"""Postgres-backed tagging queue for async job processing.

This module provides a durable queue for tagging jobs with:
- Atomic job claiming with SKIP LOCKED for multiple workers
- Retry logic with exponential backoff
- Stuck job detection and recovery
- Idempotent job creation (deduplication)

Schema:
    CREATE TABLE tagging_jobs (
        id SERIAL PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        source_url TEXT NOT NULL,

        -- Snapshot metadata (from ingestion time)
        metadata_snapshot JSONB,          -- The actual metadata to tag
        metadata_hash TEXT NOT NULL,      -- Hash of metadata for deduplication

        -- Job state
        status TEXT DEFAULT 'pending',    -- pending, processing, completed, failed
        attempts INTEGER DEFAULT 0,
        max_attempts INTEGER DEFAULT 3,

        -- Timestamps
        created_at TIMESTAMPTZ DEFAULT NOW(),
        started_at TIMESTAMPTZ,
        completed_at TIMESTAMPTZ,
        next_retry_at TIMESTAMPTZ,

        -- Results
        result JSONB,                     -- Tagging result (tags, confidence, reasoning)
        error TEXT,                       -- Last error message

        -- Deduplication key: same dataset + same metadata = same job
        UNIQUE(dataset_id, metadata_hash)
    );

    CREATE INDEX idx_tagging_jobs_status ON tagging_jobs(status);
    CREATE INDEX idx_tagging_jobs_next_retry ON tagging_jobs(next_retry_at) WHERE status = 'pending';
"""

import json
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from dataclasses import dataclass, field

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


@dataclass
class TaggingJob:
    """A tagging job from the queue."""
    id: int
    dataset_id: str
    source_url: str
    metadata_snapshot: dict[str, Any]
    metadata_hash: str
    status: str
    attempts: int
    max_attempts: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class TaggingQueue:
    """Postgres-backed queue for tagging jobs.

    Supports multiple workers with atomic job claiming using SKIP LOCKED.

    Usage:
        queue = TaggingQueue("postgresql://...")
        await queue.initialize()

        # Enqueue a job
        job_id = await queue.enqueue(
            dataset_id="ds001234",
            source_url="https://github.com/...",
            metadata_snapshot={"title": "...", "readme": "..."}
        )

        # Claim and process jobs (in worker)
        job = await queue.claim_job()
        if job:
            try:
                result = process(job)
                await queue.complete_job(job.id, result)
            except Exception as e:
                await queue.fail_job(job.id, str(e))
    """

    # Retry backoff: 1min, 5min, 30min
    RETRY_DELAYS = [60, 300, 1800]

    # Jobs stuck in "processing" for longer than this are considered dead
    STUCK_JOB_TIMEOUT = timedelta(minutes=30)

    def __init__(self, connection_string: str):
        """Initialize queue with Postgres connection string.

        Args:
            connection_string: Postgres connection URL
                e.g., "postgresql://user:pass@localhost:5432/eegdash"
        """
        self.connection_string = connection_string
        self._conn: Optional[psycopg.Connection] = None

    async def initialize(self) -> None:
        """Initialize database connection and create schema if needed."""
        self._conn = await psycopg.AsyncConnection.connect(
            self.connection_string,
            row_factory=dict_row
        )
        await self._create_schema()
        logger.info("TaggingQueue initialized")

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _create_schema(self) -> None:
        """Create queue table and indexes if they don't exist."""
        async with self._conn.cursor() as cur:
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS tagging_jobs (
                    id SERIAL PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    metadata_snapshot JSONB,
                    metadata_hash TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    attempts INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    next_retry_at TIMESTAMPTZ,
                    result JSONB,
                    error TEXT,
                    UNIQUE(dataset_id, metadata_hash)
                )
            """)

            # Create indexes
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tagging_jobs_status
                ON tagging_jobs(status)
            """)
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tagging_jobs_next_retry
                ON tagging_jobs(next_retry_at)
                WHERE status = 'pending'
            """)

            await self._conn.commit()

    @staticmethod
    def compute_metadata_hash(metadata: dict[str, Any]) -> str:
        """Compute deterministic hash of metadata for deduplication.

        Args:
            metadata: Dataset metadata dictionary

        Returns:
            SHA-256 hash of normalized JSON
        """
        # Sort keys for deterministic serialization
        normalized = json.dumps(metadata, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    async def enqueue(
        self,
        dataset_id: str,
        source_url: str,
        metadata_snapshot: dict[str, Any],
        max_attempts: int = 3,
    ) -> tuple[int, bool]:
        """Add a tagging job to the queue.

        If a job with the same dataset_id and metadata_hash already exists,
        this is a no-op (idempotent).

        Args:
            dataset_id: Dataset identifier
            source_url: GitHub URL for the dataset
            metadata_snapshot: The metadata to tag (captured at ingestion time)
            max_attempts: Maximum retry attempts

        Returns:
            Tuple of (job_id, is_new) where is_new indicates if this was a new job
        """
        metadata_hash = self.compute_metadata_hash(metadata_snapshot)

        async with self._conn.cursor() as cur:
            # Try to insert, on conflict do nothing and return existing
            await cur.execute("""
                INSERT INTO tagging_jobs (
                    dataset_id, source_url, metadata_snapshot, metadata_hash, max_attempts
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (dataset_id, metadata_hash) DO NOTHING
                RETURNING id
            """, (dataset_id, source_url, json.dumps(metadata_snapshot), metadata_hash, max_attempts))

            result = await cur.fetchone()

            if result:
                await self._conn.commit()
                logger.info(f"Enqueued new job for {dataset_id}")
                return result["id"], True

            # Job already exists, fetch its ID
            await cur.execute("""
                SELECT id FROM tagging_jobs
                WHERE dataset_id = %s AND metadata_hash = %s
            """, (dataset_id, metadata_hash))

            existing = await cur.fetchone()
            logger.debug(f"Job already exists for {dataset_id}")
            return existing["id"], False

    async def enqueue_batch(
        self,
        jobs: list[dict[str, Any]],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Enqueue multiple jobs at once.

        Args:
            jobs: List of dicts with dataset_id, source_url, metadata_snapshot
            max_attempts: Maximum retry attempts

        Returns:
            Summary with counts of queued, duplicates
        """
        queued = 0
        duplicates = 0

        for job in jobs:
            _, is_new = await self.enqueue(
                dataset_id=job["dataset_id"],
                source_url=job["source_url"],
                metadata_snapshot=job["metadata_snapshot"],
                max_attempts=max_attempts,
            )
            if is_new:
                queued += 1
            else:
                duplicates += 1

        return {"queued": queued, "duplicates": duplicates}

    async def claim_job(self, worker_id: str = "default") -> Optional[TaggingJob]:
        """Atomically claim the next available job.

        Uses SKIP LOCKED to allow multiple workers without conflicts.

        Args:
            worker_id: Identifier for the worker (for logging)

        Returns:
            TaggingJob if one was claimed, None if queue is empty
        """
        now = datetime.now(timezone.utc)

        async with self._conn.cursor() as cur:
            # Claim next pending job that's ready for processing
            # (either no next_retry_at or it's in the past)
            await cur.execute("""
                UPDATE tagging_jobs
                SET status = 'processing',
                    started_at = %s,
                    attempts = attempts + 1
                WHERE id = (
                    SELECT id FROM tagging_jobs
                    WHERE status = 'pending'
                    AND (next_retry_at IS NULL OR next_retry_at <= %s)
                    ORDER BY created_at ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING *
            """, (now, now))

            row = await cur.fetchone()
            await self._conn.commit()

            if row:
                logger.info(f"Worker {worker_id} claimed job {row['id']} for {row['dataset_id']}")
                return self._row_to_job(row)

            return None

    async def complete_job(self, job_id: int, result: dict[str, Any]) -> None:
        """Mark a job as successfully completed.

        Args:
            job_id: Job ID to complete
            result: Tagging result (tags, confidence, reasoning)
        """
        now = datetime.now(timezone.utc)

        async with self._conn.cursor() as cur:
            await cur.execute("""
                UPDATE tagging_jobs
                SET status = 'completed',
                    completed_at = %s,
                    result = %s,
                    error = NULL
                WHERE id = %s
            """, (now, json.dumps(result), job_id))
            await self._conn.commit()

        logger.info(f"Job {job_id} completed successfully")

    async def fail_job(self, job_id: int, error: str) -> None:
        """Mark a job as failed, scheduling retry if attempts remain.

        Args:
            job_id: Job ID that failed
            error: Error message
        """
        async with self._conn.cursor() as cur:
            # Get current job state
            await cur.execute("""
                SELECT attempts, max_attempts FROM tagging_jobs WHERE id = %s
            """, (job_id,))
            row = await cur.fetchone()

            if not row:
                logger.warning(f"Job {job_id} not found for failure")
                return

            attempts = row["attempts"]
            max_attempts = row["max_attempts"]

            if attempts >= max_attempts:
                # No more retries, mark as permanently failed
                await cur.execute("""
                    UPDATE tagging_jobs
                    SET status = 'failed',
                        completed_at = %s,
                        error = %s
                    WHERE id = %s
                """, (datetime.now(timezone.utc), error, job_id))
                logger.error(f"Job {job_id} permanently failed after {attempts} attempts: {error}")
            else:
                # Schedule retry with exponential backoff
                delay_idx = min(attempts - 1, len(self.RETRY_DELAYS) - 1)
                delay_seconds = self.RETRY_DELAYS[delay_idx]
                next_retry = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)

                await cur.execute("""
                    UPDATE tagging_jobs
                    SET status = 'pending',
                        started_at = NULL,
                        next_retry_at = %s,
                        error = %s
                    WHERE id = %s
                """, (next_retry, error, job_id))
                logger.warning(f"Job {job_id} failed, retry scheduled at {next_retry}: {error}")

            await self._conn.commit()

    async def recover_stuck_jobs(self) -> int:
        """Recover jobs stuck in 'processing' state.

        Jobs that have been processing for longer than STUCK_JOB_TIMEOUT
        are reset to pending for retry.

        Returns:
            Number of jobs recovered
        """
        cutoff = datetime.now(timezone.utc) - self.STUCK_JOB_TIMEOUT

        async with self._conn.cursor() as cur:
            await cur.execute("""
                UPDATE tagging_jobs
                SET status = 'pending',
                    started_at = NULL,
                    error = 'Worker timeout - job stuck in processing'
                WHERE status = 'processing'
                AND started_at < %s
            """, (cutoff,))

            recovered = cur.rowcount
            await self._conn.commit()

        if recovered > 0:
            logger.warning(f"Recovered {recovered} stuck jobs")

        return recovered

    async def get_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dict with counts by status and other metrics
        """
        async with self._conn.cursor() as cur:
            await cur.execute("""
                SELECT
                    status,
                    COUNT(*) as count
                FROM tagging_jobs
                GROUP BY status
            """)
            rows = await cur.fetchall()

            stats = {row["status"]: row["count"] for row in rows}

            # Get pending that are ready (not waiting for retry)
            await cur.execute("""
                SELECT COUNT(*) as count FROM tagging_jobs
                WHERE status = 'pending'
                AND (next_retry_at IS NULL OR next_retry_at <= NOW())
            """)
            ready = await cur.fetchone()
            stats["ready"] = ready["count"]

            return {
                "pending": stats.get("pending", 0),
                "processing": stats.get("processing", 0),
                "completed": stats.get("completed", 0),
                "failed": stats.get("failed", 0),
                "ready_to_process": stats.get("ready", 0),
            }

    async def get_job_status(self, dataset_id: str) -> Optional[dict[str, Any]]:
        """Get status of the most recent job for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Job status dict or None if no jobs exist
        """
        async with self._conn.cursor() as cur:
            await cur.execute("""
                SELECT id, status, attempts, created_at, completed_at, result, error
                FROM tagging_jobs
                WHERE dataset_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (dataset_id,))

            row = await cur.fetchone()
            if row:
                return {
                    "job_id": row["id"],
                    "status": row["status"],
                    "attempts": row["attempts"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
                    "result": row["result"],
                    "error": row["error"],
                }
            return None

    def _row_to_job(self, row: dict) -> TaggingJob:
        """Convert database row to TaggingJob dataclass."""
        metadata = row["metadata_snapshot"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        result = row.get("result")
        if isinstance(result, str):
            result = json.loads(result)

        return TaggingJob(
            id=row["id"],
            dataset_id=row["dataset_id"],
            source_url=row["source_url"],
            metadata_snapshot=metadata,
            metadata_hash=row["metadata_hash"],
            status=row["status"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            created_at=row["created_at"],
            started_at=row.get("started_at"),
            completed_at=row.get("completed_at"),
            next_retry_at=row.get("next_retry_at"),
            result=result,
            error=row.get("error"),
        )
