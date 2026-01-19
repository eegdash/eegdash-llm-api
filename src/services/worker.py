"""Background worker for processing tagging jobs.

This worker:
1. Claims jobs from the Postgres queue
2. Tags using the snapshot metadata (no re-cloning needed)
3. Updates MongoDB with results
4. Handles retries and failures

Run as a separate process/container:
    python -m src.services.worker

Or programmatically:
    worker = TaggingWorker(...)
    await worker.run()
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Optional

from eegdash_tagger.tagging import OpenRouterTagger

from .queue import TaggingQueue, TaggingJob
from .mongodb_updater import MongoDBUpdater

logger = logging.getLogger(__name__)


class TaggingWorker:
    """Background worker that processes tagging jobs.

    Features:
    - Uses snapshot metadata from queue (no re-cloning)
    - Atomic job claiming with SKIP LOCKED
    - Automatic retry with exponential backoff
    - Graceful shutdown on SIGTERM/SIGINT
    - Stuck job recovery

    Usage:
        worker = TaggingWorker(
            postgres_url="postgresql://...",
            mongodb_url="mongodb://...",
            openrouter_api_key="...",
        )
        await worker.run()
    """

    # Time between checking for new jobs when queue is empty
    POLL_INTERVAL = 5.0

    # Time between stuck job recovery checks
    RECOVERY_INTERVAL = 300.0  # 5 minutes

    def __init__(
        self,
        postgres_url: str,
        mongodb_url: str,
        mongodb_database: str = "eegdash",
        mongodb_collection: str = "datasets",
        openrouter_api_key: Optional[str] = None,
        model: str = "openai/gpt-4-turbo",
        worker_id: str = "worker-1",
    ):
        """Initialize worker.

        Args:
            postgres_url: Postgres connection URL for queue
            mongodb_url: MongoDB connection URL for tag updates
            mongodb_database: MongoDB database name
            mongodb_collection: MongoDB collection name
            openrouter_api_key: OpenRouter API key (defaults to env var)
            model: LLM model to use
            worker_id: Unique identifier for this worker
        """
        self.postgres_url = postgres_url
        self.mongodb_url = mongodb_url
        self.mongodb_database = mongodb_database
        self.mongodb_collection = mongodb_collection
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.worker_id = worker_id

        self._queue: Optional[TaggingQueue] = None
        self._mongodb: Optional[MongoDBUpdater] = None
        self._tagger: Optional[OpenRouterTagger] = None
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Track current config hash for provenance
        self._config_hash: Optional[str] = None

    async def initialize(self) -> None:
        """Initialize connections and tagger."""
        logger.info(f"Worker {self.worker_id} initializing...")

        # Initialize queue
        self._queue = TaggingQueue(self.postgres_url)
        await self._queue.initialize()

        # Initialize MongoDB
        self._mongodb = MongoDBUpdater(
            self.mongodb_url,
            self.mongodb_database,
            self.mongodb_collection,
        )
        self._mongodb.connect()

        # Initialize tagger
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        self._tagger = OpenRouterTagger(
            api_key=self.openrouter_api_key,
            model=self.model,
        )

        # Compute config hash for provenance
        self._config_hash = self._tagger.compute_config_hash()

        logger.info(f"Worker {self.worker_id} initialized with config_hash={self._config_hash[:8]}...")

    async def shutdown(self) -> None:
        """Gracefully shutdown worker."""
        logger.info(f"Worker {self.worker_id} shutting down...")
        self._running = False
        self._shutdown_event.set()

        if self._queue:
            await self._queue.close()

        if self._mongodb:
            self._mongodb.close()

        logger.info(f"Worker {self.worker_id} shutdown complete")

    async def run(self) -> None:
        """Main worker loop.

        Continuously claims and processes jobs until shutdown.
        """
        await self.initialize()
        self._running = True

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        logger.info(f"Worker {self.worker_id} started, waiting for jobs...")

        # Start recovery task
        recovery_task = asyncio.create_task(self._recovery_loop())

        try:
            while self._running:
                # Try to claim a job
                job = await self._queue.claim_job(self.worker_id)

                if job:
                    await self._process_job(job)
                else:
                    # No jobs available, wait before checking again
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(),
                            timeout=self.POLL_INTERVAL
                        )
                    except asyncio.TimeoutError:
                        pass  # Normal timeout, continue loop

        except Exception as e:
            logger.error(f"Worker {self.worker_id} error: {e}")
            raise
        finally:
            recovery_task.cancel()
            await self.shutdown()

    async def _recovery_loop(self) -> None:
        """Periodically recover stuck jobs."""
        while self._running:
            try:
                await asyncio.sleep(self.RECOVERY_INTERVAL)
                if self._queue:
                    recovered = await self._queue.recover_stuck_jobs()
                    if recovered > 0:
                        logger.info(f"Recovered {recovered} stuck jobs")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Recovery error: {e}")

    async def _process_job(self, job: TaggingJob) -> None:
        """Process a single tagging job.

        Args:
            job: The job to process
        """
        logger.info(f"Processing job {job.id} for {job.dataset_id} (attempt {job.attempts})")

        try:
            # Tag using snapshot metadata (no cloning needed!)
            result = await self._tag_from_metadata(job.metadata_snapshot)

            # Compute metadata hash for provenance
            metadata_hash = self._queue.compute_metadata_hash(job.metadata_snapshot)

            # Update MongoDB with tags
            updated = self._mongodb.update_tags(
                dataset_id=job.dataset_id,
                tags=result,
                config_hash=self._config_hash,
                metadata_hash=metadata_hash,
                model=self.model,
                reasoning=result.get("reasoning"),
            )

            if not updated:
                logger.warning(f"Dataset {job.dataset_id} not found in MongoDB")

            # Mark job as complete
            await self._queue.complete_job(job.id, {
                "tags": result,
                "config_hash": self._config_hash,
                "metadata_hash": metadata_hash,
                "mongodb_updated": updated,
            })

            logger.info(f"Job {job.id} completed for {job.dataset_id}")

        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            await self._queue.fail_job(job.id, str(e))

    async def _tag_from_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Tag a dataset using pre-extracted metadata.

        This is the key optimization: we use the snapshot metadata from
        ingestion time rather than re-cloning the repository.

        Args:
            metadata: Dataset metadata snapshot

        Returns:
            Tagging result with pathology, modality, type, confidence, reasoning
        """
        # The tagger expects metadata in a specific format
        # Convert to the format expected by tag_with_details
        result = self._tagger.tag_with_details(metadata)

        return {
            "pathology": result.get("pathology", ["Unknown"]),
            "modality": result.get("modality", ["Unknown"]),
            "type": result.get("type", ["Unknown"]),
            "confidence": result.get("confidence", {}),
            "reasoning": result.get("reasoning", {}),
        }


async def main():
    """Run worker from command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Get configuration from environment
    postgres_url = os.environ.get("POSTGRES_URL")
    mongodb_url = os.environ.get("MONGODB_URL")
    mongodb_database = os.environ.get("MONGODB_DATABASE", "eegdash")
    mongodb_collection = os.environ.get("MONGODB_COLLECTION", "datasets")
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("LLM_MODEL", "openai/gpt-4-turbo")
    worker_id = os.environ.get("WORKER_ID", f"worker-{os.getpid()}")

    if not postgres_url:
        print("ERROR: POSTGRES_URL environment variable not set")
        sys.exit(1)

    if not mongodb_url:
        print("ERROR: MONGODB_URL environment variable not set")
        sys.exit(1)

    if not openrouter_api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    worker = TaggingWorker(
        postgres_url=postgres_url,
        mongodb_url=mongodb_url,
        mongodb_database=mongodb_database,
        mongodb_collection=mongodb_collection,
        openrouter_api_key=openrouter_api_key,
        model=model,
        worker_id=worker_id,
    )

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
