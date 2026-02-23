"""Background worker for processing tagging jobs.

This worker:
1. Claims jobs from the Postgres queue
2. Tags using the snapshot metadata (no re-cloning needed)
3. Updates MongoDB with results (via direct connection OR HTTP API)
4. Handles retries and failures

Run as a separate process/container:
    python -m src.services.worker

Or programmatically:
    worker = TaggingWorker(...)
    await worker.run()

Environment variables:
    Required:
        POSTGRES_URL - Postgres connection URL for queue
        OPENROUTER_API_KEY - OpenRouter API key for LLM

    MongoDB connection (choose ONE):
        Option A - Direct connection:
            MONGODB_URL - MongoDB connection URL (e.g., mongodb://...)

        Option B - HTTP API (safer, no direct DB access):
            EEGDASH_API_URL - API URL (default: https://data.eegdash.org)
            EEGDASH_ADMIN_TOKEN - Admin bearer token for write access

    Optional:
        MONGODB_DATABASE - Database name (default: eegdash)
        LLM_MODEL - Model to use (default: openai/gpt-4-turbo)
        WORKER_ID - Unique worker identifier
"""

import asyncio
import hashlib
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Optional, Protocol

from eegdash_tagger.tagging import OpenRouterTagger

from .cache import TaggingCache, GROUND_TRUTH_HASH
from .queue import TaggingQueue, TaggingJob

logger = logging.getLogger(__name__)


class MongoUpdaterProtocol(Protocol):
    """Protocol for MongoDB updaters (direct or HTTP-based)."""

    def connect(self) -> None: ...
    def close(self) -> None: ...
    def update_tags(
        self,
        dataset_id: str,
        tags: dict[str, Any],
        config_hash: str,
        metadata_hash: str,
        model: str,
        reasoning: Optional[dict[str, str]] = None,
    ) -> bool: ...


class TaggingWorker:
    """Background worker that processes tagging jobs.

    Features:
    - Uses snapshot metadata from queue (no re-cloning)
    - Atomic job claiming with SKIP LOCKED
    - Automatic retry with exponential backoff
    - Graceful shutdown on SIGTERM/SIGINT
    - Stuck job recovery
    - Supports both direct MongoDB and HTTP API for updates

    Usage (direct MongoDB):
        worker = TaggingWorker(
            postgres_url="postgresql://...",
            mongodb_url="mongodb://...",
            openrouter_api_key="...",
        )
        await worker.run()

    Usage (HTTP API - safer):
        worker = TaggingWorker(
            postgres_url="postgresql://...",
            eegdash_api_url="https://data.eegdash.org",
            eegdash_admin_token="your-token",
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
        # Direct MongoDB connection (Option A)
        mongodb_url: Optional[str] = None,
        mongodb_database: str = "eegdash",
        mongodb_collection: str = "datasets",
        # HTTP API connection (Option B - preferred)
        eegdash_api_url: Optional[str] = None,
        eegdash_admin_token: Optional[str] = None,
        # Common settings
        openrouter_api_key: Optional[str] = None,
        model: str = "openai/gpt-4-turbo",
        worker_id: str = "worker-1",
        # Cache settings
        cache_path: Optional[Path] = None,
    ):
        """Initialize worker.

        Args:
            postgres_url: Postgres connection URL for queue
            mongodb_url: MongoDB connection URL (Option A)
            mongodb_database: MongoDB database name
            mongodb_collection: MongoDB collection name
            eegdash_api_url: EEGDash API URL (Option B)
            eegdash_admin_token: Admin token for API writes (Option B)
            openrouter_api_key: OpenRouter API key (defaults to env var)
            model: LLM model to use
            worker_id: Unique identifier for this worker
            cache_path: Path to cache file (for ground truth and LLM result caching)
        """
        self.postgres_url = postgres_url
        self.mongodb_url = mongodb_url
        self.mongodb_database = mongodb_database
        self.mongodb_collection = mongodb_collection
        self.eegdash_api_url = eegdash_api_url
        self.eegdash_admin_token = eegdash_admin_token
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.worker_id = worker_id
        self.cache_path = cache_path

        self._queue: Optional[TaggingQueue] = None
        self._mongodb: Optional[MongoUpdaterProtocol] = None
        self._tagger: Optional[OpenRouterTagger] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._use_http_api = False

        # Track current config hash for provenance
        self._config_hash: Optional[str] = None

        # Cache for ground truth and LLM results
        self._cache: Optional[TaggingCache] = None

    @staticmethod
    def _compute_config_hash(few_shot_path: Optional[Path], model: str) -> str:
        """Compute hash of few-shot examples + model for provenance tracking.

        The config hash changes when:
        - few_shot_examples.json content changes
        - model name changes

        Prompt changes do NOT invalidate cache.
        """
        hasher = hashlib.sha256()

        if few_shot_path and few_shot_path.exists():
            hasher.update(few_shot_path.read_bytes())

        hasher.update(model.encode())

        return hasher.hexdigest()[:16]

    def _create_mongodb_updater(self) -> MongoUpdaterProtocol:
        """Create the appropriate MongoDB updater based on configuration."""
        # Prefer HTTP API if configured (safer, no direct DB access)
        if self.eegdash_api_url and self.eegdash_admin_token:
            from .mongodb_http_updater import MongoDBHttpUpdater
            self._use_http_api = True
            logger.info(f"Using HTTP API for MongoDB updates: {self.eegdash_api_url}")
            return MongoDBHttpUpdater(
                api_url=self.eegdash_api_url,
                admin_token=self.eegdash_admin_token,
                database=self.mongodb_database,
            )
        elif self.mongodb_url:
            from .mongodb_updater import MongoDBUpdater
            self._use_http_api = False
            logger.info("Using direct MongoDB connection for updates")
            return MongoDBUpdater(
                self.mongodb_url,
                self.mongodb_database,
                self.mongodb_collection,
            )
        else:
            raise ValueError(
                "Either MONGODB_URL or (EEGDASH_API_URL + EEGDASH_ADMIN_TOKEN) must be set"
            )

    async def initialize(self) -> None:
        """Initialize connections and tagger."""
        logger.info(f"Worker {self.worker_id} initializing...")

        # Initialize queue
        self._queue = TaggingQueue(self.postgres_url)
        await self._queue.initialize()

        # Initialize MongoDB updater (direct or HTTP-based)
        self._mongodb = self._create_mongodb_updater()
        self._mongodb.connect()

        # Initialize tagger
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        # Get optional config paths from environment
        few_shot_path = os.environ.get("FEW_SHOT_PATH")
        prompt_path = os.environ.get("PROMPT_PATH")

        few_shot_path_obj = Path(few_shot_path) if few_shot_path else None
        prompt_path_obj = Path(prompt_path) if prompt_path else None

        self._tagger = OpenRouterTagger(
            api_key=self.openrouter_api_key,
            model=self.model,
            few_shot_path=few_shot_path_obj,
            prompt_path=prompt_path_obj,
        )

        # Compute config hash for provenance (hash of few-shot + model)
        self._config_hash = self._compute_config_hash(few_shot_path_obj, self.model)

        # Initialize cache if path provided
        if self.cache_path:
            self._cache = TaggingCache(cache_path=self.cache_path, config_hash=self._config_hash)
            gt_count = len(self._cache.list_ground_truth_datasets())
            logger.info(f"Cache initialized: {self._cache.stats()['total_entries']} entries, {gt_count} ground truth")

        mode = "HTTP API" if self._use_http_api else "direct MongoDB"
        config_hash_short = self._config_hash[:8] if self._config_hash else "unknown"
        cache_status = f", cache={self.cache_path}" if self._cache else ", no cache"
        logger.info(f"Worker {self.worker_id} initialized (mode: {mode}, config_hash={config_hash_short}...{cache_status})")

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

        Checks cache in this order:
        1. Ground truth (never invalidated)
        2. Regular cache (keyed by metadata + config + model)
        3. LLM call (if no cache hit)

        Args:
            job: The job to process
        """
        logger.info(f"Processing job {job.id} for {job.dataset_id} (attempt {job.attempts})")

        try:
            # 1. Check ground truth first (never invalidated)
            if self._cache:
                gt_result = self._cache.get_ground_truth(job.dataset_id)
                if gt_result:
                    logger.info(f"Ground truth hit for {job.dataset_id}")

                    # Update MongoDB with ground truth
                    updated = self._mongodb.update_tags(
                        dataset_id=job.dataset_id,
                        tags=gt_result,
                        config_hash=GROUND_TRUTH_HASH,
                        metadata_hash=GROUND_TRUTH_HASH,
                        model="ground_truth",
                        reasoning=gt_result.get("reasoning"),
                    )

                    if not updated:
                        logger.warning(f"Dataset {job.dataset_id} not found in MongoDB")

                    await self._queue.complete_job(job.id, {
                        "tags": gt_result,
                        "config_hash": GROUND_TRUTH_HASH,
                        "metadata_hash": GROUND_TRUTH_HASH,
                        "mongodb_updated": updated,
                        "from_ground_truth": True,
                    })

                    logger.info(f"Job {job.id} completed for {job.dataset_id} (ground truth)")
                    return

            # Compute metadata hash for cache key and provenance
            metadata_hash = self._queue.compute_metadata_hash(job.metadata_snapshot)

            # 2. Check regular cache
            if self._cache and job.metadata_snapshot:
                cache_key = self._cache.build_key(job.dataset_id, job.metadata_snapshot, self.model)
                cached_result = self._cache.get(cache_key)

                if cached_result:
                    logger.info(f"Cache hit for {job.dataset_id}")

                    # Update MongoDB with cached result
                    updated = self._mongodb.update_tags(
                        dataset_id=job.dataset_id,
                        tags=cached_result,
                        config_hash=self._config_hash,
                        metadata_hash=metadata_hash,
                        model=self.model,
                        reasoning=cached_result.get("reasoning"),
                    )

                    if not updated:
                        logger.warning(f"Dataset {job.dataset_id} not found in MongoDB")

                    await self._queue.complete_job(job.id, {
                        "tags": cached_result,
                        "config_hash": self._config_hash,
                        "metadata_hash": metadata_hash,
                        "mongodb_updated": updated,
                        "from_cache": True,
                    })

                    logger.info(f"Job {job.id} completed for {job.dataset_id} (cached)")
                    return

            # 3. No cache hit - call LLM
            logger.info(f"Cache miss for {job.dataset_id}, calling LLM...")
            result = await self._tag_from_metadata(job.metadata_snapshot)

            # Store in cache after successful tagging
            if self._cache and job.metadata_snapshot:
                cache_key = self._cache.build_key(job.dataset_id, job.metadata_snapshot, self.model)
                self._cache.set(cache_key, result, job.metadata_snapshot)
                logger.debug(f"Cached result for {job.dataset_id}")

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
                "from_llm": True,
            })

            logger.info(f"Job {job.id} completed for {job.dataset_id} (LLM)")

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
    # Load .env file if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Get configuration from environment
    postgres_url = os.environ.get("POSTGRES_URL")
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("LLM_MODEL", "openai/gpt-4-turbo")
    worker_id = os.environ.get("WORKER_ID", f"worker-{os.getpid()}")

    # MongoDB connection options
    mongodb_url = os.environ.get("MONGODB_URL")
    mongodb_database = os.environ.get("MONGODB_DATABASE", "eegdash")
    mongodb_collection = os.environ.get("MONGODB_COLLECTION", "datasets")

    # HTTP API options (preferred over direct MongoDB)
    eegdash_api_url = os.environ.get("EEGDASH_API_URL")
    eegdash_admin_token = os.environ.get("EEGDASH_ADMIN_TOKEN")

    # Validate required config
    if not postgres_url:
        print("ERROR: POSTGRES_URL environment variable not set")
        sys.exit(1)

    if not openrouter_api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    # Validate MongoDB connection options
    has_direct = bool(mongodb_url)
    has_http = bool(eegdash_api_url and eegdash_admin_token)

    if not has_direct and not has_http:
        print("ERROR: Must set either MONGODB_URL or (EEGDASH_API_URL + EEGDASH_ADMIN_TOKEN)")
        print("")
        print("Option A - Direct MongoDB connection:")
        print("  MONGODB_URL=mongodb://user:pass@host:27017/db")
        print("")
        print("Option B - HTTP API (safer, recommended):")
        print("  EEGDASH_API_URL=https://data.eegdash.org")
        print("  EEGDASH_ADMIN_TOKEN=your-admin-token")
        sys.exit(1)

    if has_http:
        print(f"Using HTTP API mode: {eegdash_api_url}")
    else:
        print("Using direct MongoDB connection mode")

    # Cache configuration (optional but recommended for ground truth support)
    cache_dir = os.environ.get("CACHE_DIR", "/tmp/eegdash-llm-api/cache")
    cache_path = Path(cache_dir) / "tagging_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using cache: {cache_path}")

    worker = TaggingWorker(
        postgres_url=postgres_url,
        mongodb_url=mongodb_url,
        mongodb_database=mongodb_database,
        mongodb_collection=mongodb_collection,
        eegdash_api_url=eegdash_api_url,
        eegdash_admin_token=eegdash_admin_token,
        openrouter_api_key=openrouter_api_key,
        model=model,
        worker_id=worker_id,
        cache_path=cache_path,
    )

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
