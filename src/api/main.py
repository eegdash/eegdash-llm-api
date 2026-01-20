"""
EEGDash LLM Tagger API Service.

FastAPI application providing REST endpoints for dataset tagging
with caching and orchestration.

Supports two modes:
1. Synchronous: POST /api/v1/tag (blocks until tagging complete)
2. Asynchronous: POST /api/v1/tag/enqueue (fire-and-forget, worker processes later)
"""
# Load .env file if present (for local development)
from dotenv import load_dotenv
load_dotenv()

import os
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from eegdash_tagger.tagging import OpenRouterTagger

from ..services.cache import TaggingCache
from ..services.orchestrator import TaggingOrchestrator
from ..services.queue import TaggingQueue


def compute_config_hash(few_shot_path: Path, prompt_path: Path) -> str:
    """
    Compute hash of few-shot examples + prompt for cache invalidation.

    Args:
        few_shot_path: Path to few_shot_examples.json
        prompt_path: Path to prompt.md

    Returns:
        First 16 characters of combined SHA-256 hash
    """
    hasher = hashlib.sha256()

    if few_shot_path.exists():
        hasher.update(few_shot_path.read_bytes())

    if prompt_path.exists():
        hasher.update(prompt_path.read_bytes())

    return hasher.hexdigest()[:16]


# Global instances
orchestrator: Optional[TaggingOrchestrator] = None
cache: Optional[TaggingCache] = None
queue: Optional[TaggingQueue] = None
queue_enabled: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Initializes cache, tagger, orchestrator, and optionally the queue on startup.
    Uses default paths from the eegdash-llm-tagger library package.
    """
    global orchestrator, cache, queue, queue_enabled

    api_key = os.getenv('OPENROUTER_API_KEY')
    model = os.getenv('LLM_MODEL', 'openai/gpt-4-turbo')
    postgres_url = os.getenv('POSTGRES_URL')

    # Cache paths - these are for the API service's own caching
    cache_dir = Path(os.getenv('CACHE_DIR', '/tmp/eegdash-llm-api/cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / 'tagging_cache.json'
    abstract_cache_path = cache_dir / 'abstract_cache.json'

    # Use library's bundled config files by default
    # Can be overridden via environment variables if needed
    few_shot_path = Path(os.getenv('FEW_SHOT_PATH')) if os.getenv('FEW_SHOT_PATH') else OpenRouterTagger.get_default_few_shot_path()
    prompt_path = Path(os.getenv('PROMPT_PATH')) if os.getenv('PROMPT_PATH') else OpenRouterTagger.get_default_prompt_path()

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable required")

    # Compute config hash for cache invalidation
    config_hash = compute_config_hash(few_shot_path, prompt_path)

    # Initialize components
    cache = TaggingCache(cache_path=cache_path, config_hash=config_hash)

    tagger = OpenRouterTagger(
        api_key=api_key,
        model=model,
        few_shot_path=few_shot_path,
        prompt_path=prompt_path
    )

    orchestrator = TaggingOrchestrator(
        tagger=tagger,
        cache=cache,
        allow_on_demand=True,
        serve_stale_on_error=True,
        abstract_cache_path=abstract_cache_path
    )

    # Initialize queue if Postgres is configured
    if postgres_url:
        try:
            queue = TaggingQueue(postgres_url)
            await queue.initialize()
            queue_enabled = True
        except Exception as e:
            import logging
            logging.warning(f"Queue initialization failed: {e}. Enqueue endpoints will be disabled.")
            queue_enabled = False

    yield

    # Cleanup
    if queue:
        await queue.close()


app = FastAPI(
    title="EEGDash LLM Tagger API",
    description="REST API for tagging EEG/MEG datasets with LLM-based classification",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response Models

class TagRequest(BaseModel):
    """Request model for tagging a dataset."""
    dataset_id: str
    source_url: str
    force_refresh: bool = False


class TagResponse(BaseModel):
    """Response model for tagging results."""
    dataset_id: str
    pathology: list[str]
    modality: list[str]
    type: list[str]
    confidence: dict
    reasoning: dict = {}
    from_cache: bool
    stale: bool = False
    cache_key: Optional[str] = None
    error: Optional[str] = None


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""
    total_entries: int
    config_hash: str
    unique_datasets: int
    datasets: list[str]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    cache_entries: int
    config_hash: Optional[str]
    queue_enabled: bool = False


# Queue Request/Response Models (for async tagging)

class EnqueueRequest(BaseModel):
    """Request model for enqueueing a tagging job."""
    dataset_id: str
    source_url: str
    metadata_snapshot: dict[str, Any]  # Pre-extracted metadata from ingestion


class EnqueueResponse(BaseModel):
    """Response model for enqueue operation."""
    status: str  # "queued" or "duplicate"
    job_id: int
    dataset_id: str
    is_new: bool


class BatchEnqueueRequest(BaseModel):
    """Request model for batch enqueueing."""
    datasets: list[EnqueueRequest]


class BatchEnqueueResponse(BaseModel):
    """Response model for batch enqueue."""
    queued: int
    duplicates: int
    total: int


class QueueStatsResponse(BaseModel):
    """Response model for queue statistics."""
    pending: int
    processing: int
    completed: int
    failed: int
    ready_to_process: int


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: Optional[int]
    dataset_id: str
    status: Optional[str]  # pending, processing, completed, failed
    attempts: Optional[int]
    created_at: Optional[str]
    completed_at: Optional[str]
    result: Optional[dict]
    error: Optional[str]


# API Endpoints

@app.post("/api/v1/tag", response_model=TagResponse)
async def tag_dataset(req: TagRequest):
    """
    Tag a dataset with LLM classification.

    Checks cache first, clones repository and calls LLM on cache miss.

    Args:
        req: TagRequest with dataset_id, source_url, and optional force_refresh

    Returns:
        TagResponse with classification results
    """
    try:
        result = orchestrator.tag(
            dataset_id=req.dataset_id,
            source_url=req.source_url,
            force_refresh=req.force_refresh
        )
        return TagResponse(
            dataset_id=req.dataset_id,
            pathology=result.get('pathology', ['Unknown']),
            modality=result.get('modality', ['Unknown']),
            type=result.get('type', ['Unknown']),
            confidence=result.get('confidence', {}),
            reasoning=result.get('reasoning', {}),
            from_cache=result.get('from_cache', False),
            stale=result.get('stale', False),
            cache_key=result.get('cache_key'),
            error=result.get('error')
        )
    except Exception as e:
        return TagResponse(
            dataset_id=req.dataset_id,
            pathology=["Unknown"],
            modality=["Unknown"],
            type=["Unknown"],
            confidence={},
            from_cache=False,
            error=str(e)
        )


@app.get("/api/v1/tags/{dataset_id}", response_model=TagResponse)
async def get_cached_tags(dataset_id: str):
    """
    Get cached tags for a dataset (no LLM call).

    Args:
        dataset_id: Dataset identifier

    Returns:
        TagResponse with cached classification results

    Raises:
        HTTPException: 404 if no cached result found
    """
    result = cache.get_any_for_dataset(dataset_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"No cached tags for {dataset_id}")

    return TagResponse(
        dataset_id=dataset_id,
        pathology=result["result"].get("pathology", ["Unknown"]),
        modality=result["result"].get("modality", ["Unknown"]),
        type=result["result"].get("type", ["Unknown"]),
        confidence=result["result"].get("confidence", {}),
        reasoning=result["result"].get("reasoning", {}),
        from_cache=True,
        stale=result.get("stale", False)
    )


@app.get("/api/v1/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """
    Get cache statistics.

    Returns:
        CacheStatsResponse with entry counts and cached datasets
    """
    stats = cache.stats()
    return CacheStatsResponse(**stats)


@app.get("/api/v1/cache/entries")
async def list_cache_entries(dataset_id: Optional[str] = None):
    """
    List cache entries, optionally filtered by dataset.

    Args:
        dataset_id: Optional filter by dataset ID

    Returns:
        List of cache entry summaries
    """
    return cache.list_entries(dataset_id=dataset_id)


@app.delete("/api/v1/cache")
async def clear_cache():
    """
    Clear entire cache.

    Returns:
        Status confirmation
    """
    cache.clear()
    return {"status": "cleared", "message": "Cache has been cleared"}


@app.delete("/api/v1/cache/{cache_key:path}")
async def delete_cache_entry(cache_key: str):
    """
    Delete a specific cache entry.

    Args:
        cache_key: Full cache key to delete

    Returns:
        Status confirmation

    Raises:
        HTTPException: 404 if cache key not found
    """
    deleted = cache.delete(cache_key)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Cache key not found: {cache_key}")
    return {"status": "deleted", "cache_key": cache_key}


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.

    Returns:
        HealthResponse with service status and cache info
    """
    return HealthResponse(
        status="healthy",
        cache_entries=len(cache._cache) if cache else 0,
        config_hash=cache.config_hash if cache else None,
        queue_enabled=queue_enabled
    )


@app.get("/")
async def root():
    """
    Root endpoint with API information.

    Returns:
        API info and available endpoints
    """
    endpoints = {
        "tag_sync": "POST /api/v1/tag",
        "get_cached": "GET /api/v1/tags/{dataset_id}",
        "cache_stats": "GET /api/v1/cache/stats",
        "cache_entries": "GET /api/v1/cache/entries",
        "clear_cache": "DELETE /api/v1/cache",
        "health": "GET /health",
    }

    # Add queue endpoints if enabled
    if queue_enabled:
        endpoints.update({
            "enqueue": "POST /api/v1/tag/enqueue",
            "enqueue_batch": "POST /api/v1/tag/enqueue/batch",
            "queue_stats": "GET /api/v1/queue/stats",
            "job_status": "GET /api/v1/queue/status/{dataset_id}",
        })

    return {
        "name": "EEGDash LLM Tagger API",
        "version": "1.0.0",
        "queue_enabled": queue_enabled,
        "endpoints": endpoints
    }


# =============================================================================
# Queue Endpoints (Async Tagging)
# =============================================================================

@app.post("/api/v1/tag/enqueue", response_model=EnqueueResponse)
async def enqueue_tagging_job(req: EnqueueRequest):
    """
    Enqueue a dataset for async tagging (fire-and-forget).

    The job is added to a queue and processed by a background worker.
    Use this for non-blocking tagging during ingestion pipelines.

    Args:
        req: EnqueueRequest with dataset_id, source_url, and metadata_snapshot

    Returns:
        EnqueueResponse with job_id and queue status

    Raises:
        HTTPException: 503 if queue is not enabled
    """
    if not queue_enabled or not queue:
        raise HTTPException(
            status_code=503,
            detail="Queue not enabled. Set POSTGRES_URL environment variable."
        )

    job_id, is_new = await queue.enqueue(
        dataset_id=req.dataset_id,
        source_url=req.source_url,
        metadata_snapshot=req.metadata_snapshot,
    )

    return EnqueueResponse(
        status="queued" if is_new else "duplicate",
        job_id=job_id,
        dataset_id=req.dataset_id,
        is_new=is_new,
    )


@app.post("/api/v1/tag/enqueue/batch", response_model=BatchEnqueueResponse)
async def enqueue_batch(req: BatchEnqueueRequest):
    """
    Enqueue multiple datasets for async tagging.

    Idempotent: duplicate jobs (same dataset_id + metadata_hash) are skipped.

    Args:
        req: BatchEnqueueRequest with list of datasets

    Returns:
        BatchEnqueueResponse with counts of queued and duplicates

    Raises:
        HTTPException: 503 if queue is not enabled
    """
    if not queue_enabled or not queue:
        raise HTTPException(
            status_code=503,
            detail="Queue not enabled. Set POSTGRES_URL environment variable."
        )

    jobs = [
        {
            "dataset_id": ds.dataset_id,
            "source_url": ds.source_url,
            "metadata_snapshot": ds.metadata_snapshot,
        }
        for ds in req.datasets
    ]

    result = await queue.enqueue_batch(jobs)

    return BatchEnqueueResponse(
        queued=result["queued"],
        duplicates=result["duplicates"],
        total=len(req.datasets),
    )


@app.get("/api/v1/queue/stats", response_model=QueueStatsResponse)
async def queue_stats():
    """
    Get queue statistics.

    Returns:
        QueueStatsResponse with job counts by status

    Raises:
        HTTPException: 503 if queue is not enabled
    """
    if not queue_enabled or not queue:
        raise HTTPException(
            status_code=503,
            detail="Queue not enabled. Set POSTGRES_URL environment variable."
        )

    stats = await queue.get_stats()

    return QueueStatsResponse(
        pending=stats["pending"],
        processing=stats["processing"],
        completed=stats["completed"],
        failed=stats["failed"],
        ready_to_process=stats["ready_to_process"],
    )


@app.get("/api/v1/queue/status/{dataset_id}", response_model=JobStatusResponse)
async def job_status(dataset_id: str):
    """
    Get status of the most recent tagging job for a dataset.

    Args:
        dataset_id: Dataset identifier

    Returns:
        JobStatusResponse with job details

    Raises:
        HTTPException: 503 if queue is not enabled
        HTTPException: 404 if no jobs found for dataset
    """
    if not queue_enabled or not queue:
        raise HTTPException(
            status_code=503,
            detail="Queue not enabled. Set POSTGRES_URL environment variable."
        )

    status = await queue.get_job_status(dataset_id)

    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"No tagging jobs found for dataset: {dataset_id}"
        )

    return JobStatusResponse(
        job_id=status.get("job_id"),
        dataset_id=dataset_id,
        status=status.get("status"),
        attempts=status.get("attempts"),
        created_at=status.get("created_at"),
        completed_at=status.get("completed_at"),
        result=status.get("result"),
        error=status.get("error"),
    )
