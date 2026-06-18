# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

REST API service that wraps the **`eegdash-tagger`** library to classify EEG/MEG datasets with an LLM (via OpenRouter). It adds caching, a durable job queue, and MongoDB write-back. The tagging logic itself (prompts, few-shot examples, BIDS metadata parsing, abstract scraping) lives in the separate `eegdash-llm-tagger` repo — this service orchestrates it.

> **Critical dependency**: `pyproject.toml` pins `eegdash-tagger @ file:///Users/kuntalkokate/neuroscience_work/eegdash-llm-tagger` — an absolute local path. The library must be checked out at that location (or the path edited) for installs to work. Symbols imported from it: `OpenRouterTagger`, `ParsedMetadata`, `build_dataset_summary_from_path`, `extract_dois_from_references`, `fetch_abstract_with_cache`.

## Commands

```bash
# Install (editable, with dev extras)
uv pip install -e ".[dev]"

# Run the test suite (pytest, asyncio_mode=auto — no @pytest.mark.asyncio needed)
uv run pytest

# Run one file / one test
uv run pytest tests/test_orchestrator.py
uv run pytest tests/test_orchestrator.py::test_name -v

# Lint (ruff, line-length 100, rules E/F/I/W)
uv run ruff check .
uv run ruff format .

# Run the API locally
uv run uvicorn src.api.main:app --reload --port 8000

# Run the async worker (separate process; needs POSTGRES_URL + Mongo creds)
uv run python -m src.services.worker

# Docker
docker-compose up -d --build
scripts/test_docker.sh         # build + health/endpoint smoke test
scripts/test_integration.sh    # real LLM call against ds002718 (needs OPENROUTER_API_KEY)
```

All unit tests mock the LLM, git clone, metadata extraction, and abstract fetching (see `tests/conftest.py`) — they make no network calls. `scripts/test_integration.sh` is the only path that hits a real LLM.

## Two execution modes — this is the central design split

### 1. Synchronous (`POST /api/v1/tag`) — `TaggingOrchestrator`
Blocks until done. On each call it **shallow-clones the dataset's git repo into a tempdir**, parses BIDS metadata, fetches up to 2 paper abstracts from DOIs, checks the cache, and calls the LLM on a miss. Used for one-off interactive tagging. Clone uses `GIT_LFS_SKIP_SMUDGE=1` and a timeout. Graceful degradation: clone or LLM failures fall back to any stale cached result for that dataset (`serve_stale_on_error`).

### 2. Asynchronous (`POST /api/v1/tag/enqueue` → Postgres → `TaggingWorker` → MongoDB)
Fire-and-forget. The **caller supplies `metadata_snapshot` at enqueue time**, so the worker never re-clones — it tags directly from the snapshot. This is the key optimization over sync mode. Flow:
- `TaggingQueue` (`queue.py`) persists jobs in a Postgres `tagging_jobs` table. Workers claim jobs atomically with `SELECT ... FOR UPDATE SKIP LOCKED` (safe for multiple workers). Retries use exponential backoff (`RETRY_DELAYS = 60/300/1800s`, `max_attempts=3`); stuck "processing" jobs older than 30 min are recovered. Dedup is enforced by `UNIQUE(dataset_id, metadata_hash)` — re-enqueueing identical metadata is a no-op.
- `TaggingWorker` (`worker.py`) polls every 5s, processes a job, then writes results back to MongoDB.

The queue is **optional**: it only initializes if `POSTGRES_URL` is set. Without it the enqueue endpoints return 503 and `queue_enabled` is false. Check `health` / the root endpoint for `queue_enabled`.

## Caching & cache invalidation

`TaggingCache` (`cache.py`) is a JSON file (`tagging_cache.json`), written atomically via temp-file + rename. Key format:

```
{dataset_id}:{metadata_hash}:{config_hash}:{model}
```

- **`config_hash`** = SHA-256 of `few_shot_examples.json` content + model name (first 16 chars). Computed identically in `api/main.py:compute_config_hash` and `worker.py:_compute_config_hash`. **Changing few-shot examples or the model invalidates the cache; changing the prompt does not** (by design).
- **`metadata_hash`** = hash of a filtered set of metadata fields (title, dataset_description, readme, participants_overview, tasks, events, paper_abstract).
- **Ground truth** entries use the sentinel `GROUND_TRUTH` for both metadata and config hash, so they are *never* invalidated. The worker checks them first.

Worker cache lookup order in `_process_job`: **ground truth → regular cache → LLM call**. Each path then writes the result to MongoDB and marks the job complete.

## MongoDB write-back — two interchangeable backends

The worker writes tags to the EEGDash dataset document under `tags` + `tagger_meta` fields, via a `MongoUpdaterProtocol` implementation chosen at runtime:
- **HTTP (preferred)** — `MongoDBHttpUpdater`: used when `EEGDASH_API_URL` + `EEGDASH_ADMIN_TOKEN` are set. Talks to the EEGDash REST API (`GET /api/{db}/datasets/{id}`, `POST /admin/{db}/datasets`). No direct DB port exposure.
- **Direct** — `MongoDBUpdater`: used when `MONGODB_URL` is set. Direct `pymongo` `$set`.

**Both always check the dataset exists first and use `upsert=False`** — they update existing docs only, never create. HTTP mode is selected if both option sets are present.

## Hooks framework (`src/hooks/`) — independent of the tagging pipeline

A standalone mini-framework for arbitrary per-dataset processing. Subclass `DatasetHook`, implement `process(dataset_id, dataset) -> dict`; the returned dict is `$set` on the MongoDB doc (only those fields). `DatasetHook.run()` handles CLI args (`--dry-run`, `--verbose`, `--database`), env-var loading, the DB round-trip (via `MongoDBHttpUpdater`), and read-back verification. Working example: `src/hooks/examples/dummy_hook.py`. This shares only `MongoDBHttpUpdater` with the tagging code — it does not touch the queue/cache/orchestrator.

## Batch operation scripts (`scripts/`)

- `enqueue_all_datasets.py` — fetch all datasets from the EEGDash API and enqueue them (`--skip-ground-truth`, `--dry-run`, `--limit`).
- `prepopulate_ground_truth.py` — load a ground-truth CSV from the `eegdash-llm-tagger` repo into the cache as never-invalidated entries.
- `batch_status_report.py` — queue + cache + ground-truth status (`--watch`).

Scripts add the repo root to `sys.path` and import `from src.services...`, so run them from the repo root with `uv run python scripts/<name>.py`.

## Environment variables

`OPENROUTER_API_KEY` is the only hard requirement for the API (startup raises without it). `LLM_MODEL` defaults to `openai/gpt-4-turbo`. `POSTGRES_URL` enables the async queue. The worker needs Postgres + either HTTP (`EEGDASH_API_URL`/`EEGDASH_ADMIN_TOKEN`) or direct (`MONGODB_URL`) Mongo creds. `FEW_SHOT_PATH`/`PROMPT_PATH` override the library's bundled config files. See the README's "Environment Variables Reference" table for the full list; `.env` is loaded automatically (`load_dotenv()`) and is gitignored.

## Production deployment (`deploy/`)

The full stack runs on the `indexing` host as a separate compose project. Artifacts and docs live in `deploy/`:
- **`deploy/README.md`** — concise deploy + day-to-day usage commands.
- **`deploy/DEPLOYMENT_PLAN.md`** — architecture, security model, and layered test plan.
- **`deploy/Dockerfile.prod`** — the dev `pyproject.toml` pins `eegdash-tagger` to a local `file://` path; this image **sed-rewrites it to the public git URL**, COPYs `README.md` (hatchling requires it), installs `httpx` (a runtime dep only listed under `[dev]`), and bakes in `deploy/vendor-config/{few_shot_examples.json,prompt.md}` (the tagger wheel doesn't ship them and its default paths break when installed non-editably). When changing build/runtime deps or the bundled config, keep these in sync.
- **`deploy/docker-compose.prod.yml`** — postgres + api + worker on the external `eegdash-competition_backend` network; **api and worker use separate cache volumes and you must run a single worker** (`cache.py` writes through one shared `.tmp` filename, so concurrent writers corrupt the cache).
