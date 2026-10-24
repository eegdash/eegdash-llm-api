# Development guidance

This service wraps `eegdash-tagger`, adding cache, synchronous HTTP orchestration,
and a PostgreSQL queue with catalogue write-back. Work from the pinned dependency
and lock file; do not restore developer-specific paths or moving Git branches.

```bash
uv sync --extra dev --frozen
uv run --extra dev pytest --cov=src
uv run uvicorn src.api.main:app --port 8000
uv run python -m src.services.worker
```

`OPENROUTER_API_KEY` is needed for service startup; unit tests use synthetic keys
and mocks. `LLM_API_BASE` selects an OpenAI-compatible endpoint per tagger instance.
The default model is `openai/gpt-5.2`. `POSTGRES_URL` enables queue endpoints.
Workers require Postgres and either `EEGDASH_API_URL`/`EEGDASH_ADMIN_TOKEN` or a
direct `MONGODB_URL`. Do not point tests at production resources.

Model/transport/parse failures propagate as `TaggingError`. A valid `Unknown`
label is not a failed call. Failed calls must never enter the cache or catalogue.
A false return or exception from catalogue persistence must call `fail_job`, not
`complete_job`, in every ground-truth/cache/fresh-model branch. Valid cached
inference may be reused after persistence failure. Synchronous failures without
stale data return HTTP 502; stale results retain explicit stale/error fields.

`services/config.py` defines the shared cache identity from the effective prompt,
loaded examples, endpoint, model, taxonomy and library version. Recompute before
each job/request so prompt-file edits invalidate cache. Examples are loaded once
by the library; restart to apply edits. Cache metadata uses the library whitelist,
including task_details. Known legacy failed-call envelopes are not served stale.
The JSON file cache still permits one writer per file. API and worker volumes
must be separate; do not scale workers sharing a cache file.

Queue deduplication is by dataset ID and metadata hash, independent of model
configuration; an already completed snapshot is not automatically queued again
when configuration changes. Keep this limitation documented rather than silently
altering production queue state.

The optional `tests/test_postgres_integration.py` uses `TEST_POSTGRES_URL`, creates
an isolated schema, runs a loopback HTTP fixture, and drops only that schema.
It exercises two failed attempts (model then catalogue), followed by completion
using the cached successful inference. Docker builds use `requirements.lock`,
exported with `uv export --frozen --no-dev --no-emit-project --no-hashes`.
Update the library revision and lock before building a release. Never deploy or
write production metadata as part of a unit/integration test.
