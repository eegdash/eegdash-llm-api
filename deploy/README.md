# Deploying & using the EEGDash LLM Tagger

Production stack on the **`indexing`** host: `tagger-postgres` (job queue) + `tagger-api`
(FastAPI) + `tagger-worker` (tags datasets → writes `tags`/`tagger_meta` to MongoDB via the
EEGDash API). Runs as a **separate compose project** on the existing
`eegdash-competition_backend` network. Full design, security, and test rationale:
[`DEPLOYMENT_PLAN.md`](./DEPLOYMENT_PLAN.md).

## Files
| File | Purpose |
|---|---|
| `Dockerfile.prod` | Prod image — installs `eegdash-tagger` from git, adds `httpx`, bakes in vendored config |
| `docker-compose.prod.yml` | The 3 services; separate cache volumes; **run a single worker** |
| `.env.prod.example` | Secrets template → copy to `deploy/.env`, `chmod 600` |
| `vendor-config/` | `few_shot_examples.json` + `prompt.md` baked to `/app/config` (the wheel doesn't ship them) |
| `caddy/tagger.eegdash.org.caddy` | Optional public site block (X-API-Key gate); only if exposing publicly |

## First deploy (on the host)
```bash
git clone https://github.com/eegdash/eegdash-llm-api.git /home/sccn/eegdash-llm-api
cd /home/sccn/eegdash-llm-api
cp deploy/.env.prod.example deploy/.env && chmod 600 deploy/.env   # then fill in secrets
docker-compose -p eegdash-llm-tagger -f deploy/docker-compose.prod.yml --env-file deploy/.env up -d --build
```
Required in `deploy/.env`: `OPENROUTER_API_KEY` (use a dedicated, **daily-capped** key),
`POSTGRES_PASSWORD` (`openssl rand -base64 32`), `EEGDASH_ADMIN_TOKEN`, `LLM_MODEL`.

## Everyday use
The API is **internal-only** (reachable on the docker network) unless Caddy is wired, so run
these on the host. `tagger-api` is reached by name from any container on the shared network.

```bash
# Health & queue status
docker exec caddy wget -qO- http://tagger-api:8000/health
docker exec tagger-api curl -sS http://localhost:8000/api/v1/queue/stats

# Tag ONE dataset synchronously — real LLM call, does NOT write to MongoDB (safe smoke test)
docker exec tagger-api curl -sS -X POST http://localhost:8000/api/v1/tag \
  -H 'Content-Type: application/json' \
  -d '{"dataset_id":"ds002718","source_url":"https://github.com/OpenNeuroDatasets/ds002718.git"}'

# Async: enqueue → worker tags → WRITES tags+tagger_meta to MongoDB
docker exec tagger-api curl -sS -X POST http://localhost:8000/api/v1/tag/enqueue \
  -H 'Content-Type: application/json' \
  -d '{"dataset_id":"dsXXXXXX","source_url":"https://github.com/OpenNeuroDatasets/dsXXXXXX.git","metadata_snapshot":{"title":"...","readme":"..."}}'

# Bulk-enqueue datasets from the EEGDash API (worker drains them)
docker exec tagger-api python scripts/enqueue_all_datasets.py --help   # --limit N, --dry-run, --skip-ground-truth

# Inspect a job + verify the Mongo write
docker exec tagger-api curl -sS http://localhost:8000/api/v1/queue/status/dsXXXXXX
docker exec tagger-api curl -sS http://eegdash-api:3000/api/eegdash/datasets/dsXXXXXX

# Logs
docker logs -f tagger-worker
```

## Lifecycle
```bash
docker-compose -p eegdash-llm-tagger -f deploy/docker-compose.prod.yml ps
docker-compose -p eegdash-llm-tagger -f deploy/docker-compose.prod.yml restart worker
docker-compose -p eegdash-llm-tagger -f deploy/docker-compose.prod.yml down        # stop (keeps queue + cache)
docker-compose -p eegdash-llm-tagger -f deploy/docker-compose.prod.yml down -v      # ⚠ also deletes queued jobs + LLM cache
```

## Gotchas (read once)
- **One worker only** — `cache.py` uses a single shared `.tmp` file; multiple writers corrupt the cache.
- **A failed Mongo write still marks the job `completed`** — monitor `result.mongodb_updated`, not just `status`.
- **Cost** — `/api/v1/tag` and `/tag/enqueue` spend the OpenRouter key and have no app-level auth. Cap the key (daily limit) and keep these endpoints internal-only or behind the Caddy X-API-Key gate.
- **`config_hash`** changes when `vendor-config/few_shot_examples.json` or the model changes, which re-tags on next run.

## Optional: public HTTPS (`tagger.eegdash.org`)
Only needed to call the API from outside the host. Add DNS `A tagger.eegdash.org → 169.228.38.92`,
then append `caddy/tagger.eegdash.org.caddy` (set a real `X-API-Key`) to the host Caddyfile and
`caddy validate && caddy reload`. The internal batch workflow above needs none of this.
