# EEGDash LLM Tagger — Deployment Plan (`tagger.eegdash.org` on `indexing`)

**Status: DEPLOYED (2026-06-18) — Layers 1–5 green; only optional public exposure remains.**
The stack is live on `indexing` (`tagger-postgres`/`tagger-api`/`tagger-worker`, `restart:
unless-stopped`). The async pipeline was validated end-to-end on production (ds002718 →
LLM `gpt-5.2` → MongoDB write with `tagger_meta`). Day-to-day commands: [`README.md`](./README.md).

This plan is reconciled against the actual artifacts in `deploy/` (those files are the
source of truth) and incorporates an adversarial gap-review.

### Decisions taken (as deployed)
- **Edge auth:** Caddy **`X-API-Key`** gate; enqueue/batch + `DELETE /cache` internal-only. (Not yet wired — gated on DNS.)
- **Write-back auth:** reuse **`ADMIN_TOKEN`**. (Investigated: the EEGDash API accepts only
  `ADMIN_TOKEN` or `CI_TOKEN`, both full-privilege — no truly scoped token exists without a
  competition-side code change.)
- **Model:** `openai/gpt-5.2` (from the operator's `.env`). Set a daily cap on the OpenRouter key.
- **Build fixes applied** (latent bugs in the repo): `Dockerfile.prod` COPYs `README.md`
  (hatchling needs it), installs `httpx` (runtime dep mis-filed under `[dev]`), and bakes in
  `vendor-config/{few_shot_examples.json,prompt.md}` (not shipped in the tagger wheel).
- **Open / optional:** DNS `A tagger.eegdash.org → 169.228.38.92` + Caddy block for public
  HTTPS. The internal batch workflow needs none of this.

---

## 1. Executive summary

We deploy the existing service **unchanged** (no new app code) as a **separate
docker-compose project** at `/home/sccn/eegdash-llm-api` on `indexing.ucsd.edu`:

- `tagger-postgres` — durable job queue (internal only)
- `tagger-api` — FastAPI (sync `/tag`, async enqueue, cache endpoints)
- `tagger-worker` — claims jobs, tags, writes `tags`/`tagger_meta` back to MongoDB

It joins the **existing** `eegdash-competition_backend` network so Caddy can front it
and the worker can reach the EEGDash API in-process. The production competition
compose file is **never edited**; the only shared change is **one appended Caddy site
block**, reloaded atomically. Public hostname: `tagger.eegdash.org` (new DNS record).

**Three decisions you must make** (details in §9): (1) how public/authenticated the
endpoints are; (2) whether the worker reuses the production `ADMIN_TOKEN` or gets a
separate write-scoped token; (3) the model + OpenRouter spend cap. The OpenRouter key
stays **server-provided** (user-provided would require new code you've excluded).

---

## 2. Analysis

**Two modes, very different risk profiles:**
- **Sync** (`POST /api/v1/tag`): clones the dataset repo, calls the LLM, caches, returns
  tags. **Spends money in-request. Never touches MongoDB.** → safe as a non-destructive
  production smoke test.
- **Async** (`/tag/enqueue` → Postgres → worker): the worker tags from the snapshot and
  **writes back to the production MongoDB** via the EEGDash admin endpoint. This is the
  only path that mutates real data.

**Build fix (mandatory):** `pyproject.toml` pins `eegdash-tagger` to a `file://` path that
exists only on a dev laptop. `deploy/Dockerfile.prod` sed-rewrites it to
`git+https://github.com/eegdash/eegdash-llm-tagger.git@main` so the image builds on any
host. Both repos are public.

**Host facts (verified):** Caddy (auto-HTTPS) routes by **container name** on
`eegdash-competition_backend`; `data.eegdash.org` → `eegdash-api:3000` →
`mongodb-production` all live on this box; `GET http://eegdash-api:3000/api/eegdash/datasets/ds002718`
returns `{success:true,data:{…}}`; Docker runs without sudo; **no passwordless sudo**;
`tagger.eegdash.org` does **not** resolve yet.

**Load-bearing behavior to monitor:** the worker marks a job `completed` **even if the
Mongo write-back fails** (`result.mongodb_updated:false`). Monitor `result.mongodb_updated`,
not just `status`.

---

## 3. Target architecture & Docker design

```
                         Internet (HTTPS)
                              │
                  ┌───────────▼────────────┐
                  │  Caddy (existing)       │  auto-TLS; routes by container name
                  │  +1 site block:         │  edge X-API-Key gate (anti-abuse)
                  │  tagger.eegdash.org     │
                  └───────────┬────────────┘
   network: eegdash-competition_backend (external, NOT created/destroyed by us)
        ┌─────────────────────┼───────────────────────────────┐
        │                     ▼                                │
        │              ┌────────────┐      ┌────────────────┐  │
        │              │ tagger-api │      │ tagger-worker  │  │
        │              │  :8000     │      │ (single replica│  │
        │              └─────┬──────┘      │  no host port) │  │
        │   cache-api vol ───┘             └───┬────────┬───┘  │
        │                                cache-worker   │ HTTP write-back
        │                                   vol         ▼  http://eegdash-api:3000
        │   ┌────────────────┐                      ┌──────────────┐
        │   │ tagger-postgres│◄── claim jobs ───────┤  eegdash-api │→ mongodb-production
        │   │  (internal)    │                      └──────────────┘   (existing prod)
        │   │  pg-data vol   │
        │   └────────────────┘
        └───────────────────────────────────────────────────────────┘
   network: internal (our own bridge) — postgres ↔ api/worker only
```

- **Networks:** `internal` (our bridge; postgres + api + worker) and `edge`
  (`external: true`, `name: eegdash-competition_backend`). `api`+`worker` are on both;
  `postgres` is **internal-only**.
- **Volumes:** `pg-data` (queue), **`cache-api`** and **`cache-worker`** — *separate*.
  Reason: `cache.py` persists via a single shared `<cache>.tmp` filename + rename; two
  processes on the same volume race on that temp file and can corrupt the cache. Ground
  truth is prepopulated into **`cache-worker`**.
- **Single worker.** `--scale worker=N` (N>1) reintroduces the same `.tmp` collision
  between replicas. Safe scaling needs a code-level cache fix (out of scope).
- **Image:** one `eegdash-llm-tagger:latest` built from `deploy/Dockerfile.prod`; the
  worker overrides the command. Resource limits + healthchecks (postgres, api) set.

---

## 4. Endpoints (danger & exposure)

| Endpoint | Spends $ | Mutates | Exposure |
|---|---|---|---|
| `POST /api/v1/tag` | **Yes (in-request LLM)** | cache only (no Mongo) | gate at edge |
| `POST /api/v1/tag/enqueue` / `…/batch` | deferred (worker) → **Mongo write** | queue + Mongo via worker | **internal-only** (batch is unbounded) |
| `DELETE /api/v1/cache` / `…/{key}` | no | **destructive** (wipes cache/ground truth) | **internal-only** |
| `GET /api/v1/tags/{id}`, `/cache/stats`, `/cache/entries`, `/queue/*`, `/health`, `/` | no | no | read-only (leaks dataset IDs) |

**No endpoint has app-level auth.** Public protection is the Caddy `X-API-Key` gate.
**Caveat:** any container on `eegdash-competition_backend` can reach `tagger-api:8000`
directly, bypassing Caddy — so the destructive/SSRF endpoints are reachable by the whole
production network. That is the real trust boundary; accept it explicitly (§9.4).

---

## 5. Implementation steps (ordered, each reversible)

0. **Gate (local):** `uv run pytest` (89 pass) + `uv run ruff check .`. Confirm the sed
   target still exists: `grep -E '"eegdash-tagger @ file://[^"]*"' pyproject.toml`.
1. **Get code to host:** `git clone https://github.com/eegdash/eegdash-llm-api.git
   /home/sccn/eegdash-llm-api`, then copy these `deploy/` artifacts over the clone (the
   public repo doesn't have them yet): `Dockerfile.prod`, `docker-compose.prod.yml`,
   `.env.prod.example`, `caddy/tagger.eegdash.org.caddy`. (Later: commit them + optionally
   a GHCR build so the host just pulls an image.)
2. **Secrets:** `cp deploy/.env.prod.example deploy/.env && chmod 600 deploy/.env`; fill
   `OPENROUTER_API_KEY` (dedicated capped key), `POSTGRES_PASSWORD` (`openssl rand -base64 32`),
   `EEGDASH_ADMIN_TOKEN`, `LLM_MODEL`.
3. **Build + verify image** (§7 Layer 2): `docker build -f deploy/Dockerfile.prod -t eegdash-llm-tagger:latest .`
   then `pip show eegdash-tagger` + import checks.
4. **Bring up (internal):** `docker-compose -p eegdash-llm-tagger -f deploy/docker-compose.prod.yml --env-file deploy/.env up -d --build`.
   Verify health + `queue_enabled:true` + worker "waiting for jobs" via `docker exec` (§7 Layer 3). **No public surface yet.**
5. **Functional smoke (§7 Layer 4):** one non-destructive `POST /api/v1/tag` for `ds002718`
   from inside the network. Proves key + tagger + clone + cache, mutates nothing.
6. **DNS (you):** create `tagger.eegdash.org A 169.228.38.92`; verify `dig +short` from
   host **and** externally **before** the next step.
7. **Caddy (only after DNS resolves):** back up the Caddyfile, append
   `deploy/caddy/tagger.eegdash.org.caddy` (set the `X-API-Key`), then `caddy validate` →
   `caddy reload`. Verify `https://tagger.eegdash.org/health`.
8. **Consented end-to-end (§7 Layer 5):** enqueue one dataset, watch the worker write back,
   verify in Mongo, keep the revert payload.

---

## 6. Security & the OpenRouter key (the central question)

**Keep the key SERVER-PROVIDED.** User-provided (per-request) keys require new code you've
excluded and fit a multi-tenant SaaS, not an internal batch tagger you own. Shrink blast
radius with **infra-only** controls (confirmed against current OpenRouter docs):

1. **Dedicated, capped OpenRouter key.** Create a key used only here with a hard daily cap
   (`limit` + `limit_reset: daily`, resets midnight UTC); keep a modest prepaid balance
   (negative balance → `402`, a hard stop). OpenRouter does **not** per-minute rate-limit a
   paid key, so this cap *is* your spend backstop. Don't enable BYOK.
2. **Secret hygiene.** `.env` `chmod 600`, owner `sccn`, gitignored, **never** in the image
   (`Dockerfile.prod` carries no secrets) or logs. Rotation: create new key → update `.env`
   → recreate `tagger-api`/`tagger-worker` → delete old key.
3. **Edge auth.** Gate `tagger.eegdash.org` with the `X-API-Key` check (or basicauth like
   Grafana) so anonymous internet traffic can't spend the key. Keep `enqueue`/`batch` and
   `DELETE /cache` **internal-only**.
4. **ADMIN_TOKEN risk.** The worker's `EEGDASH_ADMIN_TOKEN` is the production write
   credential. **Prefer a separate write-scoped token**; if reused, rotation must be
   co-sequenced with the competition `.env`. It is visible via `docker inspect tagger-worker`.
5. **SSRF / lateral note.** `POST /api/v1/tag` git-clones an arbitrary `source_url`
   (server-side fetch). With the edge gate + internal-only enqueue this is bounded, but the
   shared-network bypass (§4) means it should be signed off, not glossed.

---

## 7. Testing & production verification

Six layers; 1–4 mutate nothing, 5 writes one Mongo doc (consent), 6 is rollback.

| Layer | Proves | Mutates prod? | Pass gate |
|---|---|---|---|
| 1 Local | code + lint | No | `89 passed`, ruff clean |
| 2 Build | tagger installs from git, imports OK | No | `pip show` git install; import checks OK |
| 3 Smoke (internal) | stack up, Postgres wired | No | 200 on `/health` `/` `/queue/stats`; `queue_enabled:true`; worker draining |
| 4 **1-LLM-call prod test** | key + tagger + clone + cache | **No** | 1st call `from_cache:false,error:null`; repeat `from_cache:true` |
| 5 End-to-end | worker → Mongo write-back | **Yes (1 doc)** | job `completed`, **`mongodb_updated:true`**, fresh `tagger_meta.tagged_at`; revert verified |
| 6 Rollback | clean teardown | No | tagger gone; `data.eegdash.org`+`/live` still serving |

**Layer 3 — internal smoke (before DNS):**
```bash
docker exec caddy wget -qO- http://tagger-api:8000/health        # status:healthy, queue_enabled:true
docker exec caddy wget -qO- http://tagger-api:8000/api/v1/queue/stats   # 200 => Postgres wired
docker logs --tail=30 tagger-worker                              # "started, waiting for jobs..."
```

**Layer 4 — the simple 1-LLM-call production test (NON-DESTRUCTIVE):**
```bash
docker exec tagger-api curl -sS -X POST http://localhost:8000/api/v1/tag \
  -H 'Content-Type: application/json' \
  -d '{"dataset_id":"ds002718","source_url":"https://github.com/OpenNeuroDatasets/ds002718.git","force_refresh":false}'
```
Expect a `TagResponse` with `pathology/modality/type`, `from_cache:false`, `error:null`
(30–60s); a repeat returns `from_cache:true` instantly. **No MongoDB write occurs** — the
API process never builds a Mongo updater (only the worker does).

**Worker liveness (don't trust "started"):** the worker has no healthcheck, so a crash-loop
(bad token/URL) is invisible while jobs pile up. After enqueuing the Layer-5 job, confirm it
leaves `pending` and `queue/stats` `completed` advances; otherwise read `docker logs tagger-worker`.

**Layer 5 safety:** capture `GET /api/eegdash/datasets/ds002718` before; the write is `$set`
of `tags`+`tagger_meta` on an existing doc (`upsert=false`, existence-checked — never creates
docs). Keep the before-state to re-POST as a revert. If the doc had no prior tags, `$set`
can't `$unset`; overwrite-to-empty or remove directly in Mongo (operator decision).

---

## 8. Rollback / teardown (competition stack untouched)

- **Remove Caddy block:** restore the backed-up Caddyfile (delete only the
  `tagger.eegdash.org {…}` block) → `caddy validate` → `caddy reload`. A bad validate is
  never reloaded; `data.eegdash.org` is never interrupted. Verify `/live` still serves.
- **Stop tagger stack:** `docker-compose -p eegdash-llm-tagger -f deploy/docker-compose.prod.yml down`.
  `down` removes only this project's resources; the `external` network is **not** deleted.
- **`down -v` WARNING:** also deletes `pg-data` (queued/in-flight jobs) and the cache volumes
  (forces full LLM re-spend next run). Only use after the queue is drained.

---

## 9. Open decisions (each with a recommended default)

1. **Code delivery** — *Default: clone repo on host + overlay `deploy/` files + build.*
   Follow-up: commit `deploy/` and add a GHCR Actions build so the host just pulls an image.
2. **Public exposure & edge auth** — ✅ *CHOSEN: expose `tagger.eegdash.org` behind the
   Caddy `X-API-Key` gate; keep enqueue/batch + `DELETE /cache` internal-only.*
3. **OpenRouter key** — *Default: server-provided, dedicated key, daily cap.* (User-provided
   = new code, rejected.) You set the cap value and the model (item 5).
4. **Shared-network exposure** — *Default: accept that prod containers can reach `tagger-api`
   unauthenticated (sign-off).* True isolation needs app auth (new code) or editing the
   competition Caddy/compose (agreed off-limits).
5. **Model + spend cap** — *Default: do NOT silently ship `gpt-4-turbo`.* Pick a current,
   cost-appropriate model and set the OpenRouter daily cap to expected datasets/day.
6. **ADMIN_TOKEN** — ✅ *CHOSEN: separate write-scoped token for the worker.* Prerequisite
   (blocking for Layer 5): confirm the EEGDash API accepts a token other than `ADMIN_TOKEN`
   for `POST /admin/eegdash/datasets` (check `ADMIN_TOKEN` vs `CI_TOKEN` handling in the
   competition API), or arrange a competition-side change to honor a new token.
7. **DNS** — you create `tagger.eegdash.org A 169.228.38.92`; the Caddy block goes in only
   after it resolves (strict ordering, to avoid Let's Encrypt rate limits).
8. **SSH access for execution** — key is set up; I can drive the remote, or hand you the
   exact command blocks to run.
