# EEGDash LLM API

REST API service for tagging EEG/MEG datasets using LLM-based classification. This service wraps the [eegdash-llm-tagger](https://github.com/kuntalkokate/eegdash-llm-tagger) library with caching and orchestration.

## Prerequisites

- Docker and Docker Compose installed on the server
- OpenRouter API key ([get one here](https://openrouter.ai/))
- Git access to both repositories

## Quick Start

### 1. Clone the repositories

```bash
# Clone this API repository
git clone https://github.com/kuntalkokate/eegdash-llm-api.git
cd eegdash-llm-api

# The core library will be installed automatically via pip during Docker build
```

### 2. Configure environment

```bash
# Create environment file
cp .env.example .env

# Edit with your API key
nano .env
```

Required variables:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Optional variables:
```bash
LLM_MODEL=openai/gpt-4-turbo    # Default model
CACHE_DIR=/tmp/eegdash-llm-api/cache  # Cache location
```

### 3. Build and run

```bash
# Build and start the container
docker-compose up -d --build

# Check logs
docker-compose logs -f

# Verify it's running
curl http://localhost:8000/health
```

### 4. Test the API

```bash
# Tag a dataset
curl -X POST http://localhost:8000/api/v1/tag \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "ds001234",
    "source_url": "https://github.com/OpenNeuroDatasets/ds001234"
  }'
```

## API Endpoints

### Tag a Dataset

```bash
POST /api/v1/tag
```

Request body:
```json
{
  "dataset_id": "ds001234",
  "source_url": "https://github.com/OpenNeuroDatasets/ds001234",
  "force_refresh": false
}
```

Response:
```json
{
  "dataset_id": "ds001234",
  "pathology": ["Healthy"],
  "modality": ["Visual"],
  "type": ["Perception"],
  "confidence": {
    "pathology": 0.95,
    "modality": 0.85,
    "type": 0.80
  },
  "reasoning": {
    "few_shot_analysis": "...",
    "metadata_analysis": "...",
    "decision_summary": "..."
  },
  "from_cache": false,
  "stale": false
}
```

### Get Cached Tags

```bash
GET /api/v1/tags/{dataset_id}
```

Returns cached tags without making an LLM call. Returns 404 if not cached.

### Cache Management

```bash
# Get cache statistics
GET /api/v1/cache/stats

# List all cache entries
GET /api/v1/cache/entries

# List entries for specific dataset
GET /api/v1/cache/entries?dataset_id=ds001234

# Clear entire cache
DELETE /api/v1/cache

# Delete specific entry
DELETE /api/v1/cache/{cache_key}
```

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "cache_entries": 42,
  "config_hash": "a1b2c3d4e5f6g7h8"
}
```

## Docker Commands

### Start/Stop

```bash
# Start in background
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart
```

### View Logs

```bash
# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail 100
```

### Rebuild

```bash
# Rebuild after code changes
docker-compose up -d --build

# Force rebuild without cache
docker-compose build --no-cache
docker-compose up -d
```

### Shell Access

```bash
# Enter running container
docker-compose exec tagger-api bash

# Check installed packages
docker-compose exec tagger-api pip list | grep eegdash
```

## Production Deployment

### Using a Persistent Volume

The default `docker-compose.yml` uses a named volume for cache persistence:

```yaml
volumes:
  - cache-data:/tmp/eegdash-llm-api/cache

volumes:
  cache-data:
```

### Using Host Directory (Alternative)

To mount a host directory instead:

```yaml
volumes:
  - /path/on/host/cache:/tmp/eegdash-llm-api/cache
```

### Running on a Specific Port

```bash
# Change port in docker-compose.yml or use -p flag
docker run -p 9000:8000 eegdash-llm-api
```

### Behind a Reverse Proxy (nginx)

```nginx
server {
    listen 80;
    server_name tagger.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Updating the Core Library

When the `eegdash-llm-tagger` library is updated (new prompts, few-shot examples):

```bash
# Rebuild to pull latest library version
docker-compose build --no-cache
docker-compose up -d

# The config_hash will change automatically
# Old cache entries will be ignored, fresh LLM calls will be made
```

## Troubleshooting

### Container won't start

```bash
# Check logs for errors
docker-compose logs

# Common issues:
# - Missing OPENROUTER_API_KEY
# - Port 8000 already in use
# - Docker daemon not running
```

### API returns errors

```bash
# Check health endpoint
curl http://localhost:8000/health

# Check if API key is valid
curl http://localhost:8000/api/v1/tag \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "test", "source_url": "https://github.com/OpenNeuroDatasets/ds000001"}'
```

### Cache issues

```bash
# View cache stats
curl http://localhost:8000/api/v1/cache/stats

# Clear cache if corrupted
curl -X DELETE http://localhost:8000/api/v1/cache

# Force refresh a specific dataset
curl -X POST http://localhost:8000/api/v1/tag \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "ds001234", "source_url": "...", "force_refresh": true}'
```

### View cache files directly

```bash
# Enter container
docker-compose exec tagger-api bash

# View cache
cat /tmp/eegdash-llm-api/cache/tagging_cache.json | python -m json.tool
```

## Testing the Async Queue Workflow (Local Development)

The async workflow uses Postgres for job queuing and updates MongoDB via HTTP API. This is useful for batch processing datasets without blocking.

### Prerequisites

1. **Postgres** running locally (for job queue)
2. **EEGDash API** access (for MongoDB updates via HTTP)
3. **OpenRouter API key**

### Step 1: Start Postgres

```bash
# Start Postgres in Docker
docker run -d \
  --name eegdash-postgres \
  -e POSTGRES_USER=eegdash \
  -e POSTGRES_PASSWORD=eegdash123 \
  -e POSTGRES_DB=eegdash_queue \
  -p 5432:5432 \
  postgres:15-alpine
```

### Step 2: Configure Environment

Create/update `.env` in `eegdash-llm-api/`:

```bash
# Required: OpenRouter API key
OPENROUTER_API_KEY="your-openrouter-api-key"

# Required: Postgres URL for job queue
POSTGRES_URL=postgresql://eegdash:eegdash123@localhost:5432/eegdash_queue

# Required: EEGDash API for MongoDB updates (HTTP mode - safer)
EEGDASH_API_URL=https://data.eegdash.org
EEGDASH_ADMIN_TOKEN=your-admin-token

# Optional: Override config paths
FEW_SHOT_PATH=/path/to/eegdash-llm-tagger/data/processed/few_shot_examples.json
PROMPT_PATH=/path/to/eegdash-llm-tagger/prompt.md

# Optional: LLM model
LLM_MODEL=openai/gpt-4-turbo
```

### Step 3: Start the API Server

```bash
cd eegdash-llm-api
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 4: Verify Queue is Enabled

```bash
curl http://localhost:8000/health
```

Expected response should include `"queue_enabled": true`.

### Step 5: Enqueue a Tagging Job

```bash
curl -X POST http://localhost:8000/api/v1/tag/enqueue \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "ds003645",
    "source_url": "https://github.com/OpenNeuroDatasets/ds003645"
  }'
```

Response:
```json
{
  "success": true,
  "job_id": 1,
  "dataset_id": "ds003645",
  "message": "Job queued for processing"
}
```

### Step 6: Check Queue Status

```bash
curl http://localhost:8000/api/v1/queue/stats
```

Response:
```json
{
  "pending": 1,
  "processing": 0,
  "completed": 0,
  "failed": 0,
  "ready_to_process": 1
}
```

### Step 7: Run the Worker (Process Jobs)

In a separate terminal:

```bash
cd eegdash-llm-api

# Run worker to process one job
uv run python -c "
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from src.services.worker import TaggingWorker

async def run_once():
    worker = TaggingWorker(
        postgres_url=os.environ['POSTGRES_URL'],
        eegdash_api_url=os.environ['EEGDASH_API_URL'],
        eegdash_admin_token=os.environ['EEGDASH_ADMIN_TOKEN'],
        openrouter_api_key=os.environ['OPENROUTER_API_KEY'],
        model=os.environ.get('LLM_MODEL', 'openai/gpt-4-turbo'),
        worker_id='test-worker',
    )

    await worker.initialize()

    job = await worker._queue.claim_job(worker.worker_id)
    if job:
        print(f'Processing job {job.id} for {job.dataset_id}')
        await worker._process_job(job)
        print('Job processed!')
    else:
        print('No jobs in queue')

    await worker.shutdown()

asyncio.run(run_once())
"
```

### Step 8: Verify MongoDB Update

```bash
# Check the dataset in MongoDB (read-only GET)
curl -s "https://data.eegdash.org/api/eegdash/datasets/ds003645" | python3 -m json.tool
```

You should see new `tags` and `tagger_meta` fields:

```json
{
  "data": {
    "dataset_id": "ds003645",
    "tags": {
      "pathology": ["Healthy"],
      "modality": ["Visual"],
      "type": ["Perception"],
      "confidence": { "pathology": 0.7, "modality": 0.9, "type": 0.9 },
      "reasoning": { ... }
    },
    "tagger_meta": {
      "config_hash": "17f35c0d70bcdd4e",
      "metadata_hash": "99822e9d4040ec4f",
      "model": "openai/gpt-4-turbo",
      "tagged_at": "2026-01-20T08:04:10.118641+00:00"
    }
  }
}
```

### Running Worker Continuously

To run the worker as a continuous process (processes jobs as they arrive):

```bash
cd eegdash-llm-api
uv run python -m src.services.worker
```

The worker will:
- Poll the queue every 5 seconds
- Process jobs as they arrive
- Update MongoDB via HTTP API (safe - checks dataset exists first)
- Handle graceful shutdown on Ctrl+C

### Queue Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/tag/enqueue` | POST | Add job to queue |
| `/api/v1/queue/stats` | GET | Queue statistics |
| `/api/v1/queue/jobs` | GET | List queued jobs |
| `/api/v1/queue/jobs/{job_id}` | GET | Get specific job |

### Cleanup

```bash
# Stop Postgres
docker stop eegdash-postgres
docker rm eegdash-postgres
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter.ai API key |
| `LLM_MODEL` | No | `openai/gpt-4-turbo` | LLM model to use |
| `CACHE_DIR` | No | `/tmp/eegdash-llm-api/cache` | Cache directory path |
| `FEW_SHOT_PATH` | No | (library default) | Override few-shot examples path |
| `PROMPT_PATH` | No | (library default) | Override prompt file path |
| `POSTGRES_URL` | No* | - | Postgres URL for async queue (*required for async mode) |
| `EEGDASH_API_URL` | No* | - | EEGDash API URL (*required for worker HTTP mode) |
| `EEGDASH_ADMIN_TOKEN` | No* | - | Admin token for MongoDB writes (*required for worker HTTP mode) |
| `MONGODB_URL` | No* | - | Direct MongoDB URL (*alternative to HTTP mode) |

## Architecture

See [ARCHITECTURE.md](https://github.com/kuntalkokate/eegdash-llm-tagger/blob/main/ARCHITECTURE.md) in the core library repository for detailed system architecture.
