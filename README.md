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

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter.ai API key |
| `LLM_MODEL` | No | `openai/gpt-4-turbo` | LLM model to use |
| `CACHE_DIR` | No | `/tmp/eegdash-llm-api/cache` | Cache directory path |
| `FEW_SHOT_PATH` | No | (library default) | Override few-shot examples path |
| `PROMPT_PATH` | No | (library default) | Override prompt file path |

## Architecture

See [ARCHITECTURE.md](https://github.com/kuntalkokate/eegdash-llm-tagger/blob/main/ARCHITECTURE.md) in the core library repository for detailed system architecture.
