#!/bin/bash
# Integration test with a real dataset
# NOTE: Requires OPENROUTER_API_KEY to be set in .env

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== Integration Test - Real Dataset ==="

# Check for .env file
if [ ! -f .env ]; then
    echo "ERROR: .env file not found."
    echo "Create .env with OPENROUTER_API_KEY to run integration tests."
    exit 1
fi

# Source .env to check for API key
source .env

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set in .env"
    echo "Add your OpenRouter API key to .env to run integration tests."
    exit 1
fi

# Start container
echo ""
echo "=== Starting Docker container ==="
docker-compose up -d

# Wait for startup
echo "Waiting 15 seconds for startup..."
sleep 15

# Verify health
echo ""
echo "=== Verifying API health ==="
HEALTH=$(curl -s http://localhost:8000/health)
echo "Health: $HEALTH"

if ! echo "$HEALTH" | grep -q '"status":"healthy"'; then
    echo "API not healthy, aborting"
    docker-compose logs
    docker-compose down
    exit 1
fi

# Test with a small, well-known dataset (ds002718 is small and stable)
echo ""
echo "=== Tagging ds002718 (small dataset) ==="
echo "This will make a real LLM API call and may take 30-60 seconds..."

TAG_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/tag \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "ds002718",
    "source_url": "https://github.com/OpenNeuroDatasets/ds002718"
  }')

echo ""
echo "Tag response:"
echo "$TAG_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$TAG_RESPONSE"

# Verify response structure
if echo "$TAG_RESPONSE" | grep -q '"pathology"'; then
    echo ""
    echo "Tagging test PASSED - valid response structure"
else
    echo ""
    echo "Tagging test FAILED - invalid response structure"
    docker-compose down
    exit 1
fi

# Verify caching works
echo ""
echo "=== Testing cache (second request should be cached) ==="
TAG_RESPONSE2=$(curl -s -X POST http://localhost:8000/api/v1/tag \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "ds002718",
    "source_url": "https://github.com/OpenNeuroDatasets/ds002718"
  }')

if echo "$TAG_RESPONSE2" | grep -q '"from_cache":true'; then
    echo "Cache test PASSED - second request was cached"
else
    echo "Cache test WARNING - second request was not cached"
    echo "Response: $TAG_RESPONSE2"
fi

# Check cache stats
echo ""
echo "=== Cache statistics ==="
STATS=$(curl -s http://localhost:8000/api/v1/cache/stats)
echo "Cache stats: $STATS"

# Test force refresh
echo ""
echo "=== Testing force_refresh ==="
TAG_RESPONSE3=$(curl -s -X POST http://localhost:8000/api/v1/tag \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "ds002718",
    "source_url": "https://github.com/OpenNeuroDatasets/ds002718",
    "force_refresh": true
  }')

if echo "$TAG_RESPONSE3" | grep -q '"from_cache":false'; then
    echo "Force refresh test PASSED"
else
    echo "Force refresh test WARNING"
fi

# Cleanup
echo ""
echo "=== Cleanup ==="
docker-compose down

echo ""
echo "=== All integration tests completed ==="
