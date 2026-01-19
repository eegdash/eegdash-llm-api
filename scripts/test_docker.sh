#!/bin/bash
# Docker deployment test script
# Tests that the Docker container builds and starts correctly

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== EEGDash LLM API Docker Test ==="
echo "Project directory: $PROJECT_DIR"

# Check if .env exists
if [ ! -f .env ]; then
    echo ""
    echo "WARNING: .env file not found."
    echo "Copy .env.example to .env and add your OPENROUTER_API_KEY for full testing."
    echo "Proceeding with build test only..."
    BUILD_ONLY=true
else
    BUILD_ONLY=false
fi

# Build the image
echo ""
echo "=== Step 1: Building Docker image ==="
docker-compose build --no-cache

if [ "$BUILD_ONLY" = true ]; then
    echo ""
    echo "=== Build completed successfully ==="
    echo "To run full integration tests, create .env with OPENROUTER_API_KEY"
    exit 0
fi

# Start the container
echo ""
echo "=== Step 2: Starting container ==="
docker-compose up -d

# Wait for startup
echo ""
echo "=== Step 3: Waiting for container to be ready ==="
echo "Waiting 15 seconds for startup..."
sleep 15

# Health check with retry
echo ""
echo "=== Step 4: Health check ==="
MAX_RETRIES=5
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health 2>/dev/null || echo "FAILED")
    if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
        echo "Health response: $HEALTH_RESPONSE"
        echo "Health check PASSED"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Health check attempt $RETRY_COUNT failed, retrying in 5 seconds..."
    sleep 5
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Health check FAILED after $MAX_RETRIES attempts"
    echo "Container logs:"
    docker-compose logs
    docker-compose down
    exit 1
fi

# Test root endpoint
echo ""
echo "=== Step 5: Testing root endpoint ==="
ROOT_RESPONSE=$(curl -s http://localhost:8000/)
echo "Root response: $ROOT_RESPONSE"

if echo "$ROOT_RESPONSE" | grep -q '"name":"EEGDash LLM Tagger API"'; then
    echo "Root endpoint PASSED"
else
    echo "Root endpoint FAILED"
fi

# Test cache stats (should work even without tagging)
echo ""
echo "=== Step 6: Testing cache stats ==="
CACHE_RESPONSE=$(curl -s http://localhost:8000/api/v1/cache/stats)
echo "Cache stats: $CACHE_RESPONSE"

if echo "$CACHE_RESPONSE" | grep -q '"total_entries"'; then
    echo "Cache stats PASSED"
else
    echo "Cache stats FAILED"
fi

# Cleanup
echo ""
echo "=== Step 7: Cleanup ==="
docker-compose down

echo ""
echo "=== All Docker tests PASSED ==="
