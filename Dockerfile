FROM python:3.11-slim

# Install git for cloning repositories
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml .

# Install dependencies (includes eegdash-llm-tagger library with bundled configs)
RUN pip install --no-cache-dir -e .

# Copy application code
COPY src/ src/

# Create cache directory (will be mounted as volume in production)
RUN mkdir -p /tmp/eegdash-llm-api/cache

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
