# Tested Python base; update the digest deliberately for a new release.
FROM python:3.11-slim@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Exported from uv.lock, including the immutable library source revision.
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
# .dockerignore excludes credentials, VCS state, caches and build output.
COPY . ./
RUN pip install --no-cache-dir --no-deps . && mkdir -p /data/cache
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
