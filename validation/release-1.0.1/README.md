# Service 1.0.1 release validation

`summary.json` records the exact pinned library, executed environments, test counts,
coverage and limits. `source-sha256.json` identifies the tested runtime/test files;
the containing Git commit identifies the release candidate. All three runs passed
136 cases (the same suite, not 408 distinct tests). Locked macOS and Linux runs
covered 751/1037 source lines (72.42%). The normal wheel install used newly resolved
runtime dependencies and imported service/library from site-packages outside the
checkout. The frozen tests and Docker build used `uv.lock`/`requirements.lock`.

The actual image build used `deploy/Dockerfile.prod`; `Dockerfile` has identical
bytes. The loopback-only API health check passed. Real PostgreSQL tests each created
and dropped a unique schema. Provider and catalogue endpoints were synthetic HTTP
fixtures. No live model calls, production writes or production deployment occurred.
Direct MongoDB network I/O remains mocked. The independent review also reran the
PostgreSQL integration and the installed-wheel smoke checks.

Reproduce the locked run with `uv sync --frozen --extra dev`, then
`TEST_POSTGRES_URL=<disposable database> uv run --frozen --extra dev pytest --cov=src`.
Without `TEST_POSTGRES_URL`, the real PostgreSQL test is explicitly skipped. Build
with `docker build -f deploy/Dockerfile.prod .`. The environment JSON files record
package versions and the installed library's public Git provenance. JUnit hostnames
were removed; results/timestamps were retained. GitHub CI was configured but is not
claimed to have run here.
