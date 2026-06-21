# Connection Pooling Plan

## Problem

`DocumentStore._conn()` opens a new `psycopg2.connect()` call on every request and closes it after. Each PostgreSQL connection costs ~10MB and takes tens of milliseconds to establish. Under concurrent load on Cloud Run (concurrency=80), this will exhaust Supabase's connection limit and add unnecessary latency.

## Solution

Replace per-request `psycopg2.connect()` with a `psycopg2.pool.ThreadedConnectionPool` singleton, initialized at startup and closed at shutdown.

## Changes

### 1. Add pool config to `app/config.py`

```python
# Connection Pool
db_pool_min_conn: int = 2
db_pool_max_conn: int = 10
```

### 2. Replace `_conn()` in `app/document_store.py`

Replace the per-request connect/close pattern with a pool that checks out and returns connections:

```python
from psycopg2.pool import ThreadedConnectionPool

class DocumentStore:
    def __init__(self, settings: Settings) -> None:
        self._dsn = settings.supabase_database_url
        self._pool = ThreadedConnectionPool(
            minconn=settings.db_pool_min_conn,
            maxconn=settings.db_pool_max_conn,
            dsn=self._dsn,
        )
        ...

    @contextmanager
    def _conn(self) -> Generator:
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def close(self):
        if self._pool:
            self._pool.closeall()
```

No other methods need to change — they all go through `_conn()`.

### 3. Wire shutdown in `app/main.py` lifespan

```python
yield  # App is running

# Shutdown
if document_store:
    document_store.close()
logger.info("Shutting down...", extra={"extra_data": metrics.summary})
```

### 4. Use Supabase pooler port in production

The `SUPABASE_DATABASE_URL` secret in Cloud Run should point to port **6543** (Supabase's PgBouncer), not 5432 (direct). This gives two layers of pooling:
- **App-level** (psycopg2 ThreadedConnectionPool): reuses connections across concurrent requests within a single Cloud Run instance
- **Supabase-level** (PgBouncer on port 6543): multiplexes connections from multiple instances onto fewer backend PostgreSQL connections

Format: `postgresql://postgres.<ref>:<password>@aws-0-us-central1.pooler.supabase.com:6543/postgres`

### 5. No new dependencies needed

`psycopg2.pool` is part of `psycopg2-binary` which is already in `pyproject.toml`.

## Sizing

| Setting | Value | Rationale |
|---------|-------|-----------|
| `db_pool_min_conn` | 2 | Keep 2 warm connections to avoid cold-start latency |
| `db_pool_max_conn` | 10 | Cloud Run max-instances=3 × 10 = 30 total, well within Supabase limits |

## Rollout

1. Update the `SUPABASE_DATABASE_URL` secret in GCP Secret Manager to use port 6543
2. Implement the code changes above (config → document_store → main)
3. Test locally with `docker compose up` to verify pool initialization/shutdown logs
4. Deploy via PR and confirm health check passes
