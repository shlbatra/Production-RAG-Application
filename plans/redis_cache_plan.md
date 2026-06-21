# Plan: Replace In-Memory Cache with Redis

## Context

The current `ResponseCache` in `app/cache.py` uses a Python dict for LLM response caching. The docstring and comments explicitly call out that this should be replaced with Redis for production. The limitations are real: no persistence across restarts, no shared state across Cloud Run instances (each instance has its own cache), unbounded memory growth, and lazy-only expiration. Redis solves all of these.

## Approach

Replace the `ResponseCache` class internals with Redis while keeping the exact same public interface (`get`, `set`, `stats` property). This means `app/main.py` requires minimal changes — it already calls `cache.get()`, `cache.set()`, and `cache.stats`.

### Files to Modify

| File | Change |
|---|---|
| `app/cache.py` | Rewrite `ResponseCache` to use Redis |
| `app/config.py` | Add `redis_url` setting |
| `app/main.py` | Pass `redis_url` to `ResponseCache`, add Redis health check |
| `pyproject.toml` | Add `redis` + `fakeredis` dependencies |
| `.env.example` | Add `REDIS_URL` example |
| `docker-compose.yml` | Add Redis service |
| `tests/test_cache.py` | New test file using `fakeredis` |
| `.github/workflows/deploy-cloud-run.yml` | Add `REDIS_URL` env var |

---

## Detailed Changes

### 1. `app/config.py` — Add Redis Config

Add one field to `Settings`:

```python
redis_url: str = ""  # e.g. "redis://localhost:6379/0"
```

Add a computed property:

```python
@property
def redis_enabled(self) -> bool:
    return bool(self.redis_url)
```

---

### 2. `app/cache.py` — Redis-backed ResponseCache

Rewrite `ResponseCache` to use `redis.Redis`. Key design decisions:

- **Same public API**: `__init__(ttl_seconds)`, `get(query) -> str | None`, `set(query, response)`, `stats` property
- **Key format**: `rag:cache:{sha256_hash}` — namespaced to avoid collisions if Redis is shared
- **TTL**: Redis native `EX` parameter on `SET` — automatic expiration, no lazy cleanup needed
- **Hit/miss counters**: Stored in Redis too (`rag:cache:hits`, `rag:cache:misses`) via `INCR` — shared across instances
- **Graceful degradation**: If Redis is unreachable, log the error and return a cache miss (don't crash the request)
- **Connection**: Use `redis.Redis.from_url()` with `decode_responses=True` for string handling
- **Health check**: Add a `health_check() -> bool` method that calls `redis.ping()`

```python
import hashlib
import logging
from typing import Optional

import redis

logger = logging.getLogger(__name__)


class ResponseCache:
    KEY_PREFIX = "rag:cache:"

    def __init__(self, redis_url: str, ttl_seconds: int = 300):
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl_seconds

    def _make_key(self, query: str) -> str:
        normalized = query.lower().strip()
        return self.KEY_PREFIX + hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        try:
            value = self._redis.get(self._make_key(query))
            if value is not None:
                self._redis.incr(self.KEY_PREFIX + "hits")
                return value
            self._redis.incr(self.KEY_PREFIX + "misses")
            return None
        except redis.RedisError:
            logger.exception("Redis GET failed, treating as cache miss")
            return None

    def set(self, query: str, response: str) -> None:
        try:
            self._redis.set(self._make_key(query), response, ex=self.ttl)
        except redis.RedisError:
            logger.exception("Redis SET failed, skipping cache write")

    @property
    def stats(self) -> dict:
        try:
            hits = int(self._redis.get(self.KEY_PREFIX + "hits") or 0)
            misses = int(self._redis.get(self.KEY_PREFIX + "misses") or 0)
            total = hits + misses
            count = 0
            for _ in self._redis.scan_iter(
                match=self.KEY_PREFIX + "[a-f0-9]*", count=100
            ):
                count += 1
            return {
                "hits": hits,
                "misses": misses,
                "hit_rate": f"{(hits / total * 100) if total else 0:.1f}%",
                "cached_entries": count,
            }
        except redis.RedisError:
            logger.exception("Redis STATS failed")
            return {"hits": 0, "misses": 0, "hit_rate": "0.0%", "cached_entries": 0}

    def health_check(self) -> bool:
        try:
            return self._redis.ping()
        except redis.RedisError:
            return False
```

---

### 3. `app/main.py` — Two Small Changes

**Lifespan** — pass `redis_url` to `ResponseCache`:

```python
# Before:
cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)

# After:
cache = ResponseCache(redis_url=settings.redis_url, ttl_seconds=settings.cache_ttl_seconds)
```

**Health check** — add Redis to the checks dict:

```python
# Before:
"cache": cache is not None,

# After:
"cache": cache.health_check(),
```

---

### 4. `docker-compose.yml` — Add Redis Service

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - '6379:6379'
    healthcheck:
      test: ['CMD', 'redis-cli', 'ping']
      interval: 10s
      timeout: 5s
      retries: 3

  agent-api:
    build: .
    ports:
      - '8000:8000'
    env_file:
      - .env
    environment:
      - APP_ENV=production
      - LOG_LEVEL=INFO
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ['CMD', 'curl', '-f', 'http://localhost:8000/health']
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

---

### 5. `.env.example` — Add Redis URL

```
# Redis (required for response caching)
REDIS_URL=redis://localhost:6379/0
```

---

### 6. `pyproject.toml` — Add Dependencies

Production:
```toml
"redis>=5.0.0",
```

Dev (for testing without real Redis):
```toml
"fakeredis>=2.0.0",
```

---

### 7. `tests/test_cache.py` — Unit Tests with fakeredis

Use `fakeredis` to test without a real Redis instance:

| Test | What it verifies |
|---|---|
| `test_set_and_get` | Round-trip: set a response, get it back |
| `test_cache_miss` | Returns None for unknown query |
| `test_ttl_expiration` | Expired entries return None (use fakeredis time travel) |
| `test_key_normalization` | "What is Python?" and "what is python?" hit same key |
| `test_stats_tracking` | hits/misses/hit_rate computed correctly |
| `test_health_check` | Returns True when Redis is up |
| `test_graceful_degradation_get` | get() returns None when Redis is down |
| `test_graceful_degradation_set` | set() silently fails when Redis is down |

---

### 8. `.github/workflows/deploy-cloud-run.yml` — Add Redis URL

For GCP Memorystore (Redis), the URL is typically a private IP without auth, so it goes in env vars:

```
--set-env-vars="...,REDIS_URL=redis://10.0.0.X:6379/0"
```

If the Redis URL contains credentials, move it to `--set-secrets` instead.

---

## What Does NOT Change

- **`app/main.py` cache usage** — `cache.get()`, `cache.set()`, `cache.stats` calls stay identical
- **`/cache/stats` endpoint** — returns the same shape dict
- **`/chat` endpoint logic** — completely untouched
- **All other files** — the cache is self-contained

## Why Redis Over Alternatives

| Option | Verdict |
|---|---|
| **Redis** | Native TTL, shared across instances, persistence, battle-tested. GCP Memorystore provides managed Redis. |
| **Memcached** | Simpler but no persistence, no data structures beyond key-value. |
| **Cloud Run in-memory** | Current approach. Doesn't share across instances, lost on restart. |

---

## Verification

1. `docker compose up` — confirm Redis + API start together
2. `curl localhost:8000/health` — check Redis appears healthy in checks
3. Send two identical `/chat` requests — second should return `"cached": true`
4. `curl localhost:8000/cache/stats` — confirm hits/misses increment
5. `redis-cli KEYS "rag:cache:*"` — verify keys exist with correct prefix
6. `redis-cli TTL "rag:cache:<some-key>"` — verify TTL is set (~300s)
7. `uv run pytest tests/test_cache.py -v` — all unit tests pass
8. `uv run pytest tests/ -v` — no regressions in existing tests
9. `uv run ruff check . && uv run mypy app/` — lint and type check pass
