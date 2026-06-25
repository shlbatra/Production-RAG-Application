"""
Response Caching Layer
Redis-backed cache with TTL for LLM response deduplication.
"""

import hashlib
import logging
from typing import Optional

import redis

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    Redis-backed response cache with TTL (time-to-live).
    Uses Upstash Redis in Cloud Run, local Redis via Docker for development.
    Gracefully degrades to cache misses if Redis is unreachable.
    """

    KEY_PREFIX = "rag:cache:"

    def __init__(self, redis_url: str, ttl_seconds: int = 300):
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl_seconds

    def _make_key(self, query: str) -> str:
        """Create cache key from normalized query."""
        normalized = query.lower().strip()
        return self.KEY_PREFIX + hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """
        Get cached response if it exists and hasn't expired.
        Returns None on cache miss or Redis error.
        """
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
        """Cache a response with TTL. Silently fails if Redis is unreachable."""
        try:
            self._redis.set(self._make_key(query), response, ex=self.ttl)
        except redis.RedisError:
            logger.exception("Redis SET failed, skipping cache write")

    @property
    def stats(self) -> dict:
        """Cache performance statistics."""
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
        """Returns True if Redis is reachable."""
        try:
            return self._redis.ping()
        except redis.RedisError:
            return False
