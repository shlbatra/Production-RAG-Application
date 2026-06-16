"""
Response Caching Layer
In-memory cache with TTL for LLM response deduplication.
"""

import hashlib
import time
from typing import Optional

"""
Limitations:
  1. Lazy expiration — expired entries are only removed when someone tries to get() them. There's no background cleanup, so stale entries can sit in memory. In production with Redis, TTL expiration is automatic.
  2. Not thread-safe — if multiple requests arrive simultaneously (FastAPI is async), concurrent reads/writes to self._cache could cause issues. In practice, Python's GIL makes simple dict operations atomic enough for this use case, but Redis
  would be the proper solution.
  3. No size limit — the cache can grow unbounded. In production, you'd want a max size with an eviction policy (like LRU — least recently used).
"""


class ResponseCache:
    """
    In memory response cache with TTL (time-to-live)
    In production, replace this with Redis for:
    - Persistance across restarts
    - Shared cache across multiple instances
    - Built-in TTL management
    """

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str) -> str:
        """
        Create cache key from normalized query
        """
        normalized = query.lower().strip()
        return hashlib.sha256(
            normalized.encode()
        ).hexdigest()  # What is Python ? and what is python ?, Hashing allows Fixed length and No special characters

    def get(self, query: str) -> Optional[str]:
        """
        Get cached response if it exists and hasnt expied.
        Returns None on cache miss
        """
        key = self._make_key(query)

        if key in self._cache:
            entry = self._cache[key]
            # Check TTL
            if time.time() - entry["timestamp"] < self.ttl:
                self._hits += 1
                return entry["response"]
            else:
                # Expired: remove it
                del self._cache[key]

        self._misses += 1
        return None

    def set(self, query: str, response: str):
        """
        Cache response. If the key already exists, this overwrites it — effectively refreshing both the response and the timestamp
        """
        key = self._make_key(query)
        self._cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "query": query,
        }

    @property
    def stats(self) -> dict:
        """Cache performance statistics
        @property makes this a computed attribute — you call cache.stats (no parentheses) instead of cache.stats()
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
            "cached_entries": len(self._cache),
        }
