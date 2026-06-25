from unittest.mock import patch

import fakeredis
import pytest
import redis as redis_lib

from app.cache import ResponseCache


@pytest.fixture
def cache():
    """ResponseCache backed by fakeredis — no real Redis needed."""
    c = ResponseCache(redis_url="redis://fake", ttl_seconds=60)
    c._redis = fakeredis.FakeRedis(decode_responses=True)
    return c


class TestSetAndGet:
    def test_round_trip(self, cache):
        cache.set("What is Python?", "A programming language")
        assert cache.get("What is Python?") == "A programming language"

    def test_overwrite_existing_key(self, cache):
        cache.set("question", "old answer")
        cache.set("question", "new answer")
        assert cache.get("question") == "new answer"


class TestCacheMiss:
    def test_returns_none_for_unknown_query(self, cache):
        assert cache.get("never seen before") is None


class TestTTLExpiration:
    def test_expired_entries_return_none(self, cache):
        cache.set("temp query", "temp answer")
        assert cache.get("temp query") == "temp answer"

        # Advance all TTLs to force expiration
        for key in cache._redis.keys(cache.KEY_PREFIX + "[a-f0-9]*"):
            cache._redis.expire(key, 0)

        assert cache.get("temp query") is None


class TestKeyNormalization:
    def test_case_insensitive(self, cache):
        cache.set("What is Python?", "A language")
        assert cache.get("what is python?") == "A language"

    def test_whitespace_insensitive(self, cache):
        cache.set("  hello world  ", "greeting")
        assert cache.get("hello world") == "greeting"


class TestStatsTracking:
    def test_initial_stats(self, cache):
        stats = cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == "0.0%"
        assert stats["cached_entries"] == 0

    def test_hit_and_miss_counting(self, cache):
        cache.set("q", "a")
        cache.get("q")  # hit
        cache.get("q")  # hit
        cache.get("unknown")  # miss

        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == "66.7%"
        assert stats["cached_entries"] == 1


class TestHealthCheck:
    def test_returns_true_when_redis_is_up(self, cache):
        assert cache.health_check() is True


class TestGracefulDegradation:
    def test_get_returns_none_when_redis_is_down(self, cache):
        with patch.object(
            cache._redis,
            "get",
            side_effect=redis_lib.ConnectionError("connection refused"),
        ):
            assert cache.get("any query") is None

    def test_set_does_not_raise_when_redis_is_down(self, cache):
        with patch.object(
            cache._redis,
            "set",
            side_effect=redis_lib.ConnectionError("connection refused"),
        ):
            cache.set("any query", "any response")  # should not raise
