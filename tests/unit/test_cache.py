"""Unit tests for cache system."""

# Standard library imports
import asyncio
import pickle
from unittest.mock import AsyncMock, Mock, patch

# Third-party imports
import numpy as np
import pytest

# First-party imports
from src.core.cache import CacheEntry, CacheStats, DualLayerCache
from tests.base import AsyncBaseTest


@pytest.mark.unit
class TestDualLayerCache(AsyncBaseTest):
    """Test cases for DualLayerCache."""

    async def setup_method_async(self):
        """Setup cache test."""
        await super().setup_method_async()

        self.mock_redis = AsyncMock()
        self.cache = DualLayerCache(
            redis_client=self.mock_redis,
            memory_max_size=1024 * 1024,  # 1MB
            default_ttl=3600,
        )

    @pytest.mark.asyncio
    async def test_initialization_without_redis(self):
        """Test cache initialization without Redis."""
        cache = DualLayerCache()

        assert cache.redis_client is None
        assert cache.memory_cache == {}
        assert cache.memory_max_size == 512 * 1024 * 1024  # Default 512MB
        assert cache.default_ttl == 3600

    @pytest.mark.asyncio
    async def test_set_memory_cache_only(self):
        """Test setting value in memory cache only."""
        cache = DualLayerCache(redis_client=None)

        await cache.set("test_key", "test_value", ttl=300)

        assert "test_key" in cache.memory_cache
        entry = cache.memory_cache["test_key"]
        assert entry.value == "test_value"
        assert entry.ttl == 300

    @pytest.mark.asyncio
    async def test_set_with_redis(self):
        """Test setting value with Redis backend."""
        self.mock_redis.set.return_value = True

        await self.cache.set("test_key", "test_value", ttl=300)

        # Should set in both memory and Redis
        assert "test_key" in self.cache.memory_cache
        self.mock_redis.set.assert_called_once()
        call_args = self.mock_redis.set.call_args
        assert call_args[0][0] == "test_key"  # key
        assert call_args[1]["ex"] == 300  # TTL

    @pytest.mark.asyncio
    async def test_get_from_memory_cache(self):
        """Test getting value from memory cache."""
        # Set value in memory cache
        await self.cache.set("test_key", "test_value")

        result = await self.cache.get("test_key")

        assert result == "test_value"
        # Should not hit Redis since found in memory
        self.mock_redis.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_from_redis_cache(self):
        """Test getting value from Redis when not in memory."""
        # Mock Redis return
        test_data = {"key": "value", "number": 42}
        self.mock_redis.get.return_value = pickle.dumps(test_data)

        result = await self.cache.get("test_key")

        assert result == test_data
        self.mock_redis.get.assert_called_once_with("test_key")
        # Should also be added to memory cache
        assert "test_key" in self.cache.memory_cache

    @pytest.mark.asyncio
    async def test_get_not_found(self):
        """Test getting non-existent value."""
        self.mock_redis.get.return_value = None

        result = await self.cache.get("nonexistent_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_from_both_caches(self):
        """Test deleting from both memory and Redis."""
        # Set value first
        await self.cache.set("test_key", "test_value")

        self.mock_redis.delete.return_value = 1

        result = await self.cache.delete("test_key")

        assert result is True
        assert "test_key" not in self.cache.memory_cache
        self.mock_redis.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_exists_in_memory(self):
        """Test checking existence in memory cache."""
        await self.cache.set("test_key", "test_value")

        result = await self.cache.exists("test_key")

        assert result is True
        # Should not check Redis since found in memory
        self.mock_redis.exists.assert_not_called()

    @pytest.mark.asyncio
    async def test_exists_in_redis(self):
        """Test checking existence in Redis when not in memory."""
        self.mock_redis.exists.return_value = True

        result = await self.cache.exists("test_key")

        assert result is True
        self.mock_redis.exists.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_clear_memory_cache(self):
        """Test clearing memory cache."""
        await self.cache.set("key1", "value1")
        await self.cache.set("key2", "value2")

        await self.cache.clear_memory()

        assert len(self.cache.memory_cache) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting cache statistics."""
        # Set some values and simulate hits/misses
        await self.cache.set("key1", "value1")
        await self.cache.get("key1")  # Hit
        await self.cache.get("nonexistent")  # Miss

        self.cache.stats.memory_hits = 1
        self.cache.stats.redis_hits = 0
        self.cache.stats.misses = 1

        stats = await self.cache.get_stats()

        assert isinstance(stats, CacheStats)
        assert stats.memory_hits == 1
        assert stats.misses == 1
        assert stats.total_requests == 2
        assert stats.hit_rate == 0.5  # 1 hit out of 2 requests

    @pytest.mark.asyncio
    async def test_memory_size_limit(self):
        """Test memory cache size limiting."""
        # Create cache with very small memory limit
        cache = DualLayerCache(redis_client=None, memory_max_size=1024)  # 1KB

        # Add large values that exceed limit
        large_value = "x" * 500  # 500 bytes
        await cache.set("key1", large_value)
        await cache.set("key2", large_value)
        await cache.set("key3", large_value)  # Should trigger eviction

        # Cache should not exceed size limit significantly
        current_size = sum(entry.size for entry in cache.memory_cache.values())
        assert current_size <= cache.memory_max_size * 1.1  # Allow 10% overhead

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration in memory cache."""
        # Standard library imports
        import time

        # Set value with very short TTL
        await self.cache.set("test_key", "test_value", ttl=1)

        # Should exist immediately
        assert await self.cache.exists("test_key")

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired and removed
        result = await self.cache.get("test_key")
        assert result is None
        assert "test_key" not in self.cache.memory_cache

    @pytest.mark.asyncio
    async def test_numpy_array_serialization(self):
        """Test caching NumPy arrays."""
        array_data = np.random.randn(100, 100)

        await self.cache.set("numpy_array", array_data)
        result = await self.cache.get("numpy_array")

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(array_data, result)

    @pytest.mark.asyncio
    async def test_complex_data_serialization(self):
        """Test caching complex data structures."""
        complex_data = {
            "analysis_id": "test_123",
            "detections": [{"bbox": [100, 100, 200, 200], "confidence": 0.9}],
            "embeddings": np.random.randn(512),
            "metadata": {"timestamp": "2025-01-01T00:00:00Z", "model_version": "v1.0"},
        }

        await self.cache.set("complex_data", complex_data)
        result = await self.cache.get("complex_data")

        assert result["analysis_id"] == "test_123"
        assert len(result["detections"]) == 1
        assert isinstance(result["embeddings"], np.ndarray)
        assert result["metadata"]["model_version"] == "v1.0"

    @pytest.mark.asyncio
    async def test_redis_connection_failure(self):
        """Test handling Redis connection failures gracefully."""
        self.mock_redis.set.side_effect = Exception("Redis connection failed")

        # Should still work with memory cache only
        await self.cache.set("test_key", "test_value")

        result = await self.cache.get("test_key")
        assert result == "test_value"

        # Error should be logged but not raised
        assert "test_key" in self.cache.memory_cache


@pytest.mark.unit
class TestCacheEntry:
    """Test cases for CacheEntry dataclass."""

    def test_create_cache_entry(self):
        """Test creating CacheEntry."""
        # Standard library imports
        import time

        entry = CacheEntry(value="test_value", ttl=300, size=1024)

        assert entry.value == "test_value"
        assert entry.ttl == 300
        assert entry.size == 1024
        assert abs(entry.created_at - time.time()) < 1  # Created recently

    def test_is_expired_not_expired(self):
        """Test checking if entry is not expired."""
        entry = CacheEntry(value="test", ttl=3600)

        assert not entry.is_expired()

    def test_is_expired_expired(self):
        """Test checking if entry is expired."""
        # Standard library imports
        import time

        # Create entry that's already expired
        entry = CacheEntry(value="test", ttl=1)
        entry.created_at = time.time() - 2  # 2 seconds ago

        assert entry.is_expired()

    def test_is_expired_no_ttl(self):
        """Test entry with no TTL never expires."""
        entry = CacheEntry(value="test", ttl=None)

        assert not entry.is_expired()


@pytest.mark.unit
class TestCacheStats:
    """Test cases for CacheStats dataclass."""

    def test_create_cache_stats(self):
        """Test creating CacheStats."""
        stats = CacheStats(
            memory_hits=10,
            redis_hits=5,
            misses=2,
            memory_size=1024000,
            memory_entries=100,
        )

        assert stats.memory_hits == 10
        assert stats.redis_hits == 5
        assert stats.misses == 2
        assert stats.total_requests == 17
        assert abs(stats.hit_rate - (15 / 17)) < 0.001

    def test_hit_rate_no_requests(self):
        """Test hit rate calculation with no requests."""
        stats = CacheStats()

        assert stats.hit_rate == 0.0

    def test_total_requests_calculation(self):
        """Test total requests calculation."""
        stats = CacheStats(memory_hits=5, redis_hits=3, misses=2)

        assert stats.total_requests == 10


# Performance tests for cache
@pytest.mark.unit
@pytest.mark.performance
class TestCachePerformance(AsyncBaseTest):
    """Performance tests for cache system."""

    @pytest.mark.asyncio
    async def test_memory_cache_performance(self):
        """Test memory cache performance."""
        cache = DualLayerCache(redis_client=None)

        # Test bulk operations
        # Standard library imports
        import time

        # Bulk set operation
        start_time = time.time()
        for i in range(1000):
            await cache.set(f"key_{i}", f"value_{i}")
        set_time = time.time() - start_time

        # Bulk get operation
        start_time = time.time()
        for i in range(1000):
            await cache.get(f"key_{i}")
        get_time = time.time() - start_time

        # Performance assertions
        assert set_time < 1.0  # Should complete in under 1 second
        assert get_time < 0.5  # Gets should be even faster
        assert len(cache.memory_cache) == 1000

    @pytest.mark.asyncio
    async def test_cache_memory_efficiency(self):
        """Test cache memory usage efficiency."""
        cache = DualLayerCache(redis_client=None, memory_max_size=1024 * 1024)

        # Add data and monitor memory usage
        for i in range(100):
            data = np.random.randn(100, 100)  # ~80KB each
            await cache.set(f"array_{i}", data)

        # Check memory usage
        total_size = sum(entry.size for entry in cache.memory_cache.values())
        assert total_size <= cache.memory_max_size

        # Cache should have evicted old entries
        assert len(cache.memory_cache) < 100

    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self):
        """Test concurrent cache access performance."""
        cache = DualLayerCache(redis_client=None)

        async def cache_worker(worker_id: int):
            """Worker function for concurrent testing."""
            for i in range(100):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                await cache.set(key, value)
                retrieved = await cache.get(key)
                assert retrieved == value

        # Standard library imports
        import time

        # Run 10 concurrent workers
        start_time = time.time()
        await asyncio.gather(*[cache_worker(i) for i in range(10)])
        elapsed = time.time() - start_time

        # Should complete reasonably quickly
        assert elapsed < 5.0  # 5 seconds max for 1000 total operations

        # All data should be accessible
        assert len(cache.memory_cache) == 1000

    @pytest.mark.asyncio
    async def test_large_data_caching_performance(self):
        """Test performance with large data objects."""
        cache = DualLayerCache(redis_client=None)

        # Create large numpy array (10MB)
        large_array = np.random.randn(1000, 1000)

        # Standard library imports
        import time

        # Test large object set/get performance
        start_time = time.time()
        await cache.set("large_array", large_array)
        set_time = time.time() - start_time

        start_time = time.time()
        retrieved = await cache.get("large_array")
        get_time = time.time() - start_time

        # Performance should still be reasonable
        assert set_time < 1.0  # Set should complete in under 1 second
        assert get_time < 0.1  # Get should be very fast (memory access)

        # Data should be correct
        np.testing.assert_array_equal(large_array, retrieved)
