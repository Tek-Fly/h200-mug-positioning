"""
Dual-layer caching system for H200 Intelligent Mug Positioning System.

This module implements a two-tier caching strategy:
- L1: Redis for distributed cache with fast access
- L2: GPU memory cache for models and frequently accessed data
"""

# Standard library imports
import asyncio
import hashlib
import json
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Union

# Third-party imports
import numpy as np
import structlog
import torch

# First-party imports
from src.database.redis_client import get_redis_client

# Initialize structured logger
logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    key: str
    value: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class GPUMemoryCache:
    """
    L2 GPU memory cache for models and tensors.

    Implements LRU eviction with size-based limits and automatic
    garbage collection for optimal GPU memory usage.
    """

    def __init__(
        self,
        max_size_mb: int = 2048,
        eviction_ratio: float = 0.2,
        device: str = "cuda:0",
    ):
        """
        Initialize GPU memory cache.

        Args:
            max_size_mb: Maximum cache size in MB
            eviction_ratio: Fraction to evict when full (0.2 = 20%)
            device: GPU device to use
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.eviction_ratio = eviction_ratio
        self.device = device if torch.cuda.is_available() else "cpu"

        # LRU cache implementation
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_size = 0
        self._lock = asyncio.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        logger.info(
            "GPU memory cache initialized", max_size_mb=max_size_mb, device=self.device
        )

    async def get(self, key: str) -> Optional[Any]:
        """Get value from GPU cache."""
        async with self._lock:
            if key in self._cache:
                # Update access time and count
                entry = self._cache[key]
                entry.last_accessed = time.time()
                entry.access_count += 1

                # Move to end (most recently used)
                self._cache.move_to_end(key)

                self._hits += 1
                return entry.value

            self._misses += 1
            return None

    async def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in GPU cache."""
        # Calculate size
        size_bytes = self._calculate_size(value)

        # Check if value is too large
        if size_bytes > self.max_size_bytes:
            logger.warning(
                "Value too large for GPU cache",
                key=key,
                size_mb=size_bytes / 1024 / 1024,
                max_mb=self.max_size_bytes / 1024 / 1024,
            )
            return False

        async with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._current_size -= self._cache[key].size_bytes
                del self._cache[key]

            # Evict if necessary
            while self._current_size + size_bytes > self.max_size_bytes:
                await self._evict_lru()

            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=0,
                ttl=ttl,
            )

            self._cache[key] = entry
            self._current_size += size_bytes

            # Move model/tensor to GPU if applicable
            if isinstance(value, (torch.nn.Module, torch.Tensor)):
                entry.value = value.to(self.device)

            return True

    async def delete(self, key: str) -> bool:
        """Remove value from GPU cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._current_size -= entry.size_bytes

                # Free GPU memory if tensor/model
                if isinstance(entry.value, (torch.nn.Module, torch.Tensor)):
                    del entry.value
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear entire GPU cache."""
        async with self._lock:
            # Free GPU memory for all tensors/models
            for entry in self._cache.values():
                if isinstance(entry.value, (torch.nn.Module, torch.Tensor)):
                    del entry.value

            self._cache.clear()
            self._current_size = 0

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("GPU cache cleared")

    async def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self._cache:
            return

        # Calculate how much to evict
        target_size = int(self.max_size_bytes * (1 - self.eviction_ratio))
        evicted_count = 0
        evicted_size = 0

        # Evict oldest entries first
        while self._current_size > target_size and self._cache:
            key, entry = self._cache.popitem(last=False)
            self._current_size -= entry.size_bytes
            evicted_size += entry.size_bytes
            evicted_count += 1

            # Free GPU memory if tensor/model
            if isinstance(entry.value, (torch.nn.Module, torch.Tensor)):
                del entry.value

        self._evictions += evicted_count

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug(
            "GPU cache eviction completed",
            evicted_count=evicted_count,
            evicted_mb=evicted_size / 1024 / 1024,
        )

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        if isinstance(value, torch.Tensor):
            return value.element_size() * value.nelement()
        elif isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, torch.nn.Module):
            # Estimate model size
            total_size = 0
            for param in value.parameters():
                total_size += param.element_size() * param.nelement()
            return total_size
        else:
            # For other types, use pickle size as estimate
            try:
                return len(pickle.dumps(value))
            except:
                return 1024  # Default 1KB

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "size_mb": self._current_size / 1024 / 1024,
            "max_size_mb": self.max_size_bytes / 1024 / 1024,
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "device": self.device,
        }


class DualLayerCache:
    """
    Dual-layer caching system combining Redis L1 and GPU L2 cache.

    Provides intelligent cache warming, invalidation policies, and
    automatic promotion/demotion between cache layers.
    """

    def __init__(
        self,
        redis_prefix: str = "h200:cache:",
        gpu_cache_mb: int = 2048,
        l1_default_ttl: int = 3600,
        l2_default_ttl: int = 7200,
        warm_on_startup: bool = True,
        promotion_threshold: int = 3,
    ):
        """
        Initialize dual-layer cache.

        Args:
            redis_prefix: Prefix for Redis keys
            gpu_cache_mb: GPU cache size in MB
            l1_default_ttl: Default L1 TTL in seconds
            l2_default_ttl: Default L2 TTL in seconds
            warm_on_startup: Warm cache on initialization
            promotion_threshold: Access count for L2->L1 promotion
        """
        self.redis_prefix = redis_prefix
        self.l1_default_ttl = l1_default_ttl
        self.l2_default_ttl = l2_default_ttl
        self.warm_on_startup = warm_on_startup
        self.promotion_threshold = promotion_threshold

        # Initialize caches
        self.l1_cache: Optional[Any] = None  # Redis client
        self.l2_cache = GPUMemoryCache(max_size_mb=gpu_cache_mb)

        # Track access patterns
        self._access_counts: Dict[str, int] = {}
        self._last_access: Dict[str, float] = {}

        logger.info(
            "Dual-layer cache initialized",
            redis_prefix=redis_prefix,
            gpu_cache_mb=gpu_cache_mb,
        )

    async def initialize(self) -> None:
        """Initialize cache connections and warm if configured."""
        # Get Redis client
        self.l1_cache = await get_redis_client()

        # Warm cache if configured
        if self.warm_on_startup:
            await self.warm_cache()

        logger.info("Dual-layer cache initialization complete")

    async def get(
        self, key: str, deserializer: Optional[Callable] = None
    ) -> Optional[Any]:
        """
        Get value from cache, checking L1 then L2.

        Args:
            key: Cache key
            deserializer: Optional function to deserialize value

        Returns:
            Cached value or None
        """
        full_key = f"{self.redis_prefix}{key}"

        # Update access tracking
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
        self._last_access[key] = time.time()

        # Check L1 (Redis)
        try:
            l1_value = await self.l1_cache.get(full_key)
            if l1_value:
                # Deserialize if needed
                if deserializer:
                    return deserializer(l1_value)
                else:
                    return (
                        json.loads(l1_value) if isinstance(l1_value, str) else l1_value
                    )
        except Exception as e:
            logger.warning("L1 cache get error", key=key, error=str(e))

        # Check L2 (GPU)
        l2_value = await self.l2_cache.get(key)
        if l2_value is not None:
            # Promote to L1 if accessed frequently
            if self._access_counts[key] >= self.promotion_threshold:
                await self._promote_to_l1(key, l2_value)

            return l2_value

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serializer: Optional[Callable] = None,
        cache_level: str = "both",
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds
            serializer: Optional function to serialize value
            cache_level: Where to cache ("l1", "l2", or "both")

        Returns:
            True if successfully cached
        """
        full_key = f"{self.redis_prefix}{key}"
        success = True

        # Set in L1 if requested
        if cache_level in ("l1", "both"):
            try:
                # Serialize value
                if serializer:
                    serialized = serializer(value)
                else:
                    serialized = (
                        json.dumps(value)
                        if not isinstance(value, (str, bytes))
                        else value
                    )

                await self.l1_cache.set(
                    full_key, serialized, ex=ttl or self.l1_default_ttl
                )
            except Exception as e:
                logger.warning("L1 cache set error", key=key, error=str(e))
                success = False

        # Set in L2 if requested and suitable
        if cache_level in ("l2", "both"):
            if self._is_gpu_suitable(value):
                l2_success = await self.l2_cache.put(
                    key, value, ttl=ttl or self.l2_default_ttl
                )
                success = success and l2_success

        return success

    async def delete(self, key: str) -> bool:
        """Delete value from both cache layers."""
        full_key = f"{self.redis_prefix}{key}"

        l1_deleted = False
        l2_deleted = False

        # Delete from L1
        try:
            l1_deleted = await self.l1_cache.delete(full_key) > 0
        except Exception as e:
            logger.warning("L1 cache delete error", key=key, error=str(e))

        # Delete from L2
        l2_deleted = await self.l2_cache.delete(key)

        # Clean up tracking
        self._access_counts.pop(key, None)
        self._last_access.pop(key, None)

        return l1_deleted or l2_deleted

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        Args:
            pattern: Redis pattern (e.g., "analysis:*")

        Returns:
            Number of keys invalidated
        """
        full_pattern = f"{self.redis_prefix}{pattern}"
        count = 0

        # Invalidate L1 keys
        try:
            async for key in self.l1_cache.scan_iter(match=full_pattern):
                await self.l1_cache.delete(key)
                count += 1

                # Also remove from L2 if present
                short_key = key.replace(self.redis_prefix, "")
                await self.l2_cache.delete(short_key)
        except Exception as e:
            logger.error("Pattern invalidation error", pattern=pattern, error=str(e))

        logger.info("Cache invalidation completed", pattern=pattern, count=count)
        return count

    async def warm_cache(self, keys: Optional[List[str]] = None) -> int:
        """
        Warm cache with frequently accessed data.

        Args:
            keys: Specific keys to warm (None for automatic)

        Returns:
            Number of entries warmed
        """
        warmed = 0

        if keys:
            # Warm specific keys
            for key in keys:
                # This would typically load from database
                # For now, just mark as warmed
                warmed += 1
        else:
            # Automatic warming based on access patterns
            # Sort by access count and recency
            sorted_keys = sorted(
                self._access_counts.items(),
                key=lambda x: (x[1], self._last_access.get(x[0], 0)),
                reverse=True,
            )

            # Warm top entries
            for key, count in sorted_keys[:50]:
                if count >= self.promotion_threshold:
                    # Would load from database and cache
                    warmed += 1

        logger.info("Cache warming completed", warmed_count=warmed)
        return warmed

    async def _promote_to_l1(self, key: str, value: Any) -> None:
        """Promote frequently accessed value to L1 cache."""
        try:
            await self.set(key, value, cache_level="l1")
            logger.debug("Value promoted to L1", key=key)
        except Exception as e:
            logger.warning("L1 promotion failed", key=key, error=str(e))

    def _is_gpu_suitable(self, value: Any) -> bool:
        """Check if value is suitable for GPU cache."""
        return isinstance(value, (torch.Tensor, torch.nn.Module, np.ndarray))

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        # Get L1 stats
        l1_stats = {}
        try:
            info = await self.l1_cache.info()
            l1_stats = {
                "connected": True,
                "used_memory_mb": float(info.get("used_memory", 0)) / 1024 / 1024,
                "total_keys": (
                    info.get("db0", {}).get("keys", 0)
                    if isinstance(info.get("db0"), dict)
                    else 0
                ),
            }
        except:
            l1_stats = {"connected": False}

        # Get L2 stats
        l2_stats = self.l2_cache.get_stats()

        # Overall stats
        total_accesses = sum(self._access_counts.values())
        hot_keys = sorted(
            self._access_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "l1": l1_stats,
            "l2": l2_stats,
            "total_accesses": total_accesses,
            "unique_keys": len(self._access_counts),
            "hot_keys": hot_keys,
        }

    async def cleanup(self) -> None:
        """Cleanup cache resources."""
        await self.l2_cache.clear()
        logger.info("Dual-layer cache cleanup complete")
