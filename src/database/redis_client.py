"""
Redis async connection client for H200 Intelligent Mug Positioning System.

This module provides an async Redis client with clustering support,
connection pooling, and comprehensive error handling for high-performance caching.
"""

import asyncio
import os
import json
from typing import Optional, Dict, Any, List, Union, Set
from datetime import timedelta

import structlog
from redis.asyncio import Redis, RedisCluster
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import (
    RedisError,
    ConnectionError,
    TimeoutError,
    ResponseError,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize structured logger
logger = structlog.get_logger(__name__)


class RedisClient:
    """
    Async Redis client with clustering support and connection pooling.
    
    Supports both single-instance and clustered Redis deployments with
    automatic failover, retry logic, and comprehensive monitoring.
    
    Attributes:
        host: Redis host or cluster endpoint
        port: Redis port
        password: Redis password for authentication
        db: Database number (for non-cluster mode)
        cluster_mode: Whether to use Redis cluster
        max_connections: Maximum connections in the pool
        socket_timeout: Socket timeout in seconds
        socket_connect_timeout: Connection timeout in seconds
        retry_attempts: Number of retry attempts
        retry_delay: Delay between retries in seconds
        decode_responses: Whether to decode responses to strings
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        cluster_mode: bool = False,
        max_connections: int = 100,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        decode_responses: bool = True,
        ssl: bool = True,
    ):
        """
        Initialize Redis client with connection pooling.
        
        Args:
            host: Redis host (defaults to env variable)
            port: Redis port
            password: Redis password (defaults to env variable)
            db: Database number (ignored in cluster mode)
            cluster_mode: Enable Redis cluster mode
            max_connections: Maximum connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries
            decode_responses: Decode responses to strings
            ssl: Use SSL/TLS for connections
        """
        self.host = host or os.getenv("REDIS_HOST")
        if not self.host:
            raise ValueError("Redis host not provided or found in environment")
        
        self.port = int(os.getenv("REDIS_PORT", str(port)))
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.db = db
        self.cluster_mode = cluster_mode
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.decode_responses = decode_responses
        self.ssl = ssl
        
        self.client: Optional[Union[Redis, RedisCluster]] = None
        self._connected = False
        
        logger.info(
            "Redis client initialized",
            host=self.host,
            port=self.port,
            cluster_mode=cluster_mode,
            max_connections=max_connections
        )
    
    async def connect(self) -> None:
        """
        Establish connection to Redis with retry logic.
        
        Raises:
            ConnectionError: If connection cannot be established after retries
        """
        for attempt in range(self.retry_attempts):
            try:
                if self.cluster_mode:
                    # Redis cluster configuration
                    self.client = RedisCluster(
                        host=self.host,
                        port=self.port,
                        password=self.password,
                        decode_responses=self.decode_responses,
                        skip_full_coverage_check=True,
                        socket_timeout=self.socket_timeout,
                        socket_connect_timeout=self.socket_connect_timeout,
                        max_connections=self.max_connections,
                        ssl=self.ssl,
                    )
                else:
                    # Single Redis instance configuration
                    pool = ConnectionPool(
                        host=self.host,
                        port=self.port,
                        password=self.password,
                        db=self.db,
                        decode_responses=self.decode_responses,
                        max_connections=self.max_connections,
                        socket_timeout=self.socket_timeout,
                        socket_connect_timeout=self.socket_connect_timeout,
                        ssl=self.ssl,
                    )
                    self.client = Redis(connection_pool=pool)
                
                # Test the connection
                await self.client.ping()
                self._connected = True
                
                logger.info(
                    "Successfully connected to Redis",
                    cluster_mode=self.cluster_mode,
                    attempt=attempt + 1
                )
                return
                
            except (ConnectionError, TimeoutError) as e:
                logger.warning(
                    "Failed to connect to Redis",
                    error=str(e),
                    attempt=attempt + 1,
                    max_attempts=self.retry_attempts
                )
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("Failed to connect to Redis after all retries")
                    raise ConnectionError(f"Could not connect to Redis after {self.retry_attempts} attempts")
    
    async def disconnect(self) -> None:
        """Close the Redis connection."""
        if self.client:
            await self.client.close()
            self._connected = False
            logger.info("Redis connection closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Redis connection.
        
        Returns:
            Dict containing health status and connection details
        """
        health_status = {
            "connected": False,
            "cluster_mode": self.cluster_mode,
            "error": None,
            "info": {},
            "memory_usage": {},
            "ping_time_ms": None
        }
        
        try:
            if not self._connected:
                await self.connect()
            
            # Measure ping time
            start_time = asyncio.get_event_loop().time()
            await self.client.ping()
            ping_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            health_status["connected"] = True
            health_status["ping_time_ms"] = round(ping_time, 2)
            
            # Get server info
            if self.cluster_mode:
                # For cluster, get info from all nodes
                info = await self.client.info()
                health_status["info"] = {
                    "cluster_enabled": True,
                    "cluster_state": "ok",
                    "nodes": len(info) if isinstance(info, dict) else 1
                }
            else:
                info = await self.client.info()
                health_status["info"] = {
                    "redis_version": info.get("redis_version", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "maxmemory_human": info.get("maxmemory_human", "unknown"),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                }
            
            # Get memory usage stats
            memory_stats = await self.client.info("memory")
            health_status["memory_usage"] = {
                "used_memory": memory_stats.get("used_memory", 0),
                "used_memory_human": memory_stats.get("used_memory_human", "0"),
                "used_memory_peak_human": memory_stats.get("used_memory_peak_human", "0"),
            }
            
            logger.info("Redis health check passed", ping_ms=ping_time)
            
        except Exception as e:
            health_status["error"] = str(e)
            logger.error("Redis health check failed", error=str(e))
        
        return health_status
    
    async def execute_with_retry(
        self,
        operation,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a Redis operation with retry logic.
        
        Args:
            operation: The async operation to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            RedisError: If operation fails after all retries
        """
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                if not self._connected:
                    await self.connect()
                
                result = await operation(*args, **kwargs)
                return result
                
            except (ConnectionError, TimeoutError) as e:
                last_error = e
                self._connected = False
                logger.warning(
                    "Redis operation failed due to connection error",
                    operation=operation.__name__,
                    error=str(e),
                    attempt=attempt + 1
                )
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    
            except ResponseError as e:
                # Don't retry on response errors (e.g., wrong data type)
                logger.error(
                    "Redis operation failed with response error",
                    operation=operation.__name__,
                    error=str(e)
                )
                raise
        
        logger.error(
            "Redis operation failed after all retries",
            operation=operation.__name__,
            attempts=self.retry_attempts
        )
        raise last_error
    
    # Cache operations with automatic retry
    
    async def get(self, key: str) -> Optional[str]:
        """Get a value from cache."""
        return await self.execute_with_retry(self.client.get, key)
    
    async def set(
        self,
        key: str,
        value: Union[str, bytes, int, float],
        ex: Optional[Union[int, timedelta]] = None,
        px: Optional[Union[int, timedelta]] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """
        Set a value in cache with optional expiration.
        
        Args:
            key: Cache key
            value: Value to cache
            ex: Expire time in seconds
            px: Expire time in milliseconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists
            
        Returns:
            True if set successfully
        """
        return await self.execute_with_retry(
            self.client.set,
            key,
            value,
            ex=ex,
            px=px,
            nx=nx,
            xx=xx
        )
    
    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        return await self.execute_with_retry(self.client.delete, *keys)
    
    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        return await self.execute_with_retry(self.client.exists, *keys)
    
    async def expire(self, key: str, seconds: Union[int, timedelta]) -> bool:
        """Set expiration on a key."""
        return await self.execute_with_retry(self.client.expire, key, seconds)
    
    async def ttl(self, key: str) -> int:
        """Get time to live for a key in seconds."""
        return await self.execute_with_retry(self.client.ttl, key)
    
    # Hash operations
    
    async def hset(self, name: str, key: str, value: str) -> int:
        """Set hash field."""
        return await self.execute_with_retry(self.client.hset, name, key, value)
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field."""
        return await self.execute_with_retry(self.client.hget, name, key)
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields."""
        return await self.execute_with_retry(self.client.hgetall, name)
    
    async def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields."""
        return await self.execute_with_retry(self.client.hdel, name, *keys)
    
    # List operations
    
    async def lpush(self, key: str, *values: str) -> int:
        """Push values to the left of a list."""
        return await self.execute_with_retry(self.client.lpush, key, *values)
    
    async def rpush(self, key: str, *values: str) -> int:
        """Push values to the right of a list."""
        return await self.execute_with_retry(self.client.rpush, key, *values)
    
    async def lpop(self, key: str, count: Optional[int] = None) -> Optional[Union[str, List[str]]]:
        """Pop from the left of a list."""
        return await self.execute_with_retry(self.client.lpop, key, count)
    
    async def lrange(self, key: str, start: int, stop: int) -> List[str]:
        """Get a range of values from a list."""
        return await self.execute_with_retry(self.client.lrange, key, start, stop)
    
    # Set operations
    
    async def sadd(self, key: str, *values: str) -> int:
        """Add members to a set."""
        return await self.execute_with_retry(self.client.sadd, key, *values)
    
    async def srem(self, key: str, *values: str) -> int:
        """Remove members from a set."""
        return await self.execute_with_retry(self.client.srem, key, *values)
    
    async def smembers(self, key: str) -> Set[str]:
        """Get all members of a set."""
        return await self.execute_with_retry(self.client.smembers, key)
    
    async def sismember(self, key: str, value: str) -> bool:
        """Check if value is a member of a set."""
        return await self.execute_with_retry(self.client.sismember, key, value)
    
    # JSON operations (for complex objects)
    
    async def json_set(self, key: str, obj: Any, ex: Optional[int] = None) -> bool:
        """Store a JSON object."""
        json_str = json.dumps(obj)
        return await self.set(key, json_str, ex=ex)
    
    async def json_get(self, key: str) -> Optional[Any]:
        """Retrieve and parse a JSON object."""
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON", key=key, value=value)
                return None
        return None
    
    # Batch operations
    
    async def mget(self, *keys: str) -> List[Optional[str]]:
        """Get multiple values at once."""
        return await self.execute_with_retry(self.client.mget, *keys)
    
    async def mset(self, mapping: Dict[str, str]) -> bool:
        """Set multiple values at once."""
        return await self.execute_with_retry(self.client.mset, mapping)
    
    # Pattern operations
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern (use with caution in production)."""
        return await self.execute_with_retry(self.client.keys, pattern)
    
    async def scan_iter(self, match: Optional[str] = None, count: int = 100):
        """Iterate over keys matching a pattern."""
        async for key in self.client.scan_iter(match=match, count=count):
            yield key


# Global client instance
_redis_client: Optional[RedisClient] = None


async def get_redis_client() -> RedisClient:
    """
    Get or create the global Redis client instance.
    
    Returns:
        RedisClient instance
    """
    global _redis_client
    
    if _redis_client is None:
        _redis_client = RedisClient()
        await _redis_client.connect()
    
    return _redis_client