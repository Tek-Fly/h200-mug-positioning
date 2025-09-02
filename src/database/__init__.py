"""
Database package for H200 Intelligent Mug Positioning System.

This package provides async database connections and storage clients:
- MongoDB Atlas for document storage and vector search
- Redis for high-performance caching  
- Cloudflare R2 for object storage (S3-compatible)
"""

from .mongodb import MongoDBClient, get_mongodb_client
from .redis_client import RedisClient, get_redis_client
from .r2_storage import R2StorageClient, get_r2_client

__all__ = [
    "MongoDBClient",
    "get_mongodb_client",
    "RedisClient", 
    "get_redis_client",
    "R2StorageClient",
    "get_r2_client",
]