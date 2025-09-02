"""Database helper functions for easy access."""

# Standard library imports
from typing import Optional

# Third-party imports
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

# First-party imports
from src.database.mongodb import get_mongodb_client
from src.database.redis_client import get_redis_client


async def get_mongodb() -> AsyncIOMotorDatabase:
    """
    Get MongoDB database instance.

    Returns:
        AsyncIOMotorDatabase instance
    """
    client = await get_mongodb_client()
    return client.db


async def get_mongodb_connection() -> AsyncIOMotorClient:
    """
    Get MongoDB client instance.

    Returns:
        AsyncIOMotorClient instance
    """
    client = await get_mongodb_client()
    return client.client


async def get_redis() -> redis.Redis:
    """
    Get Redis client instance.

    Returns:
        Redis client instance
    """
    return await get_redis_client()
