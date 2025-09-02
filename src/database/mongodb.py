"""
MongoDB Atlas async connection client for H200 Intelligent Mug Positioning System.

This module provides an async MongoDB client with connection pooling,
retry logic, and comprehensive error handling.
"""

import asyncio
import os
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import structlog
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import (
    ConnectionFailure,
    ServerSelectionTimeoutError,
    OperationFailure,
    ConfigurationError
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize structured logger
logger = structlog.get_logger(__name__)


class MongoDBClient:
    """
    Async MongoDB Atlas client with connection pooling and retry logic.
    
    Attributes:
        uri: MongoDB connection URI
        database_name: Name of the database to use
        client: Motor async client instance
        db: Database instance
        max_pool_size: Maximum connection pool size
        min_pool_size: Minimum connection pool size
        max_idle_time_ms: Maximum idle time for connections
        retry_attempts: Number of retry attempts for operations
        retry_delay: Delay between retry attempts in seconds
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        database_name: str = "h200_production",
        max_pool_size: int = 100,
        min_pool_size: int = 10,
        max_idle_time_ms: int = 30000,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize MongoDB client with connection pooling.
        
        Args:
            uri: MongoDB connection URI (defaults to env variable)
            database_name: Name of the database to connect to
            max_pool_size: Maximum number of connections in the pool
            min_pool_size: Minimum number of connections in the pool
            max_idle_time_ms: Maximum idle time for connections
            retry_attempts: Number of retry attempts for operations
            retry_delay: Delay between retry attempts
        """
        self.uri = uri or os.getenv("MONGODB_ATLAS_URI")
        if not self.uri:
            raise ConfigurationError("MongoDB URI not provided or found in environment")
        
        self.database_name = database_name
        self.max_pool_size = max_pool_size
        self.min_pool_size = min_pool_size
        self.max_idle_time_ms = max_idle_time_ms
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        
        logger.info(
            "MongoDB client initialized",
            database=database_name,
            max_pool_size=max_pool_size,
            min_pool_size=min_pool_size
        )
    
    async def connect(self) -> None:
        """
        Establish connection to MongoDB Atlas with retry logic.
        
        Raises:
            ConnectionFailure: If connection cannot be established after retries
        """
        for attempt in range(self.retry_attempts):
            try:
                self.client = AsyncIOMotorClient(
                    self.uri,
                    maxPoolSize=self.max_pool_size,
                    minPoolSize=self.min_pool_size,
                    maxIdleTimeMS=self.max_idle_time_ms,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=10000,
                    socketTimeoutMS=30000,
                )
                
                # Test the connection
                await self.client.admin.command('ping')
                
                self.db = self.client[self.database_name]
                
                logger.info(
                    "Successfully connected to MongoDB Atlas",
                    database=self.database_name,
                    attempt=attempt + 1
                )
                return
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(
                    "Failed to connect to MongoDB",
                    error=str(e),
                    attempt=attempt + 1,
                    max_attempts=self.retry_attempts
                )
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("Failed to connect to MongoDB after all retries")
                    raise ConnectionFailure(f"Could not connect to MongoDB after {self.retry_attempts} attempts")
    
    async def disconnect(self) -> None:
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on MongoDB connection.
        
        Returns:
            Dict containing health status and connection details
        """
        health_status = {
            "connected": False,
            "database": self.database_name,
            "error": None,
            "collections": [],
            "stats": {}
        }
        
        try:
            if not self.client:
                await self.connect()
            
            # Ping the server
            await self.client.admin.command('ping')
            health_status["connected"] = True
            
            # Get database stats
            stats = await self.db.command("dbstats")
            health_status["stats"] = {
                "collections": stats.get("collections", 0),
                "objects": stats.get("objects", 0),
                "dataSize": stats.get("dataSize", 0),
                "indexSize": stats.get("indexSize", 0),
            }
            
            # List collections
            collections = await self.db.list_collection_names()
            health_status["collections"] = collections
            
            logger.info("MongoDB health check passed", stats=health_status["stats"])
            
        except Exception as e:
            health_status["error"] = str(e)
            logger.error("MongoDB health check failed", error=str(e))
        
        return health_status
    
    async def execute_with_retry(
        self,
        operation,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a database operation with retry logic.
        
        Args:
            operation: The async operation to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            OperationFailure: If operation fails after all retries
        """
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                result = await operation(*args, **kwargs)
                return result
                
            except OperationFailure as e:
                last_error = e
                logger.warning(
                    "Database operation failed",
                    operation=operation.__name__,
                    error=str(e),
                    attempt=attempt + 1
                )
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        logger.error(
            "Database operation failed after all retries",
            operation=operation.__name__,
            attempts=self.retry_attempts
        )
        raise last_error
    
    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for MongoDB transactions.
        
        Usage:
            async with client.transaction() as session:
                await collection.insert_one(doc, session=session)
        """
        if not self.client:
            await self.connect()
        
        async with await self.client.start_session() as session:
            async with session.start_transaction():
                try:
                    yield session
                    await session.commit_transaction()
                except Exception as e:
                    await session.abort_transaction()
                    logger.error("Transaction aborted", error=str(e))
                    raise
    
    async def create_indexes(self, collection_name: str, indexes: List[Dict[str, Any]]) -> None:
        """
        Create indexes for a collection.
        
        Args:
            collection_name: Name of the collection
            indexes: List of index specifications
        """
        collection = self.db[collection_name]
        
        for index_spec in indexes:
            try:
                await collection.create_index(
                    index_spec["keys"],
                    **index_spec.get("options", {})
                )
                logger.info(
                    "Index created",
                    collection=collection_name,
                    index=index_spec["keys"]
                )
            except Exception as e:
                logger.error(
                    "Failed to create index",
                    collection=collection_name,
                    index=index_spec["keys"],
                    error=str(e)
                )
    
    def get_collection(self, name: str):
        """
        Get a collection from the database.
        
        Args:
            name: Collection name
            
        Returns:
            AsyncIOMotorCollection instance
        """
        if not self.db:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.db[name]


# Global client instance
_mongodb_client: Optional[MongoDBClient] = None


async def get_mongodb_client() -> MongoDBClient:
    """
    Get or create the global MongoDB client instance.
    
    Returns:
        MongoDBClient instance
    """
    global _mongodb_client
    
    if _mongodb_client is None:
        _mongodb_client = MongoDBClient()
        await _mongodb_client.connect()
    
    return _mongodb_client