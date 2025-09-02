"""Database fixtures and mocks for testing."""

# Standard library imports
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

# Third-party imports
import pymongo.errors
import pytest
from bson import ObjectId


class MockMongoCollection:
    """Mock MongoDB collection with realistic behavior."""

    def __init__(self, initial_data=None):
        self._data = initial_data or {}
        self._next_id = 1

    async def insert_one(self, document):
        """Mock insert_one operation."""
        doc_id = document.get("_id") or f"mock_id_{self._next_id}"
        self._next_id += 1

        doc_copy = document.copy()
        doc_copy["_id"] = doc_id
        self._data[doc_id] = doc_copy

        result = AsyncMock()
        result.inserted_id = doc_id
        return result

    async def insert_many(self, documents):
        """Mock insert_many operation."""
        inserted_ids = []
        for doc in documents:
            result = await self.insert_one(doc)
            inserted_ids.append(result.inserted_id)

        result = AsyncMock()
        result.inserted_ids = inserted_ids
        return result

    async def find_one(self, filter_dict=None, projection=None):
        """Mock find_one operation."""
        if filter_dict is None:
            return list(self._data.values())[0] if self._data else None

        if isinstance(filter_dict, str) or "_id" in filter_dict:
            doc_id = filter_dict if isinstance(filter_dict, str) else filter_dict["_id"]
            doc = self._data.get(doc_id)
            if doc and projection:
                return self._apply_projection(doc, projection)
            return doc

        # Simple field matching
        for doc in self._data.values():
            if self._matches_filter(doc, filter_dict):
                if projection:
                    return self._apply_projection(doc, projection)
                return doc
        return None

    def find(self, filter_dict=None, projection=None, sort=None, limit=None, skip=None):
        """Mock find operation."""
        cursor = MockMongoCursor(
            data=list(self._data.values()),
            filter_dict=filter_dict,
            projection=projection,
            sort=sort,
            limit=limit,
            skip=skip,
        )
        return cursor

    async def update_one(self, filter_dict, update_dict, upsert=False):
        """Mock update_one operation."""
        doc = await self.find_one(filter_dict)
        if doc:
            self._apply_update(doc, update_dict)
            result = AsyncMock()
            result.modified_count = 1
            result.matched_count = 1
            return result
        elif upsert:
            new_doc = self._create_upsert_doc(filter_dict, update_dict)
            return await self.insert_one(new_doc)
        else:
            result = AsyncMock()
            result.modified_count = 0
            result.matched_count = 0
            return result

    async def update_many(self, filter_dict, update_dict):
        """Mock update_many operation."""
        modified_count = 0
        for doc in self._data.values():
            if self._matches_filter(doc, filter_dict):
                self._apply_update(doc, update_dict)
                modified_count += 1

        result = AsyncMock()
        result.modified_count = modified_count
        result.matched_count = modified_count
        return result

    async def delete_one(self, filter_dict):
        """Mock delete_one operation."""
        for doc_id, doc in list(self._data.items()):
            if self._matches_filter(doc, filter_dict):
                del self._data[doc_id]
                result = AsyncMock()
                result.deleted_count = 1
                return result

        result = AsyncMock()
        result.deleted_count = 0
        return result

    async def delete_many(self, filter_dict):
        """Mock delete_many operation."""
        deleted_count = 0
        for doc_id, doc in list(self._data.items()):
            if self._matches_filter(doc, filter_dict):
                del self._data[doc_id]
                deleted_count += 1

        result = AsyncMock()
        result.deleted_count = deleted_count
        return result

    async def count_documents(self, filter_dict=None):
        """Mock count_documents operation."""
        if filter_dict is None:
            return len(self._data)

        count = 0
        for doc in self._data.values():
            if self._matches_filter(doc, filter_dict):
                count += 1
        return count

    def _matches_filter(self, doc, filter_dict):
        """Check if document matches filter."""
        if not filter_dict:
            return True

        for key, value in filter_dict.items():
            if key not in doc or doc[key] != value:
                return False
        return True

    def _apply_projection(self, doc, projection):
        """Apply projection to document."""
        if projection is None:
            return doc

        if isinstance(projection, dict):
            if all(v == 1 for v in projection.values()):
                # Include only specified fields
                return {k: doc[k] for k in projection.keys() if k in doc}
            elif all(v == 0 for v in projection.values()):
                # Exclude specified fields
                return {k: v for k, v in doc.items() if k not in projection}

        return doc

    def _apply_update(self, doc, update_dict):
        """Apply update operations to document."""
        for operator, operations in update_dict.items():
            if operator == "$set":
                doc.update(operations)
            elif operator == "$unset":
                for field in operations:
                    doc.pop(field, None)
            elif operator == "$inc":
                for field, increment in operations.items():
                    doc[field] = doc.get(field, 0) + increment

    def _create_upsert_doc(self, filter_dict, update_dict):
        """Create document for upsert operation."""
        doc = filter_dict.copy()
        self._apply_update(doc, update_dict)
        return doc


class MockMongoCursor:
    """Mock MongoDB cursor with realistic behavior."""

    def __init__(
        self, data, filter_dict=None, projection=None, sort=None, limit=None, skip=None
    ):
        self._data = data
        self._filter_dict = filter_dict
        self._projection = projection
        self._sort = sort
        self._limit = limit
        self._skip = skip

    async def to_list(self, length=None):
        """Convert cursor to list."""
        results = []

        # Apply filter
        for doc in self._data:
            if self._matches_filter(doc, self._filter_dict):
                if self._projection:
                    doc = self._apply_projection(doc, self._projection)
                results.append(doc)

        # Apply sort
        if self._sort:
            for field, direction in reversed(list(self._sort.items())):
                results.sort(key=lambda x: x.get(field, ""), reverse=(direction == -1))

        # Apply skip and limit
        if self._skip:
            results = results[self._skip :]

        if self._limit:
            results = results[: self._limit]

        if length and len(results) > length:
            results = results[:length]

        return results

    def sort(self, sort_spec):
        """Add sort specification."""
        if isinstance(sort_spec, str):
            self._sort = {sort_spec: 1}
        elif isinstance(sort_spec, list):
            self._sort = dict(sort_spec)
        else:
            self._sort = sort_spec
        return self

    def limit(self, limit):
        """Add limit."""
        self._limit = limit
        return self

    def skip(self, skip):
        """Add skip."""
        self._skip = skip
        return self

    def _matches_filter(self, doc, filter_dict):
        """Check if document matches filter."""
        if not filter_dict:
            return True

        for key, value in filter_dict.items():
            if key not in doc or doc[key] != value:
                return False
        return True

    def _apply_projection(self, doc, projection):
        """Apply projection to document."""
        if projection is None:
            return doc

        if isinstance(projection, dict):
            if all(v == 1 for v in projection.values()):
                return {k: doc[k] for k in projection.keys() if k in doc}
            elif all(v == 0 for v in projection.values()):
                return {k: v for k, v in doc.items() if k not in projection}

        return doc


class MockMongoDatabase:
    """Mock MongoDB database."""

    def __init__(self):
        self._collections = {}

    def __getitem__(self, collection_name):
        """Get or create collection."""
        if collection_name not in self._collections:
            self._collections[collection_name] = MockMongoCollection()
        return self._collections[collection_name]

    def get_collection(self, collection_name):
        """Get collection."""
        return self[collection_name]

    async def list_collection_names(self):
        """List collection names."""
        return list(self._collections.keys())


class MockRedisClient:
    """Mock Redis client with realistic behavior."""

    def __init__(self):
        self._data = {}
        self._expires = {}

    async def get(self, key):
        """Get value by key."""
        if self._is_expired(key):
            await self.delete(key)
            return None
        return self._data.get(key)

    async def set(self, key, value, ex=None, px=None, nx=False, xx=False):
        """Set key-value pair."""
        if nx and key in self._data:
            return False
        if xx and key not in self._data:
            return False

        self._data[key] = value

        if ex:  # Expiry in seconds
            # Standard library imports
            import time

            self._expires[key] = time.time() + ex
        elif px:  # Expiry in milliseconds
            # Standard library imports
            import time

            self._expires[key] = time.time() + (px / 1000)

        return True

    async def delete(self, key):
        """Delete key."""
        deleted = 0
        if key in self._data:
            del self._data[key]
            deleted += 1
        if key in self._expires:
            del self._expires[key]
        return deleted

    async def exists(self, key):
        """Check if key exists."""
        if self._is_expired(key):
            await self.delete(key)
            return False
        return key in self._data

    async def expire(self, key, seconds):
        """Set expiry for key."""
        if key not in self._data:
            return False

        # Standard library imports
        import time

        self._expires[key] = time.time() + seconds
        return True

    async def ttl(self, key):
        """Get time to live for key."""
        if key not in self._expires:
            return -1 if key in self._data else -2

        # Standard library imports
        import time

        remaining = self._expires[key] - time.time()
        return max(0, int(remaining))

    async def keys(self, pattern="*"):
        """Get keys matching pattern."""
        # Standard library imports
        import fnmatch

        # Clean expired keys first
        expired_keys = []
        for key in list(self._data.keys()):
            if self._is_expired(key):
                expired_keys.append(key)

        for key in expired_keys:
            await self.delete(key)

        # Return matching keys
        if pattern == "*":
            return list(self._data.keys())

        return [k for k in self._data.keys() if fnmatch.fnmatch(k, pattern)]

    async def flushdb(self):
        """Clear database."""
        self._data.clear()
        self._expires.clear()
        return True

    async def info(self, section=None):
        """Get Redis info."""
        return {
            "used_memory": len(str(self._data)),
            "connected_clients": 1,
            "total_commands_processed": 100,
            "keyspace_hits": 50,
            "keyspace_misses": 10,
        }

    def _is_expired(self, key):
        """Check if key is expired."""
        if key not in self._expires:
            return False

        # Standard library imports
        import time

        return time.time() > self._expires[key]


@pytest.fixture
def mock_mongodb():
    """Fixture for mock MongoDB client."""
    mock_client = AsyncMock()
    mock_db = MockMongoDatabase()

    mock_client.__getitem__.return_value = mock_db
    mock_client.get_database.return_value = mock_db

    return mock_client


@pytest.fixture
def mock_redis():
    """Fixture for mock Redis client."""
    return MockRedisClient()


@pytest.fixture
def sample_analysis_data():
    """Sample analysis data for testing."""
    return {
        "_id": "analysis_123",
        "timestamp": datetime.now(timezone.utc),
        "image_hash": "abc123def456",
        "detections": [
            {
                "class": "cup",
                "confidence": 0.92,
                "bbox": [200, 200, 300, 350],
                "is_mug_related": True,
            }
        ],
        "embeddings": [0.1, 0.2, 0.3] * 170 + [0.4, 0.5],  # 512-dim vector
        "mug_positions": [
            {"x": 250, "y": 275, "confidence": 0.88, "strategy": "hybrid"}
        ],
        "confidence_scores": {"detection": 0.92, "positioning": 0.88},
        "processing_time_ms": 245,
        "gpu_memory_mb": 512,
        "cached": False,
        "user_id": "user_456",
        "metadata": {
            "model_versions": {"yolo": "yolov8n", "clip": "ViT-B/32"},
            "image_size": [640, 640],
            "device": "cuda:0",
        },
    }


@pytest.fixture
def sample_rule_data():
    """Sample rule data for testing."""
    return {
        "_id": "rule_789",
        "name": "High Confidence Alert",
        "rule_type": "conditional",
        "conditions": [
            {"field": "confidence", "operator": "greater_than", "value": 0.8}
        ],
        "actions": [
            {
                "type": "send_alert",
                "parameters": {
                    "message": "High confidence mug detected",
                    "recipient": "admin@example.com",
                },
            }
        ],
        "active": True,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "user_id": "user_456",
        "execution_count": 15,
        "last_executed": datetime.now(timezone.utc),
    }


@pytest.fixture
def mock_mongodb_with_data(mock_mongodb, sample_analysis_data, sample_rule_data):
    """MongoDB mock pre-populated with test data."""
    # Populate collections with sample data
    db = mock_mongodb.__getitem__.return_value

    # Add analysis data
    # Standard library imports
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(db.analyses.insert_one(sample_analysis_data))
        loop.run_until_complete(db.rules.insert_one(sample_rule_data))

        # Add more test data
        for i in range(5):
            analysis = sample_analysis_data.copy()
            analysis["_id"] = f"analysis_{i}"
            analysis["confidence_scores"]["detection"] = 0.7 + (i * 0.05)
            loop.run_until_complete(db.analyses.insert_one(analysis))
    finally:
        loop.close()

    return mock_mongodb


@pytest.fixture
def mock_redis_with_data(mock_redis):
    """Redis mock pre-populated with test data."""
    # Standard library imports
    import asyncio

    async def populate_redis():
        # Cache some analysis results
        await mock_redis.set("analysis:hash_123", '{"cached": true, "result": "data"}')
        await mock_redis.set("analysis:hash_456", '{"cached": true, "result": "data"}')

        # Cache some model data
        await mock_redis.set("model:yolo:loaded", "true")
        await mock_redis.set("model:clip:loaded", "true")

        # Performance metrics
        await mock_redis.set("metrics:cache_hits", "150")
        await mock_redis.set("metrics:cache_misses", "25")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(populate_redis())
    finally:
        loop.close()

    return mock_redis


class DatabaseFixtureManager:
    """Manager for database fixtures and test data."""

    def __init__(self, mongodb_mock, redis_mock):
        self.mongodb = mongodb_mock
        self.redis = redis_mock

    async def reset_all_data(self):
        """Reset all test data."""
        # Clear Redis
        await self.redis.flushdb()

        # Clear MongoDB collections
        db = self.mongodb.__getitem__.return_value
        for collection_name in await db.list_collection_names():
            collection = db[collection_name]
            await collection.delete_many({})

    async def seed_test_data(self, data_type="basic"):
        """Seed database with test data."""
        if data_type == "basic":
            await self._seed_basic_data()
        elif data_type == "performance":
            await self._seed_performance_data()
        elif data_type == "integration":
            await self._seed_integration_data()

    async def _seed_basic_data(self):
        """Seed basic test data."""
        db = self.mongodb.__getitem__.return_value

        # Basic analyses
        analyses = [
            {
                "_id": f"basic_analysis_{i}",
                "confidence_scores": {"detection": 0.8 + (i * 0.02)},
                "timestamp": datetime.now(timezone.utc),
                "user_id": "test_user",
            }
            for i in range(10)
        ]

        for analysis in analyses:
            await db.analyses.insert_one(analysis)

    async def _seed_performance_data(self):
        """Seed performance test data."""
        db = self.mongodb.__getitem__.return_value

        # Large dataset for performance testing
        for batch in range(10):  # 10 batches of 100 = 1000 records
            analyses = [
                {
                    "_id": f"perf_analysis_{batch}_{i}",
                    "processing_time_ms": 200 + (i % 100),
                    "cached": i % 3 == 0,
                    "timestamp": datetime.now(timezone.utc),
                }
                for i in range(100)
            ]

            await db.analyses.insert_many(analyses)

    async def _seed_integration_data(self):
        """Seed integration test data."""
        await self._seed_basic_data()

        # Add rules data
        db = self.mongodb.__getitem__.return_value

        rules = [
            {
                "_id": f"integration_rule_{i}",
                "name": f"Test Rule {i}",
                "active": i % 2 == 0,
                "execution_count": i * 5,
            }
            for i in range(5)
        ]

        for rule in rules:
            await db.rules.insert_one(rule)


@pytest.fixture
def db_fixture_manager(mock_mongodb, mock_redis):
    """Fixture for database fixture manager."""
    return DatabaseFixtureManager(mock_mongodb, mock_redis)
