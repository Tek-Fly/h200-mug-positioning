"""Pytest configuration and fixtures for H200 test suite."""

# Standard library imports
import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Dict
from unittest.mock import AsyncMock, Mock, patch

# Third-party imports
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

# Set test environment variables before importing application code
os.environ["TESTING"] = "true"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["MONGODB_ATLAS_URI"] = "mongodb://localhost:27017/test_h200"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
os.environ["REDIS_PASSWORD"] = "test_password"
os.environ["R2_ENDPOINT"] = "https://test.r2.cloudflarestorage.com"
os.environ["R2_ACCESS_KEY"] = "test_access_key"
os.environ["R2_SECRET_KEY"] = "test_secret_key"
os.environ["R2_BUCKET_NAME"] = "test_bucket"
os.environ["RUNPOD_API_KEY"] = "test_runpod_key"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test_project"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    # Create a 640x640 RGB test image
    image = Image.new("RGB", (640, 640), color="white")
    return image


@pytest.fixture
def sample_mug_image():
    """Create a sample image with a mug for testing."""
    # Third-party imports
    from PIL import ImageDraw

    image = Image.new("RGB", (640, 640), color="white")
    draw = ImageDraw.Draw(image)

    # Draw a "mug" (brown rectangle with handle)
    draw.rectangle([200, 200, 300, 350], fill="brown", outline="black")
    draw.ellipse([300, 250, 330, 280], outline="black")  # Handle

    # Draw a "table" surface
    draw.rectangle([100, 400, 540, 500], fill="tan", outline="black")

    return image


@pytest.fixture
def sample_batch_images(sample_image):
    """Create a batch of sample images for testing."""
    return [sample_image] * 3


@pytest.fixture
def mock_gpu_available():
    """Mock CUDA/GPU availability."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.get_device_name", return_value="Test GPU"),
    ):
        yield


@pytest.fixture
def mock_gpu_unavailable():
    """Mock CUDA/GPU unavailability."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda.device_count", return_value=0),
    ):
        yield


@pytest.fixture
async def mock_mongodb():
    """Mock MongoDB client."""
    mock_client = AsyncMock()
    mock_db = AsyncMock()
    mock_collection = AsyncMock()

    mock_client.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection

    # Mock common operations
    mock_collection.insert_one.return_value = AsyncMock(inserted_id="test_id")
    mock_collection.find_one.return_value = {"_id": "test_id", "test": "data"}
    mock_collection.find.return_value = AsyncMock()
    mock_collection.update_one.return_value = AsyncMock(modified_count=1)
    mock_collection.delete_one.return_value = AsyncMock(deleted_count=1)

    with patch("src.database.mongodb.get_mongodb_client", return_value=mock_client):
        yield mock_client


@pytest.fixture
async def mock_redis():
    """Mock Redis client."""
    mock_redis = AsyncMock()

    # Mock common Redis operations
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.exists.return_value = False
    mock_redis.expire.return_value = True
    mock_redis.keys.return_value = []
    mock_redis.info.return_value = {"used_memory": 1024}

    with patch("src.database.redis_client.get_redis_client", return_value=mock_redis):
        yield mock_redis


@pytest.fixture
async def mock_r2_storage():
    """Mock R2 storage client."""
    mock_client = AsyncMock()

    # Mock S3-like operations
    mock_client.put_object.return_value = {"ETag": "test_etag"}
    mock_client.get_object.return_value = {"Body": AsyncMock()}
    mock_client.delete_object.return_value = {"DeleteMarker": True}
    mock_client.list_objects_v2.return_value = {"Contents": []}

    with patch("src.database.r2_storage.get_r2_client", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_yolo_model():
    """Mock YOLOv8 model."""
    mock_model = Mock()

    # Mock detection results
    mock_result = Mock()
    mock_result.boxes = Mock()
    mock_result.boxes.data = torch.tensor(
        [
            [100, 100, 200, 200, 0.9, 47],  # Cup class (COCO)
            [300, 150, 450, 300, 0.8, 0],  # Person class
        ]
    )
    mock_result.boxes.conf = torch.tensor([0.9, 0.8])
    mock_result.boxes.cls = torch.tensor([47, 0])  # Cup, Person
    mock_result.boxes.xyxy = torch.tensor([[100, 100, 200, 200], [300, 150, 450, 300]])

    mock_model.return_value = [mock_result]

    return mock_model


@pytest.fixture
def mock_clip_model():
    """Mock CLIP model."""
    mock_model = Mock()

    # Mock embeddings
    mock_embeddings = torch.randn(1, 512)  # Standard CLIP embedding size
    mock_model.encode_image.return_value = mock_embeddings

    return mock_model


@pytest.fixture
async def mock_model_manager(mock_yolo_model, mock_clip_model):
    """Mock ModelManager with pre-loaded models."""
    mock_manager = AsyncMock()

    mock_manager.get_model.side_effect = lambda name: {
        "yolo": mock_yolo_model,
        "clip": mock_clip_model,
    }.get(name)

    mock_manager.is_loaded.return_value = True
    mock_manager.get_registry_info.return_value = {
        "yolo": {"description": "YOLOv8 Object Detection", "loaded": True},
        "clip": {"description": "CLIP Vision Model", "loaded": True},
    }

    return mock_manager


@pytest.fixture
def mock_runpod_client():
    """Mock RunPod API client."""
    mock_client = AsyncMock()

    # Mock pod operations
    mock_client.get_pods.return_value = {
        "data": [
            {
                "id": "test_pod_id",
                "name": "test_pod",
                "runtime": {"ports": [{"publicPort": 8000}]},
                "desiredStatus": "RUNNING",
                "lastStatusChange": "2025-01-01T00:00:00Z",
            }
        ]
    }

    mock_client.create_pod.return_value = {
        "data": {"id": "new_pod_id", "status": "CREATED"}
    }

    mock_client.stop_pod.return_value = {"status": "success"}
    mock_client.start_pod.return_value = {"status": "success"}
    mock_client.terminate_pod.return_value = {"status": "success"}

    return mock_client


@pytest.fixture
def mock_secrets_manager():
    """Mock Google Secret Manager."""
    mock_manager = Mock()
    mock_manager.get_secret.return_value = "mock_secret_value"

    with patch("src.utils.secrets.get_secret_manager", return_value=mock_manager):
        yield mock_manager


@pytest.fixture
async def test_app():
    """Create test FastAPI application."""
    # First-party imports
    from src.control.api.main import app

    # Override dependencies with mocks
    app.dependency_overrides = {}

    yield app

    # Cleanup
    app.dependency_overrides = {}


@pytest.fixture
def test_client(test_app):
    """Create test client for API testing."""
    with TestClient(test_app) as client:
        yield client


@pytest.fixture
def auth_headers():
    """Create authorization headers for API testing."""
    # Mock JWT token (in real tests, you'd generate a valid test token)
    return {"Authorization": "Bearer test_token"}


@pytest.fixture
def sample_analysis_result():
    """Create a sample analysis result for testing."""
    return {
        "analysis_id": "test_analysis_123",
        "timestamp": "2025-01-01T00:00:00Z",
        "image_hash": "test_hash_123",
        "detections": [
            {
                "class": "cup",
                "confidence": 0.9,
                "bbox": [100, 100, 200, 200],
                "is_mug_related": True,
            }
        ],
        "embeddings": np.random.randn(512).tolist(),
        "mug_positions": [
            {
                "x": 150,
                "y": 150,
                "confidence": 0.85,
                "strategy": "hybrid",
                "reasoning": "Detected mug with high confidence",
            }
        ],
        "confidence_scores": {"detection": 0.9, "positioning": 0.85},
        "processing_time_ms": 250,
        "gpu_memory_mb": 512,
        "cached": False,
    }


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for testing."""
    return {
        "cold_start_ms": 2000,  # 2 seconds max cold start
        "warm_start_ms": 100,  # 100ms max warm start
        "image_processing_ms": 500,  # 500ms max for 1080p image
        "cache_hit_rate": 0.85,  # 85% min cache hit rate
        "gpu_utilization": 0.70,  # 70% min GPU utilization
        "api_latency_p95_ms": 200,  # 200ms max p95 latency
    }


# Utility functions for tests
def create_test_tensor(shape=(1, 3, 640, 640), dtype=torch.float32):
    """Create a test tensor with specified shape."""
    return torch.randn(shape, dtype=dtype)


def create_mock_detection(bbox, confidence, class_id, class_name):
    """Create a mock YOLO detection."""
    return {
        "bbox": bbox,
        "confidence": confidence,
        "class_id": class_id,
        "class": class_name,
        "is_mug_related": class_name in ["cup", "mug", "bottle"],
    }


def assert_performance_threshold(actual_ms: float, threshold_ms: float, operation: str):
    """Assert that performance meets threshold."""
    assert (
        actual_ms <= threshold_ms
    ), f"{operation} took {actual_ms}ms, exceeding threshold of {threshold_ms}ms"


def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")


def skip_if_no_external_services():
    """Skip test if external services are not available."""
    return pytest.mark.skipif(
        os.environ.get("SKIP_EXTERNAL_TESTS", "false").lower() == "true",
        reason="External services not available",
    )


# Test data generators
@pytest.fixture
def generate_test_images():
    """Generator for creating multiple test images."""

    def _generate(count: int, size=(640, 640)):
        images = []
        for i in range(count):
            image = Image.new("RGB", size, color=(i * 50 % 255, 100, 150))
            images.append(image)
        return images

    return _generate


@pytest.fixture
def mock_webhook_server():
    """Mock webhook server for notification testing."""
    # Standard library imports
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from unittest.mock import Mock

    received_requests = []

    class MockHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            received_requests.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": body.decode("utf-8"),
                }
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

    # Find available port
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = HTTPServer(("localhost", port), MockHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    yield {
        "url": f"http://localhost:{port}",
        "requests": received_requests,
        "server": server,
    }

    server.shutdown()
    server.server_close()


# Pytest plugins and hooks
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU access"
    )
    config.addinivalue_line(
        "markers", "requires_external: mark test as requiring external services"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test paths."""
    for item in items:
        # Add markers based on test file paths
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        if "gpu" in str(item.fspath) or "model" in str(item.fspath):
            item.add_marker(pytest.mark.gpu)
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Automatically cleanup GPU memory after tests."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
