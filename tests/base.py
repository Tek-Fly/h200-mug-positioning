"""Base test classes for H200 test suite."""

# Standard library imports
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

# Third-party imports
import pytest
import structlog

# Suppress logging during tests
structlog.configure(
    processors=[structlog.testing.LogCapture()],
    logger_factory=structlog.testing.LogCapture,
    cache_logger_on_first_use=True,
)


class BaseTest(ABC):
    """Base class for all tests with common utilities."""

    def setup_method(self):
        """Setup method called before each test."""
        self.start_time = time.time()

    def teardown_method(self):
        """Teardown method called after each test."""
        elapsed = time.time() - self.start_time
        print(f"Test completed in {elapsed:.3f}s")

    def assert_performance(self, actual_ms: float, threshold_ms: float, operation: str):
        """Assert performance meets threshold."""
        assert (
            actual_ms <= threshold_ms
        ), f"{operation} took {actual_ms}ms, exceeding threshold of {threshold_ms}ms"

    def assert_gpu_memory_reasonable(self, memory_mb: float, max_mb: float = 2048):
        """Assert GPU memory usage is reasonable."""
        assert (
            0 <= memory_mb <= max_mb
        ), f"GPU memory usage {memory_mb}MB is not reasonable (max: {max_mb}MB)"

    def assert_confidence_scores(self, scores: Dict[str, float]):
        """Assert confidence scores are valid."""
        for key, score in scores.items():
            assert (
                0.0 <= score <= 1.0
            ), f"Confidence score '{key}': {score} not in valid range [0.0, 1.0]"

    def assert_bbox_valid(self, bbox: List[float], image_size: tuple = (640, 640)):
        """Assert bounding box is valid."""
        x1, y1, x2, y2 = bbox
        width, height = image_size

        assert 0 <= x1 < x2 <= width, f"Invalid bbox x coordinates: {x1}, {x2}"
        assert 0 <= y1 < y2 <= height, f"Invalid bbox y coordinates: {y1}, {y2}"


class AsyncBaseTest(BaseTest):
    """Base class for async tests."""

    async def setup_method_async(self):
        """Async setup method."""
        await asyncio.sleep(0)  # Ensure event loop is running

    async def teardown_method_async(self):
        """Async teardown method."""
        await asyncio.sleep(0)  # Cleanup

    async def wait_for_condition(
        self, condition_func, timeout_s: float = 5.0, check_interval_s: float = 0.1
    ):
        """Wait for a condition to become True."""
        start_time = time.time()
        while time.time() - start_time < timeout_s:
            if (
                await condition_func()
                if asyncio.iscoroutinefunction(condition_func)
                else condition_func()
            ):
                return True
            await asyncio.sleep(check_interval_s)

        raise TimeoutError(f"Condition not met within {timeout_s}s")


class ModelTestBase(AsyncBaseTest):
    """Base class for model-related tests."""

    def setup_method(self):
        """Setup for model tests."""
        super().setup_method()
        self.mock_gpu_memory = 512.0
        self.mock_inference_time = 50.0

    def create_mock_yolo_result(self, detections: List[Dict[str, Any]]):
        """Create mock YOLO detection result."""
        # Standard library imports
        from unittest.mock import Mock

        # Third-party imports
        import torch

        mock_result = Mock()
        mock_boxes = Mock()

        if detections:
            # Convert detections to tensors
            data = []
            confs = []
            classes = []
            xyxy = []

            for det in detections:
                bbox = det["bbox"]
                conf = det["confidence"]
                cls_id = det.get("class_id", 0)

                data.append(bbox + [conf, cls_id])
                confs.append(conf)
                classes.append(cls_id)
                xyxy.append(bbox)

            mock_boxes.data = torch.tensor(data)
            mock_boxes.conf = torch.tensor(confs)
            mock_boxes.cls = torch.tensor(classes, dtype=torch.int64)
            mock_boxes.xyxy = torch.tensor(xyxy)
        else:
            # Empty detection
            mock_boxes.data = torch.empty((0, 6))
            mock_boxes.conf = torch.empty(0)
            mock_boxes.cls = torch.empty(0, dtype=torch.int64)
            mock_boxes.xyxy = torch.empty((0, 4))

        mock_result.boxes = mock_boxes
        return [mock_result]

    def create_mock_clip_embedding(self, size: int = 512):
        """Create mock CLIP embedding."""
        # Third-party imports
        import torch

        return torch.randn(1, size)


class APITestBase(BaseTest):
    """Base class for API tests."""

    def setup_method(self):
        """Setup for API tests."""
        super().setup_method()
        self.base_url = "/api/v1"
        self.auth_headers = {"Authorization": "Bearer test_token"}

    def assert_api_response_format(
        self, response_data: Dict[str, Any], required_fields: List[str]
    ):
        """Assert API response has required format."""
        for field in required_fields:
            assert field in response_data, f"Missing required field: {field}"

    def assert_error_response(self, response_data: Dict[str, Any], expected_code: str):
        """Assert error response format."""
        assert "error" in response_data
        assert response_data.get("code") == expected_code


class IntegrationTestBase(AsyncBaseTest):
    """Base class for integration tests."""

    async def setup_method_async(self):
        """Setup for integration tests."""
        await super().setup_method_async()

        # Mock external services
        self.mock_mongodb = AsyncMock()
        self.mock_redis = AsyncMock()
        self.mock_r2 = AsyncMock()
        self.mock_runpod = AsyncMock()

    async def mock_database_operations(self):
        """Setup mock database operations."""
        # MongoDB mocks
        self.mock_mongodb.insert_one.return_value = AsyncMock(inserted_id="test_id")
        self.mock_mongodb.find_one.return_value = {"_id": "test_id", "data": "test"}
        self.mock_mongodb.update_one.return_value = AsyncMock(modified_count=1)

        # Redis mocks
        self.mock_redis.get.return_value = None
        self.mock_redis.set.return_value = True
        self.mock_redis.delete.return_value = 1

        # R2 mocks
        self.mock_r2.put_object.return_value = {"ETag": "test_etag"}
        self.mock_r2.get_object.return_value = {"Body": AsyncMock()}


class E2ETestBase(AsyncBaseTest):
    """Base class for end-to-end tests."""

    async def setup_method_async(self):
        """Setup for E2E tests."""
        await super().setup_method_async()
        self.test_images = []
        self.cleanup_tasks = []

    async def teardown_method_async(self):
        """Cleanup after E2E tests."""
        # Run cleanup tasks
        for task in self.cleanup_tasks:
            try:
                await task()
            except Exception as e:
                print(f"Cleanup task failed: {e}")

        await super().teardown_method_async()

    def add_cleanup_task(self, task):
        """Add cleanup task to run after test."""
        self.cleanup_tasks.append(task)

    async def simulate_user_workflow(self, steps: List[Dict[str, Any]]):
        """Simulate a complete user workflow."""
        results = []

        for step in steps:
            step_type = step.get("type")
            step_data = step.get("data", {})

            if step_type == "upload_image":
                result = await self._simulate_image_upload(step_data)
            elif step_type == "analyze_image":
                result = await self._simulate_image_analysis(step_data)
            elif step_type == "update_rules":
                result = await self._simulate_rule_update(step_data)
            elif step_type == "check_status":
                result = await self._simulate_status_check(step_data)
            else:
                raise ValueError(f"Unknown step type: {step_type}")

            results.append(result)

            # Add delay between steps
            await asyncio.sleep(step.get("delay", 0.1))

        return results

    async def _simulate_image_upload(self, data):
        """Simulate image upload step."""
        # Mock implementation
        return {"status": "uploaded", "image_id": "test_image_123"}

    async def _simulate_image_analysis(self, data):
        """Simulate image analysis step."""
        # Mock implementation
        return {
            "status": "analyzed",
            "analysis_id": "analysis_123",
            "detections": [],
            "positions": [],
        }

    async def _simulate_rule_update(self, data):
        """Simulate rule update step."""
        # Mock implementation
        return {"status": "updated", "rule_id": "rule_123"}

    async def _simulate_status_check(self, data):
        """Simulate status check step."""
        # Mock implementation
        return {"status": "running", "health": "healthy"}


class PerformanceTestBase(AsyncBaseTest):
    """Base class for performance tests."""

    def __init__(self):
        super().__init__()
        self.performance_metrics = {}

    async def setup_method_async(self):
        """Setup for performance tests."""
        await super().setup_method_async()
        self.performance_metrics.clear()

    def record_metric(self, name: str, value: float, unit: str = "ms"):
        """Record a performance metric."""
        self.performance_metrics[name] = {
            "value": value,
            "unit": unit,
            "timestamp": time.time(),
        }

    async def measure_async_operation(self, operation, operation_name: str):
        """Measure async operation performance."""
        start_time = time.time()
        result = await operation()
        end_time = time.time()

        duration_ms = (end_time - start_time) * 1000
        self.record_metric(operation_name, duration_ms)

        return result, duration_ms

    def measure_sync_operation(self, operation, operation_name: str):
        """Measure sync operation performance."""
        start_time = time.time()
        result = operation()
        end_time = time.time()

        duration_ms = (end_time - start_time) * 1000
        self.record_metric(operation_name, duration_ms)

        return result, duration_ms

    def assert_performance_targets(self, targets: Dict[str, float]):
        """Assert all performance targets are met."""
        for metric_name, target_value in targets.items():
            metric = self.performance_metrics.get(metric_name)
            assert (
                metric is not None
            ), f"Performance metric '{metric_name}' not recorded"

            actual_value = metric["value"]
            assert actual_value <= target_value, (
                f"Performance target not met for '{metric_name}': "
                f"{actual_value}{metric['unit']} > {target_value}{metric['unit']}"
            )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all performance metrics."""
        return {
            "metrics": self.performance_metrics,
            "total_metrics": len(self.performance_metrics),
            "test_duration": time.time() - self.start_time,
        }


class MockServiceMixin:
    """Mixin for common service mocking."""

    def setup_mock_services(self):
        """Setup all mock external services."""
        self.setup_mock_mongodb()
        self.setup_mock_redis()
        self.setup_mock_r2()
        self.setup_mock_runpod()
        self.setup_mock_models()

    def setup_mock_mongodb(self):
        """Setup MongoDB mocks."""
        self.mock_mongodb = AsyncMock()
        # Add standard mock responses

    def setup_mock_redis(self):
        """Setup Redis mocks."""
        self.mock_redis = AsyncMock()
        # Add standard mock responses

    def setup_mock_r2(self):
        """Setup R2 storage mocks."""
        self.mock_r2 = AsyncMock()
        # Add standard mock responses

    def setup_mock_runpod(self):
        """Setup RunPod API mocks."""
        self.mock_runpod = AsyncMock()
        # Add standard mock responses

    def setup_mock_models(self):
        """Setup model mocks."""
        self.mock_yolo = AsyncMock()
        self.mock_clip = AsyncMock()
        # Add standard mock responses


# Utility decorators for tests
def requires_gpu(func):
    """Decorator to skip test if GPU is not available."""
    # Third-party imports
    import torch

    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not available"
    )(func)


def requires_external_services(func):
    """Decorator to skip test if external services are not available."""
    # Standard library imports
    import os

    return pytest.mark.skipif(
        os.environ.get("SKIP_EXTERNAL_TESTS", "false").lower() == "true",
        reason="External services not available",
    )(func)


def slow_test(func):
    """Decorator to mark test as slow."""
    return pytest.mark.slow(func)


def performance_test(func):
    """Decorator to mark test as performance test."""
    return pytest.mark.performance(func)
