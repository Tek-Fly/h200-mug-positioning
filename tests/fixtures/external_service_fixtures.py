"""Fixtures for external service mocking."""

# Standard library imports
import asyncio
import io
import json
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

# Third-party imports
import pytest


class MockRunPodClient:
    """Mock RunPod API client."""

    def __init__(self):
        self._pods = {}
        self._next_pod_id = 1
        self._api_calls = []

    def _generate_pod_id(self):
        """Generate unique pod ID."""
        pod_id = f"mock_pod_{self._next_pod_id}"
        self._next_pod_id += 1
        return pod_id

    def _record_api_call(self, method, endpoint, **kwargs):
        """Record API call for testing."""
        self._api_calls.append(
            {
                "method": method,
                "endpoint": endpoint,
                "timestamp": datetime.now(timezone.utc),
                **kwargs,
            }
        )

    async def get_pods(self, pod_id=None):
        """Mock get_pods API call."""
        self._record_api_call("GET", "/pods", pod_id=pod_id)

        if pod_id:
            pod = self._pods.get(pod_id)
            return {"data": pod} if pod else {"data": None}

        return {"data": list(self._pods.values())}

    async def create_pod(self, pod_config):
        """Mock create_pod API call."""
        pod_id = self._generate_pod_id()

        pod = {
            "id": pod_id,
            "name": pod_config.get("name", f"pod_{pod_id}"),
            "imageName": pod_config.get("imageName", "default:latest"),
            "gpuType": pod_config.get("gpuType", "RTX4090"),
            "vcpuCount": pod_config.get("vcpuCount", 4),
            "memoryInGb": pod_config.get("memoryInGb", 16),
            "diskInGb": pod_config.get("diskInGb", 50),
            "desiredStatus": "RUNNING",
            "actualStatus": "PROVISIONING",
            "runtime": {
                "ports": [{"privatePort": 8000, "publicPort": 8000, "type": "http"}]
            },
            "machine": {"podHostId": "mock_host_123"},
            "lastStatusChange": datetime.now(timezone.utc).isoformat(),
            "containerDiskInGb": pod_config.get("containerDiskInGb", 20),
            "volumeInGb": pod_config.get("volumeInGb", 30),
            "costPerHr": 0.50,  # $0.50/hr mock cost
            "uptimeSeconds": 0,
        }

        self._pods[pod_id] = pod
        self._record_api_call("POST", "/pods", pod_config=pod_config, pod_id=pod_id)

        # Simulate provisioning delay
        await asyncio.sleep(0.1)
        pod["actualStatus"] = "RUNNING"

        return {"data": pod}

    async def start_pod(self, pod_id):
        """Mock start_pod API call."""
        self._record_api_call("POST", f"/pods/{pod_id}/start", pod_id=pod_id)

        if pod_id not in self._pods:
            return {"error": "Pod not found"}

        pod = self._pods[pod_id]
        pod["desiredStatus"] = "RUNNING"
        pod["actualStatus"] = "STARTING"
        pod["lastStatusChange"] = datetime.now(timezone.utc).isoformat()

        # Simulate startup delay
        await asyncio.sleep(0.1)
        pod["actualStatus"] = "RUNNING"
        pod["uptimeSeconds"] = 0

        return {"data": pod}

    async def stop_pod(self, pod_id):
        """Mock stop_pod API call."""
        self._record_api_call("POST", f"/pods/{pod_id}/stop", pod_id=pod_id)

        if pod_id not in self._pods:
            return {"error": "Pod not found"}

        pod = self._pods[pod_id]
        pod["desiredStatus"] = "STOPPED"
        pod["actualStatus"] = "STOPPING"
        pod["lastStatusChange"] = datetime.now(timezone.utc).isoformat()

        # Simulate stop delay
        await asyncio.sleep(0.05)
        pod["actualStatus"] = "STOPPED"

        return {"data": pod}

    async def terminate_pod(self, pod_id):
        """Mock terminate_pod API call."""
        self._record_api_call("POST", f"/pods/{pod_id}/terminate", pod_id=pod_id)

        if pod_id not in self._pods:
            return {"error": "Pod not found"}

        del self._pods[pod_id]
        return {"status": "success", "message": f"Pod {pod_id} terminated"}

    async def get_pod_logs(self, pod_id, lines=100):
        """Mock get_pod_logs API call."""
        self._record_api_call("GET", f"/pods/{pod_id}/logs", pod_id=pod_id, lines=lines)

        if pod_id not in self._pods:
            return {"error": "Pod not found"}

        # Generate mock logs
        mock_logs = []
        for i in range(min(lines, 50)):  # Max 50 mock log lines
            timestamp = datetime.now(timezone.utc).isoformat()
            log_levels = ["INFO", "DEBUG", "WARNING", "ERROR"]
            level = log_levels[i % len(log_levels)]
            messages = [
                "Server started successfully",
                "Model loaded: YOLOv8n",
                "Processing image analysis request",
                "Analysis completed",
                "GPU memory usage: 512MB",
                "Cache hit for image hash",
                "Positioning calculation completed",
            ]
            message = messages[i % len(messages)]

            mock_logs.append(f"{timestamp} [{level}] {message}")

        return {"data": "\n".join(mock_logs)}

    async def get_gpu_types(self):
        """Mock get_gpu_types API call."""
        self._record_api_call("GET", "/gpu-types")

        return {
            "data": [
                {
                    "id": "RTX4090",
                    "displayName": "RTX 4090",
                    "memoryInGb": 24,
                    "secureCloud": True,
                    "communityCloud": True,
                    "lowestPrice": {
                        "minimumBidPrice": 0.40,
                        "uninterruptablePrice": 0.70,
                    },
                },
                {
                    "id": "H100",
                    "displayName": "H100 80GB HBM3",
                    "memoryInGb": 80,
                    "secureCloud": True,
                    "communityCloud": False,
                    "lowestPrice": {
                        "minimumBidPrice": 2.50,
                        "uninterruptablePrice": 4.00,
                    },
                },
            ]
        }

    def get_api_call_history(self):
        """Get history of API calls for testing."""
        return self._api_calls

    def clear_api_call_history(self):
        """Clear API call history."""
        self._api_calls.clear()

    def reset_state(self):
        """Reset mock state."""
        self._pods.clear()
        self._api_calls.clear()
        self._next_pod_id = 1


class MockR2StorageClient:
    """Mock Cloudflare R2 storage client."""

    def __init__(self):
        self._objects = {}
        self._api_calls = []

    def _record_api_call(self, operation, **kwargs):
        """Record API call for testing."""
        self._api_calls.append(
            {"operation": operation, "timestamp": datetime.now(timezone.utc), **kwargs}
        )

    async def put_object(self, bucket, key, body, **kwargs):
        """Mock put_object operation."""
        self._record_api_call("put_object", bucket=bucket, key=key)

        # Simulate upload
        if isinstance(body, (str, bytes)):
            content = body
        else:
            content = body.read()

        object_info = {
            "key": key,
            "bucket": bucket,
            "content": content,
            "content_length": len(content),
            "content_type": kwargs.get("ContentType", "application/octet-stream"),
            "metadata": kwargs.get("Metadata", {}),
            "etag": f"mock_etag_{hash(content) % 10000}",
            "last_modified": datetime.now(timezone.utc),
            "storage_class": "STANDARD",
        }

        self._objects[f"{bucket}/{key}"] = object_info

        return {
            "ETag": object_info["etag"],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    async def get_object(self, bucket, key):
        """Mock get_object operation."""
        self._record_api_call("get_object", bucket=bucket, key=key)

        object_key = f"{bucket}/{key}"
        if object_key not in self._objects:
            raise Exception("NoSuchKey")

        obj = self._objects[object_key]

        # Create mock body stream
        body_stream = io.BytesIO(
            obj["content"]
            if isinstance(obj["content"], bytes)
            else obj["content"].encode()
        )

        return {
            "Body": body_stream,
            "ContentLength": obj["content_length"],
            "ContentType": obj["content_type"],
            "ETag": obj["etag"],
            "LastModified": obj["last_modified"],
            "Metadata": obj["metadata"],
        }

    async def delete_object(self, bucket, key):
        """Mock delete_object operation."""
        self._record_api_call("delete_object", bucket=bucket, key=key)

        object_key = f"{bucket}/{key}"
        if object_key in self._objects:
            del self._objects[object_key]
            return {"DeleteMarker": True}

        return {"DeleteMarker": False}

    async def list_objects_v2(self, bucket, prefix=None, max_keys=1000):
        """Mock list_objects_v2 operation."""
        self._record_api_call(
            "list_objects_v2", bucket=bucket, prefix=prefix, max_keys=max_keys
        )

        objects = []
        count = 0

        for object_key, obj_info in self._objects.items():
            if not object_key.startswith(f"{bucket}/"):
                continue

            key = object_key[len(bucket) + 1 :]  # Remove bucket prefix

            if prefix and not key.startswith(prefix):
                continue

            if count >= max_keys:
                break

            objects.append(
                {
                    "Key": key,
                    "LastModified": obj_info["last_modified"],
                    "ETag": obj_info["etag"],
                    "Size": obj_info["content_length"],
                    "StorageClass": obj_info["storage_class"],
                }
            )
            count += 1

        return {
            "Contents": objects,
            "KeyCount": len(objects),
            "MaxKeys": max_keys,
            "IsTruncated": False,
            "Name": bucket,
        }

    async def head_object(self, bucket, key):
        """Mock head_object operation."""
        self._record_api_call("head_object", bucket=bucket, key=key)

        object_key = f"{bucket}/{key}"
        if object_key not in self._objects:
            raise Exception("NoSuchKey")

        obj = self._objects[object_key]

        return {
            "ContentLength": obj["content_length"],
            "ContentType": obj["content_type"],
            "ETag": obj["etag"],
            "LastModified": obj["last_modified"],
            "Metadata": obj["metadata"],
        }

    def get_api_call_history(self):
        """Get history of API calls."""
        return self._api_calls

    def clear_api_call_history(self):
        """Clear API call history."""
        self._api_calls.clear()

    def reset_state(self):
        """Reset mock state."""
        self._objects.clear()
        self._api_calls.clear()


class MockGoogleSecretManager:
    """Mock Google Secret Manager client."""

    def __init__(self):
        self._secrets = {
            "projects/test-project/secrets/runpod-api-key/versions/latest": "mock_runpod_key_123",
            "projects/test-project/secrets/mongodb-uri/versions/latest": "mongodb://mock:27017/test",
            "projects/test-project/secrets/redis-password/versions/latest": "mock_redis_pass",
            "projects/test-project/secrets/r2-access-key/versions/latest": "mock_r2_access_key",
            "projects/test-project/secrets/r2-secret-key/versions/latest": "mock_r2_secret_key",
            "projects/test-project/secrets/jwt-secret/versions/latest": "mock_jwt_secret_key_456",
            "projects/test-project/secrets/webhook-token/versions/latest": "mock_webhook_token_789",
        }
        self._api_calls = []

    def _record_api_call(self, operation, **kwargs):
        """Record API call."""
        self._api_calls.append(
            {"operation": operation, "timestamp": datetime.now(timezone.utc), **kwargs}
        )

    async def access_secret_version(self, request):
        """Mock access_secret_version."""
        name = request["name"]
        self._record_api_call("access_secret_version", name=name)

        if name not in self._secrets:
            raise Exception(f"Secret {name} not found")

        secret_value = self._secrets[name]

        # Mock response structure
        response = Mock()
        response.payload = Mock()
        response.payload.data = secret_value.encode("utf-8")

        return response

    def add_secret(self, name, value):
        """Add secret for testing."""
        self._secrets[name] = value

    def get_api_call_history(self):
        """Get API call history."""
        return self._api_calls

    def clear_api_call_history(self):
        """Clear API call history."""
        self._api_calls.clear()


class MockWebhookServer:
    """Mock webhook server for notification testing."""

    def __init__(self, port=8080):
        self.port = port
        self.received_webhooks = []
        self.server = None
        self.is_running = False

    async def start(self):
        """Start mock webhook server."""
        # Third-party imports
        import aiohttp
        from aiohttp import ClientSession, web

        async def webhook_handler(request):
            """Handle incoming webhooks."""
            try:
                # Record the webhook
                webhook_data = {
                    "timestamp": datetime.now(timezone.utc),
                    "method": request.method,
                    "path": request.path,
                    "headers": dict(request.headers),
                    "query": dict(request.query),
                    "body": await request.text() if request.content_length else None,
                }

                if request.content_type == "application/json":
                    webhook_data["json"] = await request.json()

                self.received_webhooks.append(webhook_data)

                return web.json_response(
                    {"status": "received", "id": len(self.received_webhooks)}
                )

            except Exception as e:
                return web.json_response({"error": str(e)}, status=400)

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", webhook_handler)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, "localhost", self.port)
        await site.start()

        self.server = runner
        self.is_running = True

        return f"http://localhost:{self.port}"

    async def stop(self):
        """Stop mock webhook server."""
        if self.server and self.is_running:
            await self.server.cleanup()
            self.is_running = False

    def get_received_webhooks(self):
        """Get all received webhooks."""
        return self.received_webhooks

    def clear_webhooks(self):
        """Clear received webhooks."""
        self.received_webhooks.clear()


@pytest.fixture
def mock_runpod_client():
    """Fixture for mock RunPod client."""
    return MockRunPodClient()


@pytest.fixture
def mock_r2_client():
    """Fixture for mock R2 storage client."""
    return MockR2StorageClient()


@pytest.fixture
def mock_secret_manager():
    """Fixture for mock Google Secret Manager."""
    return MockGoogleSecretManager()


@pytest.fixture
async def mock_webhook_server():
    """Fixture for mock webhook server."""
    server = MockWebhookServer()
    url = await server.start()

    yield {
        "url": url,
        "server": server,
        "get_webhooks": server.get_received_webhooks,
        "clear_webhooks": server.clear_webhooks,
    }

    await server.stop()


@pytest.fixture
def mock_external_services(mock_runpod_client, mock_r2_client, mock_secret_manager):
    """Combined fixture for all external services."""
    return {
        "runpod": mock_runpod_client,
        "r2_storage": mock_r2_client,
        "secret_manager": mock_secret_manager,
    }


class MockNotificationService:
    """Mock notification service."""

    def __init__(self):
        self.sent_notifications = []
        self.webhook_endpoints = []
        self.email_config = None
        self.slack_config = None

    async def send_webhook_notification(self, url, payload, headers=None):
        """Mock webhook notification."""
        notification = {
            "type": "webhook",
            "url": url,
            "payload": payload,
            "headers": headers or {},
            "timestamp": datetime.now(timezone.utc),
            "status": "sent",
            "id": f"webhook_{len(self.sent_notifications) + 1}",
        }

        self.sent_notifications.append(notification)

        # Simulate success/failure
        if "fail" in url:
            notification["status"] = "failed"
            notification["error"] = "Connection refused"
            return False

        return True

    async def send_email_notification(self, to_email, subject, body):
        """Mock email notification."""
        notification = {
            "type": "email",
            "to": to_email,
            "subject": subject,
            "body": body,
            "timestamp": datetime.now(timezone.utc),
            "status": "sent",
            "id": f"email_{len(self.sent_notifications) + 1}",
        }

        self.sent_notifications.append(notification)
        return True

    async def send_slack_notification(self, channel, message, attachments=None):
        """Mock Slack notification."""
        notification = {
            "type": "slack",
            "channel": channel,
            "message": message,
            "attachments": attachments,
            "timestamp": datetime.now(timezone.utc),
            "status": "sent",
            "id": f"slack_{len(self.sent_notifications) + 1}",
        }

        self.sent_notifications.append(notification)
        return True

    def get_sent_notifications(self, notification_type=None):
        """Get sent notifications."""
        if notification_type:
            return [
                n for n in self.sent_notifications if n["type"] == notification_type
            ]
        return self.sent_notifications

    def clear_notifications(self):
        """Clear sent notifications."""
        self.sent_notifications.clear()

    def configure_webhook(self, url, headers=None):
        """Configure webhook endpoint."""
        self.webhook_endpoints.append({"url": url, "headers": headers or {}})

    def configure_email(self, smtp_server, username, password):
        """Configure email settings."""
        self.email_config = {
            "smtp_server": smtp_server,
            "username": username,
            "password": password,
        }

    def configure_slack(self, webhook_url, token=None):
        """Configure Slack settings."""
        self.slack_config = {"webhook_url": webhook_url, "token": token}


@pytest.fixture
def mock_notification_service():
    """Fixture for mock notification service."""
    return MockNotificationService()


class ExternalServiceManager:
    """Manager for coordinating external service mocks."""

    def __init__(self, runpod_client, r2_client, secret_manager, notification_service):
        self.runpod = runpod_client
        self.r2_storage = r2_client
        self.secret_manager = secret_manager
        self.notifications = notification_service

    def reset_all_mocks(self):
        """Reset all mock states."""
        self.runpod.reset_state()
        self.r2_storage.reset_state()
        self.secret_manager.clear_api_call_history()
        self.notifications.clear_notifications()

    def get_all_api_calls(self):
        """Get API calls from all services."""
        return {
            "runpod": self.runpod.get_api_call_history(),
            "r2_storage": self.r2_storage.get_api_call_history(),
            "secret_manager": self.secret_manager.get_api_call_history(),
        }

    async def simulate_deployment_workflow(self):
        """Simulate complete deployment workflow."""
        # Create pod
        pod_config = {
            "name": "h200-test-pod",
            "imageName": "h200-mug-analyzer:latest",
            "gpuType": "RTX4090",
        }

        create_result = await self.runpod.create_pod(pod_config)
        pod_id = create_result["data"]["id"]

        # Upload config to R2
        config_data = json.dumps({"model_config": "test"})
        await self.r2_storage.put_object(
            "config-bucket", f"{pod_id}/config.json", config_data
        )

        # Send notification
        await self.notifications.send_webhook_notification(
            "http://localhost:8080/webhook", {"event": "pod_created", "pod_id": pod_id}
        )

        return pod_id

    async def simulate_error_scenarios(self):
        """Simulate various error scenarios."""
        errors = []

        # Test RunPod API error
        try:
            await self.runpod.get_pods("nonexistent_pod")
        except Exception as e:
            errors.append(f"runpod_error: {str(e)}")

        # Test R2 storage error
        try:
            await self.r2_storage.get_object("nonexistent-bucket", "nonexistent-key")
        except Exception as e:
            errors.append(f"r2_error: {str(e)}")

        return errors


@pytest.fixture
def external_service_manager(mock_external_services, mock_notification_service):
    """Fixture for external service manager."""
    return ExternalServiceManager(
        runpod_client=mock_external_services["runpod"],
        r2_client=mock_external_services["r2_storage"],
        secret_manager=mock_external_services["secret_manager"],
        notification_service=mock_notification_service,
    )
