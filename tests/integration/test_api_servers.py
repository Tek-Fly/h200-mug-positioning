"""Integration tests for servers API endpoints."""

# Standard library imports
from unittest.mock import AsyncMock, patch

# Third-party imports
import pytest
from fastapi import status

# First-party imports
from tests.base import APITestBase, IntegrationTestBase


@pytest.mark.integration
class TestServersAPI(APITestBase, IntegrationTestBase):
    """Test cases for servers API endpoints."""

    async def setup_method_async(self):
        """Setup servers API tests."""
        await super().setup_method_async()
        await self.mock_database_operations()

    def test_list_servers(self, test_client, auth_headers):
        """Test listing all servers."""
        mock_servers = {
            "serverless": {
                "type": "serverless",
                "status": "running",
                "instances": [
                    {
                        "id": "pod_serverless_1",
                        "status": "running",
                        "uptime_seconds": 3600,
                        "requests_handled": 150,
                        "cpu_usage": 45.2,
                        "memory_usage": 67.8,
                        "gpu_usage": 78.5,
                    }
                ],
                "auto_shutdown_enabled": True,
                "idle_timeout_seconds": 600,
            },
            "timed": {
                "type": "timed",
                "status": "stopped",
                "instances": [],
                "schedule": {
                    "start_time": "08:00",
                    "stop_time": "18:00",
                    "timezone": "UTC",
                    "enabled": True,
                },
                "next_scheduled_start": "2025-01-02T08:00:00Z",
            },
        }

        with patch(
            "src.control.manager.orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.list_servers.return_value = mock_servers
            mock_get_orch.return_value = mock_orchestrator

            response = test_client.get(f"{self.base_url}/servers", headers=auth_headers)

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert "serverless" in data
            assert "timed" in data
            assert data["serverless"]["status"] == "running"
            assert len(data["serverless"]["instances"]) == 1
            assert data["timed"]["status"] == "stopped"

    def test_get_server_details(self, test_client, auth_headers):
        """Test getting specific server details."""
        server_type = "serverless"

        mock_server_details = {
            "type": "serverless",
            "status": "running",
            "instances": [
                {
                    "id": "pod_serverless_1",
                    "status": "running",
                    "created_at": "2025-01-01T10:00:00Z",
                    "uptime_seconds": 7200,
                    "requests_handled": 300,
                    "avg_response_time_ms": 180,
                    "last_request_at": "2025-01-01T12:00:00Z",
                    "resources": {
                        "cpu_usage": 45.2,
                        "memory_usage_mb": 2048,
                        "memory_limit_mb": 4096,
                        "gpu_usage": 78.5,
                        "gpu_memory_mb": 1536,
                        "gpu_memory_limit_mb": 2048,
                    },
                    "network": {
                        "public_ip": "192.168.1.100",
                        "ports": [{"internal": 8000, "external": 80}],
                    },
                }
            ],
            "configuration": {
                "auto_shutdown_enabled": True,
                "idle_timeout_seconds": 600,
                "max_instances": 5,
                "min_instances": 0,
            },
            "metrics": {
                "total_requests": 300,
                "successful_requests": 295,
                "failed_requests": 5,
                "avg_processing_time_ms": 245,
            },
        }

        with patch(
            "src.control.manager.orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_server_details.return_value = mock_server_details
            mock_get_orch.return_value = mock_orchestrator

            response = test_client.get(
                f"{self.base_url}/servers/{server_type}", headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["type"] == "serverless"
            assert data["status"] == "running"
            assert len(data["instances"]) == 1
            assert "configuration" in data
            assert "metrics" in data

    def test_start_server_success(self, test_client, auth_headers):
        """Test starting a server successfully."""
        server_type = "serverless"

        with patch(
            "src.control.manager.orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.start_server.return_value = {
                "success": True,
                "message": "Server started successfully",
                "instance_id": "pod_new_123",
                "startup_time_seconds": 45,
            }
            mock_get_orch.return_value = mock_orchestrator

            response = test_client.post(
                f"{self.base_url}/servers/{server_type}/control",
                json={"action": "start"},
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["success"] is True
            assert data["message"] == "Server started successfully"
            assert "instance_id" in data
            mock_orchestrator.start_server.assert_called_once_with(server_type)

    def test_stop_server_success(self, test_client, auth_headers):
        """Test stopping a server successfully."""
        server_type = "serverless"

        with patch(
            "src.control.manager.orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.stop_server.return_value = {
                "success": True,
                "message": "Server stopped successfully",
                "instances_stopped": 2,
                "shutdown_time_seconds": 15,
            }
            mock_get_orch.return_value = mock_orchestrator

            response = test_client.post(
                f"{self.base_url}/servers/{server_type}/control",
                json={"action": "stop"},
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["success"] is True
            assert data["instances_stopped"] == 2
            mock_orchestrator.stop_server.assert_called_once_with(server_type)

    def test_restart_server_success(self, test_client, auth_headers):
        """Test restarting a server successfully."""
        server_type = "timed"

        with patch(
            "src.control.manager.orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.restart_server.return_value = {
                "success": True,
                "message": "Server restarted successfully",
                "old_instances_stopped": 1,
                "new_instance_id": "pod_restart_456",
                "total_restart_time_seconds": 60,
            }
            mock_get_orch.return_value = mock_orchestrator

            response = test_client.post(
                f"{self.base_url}/servers/{server_type}/control",
                json={"action": "restart"},
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["success"] is True
            assert "new_instance_id" in data
            mock_orchestrator.restart_server.assert_called_once_with(server_type)

    def test_server_control_invalid_action(self, test_client, auth_headers):
        """Test server control with invalid action."""
        server_type = "serverless"

        response = test_client.post(
            f"{self.base_url}/servers/{server_type}/control",
            json={"action": "invalid_action"},
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        self.assert_error_response(data, "INVALID_ACTION")

    def test_server_control_failure(self, test_client, auth_headers):
        """Test server control operation failure."""
        server_type = "serverless"

        with patch(
            "src.control.manager.orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.start_server.side_effect = Exception("RunPod API error")
            mock_get_orch.return_value = mock_orchestrator

            response = test_client.post(
                f"{self.base_url}/servers/{server_type}/control",
                json={"action": "start"},
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            assert "error" in data

    def test_get_server_logs(self, test_client, auth_headers):
        """Test getting server logs."""
        server_type = "serverless"
        instance_id = "pod_123"

        mock_logs = {
            "instance_id": instance_id,
            "logs": [
                {
                    "timestamp": "2025-01-01T12:00:00Z",
                    "level": "INFO",
                    "message": "Server started successfully",
                    "source": "main",
                },
                {
                    "timestamp": "2025-01-01T12:01:00Z",
                    "level": "INFO",
                    "message": "Model loaded: YOLOv8n",
                    "source": "model_manager",
                },
                {
                    "timestamp": "2025-01-01T12:02:00Z",
                    "level": "DEBUG",
                    "message": "Processing image analysis request",
                    "source": "analyzer",
                },
            ],
            "pagination": {"offset": 0, "limit": 100, "total": 3, "has_more": False},
        }

        with patch(
            "src.control.manager.orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_server_logs.return_value = mock_logs
            mock_get_orch.return_value = mock_orchestrator

            response = test_client.get(
                f"{self.base_url}/servers/{server_type}/logs",
                params={"instance_id": instance_id, "limit": 100, "level": "INFO"},
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["instance_id"] == instance_id
            assert len(data["logs"]) == 3
            assert data["logs"][0]["level"] == "INFO"
            assert "pagination" in data

    def test_get_server_metrics(self, test_client, auth_headers):
        """Test getting server metrics."""
        server_type = "serverless"

        mock_metrics = {
            "server_type": server_type,
            "current_metrics": {
                "cpu_usage": 45.2,
                "memory_usage_mb": 2048,
                "gpu_usage": 78.5,
                "network_io_mb": 150.5,
                "requests_per_minute": 25,
            },
            "historical_metrics": {
                "timestamps": [
                    "2025-01-01T11:00:00Z",
                    "2025-01-01T11:30:00Z",
                    "2025-01-01T12:00:00Z",
                ],
                "cpu_usage": [42.1, 44.8, 45.2],
                "memory_usage_mb": [1950, 2000, 2048],
                "gpu_usage": [75.2, 76.8, 78.5],
            },
            "performance_metrics": {
                "avg_response_time_ms": 180,
                "requests_handled": 300,
                "success_rate": 0.98,
                "cache_hit_rate": 0.87,
            },
        }

        with patch(
            "src.control.manager.metrics.get_server_metrics"
        ) as mock_get_metrics:
            mock_get_metrics.return_value = mock_metrics

            response = test_client.get(
                f"{self.base_url}/servers/{server_type}/metrics",
                params={"time_range": "1h", "granularity": "30m"},
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["server_type"] == server_type
            assert "current_metrics" in data
            assert "historical_metrics" in data
            assert data["current_metrics"]["cpu_usage"] == 45.2
            assert len(data["historical_metrics"]["timestamps"]) == 3

    def test_update_server_configuration(self, test_client, auth_headers):
        """Test updating server configuration."""
        server_type = "serverless"

        config_updates = {
            "auto_shutdown_enabled": False,
            "idle_timeout_seconds": 1200,
            "max_instances": 10,
            "min_instances": 1,
        }

        with patch(
            "src.control.manager.orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.update_server_config.return_value = {
                "success": True,
                "message": "Configuration updated successfully",
                "updated_fields": list(config_updates.keys()),
            }
            mock_get_orch.return_value = mock_orchestrator

            response = test_client.patch(
                f"{self.base_url}/servers/{server_type}/config",
                json=config_updates,
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["success"] is True
            assert len(data["updated_fields"]) == 4
            mock_orchestrator.update_server_config.assert_called_once_with(
                server_type, config_updates
            )

    def test_scale_server_instances(self, test_client, auth_headers):
        """Test scaling server instances."""
        server_type = "serverless"

        with patch(
            "src.control.manager.orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.scale_server.return_value = {
                "success": True,
                "message": "Server scaled successfully",
                "current_instances": 3,
                "target_instances": 3,
                "scaling_time_seconds": 30,
            }
            mock_get_orch.return_value = mock_orchestrator

            response = test_client.post(
                f"{self.base_url}/servers/{server_type}/scale",
                json={"target_instances": 3},
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["success"] is True
            assert data["current_instances"] == 3
            mock_orchestrator.scale_server.assert_called_once_with(server_type, 3)

    def test_get_server_health(self, test_client, auth_headers):
        """Test getting server health status."""
        server_type = "serverless"

        mock_health = {
            "server_type": server_type,
            "overall_status": "healthy",
            "instances": [
                {
                    "id": "pod_123",
                    "status": "healthy",
                    "health_checks": {
                        "api_responsive": True,
                        "models_loaded": True,
                        "gpu_accessible": True,
                        "database_connected": True,
                        "cache_connected": True,
                    },
                    "last_health_check": "2025-01-01T12:00:00Z",
                    "uptime_seconds": 7200,
                }
            ],
            "aggregated_health": {
                "healthy_instances": 1,
                "total_instances": 1,
                "health_percentage": 100.0,
            },
        }

        with patch(
            "src.control.manager.orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_server_health.return_value = mock_health
            mock_get_orch.return_value = mock_orchestrator

            response = test_client.get(
                f"{self.base_url}/servers/{server_type}/health", headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["overall_status"] == "healthy"
            assert len(data["instances"]) == 1
            assert data["instances"][0]["health_checks"]["models_loaded"] is True
            assert data["aggregated_health"]["health_percentage"] == 100.0

    def test_server_not_found(self, test_client, auth_headers):
        """Test accessing non-existent server type."""
        invalid_server_type = "nonexistent"

        response = test_client.get(
            f"{self.base_url}/servers/{invalid_server_type}", headers=auth_headers
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_servers_no_auth(self, test_client):
        """Test servers endpoints without authentication."""
        response = test_client.get(f"{self.base_url}/servers")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.integration
class TestServersWebSocket(IntegrationTestBase):
    """Test cases for servers WebSocket endpoints."""

    def test_server_logs_websocket(self, test_client):
        """Test real-time server logs via WebSocket."""
        server_type = "serverless"
        instance_id = "pod_123"

        with patch(
            "src.control.api.routers.websocket.LogStreamManager"
        ) as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager

            websocket_url = f"/ws/servers/{server_type}/logs?instance_id={instance_id}"

            with test_client.websocket_connect(websocket_url) as websocket:
                # Simulate receiving log message
                mock_log_message = {
                    "timestamp": "2025-01-01T12:00:00Z",
                    "level": "INFO",
                    "message": "Processing image analysis",
                    "source": "analyzer",
                }

                # In real implementation, logs would be streamed
                assert websocket is not None

    def test_server_metrics_websocket(self, test_client):
        """Test real-time server metrics via WebSocket."""
        server_type = "serverless"

        with patch(
            "src.control.api.routers.websocket.MetricsStreamManager"
        ) as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager

            websocket_url = f"/ws/servers/{server_type}/metrics"

            with test_client.websocket_connect(websocket_url) as websocket:
                # Simulate receiving metrics update
                mock_metrics = {
                    "timestamp": "2025-01-01T12:00:00Z",
                    "cpu_usage": 45.2,
                    "memory_usage": 67.8,
                    "gpu_usage": 78.5,
                }

                assert websocket is not None
