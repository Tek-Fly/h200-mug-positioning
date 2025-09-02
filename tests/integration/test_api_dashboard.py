"""Integration tests for dashboard API endpoints."""

# Standard library imports
from unittest.mock import AsyncMock, patch

# Third-party imports
import pytest
from fastapi import status

# First-party imports
from tests.base import APITestBase, IntegrationTestBase


@pytest.mark.integration
class TestDashboardAPI(APITestBase, IntegrationTestBase):
    """Test cases for dashboard API endpoints."""

    async def setup_method_async(self):
        """Setup dashboard API tests."""
        await super().setup_method_async()
        await self.mock_database_operations()

    def test_get_dashboard_overview(self, test_client, auth_headers):
        """Test getting dashboard overview."""
        mock_dashboard_data = {
            "system_status": "healthy",
            "active_servers": {
                "serverless": {"status": "running", "instances": 2},
                "timed": {"status": "stopped", "instances": 0},
            },
            "recent_analyses": {
                "total_today": 150,
                "successful": 145,
                "failed": 5,
                "avg_processing_time_ms": 245,
            },
            "model_status": {
                "yolo": {"loaded": True, "memory_mb": 512},
                "clip": {"loaded": True, "memory_mb": 1024},
            },
            "cache_stats": {
                "hit_rate": 0.87,
                "memory_usage_mb": 256,
                "redis_connected": True,
            },
        }

        with patch(
            "src.control.manager.orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_dashboard_data.return_value = mock_dashboard_data
            mock_get_orch.return_value = mock_orchestrator

            response = test_client.get(
                f"{self.base_url}/dashboard", headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Verify response structure
            required_fields = [
                "system_status",
                "active_servers",
                "recent_analyses",
                "model_status",
                "cache_stats",
            ]
            self.assert_api_response_format(data, required_fields)

            assert data["system_status"] == "healthy"
            assert data["recent_analyses"]["total_today"] == 150
            assert data["cache_stats"]["hit_rate"] == 0.87

    def test_get_system_metrics(self, test_client, auth_headers):
        """Test getting system metrics."""
        mock_metrics = {
            "timestamp": "2025-01-01T12:00:00Z",
            "server_metrics": {
                "cpu_usage_percent": 65.2,
                "memory_usage_percent": 78.5,
                "gpu_usage_percent": 45.3,
                "gpu_memory_usage_percent": 67.8,
            },
            "api_metrics": {
                "requests_per_minute": 25.5,
                "avg_response_time_ms": 180,
                "error_rate_percent": 2.1,
            },
            "processing_metrics": {
                "analyses_per_hour": 120,
                "avg_processing_time_ms": 245,
                "cache_hit_rate": 0.89,
            },
        }

        with patch(
            "src.control.manager.metrics.get_system_metrics"
        ) as mock_get_metrics:
            mock_get_metrics.return_value = mock_metrics

            response = test_client.get(
                f"{self.base_url}/dashboard/metrics", headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert "server_metrics" in data
            assert "api_metrics" in data
            assert "processing_metrics" in data
            assert data["server_metrics"]["cpu_usage_percent"] == 65.2
            assert data["api_metrics"]["requests_per_minute"] == 25.5

    def test_get_performance_charts(self, test_client, auth_headers):
        """Test getting performance chart data."""
        mock_chart_data = {
            "processing_times": {
                "timestamps": [
                    "2025-01-01T10:00:00Z",
                    "2025-01-01T11:00:00Z",
                    "2025-01-01T12:00:00Z",
                ],
                "values": [250, 245, 260],
            },
            "throughput": {
                "timestamps": [
                    "2025-01-01T10:00:00Z",
                    "2025-01-01T11:00:00Z",
                    "2025-01-01T12:00:00Z",
                ],
                "values": [15, 18, 22],
            },
            "cache_hit_rates": {
                "timestamps": [
                    "2025-01-01T10:00:00Z",
                    "2025-01-01T11:00:00Z",
                    "2025-01-01T12:00:00Z",
                ],
                "values": [0.85, 0.87, 0.89],
            },
        }

        with patch(
            "src.control.manager.metrics.get_performance_chart_data"
        ) as mock_get_charts:
            mock_get_charts.return_value = mock_chart_data

            response = test_client.get(
                f"{self.base_url}/dashboard/charts/performance",
                params={"timerange": "24h", "interval": "1h"},
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert "processing_times" in data
            assert "throughput" in data
            assert "cache_hit_rates" in data
            assert len(data["processing_times"]["timestamps"]) == 3
            assert len(data["throughput"]["values"]) == 3

    def test_get_cost_breakdown(self, test_client, auth_headers):
        """Test getting cost breakdown data."""
        mock_cost_data = {
            "current_month": {
                "total_usd": 245.67,
                "breakdown": {
                    "gpu_compute": 180.45,
                    "storage": 25.30,
                    "network": 15.20,
                    "other": 24.72,
                },
            },
            "daily_costs": {
                "dates": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "costs": [8.50, 7.80, 9.20],
            },
            "cost_per_analysis": 0.05,
            "projected_monthly": 312.45,
        }

        with patch("src.control.manager.metrics.get_cost_breakdown") as mock_get_costs:
            mock_get_costs.return_value = mock_cost_data

            response = test_client.get(
                f"{self.base_url}/dashboard/costs", headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert "current_month" in data
            assert "daily_costs" in data
            assert data["current_month"]["total_usd"] == 245.67
            assert data["cost_per_analysis"] == 0.05

    def test_get_activity_feed(self, test_client, auth_headers):
        """Test getting activity feed."""
        mock_activities = {
            "activities": [
                {
                    "id": "activity_1",
                    "timestamp": "2025-01-01T12:00:00Z",
                    "type": "analysis_completed",
                    "message": "Image analysis completed successfully",
                    "details": {
                        "analysis_id": "analysis_123",
                        "processing_time_ms": 245,
                    },
                },
                {
                    "id": "activity_2",
                    "timestamp": "2025-01-01T11:55:00Z",
                    "type": "rule_triggered",
                    "message": "High confidence rule triggered",
                    "details": {"rule_id": "rule_456", "confidence": 0.92},
                },
                {
                    "id": "activity_3",
                    "timestamp": "2025-01-01T11:50:00Z",
                    "type": "server_started",
                    "message": "Serverless instance started",
                    "details": {"server_type": "serverless", "instance_id": "pod_789"},
                },
            ],
            "pagination": {"offset": 0, "limit": 20, "total": 3, "has_more": False},
        }

        with patch(
            "src.control.manager.metrics.get_activity_feed"
        ) as mock_get_activities:
            mock_get_activities.return_value = mock_activities

            response = test_client.get(
                f"{self.base_url}/dashboard/activity",
                params={"limit": 20, "offset": 0},
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert "activities" in data
            assert "pagination" in data
            assert len(data["activities"]) == 3
            assert data["activities"][0]["type"] == "analysis_completed"
            assert data["pagination"]["total"] == 3

    def test_get_server_status(self, test_client, auth_headers):
        """Test getting server status information."""
        mock_server_status = {
            "serverless": {
                "status": "running",
                "instances": [
                    {
                        "id": "pod_1",
                        "status": "running",
                        "cpu_usage": 45.2,
                        "memory_usage": 67.8,
                        "gpu_usage": 78.5,
                        "uptime_seconds": 3600,
                        "last_request": "2025-01-01T12:00:00Z",
                    }
                ],
                "total_requests": 1250,
                "avg_response_time_ms": 180,
            },
            "timed": {
                "status": "stopped",
                "instances": [],
                "last_active": "2025-01-01T08:00:00Z",
                "scheduled_start": "2025-01-01T18:00:00Z",
            },
        }

        with patch(
            "src.control.manager.orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_server_status.return_value = mock_server_status
            mock_get_orch.return_value = mock_orchestrator

            response = test_client.get(
                f"{self.base_url}/dashboard/servers", headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert "serverless" in data
            assert "timed" in data
            assert data["serverless"]["status"] == "running"
            assert len(data["serverless"]["instances"]) == 1
            assert data["timed"]["status"] == "stopped"

    def test_get_alert_summary(self, test_client, auth_headers):
        """Test getting alert summary."""
        mock_alerts = {
            "active_alerts": [
                {
                    "id": "alert_1",
                    "severity": "warning",
                    "message": "GPU memory usage above 90%",
                    "timestamp": "2025-01-01T12:00:00Z",
                    "source": "resource_monitor",
                },
                {
                    "id": "alert_2",
                    "severity": "info",
                    "message": "Cache hit rate dropped below 85%",
                    "timestamp": "2025-01-01T11:45:00Z",
                    "source": "performance_monitor",
                },
            ],
            "alert_counts": {"critical": 0, "warning": 1, "info": 1, "total": 2},
            "recent_resolved": 5,
        }

        with patch("src.control.manager.metrics.get_alert_summary") as mock_get_alerts:
            mock_get_alerts.return_value = mock_alerts

            response = test_client.get(
                f"{self.base_url}/dashboard/alerts", headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert "active_alerts" in data
            assert "alert_counts" in data
            assert len(data["active_alerts"]) == 2
            assert data["alert_counts"]["warning"] == 1
            assert data["alert_counts"]["total"] == 2

    def test_get_health_check(self, test_client, auth_headers):
        """Test getting system health check."""
        mock_health = {
            "overall_status": "healthy",
            "components": {
                "api": {"status": "healthy", "response_time_ms": 5},
                "database": {"status": "healthy", "connection_pool": "80%"},
                "redis": {"status": "healthy", "memory_usage": "45%"},
                "gpu": {"status": "healthy", "utilization": "67%"},
                "storage": {"status": "healthy", "free_space": "78%"},
            },
            "uptime_seconds": 86400,
            "last_check": "2025-01-01T12:00:00Z",
        }

        with patch("src.control.manager.metrics.get_health_status") as mock_get_health:
            mock_get_health.return_value = mock_health

            response = test_client.get(
                f"{self.base_url}/dashboard/health", headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["overall_status"] == "healthy"
            assert "components" in data
            assert data["components"]["api"]["status"] == "healthy"
            assert data["components"]["gpu"]["status"] == "healthy"
            assert "uptime_seconds" in data

    def test_get_model_performance(self, test_client, auth_headers):
        """Test getting model performance metrics."""
        mock_model_perf = {
            "yolo": {
                "avg_inference_time_ms": 45,
                "total_inferences": 5000,
                "accuracy_metrics": {
                    "precision": 0.92,
                    "recall": 0.88,
                    "f1_score": 0.90,
                },
                "gpu_memory_mb": 512,
                "model_version": "yolov8n",
            },
            "clip": {
                "avg_inference_time_ms": 85,
                "total_inferences": 5000,
                "embedding_quality": 0.94,
                "gpu_memory_mb": 1024,
                "model_version": "ViT-B/32",
            },
            "positioning": {
                "avg_calculation_time_ms": 15,
                "accuracy_score": 0.87,
                "total_calculations": 4800,
            },
        }

        with patch(
            "src.control.manager.metrics.get_model_performance"
        ) as mock_get_model_perf:
            mock_get_model_perf.return_value = mock_model_perf

            response = test_client.get(
                f"{self.base_url}/dashboard/models/performance", headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert "yolo" in data
            assert "clip" in data
            assert "positioning" in data
            assert data["yolo"]["avg_inference_time_ms"] == 45
            assert data["clip"]["embedding_quality"] == 0.94

    def test_dashboard_no_auth(self, test_client):
        """Test dashboard access without authentication."""
        response = test_client.get(f"{self.base_url}/dashboard")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_dashboard_with_time_range_filter(self, test_client, auth_headers):
        """Test dashboard data with time range filters."""
        with patch(
            "src.control.manager.orchestrator.get_orchestrator"
        ) as mock_get_orch:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.get_dashboard_data.return_value = {
                "system_status": "healthy",
                "time_range": "24h",
                "data_points": 24,
            }
            mock_get_orch.return_value = mock_orchestrator

            response = test_client.get(
                f"{self.base_url}/dashboard",
                params={"time_range": "24h", "granularity": "1h"},
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["time_range"] == "24h"
            mock_orchestrator.get_dashboard_data.assert_called_once()


@pytest.mark.integration
class TestDashboardWebSocket(IntegrationTestBase):
    """Test cases for dashboard WebSocket endpoints."""

    def test_websocket_connection(self, test_client):
        """Test WebSocket connection for real-time updates."""
        with patch(
            "src.control.api.routers.websocket.ConnectionManager"
        ) as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager

            with test_client.websocket_connect("/ws/dashboard") as websocket:
                # Simulate receiving real-time data
                mock_data = {
                    "type": "metrics_update",
                    "data": {
                        "cpu_usage": 65.2,
                        "memory_usage": 78.5,
                        "timestamp": "2025-01-01T12:00:00Z",
                    },
                }

                # In real implementation, this would be sent by the server
                # Here we're just testing the connection works
                assert websocket is not None

    def test_websocket_authentication(self, test_client, auth_headers):
        """Test WebSocket authentication."""
        # WebSocket authentication testing would depend on specific implementation
        # This is a placeholder for the actual WebSocket auth test
        pass
