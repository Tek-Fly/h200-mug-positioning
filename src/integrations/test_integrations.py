"""Tests for external integrations."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any

from .templated import TemplatedClient, TemplatedAPIError
from .n8n import N8NClient, N8NEventType, N8NWebhookError
from .notifications import NotificationClient, NotificationType, NotificationChannel
from .manager import IntegrationManager


class TestTemplatedClient:
    """Test Templated.io client."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session."""
        session = AsyncMock()
        response = AsyncMock()
        response.status = 200
        response.json.return_value = {"render_url": "https://example.com/render.png"}
        session.request.return_value.__aenter__.return_value = response
        return session
    
    @pytest.mark.asyncio
    async def test_render_design(self, mock_session):
        """Test design rendering."""
        with patch.dict("os.environ", {"TEMPLATED_API_KEY": "test_key"}):
            client = TemplatedClient()
            client.session = mock_session
            
            positioning_data = {
                "mug_position": {"x": 100, "y": 200, "width": 120, "height": 180},
                "confidence": 0.92,
                "template_params": {"text": "Test"}
            }
            
            result = await client.render_design("test-template", positioning_data)
            
            assert result["render_url"] == "https://example.com/render.png"
            mock_session.request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transform_positioning_to_template(self):
        """Test positioning data transformation."""
        with patch.dict("os.environ", {"TEMPLATED_API_KEY": "test_key"}):
            client = TemplatedClient()
            
            positioning_data = {
                "mug_position": {"x": 100, "y": 200, "width": 120, "height": 180},
                "confidence": 0.85,
                "bounding_box": {"x1": 50, "y1": 150, "x2": 170, "y2": 330},
                "template_params": {"custom_text": "Hello"}
            }
            
            result = client._transform_positioning_to_template(positioning_data)
            
            assert result["mug_x"] == 100
            assert result["mug_y"] == 200
            assert result["confidence_score"] == 0.85
            assert result["positioning_quality"] == "good"
            assert result["custom_text"] == "Hello"
            assert result["bbox_x1"] == 50


class TestN8NClient:
    """Test N8N client."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session."""
        session = AsyncMock()
        response = AsyncMock()
        response.status = 200
        response.json.return_value = {"success": True}
        session.post.return_value.__aenter__.return_value = response
        return session
    
    @pytest.mark.asyncio
    async def test_send_event(self, mock_session):
        """Test sending event to N8N."""
        with patch.dict("os.environ", {"N8N_WEBHOOK_URL": "https://test.n8n.com/webhook"}):
            client = N8NClient()
            client.session = mock_session
            
            result = await client.send_event(
                N8NEventType.MUG_DETECTED,
                {"image_id": "test_123", "mug_count": 2}
            )
            
            assert result["success"] is True
            mock_session.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enrich_event(self):
        """Test event enrichment."""
        with patch.dict("os.environ", {"N8N_WEBHOOK_URL": "https://test.n8n.com/webhook"}):
            client = N8NClient()
            
            enriched = client._enrich_event(
                N8NEventType.MUG_DETECTED,
                {"test": "data"}
            )
            
            assert enriched["event_type"] == "mug_detected"
            assert enriched["source"] == "h200-mug-positioning"
            assert enriched["version"] == "1.0"
            assert enriched["data"]["test"] == "data"
            assert "timestamp" in enriched
    
    @pytest.mark.asyncio
    async def test_batch_events(self, mock_session):
        """Test batch event sending."""
        with patch.dict("os.environ", {"N8N_WEBHOOK_URL": "https://test.n8n.com/webhook"}):
            client = N8NClient()
            client.session = mock_session
            
            events = [
                {"type": N8NEventType.MUG_DETECTED, "data": {"image_id": "1"}},
                {"type": N8NEventType.POSITIONING_ANALYZED, "data": {"image_id": "2"}}
            ]
            
            result = await client.send_batch_events(events)
            
            assert result["success"] is True
            mock_session.post.assert_called_once()


class TestNotificationClient:
    """Test notification client."""
    
    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session."""
        session = AsyncMock()
        response = AsyncMock()
        response.status = 200
        response.raise_for_status = MagicMock()
        session.post.return_value.__aenter__.return_value = response
        return session
    
    @pytest.mark.asyncio
    async def test_send_notification_webhook(self, mock_session):
        """Test webhook notification."""
        with patch.dict("os.environ", {"WEBHOOK_URL": "https://test.webhook.com"}):
            client = NotificationClient()
            client.session = mock_session
            
            results = await client.send_notification(
                title="Test Alert",
                message="This is a test",
                notification_type=NotificationType.INFO,
                channels=[NotificationChannel.WEBHOOK]
            )
            
            assert "webhook" in results
            mock_session.post.assert_called_once()
    
    def test_get_channels_for_type(self):
        """Test channel selection based on notification type."""
        with patch.dict("os.environ", {
            "WEBHOOK_URL": "https://test.webhook.com",
            "SLACK_WEBHOOK_URL": "https://test.slack.com"
        }):
            client = NotificationClient()
            
            # Critical should use all available channels
            critical_channels = client._get_channels_for_type(NotificationType.CRITICAL)
            assert NotificationChannel.WEBHOOK in critical_channels
            assert NotificationChannel.SLACK in critical_channels
            
            # Info should use webhook and discord only
            info_channels = client._get_channels_for_type(NotificationType.INFO)
            assert NotificationChannel.WEBHOOK in info_channels
    
    def test_format_email_html(self):
        """Test email HTML formatting."""
        client = NotificationClient()
        
        data = {
            "title": "Test Alert",
            "message": "This is a test message",
            "type": "warning",
            "timestamp": "2024-01-01T12:00:00Z",
            "data": {"key": "value"}
        }
        
        html = client._format_email_html(data)
        
        assert "Test Alert" in html
        assert "This is a test message" in html
        assert "WARNING" in html
        assert "2024-01-01T12:00:00Z" in html
        assert '"key": "value"' in html


class TestIntegrationManager:
    """Test integration manager."""
    
    @pytest.fixture
    def mock_clients(self):
        """Mock all integration clients."""
        templated = AsyncMock()
        n8n = AsyncMock()
        notifications = AsyncMock()
        
        return {
            "templated": templated,
            "n8n": n8n,
            "notifications": notifications
        }
    
    @pytest.mark.asyncio
    async def test_process_positioning_complete(self, mock_clients):
        """Test complete positioning workflow."""
        manager = IntegrationManager()
        manager.templated_client = mock_clients["templated"]
        manager.n8n_client = mock_clients["n8n"]
        manager.notification_client = mock_clients["notifications"]
        manager._health_status = {"templated": True, "n8n": True, "notifications": True}
        
        # Mock render result
        mock_clients["templated"].render_design.return_value = {
            "render_url": "https://example.com/render.png"
        }
        
        positioning_data = {
            "mug_position": {"x": 100, "y": 200},
            "confidence": 0.92,
            "rules_applied": ["test_rule"],
            "recommendations": ["Perfect position"]
        }
        
        results = await manager.process_positioning_complete(
            image_id="test_123",
            positioning_data=positioning_data,
            render_template="test-template"
        )
        
        assert "templated" in results
        assert "n8n" in results
        assert "notifications" in results
        assert results["templated"].success
        assert results["n8n"].success
        assert results["notifications"].success
    
    @pytest.mark.asyncio
    async def test_send_system_alert(self, mock_clients):
        """Test system alert sending."""
        manager = IntegrationManager()
        manager.n8n_client = mock_clients["n8n"]
        manager.notification_client = mock_clients["notifications"]
        manager._health_status = {"n8n": True, "notifications": True}
        
        results = await manager.send_system_alert(
            title="Test Alert",
            message="This is a test alert",
            severity="warning",
            data={"metric": "cpu_usage", "value": 85}
        )
        
        assert "n8n" in results
        assert "notifications" in results
        mock_clients["n8n"].send_system_alert.assert_called_once()
        mock_clients["notifications"].send_notification.assert_called_once()
    
    def test_get_health_status(self):
        """Test health status retrieval."""
        manager = IntegrationManager()
        manager._health_status = {
            "templated": True,
            "n8n": False,
            "notifications": True
        }
        
        health = manager.get_health_status()
        
        assert health["templated"] is True
        assert health["n8n"] is False
        assert health["notifications"] is True


# Integration test examples
async def test_real_integrations():
    """
    Integration test examples (requires real API keys).
    Run with: pytest -s tests/test_integrations.py::test_real_integrations
    """
    print("\n=== Integration Test Examples ===")
    
    # Test with actual environment variables
    async with IntegrationManager() as manager:
        health = await manager.check_integration_health()
        print(f"Integration health: {health}")
        
        # If any integrations are healthy, test them
        if health.get("notifications", {}).get("healthy"):
            print("Testing notification system...")
            try:
                await manager.send_system_alert(
                    title="Integration Test",
                    message="Testing notification system from pytest",
                    severity="info",
                    data={"test": True, "timestamp": "2024-01-01T12:00:00Z"}
                )
                print("✓ Notification test passed")
            except Exception as e:
                print(f"✗ Notification test failed: {e}")
        
        if health.get("n8n", {}).get("healthy"):
            print("Testing N8N integration...")
            try:
                await manager.n8n_client.send_event(
                    N8NEventType.SYSTEM_ALERT,
                    {"message": "Integration test from pytest", "test": True}
                )
                print("✓ N8N test passed")
            except Exception as e:
                print(f"✗ N8N test failed: {e}")


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_real_integrations())