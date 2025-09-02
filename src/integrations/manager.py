"""Integration manager for coordinating external service integrations."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass

from .templated import TemplatedClient
from .n8n import N8NClient, N8NEventType
from .notifications import NotificationClient, NotificationType, NotificationChannel
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class IntegrationResult:
    """Result from integration operation."""
    success: bool
    service: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    response_time_ms: Optional[float] = None


class IntegrationManager:
    """
    Centralized manager for all external integrations.
    
    Features:
    - Coordinate multiple integration services
    - Batch operations across services
    - Health monitoring for integrations
    - Automatic fallback handling
    - Performance metrics collection
    
    Example:
        ```python
        manager = IntegrationManager()
        
        # Process complete positioning workflow
        await manager.process_positioning_complete(
            image_id="img_123",
            positioning_data={...},
            render_template="default-overlay"
        )
        
        # Send system alert to all channels
        await manager.send_system_alert(
            "High GPU Usage",
            "GPU usage exceeded 90%",
            severity="warning"
        )
        ```
    """
    
    def __init__(self):
        """Initialize integration manager."""
        self.templated_client: Optional[TemplatedClient] = None
        self.n8n_client: Optional[N8NClient] = None
        self.notification_client: Optional[NotificationClient] = None
        
        self._clients_initialized = False
        self._health_status: Dict[str, bool] = {}
        
        logger.info("Initialized integration manager")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_clients()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_clients()
    
    async def _initialize_clients(self):
        """Initialize all integration clients."""
        try:
            self.templated_client = TemplatedClient()
            await self.templated_client.__aenter__()
            self._health_status["templated"] = True
            logger.info("Templated.io client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Templated.io client: {e}")
            self._health_status["templated"] = False
        
        try:
            self.n8n_client = N8NClient()
            await self.n8n_client.__aenter__()
            self._health_status["n8n"] = True
            logger.info("N8N client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize N8N client: {e}")
            self._health_status["n8n"] = False
        
        try:
            self.notification_client = NotificationClient()
            await self.notification_client.__aenter__()
            self._health_status["notifications"] = True
            logger.info("Notification client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize notification client: {e}")
            self._health_status["notifications"] = False
        
        self._clients_initialized = True
        
        # Send startup notification
        if self._health_status.get("notifications"):
            try:
                await self.notification_client.send_system_startup()
            except Exception as e:
                logger.error(f"Failed to send startup notification: {e}")
    
    async def _close_clients(self):
        """Close all integration clients."""
        if self.templated_client:
            await self.templated_client.__aexit__(None, None, None)
        if self.n8n_client:
            await self.n8n_client.__aexit__(None, None, None)
        if self.notification_client:
            await self.notification_client.__aexit__(None, None, None)
    
    def get_health_status(self) -> Dict[str, bool]:
        """Get health status of all integrations."""
        return self._health_status.copy()
    
    async def check_integration_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Check health of all integrations.
        
        Returns:
            Health status for each integration
        """
        health_results = {}
        
        # Check Templated.io
        if self.templated_client:
            try:
                start_time = datetime.now()
                await self.templated_client.list_templates()
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                health_results["templated"] = {
                    "healthy": True,
                    "response_time_ms": response_time
                }
            except Exception as e:
                health_results["templated"] = {
                    "healthy": False,
                    "error": str(e)
                }
        
        # For webhook-based services, we'll mark as healthy if they're configured
        health_results["n8n"] = {
            "healthy": self._health_status.get("n8n", False),
            "configured": bool(self.n8n_client)
        }
        
        health_results["notifications"] = {
            "healthy": self._health_status.get("notifications", False),
            "configured": bool(self.notification_client),
            "channels": []
        }
        
        if self.notification_client:
            config = self.notification_client.config
            if config.email_enabled:
                health_results["notifications"]["channels"].append("email")
            if config.webhook_enabled:
                health_results["notifications"]["channels"].append("webhook")
            if config.slack_enabled:
                health_results["notifications"]["channels"].append("slack")
            if config.discord_enabled:
                health_results["notifications"]["channels"].append("discord")
        
        return health_results
    
    async def process_positioning_complete(
        self,
        image_id: str,
        positioning_data: Dict[str, Any],
        render_template: Optional[str] = None,
        notify_completion: bool = True
    ) -> Dict[str, IntegrationResult]:
        """
        Process complete positioning workflow across all integrations.
        
        Args:
            image_id: ID of the analyzed image
            positioning_data: Complete positioning analysis data
            render_template: Template ID for rendering (optional)
            notify_completion: Whether to send completion notification
            
        Returns:
            Results from each integration
        """
        results = {}
        
        # 1. Send positioning analysis to N8N
        if self.n8n_client and self._health_status.get("n8n"):
            try:
                start_time = datetime.now()
                await self.n8n_client.send_positioning_analysis(
                    image_id=image_id,
                    positioning_result=positioning_data,
                    rules_applied=positioning_data.get("rules_applied", []),
                    recommendations=positioning_data.get("recommendations", [])
                )
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                results["n8n"] = IntegrationResult(
                    success=True,
                    service="n8n",
                    response_time_ms=response_time
                )
            except Exception as e:
                logger.error(f"Failed to send N8N event: {e}")
                results["n8n"] = IntegrationResult(
                    success=False,
                    service="n8n",
                    error=str(e)
                )
        
        # 2. Render design if template specified
        if render_template and self.templated_client and self._health_status.get("templated"):
            try:
                start_time = datetime.now()
                render_result = await self.templated_client.render_design(
                    template_id=render_template,
                    positioning_data=positioning_data
                )
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                results["templated"] = IntegrationResult(
                    success=True,
                    service="templated",
                    data=render_result,
                    response_time_ms=response_time
                )
            except Exception as e:
                logger.error(f"Failed to render design: {e}")
                results["templated"] = IntegrationResult(
                    success=False,
                    service="templated",
                    error=str(e)
                )
        
        # 3. Send completion notification if requested
        if notify_completion and self.notification_client and self._health_status.get("notifications"):
            try:
                start_time = datetime.now()
                confidence = positioning_data.get("confidence", 0)
                notification_type = NotificationType.SUCCESS if confidence > 0.8 else NotificationType.WARNING
                
                await self.notification_client.send_notification(
                    title="Positioning Analysis Complete",
                    message=f"Image {image_id} analyzed with {confidence:.1%} confidence",
                    notification_type=notification_type,
                    data={
                        "image_id": image_id,
                        "confidence": confidence,
                        "render_created": bool(render_template and results.get("templated", {}).success)
                    }
                )
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                results["notifications"] = IntegrationResult(
                    success=True,
                    service="notifications",
                    response_time_ms=response_time
                )
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
                results["notifications"] = IntegrationResult(
                    success=False,
                    service="notifications",
                    error=str(e)
                )
        
        return results
    
    async def send_system_alert(
        self,
        title: str,
        message: str,
        severity: str = "warning",
        data: Optional[Dict[str, Any]] = None,
        channels: Optional[List[str]] = None
    ) -> Dict[str, IntegrationResult]:
        """
        Send system alert across multiple integrations.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity (info, warning, error, critical)
            data: Additional alert data
            channels: Specific channels to use
            
        Returns:
            Results from each integration
        """
        results = {}
        
        # Map severity to notification type
        severity_map = {
            "info": NotificationType.INFO,
            "warning": NotificationType.WARNING,
            "error": NotificationType.ERROR,
            "critical": NotificationType.CRITICAL,
            "success": NotificationType.SUCCESS
        }
        notification_type = severity_map.get(severity, NotificationType.WARNING)
        
        # Send to N8N
        if self.n8n_client and self._health_status.get("n8n"):
            try:
                start_time = datetime.now()
                await self.n8n_client.send_system_alert(
                    alert_type=severity,
                    message=f"{title}: {message}",
                    metrics=data
                )
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                results["n8n"] = IntegrationResult(
                    success=True,
                    service="n8n",
                    response_time_ms=response_time
                )
            except Exception as e:
                logger.error(f"Failed to send N8N alert: {e}")
                results["n8n"] = IntegrationResult(
                    success=False,
                    service="n8n",
                    error=str(e)
                )
        
        # Send to notification channels
        if self.notification_client and self._health_status.get("notifications"):
            try:
                start_time = datetime.now()
                
                # Convert string channels to enum
                channel_enums = None
                if channels:
                    channel_map = {
                        "email": NotificationChannel.EMAIL,
                        "webhook": NotificationChannel.WEBHOOK,
                        "slack": NotificationChannel.SLACK,
                        "discord": NotificationChannel.DISCORD
                    }
                    channel_enums = [channel_map[ch] for ch in channels if ch in channel_map]
                
                await self.notification_client.send_notification(
                    title=title,
                    message=message,
                    notification_type=notification_type,
                    data=data,
                    channels=channel_enums
                )
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                results["notifications"] = IntegrationResult(
                    success=True,
                    service="notifications",
                    response_time_ms=response_time
                )
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
                results["notifications"] = IntegrationResult(
                    success=False,
                    service="notifications",
                    error=str(e)
                )
        
        return results
    
    async def process_error_event(
        self,
        error_type: str,
        error_message: str,
        context: Dict[str, Any],
        severity: str = "error"
    ) -> Dict[str, IntegrationResult]:
        """
        Process error event across all relevant integrations.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Error context
            severity: Error severity
            
        Returns:
            Results from each integration
        """
        results = {}
        
        # Send to N8N
        if self.n8n_client and self._health_status.get("n8n"):
            try:
                start_time = datetime.now()
                await self.n8n_client.send_error(
                    error_type=error_type,
                    error_message=error_message,
                    context=context,
                    severity=severity
                )
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                results["n8n"] = IntegrationResult(
                    success=True,
                    service="n8n",
                    response_time_ms=response_time
                )
            except Exception as e:
                logger.error(f"Failed to send error to N8N: {e}")
                results["n8n"] = IntegrationResult(
                    success=False,
                    service="n8n",
                    error=str(e)
                )
        
        # Send error notification
        if self.notification_client and self._health_status.get("notifications"):
            try:
                start_time = datetime.now()
                await self.notification_client.send_error_alert(
                    error_type=error_type,
                    error_message=error_message,
                    context=context
                )
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                results["notifications"] = IntegrationResult(
                    success=True,
                    service="notifications",
                    response_time_ms=response_time
                )
            except Exception as e:
                logger.error(f"Failed to send error notification: {e}")
                results["notifications"] = IntegrationResult(
                    success=False,
                    service="notifications",
                    error=str(e)
                )
        
        return results
    
    async def process_batch_complete(
        self,
        batch_id: str,
        batch_stats: Dict[str, Any],
        render_results: bool = False
    ) -> Dict[str, IntegrationResult]:
        """
        Process batch completion event.
        
        Args:
            batch_id: ID of completed batch
            batch_stats: Statistics about the batch
            render_results: Whether to render result visualizations
            
        Returns:
            Results from each integration
        """
        results = {}
        
        # Send batch completion to N8N
        if self.n8n_client and self._health_status.get("n8n"):
            try:
                start_time = datetime.now()
                await self.n8n_client.send_event(
                    N8NEventType.BATCH_COMPLETED,
                    {
                        "batch_id": batch_id,
                        "stats": batch_stats,
                        "completed_at": datetime.utcnow().isoformat()
                    }
                )
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                results["n8n"] = IntegrationResult(
                    success=True,
                    service="n8n",
                    response_time_ms=response_time
                )
            except Exception as e:
                logger.error(f"Failed to send batch event to N8N: {e}")
                results["n8n"] = IntegrationResult(
                    success=False,
                    service="n8n",
                    error=str(e)
                )
        
        # Render batch summary if requested
        if render_results and self.templated_client and self._health_status.get("templated"):
            try:
                start_time = datetime.now()
                render_result = await self.templated_client.render_design(
                    template_id="batch-summary",
                    positioning_data={
                        "batch_id": batch_id,
                        "template_params": batch_stats
                    }
                )
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                results["templated"] = IntegrationResult(
                    success=True,
                    service="templated",
                    data=render_result,
                    response_time_ms=response_time
                )
            except Exception as e:
                logger.error(f"Failed to render batch summary: {e}")
                results["templated"] = IntegrationResult(
                    success=False,
                    service="templated",
                    error=str(e)
                )
        
        # Send completion notification
        if self.notification_client and self._health_status.get("notifications"):
            try:
                start_time = datetime.now()
                success_rate = batch_stats.get("success_rate", 0)
                notification_type = NotificationType.SUCCESS if success_rate > 0.95 else NotificationType.WARNING
                
                await self.notification_client.send_notification(
                    title="Batch Processing Complete",
                    message=f"Batch {batch_id} completed with {success_rate:.1%} success rate",
                    notification_type=notification_type,
                    data=batch_stats
                )
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                results["notifications"] = IntegrationResult(
                    success=True,
                    service="notifications",
                    response_time_ms=response_time
                )
            except Exception as e:
                logger.error(f"Failed to send batch notification: {e}")
                results["notifications"] = IntegrationResult(
                    success=False,
                    service="notifications",
                    error=str(e)
                )
        
        return results
    
    async def process_model_update(
        self,
        model_name: str,
        version: str,
        performance_metrics: Dict[str, Any]
    ) -> Dict[str, IntegrationResult]:
        """
        Process model update event.
        
        Args:
            model_name: Name of updated model
            version: New model version
            performance_metrics: Model performance data
            
        Returns:
            Results from each integration
        """
        results = {}
        
        # Send to N8N
        if self.n8n_client and self._health_status.get("n8n"):
            try:
                start_time = datetime.now()
                await self.n8n_client.send_event(
                    N8NEventType.MODEL_UPDATED,
                    {
                        "model_name": model_name,
                        "version": version,
                        "metrics": performance_metrics,
                        "updated_at": datetime.utcnow().isoformat()
                    }
                )
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                results["n8n"] = IntegrationResult(
                    success=True,
                    service="n8n",
                    response_time_ms=response_time
                )
            except Exception as e:
                results["n8n"] = IntegrationResult(
                    success=False,
                    service="n8n",
                    error=str(e)
                )
        
        # Send notification
        if self.notification_client and self._health_status.get("notifications"):
            try:
                start_time = datetime.now()
                await self.notification_client.send_notification(
                    title=f"Model Updated: {model_name}",
                    message=f"Model {model_name} updated to version {version}",
                    notification_type=NotificationType.INFO,
                    data={
                        "model": model_name,
                        "version": version,
                        **performance_metrics
                    }
                )
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                results["notifications"] = IntegrationResult(
                    success=True,
                    service="notifications",
                    response_time_ms=response_time
                )
            except Exception as e:
                results["notifications"] = IntegrationResult(
                    success=False,
                    service="notifications",
                    error=str(e)
                )
        
        return results
    
    async def get_integration_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from all integrations.
        
        Returns:
            Combined metrics from all services
        """
        metrics = {
            "health": self.get_health_status(),
            "services": {
                "templated": {
                    "enabled": bool(self.templated_client),
                    "healthy": self._health_status.get("templated", False)
                },
                "n8n": {
                    "enabled": bool(self.n8n_client),
                    "healthy": self._health_status.get("n8n", False)
                },
                "notifications": {
                    "enabled": bool(self.notification_client),
                    "healthy": self._health_status.get("notifications", False)
                }
            }
        }
        
        # Add notification channel details
        if self.notification_client:
            config = self.notification_client.config
            metrics["services"]["notifications"]["channels"] = {
                "email": config.email_enabled,
                "webhook": config.webhook_enabled,
                "slack": config.slack_enabled,
                "discord": config.discord_enabled
            }
        
        return metrics


# Usage example
async def main():
    """Example usage of IntegrationManager."""
    async with IntegrationManager() as manager:
        # Example 1: Process complete positioning workflow
        positioning_data = {
            "mug_position": {"x": 150, "y": 200, "width": 120, "height": 180},
            "confidence": 0.92,
            "rules_applied": ["minimum_spacing", "edge_detection"],
            "recommendations": ["Position is optimal"],
            "quality_score": 0.91
        }
        
        results = await manager.process_positioning_complete(
            image_id="img_12345",
            positioning_data=positioning_data,
            render_template="overlay-template",
            notify_completion=True
        )
        print(f"Positioning workflow results: {results}")
        
        # Example 2: Send system alert
        alert_results = await manager.send_system_alert(
            title="High GPU Usage",
            message="GPU usage has exceeded 90% for 5 minutes",
            severity="warning",
            data={
                "gpu_usage": 92.5,
                "threshold": 90,
                "duration": "5 minutes"
            }
        )
        print(f"Alert results: {alert_results}")
        
        # Example 3: Check integration health
        health = await manager.check_integration_health()
        print(f"Integration health: {health}")
        
        # Example 4: Process model update
        model_results = await manager.process_model_update(
            model_name="YOLO",
            version="1.2.0",
            performance_metrics={
                "accuracy": 0.95,
                "inference_time_ms": 45,
                "model_size_mb": 250
            }
        )
        print(f"Model update results: {model_results}")


if __name__ == "__main__":
    asyncio.run(main())