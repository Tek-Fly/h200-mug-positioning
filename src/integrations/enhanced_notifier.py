"""Enhanced notification system that integrates WebSocket and external integrations."""

# Standard library imports
import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# Local imports
from ..control.manager.notifier import (
    NotificationPriority,
    NotificationTopic,
    WebSocketNotifier,
)
from ..utils.logging_config import get_logger
from .manager import IntegrationManager
from .n8n import N8NEventType
from .notifications import NotificationType

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EnhancedNotifier:
    """
    Enhanced notification system that combines WebSocket notifications
    with external integrations (N8N, email, Slack, Discord).

    Features:
    - Unified notification interface
    - Multi-channel delivery (WebSocket + external)
    - Priority-based routing
    - Automatic escalation for critical alerts
    - Performance monitoring and metrics

    Example:
        ```python
        notifier = EnhancedNotifier()
        await notifier.start()

        # Send system alert to all channels
        await notifier.send_system_alert(
            title="High GPU Usage",
            message="GPU usage exceeded 90%",
            severity=AlertSeverity.WARNING,
            data={"gpu_usage": 92.5, "threshold": 90}
        )

        # Send positioning complete event
        await notifier.send_positioning_complete(
            image_id="img_123",
            positioning_data={...},
            render_template="overlay-template"
        )
        ```
    """

    def __init__(self):
        """Initialize enhanced notifier."""
        self.ws_notifier = WebSocketNotifier()
        self.integration_manager: Optional[IntegrationManager] = None
        self.is_running = False

        # Performance metrics
        self.metrics = {
            "notifications_sent": 0,
            "external_notifications_sent": 0,
            "websocket_notifications_sent": 0,
            "failed_notifications": 0,
            "average_response_time_ms": 0,
        }

        logger.info("Initialized enhanced notifier")

    async def start(self):
        """Start the enhanced notification system."""
        if self.is_running:
            return

        logger.info("Starting enhanced notification system")
        self.is_running = True

        # Start WebSocket notifier
        await self.ws_notifier.start()

        # Initialize integration manager
        try:
            self.integration_manager = IntegrationManager()
            await self.integration_manager.__aenter__()
            logger.info("External integrations initialized")
        except Exception as e:
            logger.error(f"Failed to initialize external integrations: {e}")
            self.integration_manager = None

        # Send startup notification
        await self.send_system_alert(
            title="Enhanced Notification System Started",
            message="All notification channels are now active",
            severity=AlertSeverity.INFO,
            data={"startup_time": datetime.utcnow().isoformat()},
        )

    async def stop(self):
        """Stop the enhanced notification system."""
        if not self.is_running:
            return

        logger.info("Stopping enhanced notification system")
        self.is_running = False

        # Stop WebSocket notifier
        await self.ws_notifier.stop()

        # Close integration manager
        if self.integration_manager:
            await self.integration_manager.__aexit__(None, None, None)
            self.integration_manager = None

    async def send_system_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        data: Optional[Dict[str, Any]] = None,
        channels: Optional[List[str]] = None,
    ):
        """
        Send system alert to all configured channels.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            data: Additional alert data
            channels: Specific external channels to use
        """
        start_time = datetime.now()
        alert_data = data or {}

        # Add metadata
        alert_data.update(
            {
                "alert_id": f"alert_{int(datetime.now().timestamp())}",
                "severity": severity.value,
                "source": "h200-positioning-system",
            }
        )

        # Send to WebSocket clients
        try:
            ws_priority = self._map_severity_to_ws_priority(severity)
            await self.ws_notifier.notify_alert(
                alert_type="system",
                severity=severity.value,
                message=f"{title}: {message}",
                metadata=alert_data,
            )
            self.metrics["websocket_notifications_sent"] += 1
        except Exception as e:
            logger.error(f"Failed to send WebSocket alert: {e}")
            self.metrics["failed_notifications"] += 1

        # Send to external integrations
        if self.integration_manager:
            try:
                await self.integration_manager.send_system_alert(
                    title=title,
                    message=message,
                    severity=severity.value,
                    data=alert_data,
                    channels=channels,
                )
                self.metrics["external_notifications_sent"] += 1
            except Exception as e:
                logger.error(f"Failed to send external alert: {e}")
                self.metrics["failed_notifications"] += 1

        # Update metrics
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics["notifications_sent"] += 1
        self._update_average_response_time(response_time)

    async def send_positioning_complete(
        self,
        image_id: str,
        positioning_data: Dict[str, Any],
        render_template: Optional[str] = None,
    ):
        """
        Send positioning analysis complete event.

        Args:
            image_id: ID of analyzed image
            positioning_data: Complete positioning analysis data
            render_template: Optional template for rendering
        """
        start_time = datetime.now()

        # Send to WebSocket clients
        try:
            await self.ws_notifier.notify_activity(
                activity_type="positioning_complete",
                description=f"Positioning analysis completed for image {image_id}",
                metadata={
                    "image_id": image_id,
                    "confidence": positioning_data.get("confidence", 0),
                    "processing_time": positioning_data.get("processing_time", 0),
                },
            )
            self.metrics["websocket_notifications_sent"] += 1
        except Exception as e:
            logger.error(f"Failed to send WebSocket activity: {e}")
            self.metrics["failed_notifications"] += 1

        # Send to external integrations
        if self.integration_manager:
            try:
                results = await self.integration_manager.process_positioning_complete(
                    image_id=image_id,
                    positioning_data=positioning_data,
                    render_template=render_template,
                    notify_completion=True,
                )

                # Count successful external notifications
                successful_external = sum(1 for r in results.values() if r.success)
                self.metrics["external_notifications_sent"] += successful_external

                # Log render URL if available
                if "templated" in results and results["templated"].success:
                    render_url = results["templated"].data.get("render_url")
                    if render_url:
                        logger.info(f"Design rendered for {image_id}: {render_url}")

                        # Send render complete notification
                        await self.ws_notifier.notify_activity(
                            activity_type="design_rendered",
                            description=f"Design rendered for image {image_id}",
                            metadata={
                                "image_id": image_id,
                                "render_url": render_url,
                                "template": render_template,
                            },
                        )

            except Exception as e:
                logger.error(f"Failed to process external integrations: {e}")
                self.metrics["failed_notifications"] += 1

        # Update metrics
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics["notifications_sent"] += 1
        self._update_average_response_time(response_time)

    async def send_batch_complete(
        self,
        batch_id: str,
        batch_stats: Dict[str, Any],
        create_summary_report: bool = True,
    ):
        """
        Send batch processing complete event.

        Args:
            batch_id: ID of completed batch
            batch_stats: Batch processing statistics
            create_summary_report: Whether to create visual summary report
        """
        start_time = datetime.now()

        # Send to WebSocket clients
        try:
            await self.ws_notifier.notify_activity(
                activity_type="batch_complete",
                description=f"Batch {batch_id} processing completed",
                metadata={"batch_id": batch_id, "stats": batch_stats},
            )
            self.metrics["websocket_notifications_sent"] += 1
        except Exception as e:
            logger.error(f"Failed to send WebSocket batch notification: {e}")
            self.metrics["failed_notifications"] += 1

        # Send to external integrations
        if self.integration_manager:
            try:
                results = await self.integration_manager.process_batch_complete(
                    batch_id=batch_id,
                    batch_stats=batch_stats,
                    render_results=create_summary_report,
                )

                successful_external = sum(1 for r in results.values() if r.success)
                self.metrics["external_notifications_sent"] += successful_external

            except Exception as e:
                logger.error(f"Failed to process batch external integrations: {e}")
                self.metrics["failed_notifications"] += 1

        # Update metrics
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics["notifications_sent"] += 1
        self._update_average_response_time(response_time)

    async def send_error_event(
        self,
        error_type: str,
        error_message: str,
        context: Dict[str, Any],
        severity: AlertSeverity = AlertSeverity.ERROR,
    ):
        """
        Send error event to all channels.

        Args:
            error_type: Type of error
            error_message: Error message
            context: Error context
            severity: Error severity
        """
        start_time = datetime.now()

        # Send to WebSocket clients
        try:
            await self.ws_notifier.notify_alert(
                alert_type=error_type,
                severity=severity.value,
                message=error_message,
                metadata=context,
            )
            self.metrics["websocket_notifications_sent"] += 1
        except Exception as e:
            logger.error(f"Failed to send WebSocket error: {e}")
            self.metrics["failed_notifications"] += 1

        # Send to external integrations
        if self.integration_manager:
            try:
                await self.integration_manager.process_error_event(
                    error_type=error_type,
                    error_message=error_message,
                    context=context,
                    severity=severity.value,
                )
                self.metrics["external_notifications_sent"] += 1
            except Exception as e:
                logger.error(f"Failed to send external error notification: {e}")
                self.metrics["failed_notifications"] += 1

        # Update metrics
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics["notifications_sent"] += 1
        self._update_average_response_time(response_time)

    async def send_performance_alert(
        self,
        metric_name: str,
        current_value: float,
        threshold: float,
        trend: Optional[str] = None,
    ):
        """
        Send performance alert.

        Args:
            metric_name: Name of the performance metric
            current_value: Current metric value
            threshold: Threshold value
            trend: Performance trend (increasing, decreasing, stable)
        """
        # Determine severity based on how much threshold is exceeded
        exceeded_ratio = current_value / threshold
        if exceeded_ratio > 2.0:
            severity = AlertSeverity.CRITICAL
        elif exceeded_ratio > 1.5:
            severity = AlertSeverity.ERROR
        elif exceeded_ratio > 1.0:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        await self.send_system_alert(
            title=f"Performance Alert: {metric_name}",
            message=f"{metric_name} is {current_value:.2f} (threshold: {threshold:.2f})",
            severity=severity,
            data={
                "metric_name": metric_name,
                "current_value": current_value,
                "threshold": threshold,
                "exceeded_by_percent": f"{((exceeded_ratio - 1) * 100):.1f}%",
                "trend": trend,
            },
        )

    async def send_model_update(
        self, model_name: str, version: str, performance_metrics: Dict[str, Any]
    ):
        """
        Send model update notification.

        Args:
            model_name: Name of updated model
            version: New version
            performance_metrics: Performance data
        """
        start_time = datetime.now()

        # Send to WebSocket clients
        try:
            await self.ws_notifier.notify_activity(
                activity_type="model_updated",
                description=f"Model {model_name} updated to version {version}",
                metadata={
                    "model_name": model_name,
                    "version": version,
                    "metrics": performance_metrics,
                },
            )
            self.metrics["websocket_notifications_sent"] += 1
        except Exception as e:
            logger.error(f"Failed to send WebSocket model update: {e}")
            self.metrics["failed_notifications"] += 1

        # Send to external integrations
        if self.integration_manager:
            try:
                await self.integration_manager.process_model_update(
                    model_name=model_name,
                    version=version,
                    performance_metrics=performance_metrics,
                )
                self.metrics["external_notifications_sent"] += 1
            except Exception as e:
                logger.error(f"Failed to send external model update: {e}")
                self.metrics["failed_notifications"] += 1

        # Update metrics
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics["notifications_sent"] += 1
        self._update_average_response_time(response_time)

    async def send_mug_detection(
        self,
        image_id: str,
        mug_count: int,
        positions: List[Dict[str, Any]],
        confidence_scores: List[float],
        processing_time: float,
    ):
        """
        Send mug detection event.

        Args:
            image_id: ID of analyzed image
            mug_count: Number of mugs detected
            positions: List of mug positions
            confidence_scores: Confidence scores
            processing_time: Processing time in seconds
        """
        start_time = datetime.now()

        # Send to WebSocket clients
        try:
            await self.ws_notifier.notify_activity(
                activity_type="mug_detected",
                description=f"Detected {mug_count} mugs in image {image_id}",
                metadata={
                    "image_id": image_id,
                    "mug_count": mug_count,
                    "positions": positions,
                    "confidence_scores": confidence_scores,
                    "processing_time": processing_time,
                    "average_confidence": (
                        sum(confidence_scores) / len(confidence_scores)
                        if confidence_scores
                        else 0
                    ),
                },
            )
            self.metrics["websocket_notifications_sent"] += 1
        except Exception as e:
            logger.error(f"Failed to send WebSocket detection: {e}")
            self.metrics["failed_notifications"] += 1

        # Send to N8N via integration manager
        if self.integration_manager and self.integration_manager.n8n_client:
            try:
                await self.integration_manager.n8n_client.send_mug_detection(
                    image_id=image_id,
                    mug_count=mug_count,
                    positions=positions,
                    confidence_scores=confidence_scores,
                    processing_time=processing_time,
                )
                self.metrics["external_notifications_sent"] += 1
            except Exception as e:
                logger.error(f"Failed to send N8N detection event: {e}")
                self.metrics["failed_notifications"] += 1

        # Update metrics
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics["notifications_sent"] += 1
        self._update_average_response_time(response_time)

    async def send_rule_triggered(
        self,
        rule_id: str,
        rule_name: str,
        triggered_by: Dict[str, Any],
        action_taken: str,
    ):
        """
        Send rule triggered event.

        Args:
            rule_id: ID of triggered rule
            rule_name: Name of the rule
            triggered_by: Data that triggered the rule
            action_taken: Action taken as result
        """
        start_time = datetime.now()

        # Send to WebSocket clients
        try:
            await self.ws_notifier.notify_activity(
                activity_type="rule_triggered",
                description=f"Rule '{rule_name}' triggered",
                metadata={
                    "rule_id": rule_id,
                    "rule_name": rule_name,
                    "triggered_by": triggered_by,
                    "action_taken": action_taken,
                },
            )
            self.metrics["websocket_notifications_sent"] += 1
        except Exception as e:
            logger.error(f"Failed to send WebSocket rule event: {e}")
            self.metrics["failed_notifications"] += 1

        # Send to N8N
        if self.integration_manager and self.integration_manager.n8n_client:
            try:
                await self.integration_manager.n8n_client.send_event(
                    N8NEventType.RULE_TRIGGERED,
                    {
                        "rule_id": rule_id,
                        "rule_name": rule_name,
                        "triggered_by": triggered_by,
                        "action_taken": action_taken,
                        "triggered_at": datetime.utcnow().isoformat(),
                    },
                )
                self.metrics["external_notifications_sent"] += 1
            except Exception as e:
                logger.error(f"Failed to send N8N rule event: {e}")
                self.metrics["failed_notifications"] += 1

        # Update metrics
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics["notifications_sent"] += 1
        self._update_average_response_time(response_time)

    async def send_system_metrics(self, metrics: Dict[str, Any]):
        """
        Send system metrics update.

        Args:
            metrics: System metrics data
        """
        # Send to WebSocket clients (this is the primary channel for metrics)
        try:
            await self.ws_notifier.notify_metrics(metrics)
            self.metrics["websocket_notifications_sent"] += 1
        except Exception as e:
            logger.error(f"Failed to send WebSocket metrics: {e}")
            self.metrics["failed_notifications"] += 1

        # Check for alert conditions in metrics
        await self._check_metric_alerts(metrics)

    async def _check_metric_alerts(self, metrics: Dict[str, Any]):
        """Check metrics for alert conditions."""
        # GPU usage alert
        gpu_usage = metrics.get("gpu_usage_percent", 0)
        if gpu_usage > 90:
            await self.send_performance_alert(
                metric_name="GPU Usage",
                current_value=gpu_usage,
                threshold=90,
                trend=metrics.get("gpu_trend", "unknown"),
            )

        # Memory usage alert
        memory_usage = metrics.get("memory_usage_percent", 0)
        if memory_usage > 85:
            await self.send_performance_alert(
                metric_name="Memory Usage",
                current_value=memory_usage,
                threshold=85,
                trend=metrics.get("memory_trend", "unknown"),
            )

        # API response time alert
        api_response_time = metrics.get("api_p95_response_time_ms", 0)
        if api_response_time > 500:
            await self.send_performance_alert(
                metric_name="API Response Time (P95)",
                current_value=api_response_time,
                threshold=500,
                trend=metrics.get("response_time_trend", "unknown"),
            )

        # Cache hit rate alert
        cache_hit_rate = metrics.get("redis_cache_hit_rate", 1.0)
        if cache_hit_rate < 0.8:
            await self.send_performance_alert(
                metric_name="Cache Hit Rate",
                current_value=cache_hit_rate * 100,
                threshold=80,
                trend=metrics.get("cache_trend", "unknown"),
            )

    def _map_severity_to_ws_priority(
        self, severity: AlertSeverity
    ) -> NotificationPriority:
        """Map alert severity to WebSocket priority."""
        mapping = {
            AlertSeverity.INFO: NotificationPriority.LOW,
            AlertSeverity.WARNING: NotificationPriority.MEDIUM,
            AlertSeverity.ERROR: NotificationPriority.HIGH,
            AlertSeverity.CRITICAL: NotificationPriority.CRITICAL,
        }
        return mapping.get(severity, NotificationPriority.MEDIUM)

    def _update_average_response_time(self, response_time: float):
        """Update average response time metric."""
        current_avg = self.metrics["average_response_time_ms"]
        count = self.metrics["notifications_sent"]

        # Calculate new average
        if count == 1:
            self.metrics["average_response_time_ms"] = response_time
        else:
            self.metrics["average_response_time_ms"] = (
                current_avg * (count - 1) + response_time
            ) / count

    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status including all integrations.

        Returns:
            Complete system status
        """
        status = {
            "notifier": {"is_running": self.is_running, "metrics": self.metrics.copy()},
            "websocket": {
                "is_running": self.ws_notifier.is_running,
                "stats": self.ws_notifier.get_statistics(),
            },
        }

        # Add integration status
        if self.integration_manager:
            try:
                health = await self.integration_manager.check_integration_health()
                metrics = await self.integration_manager.get_integration_metrics()
                status["integrations"] = {"health": health, "metrics": metrics}
            except Exception as e:
                logger.error(f"Failed to get integration status: {e}")
                status["integrations"] = {"error": str(e)}
        else:
            status["integrations"] = {"status": "not_initialized"}

        return status

    async def test_all_channels(self) -> Dict[str, Any]:
        """
        Test all notification channels.

        Returns:
            Test results for each channel
        """
        test_results = {}

        # Test WebSocket
        try:
            await self.ws_notifier.notify_alert(
                alert_type="test",
                severity="info",
                message="WebSocket test notification",
                metadata={"test": True},
            )
            test_results["websocket"] = {"success": True}
        except Exception as e:
            test_results["websocket"] = {"success": False, "error": str(e)}

        # Test external integrations
        if self.integration_manager:
            try:
                ext_results = await self.integration_manager.send_system_alert(
                    title="Integration Test",
                    message="Testing all external notification channels",
                    severity="info",
                    data={"test": True, "timestamp": datetime.utcnow().isoformat()},
                )
                test_results["external"] = {
                    service: {"success": result.success, "error": result.error}
                    for service, result in ext_results.items()
                }
            except Exception as e:
                test_results["external"] = {"error": str(e)}
        else:
            test_results["external"] = {"status": "not_available"}

        return test_results


# Usage example
async def main():
    """Example usage of EnhancedNotifier."""
    notifier = EnhancedNotifier()

    try:
        # Start the enhanced notifier
        await notifier.start()

        # Example 1: Send system alert
        await notifier.send_system_alert(
            title="System Test",
            message="Testing enhanced notification system",
            severity=AlertSeverity.INFO,
            data={"test_mode": True},
        )

        # Example 2: Send positioning complete event
        positioning_data = {
            "mug_position": {"x": 150, "y": 200, "width": 120, "height": 180},
            "confidence": 0.92,
            "processing_time": 0.245,
            "rules_applied": ["minimum_spacing", "edge_detection"],
            "recommendations": ["Position is optimal"],
        }

        await notifier.send_positioning_complete(
            image_id="test_img_456",
            positioning_data=positioning_data,
            render_template="overlay-template",
        )

        # Example 3: Send performance alert
        await notifier.send_performance_alert(
            metric_name="GPU Usage",
            current_value=92.5,
            threshold=90.0,
            trend="increasing",
        )

        # Example 4: Test all channels
        test_results = await notifier.test_all_channels()
        logger.info(f"Channel test results: {test_results}")

        # Example 5: Get system status
        status = await notifier.get_system_status()
        logger.info(f"System status: {status}")

        # Wait a bit to see notifications
        await asyncio.sleep(2)

    finally:
        # Always stop the notifier
        await notifier.stop()


if __name__ == "__main__":
    asyncio.run(main())
