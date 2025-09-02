"""Example usage of integrations in the H200 system context."""

# Standard library imports
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

# Local imports
from ..utils.logging_config import get_logger
from .enhanced_notifier import AlertSeverity, EnhancedNotifier
from .manager import IntegrationManager
from .n8n import N8NClient, N8NEventType
from .notifications import NotificationClient, NotificationType
from .templated import TemplatedClient

logger = get_logger(__name__)


class H200IntegrationExample:
    """
    Example showing how to integrate external services into the H200 system.

    This class demonstrates:
    1. How to integrate with the positioning analysis pipeline
    2. How to send events to N8N workflows
    3. How to render designs based on positioning results
    4. How to send notifications across multiple channels
    5. How to handle errors and monitoring
    """

    def __init__(self):
        """Initialize integration example."""
        self.notifier = EnhancedNotifier()
        self.is_running = False

    async def start(self):
        """Start the integration example."""
        if self.is_running:
            return

        logger.info("Starting H200 integration example")
        await self.notifier.start()
        self.is_running = True

    async def stop(self):
        """Stop the integration example."""
        if not self.is_running:
            return

        logger.info("Stopping H200 integration example")
        await self.notifier.stop()
        self.is_running = False

    async def simulate_image_analysis_workflow(self):
        """Simulate a complete image analysis workflow with integrations."""
        logger.info("=== Simulating Image Analysis Workflow ===")

        # Step 1: Simulate mug detection
        image_id = f"sim_img_{int(datetime.now().timestamp())}"

        await self.notifier.send_mug_detection(
            image_id=image_id,
            mug_count=2,
            positions=[
                {"x": 100, "y": 150, "width": 80, "height": 120},
                {"x": 300, "y": 180, "width": 85, "height": 125},
            ],
            confidence_scores=[0.95, 0.88],
            processing_time=0.156,
        )

        # Step 2: Simulate positioning analysis
        positioning_data = {
            "mug_position": {"x": 100, "y": 150, "width": 80, "height": 120},
            "confidence": 0.95,
            "processing_time": 0.089,
            "rules_applied": ["minimum_spacing", "edge_detection", "symmetry_check"],
            "recommendations": ["Position is optimal", "Good spacing between mugs"],
            "quality_score": 0.92,
            "bounding_box": {"x1": 60, "y1": 90, "x2": 180, "y2": 270},
            "template_params": {
                "overlay_color": "#00FF00",
                "show_confidence": True,
                "highlight_optimal": True,
            },
        }

        await self.notifier.send_positioning_complete(
            image_id=image_id,
            positioning_data=positioning_data,
            render_template="mug-positioning-overlay",
        )

        # Step 3: Simulate rule trigger
        await self.notifier.send_rule_triggered(
            rule_id="symmetry_check_001",
            rule_name="Symmetry Check",
            triggered_by={
                "image_id": image_id,
                "symmetry_score": 0.91,
                "threshold": 0.85,
            },
            action_taken="Applied symmetry correction overlay",
        )

        logger.info("✓ Image analysis workflow simulation complete")

    async def simulate_batch_processing(self):
        """Simulate batch processing workflow."""
        logger.info("=== Simulating Batch Processing ===")

        batch_id = f"batch_{int(datetime.now().timestamp())}"

        # Simulate processing multiple images
        for i in range(5):
            image_id = f"batch_img_{i+1}"

            # Simulate detection for each image
            await self.notifier.send_mug_detection(
                image_id=image_id,
                mug_count=1 + (i % 3),  # Vary mug count
                positions=[{"x": 100 + i * 50, "y": 200, "width": 80, "height": 120}],
                confidence_scores=[0.9 - i * 0.05],  # Decreasing confidence
                processing_time=0.1 + i * 0.02,
            )

            # Small delay between detections
            await asyncio.sleep(0.1)

        # Send batch completion
        batch_stats = {
            "batch_id": batch_id,
            "total_images": 5,
            "successful_detections": 5,
            "failed_detections": 0,
            "success_rate": 1.0,
            "average_confidence": 0.82,
            "total_processing_time": 0.6,
            "mugs_detected": 10,
        }

        await self.notifier.send_batch_complete(
            batch_id=batch_id, batch_stats=batch_stats, create_summary_report=True
        )

        logger.info("✓ Batch processing simulation complete")

    async def simulate_system_monitoring(self):
        """Simulate system monitoring with alerts."""
        logger.info("=== Simulating System Monitoring ===")

        # Send normal metrics
        normal_metrics = {
            "gpu_usage_percent": 65.5,
            "memory_usage_percent": 42.3,
            "api_p95_response_time_ms": 145,
            "redis_cache_hit_rate": 0.94,
            "active_connections": 12,
            "requests_per_second": 8.5,
        }

        await self.notifier.send_system_metrics(normal_metrics)
        await asyncio.sleep(1)

        # Simulate high GPU usage alert
        high_gpu_metrics = {
            "gpu_usage_percent": 93.2,
            "memory_usage_percent": 45.1,
            "api_p95_response_time_ms": 178,
            "redis_cache_hit_rate": 0.91,
            "gpu_trend": "increasing",
        }

        await self.notifier.send_system_metrics(high_gpu_metrics)
        await asyncio.sleep(1)

        # Simulate critical memory usage
        critical_metrics = {
            "gpu_usage_percent": 95.8,
            "memory_usage_percent": 88.7,
            "api_p95_response_time_ms": 650,
            "redis_cache_hit_rate": 0.76,
            "memory_trend": "rapidly_increasing",
            "response_time_trend": "increasing",
        }

        await self.notifier.send_system_metrics(critical_metrics)

        logger.info("✓ System monitoring simulation complete")

    async def simulate_model_update(self):
        """Simulate model update workflow."""
        logger.info("=== Simulating Model Update ===")

        # Simulate YOLO model update
        await self.notifier.send_model_update(
            model_name="YOLO",
            version="8.2.0",
            performance_metrics={
                "accuracy": 0.96,
                "precision": 0.94,
                "recall": 0.92,
                "inference_time_ms": 42,
                "model_size_mb": 248,
                "improvement_over_previous": "3.2%",
            },
        )

        # Simulate CLIP model update
        await self.notifier.send_model_update(
            model_name="CLIP",
            version="1.5.1",
            performance_metrics={
                "embedding_quality": 0.89,
                "inference_time_ms": 28,
                "model_size_mb": 512,
                "improvement_over_previous": "1.8%",
            },
        )

        logger.info("✓ Model update simulation complete")

    async def simulate_error_scenarios(self):
        """Simulate various error scenarios."""
        logger.info("=== Simulating Error Scenarios ===")

        # Simulate GPU memory error
        await self.notifier.send_error_event(
            error_type="GPUMemoryError",
            error_message="GPU out of memory during model inference",
            context={
                "available_memory": "512MB",
                "required_memory": "2GB",
                "model": "YOLO",
                "batch_size": 32,
                "image_resolution": "1920x1080",
            },
            severity=AlertSeverity.ERROR,
        )

        # Simulate critical database connection failure
        await self.notifier.send_error_event(
            error_type="DatabaseConnectionError",
            error_message="Failed to connect to MongoDB Atlas after 3 retries",
            context={
                "connection_string": "mongodb+srv://***",
                "last_error": "connection timeout",
                "retry_count": 3,
                "service_affected": "positioning_rules",
            },
            severity=AlertSeverity.CRITICAL,
        )

        # Simulate warning for high API latency
        await self.notifier.send_error_event(
            error_type="PerformanceWarning",
            error_message="API response time consistently above threshold",
            context={
                "current_p95": "847ms",
                "threshold": "500ms",
                "duration": "5 minutes",
                "affected_endpoints": ["/api/v1/analyze", "/api/v1/dashboard"],
            },
            severity=AlertSeverity.WARNING,
        )

        logger.info("✓ Error scenario simulation complete")

    async def test_individual_integrations(self):
        """Test each integration individually."""
        logger.info("=== Testing Individual Integrations ===")

        async with IntegrationManager() as manager:
            # Test integration health
            health = await manager.check_integration_health()
            logger.info(f"Integration health: {health}")

            # Test Templated.io if available
            if health.get("templated", {}).get("healthy"):
                logger.info("Testing Templated.io integration...")
                try:
                    if manager.templated_client:
                        templates = await manager.templated_client.list_templates(
                            category="mug-positioning"
                        )
                        logger.info(f"✓ Found {len(templates)} templates")

                        # Test render with sample data
                        render_result = await manager.templated_client.render_design(
                            template_id="sample-template",
                            positioning_data={
                                "mug_position": {"x": 200, "y": 300},
                                "confidence": 0.88,
                                "template_params": {
                                    "title": "Integration Test",
                                    "color": "#FF6B35",
                                },
                            },
                        )
                        logger.info(
                            f"✓ Render created: {render_result.get('render_url', 'N/A')}"
                        )
                except Exception as e:
                    logger.error(f"✗ Templated.io test failed: {e}")

            # Test N8N if available
            if health.get("n8n", {}).get("healthy"):
                logger.info("Testing N8N integration...")
                try:
                    if manager.n8n_client:
                        await manager.n8n_client.send_event(
                            N8NEventType.SYSTEM_ALERT,
                            {
                                "test": True,
                                "message": "Integration test from H200 system",
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                        )
                        logger.info("✓ N8N event sent successfully")
                except Exception as e:
                    logger.error(f"✗ N8N test failed: {e}")

            # Test notifications if available
            if health.get("notifications", {}).get("healthy"):
                logger.info("Testing notification system...")
                try:
                    if manager.notification_client:
                        await manager.notification_client.send_notification(
                            title="Integration Test",
                            message="Testing notification system from H200",
                            notification_type=NotificationType.INFO,
                            data={
                                "test": True,
                                "system": "h200-mug-positioning",
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                        )
                        logger.info("✓ Notification sent successfully")
                except Exception as e:
                    logger.error(f"✗ Notification test failed: {e}")

        logger.info("✓ Individual integration testing complete")

    async def run_complete_demo(self):
        """Run a complete demonstration of all integrations."""
        logger.info("=" * 50)
        logger.info("H200 INTEGRATION SYSTEM DEMONSTRATION")
        logger.info("=" * 50)

        try:
            await self.start()

            # Run all simulation scenarios
            await self.simulate_image_analysis_workflow()
            await asyncio.sleep(1)

            await self.simulate_batch_processing()
            await asyncio.sleep(1)

            await self.simulate_system_monitoring()
            await asyncio.sleep(1)

            await self.simulate_model_update()
            await asyncio.sleep(1)

            await self.simulate_error_scenarios()
            await asyncio.sleep(1)

            await self.test_individual_integrations()

            # Get final system status
            status = await self.notifier.get_system_status()
            logger.info("=" * 30)
            logger.info("FINAL SYSTEM STATUS")
            logger.info("=" * 30)
            logger.info(
                f"Notifications sent: {status['notifier']['metrics']['notifications_sent']}"
            )
            logger.info(
                f"WebSocket notifications: {status['notifier']['metrics']['websocket_notifications_sent']}"
            )
            logger.info(
                f"External notifications: {status['notifier']['metrics']['external_notifications_sent']}"
            )
            logger.info(
                f"Failed notifications: {status['notifier']['metrics']['failed_notifications']}"
            )
            logger.info(
                f"Average response time: {status['notifier']['metrics']['average_response_time_ms']:.2f}ms"
            )

            if "integrations" in status:
                integration_health = status["integrations"].get("health", {})
                logger.info(f"Integration health: {integration_health}")

        finally:
            await self.stop()


# Practical usage examples for different components


async def integrate_with_api_endpoint():
    """Example of integrating with FastAPI endpoint."""
    # Third-party imports
    from fastapi import BackgroundTasks

    # This would be used in an actual API endpoint
    async def analyze_image_endpoint(
        image_data: bytes, background_tasks: BackgroundTasks, notifier: EnhancedNotifier
    ):
        """Example API endpoint with integration."""
        try:
            # Perform image analysis (simulated)
            image_id = "api_img_123"
            positioning_result = {
                "mug_position": {"x": 150, "y": 200},
                "confidence": 0.92,
                "processing_time": 0.234,
            }

            # Send completion event in background
            background_tasks.add_task(
                notifier.send_positioning_complete,
                image_id,
                positioning_result,
                "default-overlay",
            )

            return {"success": True, "image_id": image_id}

        except Exception as e:
            # Send error event
            background_tasks.add_task(
                notifier.send_error_event,
                "ImageAnalysisError",
                str(e),
                {"image_id": image_id},
                AlertSeverity.ERROR,
            )
            raise


async def integrate_with_control_plane():
    """Example of integrating with the control plane."""
    # Local imports
    from ..control.manager.orchestrator import ControlPlaneOrchestrator

    class IntegratedControlPlane(ControlPlaneOrchestrator):
        """Control plane with enhanced notifications."""

        def __init__(self):
            super().__init__()
            self.notifier = EnhancedNotifier()

        async def start(self):
            """Start control plane with notifications."""
            await super().start()
            await self.notifier.start()

        async def stop(self):
            """Stop control plane with notifications."""
            await self.notifier.stop()
            await super().stop()

        async def on_server_start(self, server_id: str, server_type: str):
            """Called when a server starts."""
            await super().on_server_start(server_id, server_type)

            # Send notification
            await self.notifier.send_system_alert(
                title=f"Server Started",
                message=f"{server_type} server {server_id} is now running",
                severity=AlertSeverity.INFO,
                data={
                    "server_id": server_id,
                    "server_type": server_type,
                    "action": "start",
                },
            )

        async def on_server_error(self, server_id: str, error: str):
            """Called when a server encounters an error."""
            await super().on_server_error(server_id, error)

            # Send error notification
            await self.notifier.send_error_event(
                error_type="ServerError",
                error_message=f"Server {server_id} encountered an error",
                context={"server_id": server_id, "error_details": error},
                severity=AlertSeverity.ERROR,
            )


async def integrate_with_rules_engine():
    """Example of integrating with the rules engine."""
    # Local imports
    from ..core.rules.engine import RuleEngine

    class IntegratedRuleEngine(RuleEngine):
        """Rules engine with enhanced notifications."""

        def __init__(self):
            super().__init__()
            self.notifier = EnhancedNotifier()

        async def execute_rule(self, rule_id: str, input_data: Dict[str, Any]):
            """Execute rule with notification."""
            try:
                # Execute the rule
                result = await super().execute_rule(rule_id, input_data)

                # Send rule triggered notification
                if result.get("triggered"):
                    await self.notifier.send_rule_triggered(
                        rule_id=rule_id,
                        rule_name=result.get("rule_name", rule_id),
                        triggered_by=input_data,
                        action_taken=result.get("action", "Rule executed"),
                    )

                return result

            except Exception as e:
                # Send error notification
                await self.notifier.send_error_event(
                    error_type="RuleExecutionError",
                    error_message=f"Failed to execute rule {rule_id}",
                    context={
                        "rule_id": rule_id,
                        "input_data": input_data,
                        "error": str(e),
                    },
                    severity=AlertSeverity.ERROR,
                )
                raise


# Main demonstration
async def main():
    """Run the complete integration demonstration."""
    demo = H200IntegrationExample()
    await demo.run_complete_demo()


if __name__ == "__main__":
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run demo
    asyncio.run(main())
