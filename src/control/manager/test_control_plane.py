"""Test script for control plane functionality."""

# Standard library imports
import asyncio
import logging
from datetime import datetime

# First-party imports
from src.control.api.models.servers import ServerConfig, ServerType
from src.control.manager.orchestrator import ControlPlaneOrchestrator
from src.utils.logging_config import setup_logging

# Setup logging
logger = setup_logging(__name__)


async def test_control_plane():
    """Test control plane components."""
    logger.info("Starting control plane test")

    # Create orchestrator (without real RunPod API key for testing)
    orchestrator = ControlPlaneOrchestrator(
        runpod_api_key="test-api-key",  # Use mock key for testing
        idle_timeout_seconds=60,  # 1 minute for testing
        enable_auto_shutdown=True,
    )

    try:
        # Start orchestrator
        logger.info("Starting orchestrator...")
        await orchestrator.start()

        # Test 1: Resource monitoring
        logger.info("\n=== Testing Resource Monitor ===")
        gpu_info = await orchestrator.resource_monitor.get_gpu_info()
        system_info = await orchestrator.resource_monitor.get_system_resources()

        logger.info(f"GPU count: {len(gpu_info)}")
        logger.info(f"CPU usage: {system_info.cpu_percent}%")
        logger.info(f"Memory usage: {system_info.memory_percent}%")

        # Test 2: Metrics collection
        logger.info("\n=== Testing Metrics Collector ===")

        # Record some test metrics
        orchestrator.metrics_collector.record_request(
            server_id="test-server",
            latency_ms=150,
            success=True,
        )

        orchestrator.metrics_collector.record_model_metrics(
            load_time_ms=1200,
            inference_time_ms=95,
            cache_hit=True,
            batch_size=1,
        )

        # Get dashboard metrics
        dashboard_metrics = orchestrator.metrics_collector.get_dashboard_metrics()
        logger.info(
            f"Total requests: {dashboard_metrics['requests']['total_requests']}"
        )

        # Test 3: Status tracking
        logger.info("\n=== Testing Status Tracker ===")

        # Track a test deployment
        deployment = orchestrator.status_tracker.track_deployment(
            deployment_id="test-deploy-1",
            server_id="test-server",
            server_type=ServerType.SERVERLESS,
            server_state="running",
        )

        logger.info(f"Deployment tracked: {deployment.deployment_id}")

        # Test 4: WebSocket notifications
        logger.info("\n=== Testing WebSocket Notifier ===")

        # Send test notifications
        await orchestrator.notifier.notify_activity(
            activity_type="test",
            description="Control plane test activity",
            user_id="test-user",
        )

        await orchestrator.notifier.notify_alert(
            alert_type="test_alert",
            severity="low",
            message="This is a test alert",
        )

        stats = orchestrator.notifier.get_statistics()
        logger.info(f"Notifications sent: {stats['total_sent']}")

        # Test 5: Auto-shutdown scheduler
        logger.info("\n=== Testing Auto-Shutdown Scheduler ===")

        # Record activity
        orchestrator.auto_shutdown.record_activity("test-server")

        # Check statistics
        shutdown_stats = orchestrator.auto_shutdown.get_statistics()
        logger.info(f"Auto-shutdown enabled: {shutdown_stats['is_running']}")
        logger.info(f"Servers tracking: {shutdown_stats['servers_tracking']}")

        # Test 6: Dashboard data
        logger.info("\n=== Testing Dashboard Data ===")

        dashboard_data = await orchestrator.get_dashboard_data()
        logger.info(f"Dashboard timestamp: {dashboard_data['timestamp']}")
        logger.info(f"Total servers: {len(dashboard_data['servers'])}")
        logger.info(f"Resource summary: {dashboard_data['resources']['system']}")

        # Wait a bit to see background tasks working
        logger.info("\n=== Waiting 10 seconds to observe background tasks ===")
        await asyncio.sleep(10)

        logger.info("\n=== Test completed successfully ===")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

    finally:
        # Stop orchestrator
        logger.info("Stopping orchestrator...")
        await orchestrator.stop()


if __name__ == "__main__":
    # Run test
    asyncio.run(test_control_plane())
