"""Control plane orchestrator that coordinates all manager components."""

# Standard library imports
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# First-party imports
from src.control.api.models.servers import (
    ServerConfig,
    ServerInfo,
    ServerState,
    ServerType,
)
from src.control.manager.auto_shutdown import AutoShutdownScheduler
from src.control.manager.metrics import MetricsCollector
from src.control.manager.notifier import WebSocketNotifier
from src.control.manager.resource_monitor import ResourceMonitor
from src.control.manager.server_manager import ServerManager
from src.control.manager.status_tracker import StatusTracker
from src.database.mongodb import get_database
from src.database.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class ControlPlaneOrchestrator:
    """Orchestrates all control plane components."""

    def __init__(
        self,
        runpod_api_key: Optional[str] = None,
        idle_timeout_seconds: int = 600,
        enable_auto_shutdown: bool = True,
    ):
        """Initialize control plane orchestrator."""
        # Core components
        self.server_manager = ServerManager(runpod_api_key)
        self.resource_monitor = ResourceMonitor(poll_interval=5)
        self.metrics_collector = MetricsCollector(retention_hours=24)
        self.status_tracker = StatusTracker(health_check_interval=30)
        self.notifier = WebSocketNotifier(batch_interval=0.1)

        # Auto-shutdown scheduler
        self.auto_shutdown = AutoShutdownScheduler(
            server_manager=self.server_manager,
            idle_timeout_seconds=idle_timeout_seconds,
            check_interval_seconds=30,
        )
        self.enable_auto_shutdown = enable_auto_shutdown

        # Database clients
        self.mongodb = None
        self.redis = None

        # State
        self.is_running = False
        self._background_tasks: List[asyncio.Task] = []

        # Wire up callbacks
        self._setup_callbacks()

    def _setup_callbacks(self):
        """Set up component callbacks."""

        # Resource monitor -> Metrics collector
        async def resource_callback(gpus, system):
            self.metrics_collector.record_gpu_metrics(gpus)
            self.metrics_collector.record_system_metrics(system)

            # Check for alerts
            alerts = await self.resource_monitor.check_resource_alerts()
            for alert in alerts:
                await self.notifier.notify_alert(
                    alert_type=alert["type"],
                    severity=alert["severity"],
                    message=alert["message"],
                    metadata=alert,
                )

        self.resource_monitor.add_callback(resource_callback)

        # Auto-shutdown -> Notifier
        async def shutdown_callback(event):
            await self.notifier.notify_activity(
                activity_type=event["event"],
                description=event.get(
                    "message", f"Auto-shutdown event: {event['event']}"
                ),
                metadata=event,
            )

            # Update status tracker
            if "server_id" in event:
                if event["event"] == "auto_shutdown":
                    self.status_tracker.update_server_state(
                        event["server_id"],
                        ServerState.STOPPED,
                    )

        self.auto_shutdown.add_shutdown_callback(shutdown_callback)

        # Status tracker -> Notifier
        async def status_callback(event):
            if event["event"] == "alert":
                await self.notifier.notify_alert(
                    alert_type=event["alert"]["type"],
                    severity=event["alert"].get("severity", "warning"),
                    message=event["alert"]["message"],
                    metadata=event,
                )
            elif event["event"] in ["status_change", "health_change"]:
                await self.notifier.notify_activity(
                    activity_type=event["event"],
                    description=f"Server {event['server_id']} {event['event']}",
                    metadata=event,
                )

        self.status_tracker.add_callback(status_callback)

    async def start(self):
        """Start all control plane components."""
        if self.is_running:
            return

        logger.info("Starting control plane orchestrator")

        try:
            # Initialize database connections
            self.mongodb = await get_database()
            self.redis = await get_redis_client()

            # Initialize and start components
            await self.server_manager.initialize()
            await self.resource_monitor.start()
            await self.metrics_collector.start()
            await self.status_tracker.start()
            await self.notifier.start()

            if self.enable_auto_shutdown:
                await self.auto_shutdown.start()

            # Start background tasks
            self._background_tasks = [
                asyncio.create_task(self._metrics_broadcast_loop()),
                asyncio.create_task(self._status_update_loop()),
                asyncio.create_task(self._cost_tracking_loop()),
            ]

            self.is_running = True
            logger.info("Control plane orchestrator started successfully")

        except Exception as e:
            logger.error(f"Failed to start orchestrator: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop all control plane components."""
        if not self.is_running:
            return

        logger.info("Stopping control plane orchestrator")

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Stop components
        if self.enable_auto_shutdown:
            await self.auto_shutdown.stop()

        await self.notifier.stop()
        await self.status_tracker.stop()
        await self.metrics_collector.stop()
        await self.resource_monitor.stop()
        await self.server_manager.shutdown()

        # Close database connections
        if self.redis:
            await self.redis.close()

        self.is_running = False
        logger.info("Control plane orchestrator stopped")

    async def deploy_server(
        self,
        server_type: ServerType,
        config: ServerConfig,
        auto_start: bool = True,
        tags: Optional[Dict[str, str]] = None,
    ) -> ServerInfo:
        """Deploy a new server with full tracking."""
        logger.info(f"Deploying {server_type} server")

        try:
            # Deploy server
            server = await self.server_manager.deploy_server(
                server_type=server_type,
                config=config,
                tags=tags,
            )

            # Track deployment
            deployment_id = f"deploy-{datetime.utcnow().timestamp()}"
            self.status_tracker.track_deployment(
                deployment_id=deployment_id,
                server_id=server.id,
                server_type=server_type,
                server_state=server.state,
            )

            # Start if requested
            if auto_start and server.state == ServerState.STOPPED:
                server = await self.start_server(server.id)

            # Log deployment
            await self.notifier.notify_activity(
                activity_type="server_deployed",
                description=f"Deployed {server_type} server {server.id}",
                metadata={
                    "server_id": server.id,
                    "server_type": server_type.value,
                    "config": config.model_dump(),
                },
            )

            # Store in MongoDB
            if self.mongodb:
                await self.mongodb.deployments.insert_one(
                    {
                        "deployment_id": deployment_id,
                        "server_id": server.id,
                        "server_type": server_type.value,
                        "config": config.model_dump(),
                        "created_at": datetime.utcnow(),
                        "tags": tags or {},
                    }
                )

            return server

        except Exception as e:
            logger.error(f"Failed to deploy server: {e}")
            await self.notifier.notify_alert(
                alert_type="deployment_failed",
                severity="critical",
                message=f"Failed to deploy {server_type} server: {str(e)}",
            )
            raise

    async def start_server(self, server_id: str) -> ServerInfo:
        """Start a server with tracking."""
        logger.info(f"Starting server {server_id}")

        try:
            # Start server
            server = await self.server_manager.start_server(server_id)

            # Update status
            self.status_tracker.update_server_state(server_id, ServerState.RUNNING)

            # Record activity
            self.auto_shutdown.record_activity(server_id)

            # Notify
            await self.notifier.notify_server_event(
                server_id=server_id,
                event_type="start",
                details={
                    "state": server.state.value,
                    "endpoint_url": server.endpoint_url,
                },
            )

            return server

        except Exception as e:
            logger.error(f"Failed to start server {server_id}: {e}")
            self.status_tracker.update_server_state(server_id, ServerState.ERROR)
            raise

    async def stop_server(self, server_id: str, force: bool = False) -> ServerInfo:
        """Stop a server with tracking."""
        logger.info(f"Stopping server {server_id} (force={force})")

        try:
            # Stop server
            server = await self.server_manager.stop_server(server_id, force)

            # Update status
            self.status_tracker.update_server_state(server_id, ServerState.STOPPED)

            # Notify
            await self.notifier.notify_server_event(
                server_id=server_id,
                event_type="stop",
                details={
                    "state": server.state.value,
                    "forced": force,
                },
            )

            return server

        except Exception as e:
            logger.error(f"Failed to stop server {server_id}: {e}")
            if not force:
                raise
            return await self.server_manager.get_server_status(server_id)

    async def restart_server(self, server_id: str) -> ServerInfo:
        """Restart a server."""
        logger.info(f"Restarting server {server_id}")

        server = await self.stop_server(server_id)
        await asyncio.sleep(2)  # Brief pause
        return await self.start_server(server_id)

    async def scale_server(
        self,
        server_id: str,
        min_instances: Optional[int] = None,
        max_instances: Optional[int] = None,
    ) -> ServerInfo:
        """Scale a serverless endpoint."""
        logger.info(
            f"Scaling server {server_id}: min={min_instances}, max={max_instances}"
        )

        try:
            # Scale server
            server = await self.server_manager.scale_server(
                server_id=server_id,
                min_instances=min_instances,
                max_instances=max_instances,
            )

            # Notify
            await self.notifier.notify_server_event(
                server_id=server_id,
                event_type="scale",
                details={
                    "min_instances": min_instances,
                    "max_instances": max_instances,
                },
            )

            return server

        except Exception as e:
            logger.error(f"Failed to scale server {server_id}: {e}")
            raise

    def record_request(
        self,
        server_id: str,
        latency_ms: float,
        success: bool = True,
        endpoint: Optional[str] = None,
    ):
        """Record a request for tracking."""
        # Update metrics
        self.metrics_collector.record_request(
            server_id=server_id,
            latency_ms=latency_ms,
            success=success,
            metadata={"endpoint": endpoint} if endpoint else None,
        )

        # Update status tracker
        self.status_tracker.record_request(server_id, success)

        # Update auto-shutdown
        self.auto_shutdown.record_activity(server_id)

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        # Get all data sources
        servers = await self.server_manager.list_servers()
        metrics = self.metrics_collector.get_dashboard_metrics()
        deployments = self.status_tracker.get_all_deployments()
        auto_shutdown_stats = self.auto_shutdown.get_statistics()
        notification_stats = self.notifier.get_statistics()

        # Get current resource usage
        gpu_info = await self.resource_monitor.get_gpu_info()
        system_info = await self.resource_monitor.get_system_resources()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "servers": [
                {
                    "id": server.id,
                    "type": server.type.value,
                    "state": server.state.value,
                    "endpoint_url": server.endpoint_url,
                    "cost_per_hour": server.cost_per_hour,
                    "total_cost": server.total_cost,
                    "created_at": (
                        server.created_at.isoformat() if server.created_at else None
                    ),
                }
                for server in servers
            ],
            "deployments": [
                {
                    "deployment_id": d.deployment_id,
                    "server_id": d.server_id,
                    "health": d.health.value,
                    "uptime_seconds": d.uptime_seconds,
                    "total_requests": d.total_requests,
                    "error_count": d.error_count,
                    "active_alerts": len(d.active_alerts),
                }
                for d in deployments
            ],
            "resources": {
                "gpu": self.resource_monitor.get_gpu_summary(gpu_info),
                "system": {
                    "cpu_percent": system_info.cpu_percent,
                    "memory_percent": system_info.memory_percent,
                    "disk_percent": system_info.disk_percent,
                },
            },
            "metrics": metrics,
            "auto_shutdown": auto_shutdown_stats,
            "notifications": notification_stats,
            "status_summary": self.status_tracker.get_summary(),
        }

    async def _metrics_broadcast_loop(self):
        """Periodically broadcast metrics."""
        while self.is_running:
            try:
                # Get current metrics
                servers = await self.server_manager.list_servers()

                for server in servers:
                    if server.state == ServerState.RUNNING:
                        metrics = self.metrics_collector.get_server_metrics(server.id)

                        await self.notifier.notify_metrics(
                            {
                                "server_id": server.id,
                                "metrics": metrics.model_dump(),
                            }
                        )

                await asyncio.sleep(5)  # Broadcast every 5 seconds

            except Exception as e:
                logger.error(f"Metrics broadcast error: {e}")
                await asyncio.sleep(5)

    async def _status_update_loop(self):
        """Periodically update and broadcast status."""
        while self.is_running:
            try:
                # Update server statuses
                servers = await self.server_manager.list_servers()

                for server in servers:
                    # Get latest status from RunPod
                    updated_server = await self.server_manager.get_server_status(
                        server.id
                    )

                    # Update tracker if state changed
                    if updated_server.state != server.state:
                        self.status_tracker.update_server_state(
                            server.id,
                            updated_server.state,
                        )

                # Broadcast system status
                status_summary = self.status_tracker.get_summary()
                await self.notifier.send_system_status(status_summary)

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Status update error: {e}")
                await asyncio.sleep(30)

    async def _cost_tracking_loop(self):
        """Track and update costs."""
        while self.is_running:
            try:
                # Update costs for running servers
                servers = await self.server_manager.list_servers()

                for server in servers:
                    if server.state == ServerState.RUNNING and server.started_at:
                        # Calculate current session cost
                        runtime = datetime.utcnow() - server.started_at
                        hours = runtime.total_seconds() / 3600
                        session_cost = hours * server.cost_per_hour

                        # Record cost
                        self.metrics_collector.record_cost(server.id, session_cost)

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Cost tracking error: {e}")
                await asyncio.sleep(60)
