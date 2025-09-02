"""Server lifecycle management for RunPod GPU instances."""

# Standard library imports
import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

# First-party imports
from src.control.api.models.servers import (
    ServerAction,
    ServerConfig,
    ServerInfo,
    ServerState,
    ServerType,
)
from src.deployment import (
    DeploymentConfig,
    DeploymentManager,
    DeploymentMode,
    DeploymentOrchestrator,
    DeploymentStatus,
    DeploymentStrategy,
    GPUConfig,
    GPUType,
    ResourceConfig,
    ScalingConfig,
)
from src.utils.secrets import get_secret

logger = logging.getLogger(__name__)


class ServerManager:
    """Manages server lifecycle on RunPod using the deployment module."""

    def __init__(self, runpod_api_key: Optional[str] = None):
        """Initialize server manager."""
        self.api_key = runpod_api_key or get_secret("RUNPOD_API_KEY")
        self.deployment_manager = DeploymentManager(self.api_key)
        self.orchestrator = DeploymentOrchestrator(self.deployment_manager)
        self.servers: Dict[str, ServerInfo] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize the deployment manager."""
        if not self._initialized:
            await self.deployment_manager.initialize()
            self._initialized = True

            # Load existing deployments
            await self._sync_deployments()

    async def shutdown(self):
        """Shutdown the server manager."""
        if self._initialized:
            await self.deployment_manager.shutdown()
            self._initialized = False

    async def _sync_deployments(self):
        """Sync local server info with deployment states."""
        try:
            # Get all deployments
            deployments = await self.deployment_manager.list_deployments()

            # Update server info
            for state in deployments:
                if state.status != DeploymentStatus.TERMINATED:
                    server_info = await self._deployment_state_to_server_info(state)
                    self.servers[server_info.id] = server_info

        except Exception as e:
            logger.error(f"Failed to sync deployments: {e}")

    async def _deployment_state_to_server_info(self, state) -> ServerInfo:
        """Convert deployment state to server info."""
        # Map deployment mode to server type
        server_type = (
            ServerType.SERVERLESS
            if state.mode == DeploymentMode.SERVERLESS
            else ServerType.TIMED
        )

        # Map deployment status to server state
        state_map = {
            DeploymentStatus.PENDING: ServerState.STARTING,
            DeploymentStatus.PROVISIONING: ServerState.STARTING,
            DeploymentStatus.STARTING: ServerState.STARTING,
            DeploymentStatus.RUNNING: ServerState.RUNNING,
            DeploymentStatus.STOPPING: ServerState.STOPPING,
            DeploymentStatus.STOPPED: ServerState.STOPPED,
            DeploymentStatus.FAILED: ServerState.ERROR,
            DeploymentStatus.TERMINATED: ServerState.STOPPED,
        }

        # Get metrics if available
        metrics = state.metrics

        # Create server config from deployment config
        server_config = ServerConfig(
            docker_image=state.config.docker_image,
            gpu_count=state.config.gpu.count,
            cpu_cores=state.config.resources.cpu_cores,
            memory_gb=state.config.resources.memory_gb,
            min_instances=(
                state.config.scaling.min_workers if state.config.scaling else 0
            ),
            max_instances=(
                state.config.scaling.max_workers if state.config.scaling else 1
            ),
            environment_variables=state.config.env_vars,
        )

        # Create server info
        return ServerInfo(
            id=state.deployment_id,
            type=server_type,
            state=state_map.get(state.status, ServerState.ERROR),
            config=server_config,
            created_at=state.created_at,
            started_at=(
                state.created_at if state.status == DeploymentStatus.RUNNING else None
            ),
            endpoint_url=metrics.get("endpoint_url") if metrics else None,
            metrics_url=metrics.get("metrics_url") if metrics else None,
            current_instances=(
                metrics.get("workers", {}).get("running", 0) if metrics else 0
            ),
            total_requests=metrics.get("requests_total", 0) if metrics else 0,
            error_count=metrics.get("requests_failed", 0) if metrics else 0,
            cost_per_hour=3.50,  # H100 pricing
        )

    def _server_config_to_deployment_config(
        self,
        server_type: ServerType,
        config: ServerConfig,
        tags: Optional[Dict[str, str]] = None,
    ) -> DeploymentConfig:
        """Convert server config to deployment config."""
        # Map GPU type (defaulting to H100 for H200)
        gpu_type = GPUType.H100  # H200 not in enum yet, using H100

        # Create deployment config
        deployment_config = DeploymentConfig(
            name=f"h200-mug-positioning-{server_type.value}",
            mode=(
                DeploymentMode.SERVERLESS
                if server_type == ServerType.SERVERLESS
                else DeploymentMode.TIMED
            ),
            environment=os.getenv("DEPLOYMENT_ENV", "production"),
            docker_image=config.docker_image,
            docker_tag=config.docker_tag or "latest",
            gpu=GPUConfig(type=gpu_type, count=config.gpu_count),
            resources=ResourceConfig(
                cpu_cores=config.cpu_cores,
                memory_gb=config.memory_gb,
                container_disk_gb=config.container_disk_gb,
                volume_size_gb=config.volume_size_gb,
                network_volume_size_gb=100,  # For model storage
            ),
            env_vars=config.environment_variables,
            idle_timeout_minutes=config.idle_timeout_minutes,
            labels=tags or {},
        )

        # Add scaling config for serverless
        if server_type == ServerType.SERVERLESS:
            deployment_config.scaling = ScalingConfig(
                min_workers=config.min_instances,
                max_workers=config.max_instances,
                target_requests_per_second=20,
                max_requests_per_worker=100,
            )

        return deployment_config

    async def deploy_server(
        self,
        server_type: ServerType,
        config: ServerConfig,
        tags: Optional[Dict[str, str]] = None,
    ) -> ServerInfo:
        """Deploy a new server on RunPod."""
        logger.info(f"Deploying {server_type} server with config: {config}")

        # Ensure initialized
        await self.initialize()

        try:
            # Convert to deployment config
            deployment_config = self._server_config_to_deployment_config(
                server_type, config, tags
            )

            # Deploy using the deployment manager
            state = await self.deployment_manager.deploy(deployment_config)

            # Convert to server info
            server = await self._deployment_state_to_server_info(state)

            # Store server
            self.servers[server.id] = server

            logger.info(f"Server {server.id} deployed successfully")
            return server

        except Exception as e:
            logger.error(f"Failed to deploy server: {e}")
            raise

    async def start_server(self, server_id: str) -> ServerInfo:
        """Start a stopped server."""
        if server_id not in self.servers:
            raise ValueError(f"Server {server_id} not found")

        server = self.servers[server_id]

        if server.state != ServerState.STOPPED:
            raise ValueError(f"Cannot start server in {server.state} state")

        logger.info(f"Starting server {server_id}")

        try:
            # Start using deployment manager
            state = await self.deployment_manager.start_deployment(server_id)

            # Update server info
            server = await self._deployment_state_to_server_info(state)
            self.servers[server.id] = server

            return server

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise

    async def stop_server(self, server_id: str, force: bool = False) -> ServerInfo:
        """Stop a running server."""
        if server_id not in self.servers:
            raise ValueError(f"Server {server_id} not found")

        server = self.servers[server_id]

        if server.state not in [ServerState.RUNNING, ServerState.ERROR]:
            raise ValueError(f"Cannot stop server in {server.state} state")

        logger.info(f"Stopping server {server_id} (force={force})")

        try:
            # Stop using deployment manager
            state = await self.deployment_manager.stop_deployment(server_id)

            # Update server info
            server = await self._deployment_state_to_server_info(state)
            server.stopped_at = datetime.utcnow()

            # Calculate total runtime and cost
            if server.started_at and server.stopped_at:
                runtime = server.stopped_at - server.started_at
                hours = runtime.total_seconds() / 3600
                server.total_cost += hours * server.cost_per_hour

            self.servers[server.id] = server

            return server

        except Exception as e:
            if force:
                server.state = ServerState.ERROR
                return server
            logger.error(f"Failed to stop server: {e}")
            raise

    async def restart_server(self, server_id: str) -> ServerInfo:
        """Restart a server."""
        logger.info(f"Restarting server {server_id}")

        # Stop then start
        await self.stop_server(server_id)
        await asyncio.sleep(2)  # Brief pause
        return await self.start_server(server_id)

    async def scale_server(
        self,
        server_id: str,
        min_instances: Optional[int] = None,
        max_instances: Optional[int] = None,
    ) -> ServerInfo:
        """Scale a serverless endpoint."""
        if server_id not in self.servers:
            raise ValueError(f"Server {server_id} not found")

        server = self.servers[server_id]

        if server.type != ServerType.SERVERLESS:
            raise ValueError("Can only scale serverless endpoints")

        logger.info(
            f"Scaling server {server_id}: min={min_instances}, max={max_instances}"
        )

        try:
            # Scale using deployment manager
            state = await self.deployment_manager.scale_deployment(
                server_id,
                min_instances or server.config.min_instances,
                max_instances or server.config.max_instances,
            )

            # Update server info
            server = await self._deployment_state_to_server_info(state)
            self.servers[server.id] = server

            return server

        except Exception as e:
            logger.error(f"Failed to scale server: {e}")
            raise

    async def delete_server(self, server_id: str) -> bool:
        """Delete a server deployment."""
        if server_id not in self.servers:
            raise ValueError(f"Server {server_id} not found")

        server = self.servers[server_id]

        # Ensure server is stopped
        if server.state != ServerState.STOPPED:
            await self.stop_server(server_id)

        logger.info(f"Deleting server {server_id}")

        try:
            # Terminate using deployment manager
            await self.deployment_manager.terminate_deployment(server_id)

            # Remove from local cache
            del self.servers[server_id]
            return True

        except Exception as e:
            logger.error(f"Failed to delete server: {e}")
            raise

    async def get_server_status(self, server_id: str) -> ServerInfo:
        """Get current server status from RunPod."""
        if server_id not in self.servers:
            raise ValueError(f"Server {server_id} not found")

        try:
            # Get deployment status
            state = await self.deployment_manager.get_deployment(server_id)
            if not state:
                raise ValueError(f"No deployment state for server {server_id}")

            # Get latest metrics
            metrics = await self.deployment_manager.get_deployment_metrics(server_id)

            # Update server info
            server = await self._deployment_state_to_server_info(state)

            # Update with latest metrics
            if "current" in metrics:
                current = metrics["current"]
                if "workers" in current:
                    server.current_instances = current["workers"].get("running", 0)
                if "metrics" in current:
                    server.total_requests = current["metrics"].get("requests_total", 0)
                    server.error_count = current["metrics"].get("requests_failed", 0)

            self.servers[server.id] = server
            return server

        except Exception as e:
            logger.error(f"Failed to get server status: {e}")
            raise

    async def list_servers(
        self, server_type: Optional[ServerType] = None
    ) -> List[ServerInfo]:
        """List all managed servers."""
        # Sync with deployment manager
        await self._sync_deployments()

        servers = list(self.servers.values())

        if server_type:
            servers = [s for s in servers if s.type == server_type]

        return servers

    async def get_server_logs(self, server_id: str, lines: int = 100) -> str:
        """Get server logs."""
        if server_id not in self.servers:
            raise ValueError(f"Server {server_id} not found")

        try:
            return await self.deployment_manager.get_deployment_logs(server_id, lines)
        except Exception as e:
            logger.error(f"Failed to get server logs: {e}")
            raise

    async def get_server_metrics(self, server_id: str) -> Dict[str, Any]:
        """Get detailed server metrics."""
        if server_id not in self.servers:
            raise ValueError(f"Server {server_id} not found")

        try:
            return await self.deployment_manager.get_deployment_metrics(server_id)
        except Exception as e:
            logger.error(f"Failed to get server metrics: {e}")
            raise

    async def perform_health_check(self, server_id: str) -> Dict[str, Any]:
        """Perform health check on server."""
        if server_id not in self.servers:
            raise ValueError(f"Server {server_id} not found")

        try:
            return await self.deployment_manager.health_check(server_id)
        except Exception as e:
            logger.error(f"Failed to perform health check: {e}")
            raise
