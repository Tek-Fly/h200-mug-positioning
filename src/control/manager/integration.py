"""Integration helpers for control plane manager."""

import logging
from typing import Optional

from fastapi import FastAPI, Request
from contextlib import asynccontextmanager

from src.control.manager.orchestrator import ControlPlaneOrchestrator
from src.control.api.models.servers import ServerInfo
from src.utils.secrets import get_secret

logger = logging.getLogger(__name__)


# Global orchestrator instance
_orchestrator: Optional[ControlPlaneOrchestrator] = None


def get_orchestrator() -> ControlPlaneOrchestrator:
    """Get the global orchestrator instance."""
    if _orchestrator is None:
        raise RuntimeError("Orchestrator not initialized")
    return _orchestrator


async def init_orchestrator(
    runpod_api_key: Optional[str] = None,
    idle_timeout_seconds: int = 600,
    enable_auto_shutdown: bool = True,
) -> ControlPlaneOrchestrator:
    """Initialize the global orchestrator instance."""
    global _orchestrator
    
    if _orchestrator is not None:
        return _orchestrator
    
    logger.info("Initializing control plane orchestrator")
    
    # Get API key if not provided
    if not runpod_api_key:
        runpod_api_key = get_secret("RUNPOD_API_KEY")
    
    # Create orchestrator
    _orchestrator = ControlPlaneOrchestrator(
        runpod_api_key=runpod_api_key,
        idle_timeout_seconds=idle_timeout_seconds,
        enable_auto_shutdown=enable_auto_shutdown,
    )
    
    # Start orchestrator
    await _orchestrator.start()
    
    return _orchestrator


async def shutdown_orchestrator():
    """Shutdown the global orchestrator instance."""
    global _orchestrator
    
    if _orchestrator is None:
        return
    
    logger.info("Shutting down control plane orchestrator")
    await _orchestrator.stop()
    _orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    # Startup
    logger.info("Starting control plane integration")
    
    # Initialize orchestrator
    orchestrator = await init_orchestrator()
    
    # Store in app state
    app.state.orchestrator = orchestrator
    
    # Update server manager reference in routers
    from src.control.api.routers import servers
    servers.server_manager = orchestrator.server_manager
    
    yield
    
    # Shutdown
    logger.info("Stopping control plane integration")
    await shutdown_orchestrator()


def setup_control_plane(app: FastAPI):
    """Set up control plane integration with FastAPI app."""
    # Add lifespan handler
    app.router.lifespan_context = lifespan
    
    # Add middleware to track requests
    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        """Track all requests for auto-shutdown."""
        # Get orchestrator
        orchestrator = getattr(request.app.state, "orchestrator", None)
        
        if orchestrator and request.url.path.startswith("/api/v1/analyze"):
            # Extract server ID from headers or default
            server_id = request.headers.get("X-Server-ID", "default")
            
            # Record activity
            orchestrator.auto_shutdown.record_activity(server_id)
        
        # Process request
        response = await call_next(request)
        
        # Record metrics
        if orchestrator and hasattr(request.state, "start_time"):
            latency_ms = (request.state.end_time - request.state.start_time) * 1000
            server_id = request.headers.get("X-Server-ID", "default")
            
            orchestrator.record_request(
                server_id=server_id,
                latency_ms=latency_ms,
                success=response.status_code < 400,
                endpoint=str(request.url.path),
            )
        
        return response
    
    logger.info("Control plane integration configured")


# Helper functions for API endpoints
async def get_server_by_type(server_type: str) -> Optional[ServerInfo]:
    """Get a server by type from the orchestrator."""
    orchestrator = get_orchestrator()
    servers = await orchestrator.server_manager.list_servers()
    
    for server in servers:
        if server.type.value == server_type:
            return server
    
    return None


async def ensure_server_running(server_type: str) -> ServerInfo:
    """Ensure a server of the given type is running."""
    orchestrator = get_orchestrator()
    
    # Get server
    server = await get_server_by_type(server_type)
    
    if not server:
        raise ValueError(f"No {server_type} server found")
    
    # Start if not running
    if server.state.value != "running":
        server = await orchestrator.start_server(server.id)
    
    return server


async def record_model_performance(
    load_time_ms: float,
    inference_time_ms: float,
    cache_hit: bool,
    batch_size: int = 1,
):
    """Record model performance metrics."""
    orchestrator = get_orchestrator()
    orchestrator.metrics_collector.record_model_metrics(
        load_time_ms=load_time_ms,
        inference_time_ms=inference_time_ms,
        cache_hit=cache_hit,
        batch_size=batch_size,
    )