"""Server control API endpoints."""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.control.api.middleware.auth import get_current_user, require_permission
from src.control.api.models.servers import (
    DeploymentRequest,
    DeploymentResponse,
    ServerAction,
    ServerConfig,
    ServerControlRequest,
    ServerControlResponse,
    ServerInfo,
    ServerLogsRequest,
    ServerLogsResponse,
    ServerMetrics,
    ServerState,
    ServerType,
)
from src.control.manager.orchestrator import ControlPlaneOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_orchestrator(request: Request) -> ControlPlaneOrchestrator:
    """Get orchestrator from app state."""
    if not hasattr(request.app.state, "orchestrator"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Control plane not initialized",
        )
    
    return request.app.state.orchestrator


@router.post("/servers/{server_type}/control", response_model=ServerControlResponse)
async def control_server(
    request: Request,
    server_type: ServerType,
    control_request: ServerControlRequest,
    current_user: str = Depends(get_current_user),
    orchestrator: ControlPlaneOrchestrator = Depends(get_orchestrator),
) -> ServerControlResponse:
    """
    Control server lifecycle (start, stop, restart, scale).
    
    Actions:
    - start: Start a stopped server
    - stop: Stop a running server
    - restart: Restart a running server
    - scale: Update server configuration (serverless only)
    """
    start_time = time.time()
    
    try:
        # Find server by type
        servers = await orchestrator.server_manager.list_servers(server_type)
        if not servers:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No {server_type} server found",
            )
        
        server = servers[0]  # Get first server of type
        
        # Execute action
        if control_request.action == ServerAction.START:
            server = await orchestrator.start_server(server.id)
            message = f"{server_type} server started successfully"
        
        elif control_request.action == ServerAction.STOP:
            server = await orchestrator.stop_server(server.id, control_request.force)
            message = f"{server_type} server stopped successfully"
        
        elif control_request.action == ServerAction.RESTART:
            server = await orchestrator.restart_server(server.id)
            message = f"{server_type} server restarted successfully"
        
        elif control_request.action == ServerAction.SCALE:
            if server_type != ServerType.SERVERLESS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Can only scale serverless endpoints",
                )
            
            if not control_request.config:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Configuration required for scale action",
                )
            
            server = await orchestrator.scale_server(
                server.id,
                min_instances=control_request.config.min_instances,
                max_instances=control_request.config.max_instances,
            )
            message = f"{server_type} server scaled successfully"
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown action: {control_request.action}",
            )
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log activity
        if hasattr(request.app.state, "mongodb"):
            await request.app.state.mongodb.server_activity.insert_one({
                "server_id": server.id,
                "server_type": server.type,
                "action": control_request.action,
                "user_id": current_user,
                "timestamp": datetime.utcnow(),
                "duration_seconds": duration,
                "success": True,
            })
        
        return ServerControlResponse(
            success=True,
            server=server,
            message=message,
            duration_seconds=duration,
            warnings=[],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error controlling server: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to control server: {str(e)}",
        )


@router.get("/servers", response_model=List[ServerInfo])
async def list_servers(
    request: Request,
    server_type: Optional[ServerType] = None,
    current_user: str = Depends(get_current_user),
    orchestrator: ControlPlaneOrchestrator = Depends(get_orchestrator),
) -> List[ServerInfo]:
    """List all configured servers."""
    return await orchestrator.server_manager.list_servers(server_type)


@router.get("/servers/{server_id}", response_model=ServerInfo)
async def get_server(
    request: Request,
    server_id: str,
    current_user: str = Depends(get_current_user),
    orchestrator: ControlPlaneOrchestrator = Depends(get_orchestrator),
) -> ServerInfo:
    """Get detailed information about a specific server."""
    servers = await orchestrator.server_manager.list_servers()
    
    for server in servers:
        if server.id == server_id:
            return server
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Server {server_id} not found",
    )


@router.post("/servers/deploy", response_model=DeploymentResponse)
async def deploy_server(
    request: Request,
    deployment: DeploymentRequest,
    current_user: str = Depends(get_current_user),
    orchestrator: ControlPlaneOrchestrator = Depends(get_orchestrator),
) -> DeploymentResponse:
    """Deploy a new server instance."""
    start_time = time.time()
    
    try:
        # Deploy server through orchestrator
        server = await orchestrator.deploy_server(
            server_type=deployment.type,
            config=deployment.config,
            auto_start=deployment.auto_start,
            tags=deployment.tags,
        )
        
        # Calculate deployment time
        deployment_time = time.time() - start_time
        
        return DeploymentResponse(
            deployment_id=str(uuid4()),
            server=server,
            deployment_time_seconds=deployment_time,
            message=f"Server {server.id} deployed successfully",
        )
        
    except Exception as e:
        logger.error(f"Error deploying server: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deploy server: {str(e)}",
        )


@router.get("/servers/{server_id}/metrics", response_model=ServerMetrics)
async def get_server_metrics(
    request: Request,
    server_id: str,
    current_user: str = Depends(get_current_user),
    orchestrator: ControlPlaneOrchestrator = Depends(get_orchestrator),
) -> ServerMetrics:
    """Get current performance metrics for a server."""
    # Verify server exists
    servers = await orchestrator.server_manager.list_servers()
    server = None
    
    for s in servers:
        if s.id == server_id:
            server = s
            break
    
    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server {server_id} not found",
        )
    
    if server.state != ServerState.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Server is not running (current state: {server.state})",
        )
    
    # Get metrics from orchestrator
    return orchestrator.metrics_collector.get_server_metrics(server_id)


@router.post("/servers/{server_id}/logs", response_model=ServerLogsResponse)
async def get_server_logs(
    request: Request,
    server_id: str,
    logs_request: ServerLogsRequest,
    current_user: str = Depends(get_current_user),
    orchestrator: ControlPlaneOrchestrator = Depends(get_orchestrator),
) -> ServerLogsResponse:
    """Get logs from a server."""
    # Verify server exists
    servers = await orchestrator.server_manager.list_servers()
    server = None
    
    for s in servers:
        if s.id == server_id:
            server = s
            break
    
    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server {server_id} not found",
        )
    
    # TODO: Implement actual log retrieval from RunPod
    # For now, return simulated logs
    logs = []
    for i in range(min(logs_request.lines, 10)):
        logs.append({
            "timestamp": datetime.utcnow().isoformat(),
            "level": "INFO" if i % 3 == 0 else "DEBUG",
            "message": f"Sample log entry {i + 1}",
            "context": {
                "request_id": str(uuid4()),
                "user_id": current_user,
            },
        })
    
    return ServerLogsResponse(
        server_id=server_id,
        total_lines=100,  # Simulated total
        returned_lines=len(logs),
        logs=logs,
        next_cursor=None,
    )


@router.delete("/servers/{server_id}")
async def delete_server(
    request: Request,
    server_id: str,
    current_user: str = Depends(get_current_user),
    orchestrator: ControlPlaneOrchestrator = Depends(get_orchestrator),
) -> dict:
    """Delete a server deployment."""
    try:
        # Delete through orchestrator
        success = await orchestrator.server_manager.delete_server(server_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete server",
            )
        
        # Log deletion
        if hasattr(request.app.state, "mongodb"):
            await request.app.state.mongodb.server_activity.insert_one({
                "server_id": server_id,
                "action": "delete",
                "user_id": current_user,
                "timestamp": datetime.utcnow(),
            })
        
        return {
            "success": True,
            "message": f"Server {server_id} deleted successfully",
        }
        
    except Exception as e:
        logger.error(f"Error deleting server: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete server: {str(e)}",
        )


@router.get("/servers/{server_id}/status")
async def get_server_status(
    request: Request,
    server_id: str,
    current_user: str = Depends(get_current_user),
    orchestrator: ControlPlaneOrchestrator = Depends(get_orchestrator),
) -> dict:
    """Get detailed status information for a server."""
    # Get deployment status
    deployment_status = orchestrator.status_tracker.get_deployment_status(server_id)
    
    if not deployment_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No status information for server {server_id}",
        )
    
    return {
        "server_id": server_id,
        "deployment_id": deployment_status.deployment_id,
        "state": deployment_status.state.value,
        "health": deployment_status.health.value,
        "uptime_seconds": deployment_status.uptime_seconds,
        "total_requests": deployment_status.total_requests,
        "error_count": deployment_status.error_count,
        "health_checks": {
            name: {
                "status": check.status.value,
                "message": check.message,
                "last_check": check.last_check.isoformat(),
            }
            for name, check in deployment_status.health_checks.items()
        },
        "active_alerts": deployment_status.active_alerts,
        "last_updated": deployment_status.last_updated.isoformat(),
    }


@router.post("/servers/{server_id}/protect")
async def protect_server(
    request: Request,
    server_id: str,
    current_user: str = Depends(get_current_user),
    orchestrator: ControlPlaneOrchestrator = Depends(get_orchestrator),
) -> dict:
    """Protect a server from auto-shutdown."""
    orchestrator.auto_shutdown.protect_server(server_id)
    
    return {
        "success": True,
        "message": f"Server {server_id} protected from auto-shutdown",
    }


@router.delete("/servers/{server_id}/protect")
async def unprotect_server(
    request: Request,
    server_id: str,
    current_user: str = Depends(get_current_user),
    orchestrator: ControlPlaneOrchestrator = Depends(get_orchestrator),
) -> dict:
    """Remove auto-shutdown protection from a server."""
    orchestrator.auto_shutdown.unprotect_server(server_id)
    
    return {
        "success": True,
        "message": f"Server {server_id} auto-shutdown protection removed",
    }