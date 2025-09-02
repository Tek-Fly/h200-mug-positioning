"""Server control API endpoints."""

# Standard library imports
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

# Third-party imports
from aiohttp import ClientError
from fastapi import APIRouter, Depends, HTTPException, Request, status

# First-party imports
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
from src.deployment.client import RunPodAPIError

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
            await request.app.state.mongodb.server_activity.insert_one(
                {
                    "server_id": server.id,
                    "server_type": server.type,
                    "action": control_request.action,
                    "user_id": current_user,
                    "timestamp": datetime.utcnow(),
                    "duration_seconds": duration,
                    "success": True,
                }
            )

        return ServerControlResponse(
            success=True,
            server=server,
            message=message,
            duration_seconds=duration,
            warnings=[],
        )

    except HTTPException:
        raise
    except ValueError as e:
        # Invalid server ID or configuration
        logger.error(f"Invalid request for server control: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RunPodAPIError as e:
        # RunPod API specific errors
        logger.error(f"RunPod API error controlling server: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"RunPod API error: {str(e)}",
        )
    except ClientError as e:
        # Network/connection errors
        logger.error(f"Network error controlling server: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service temporarily unavailable: {str(e)}",
        )
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error controlling server: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
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

    except ValueError as e:
        # Invalid configuration
        logger.error(f"Invalid deployment configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RunPodAPIError as e:
        # RunPod API specific errors
        logger.error(f"RunPod API error during deployment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"RunPod API error: {str(e)}",
        )
    except ClientError as e:
        # Network/connection errors
        logger.error(f"Network error during deployment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service temporarily unavailable: {str(e)}",
        )
    except RuntimeError as e:
        # Deployment runtime errors (e.g., resource limits)
        logger.error(f"Deployment runtime error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail=f"Insufficient resources: {str(e)}",
        )
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error deploying server: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
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

    # Get actual logs from RunPod
    try:
        # Get raw logs from server manager
        raw_logs = await orchestrator.server_manager.get_server_logs(
            server_id, lines=logs_request.lines
        )

        # Parse logs into structured format
        logs = []
        log_lines = raw_logs.strip().split("\n") if raw_logs else []

        for line in log_lines:
            if not line.strip():
                continue

            # Parse log line - RunPod logs typically have timestamp and content
            # Format varies but often: [TIMESTAMP] LEVEL: MESSAGE
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),  # Default timestamp
                "level": "INFO",  # Default level
                "message": line,
                "context": {
                    "server_id": server_id,
                    "user_id": current_user,
                },
            }

            # Try to parse structured log format
            if line.startswith("["):
                # Extract timestamp if present
                try:
                    end_bracket = line.find("]")
                    if end_bracket > 0:
                        timestamp_str = line[1:end_bracket]
                        # Parse timestamp - adjust format as needed
                        try:
                            # Try to parse as ISO format first
                            parsed_time = datetime.fromisoformat(
                                timestamp_str.replace("Z", "+00:00")
                            )
                            log_entry["timestamp"] = parsed_time.isoformat()
                            line = line[end_bracket + 1 :].strip()
                        except:
                            pass
                except:
                    pass

            # Extract log level if present
            for level in ["ERROR", "WARN", "WARNING", "INFO", "DEBUG"]:
                if line.startswith(f"{level}:") or f" {level}:" in line[:20]:
                    log_entry["level"] = level if level != "WARNING" else "WARN"
                    break

            # Apply filters if requested
            if logs_request.level:
                level_priority = {"ERROR": 4, "WARN": 3, "INFO": 2, "DEBUG": 1}
                min_priority = level_priority.get(logs_request.level.upper(), 0)
                entry_priority = level_priority.get(log_entry["level"], 0)
                if entry_priority < min_priority:
                    continue

            if logs_request.search and logs_request.search.lower() not in line.lower():
                continue

            if logs_request.since:
                try:
                    entry_time = datetime.fromisoformat(
                        log_entry["timestamp"].replace("Z", "+00:00")
                    )
                    if entry_time.replace(tzinfo=None) < logs_request.since.replace(
                        tzinfo=None
                    ):
                        continue
                except:
                    pass

            logs.append(log_entry)

        # Limit to requested number of lines
        logs = logs[-logs_request.lines :] if len(logs) > logs_request.lines else logs

        return ServerLogsResponse(
            server_id=server_id,
            total_lines=len(log_lines),
            returned_lines=len(logs),
            logs=logs,
            next_cursor=None,  # RunPod doesn't support pagination for logs
        )

    except ValueError as e:
        # Server not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except RunPodAPIError as e:
        # RunPod API specific errors
        logger.error(f"RunPod API error retrieving logs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"RunPod API error: {str(e)}",
        )
    except ClientError as e:
        # Network/connection errors
        logger.error(f"Network error retrieving logs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service temporarily unavailable: {str(e)}",
        )
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(
            f"Unexpected error retrieving logs for server {server_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
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
            await request.app.state.mongodb.server_activity.insert_one(
                {
                    "server_id": server_id,
                    "action": "delete",
                    "user_id": current_user,
                    "timestamp": datetime.utcnow(),
                }
            )

        return {
            "success": True,
            "message": f"Server {server_id} deleted successfully",
        }

    except ValueError as e:
        # Server not found or invalid request
        logger.error(f"Invalid delete request: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except RunPodAPIError as e:
        # RunPod API specific errors
        logger.error(f"RunPod API error deleting server: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"RunPod API error: {str(e)}",
        )
    except ClientError as e:
        # Network/connection errors
        logger.error(f"Network error deleting server: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service temporarily unavailable: {str(e)}",
        )
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error deleting server: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
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
