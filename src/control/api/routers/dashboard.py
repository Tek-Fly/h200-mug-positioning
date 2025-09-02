"""Dashboard API endpoints."""

# Standard library imports
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

# Third-party imports
import psutil
from fastapi import APIRouter, Depends, HTTPException, Request, status

# First-party imports
from src.control.api.middleware.auth import get_current_user
from src.control.api.models.dashboard import (
    ActivityLog,
    CostMetrics,
    DashboardRequest,
    DashboardResponse,
    MetricType,
    PerformanceMetrics,
    ResourceUsage,
    ServiceHealth,
    ServiceStatus,
    SystemMetric,
)
from src.control.manager.orchestrator import ControlPlaneOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_service_health(request: Request) -> Dict[str, ServiceHealth]:
    """Check health of all services."""
    services = {}

    # MongoDB health
    mongodb_health = ServiceHealth(
        name="MongoDB",
        status=ServiceStatus.UNKNOWN,
        last_check=datetime.utcnow(),
        details={},
    )

    try:
        if hasattr(request.app.state, "mongodb"):
            await request.app.state.mongodb.admin.command("ping")
            mongodb_health.status = ServiceStatus.HEALTHY

            # Get connection info
            server_info = await request.app.state.mongodb.server_info()
            mongodb_health.details = {
                "version": server_info.get("version"),
                "uptime": server_info.get("uptimeMillis", 0) / 1000,
            }
            mongodb_health.uptime_seconds = mongodb_health.details["uptime"]
    except Exception as e:
        mongodb_health.status = ServiceStatus.UNHEALTHY
        mongodb_health.details["error"] = str(e)

    services["mongodb"] = mongodb_health

    # Redis health
    redis_health = ServiceHealth(
        name="Redis",
        status=ServiceStatus.UNKNOWN,
        last_check=datetime.utcnow(),
        details={},
    )

    try:
        if hasattr(request.app.state, "redis"):
            await request.app.state.redis.ping()
            redis_health.status = ServiceStatus.HEALTHY

            # Get Redis info
            info = await request.app.state.redis.info()
            redis_health.details = {
                "version": info.get("redis_version"),
                "memory_used_mb": info.get("used_memory", 0) / 1024 / 1024,
                "connected_clients": info.get("connected_clients", 0),
            }
            redis_health.uptime_seconds = info.get("uptime_in_seconds", 0)
    except Exception as e:
        redis_health.status = ServiceStatus.UNHEALTHY
        redis_health.details["error"] = str(e)

    services["redis"] = redis_health

    # Model service health
    model_health = ServiceHealth(
        name="ML Models",
        status=ServiceStatus.UNKNOWN,
        last_check=datetime.utcnow(),
        details={},
    )

    try:
        if hasattr(request.app.state, "model_manager"):
            if request.app.state.model_manager.is_initialized:
                model_health.status = ServiceStatus.HEALTHY
                model_health.details = {
                    "models_loaded": len(request.app.state.model_manager._models),
                    "cache_enabled": True,
                }
            else:
                model_health.status = ServiceStatus.DEGRADED
                model_health.details["message"] = "Models not fully initialized"
    except Exception as e:
        model_health.status = ServiceStatus.UNHEALTHY
        model_health.details["error"] = str(e)

    services["models"] = model_health

    # Control plane health
    control_health = ServiceHealth(
        name="Control Plane",
        status=ServiceStatus.UNKNOWN,
        last_check=datetime.utcnow(),
        details={},
    )

    try:
        if hasattr(request.app.state, "orchestrator"):
            orchestrator = request.app.state.orchestrator
            if orchestrator.is_running:
                control_health.status = ServiceStatus.HEALTHY
                control_health.details = {
                    "servers": len(await orchestrator.server_manager.list_servers()),
                    "auto_shutdown_enabled": orchestrator.enable_auto_shutdown,
                    "active_notifications": orchestrator.notifier.get_statistics()[
                        "connected_clients"
                    ],
                }
            else:
                control_health.status = ServiceStatus.DEGRADED
                control_health.details["message"] = (
                    "Control plane not fully operational"
                )
    except Exception as e:
        control_health.status = ServiceStatus.UNHEALTHY
        control_health.details["error"] = str(e)

    services["control_plane"] = control_health

    return services


async def get_performance_metrics(request: Request) -> PerformanceMetrics:
    """Get system performance metrics."""
    # These would typically come from Prometheus or similar monitoring system
    # For now, returning placeholder values

    metrics = PerformanceMetrics(
        cold_start_ms=1500.0,  # FlashBoot cold start
        warm_start_ms=50.0,
        image_processing_ms=450.0,
        api_latency_p95_ms=180.0,
        gpu_utilization_percent=0.0,  # Would need nvidia-ml-py for real GPU stats
        cache_hit_rate=0.0,
        requests_per_second=0.0,
    )

    # Try to get cache hit rate from Redis
    try:
        if hasattr(request.app.state, "redis"):
            info = await request.app.state.redis.info("stats")
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            total = hits + misses
            if total > 0:
                metrics.cache_hit_rate = (hits / total) * 100
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")

    return metrics


async def get_resource_usage() -> ResourceUsage:
    """Get current resource usage."""
    # CPU and memory stats
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    usage = ResourceUsage(
        cpu_percent=cpu_percent,
        memory_used_mb=memory.used / 1024 / 1024,
        memory_total_mb=memory.total / 1024 / 1024,
        disk_used_gb=disk.used / 1024 / 1024 / 1024,
        disk_total_gb=disk.total / 1024 / 1024 / 1024,
    )

    # Try to get GPU stats (requires nvidia-ml-py)
    try:
        # Third-party imports
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            usage.gpu_memory_used_mb = mem_info.used / 1024 / 1024
            usage.gpu_memory_total_mb = mem_info.total / 1024 / 1024
    except Exception:
        # GPU stats not available
        pass

    return usage


async def get_cost_metrics() -> CostMetrics:
    """Calculate cost metrics."""
    # These are example calculations - adjust based on actual pricing

    # H200 GPU pricing (example: $3.50/hour)
    gpu_hourly_rate = 3.50

    # Storage pricing (example: $0.023/GB/month)
    storage_gb_monthly = 0.023

    # Network pricing (example: $0.09/GB egress)
    network_gb_rate = 0.09

    # Calculate daily costs
    compute_cost = gpu_hourly_rate * 24  # Assuming 24/7 operation
    storage_cost = (100 * storage_gb_monthly) / 30  # Assuming 100GB storage
    network_cost = 10 * network_gb_rate  # Assuming 10GB daily transfer

    total_cost = compute_cost + storage_cost + network_cost

    return CostMetrics(
        period="daily",
        compute_cost=compute_cost,
        storage_cost=storage_cost,
        network_cost=network_cost,
        total_cost=total_cost,
        cost_per_request=total_cost / 10000,  # Assuming 10k requests/day
        breakdown={
            "gpu_hours": 24,
            "storage_gb": 100,
            "network_gb": 10,
        },
    )


async def get_recent_activity(
    request: Request,
    limit: int = 50,
) -> list[ActivityLog]:
    """Get recent system activity."""
    activities = []

    try:
        if hasattr(request.app.state, "mongodb"):
            # Get recent analysis requests
            cursor = (
                request.app.state.mongodb.analysis_results.find(
                    {},
                    {
                        "request_id": 1,
                        "user_id": 1,
                        "timestamp": 1,
                        "processing_time_ms": 1,
                    },
                )
                .sort("timestamp", -1)
                .limit(limit)
            )

            async for doc in cursor:
                activity = ActivityLog(
                    timestamp=doc["timestamp"],
                    type="analysis",
                    user_id=doc.get("user_id"),
                    action="Image analysis completed",
                    details={
                        "request_id": doc["request_id"],
                    },
                    duration_ms=doc.get("processing_time_ms"),
                )
                activities.append(activity)

    except Exception as e:
        logger.error(f"Error getting activity logs: {e}")

    return activities


async def get_orchestrator(request: Request) -> Optional[ControlPlaneOrchestrator]:
    """Get orchestrator from app state if available."""
    return getattr(request.app.state, "orchestrator", None)


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    request: Request,
    include_metrics: bool = True,
    include_health: bool = True,
    include_resources: bool = True,
    include_costs: bool = True,
    include_activity: bool = True,
    activity_limit: int = 50,
    current_user: str = Depends(get_current_user),
    orchestrator: Optional[ControlPlaneOrchestrator] = Depends(get_orchestrator),
) -> DashboardResponse:
    """
    Get complete dashboard data including:
    - Service health status
    - Performance metrics
    - Resource usage
    - Cost tracking
    - Recent activity
    """
    response = DashboardResponse(timestamp=datetime.utcnow())

    try:
        # Service health
        if include_health:
            services = await get_service_health(request)
            response.services = list(services.values())

            # Determine overall health
            statuses = [s.status for s in services.values()]
            if all(s == ServiceStatus.HEALTHY for s in statuses):
                response.overall_health = ServiceStatus.HEALTHY
            elif any(s == ServiceStatus.UNHEALTHY for s in statuses):
                response.overall_health = ServiceStatus.UNHEALTHY
            else:
                response.overall_health = ServiceStatus.DEGRADED

        # Performance metrics
        if include_metrics:
            response.performance = await get_performance_metrics(request)

            # Add additional system metrics
            response.metrics = [
                SystemMetric(
                    name="api_requests_total",
                    type=MetricType.COUNTER,
                    value=12345,  # Would come from metrics system
                    timestamp=datetime.utcnow(),
                ),
                SystemMetric(
                    name="active_connections",
                    type=MetricType.GAUGE,
                    value=42,
                    timestamp=datetime.utcnow(),
                ),
            ]

        # Resource usage
        if include_resources:
            response.resources = await get_resource_usage()

        # Cost metrics
        if include_costs:
            response.costs = await get_cost_metrics()

        # Recent activity
        if include_activity:
            response.recent_activity = await get_recent_activity(
                request, activity_limit
            )

        # Summary statistics
        response.summary = {
            "total_requests_today": 12345,
            "average_response_time_ms": 180,
            "error_rate_percent": 0.5,
            "active_users": 42,
        }

        # Add control plane data if available
        if orchestrator and include_metrics:
            try:
                # Get comprehensive dashboard data from orchestrator
                control_data = await orchestrator.get_dashboard_data()

                # Update response with real data
                if "metrics" in control_data:
                    metrics_data = control_data["metrics"]

                    # Update performance metrics with real data
                    if "request" in metrics_data:
                        response.performance.api_latency_p95_ms = (
                            metrics_data["request"]
                            .get("average_latency", {})
                            .get("p95", 180.0)
                        )
                        response.performance.requests_per_second = (
                            metrics_data["requests"].get("total_requests", 0) / 3600
                        )  # Convert to RPS

                    if "gpu" in metrics_data:
                        response.performance.gpu_utilization_percent = (
                            metrics_data["gpu"]
                            .get("gpu_utilization", {})
                            .get("avg", 0.0)
                        )

                    if "model" in metrics_data:
                        response.performance.cache_hit_rate = (
                            metrics_data["model"]
                            .get("cache_hit_rate", {})
                            .get("avg", 0.0)
                        )

                # Update costs with real data
                if "costs" in control_data and include_costs:
                    cost_data = control_data["costs"]
                    response.costs.total_cost = cost_data.get(
                        "total", response.costs.total_cost
                    )
                    response.costs.breakdown["servers"] = cost_data.get("by_server", {})

                # Add server status to summary
                if "servers" in control_data:
                    response.summary["active_servers"] = len(
                        [s for s in control_data["servers"] if s["state"] == "running"]
                    )
                    response.summary["total_servers"] = len(control_data["servers"])

                # Add auto-shutdown stats
                if "auto_shutdown" in control_data:
                    shutdown_stats = control_data["auto_shutdown"]
                    response.summary["auto_shutdowns_today"] = shutdown_stats.get(
                        "total_shutdowns", 0
                    )
                    response.summary["cost_savings_today"] = shutdown_stats.get(
                        "total_savings", 0.0
                    )

            except Exception as e:
                logger.error(f"Error getting control plane data: {e}")

        return response

    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dashboard data: {str(e)}",
        )


@router.get("/dashboard/metrics/{metric_name}")
async def get_metric_history(
    request: Request,
    metric_name: str,
    time_range: str = "1h",
    current_user: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get historical data for a specific metric."""
    # This would typically query a time-series database
    # For now, returning placeholder data

    return {
        "metric": metric_name,
        "time_range": time_range,
        "data_points": [
            {
                "timestamp": datetime.utcnow() - timedelta(minutes=i),
                "value": 50 + (i % 10) * 5,
            }
            for i in range(60, 0, -5)
        ],
    }
