"""Server and deployment status tracking."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from dataclasses import dataclass, field

from src.control.api.models.servers import ServerInfo, ServerState, ServerType

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str
    last_check: datetime
    check_duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentStatus:
    """Deployment status information."""
    deployment_id: str
    server_id: str
    server_type: ServerType
    state: ServerState
    health: HealthStatus
    
    # Timestamps
    created_at: datetime
    last_updated: datetime
    last_health_check: Optional[datetime] = None
    
    # Health checks
    health_checks: Dict[str, HealthCheck] = field(default_factory=dict)
    
    # Metrics
    uptime_seconds: float = 0
    total_requests: int = 0
    error_count: int = 0
    
    # Alerts
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)


class StatusTracker:
    """Tracks server and deployment status."""
    
    def __init__(
        self,
        health_check_interval: int = 30,
        unhealthy_threshold: int = 3,
    ):
        """Initialize status tracker."""
        self.health_check_interval = health_check_interval
        self.unhealthy_threshold = unhealthy_threshold
        
        self.deployments: Dict[str, DeploymentStatus] = {}
        self.health_check_failures: Dict[str, int] = {}
        
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._status_callbacks = []
    
    async def start(self):
        """Start status tracking."""
        if self.is_running:
            return
        
        logger.info("Starting status tracker")
        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop status tracking."""
        if not self.is_running:
            return
        
        logger.info("Stopping status tracker")
        self.is_running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    def add_callback(self, callback):
        """Add status update callback."""
        self._status_callbacks.append(callback)
    
    def track_deployment(
        self,
        deployment_id: str,
        server_id: str,
        server_type: ServerType,
        server_state: ServerState,
    ) -> DeploymentStatus:
        """Start tracking a deployment."""
        status = DeploymentStatus(
            deployment_id=deployment_id,
            server_id=server_id,
            server_type=server_type,
            state=server_state,
            health=HealthStatus.UNKNOWN,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
        )
        
        self.deployments[server_id] = status
        logger.info(f"Tracking deployment {deployment_id} for server {server_id}")
        
        return status
    
    def update_server_state(self, server_id: str, state: ServerState):
        """Update server state."""
        if server_id not in self.deployments:
            return
        
        deployment = self.deployments[server_id]
        deployment.state = state
        deployment.last_updated = datetime.utcnow()
        
        # Clear failure count on state change
        if state == ServerState.RUNNING:
            self.health_check_failures.pop(server_id, None)
        
        # Notify callbacks
        asyncio.create_task(self._notify_status_change(deployment))
    
    def record_request(self, server_id: str, success: bool = True):
        """Record a request for a server."""
        if server_id not in self.deployments:
            return
        
        deployment = self.deployments[server_id]
        deployment.total_requests += 1
        if not success:
            deployment.error_count += 1
        
        deployment.last_updated = datetime.utcnow()
    
    def add_alert(self, server_id: str, alert: Dict[str, Any]):
        """Add an alert for a server."""
        if server_id not in self.deployments:
            return
        
        deployment = self.deployments[server_id]
        
        # Add timestamp if not present
        if "timestamp" not in alert:
            alert["timestamp"] = datetime.utcnow().isoformat()
        
        deployment.active_alerts.append(alert)
        deployment.last_updated = datetime.utcnow()
        
        # Notify callbacks
        asyncio.create_task(self._notify_alert(deployment, alert))
    
    def clear_alerts(self, server_id: str, alert_type: Optional[str] = None):
        """Clear alerts for a server."""
        if server_id not in self.deployments:
            return
        
        deployment = self.deployments[server_id]
        
        if alert_type:
            # Clear specific type
            deployment.active_alerts = [
                a for a in deployment.active_alerts
                if a.get("type") != alert_type
            ]
        else:
            # Clear all
            deployment.active_alerts.clear()
        
        deployment.last_updated = datetime.utcnow()
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Check all deployments
                for server_id, deployment in self.deployments.items():
                    if deployment.state == ServerState.RUNNING:
                        await self._check_deployment_health(deployment)
                
                # Wait for next check
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_deployment_health(self, deployment: DeploymentStatus):
        """Check health of a deployment."""
        logger.debug(f"Checking health for {deployment.server_id}")
        
        # Define health checks
        checks = [
            ("endpoint", self._check_endpoint_health),
            ("metrics", self._check_metrics_health),
            ("resources", self._check_resource_health),
        ]
        
        overall_health = HealthStatus.HEALTHY
        check_results = {}
        
        for check_name, check_func in checks:
            try:
                start_time = datetime.utcnow()
                health_check = await check_func(deployment)
                health_check.check_duration_ms = (
                    datetime.utcnow() - start_time
                ).total_seconds() * 1000
                
                check_results[check_name] = health_check
                
                # Update overall health
                if health_check.status == HealthStatus.UNHEALTHY:
                    overall_health = HealthStatus.UNHEALTHY
                elif health_check.status == HealthStatus.DEGRADED and overall_health == HealthStatus.HEALTHY:
                    overall_health = HealthStatus.DEGRADED
                
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                check_results[check_name] = HealthCheck(
                    name=check_name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {str(e)}",
                    last_check=datetime.utcnow(),
                    check_duration_ms=0,
                )
                overall_health = HealthStatus.UNHEALTHY
        
        # Update deployment
        deployment.health_checks = check_results
        deployment.last_health_check = datetime.utcnow()
        
        # Update health status
        previous_health = deployment.health
        deployment.health = overall_health
        
        # Track failures
        if overall_health == HealthStatus.UNHEALTHY:
            self.health_check_failures[deployment.server_id] = (
                self.health_check_failures.get(deployment.server_id, 0) + 1
            )
            
            # Check if exceeded threshold
            if self.health_check_failures[deployment.server_id] >= self.unhealthy_threshold:
                self.add_alert(deployment.server_id, {
                    "type": "health_threshold_exceeded",
                    "severity": "critical",
                    "message": f"Server unhealthy for {self.unhealthy_threshold} consecutive checks",
                    "failure_count": self.health_check_failures[deployment.server_id],
                })
        else:
            # Clear failure count
            self.health_check_failures.pop(deployment.server_id, None)
        
        # Update uptime
        if deployment.created_at:
            deployment.uptime_seconds = (
                datetime.utcnow() - deployment.created_at
            ).total_seconds()
        
        # Notify if health changed
        if previous_health != overall_health:
            await self._notify_health_change(deployment, previous_health)
    
    async def _check_endpoint_health(self, deployment: DeploymentStatus) -> HealthCheck:
        """Check endpoint connectivity."""
        # In real implementation, would make HTTP request to endpoint
        # For now, simulate based on state
        if deployment.state == ServerState.RUNNING:
            return HealthCheck(
                name="endpoint",
                status=HealthStatus.HEALTHY,
                message="Endpoint responding",
                last_check=datetime.utcnow(),
                check_duration_ms=0,
                metadata={"response_time_ms": 50},
            )
        else:
            return HealthCheck(
                name="endpoint",
                status=HealthStatus.UNHEALTHY,
                message=f"Server not running (state: {deployment.state})",
                last_check=datetime.utcnow(),
                check_duration_ms=0,
            )
    
    async def _check_metrics_health(self, deployment: DeploymentStatus) -> HealthCheck:
        """Check metrics availability."""
        # In real implementation, would check metrics endpoint
        # For now, check error rate
        if deployment.total_requests > 0:
            error_rate = deployment.error_count / deployment.total_requests
            if error_rate > 0.1:  # 10% error rate
                return HealthCheck(
                    name="metrics",
                    status=HealthStatus.UNHEALTHY,
                    message=f"High error rate: {error_rate:.1%}",
                    last_check=datetime.utcnow(),
                    check_duration_ms=0,
                    metadata={"error_rate": error_rate},
                )
            elif error_rate > 0.05:  # 5% error rate
                return HealthCheck(
                    name="metrics",
                    status=HealthStatus.DEGRADED,
                    message=f"Elevated error rate: {error_rate:.1%}",
                    last_check=datetime.utcnow(),
                    check_duration_ms=0,
                    metadata={"error_rate": error_rate},
                )
        
        return HealthCheck(
            name="metrics",
            status=HealthStatus.HEALTHY,
            message="Metrics normal",
            last_check=datetime.utcnow(),
            check_duration_ms=0,
        )
    
    async def _check_resource_health(self, deployment: DeploymentStatus) -> HealthCheck:
        """Check resource utilization."""
        # In real implementation, would check actual resources
        # For now, always healthy
        return HealthCheck(
            name="resources",
            status=HealthStatus.HEALTHY,
            message="Resources within limits",
            last_check=datetime.utcnow(),
            check_duration_ms=0,
            metadata={
                "gpu_utilization": 75,
                "memory_utilization": 60,
            },
        )
    
    async def _notify_status_change(self, deployment: DeploymentStatus):
        """Notify callbacks of status change."""
        for callback in self._status_callbacks:
            try:
                await callback({
                    "event": "status_change",
                    "server_id": deployment.server_id,
                    "deployment_id": deployment.deployment_id,
                    "state": deployment.state.value,
                    "health": deployment.health.value,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    async def _notify_health_change(
        self,
        deployment: DeploymentStatus,
        previous_health: HealthStatus,
    ):
        """Notify callbacks of health change."""
        for callback in self._status_callbacks:
            try:
                await callback({
                    "event": "health_change",
                    "server_id": deployment.server_id,
                    "deployment_id": deployment.deployment_id,
                    "previous_health": previous_health.value,
                    "current_health": deployment.health.value,
                    "health_checks": {
                        name: {
                            "status": check.status.value,
                            "message": check.message,
                        }
                        for name, check in deployment.health_checks.items()
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                })
            except Exception as e:
                logger.error(f"Health callback error: {e}")
    
    async def _notify_alert(self, deployment: DeploymentStatus, alert: Dict[str, Any]):
        """Notify callbacks of new alert."""
        for callback in self._status_callbacks:
            try:
                await callback({
                    "event": "alert",
                    "server_id": deployment.server_id,
                    "deployment_id": deployment.deployment_id,
                    "alert": alert,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def get_deployment_status(self, server_id: str) -> Optional[DeploymentStatus]:
        """Get deployment status for a server."""
        return self.deployments.get(server_id)
    
    def get_all_deployments(self) -> List[DeploymentStatus]:
        """Get all deployment statuses."""
        return list(self.deployments.values())
    
    def get_unhealthy_deployments(self) -> List[DeploymentStatus]:
        """Get unhealthy deployments."""
        return [
            d for d in self.deployments.values()
            if d.health in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get status summary."""
        deployments = list(self.deployments.values())
        
        # Count by state
        state_counts = {}
        for state in ServerState:
            state_counts[state.value] = sum(
                1 for d in deployments if d.state == state
            )
        
        # Count by health
        health_counts = {}
        for health in HealthStatus:
            health_counts[health.value] = sum(
                1 for d in deployments if d.health == health
            )
        
        # Active alerts
        total_alerts = sum(len(d.active_alerts) for d in deployments)
        
        return {
            "total_deployments": len(deployments),
            "states": state_counts,
            "health": health_counts,
            "active_alerts": total_alerts,
            "unhealthy_servers": [
                d.server_id for d in self.get_unhealthy_deployments()
            ],
        }