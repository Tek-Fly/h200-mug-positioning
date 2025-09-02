"""Models for server control endpoints."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, ConfigDict


class ServerType(str, Enum):
    """Server deployment types."""
    SERVERLESS = "serverless"
    TIMED = "timed"


class ServerAction(str, Enum):
    """Server control actions."""
    START = "start"
    STOP = "stop"
    RESTART = "restart"
    SCALE = "scale"
    UPDATE = "update"


class ServerState(str, Enum):
    """Server states."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    UPDATING = "updating"


class ServerConfig(BaseModel):
    """Server configuration."""
    gpu_type: str = Field(default="H200", description="GPU type")
    gpu_count: int = Field(default=1, ge=1, description="Number of GPUs")
    memory_gb: int = Field(default=32, ge=1, description="Memory in GB")
    cpu_cores: int = Field(default=8, ge=1, description="Number of CPU cores")
    
    # Scaling configuration
    min_instances: int = Field(default=0, ge=0, description="Minimum instances (serverless)")
    max_instances: int = Field(default=10, ge=1, description="Maximum instances (serverless)")
    target_utilization: float = Field(default=0.7, ge=0.1, le=1.0, description="Target GPU utilization")
    
    # Timeout configuration
    idle_timeout_seconds: int = Field(default=600, ge=60, description="Idle timeout before shutdown")
    max_runtime_seconds: Optional[int] = Field(None, description="Maximum runtime (timed instances)")
    
    # Environment
    environment_variables: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    docker_image: str = Field(..., description="Docker image to deploy")


class ServerInfo(BaseModel):
    """Server information."""
    model_config = ConfigDict(from_attributes=True)
    
    id: str = Field(..., description="Server instance ID")
    type: ServerType = Field(..., description="Server type")
    state: ServerState = Field(..., description="Current state")
    config: ServerConfig = Field(..., description="Server configuration")
    
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Last start timestamp")
    stopped_at: Optional[datetime] = Field(None, description="Last stop timestamp")
    
    endpoint_url: Optional[str] = Field(None, description="Server endpoint URL")
    metrics_url: Optional[str] = Field(None, description="Metrics endpoint URL")
    
    current_instances: int = Field(default=0, description="Current running instances")
    total_requests: int = Field(default=0, description="Total requests processed")
    error_count: int = Field(default=0, description="Total errors")
    
    cost_per_hour: float = Field(..., description="Estimated cost per hour")
    total_cost: float = Field(default=0.0, description="Total accumulated cost")


class ServerControlRequest(BaseModel):
    """Server control request."""
    action: ServerAction = Field(..., description="Action to perform")
    config: Optional[ServerConfig] = Field(None, description="New configuration (for update/scale)")
    force: bool = Field(default=False, description="Force action even if risky")
    wait_for_completion: bool = Field(default=True, description="Wait for action to complete")


class ServerControlResponse(BaseModel):
    """Server control response."""
    success: bool = Field(..., description="Whether action succeeded")
    server: ServerInfo = Field(..., description="Updated server information")
    message: str = Field(..., description="Action result message")
    duration_seconds: float = Field(..., description="Action duration")
    warnings: List[str] = Field(default_factory=list, description="Any warnings")


class ServerMetrics(BaseModel):
    """Server performance metrics."""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    
    # Resource metrics
    gpu_utilization: float = Field(..., description="GPU utilization percentage")
    gpu_memory_used_mb: float = Field(..., description="GPU memory used")
    cpu_utilization: float = Field(..., description="CPU utilization percentage")
    memory_used_mb: float = Field(..., description="System memory used")
    
    # Request metrics
    active_requests: int = Field(..., description="Currently active requests")
    requests_per_second: float = Field(..., description="Request rate")
    average_latency_ms: float = Field(..., description="Average request latency")
    p95_latency_ms: float = Field(..., description="95th percentile latency")
    p99_latency_ms: float = Field(..., description="99th percentile latency")
    
    # Model metrics
    model_load_time_ms: float = Field(..., description="Model loading time")
    inference_time_ms: float = Field(..., description="Average inference time")
    cache_hit_rate: float = Field(..., description="Model cache hit rate")


class ServerLogsRequest(BaseModel):
    """Server logs request."""
    lines: int = Field(default=100, ge=1, le=10000, description="Number of log lines")
    since: Optional[datetime] = Field(None, description="Logs since timestamp")
    level: Optional[str] = Field(None, description="Minimum log level")
    search: Optional[str] = Field(None, description="Search text in logs")


class ServerLogsResponse(BaseModel):
    """Server logs response."""
    server_id: str = Field(..., description="Server ID")
    total_lines: int = Field(..., description="Total matching lines")
    returned_lines: int = Field(..., description="Number of lines returned")
    logs: List[Dict[str, Any]] = Field(..., description="Log entries")
    next_cursor: Optional[str] = Field(None, description="Cursor for pagination")


class DeploymentRequest(BaseModel):
    """New deployment request."""
    type: ServerType = Field(..., description="Server type to deploy")
    config: ServerConfig = Field(..., description="Server configuration")
    auto_start: bool = Field(default=True, description="Start immediately after deployment")
    tags: Dict[str, str] = Field(default_factory=dict, description="Deployment tags")


class DeploymentResponse(BaseModel):
    """Deployment response."""
    deployment_id: str = Field(..., description="Deployment ID")
    server: ServerInfo = Field(..., description="Deployed server information")
    deployment_time_seconds: float = Field(..., description="Deployment duration")
    message: str = Field(..., description="Deployment message")