"""Models for dashboard endpoints."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, ConfigDict


class ServiceStatus(str, Enum):
    """Service status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class TimeRange(str, Enum):
    """Predefined time ranges."""
    LAST_HOUR = "1h"
    LAST_DAY = "24h"
    LAST_WEEK = "7d"
    LAST_MONTH = "30d"


class ServiceHealth(BaseModel):
    """Service health information."""
    name: str = Field(..., description="Service name")
    status: ServiceStatus = Field(..., description="Current status")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime")
    last_check: datetime = Field(..., description="Last health check time")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


class SystemMetric(BaseModel):
    """System metric data."""
    name: str = Field(..., description="Metric name")
    type: MetricType = Field(..., description="Metric type")
    value: float = Field(..., description="Current value")
    unit: Optional[str] = Field(None, description="Metric unit")
    labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels")
    timestamp: datetime = Field(..., description="Measurement timestamp")


class PerformanceMetrics(BaseModel):
    """Performance metrics summary."""
    cold_start_ms: float = Field(..., description="Average cold start time")
    warm_start_ms: float = Field(..., description="Average warm start time")
    image_processing_ms: float = Field(..., description="Average image processing time")
    api_latency_p95_ms: float = Field(..., description="95th percentile API latency")
    gpu_utilization_percent: float = Field(..., description="Current GPU utilization")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    requests_per_second: float = Field(..., description="Current request rate")


class ResourceUsage(BaseModel):
    """Resource usage information."""
    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_used_mb: float = Field(..., description="Memory used in MB")
    memory_total_mb: float = Field(..., description="Total memory in MB")
    gpu_memory_used_mb: Optional[float] = Field(None, description="GPU memory used")
    gpu_memory_total_mb: Optional[float] = Field(None, description="Total GPU memory")
    disk_used_gb: float = Field(..., description="Disk space used in GB")
    disk_total_gb: float = Field(..., description="Total disk space in GB")


class CostMetrics(BaseModel):
    """Cost tracking metrics."""
    period: str = Field(..., description="Cost period (e.g., 'daily', 'monthly')")
    compute_cost: float = Field(..., description="Compute cost in USD")
    storage_cost: float = Field(..., description="Storage cost in USD")
    network_cost: float = Field(..., description="Network cost in USD")
    total_cost: float = Field(..., description="Total cost in USD")
    cost_per_request: float = Field(..., description="Average cost per request")
    breakdown: Dict[str, float] = Field(default_factory=dict, description="Detailed cost breakdown")


class ActivityLog(BaseModel):
    """Activity log entry."""
    timestamp: datetime = Field(..., description="Activity timestamp")
    type: str = Field(..., description="Activity type")
    user_id: Optional[str] = Field(None, description="User who performed the action")
    action: str = Field(..., description="Action performed")
    details: Dict[str, Any] = Field(default_factory=dict, description="Activity details")
    duration_ms: Optional[float] = Field(None, description="Action duration")


class DashboardRequest(BaseModel):
    """Dashboard data request."""
    include_metrics: bool = Field(default=True, description="Include performance metrics")
    include_health: bool = Field(default=True, description="Include service health")
    include_resources: bool = Field(default=True, description="Include resource usage")
    include_costs: bool = Field(default=True, description="Include cost metrics")
    include_activity: bool = Field(default=True, description="Include recent activity")
    time_range: TimeRange = Field(default=TimeRange.LAST_HOUR, description="Time range for metrics")
    activity_limit: int = Field(default=50, ge=1, le=1000, description="Number of activity entries")


class DashboardResponse(BaseModel):
    """Complete dashboard data."""
    model_config = ConfigDict(from_attributes=True)
    
    timestamp: datetime = Field(..., description="Dashboard data timestamp")
    
    # Service health
    services: Optional[List[ServiceHealth]] = Field(None, description="Service health status")
    overall_health: Optional[ServiceStatus] = Field(None, description="Overall system health")
    
    # Performance metrics
    performance: Optional[PerformanceMetrics] = Field(None, description="Performance metrics")
    metrics: Optional[List[SystemMetric]] = Field(None, description="Additional system metrics")
    
    # Resource usage
    resources: Optional[ResourceUsage] = Field(None, description="Resource usage")
    
    # Cost tracking
    costs: Optional[CostMetrics] = Field(None, description="Cost metrics")
    
    # Recent activity
    recent_activity: Optional[List[ActivityLog]] = Field(None, description="Recent activity log")
    
    # Summary statistics
    summary: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics")


class AlertRule(BaseModel):
    """Alert rule configuration."""
    id: str = Field(..., description="Alert rule ID")
    name: str = Field(..., description="Alert name")
    metric: str = Field(..., description="Metric to monitor")
    condition: str = Field(..., description="Alert condition (e.g., '> 90')")
    threshold: float = Field(..., description="Alert threshold")
    duration_seconds: int = Field(..., description="Duration before triggering")
    enabled: bool = Field(default=True, description="Whether alert is enabled")
    notification_channels: List[str] = Field(default_factory=list, description="Notification channels")


class Alert(BaseModel):
    """Active alert."""
    id: str = Field(..., description="Alert ID")
    rule_id: str = Field(..., description="Alert rule ID")
    severity: str = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    triggered_at: datetime = Field(..., description="When alert was triggered")
    acknowledged: bool = Field(default=False, description="Whether alert was acknowledged")
    resolved_at: Optional[datetime] = Field(None, description="When alert was resolved")
    details: Dict[str, Any] = Field(default_factory=dict, description="Alert details")