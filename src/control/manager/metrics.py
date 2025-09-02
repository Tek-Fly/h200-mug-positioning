"""Metrics collection and aggregation."""

# Standard library imports
import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional

# Third-party imports
import numpy as np
from pydantic import BaseModel, Field

# First-party imports
from src.control.api.models.servers import ServerInfo, ServerMetrics
from src.control.manager.resource_monitor import GPUInfo, SystemResources

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series for a metric."""

    name: str
    unit: str
    points: Deque[MetricPoint] = field(default_factory=lambda: deque(maxlen=1000))

    def add_point(self, value: float, metadata: Optional[Dict] = None):
        """Add a data point."""
        self.points.append(
            MetricPoint(
                timestamp=datetime.utcnow(),
                value=value,
                metadata=metadata or {},
            )
        )

    def get_recent(self, seconds: int = 300) -> List[MetricPoint]:
        """Get recent points within time window."""
        cutoff = datetime.utcnow() - timedelta(seconds=seconds)
        return [p for p in self.points if p.timestamp > cutoff]

    def get_statistics(self, seconds: int = 300) -> Dict[str, float]:
        """Get statistics for recent points."""
        recent = self.get_recent(seconds)
        if not recent:
            return {
                "min": 0,
                "max": 0,
                "avg": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0,
                "count": 0,
            }

        values = [p.value for p in recent]
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "p50": np.percentile(values, 50),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
            "count": len(values),
        }


class MetricsCollector:
    """Collects and aggregates system metrics."""

    def __init__(self, retention_hours: int = 24):
        """Initialize metrics collector."""
        self.retention_hours = retention_hours
        self.metrics: Dict[str, Dict[str, MetricSeries]] = defaultdict(dict)

        # Request tracking
        self.request_counters: Dict[str, int] = defaultdict(int)
        self.error_counters: Dict[str, int] = defaultdict(int)
        self.latency_histograms: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=10000)
        )

        # Cost tracking
        self.cost_accumulator: Dict[str, float] = defaultdict(float)

        # Active request tracking
        self.active_requests: Dict[str, int] = defaultdict(int)

        # Network throughput tracking
        self.last_network_bytes_in: Optional[int] = None
        self.last_network_bytes_out: Optional[int] = None
        self.last_network_timestamp: Optional[datetime] = None

        # Initialize standard metrics
        self._initialize_metrics()

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False

    def _initialize_metrics(self):
        """Initialize standard metric series."""
        # GPU metrics
        gpu_metrics = [
            ("gpu_utilization", "%"),
            ("gpu_memory_used", "MB"),
            ("gpu_temperature", "°C"),
            ("gpu_power_draw", "W"),
        ]

        # System metrics
        system_metrics = [
            ("cpu_utilization", "%"),
            ("memory_utilization", "%"),
            ("memory_used", "MB"),
            ("disk_utilization", "%"),
            ("network_throughput_in", "MB/s"),
            ("network_throughput_out", "MB/s"),
        ]

        # Request metrics
        request_metrics = [
            ("requests_per_second", "req/s"),
            ("average_latency", "ms"),
            ("p95_latency", "ms"),
            ("p99_latency", "ms"),
            ("error_rate", "%"),
        ]

        # Model metrics
        model_metrics = [
            ("model_load_time", "ms"),
            ("inference_time", "ms"),
            ("cache_hit_rate", "%"),
            ("batch_size", "images"),
        ]

        # Initialize all metrics
        for category, metrics in [
            ("gpu", gpu_metrics),
            ("system", system_metrics),
            ("request", request_metrics),
            ("model", model_metrics),
        ]:
            for name, unit in metrics:
                self.metrics[category][name] = MetricSeries(name=name, unit=unit)

    async def start(self):
        """Start metrics collection."""
        if self.is_running:
            return

        logger.info("Starting metrics collection")
        self.is_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """Stop metrics collection."""
        if not self.is_running:
            return

        logger.info("Stopping metrics collection")
        self.is_running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    def record_gpu_metrics(self, gpus: List[GPUInfo]):
        """Record GPU metrics."""
        if not gpus:
            return

        # Aggregate GPU metrics
        avg_utilization = sum(gpu.utilization for gpu in gpus) / len(gpus)
        total_memory_used = sum(gpu.memory_used for gpu in gpus)
        avg_temperature = sum(gpu.temperature for gpu in gpus) / len(gpus)
        total_power_draw = sum(gpu.power_draw for gpu in gpus)

        # Record metrics
        self.metrics["gpu"]["gpu_utilization"].add_point(avg_utilization)
        self.metrics["gpu"]["gpu_memory_used"].add_point(total_memory_used)
        self.metrics["gpu"]["gpu_temperature"].add_point(avg_temperature)
        self.metrics["gpu"]["gpu_power_draw"].add_point(total_power_draw)

        # Record per-GPU metrics
        for gpu in gpus:
            self.metrics[f"gpu_{gpu.index}"]["utilization"].add_point(
                gpu.utilization,
                {"gpu_name": gpu.name},
            )

    def record_system_metrics(self, system: SystemResources):
        """Record system metrics."""
        self.metrics["system"]["cpu_utilization"].add_point(system.cpu_percent)
        self.metrics["system"]["memory_utilization"].add_point(system.memory_percent)
        self.metrics["system"]["memory_used"].add_point(system.memory_used)
        self.metrics["system"]["disk_utilization"].add_point(system.disk_percent)

        # Calculate network throughput
        current_time = datetime.utcnow()
        if (
            self.last_network_bytes_in is not None
            and self.last_network_timestamp is not None
        ):
            time_delta = (current_time - self.last_network_timestamp).total_seconds()
            if time_delta > 0:
                # Calculate throughput in MB/s
                bytes_in_delta = system.network_recv_bytes - self.last_network_bytes_in
                bytes_out_delta = (
                    system.network_sent_bytes - self.last_network_bytes_out
                )

                throughput_in = (bytes_in_delta / (1024 * 1024)) / time_delta  # MB/s
                throughput_out = (bytes_out_delta / (1024 * 1024)) / time_delta  # MB/s

                # Only record positive values (counter might reset)
                if throughput_in >= 0:
                    self.metrics["system"]["network_throughput_in"].add_point(
                        throughput_in
                    )
                if throughput_out >= 0:
                    self.metrics["system"]["network_throughput_out"].add_point(
                        throughput_out
                    )

        # Update last values
        self.last_network_bytes_in = system.network_recv_bytes
        self.last_network_bytes_out = system.network_sent_bytes
        self.last_network_timestamp = current_time

    def record_request(
        self,
        server_id: str,
        latency_ms: float,
        success: bool = True,
        metadata: Optional[Dict] = None,
    ):
        """Record a request."""
        # Update counters
        self.request_counters[server_id] += 1
        if not success:
            self.error_counters[server_id] += 1

        # Record latency
        self.latency_histograms[server_id].append(latency_ms)

        # Update metrics
        self.metrics["request"]["average_latency"].add_point(
            latency_ms,
            {"server_id": server_id, "success": success, **(metadata or {})},
        )

    def record_request_start(self, server_id: str):
        """Record request start (for tracking active requests)."""
        self.active_requests[server_id] += 1

    def record_request_end(self, server_id: str):
        """Record request end (for tracking active requests)."""
        self.active_requests[server_id] = max(0, self.active_requests[server_id] - 1)

    def record_model_metrics(
        self,
        load_time_ms: float,
        inference_time_ms: float,
        cache_hit: bool,
        batch_size: int = 1,
    ):
        """Record model performance metrics."""
        self.metrics["model"]["model_load_time"].add_point(load_time_ms)
        self.metrics["model"]["inference_time"].add_point(inference_time_ms)
        self.metrics["model"]["batch_size"].add_point(batch_size)

        # Update cache hit rate (would need running average in real impl)
        cache_hit_value = 100.0 if cache_hit else 0.0
        self.metrics["model"]["cache_hit_rate"].add_point(cache_hit_value)

    def record_cost(self, server_id: str, cost: float):
        """Record server cost."""
        self.cost_accumulator[server_id] += cost

    def get_server_metrics(self, server_id: str) -> ServerMetrics:
        """Get current metrics for a server."""
        # Calculate request metrics
        latencies = list(self.latency_histograms.get(server_id, []))
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
        else:
            avg_latency = p95_latency = p99_latency = 0

        # Calculate error rate
        total_requests = self.request_counters.get(server_id, 0)
        total_errors = self.error_counters.get(server_id, 0)
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0

        # Get recent metrics
        gpu_stats = self.metrics["gpu"]["gpu_utilization"].get_statistics(60)
        memory_stats = self.metrics["gpu"]["gpu_memory_used"].get_statistics(60)
        cpu_stats = self.metrics["system"]["cpu_utilization"].get_statistics(60)

        # Calculate RPS (requests per second over last minute)
        recent_requests = self.metrics["request"]["average_latency"].get_recent(60)
        requests_per_second = len(recent_requests) / 60 if recent_requests else 0

        return ServerMetrics(
            timestamp=datetime.utcnow(),
            # Resource metrics
            gpu_utilization=gpu_stats["avg"],
            gpu_memory_used_mb=memory_stats["avg"],
            cpu_utilization=cpu_stats["avg"],
            memory_used_mb=(
                self.metrics["system"]["memory_used"].get_statistics(60)["avg"]
                if "memory_used" in self.metrics["system"]
                else 0
            ),
            # Request metrics
            active_requests=self.active_requests.get(server_id, 0),
            requests_per_second=requests_per_second,
            average_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            # Model metrics
            model_load_time_ms=self.metrics["model"]["model_load_time"].get_statistics(
                300
            )["avg"],
            inference_time_ms=self.metrics["model"]["inference_time"].get_statistics(
                300
            )["avg"],
            cache_hit_rate=self.metrics["model"]["cache_hit_rate"].get_statistics(300)[
                "avg"
            ]
            / 100,
        )

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get comprehensive dashboard metrics."""
        # Aggregate all metrics
        dashboard = {
            "timestamp": datetime.utcnow().isoformat(),
            "gpu": {},
            "system": {},
            "requests": {},
            "model": {},
            "costs": {},
        }

        # GPU metrics
        for metric_name, series in self.metrics["gpu"].items():
            dashboard["gpu"][metric_name] = series.get_statistics(300)

        # System metrics
        for metric_name, series in self.metrics["system"].items():
            dashboard["system"][metric_name] = series.get_statistics(300)

        # Request metrics
        total_requests = sum(self.request_counters.values())
        total_errors = sum(self.error_counters.values())

        dashboard["requests"] = {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (
                (total_errors / total_requests * 100) if total_requests > 0 else 0
            ),
            "servers": {
                server_id: {
                    "requests": count,
                    "errors": self.error_counters.get(server_id, 0),
                }
                for server_id, count in self.request_counters.items()
            },
        }

        # Model metrics
        for metric_name, series in self.metrics["model"].items():
            dashboard["model"][metric_name] = series.get_statistics(300)

        # Cost metrics
        dashboard["costs"] = {
            "total": sum(self.cost_accumulator.values()),
            "by_server": dict(self.cost_accumulator),
        }

        return dashboard

    async def _cleanup_loop(self):
        """Periodically clean up old metrics."""
        while self.is_running:
            try:
                # Clean up old metric points
                cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)

                for category in self.metrics.values():
                    for series in category.values():
                        # Remove old points
                        while series.points and series.points[0].timestamp < cutoff:
                            series.points.popleft()

                # Wait for next cleanup
                await asyncio.sleep(3600)  # Run hourly

            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)

    def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format."""
        if format == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for category, metrics in self.metrics.items():
            for metric_name, series in metrics.items():
                # Get latest value
                if series.points:
                    latest = series.points[-1]
                    prometheus_name = f"h200_{category}_{metric_name}"

                    # Add metric description
                    lines.append(f"# HELP {prometheus_name} {series.name}")
                    lines.append(f"# TYPE {prometheus_name} gauge")

                    # Add metric value
                    labels = []
                    if latest.metadata:
                        labels = [f'{k}="{v}"' for k, v in latest.metadata.items()]

                    label_str = f"{{{','.join(labels)}}}" if labels else ""
                    lines.append(f"{prometheus_name}{label_str} {latest.value}")

        return "\n".join(lines)
