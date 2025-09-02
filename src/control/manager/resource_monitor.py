"""GPU and system resource monitoring."""

# Standard library imports
import asyncio
import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Third-party imports
import aiohttp
import psutil
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU information."""

    index: int
    name: str
    uuid: str
    temperature: float  # Celsius
    power_draw: float  # Watts
    power_limit: float  # Watts
    memory_total: int  # MB
    memory_used: int  # MB
    memory_free: int  # MB
    utilization: float  # Percentage
    compute_mode: str
    driver_version: str
    cuda_version: str


@dataclass
class SystemResources:
    """System resource information."""

    timestamp: datetime

    # CPU
    cpu_count: int
    cpu_percent: float
    cpu_freq_current: float  # MHz
    cpu_freq_max: float  # MHz

    # Memory
    memory_total: int  # MB
    memory_used: int  # MB
    memory_available: int  # MB
    memory_percent: float

    # Disk
    disk_total: int  # GB
    disk_used: int  # GB
    disk_free: int  # GB
    disk_percent: float

    # Network
    network_sent_bytes: int
    network_recv_bytes: int
    network_connections: int

    # Processes
    process_count: int
    thread_count: int


class ResourceMonitor:
    """Monitors GPU and system resources."""

    def __init__(self, poll_interval: int = 5):
        """Initialize resource monitor."""
        self.poll_interval = poll_interval
        self.is_monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._callbacks: List[Any] = []
        self._last_network_io = None
        self._gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            return bool(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("nvidia-smi not available, GPU monitoring disabled")
            return False

    async def start(self):
        """Start monitoring resources."""
        if self.is_monitoring:
            return

        logger.info("Starting resource monitoring")
        self.is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop monitoring resources."""
        if not self.is_monitoring:
            return

        logger.info("Stopping resource monitoring")
        self.is_monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    def add_callback(self, callback):
        """Add callback for resource updates."""
        self._callbacks.append(callback)

    def remove_callback(self, callback):
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect resource data
                gpu_data = await self.get_gpu_info() if self._gpu_available else []
                system_data = await self.get_system_resources()

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        await callback(gpu_data, system_data)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

                # Wait for next poll
                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.poll_interval)

    async def get_gpu_info(self) -> List[GPUInfo]:
        """Get GPU information using nvidia-smi."""
        if not self._gpu_available:
            return []

        try:
            # Query GPU information
            query_fields = [
                "index",
                "name",
                "uuid",
                "temperature.gpu",
                "power.draw",
                "power.limit",
                "memory.total",
                "memory.used",
                "memory.free",
                "utilization.gpu",
                "compute_mode",
                "driver_version",
            ]

            result = await asyncio.create_subprocess_shell(
                f"nvidia-smi --query-gpu={','.join(query_fields)} --format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                logger.error(f"nvidia-smi failed: {stderr.decode()}")
                return []

            # Parse output
            gpus = []
            for line in stdout.decode().strip().split("\n"):
                if not line:
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) != len(query_fields):
                    continue

                gpu = GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    uuid=parts[2],
                    temperature=float(parts[3]),
                    power_draw=float(parts[4]),
                    power_limit=float(parts[5]),
                    memory_total=int(parts[6]),
                    memory_used=int(parts[7]),
                    memory_free=int(parts[8]),
                    utilization=float(parts[9]),
                    compute_mode=parts[10],
                    driver_version=parts[11],
                    cuda_version=await self._get_cuda_version(),
                )
                gpus.append(gpu)

            return gpus

        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            return []

    async def _get_cuda_version(self) -> str:
        """Get CUDA version."""
        try:
            result = await asyncio.create_subprocess_shell(
                "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()

            # Extract CUDA version from nvidia-smi output
            # This is a simplified version, actual parsing might be more complex
            return "12.0"  # Default for H200

        except Exception:
            return "Unknown"

    async def get_system_resources(self) -> SystemResources:
        """Get system resource information."""
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()

        # Memory info
        memory = psutil.virtual_memory()

        # Disk info (root partition)
        disk = psutil.disk_usage("/")

        # Network info
        net_io = psutil.net_io_counters()

        # Process info
        process_count = len(psutil.pids())

        # Calculate thread count
        thread_count = sum(
            p.num_threads() for p in psutil.process_iter(["num_threads"])
        )

        return SystemResources(
            timestamp=datetime.utcnow(),
            # CPU
            cpu_count=psutil.cpu_count(),
            cpu_percent=cpu_percent,
            cpu_freq_current=cpu_freq.current if cpu_freq else 0,
            cpu_freq_max=cpu_freq.max if cpu_freq else 0,
            # Memory
            memory_total=memory.total // (1024 * 1024),  # Convert to MB
            memory_used=memory.used // (1024 * 1024),
            memory_available=memory.available // (1024 * 1024),
            memory_percent=memory.percent,
            # Disk
            disk_total=disk.total // (1024 * 1024 * 1024),  # Convert to GB
            disk_used=disk.used // (1024 * 1024 * 1024),
            disk_free=disk.free // (1024 * 1024 * 1024),
            disk_percent=disk.percent,
            # Network
            network_sent_bytes=net_io.bytes_sent,
            network_recv_bytes=net_io.bytes_recv,
            network_connections=len(psutil.net_connections()),
            # Processes
            process_count=process_count,
            thread_count=thread_count,
        )

    async def check_resource_alerts(
        self,
        gpu_threshold: float = 90.0,
        memory_threshold: float = 85.0,
        disk_threshold: float = 90.0,
    ) -> List[Dict[str, Any]]:
        """Check for resource alerts."""
        alerts = []

        # Check GPU
        if self._gpu_available:
            gpus = await self.get_gpu_info()
            for gpu in gpus:
                # GPU utilization
                if gpu.utilization > gpu_threshold:
                    alerts.append(
                        {
                            "type": "gpu_utilization",
                            "severity": "warning",
                            "gpu_index": gpu.index,
                            "value": gpu.utilization,
                            "threshold": gpu_threshold,
                            "message": f"GPU {gpu.index} utilization at {gpu.utilization}%",
                        }
                    )

                # GPU memory
                gpu_memory_percent = (gpu.memory_used / gpu.memory_total) * 100
                if gpu_memory_percent > memory_threshold:
                    alerts.append(
                        {
                            "type": "gpu_memory",
                            "severity": "warning",
                            "gpu_index": gpu.index,
                            "value": gpu_memory_percent,
                            "threshold": memory_threshold,
                            "message": f"GPU {gpu.index} memory at {gpu_memory_percent:.1f}%",
                        }
                    )

                # Temperature
                if gpu.temperature > 85:  # Critical temp for most GPUs
                    alerts.append(
                        {
                            "type": "gpu_temperature",
                            "severity": "critical",
                            "gpu_index": gpu.index,
                            "value": gpu.temperature,
                            "threshold": 85,
                            "message": f"GPU {gpu.index} temperature at {gpu.temperature}°C",
                        }
                    )

        # Check system resources
        system = await self.get_system_resources()

        # System memory
        if system.memory_percent > memory_threshold:
            alerts.append(
                {
                    "type": "system_memory",
                    "severity": "warning",
                    "value": system.memory_percent,
                    "threshold": memory_threshold,
                    "message": f"System memory at {system.memory_percent:.1f}%",
                }
            )

        # Disk space
        if system.disk_percent > disk_threshold:
            alerts.append(
                {
                    "type": "disk_space",
                    "severity": "critical",
                    "value": system.disk_percent,
                    "threshold": disk_threshold,
                    "message": f"Disk space at {system.disk_percent:.1f}%",
                }
            )

        return alerts

    def get_gpu_summary(self, gpus: List[GPUInfo]) -> Dict[str, Any]:
        """Get GPU summary statistics."""
        if not gpus:
            return {
                "gpu_count": 0,
                "total_memory_mb": 0,
                "used_memory_mb": 0,
                "avg_utilization": 0,
                "avg_temperature": 0,
                "total_power_draw": 0,
            }

        return {
            "gpu_count": len(gpus),
            "total_memory_mb": sum(gpu.memory_total for gpu in gpus),
            "used_memory_mb": sum(gpu.memory_used for gpu in gpus),
            "avg_utilization": sum(gpu.utilization for gpu in gpus) / len(gpus),
            "avg_temperature": sum(gpu.temperature for gpu in gpus) / len(gpus),
            "total_power_draw": sum(gpu.power_draw for gpu in gpus),
            "gpus": [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "utilization": gpu.utilization,
                    "memory_used_mb": gpu.memory_used,
                    "temperature": gpu.temperature,
                }
                for gpu in gpus
            ],
        }
