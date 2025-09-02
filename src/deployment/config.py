"""
Deployment configuration models and constants
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime


class DeploymentMode(str, Enum):
    """Deployment mode enumeration"""
    SERVERLESS = "serverless"
    TIMED = "timed"
    BOTH = "both"


class GPUType(str, Enum):
    """Available GPU types"""
    H100 = "H100"
    A100 = "A100"
    RTX4090 = "RTX4090"
    RTX3090 = "RTX3090"
    V100 = "V100"


class DeploymentStatus(str, Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    PROVISIONING = "provisioning"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    TERMINATED = "terminated"


class EnvironmentType(str, Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class GPUConfig:
    """GPU configuration"""
    type: GPUType
    count: int = 1
    memory_gb: Optional[int] = None  # Minimum GPU memory required
    compute_capability: Optional[float] = None  # Minimum compute capability


@dataclass
class ResourceConfig:
    """Resource configuration"""
    cpu_cores: int = 4
    memory_gb: int = 16
    container_disk_gb: int = 20
    volume_size_gb: int = 50
    network_volume_size_gb: Optional[int] = 100  # For model storage


@dataclass
class ScalingConfig:
    """Auto-scaling configuration for serverless"""
    min_workers: int = 0
    max_workers: int = 3
    target_requests_per_second: int = 10
    max_requests_per_worker: int = 100
    scale_up_threshold: float = 0.8  # CPU/memory usage threshold
    scale_down_threshold: float = 0.2
    scale_up_delay_seconds: int = 30
    scale_down_delay_seconds: int = 300  # 5 minutes


@dataclass
class NetworkConfig:
    """Network configuration"""
    ports: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"port": 8000, "protocol": "http", "name": "api"},
        {"port": 8001, "protocol": "http", "name": "control"},
        {"port": 8002, "protocol": "http", "name": "metrics"},
        {"port": 9090, "protocol": "http", "name": "prometheus"}
    ])
    enable_public_ip: bool = True
    enable_private_networking: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_jwt_auth: bool = True
    enable_api_key_auth: bool = True
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    enable_tls: bool = True
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    enable_prometheus: bool = True
    enable_structured_logging: bool = True
    log_level: str = "INFO"
    enable_alerts: bool = True
    alert_webhook_url: Optional[str] = None
    metrics_retention_days: int = 30


@dataclass
class FlashBootConfig:
    """FlashBoot configuration for fast cold starts"""
    enabled: bool = True
    cache_models: bool = True
    preload_models: List[str] = field(default_factory=lambda: [
        "openai/clip-vit-base-patch32",
        "ultralytics/yolov8n"
    ])
    warm_start_timeout_seconds: int = 30
    health_check_interval_seconds: int = 10


@dataclass
class DeploymentConfig:
    """Complete deployment configuration"""
    # Basic settings
    name: str
    mode: DeploymentMode
    environment: EnvironmentType
    docker_image: str
    docker_tag: str = "latest"
    
    # Resource configuration
    gpu: GPUConfig = field(default_factory=lambda: GPUConfig(type=GPUType.H100))
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    
    # Scaling (for serverless)
    scaling: Optional[ScalingConfig] = field(default_factory=ScalingConfig)
    
    # Network configuration
    network: NetworkConfig = field(default_factory=NetworkConfig)
    
    # Security configuration
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Monitoring configuration
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # FlashBoot configuration
    flashboot: FlashBootConfig = field(default_factory=FlashBootConfig)
    
    # Environment variables
    env_vars: Dict[str, str] = field(default_factory=dict)
    
    # Advanced settings
    idle_timeout_minutes: int = 10
    max_runtime_hours: Optional[int] = None  # For timed deployments
    enable_auto_shutdown: bool = True
    enable_auto_restart: bool = True
    restart_policy: str = "on-failure"
    max_restart_attempts: int = 3
    
    # Deployment metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    def to_runpod_config(self) -> Dict[str, Any]:
        """Convert to RunPod API configuration"""
        if self.mode == DeploymentMode.SERVERLESS:
            return self._to_serverless_config()
        else:
            return self._to_timed_config()
    
    def _to_serverless_config(self) -> Dict[str, Any]:
        """Convert to serverless endpoint configuration"""
        return {
            "name": self.name,
            "docker_image": f"{self.docker_image}:{self.docker_tag}",
            "gpu_type": self.gpu.type.value,
            "gpu_count": self.gpu.count,
            "container_disk_size": self.resources.container_disk_gb,
            "volume_size": self.resources.volume_size_gb,
            "min_workers": self.scaling.min_workers,
            "max_workers": self.scaling.max_workers,
            "idle_timeout": self.idle_timeout_minutes * 60,  # Convert to seconds
            "flashboot": self.flashboot.enabled,
            "env": self._prepare_env_vars(),
            "scaling": {
                "type": "requests_per_second",
                "target": self.scaling.target_requests_per_second,
                "max_requests_per_worker": self.scaling.max_requests_per_worker
            }
        }
    
    def _to_timed_config(self) -> Dict[str, Any]:
        """Convert to timed pod configuration"""
        ports = ",".join([
            f"{p['port']}/{p['protocol']}" 
            for p in self.network.ports
        ])
        
        return {
            "name": self.name,
            "image_name": f"{self.docker_image}:{self.docker_tag}",
            "gpu_type_id": self.gpu.type.value.lower(),
            "gpu_count": self.gpu.count,
            "container_disk_in_gb": self.resources.container_disk_gb,
            "volume_in_gb": self.resources.volume_size_gb,
            "ports": ports,
            "env": self._prepare_env_vars(),
            "docker_args": f"--restart {self.restart_policy}"
        }
    
    def _prepare_env_vars(self) -> Dict[str, str]:
        """Prepare environment variables for deployment"""
        env = self.env_vars.copy()
        
        # Add deployment-specific variables
        env.update({
            "DEPLOYMENT_MODE": self.mode.value,
            "DEPLOYMENT_ENVIRONMENT": self.environment.value,
            "ENABLE_AUTO_SHUTDOWN": str(self.enable_auto_shutdown).lower(),
            "IDLE_TIMEOUT_MINUTES": str(self.idle_timeout_minutes),
            "LOG_LEVEL": self.monitoring.log_level,
            "ENABLE_PROMETHEUS": str(self.monitoring.enable_prometheus).lower(),
            "ENABLE_JWT_AUTH": str(self.security.enable_jwt_auth).lower(),
            "ENABLE_RATE_LIMITING": str(self.security.enable_rate_limiting).lower(),
            "MAX_REQUESTS_PER_MINUTE": str(self.security.max_requests_per_minute),
        })
        
        # Add FlashBoot configuration
        if self.flashboot.enabled:
            env.update({
                "FLASHBOOT_ENABLED": "true",
                "MODEL_CACHE_ENABLED": str(self.flashboot.cache_models).lower(),
                "PRELOAD_MODELS": ",".join(self.flashboot.preload_models)
            })
        
        return env


# Preset configurations for different environments
DEPLOYMENT_PRESETS = {
    "production": DeploymentConfig(
        name="h200-mug-positioning-prod",
        mode=DeploymentMode.SERVERLESS,
        environment=EnvironmentType.PRODUCTION,
        docker_image="h200-mug-positioning",
        gpu=GPUConfig(type=GPUType.H100, count=1),
        resources=ResourceConfig(
            cpu_cores=8,
            memory_gb=32,
            container_disk_gb=50,
            volume_size_gb=100,
            network_volume_size_gb=200
        ),
        scaling=ScalingConfig(
            min_workers=1,
            max_workers=10,
            target_requests_per_second=20
        ),
        security=SecurityConfig(
            enable_jwt_auth=True,
            enable_rate_limiting=True,
            max_requests_per_minute=100
        ),
        idle_timeout_minutes=15
    ),
    "staging": DeploymentConfig(
        name="h200-mug-positioning-staging",
        mode=DeploymentMode.TIMED,
        environment=EnvironmentType.STAGING,
        docker_image="h200-mug-positioning",
        gpu=GPUConfig(type=GPUType.A100, count=1),
        resources=ResourceConfig(
            cpu_cores=4,
            memory_gb=16,
            container_disk_gb=30,
            volume_size_gb=50
        ),
        idle_timeout_minutes=30,
        max_runtime_hours=24
    ),
    "development": DeploymentConfig(
        name="h200-mug-positioning-dev",
        mode=DeploymentMode.TIMED,
        environment=EnvironmentType.DEVELOPMENT,
        docker_image="h200-mug-positioning",
        gpu=GPUConfig(type=GPUType.RTX3090, count=1),
        resources=ResourceConfig(
            cpu_cores=2,
            memory_gb=8,
            container_disk_gb=20,
            volume_size_gb=30
        ),
        security=SecurityConfig(
            enable_jwt_auth=False,
            enable_rate_limiting=False
        ),
        monitoring=MonitoringConfig(
            log_level="DEBUG",
            enable_alerts=False
        ),
        idle_timeout_minutes=60
    )
}