"""
Deployment validation and rollback capabilities
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from .config import DeploymentConfig, DeploymentMode, GPUType
from .client import RunPodClient

logger = logging.getLogger(__name__)


class DeploymentValidator:
    """Validates deployment configurations and performs pre-deployment checks"""
    
    def __init__(self):
        """Initialize validator"""
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules"""
        return {
            "gpu_limits": {
                GPUType.H100: {"max_count": 8, "min_memory_gb": 80},
                GPUType.A100: {"max_count": 8, "min_memory_gb": 40},
                GPUType.RTX4090: {"max_count": 4, "min_memory_gb": 24},
                GPUType.RTX3090: {"max_count": 4, "min_memory_gb": 24},
                GPUType.V100: {"max_count": 8, "min_memory_gb": 16}
            },
            "resource_limits": {
                "max_cpu_cores": 128,
                "max_memory_gb": 512,
                "max_container_disk_gb": 1000,
                "max_volume_size_gb": 1000,
                "max_network_volume_size_gb": 5000
            },
            "scaling_limits": {
                "max_workers": 100,
                "max_requests_per_second": 1000,
                "max_requests_per_worker": 1000
            },
            "required_env_vars": [
                "MONGODB_ATLAS_URI",
                "REDIS_HOST",
                "REDIS_PASSWORD",
                "JWT_SECRET"
            ]
        }
    
    async def validate_config(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Validate deployment configuration"""
        logger.info(f"Validating deployment configuration for {config.name}")
        
        errors = []
        warnings = []
        
        # Validate GPU configuration
        gpu_errors = self._validate_gpu_config(config.gpu)
        errors.extend(gpu_errors)
        
        # Validate resource configuration
        resource_errors = self._validate_resource_config(config.resources)
        errors.extend(resource_errors)
        
        # Validate scaling configuration (for serverless)
        if config.mode == DeploymentMode.SERVERLESS and config.scaling:
            scaling_errors = self._validate_scaling_config(config.scaling)
            errors.extend(scaling_errors)
        
        # Validate environment variables
        env_errors = self._validate_env_vars(config.env_vars)
        errors.extend(env_errors)
        
        # Validate network configuration
        network_warnings = self._validate_network_config(config.network)
        warnings.extend(network_warnings)
        
        # Validate security configuration
        security_warnings = self._validate_security_config(config.security)
        warnings.extend(security_warnings)
        
        # Mode-specific validation
        if config.mode == DeploymentMode.SERVERLESS:
            serverless_errors = self._validate_serverless_config(config)
            errors.extend(serverless_errors)
        elif config.mode == DeploymentMode.TIMED:
            timed_errors = self._validate_timed_config(config)
            errors.extend(timed_errors)
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "validated_at": datetime.utcnow().isoformat()
        }
    
    def _validate_gpu_config(self, gpu_config) -> List[str]:
        """Validate GPU configuration"""
        errors = []
        
        gpu_limits = self.validation_rules["gpu_limits"].get(gpu_config.type)
        if not gpu_limits:
            errors.append(f"Unknown GPU type: {gpu_config.type}")
            return errors
        
        if gpu_config.count > gpu_limits["max_count"]:
            errors.append(f"GPU count {gpu_config.count} exceeds maximum {gpu_limits['max_count']} for {gpu_config.type}")
        
        if gpu_config.count < 1:
            errors.append("GPU count must be at least 1")
        
        if gpu_config.memory_gb and gpu_config.memory_gb < gpu_limits["min_memory_gb"]:
            errors.append(f"GPU memory requirement {gpu_config.memory_gb}GB is less than minimum {gpu_limits['min_memory_gb']}GB for {gpu_config.type}")
        
        return errors
    
    def _validate_resource_config(self, resource_config) -> List[str]:
        """Validate resource configuration"""
        errors = []
        limits = self.validation_rules["resource_limits"]
        
        if resource_config.cpu_cores > limits["max_cpu_cores"]:
            errors.append(f"CPU cores {resource_config.cpu_cores} exceeds maximum {limits['max_cpu_cores']}")
        
        if resource_config.memory_gb > limits["max_memory_gb"]:
            errors.append(f"Memory {resource_config.memory_gb}GB exceeds maximum {limits['max_memory_gb']}GB")
        
        if resource_config.container_disk_gb > limits["max_container_disk_gb"]:
            errors.append(f"Container disk {resource_config.container_disk_gb}GB exceeds maximum {limits['max_container_disk_gb']}GB")
        
        if resource_config.volume_size_gb > limits["max_volume_size_gb"]:
            errors.append(f"Volume size {resource_config.volume_size_gb}GB exceeds maximum {limits['max_volume_size_gb']}GB")
        
        if resource_config.network_volume_size_gb and resource_config.network_volume_size_gb > limits["max_network_volume_size_gb"]:
            errors.append(f"Network volume size {resource_config.network_volume_size_gb}GB exceeds maximum {limits['max_network_volume_size_gb']}GB")
        
        # Minimum requirements
        if resource_config.cpu_cores < 1:
            errors.append("CPU cores must be at least 1")
        
        if resource_config.memory_gb < 1:
            errors.append("Memory must be at least 1GB")
        
        if resource_config.container_disk_gb < 10:
            errors.append("Container disk must be at least 10GB")
        
        return errors
    
    def _validate_scaling_config(self, scaling_config) -> List[str]:
        """Validate scaling configuration"""
        errors = []
        limits = self.validation_rules["scaling_limits"]
        
        if scaling_config.max_workers > limits["max_workers"]:
            errors.append(f"Max workers {scaling_config.max_workers} exceeds maximum {limits['max_workers']}")
        
        if scaling_config.min_workers > scaling_config.max_workers:
            errors.append("Min workers cannot be greater than max workers")
        
        if scaling_config.min_workers < 0:
            errors.append("Min workers cannot be negative")
        
        if scaling_config.target_requests_per_second > limits["max_requests_per_second"]:
            errors.append(f"Target RPS {scaling_config.target_requests_per_second} exceeds maximum {limits['max_requests_per_second']}")
        
        if scaling_config.max_requests_per_worker > limits["max_requests_per_worker"]:
            errors.append(f"Max requests per worker {scaling_config.max_requests_per_worker} exceeds maximum {limits['max_requests_per_worker']}")
        
        if scaling_config.scale_up_threshold >= scaling_config.scale_down_threshold:
            errors.append("Scale up threshold must be less than scale down threshold")
        
        return errors
    
    def _validate_env_vars(self, env_vars: Dict[str, str]) -> List[str]:
        """Validate environment variables"""
        errors = []
        
        # Check required environment variables
        for required_var in self.validation_rules["required_env_vars"]:
            if required_var not in env_vars:
                errors.append(f"Missing required environment variable: {required_var}")
        
        # Check for empty values
        for key, value in env_vars.items():
            if value == "":
                errors.append(f"Environment variable {key} has empty value")
        
        # Check for sensitive data patterns
        sensitive_patterns = ["password", "secret", "key", "token"]
        for key in env_vars:
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in sensitive_patterns):
                if len(env_vars[key]) < 8:
                    errors.append(f"Sensitive environment variable {key} appears too short")
        
        return errors
    
    def _validate_network_config(self, network_config) -> List[str]:
        """Validate network configuration"""
        warnings = []
        
        # Check ports
        reserved_ports = {22, 80, 443}  # SSH, HTTP, HTTPS
        for port_config in network_config.ports:
            port = port_config["port"]
            if port in reserved_ports:
                warnings.append(f"Port {port} is commonly reserved, consider using a different port")
            
            if port < 1024:
                warnings.append(f"Port {port} is in privileged range, may require special permissions")
        
        # Check allowed origins
        if "*" in network_config.allowed_origins:
            warnings.append("CORS is configured to allow all origins, consider restricting for production")
        
        return warnings
    
    def _validate_security_config(self, security_config) -> List[str]:
        """Validate security configuration"""
        warnings = []
        
        if not security_config.enable_jwt_auth and not security_config.enable_api_key_auth:
            warnings.append("No authentication method enabled, API will be publicly accessible")
        
        if not security_config.enable_rate_limiting:
            warnings.append("Rate limiting is disabled, API may be vulnerable to abuse")
        
        if security_config.enable_tls and not (security_config.tls_cert_path and security_config.tls_key_path):
            warnings.append("TLS is enabled but certificate paths are not provided")
        
        if security_config.max_requests_per_minute > 1000:
            warnings.append(f"Rate limit of {security_config.max_requests_per_minute} requests/minute is very high")
        
        return warnings
    
    def _validate_serverless_config(self, config: DeploymentConfig) -> List[str]:
        """Validate serverless-specific configuration"""
        errors = []
        
        if not config.flashboot.enabled:
            errors.append("FlashBoot must be enabled for serverless deployments")
        
        if config.idle_timeout_minutes < 1:
            errors.append("Idle timeout must be at least 1 minute for serverless")
        
        if config.idle_timeout_minutes > 60:
            errors.append("Idle timeout cannot exceed 60 minutes for serverless")
        
        return errors
    
    def _validate_timed_config(self, config: DeploymentConfig) -> List[str]:
        """Validate timed deployment configuration"""
        errors = []
        
        if config.max_runtime_hours and config.max_runtime_hours > 168:  # 1 week
            errors.append("Max runtime cannot exceed 168 hours (1 week)")
        
        if config.max_runtime_hours and config.max_runtime_hours < 1:
            errors.append("Max runtime must be at least 1 hour")
        
        return errors
    
    async def validate_deployment_health(self, client: RunPodClient, 
                                       deployment_id: str,
                                       mode: DeploymentMode,
                                       checks: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate deployment health with custom checks"""
        if checks is None:
            checks = ["status", "connectivity", "resources", "performance"]
        
        logger.info(f"Validating deployment health for {deployment_id}")
        
        results = {
            "deployment_id": deployment_id,
            "mode": mode.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "healthy": True
        }
        
        # Status check
        if "status" in checks:
            try:
                if mode == DeploymentMode.SERVERLESS:
                    endpoint = await client.get_endpoint(deployment_id)
                    status_healthy = endpoint.get("status", "").lower() == "running"
                    workers = endpoint.get("workers", {})
                    has_workers = workers.get("running", 0) > 0
                    results["checks"]["status"] = {
                        "passed": status_healthy and has_workers,
                        "details": {
                            "status": endpoint.get("status"),
                            "workers": workers
                        }
                    }
                else:
                    pod = await client.get_pod(deployment_id)
                    status_healthy = pod.get("status", "").lower() == "running"
                    results["checks"]["status"] = {
                        "passed": status_healthy,
                        "details": {
                            "status": pod.get("status")
                        }
                    }
            except Exception as e:
                results["checks"]["status"] = {
                    "passed": False,
                    "error": str(e)
                }
                results["healthy"] = False
        
        # Connectivity check
        if "connectivity" in checks and mode == DeploymentMode.TIMED:
            try:
                pod = await client.get_pod(deployment_id)
                has_ip = bool(pod.get("ip"))
                results["checks"]["connectivity"] = {
                    "passed": has_ip,
                    "details": {
                        "ip": pod.get("ip", "none")
                    }
                }
            except Exception as e:
                results["checks"]["connectivity"] = {
                    "passed": False,
                    "error": str(e)
                }
                results["healthy"] = False
        
        # Resource utilization check
        if "resources" in checks and mode == DeploymentMode.TIMED:
            try:
                metrics = await client.get_pod_metrics(deployment_id)
                cpu_usage = metrics.get("cpu_usage_percent", 0)
                memory_usage = metrics.get("memory_usage_percent", 0)
                gpu_usage = metrics.get("gpu_usage_percent", 0)
                
                resources_healthy = (
                    cpu_usage < 90 and
                    memory_usage < 90 and
                    gpu_usage < 95
                )
                
                results["checks"]["resources"] = {
                    "passed": resources_healthy,
                    "details": {
                        "cpu_usage_percent": cpu_usage,
                        "memory_usage_percent": memory_usage,
                        "gpu_usage_percent": gpu_usage
                    }
                }
            except Exception as e:
                results["checks"]["resources"] = {
                    "passed": False,
                    "error": str(e)
                }
        
        # Performance check (for serverless)
        if "performance" in checks and mode == DeploymentMode.SERVERLESS:
            try:
                metrics = await client.get_endpoint_metrics(deployment_id)
                avg_latency = metrics.get("average_latency_ms", 0)
                error_rate = self._calculate_error_rate(metrics)
                
                performance_healthy = (
                    avg_latency < 1000 and  # Less than 1 second
                    error_rate < 0.05  # Less than 5% errors
                )
                
                results["checks"]["performance"] = {
                    "passed": performance_healthy,
                    "details": {
                        "average_latency_ms": avg_latency,
                        "error_rate": error_rate
                    }
                }
            except Exception as e:
                results["checks"]["performance"] = {
                    "passed": False,
                    "error": str(e)
                }
        
        # Overall health
        results["healthy"] = all(
            check.get("passed", False) 
            for check in results["checks"].values()
        )
        
        return results
    
    def _calculate_error_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate error rate from metrics"""
        total = metrics.get("requests_total", 0)
        failed = metrics.get("requests_failed", 0)
        
        if total == 0:
            return 0.0
        
        return failed / total
    
    async def validate_rollback_safety(self, current_config: DeploymentConfig,
                                     target_config: DeploymentConfig) -> Dict[str, Any]:
        """Validate if rollback is safe"""
        logger.info("Validating rollback safety")
        
        warnings = []
        errors = []
        
        # Check configuration compatibility
        if current_config.gpu.type != target_config.gpu.type:
            warnings.append(f"GPU type will change from {current_config.gpu.type} to {target_config.gpu.type}")
        
        if current_config.gpu.count != target_config.gpu.count:
            warnings.append(f"GPU count will change from {current_config.gpu.count} to {target_config.gpu.count}")
        
        # Check resource changes
        if target_config.resources.memory_gb < current_config.resources.memory_gb:
            warnings.append(f"Memory will decrease from {current_config.resources.memory_gb}GB to {target_config.resources.memory_gb}GB")
        
        if target_config.resources.cpu_cores < current_config.resources.cpu_cores:
            warnings.append(f"CPU cores will decrease from {current_config.resources.cpu_cores} to {target_config.resources.cpu_cores}")
        
        # Check for missing environment variables
        current_env_keys = set(current_config.env_vars.keys())
        target_env_keys = set(target_config.env_vars.keys())
        missing_vars = current_env_keys - target_env_keys
        
        if missing_vars:
            errors.append(f"Target configuration missing environment variables: {missing_vars}")
        
        # Check scaling changes (for serverless)
        if current_config.mode == DeploymentMode.SERVERLESS and target_config.mode == DeploymentMode.SERVERLESS:
            if target_config.scaling.max_workers < current_config.scaling.min_workers:
                errors.append("Target max workers is less than current min workers")
        
        return {
            "safe": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "validated_at": datetime.utcnow().isoformat()
        }