"""
Core RunPod deployment functionality
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .client import RunPodClient, RunPodAPIError
from .config import DeploymentConfig, DeploymentMode, DeploymentStatus
from .validator import DeploymentValidator
from .volume import VolumeManager

logger = logging.getLogger(__name__)


class RunPodDeployer:
    """Handles deployment operations to RunPod"""
    
    def __init__(self, client: RunPodClient, volume_manager: Optional[VolumeManager] = None):
        """Initialize deployer"""
        self.client = client
        self.volume_manager = volume_manager or VolumeManager(client)
        self.validator = DeploymentValidator()
    
    async def deploy(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy to RunPod based on configuration"""
        # Validate configuration
        validation_result = await self.validator.validate_config(config)
        if not validation_result["valid"]:
            raise ValueError(f"Invalid deployment configuration: {validation_result['errors']}")
        
        # Deploy based on mode
        if config.mode == DeploymentMode.SERVERLESS:
            return await self.deploy_serverless(config)
        elif config.mode == DeploymentMode.TIMED:
            return await self.deploy_timed(config)
        elif config.mode == DeploymentMode.BOTH:
            return await self.deploy_both(config)
        else:
            raise ValueError(f"Unknown deployment mode: {config.mode}")
    
    async def deploy_serverless(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy serverless endpoint"""
        logger.info(f"Deploying serverless endpoint: {config.name}")
        
        try:
            # Check for existing endpoint
            endpoints = await self.client.list_endpoints()
            existing = next(
                (ep for ep in endpoints if ep["name"] == config.name),
                None
            )
            
            # Prepare configuration
            runpod_config = config.to_runpod_config()
            
            # Create or update endpoint
            if existing:
                logger.info(f"Updating existing endpoint: {existing['id']}")
                result = await self.client.update_endpoint(
                    existing["id"],
                    runpod_config
                )
                deployment_id = existing["id"]
            else:
                logger.info("Creating new serverless endpoint")
                result = await self.client.create_endpoint(runpod_config)
                deployment_id = result["id"]
            
            # Wait for endpoint to be ready
            ready = await self.client.wait_for_endpoint_ready(
                deployment_id,
                timeout=300,
                callback=self._deployment_progress_callback
            )
            
            if not ready:
                raise RuntimeError(f"Endpoint {deployment_id} failed to become ready")
            
            # Get final status
            endpoint = await self.client.get_endpoint(deployment_id)
            
            return {
                "id": deployment_id,
                "mode": "serverless",
                "status": DeploymentStatus.RUNNING,
                "endpoint_url": endpoint.get("endpoint", ""),
                "workers": endpoint.get("workers", {}),
                "created_at": datetime.utcnow().isoformat(),
                "config": config.dict()
            }
            
        except RunPodAPIError as e:
            logger.error(f"RunPod API error during serverless deployment: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to deploy serverless endpoint: {e}")
            raise
    
    async def deploy_timed(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy timed GPU pod"""
        logger.info(f"Deploying timed GPU pod: {config.name}")
        
        try:
            # Check for existing pod
            pods = await self.client.list_pods()
            existing = next(
                (pod for pod in pods if pod["name"] == config.name),
                None
            )
            
            # Handle existing pod
            if existing:
                logger.info(f"Found existing pod: {existing['id']}")
                
                # Stop if running
                if existing["status"].lower() in ["running", "starting"]:
                    logger.info("Stopping existing pod")
                    await self.client.stop_pod(existing["id"])
                    await asyncio.sleep(5)  # Wait for stop
                
                # Terminate old pod
                logger.info("Terminating existing pod")
                await self.client.terminate_pod(existing["id"])
                await asyncio.sleep(5)  # Wait for termination
            
            # Prepare configuration
            runpod_config = config.to_runpod_config()
            
            # Attach network volume if configured
            volume_id = None
            if config.resources.network_volume_size_gb:
                volume_id = await self.volume_manager.ensure_volume(
                    name=f"{config.name}-models",
                    size_gb=config.resources.network_volume_size_gb,
                    labels={"deployment": config.name, "type": "models"}
                )
                runpod_config["volume_mount_path"] = "/models"
                runpod_config["volume_id"] = volume_id
            
            # Create new pod
            logger.info("Creating new timed GPU pod")
            result = await self.client.create_pod(runpod_config)
            deployment_id = result["id"]
            
            # Wait for pod to be ready
            ready = await self.client.wait_for_pod_ready(
                deployment_id,
                timeout=300,
                callback=self._deployment_progress_callback
            )
            
            if not ready:
                raise RuntimeError(f"Pod {deployment_id} failed to become ready")
            
            # Get final status
            pod = await self.client.get_pod(deployment_id)
            
            return {
                "id": deployment_id,
                "mode": "timed",
                "status": DeploymentStatus.RUNNING,
                "pod_ip": pod.get("ip", ""),
                "ssh_command": pod.get("ssh_command", ""),
                "ports": pod.get("ports", {}),
                "volume_id": volume_id,
                "created_at": datetime.utcnow().isoformat(),
                "config": config.dict()
            }
            
        except RunPodAPIError as e:
            logger.error(f"RunPod API error during timed deployment: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to deploy timed pod: {e}")
            raise
    
    async def deploy_both(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy both serverless and timed instances"""
        logger.info(f"Deploying both serverless and timed instances for: {config.name}")
        
        results = {}
        errors = []
        
        # Deploy serverless
        serverless_config = config.copy()
        serverless_config.mode = DeploymentMode.SERVERLESS
        serverless_config.name = f"{config.name}-serverless"
        
        try:
            results["serverless"] = await self.deploy_serverless(serverless_config)
        except Exception as e:
            logger.error(f"Failed to deploy serverless: {e}")
            errors.append(("serverless", str(e)))
        
        # Deploy timed
        timed_config = config.copy()
        timed_config.mode = DeploymentMode.TIMED
        timed_config.name = f"{config.name}-timed"
        
        try:
            results["timed"] = await self.deploy_timed(timed_config)
        except Exception as e:
            logger.error(f"Failed to deploy timed: {e}")
            errors.append(("timed", str(e)))
        
        # Check if any deployments succeeded
        if not results:
            raise RuntimeError(f"All deployments failed: {errors}")
        
        return {
            "deployments": results,
            "errors": errors,
            "status": DeploymentStatus.RUNNING if not errors else DeploymentStatus.PARTIALLY_FAILED,
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def update_deployment(self, deployment_id: str, mode: DeploymentMode,
                               updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing deployment"""
        logger.info(f"Updating {mode} deployment: {deployment_id}")
        
        try:
            if mode == DeploymentMode.SERVERLESS:
                result = await self.client.update_endpoint(deployment_id, updates)
            else:
                # For pods, we need to stop and recreate with new config
                pod = await self.client.get_pod(deployment_id)
                
                # Stop pod
                await self.client.stop_pod(deployment_id)
                await asyncio.sleep(5)
                
                # Update configuration
                # Note: RunPod doesn't support in-place pod updates,
                # so this is a placeholder for future functionality
                result = {"message": "Pod updates require recreation"}
            
            return {
                "id": deployment_id,
                "mode": mode,
                "updates": updates,
                "result": result,
                "updated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to update deployment: {e}")
            raise
    
    async def stop_deployment(self, deployment_id: str, mode: DeploymentMode) -> Dict[str, Any]:
        """Stop deployment"""
        logger.info(f"Stopping {mode} deployment: {deployment_id}")
        
        try:
            if mode == DeploymentMode.SERVERLESS:
                # Serverless endpoints can't be stopped, only scaled to 0
                result = await self.client.update_endpoint(
                    deployment_id,
                    {"min_workers": 0, "max_workers": 0}
                )
            else:
                result = await self.client.stop_pod(deployment_id)
            
            return {
                "id": deployment_id,
                "mode": mode,
                "status": DeploymentStatus.STOPPED,
                "stopped_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to stop deployment: {e}")
            raise
    
    async def start_deployment(self, deployment_id: str, mode: DeploymentMode,
                              config: Optional[DeploymentConfig] = None) -> Dict[str, Any]:
        """Start stopped deployment"""
        logger.info(f"Starting {mode} deployment: {deployment_id}")
        
        try:
            if mode == DeploymentMode.SERVERLESS:
                # Scale serverless endpoint back up
                scaling = config.scaling if config else {"min_workers": 0, "max_workers": 3}
                result = await self.client.update_endpoint(
                    deployment_id,
                    {
                        "min_workers": scaling.min_workers,
                        "max_workers": scaling.max_workers
                    }
                )
            else:
                result = await self.client.start_pod(deployment_id)
            
            # Wait for ready
            if mode == DeploymentMode.SERVERLESS:
                ready = await self.client.wait_for_endpoint_ready(deployment_id)
            else:
                ready = await self.client.wait_for_pod_ready(deployment_id)
            
            return {
                "id": deployment_id,
                "mode": mode,
                "status": DeploymentStatus.RUNNING if ready else DeploymentStatus.FAILED,
                "started_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to start deployment: {e}")
            raise
    
    async def terminate_deployment(self, deployment_id: str, mode: DeploymentMode) -> Dict[str, Any]:
        """Terminate deployment permanently"""
        logger.info(f"Terminating {mode} deployment: {deployment_id}")
        
        try:
            if mode == DeploymentMode.SERVERLESS:
                result = await self.client.delete_endpoint(deployment_id)
            else:
                result = await self.client.terminate_pod(deployment_id)
            
            return {
                "id": deployment_id,
                "mode": mode,
                "status": DeploymentStatus.TERMINATED,
                "terminated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to terminate deployment: {e}")
            raise
    
    async def get_deployment_status(self, deployment_id: str, mode: DeploymentMode) -> Dict[str, Any]:
        """Get deployment status"""
        try:
            if mode == DeploymentMode.SERVERLESS:
                endpoint = await self.client.get_endpoint(deployment_id)
                return {
                    "id": deployment_id,
                    "mode": mode,
                    "status": self._map_endpoint_status(endpoint.get("status", "")),
                    "endpoint_url": endpoint.get("endpoint", ""),
                    "workers": endpoint.get("workers", {}),
                    "metrics": {
                        "requests_total": endpoint.get("requests_total", 0),
                        "requests_failed": endpoint.get("requests_failed", 0),
                        "average_latency_ms": endpoint.get("average_latency_ms", 0)
                    }
                }
            else:
                pod = await self.client.get_pod(deployment_id)
                return {
                    "id": deployment_id,
                    "mode": mode,
                    "status": self._map_pod_status(pod.get("status", "")),
                    "pod_ip": pod.get("ip", ""),
                    "gpu_type": pod.get("gpu_type", ""),
                    "runtime_seconds": pod.get("runtime", 0),
                    "metrics": {
                        "cpu_usage_percent": pod.get("cpu_usage", 0),
                        "memory_usage_mb": pod.get("memory_usage", 0),
                        "gpu_usage_percent": pod.get("gpu_usage", 0),
                        "gpu_memory_usage_mb": pod.get("gpu_memory_usage", 0)
                    }
                }
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {
                "id": deployment_id,
                "mode": mode,
                "status": DeploymentStatus.FAILED,
                "error": str(e)
            }
    
    async def get_deployment_logs(self, deployment_id: str, mode: DeploymentMode,
                                 lines: int = 100) -> str:
        """Get deployment logs"""
        try:
            if mode == DeploymentMode.TIMED:
                return await self.client.get_pod_logs(deployment_id, lines)
            else:
                # Serverless endpoints don't have direct log access
                return "Serverless endpoint logs are available through RunPod dashboard"
        except Exception as e:
            logger.error(f"Failed to get deployment logs: {e}")
            return f"Error fetching logs: {e}"
    
    def _deployment_progress_callback(self, status: Dict[str, Any]):
        """Callback for deployment progress updates"""
        logger.info(f"Deployment progress: {status}")
    
    def _map_endpoint_status(self, status: str) -> DeploymentStatus:
        """Map RunPod endpoint status to DeploymentStatus"""
        status_map = {
            "initializing": DeploymentStatus.PROVISIONING,
            "provisioning": DeploymentStatus.PROVISIONING,
            "starting": DeploymentStatus.STARTING,
            "running": DeploymentStatus.RUNNING,
            "stopping": DeploymentStatus.STOPPING,
            "stopped": DeploymentStatus.STOPPED,
            "failed": DeploymentStatus.FAILED,
            "terminated": DeploymentStatus.TERMINATED
        }
        return status_map.get(status.lower(), DeploymentStatus.PENDING)
    
    def _map_pod_status(self, status: str) -> DeploymentStatus:
        """Map RunPod pod status to DeploymentStatus"""
        status_map = {
            "pending": DeploymentStatus.PENDING,
            "provisioning": DeploymentStatus.PROVISIONING,
            "starting": DeploymentStatus.STARTING,
            "running": DeploymentStatus.RUNNING,
            "stopping": DeploymentStatus.STOPPING,
            "stopped": DeploymentStatus.STOPPED,
            "failed": DeploymentStatus.FAILED,
            "terminated": DeploymentStatus.TERMINATED
        }
        return status_map.get(status.lower(), DeploymentStatus.PENDING)