"""
Network volume management for model storage
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .client import RunPodClient

logger = logging.getLogger(__name__)


class VolumeManager:
    """Manages RunPod network volumes for persistent model storage"""
    
    def __init__(self, client: RunPodClient):
        """Initialize volume manager"""
        self.client = client
    
    async def ensure_volume(self, name: str, size_gb: int, 
                          region: str = "us-east-1",
                          labels: Optional[Dict[str, str]] = None) -> str:
        """Ensure volume exists with specified size"""
        logger.info(f"Ensuring volume {name} with size {size_gb}GB")
        
        try:
            # Check if volume already exists
            volumes = await self.client.list_volumes()
            existing = next(
                (vol for vol in volumes if vol["name"] == name),
                None
            )
            
            if existing:
                volume_id = existing["id"]
                current_size = existing.get("size", 0)
                
                # Check if resize is needed
                if current_size < size_gb:
                    logger.info(f"Resizing volume {volume_id} from {current_size}GB to {size_gb}GB")
                    await self.client.update_volume(volume_id, size_gb)
                else:
                    logger.info(f"Volume {volume_id} already exists with adequate size")
                
                return volume_id
            
            else:
                # Create new volume
                logger.info(f"Creating new volume {name}")
                result = await self.client.create_volume(name, size_gb, region)
                return result["id"]
                
        except Exception as e:
            logger.error(f"Failed to ensure volume: {e}")
            raise
    
    async def list_volumes(self, labels: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """List volumes with optional label filtering"""
        volumes = await self.client.list_volumes()
        
        if labels:
            # Filter by labels if RunPod supports it
            # This is a placeholder as RunPod API might not support labels yet
            filtered = []
            for vol in volumes:
                vol_labels = vol.get("labels", {})
                if all(vol_labels.get(k) == v for k, v in labels.items()):
                    filtered.append(vol)
            return filtered
        
        return volumes
    
    async def get_volume_info(self, volume_id: str) -> Dict[str, Any]:
        """Get detailed volume information"""
        volume = await self.client.get_volume(volume_id)
        
        # Add computed fields
        volume["usage_percent"] = self._calculate_usage_percent(volume)
        volume["last_accessed"] = volume.get("last_accessed", "unknown")
        
        return volume
    
    async def cleanup_unused_volumes(self, days_unused: int = 30) -> List[str]:
        """Clean up volumes not used for specified days"""
        logger.info(f"Cleaning up volumes unused for {days_unused} days")
        
        cleaned = []
        volumes = await self.client.list_volumes()
        
        for volume in volumes:
            # Check if volume is attached to any pod
            if volume.get("attached_to"):
                continue
            
            # Check last access time
            last_accessed = volume.get("last_accessed")
            if not last_accessed:
                continue
            
            # Parse last accessed time and check age
            try:
                last_accessed_dt = datetime.fromisoformat(last_accessed)
                age_days = (datetime.utcnow() - last_accessed_dt).days
                
                if age_days > days_unused:
                    logger.info(f"Deleting unused volume {volume['id']} (unused for {age_days} days)")
                    await self.client.delete_volume(volume["id"])
                    cleaned.append(volume["id"])
                    
            except Exception as e:
                logger.error(f"Error checking volume {volume['id']}: {e}")
        
        return cleaned
    
    async def attach_to_deployment(self, volume_id: str, deployment_id: str,
                                  mount_path: str = "/models") -> Dict[str, Any]:
        """Attach volume to deployment"""
        logger.info(f"Attaching volume {volume_id} to deployment {deployment_id}")
        
        try:
            result = await self.client.attach_volume(
                deployment_id,
                volume_id,
                mount_path
            )
            
            return {
                "volume_id": volume_id,
                "deployment_id": deployment_id,
                "mount_path": mount_path,
                "attached_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to attach volume: {e}")
            raise
    
    async def detach_from_deployment(self, volume_id: str, deployment_id: str) -> Dict[str, Any]:
        """Detach volume from deployment"""
        logger.info(f"Detaching volume {volume_id} from deployment {deployment_id}")
        
        try:
            result = await self.client.detach_volume(deployment_id, volume_id)
            
            return {
                "volume_id": volume_id,
                "deployment_id": deployment_id,
                "detached_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to detach volume: {e}")
            raise
    
    async def sync_models_to_volume(self, volume_id: str, models: List[str]) -> Dict[str, Any]:
        """Sync models to volume (requires running pod with volume attached)"""
        # This would typically be implemented by:
        # 1. Ensuring a temporary pod with the volume attached
        # 2. Running sync commands to download models
        # 3. Verifying model integrity
        # 4. Updating volume metadata
        
        logger.info(f"Model sync to volume {volume_id} would sync: {models}")
        
        return {
            "volume_id": volume_id,
            "models": models,
            "status": "sync_required",
            "message": "Model sync requires implementation with running pod"
        }
    
    async def get_volume_metrics(self, volume_id: str) -> Dict[str, Any]:
        """Get volume usage metrics"""
        volume = await self.get_volume_info(volume_id)
        
        return {
            "volume_id": volume_id,
            "size_gb": volume.get("size", 0),
            "used_gb": volume.get("used", 0),
            "usage_percent": self._calculate_usage_percent(volume),
            "iops": volume.get("iops", 0),
            "throughput_mbps": volume.get("throughput", 0),
            "attached_deployments": volume.get("attached_to", [])
        }
    
    def _calculate_usage_percent(self, volume: Dict[str, Any]) -> float:
        """Calculate volume usage percentage"""
        size = volume.get("size", 0)
        used = volume.get("used", 0)
        
        if size > 0:
            return round((used / size) * 100, 2)
        return 0.0
    
    async def create_volume_snapshot(self, volume_id: str, name: str) -> Dict[str, Any]:
        """Create volume snapshot (if supported by RunPod)"""
        # Placeholder for future RunPod snapshot support
        logger.info(f"Snapshot creation for volume {volume_id} with name {name}")
        
        return {
            "volume_id": volume_id,
            "snapshot_name": name,
            "status": "not_supported",
            "message": "Volume snapshots are not yet supported by RunPod API"
        }
    
    async def restore_volume_snapshot(self, snapshot_id: str, volume_name: str) -> Dict[str, Any]:
        """Restore volume from snapshot (if supported by RunPod)"""
        # Placeholder for future RunPod snapshot support
        logger.info(f"Snapshot restore for {snapshot_id} to volume {volume_name}")
        
        return {
            "snapshot_id": snapshot_id,
            "volume_name": volume_name,
            "status": "not_supported",
            "message": "Volume snapshots are not yet supported by RunPod API"
        }