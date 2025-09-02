"""
RunPod API client wrapper with retry logic and error handling
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)


class RunPodAPIError(Exception):
    """RunPod API error"""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class RunPodClient:
    """Async RunPod API client with retry logic"""
    
    BASE_URL = "https://api.runpod.io/v2"
    
    def __init__(self, api_key: str, timeout: int = 30):
        """Initialize RunPod client"""
        self.api_key = api_key
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_remaining = 100
        self._rate_limit_reset = datetime.utcnow()
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(aiohttp.ClientError),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make API request with retry logic"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with context manager.")
        
        # Check rate limiting
        await self._check_rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                # Update rate limit info
                self._update_rate_limit(response.headers)
                
                # Check response status
                if response.status >= 400:
                    error_data = await response.json() if response.content_type == "application/json" else None
                    raise RunPodAPIError(
                        f"API request failed: {response.status}",
                        status_code=response.status,
                        response=error_data
                    )
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"RunPod API request failed: {e}")
            raise
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        if self._rate_limit_remaining <= 0:
            wait_time = (self._rate_limit_reset - datetime.utcnow()).total_seconds()
            if wait_time > 0:
                logger.warning(f"Rate limit exceeded. Waiting {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
    
    def _update_rate_limit(self, headers: Dict[str, str]):
        """Update rate limit information from response headers"""
        if "X-RateLimit-Remaining" in headers:
            self._rate_limit_remaining = int(headers["X-RateLimit-Remaining"])
        
        if "X-RateLimit-Reset" in headers:
            self._rate_limit_reset = datetime.fromtimestamp(int(headers["X-RateLimit-Reset"]))
    
    # Endpoint methods
    
    async def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all serverless endpoints"""
        response = await self._request("GET", "/serverless/endpoints")
        return response.get("endpoints", [])
    
    async def get_endpoint(self, endpoint_id: str) -> Dict[str, Any]:
        """Get serverless endpoint details"""
        return await self._request("GET", f"/serverless/endpoints/{endpoint_id}")
    
    async def create_endpoint(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create serverless endpoint"""
        return await self._request("POST", "/serverless/endpoints", json=config)
    
    async def update_endpoint(self, endpoint_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update serverless endpoint"""
        return await self._request("PATCH", f"/serverless/endpoints/{endpoint_id}", json=config)
    
    async def delete_endpoint(self, endpoint_id: str) -> Dict[str, Any]:
        """Delete serverless endpoint"""
        return await self._request("DELETE", f"/serverless/endpoints/{endpoint_id}")
    
    async def list_pods(self) -> List[Dict[str, Any]]:
        """List all GPU pods"""
        response = await self._request("GET", "/pods")
        return response.get("pods", [])
    
    async def get_pod(self, pod_id: str) -> Dict[str, Any]:
        """Get GPU pod details"""
        return await self._request("GET", f"/pods/{pod_id}")
    
    async def create_pod(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create GPU pod"""
        return await self._request("POST", "/pods", json=config)
    
    async def start_pod(self, pod_id: str) -> Dict[str, Any]:
        """Start GPU pod"""
        return await self._request("POST", f"/pods/{pod_id}/start")
    
    async def stop_pod(self, pod_id: str) -> Dict[str, Any]:
        """Stop GPU pod"""
        return await self._request("POST", f"/pods/{pod_id}/stop")
    
    async def restart_pod(self, pod_id: str) -> Dict[str, Any]:
        """Restart GPU pod"""
        return await self._request("POST", f"/pods/{pod_id}/restart")
    
    async def terminate_pod(self, pod_id: str) -> Dict[str, Any]:
        """Terminate GPU pod"""
        return await self._request("DELETE", f"/pods/{pod_id}")
    
    async def get_pod_logs(self, pod_id: str, lines: int = 100) -> str:
        """Get pod logs"""
        response = await self._request("GET", f"/pods/{pod_id}/logs", params={"lines": lines})
        return response.get("logs", "")
    
    # Volume management
    
    async def list_volumes(self) -> List[Dict[str, Any]]:
        """List network volumes"""
        response = await self._request("GET", "/volumes")
        return response.get("volumes", [])
    
    async def get_volume(self, volume_id: str) -> Dict[str, Any]:
        """Get volume details"""
        return await self._request("GET", f"/volumes/{volume_id}")
    
    async def create_volume(self, name: str, size_gb: int, region: str = "us-east-1") -> Dict[str, Any]:
        """Create network volume"""
        config = {
            "name": name,
            "size": size_gb,
            "region": region
        }
        return await self._request("POST", "/volumes", json=config)
    
    async def update_volume(self, volume_id: str, size_gb: int) -> Dict[str, Any]:
        """Update volume size"""
        return await self._request("PATCH", f"/volumes/{volume_id}", json={"size": size_gb})
    
    async def delete_volume(self, volume_id: str) -> Dict[str, Any]:
        """Delete volume"""
        return await self._request("DELETE", f"/volumes/{volume_id}")
    
    async def attach_volume(self, pod_id: str, volume_id: str, mount_path: str = "/workspace") -> Dict[str, Any]:
        """Attach volume to pod"""
        config = {
            "volume_id": volume_id,
            "mount_path": mount_path
        }
        return await self._request("POST", f"/pods/{pod_id}/volumes", json=config)
    
    async def detach_volume(self, pod_id: str, volume_id: str) -> Dict[str, Any]:
        """Detach volume from pod"""
        return await self._request("DELETE", f"/pods/{pod_id}/volumes/{volume_id}")
    
    # Metrics and monitoring
    
    async def get_endpoint_metrics(self, endpoint_id: str, 
                                  start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get endpoint metrics"""
        params = {}
        if start_time:
            params["start"] = start_time.isoformat()
        if end_time:
            params["end"] = end_time.isoformat()
        
        return await self._request("GET", f"/serverless/endpoints/{endpoint_id}/metrics", params=params)
    
    async def get_pod_metrics(self, pod_id: str,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get pod metrics"""
        params = {}
        if start_time:
            params["start"] = start_time.isoformat()
        if end_time:
            params["end"] = end_time.isoformat()
        
        return await self._request("GET", f"/pods/{pod_id}/metrics", params=params)
    
    # Health checks
    
    async def health_check(self) -> bool:
        """Check API health"""
        try:
            response = await self._request("GET", "/health")
            return response.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    # Webhook management
    
    async def register_webhook(self, endpoint_id: str, webhook_url: str, 
                             events: List[str] = None) -> Dict[str, Any]:
        """Register webhook for endpoint events"""
        if events is None:
            events = ["job.completed", "job.failed", "worker.started", "worker.stopped"]
        
        config = {
            "url": webhook_url,
            "events": events
        }
        return await self._request("POST", f"/serverless/endpoints/{endpoint_id}/webhooks", json=config)
    
    async def list_webhooks(self, endpoint_id: str) -> List[Dict[str, Any]]:
        """List webhooks for endpoint"""
        response = await self._request("GET", f"/serverless/endpoints/{endpoint_id}/webhooks")
        return response.get("webhooks", [])
    
    async def delete_webhook(self, endpoint_id: str, webhook_id: str) -> Dict[str, Any]:
        """Delete webhook"""
        return await self._request("DELETE", f"/serverless/endpoints/{endpoint_id}/webhooks/{webhook_id}")
    
    # Helper methods
    
    async def wait_for_pod_ready(self, pod_id: str, timeout: int = 300, 
                                callback: Optional[Callable] = None) -> bool:
        """Wait for pod to be ready"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            try:
                pod = await self.get_pod(pod_id)
                status = pod.get("status", "").lower()
                
                if callback:
                    await callback(pod)
                
                if status == "running":
                    logger.info(f"Pod {pod_id} is ready")
                    return True
                elif status in ["failed", "terminated"]:
                    logger.error(f"Pod {pod_id} failed with status: {status}")
                    return False
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error checking pod status: {e}")
                await asyncio.sleep(5)
        
        logger.error(f"Pod {pod_id} not ready after {timeout} seconds")
        return False
    
    async def wait_for_endpoint_ready(self, endpoint_id: str, timeout: int = 300,
                                     callback: Optional[Callable] = None) -> bool:
        """Wait for endpoint to be ready"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            try:
                endpoint = await self.get_endpoint(endpoint_id)
                status = endpoint.get("status", "").lower()
                workers = endpoint.get("workers", {})
                
                if callback:
                    await callback(endpoint)
                
                if status == "running" and workers.get("running", 0) > 0:
                    logger.info(f"Endpoint {endpoint_id} is ready with {workers['running']} workers")
                    return True
                elif status == "failed":
                    logger.error(f"Endpoint {endpoint_id} failed")
                    return False
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error checking endpoint status: {e}")
                await asyncio.sleep(5)
        
        logger.error(f"Endpoint {endpoint_id} not ready after {timeout} seconds")
        return False