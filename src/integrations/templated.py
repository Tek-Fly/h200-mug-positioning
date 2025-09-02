"""Templated.io integration for design rendering based on mug positioning results."""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin

import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class TemplatedAPIError(Exception):
    """Templated.io API error."""
    pass


class TemplatedClient:
    """
    Async client for Templated.io API integration.
    
    Features:
    - Design rendering based on mug positioning data
    - Template management and customization
    - Async operations with proper retry logic
    - Comprehensive error handling
    
    Example:
        ```python
        client = TemplatedClient()
        
        # Render design based on positioning
        positioning_data = {
            "mug_position": {"x": 100, "y": 200},
            "confidence": 0.95,
            "template_params": {
                "text": "Hello World",
                "color": "#FF0000"
            }
        }
        
        result = await client.render_design(
            template_id="mug-template-1",
            positioning_data=positioning_data
        )
        ```
    """
    
    BASE_URL = "https://api.templated.io/v1"
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Templated.io client.
        
        Args:
            api_key: API key for authentication. If not provided, reads from TEMPLATED_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("TEMPLATED_API_KEY")
        if not self.api_key:
            raise ValueError("Templated API key not provided")
            
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info("Initialized Templated.io client")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            Response data
            
        Raises:
            TemplatedAPIError: On API errors
        """
        await self._ensure_session()
        
        url = urljoin(self.BASE_URL, endpoint)
        timeout = aiohttp.ClientTimeout(total=kwargs.pop("timeout", self.DEFAULT_TIMEOUT))
        
        try:
            async with self.session.request(
                method, url, timeout=timeout, **kwargs
            ) as response:
                data = await response.json()
                
                if response.status >= 400:
                    error_msg = data.get("error", "Unknown error")
                    raise TemplatedAPIError(
                        f"API error {response.status}: {error_msg}"
                    )
                
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise TemplatedAPIError(f"Request failed: {e}")
    
    async def render_design(
        self,
        template_id: str,
        positioning_data: Dict[str, Any],
        output_format: str = "png",
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Render a design based on mug positioning data.
        
        Args:
            template_id: ID of the template to use
            positioning_data: Mug positioning data and template parameters
            output_format: Output format (png, jpg, svg, pdf)
            width: Optional output width
            height: Optional output height
            
        Returns:
            Render result with URL and metadata
        """
        # Transform positioning data into template parameters
        template_params = self._transform_positioning_to_template(positioning_data)
        
        request_data = {
            "template_id": template_id,
            "params": template_params,
            "output_format": output_format
        }
        
        if width:
            request_data["width"] = width
        if height:
            request_data["height"] = height
        
        logger.info(f"Rendering design with template {template_id}")
        result = await self._make_request(
            "POST",
            "/render",
            json=request_data
        )
        
        logger.info(f"Design rendered successfully: {result.get('render_url')}")
        return result
    
    def _transform_positioning_to_template(
        self,
        positioning_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transform mug positioning data into template parameters.
        
        Args:
            positioning_data: Raw positioning data
            
        Returns:
            Template parameters
        """
        # Extract base positioning information
        mug_position = positioning_data.get("mug_position", {})
        confidence = positioning_data.get("confidence", 0)
        
        # Start with any custom template parameters
        template_params = positioning_data.get("template_params", {})
        
        # Add positioning-specific parameters
        template_params.update({
            "mug_x": mug_position.get("x", 0),
            "mug_y": mug_position.get("y", 0),
            "mug_width": mug_position.get("width", 100),
            "mug_height": mug_position.get("height", 100),
            "confidence_score": confidence,
            "positioning_quality": self._get_quality_label(confidence)
        })
        
        # Add any bounding box information
        if "bounding_box" in positioning_data:
            bbox = positioning_data["bounding_box"]
            template_params.update({
                "bbox_x1": bbox.get("x1", 0),
                "bbox_y1": bbox.get("y1", 0),
                "bbox_x2": bbox.get("x2", 0),
                "bbox_y2": bbox.get("y2", 0)
            })
        
        return template_params
    
    def _get_quality_label(self, confidence: float) -> str:
        """Get quality label based on confidence score."""
        if confidence >= 0.9:
            return "excellent"
        elif confidence >= 0.7:
            return "good"
        elif confidence >= 0.5:
            return "fair"
        else:
            return "poor"
    
    async def list_templates(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available templates.
        
        Args:
            category: Filter by category
            search: Search term
            
        Returns:
            List of templates
        """
        params = {}
        if category:
            params["category"] = category
        if search:
            params["search"] = search
        
        result = await self._make_request(
            "GET",
            "/templates",
            params=params
        )
        
        return result.get("templates", [])
    
    async def get_template(self, template_id: str) -> Dict[str, Any]:
        """
        Get template details.
        
        Args:
            template_id: Template ID
            
        Returns:
            Template details
        """
        return await self._make_request(
            "GET",
            f"/templates/{template_id}"
        )
    
    async def create_custom_template(
        self,
        name: str,
        description: str,
        template_data: Dict[str, Any],
        category: str = "mug-positioning"
    ) -> Dict[str, Any]:
        """
        Create a custom template for mug positioning.
        
        Args:
            name: Template name
            description: Template description
            template_data: Template configuration
            category: Template category
            
        Returns:
            Created template details
        """
        request_data = {
            "name": name,
            "description": description,
            "category": category,
            "template": template_data
        }
        
        result = await self._make_request(
            "POST",
            "/templates",
            json=request_data
        )
        
        logger.info(f"Created custom template: {result.get('template_id')}")
        return result
    
    async def batch_render(
        self,
        renders: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Batch render multiple designs.
        
        Args:
            renders: List of render configurations
            
        Returns:
            List of render results
        """
        request_data = {"renders": renders}
        
        result = await self._make_request(
            "POST",
            "/batch-render",
            json=request_data
        )
        
        return result.get("renders", [])


# Usage example
async def main():
    """Example usage of TemplatedClient."""
    # Initialize client
    async with TemplatedClient() as client:
        # Example 1: Render design based on positioning
        positioning_data = {
            "mug_position": {
                "x": 150,
                "y": 200,
                "width": 120,
                "height": 180
            },
            "confidence": 0.92,
            "bounding_box": {
                "x1": 150,
                "y1": 200,
                "x2": 270,
                "y2": 380
            },
            "template_params": {
                "text": "Perfect Mug Position!",
                "background_color": "#F0F0F0",
                "border_color": "#00FF00"
            }
        }
        
        # Render the design
        result = await client.render_design(
            template_id="mug-overlay-template",
            positioning_data=positioning_data,
            output_format="png",
            width=800,
            height=600
        )
        print(f"Rendered design: {result}")
        
        # Example 2: List available templates
        templates = await client.list_templates(category="mug-positioning")
        print(f"Available templates: {len(templates)}")
        
        # Example 3: Batch render multiple designs
        batch_renders = [
            {
                "template_id": "mug-template-1",
                "params": {"text": "Mug 1", "mug_x": 100, "mug_y": 100}
            },
            {
                "template_id": "mug-template-2",
                "params": {"text": "Mug 2", "mug_x": 200, "mug_y": 200}
            }
        ]
        
        batch_results = await client.batch_render(batch_renders)
        print(f"Batch rendered {len(batch_results)} designs")


if __name__ == "__main__":
    asyncio.run(main())