"""
MCP Client Implementation.

Example client for connecting to the MCP server and using its tools.
"""

# Standard library imports
import asyncio
import base64
import io
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Third-party imports
import aiohttp
import structlog
import websockets
from PIL import Image

# Local imports
from .auth import MCPAuthenticator
from .models import (
    MCPAuthType,
    MCPBatchRequest,
    MCPCapabilities,
    MCPRequest,
    MCPResponse,
)

logger = structlog.get_logger(__name__)


class MCPClient:
    """
    Client for interacting with the MCP server.

    Supports both HTTP and WebSocket connections.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8766",
        websocket_url: str = "ws://localhost:8765",
        auth_type: MCPAuthType = MCPAuthType.JWT,
        auth_token: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize MCP client."""
        self.base_url = base_url
        self.websocket_url = websocket_url
        self.auth_type = auth_type
        self.auth_token = auth_token
        self.timeout = timeout

        self._websocket = None
        self._capabilities = None
        self._session = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self):
        """Connect to the MCP server."""
        # Create HTTP session
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )

        # Get capabilities
        await self.get_capabilities()

        logger.info("Connected to MCP server", base_url=self.base_url)

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("Disconnected from MCP server")

    async def connect_websocket(self):
        """Connect via WebSocket for streaming."""
        if self._websocket:
            return

        self._websocket = await websockets.connect(
            self.websocket_url, ping_interval=30, ping_timeout=10
        )

        logger.info("WebSocket connected", url=self.websocket_url)

    def _get_auth_header(self) -> Dict[str, str]:
        """Get authorization header."""
        if not self.auth_token:
            return {}

        if self.auth_type == MCPAuthType.JWT:
            return {"Authorization": f"Bearer {self.auth_token}"}
        elif self.auth_type == MCPAuthType.API_KEY:
            return {"X-API-Key": self.auth_token}

        return {}

    def _get_auth_data(self) -> Dict[str, Any]:
        """Get auth data for request."""
        if not self.auth_token:
            return None

        if self.auth_type == MCPAuthType.JWT:
            return {"type": MCPAuthType.JWT.value, "token": self.auth_token}
        elif self.auth_type == MCPAuthType.API_KEY:
            return {"type": MCPAuthType.API_KEY.value, "key": self.auth_token}

        return None

    async def get_capabilities(self) -> MCPCapabilities:
        """Get server capabilities."""
        if self._capabilities:
            return self._capabilities

        async with self._session.get(
            f"{self.base_url}/mcp/v1/capabilities", headers=self._get_auth_header()
        ) as response:
            response.raise_for_status()
            data = await response.json()
            self._capabilities = MCPCapabilities(**data)
            return self._capabilities

    async def execute(
        self,
        tool: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MCPResponse:
        """Execute a tool on the server."""
        request = MCPRequest(
            tool=tool,
            parameters=parameters,
            auth=self._get_auth_data(),
            metadata=metadata,
        )

        async with self._session.post(
            f"{self.base_url}/mcp/v1/execute",
            json=request.dict(),
            headers=self._get_auth_header(),
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return MCPResponse(**data)

    async def execute_batch(
        self,
        requests: List[Dict[str, Any]],
        parallel: bool = True,
        stop_on_error: bool = False,
    ) -> Dict[str, Any]:
        """Execute multiple requests as a batch."""
        # Create batch request
        mcp_requests = [
            MCPRequest(
                tool=req["tool"],
                parameters=req["parameters"],
                auth=self._get_auth_data(),
                metadata=req.get("metadata"),
            )
            for req in requests
        ]

        batch_request = MCPBatchRequest(
            requests=mcp_requests, parallel=parallel, stop_on_error=stop_on_error
        )

        async with self._session.post(
            f"{self.base_url}/mcp/v1/batch",
            json=batch_request.dict(),
            headers=self._get_auth_header(),
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def analyze_image(
        self,
        image: Image.Image,
        apply_rules: bool = True,
        return_embeddings: bool = False,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Analyze an image for mug positioning."""
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        response = await self.execute(
            "analyze_image",
            {
                "image": image_base64,
                "image_format": "jpeg",
                "apply_rules": apply_rules,
                "return_embeddings": return_embeddings,
                "confidence_threshold": confidence_threshold,
            },
        )

        if not response.result.success:
            raise Exception(f"Analysis failed: {response.result.error.message}")

        return response.result.data

    async def apply_rule(
        self,
        rule_text: str,
        priority: int = 5,
        tags: List[str] = None,
        validate_only: bool = False,
    ) -> Dict[str, Any]:
        """Apply a natural language rule."""
        response = await self.execute(
            "apply_rules",
            {
                "rule_text": rule_text,
                "priority": priority,
                "tags": tags or [],
                "validate_only": validate_only,
            },
        )

        if not response.result.success:
            raise Exception(f"Rule application failed: {response.result.error.message}")

        return response.result.data

    async def get_suggestions(
        self,
        scene_context: Dict[str, Any],
        strategy: str = "balanced",
        constraints: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Get positioning suggestions."""
        response = await self.execute(
            "get_suggestions",
            {
                "scene_context": scene_context,
                "strategy": strategy,
                "constraints": constraints or {},
            },
        )

        if not response.result.success:
            raise Exception(
                f"Suggestion generation failed: {response.result.error.message}"
            )

        return response.result.data

    async def update_positioning(
        self, analysis_id: str, feedback: Dict[str, Any], update_model: bool = False
    ) -> Dict[str, Any]:
        """Update positioning based on feedback."""
        response = await self.execute(
            "update_positioning",
            {
                "analysis_id": analysis_id,
                "feedback": feedback,
                "update_model": update_model,
            },
        )

        if not response.result.success:
            raise Exception(
                f"Positioning update failed: {response.result.error.message}"
            )

        return response.result.data

    async def stream_updates(self, callback):
        """Stream real-time updates via WebSocket."""
        await self.connect_websocket()

        async for message in self._websocket:
            try:
                data = json.loads(message)
                await callback(data)
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON in stream", error=str(e))
            except Exception as e:
                logger.error("Stream callback error", error=str(e))

    async def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        async with self._session.get(
            f"{self.base_url}/mcp/v1/health", headers=self._get_auth_header()
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        async with self._session.get(
            f"{self.base_url}/mcp/v1/stats", headers=self._get_auth_header()
        ) as response:
            response.raise_for_status()
            return await response.json()


async def example_usage():
    """Example of using the MCP client."""
    # Initialize client
    async with MCPClient(
        auth_type=MCPAuthType.JWT, auth_token="your-jwt-token"
    ) as client:

        # Check capabilities
        capabilities = await client.get_capabilities()
        print(f"Server version: {capabilities.version}")
        print(f"Available tools: {[t.name for t in capabilities.tools]}")

        # Analyze an image
        image = Image.open("test_image.jpg")
        analysis = await client.analyze_image(image)
        print(f"Analysis ID: {analysis['analysis_id']}")
        print(f"Detected {len(analysis['detections'])} objects")

        # Apply a rule
        rule_result = await client.apply_rule(
            "Keep mugs at least 6 inches away from electronic devices",
            priority=8,
            tags=["safety", "electronics"],
        )
        print(f"Rule ID: {rule_result['rule_id']}")

        # Get suggestions
        suggestions = await client.get_suggestions(
            scene_context={
                "objects": analysis["detections"],
                "environment": {"type": "office", "desk_size": "large"},
            },
            strategy="safety_first",
        )
        print(f"Got {len(suggestions['suggestions'])} suggestions")

        # Update based on feedback
        if suggestions["suggestions"]:
            feedback_result = await client.update_positioning(
                analysis_id=analysis["analysis_id"],
                feedback={
                    "type": "user_preference",
                    "suggestions_accepted": [
                        s["mug_id"] for s in suggestions["suggestions"][:2]
                    ],
                    "suggestions_rejected": [],
                    "reason": "Looks good!",
                },
            )
            print(f"Feedback ID: {feedback_result['feedback_id']}")

        # Check server health
        health = await client.health_check()
        print(f"Server status: {health['status']}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
