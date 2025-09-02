"""
MCP Integration with H200 Control Plane.

Provides integration points for the MCP server with the main
H200 control plane and FastAPI application.
"""

# Standard library imports
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

# Third-party imports
import structlog
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

# First-party imports
from src.core.mcp.auth import MCPAuthenticator
from src.core.mcp.models import MCPAuthType, MCPRequest
from src.core.mcp.server import MCPServer

logger = structlog.get_logger(__name__)


class MCPIntegration:
    """
    Integration layer for MCP server with FastAPI.

    This class provides:
    - Lifecycle management for MCP server
    - FastAPI route integration
    - Shared component management
    """

    def __init__(self):
        self.mcp_server: Optional[MCPServer] = None
        self._server_task: Optional[asyncio.Task] = None

    async def startup(self):
        """Initialize and start MCP server."""
        logger.info("Starting MCP integration...")

        # Create MCP server
        self.mcp_server = MCPServer(
            host="0.0.0.0",
            port=8765,
            enable_auth=True,
            enable_websocket=True,
            enable_http=False,  # We'll use FastAPI routes instead
        )

        # Initialize server
        await self.mcp_server.initialize()

        # Start server in background
        self._server_task = asyncio.create_task(self._run_mcp_server())

        logger.info("MCP integration started")

    async def shutdown(self):
        """Shutdown MCP server."""
        logger.info("Shutting down MCP integration...")

        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass

        if self.mcp_server:
            await self.mcp_server.shutdown()

        logger.info("MCP integration shutdown complete")

    async def _run_mcp_server(self):
        """Run MCP server (WebSocket only)."""
        try:
            await self.mcp_server._start_websocket_server()
        except asyncio.CancelledError:
            logger.info("MCP server task cancelled")
        except Exception as e:
            logger.error("MCP server error", error=str(e))

    def get_capabilities(self):
        """Get MCP server capabilities."""
        if not self.mcp_server:
            raise HTTPException(status_code=503, detail="MCP server not initialized")

        return self.mcp_server.get_capabilities()

    async def execute_tool(
        self, request: MCPRequest, auth_header: Optional[str] = None
    ):
        """Execute MCP tool."""
        if not self.mcp_server:
            raise HTTPException(status_code=503, detail="MCP server not initialized")

        # Add auth to request if provided
        if auth_header and auth_header.startswith("Bearer "):
            request.auth = {"type": MCPAuthType.JWT.value, "token": auth_header[7:]}

        # Execute request
        response = await self.mcp_server._handle_request(request)

        return response

    def create_jwt_token(
        self, client_id: str, scopes: list = None, metadata: dict = None
    ) -> str:
        """Create JWT token for MCP access."""
        if not self.mcp_server or not self.mcp_server.authenticator:
            raise HTTPException(status_code=503, detail="Authentication not available")

        return self.mcp_server.authenticator.generate_jwt_token(
            client_id=client_id,
            scopes=scopes or ["analyze", "suggest"],
            metadata=metadata,
        )

    def create_api_key(
        self, client_id: str, name: str, scopes: list = None, rate_limit: dict = None
    ) -> str:
        """Create API key for MCP access."""
        if not self.mcp_server or not self.mcp_server.authenticator:
            raise HTTPException(status_code=503, detail="Authentication not available")

        return self.mcp_server.authenticator.generate_api_key(
            client_id=client_id, name=name, scopes=scopes, rate_limit=rate_limit
        )


# Global instance
mcp_integration = MCPIntegration()


@asynccontextmanager
async def mcp_lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    # Startup
    await mcp_integration.startup()

    yield

    # Shutdown
    await mcp_integration.shutdown()


def setup_mcp_routes(app: FastAPI):
    """Setup MCP routes in FastAPI application."""

    @app.get("/api/v1/mcp/capabilities")
    async def get_mcp_capabilities():
        """Get MCP server capabilities."""
        capabilities = mcp_integration.get_capabilities()
        return JSONResponse(content=capabilities.dict())

    @app.post("/api/v1/mcp/execute")
    async def execute_mcp_tool(
        request: MCPRequest, authorization: Optional[str] = Header(None)
    ):
        """Execute MCP tool."""
        response = await mcp_integration.execute_tool(request, authorization)
        return JSONResponse(content=response.dict())

    @app.post("/api/v1/mcp/auth/token")
    async def create_mcp_token(
        client_id: str, scopes: list = None, metadata: dict = None
    ):
        """Create MCP JWT token."""
        token = mcp_integration.create_jwt_token(client_id, scopes, metadata)
        return {"token": token, "type": "Bearer", "expires_in": 86400}  # 24 hours

    @app.post("/api/v1/mcp/auth/api-key")
    async def create_mcp_api_key(
        client_id: str, name: str, scopes: list = None, rate_limit: dict = None
    ):
        """Create MCP API key."""
        api_key = mcp_integration.create_api_key(client_id, name, scopes, rate_limit)
        return {"api_key": api_key, "name": name, "client_id": client_id}

    @app.get("/api/v1/mcp/health")
    async def mcp_health_check():
        """Check MCP server health."""
        if mcp_integration.mcp_server and mcp_integration.mcp_server._running:
            return {
                "status": "healthy",
                "websocket_port": 8765,
                "active_connections": len(
                    mcp_integration.mcp_server._connected_clients
                ),
            }
        else:
            raise HTTPException(status_code=503, detail="MCP server not running")

    logger.info("MCP routes configured")


# Example usage in main FastAPI app:
"""
from fastapi import FastAPI
from src.core.mcp.integration import mcp_lifespan, setup_mcp_routes

app = FastAPI(lifespan=mcp_lifespan)
setup_mcp_routes(app)
"""
