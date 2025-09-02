"""
MCP Server Implementation.

Main server class that handles MCP protocol communication,
tool registration, and request routing.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import signal
import sys

import structlog
import websockets
from websockets.server import WebSocketServerProtocol
import aiohttp

from src.core.analyzer import ImageAnalyzer
from src.core.rules.engine import RuleEngine
from src.core.positioning import MugPositioningEngine
from src.utils.logging import setup_structured_logging
from src.utils.secrets import get_secret

from .models import (
    MCPRequest,
    MCPResponse,
    MCPCapabilities,
    MCPVersion,
    MCPAuthType,
    MCPError,
    MCPToolResult,
    MCPBatchRequest,
    MCPBatchResponse,
    MCPStreamChunk
)
from .auth import MCPAuthenticator
from .tools import (
    AVAILABLE_TOOLS,
    BaseMCPTool,
    AnalyzeImageTool,
    ApplyRulesTool,
    GetSuggestionsTool,
    UpdatePositioningTool
)


logger = structlog.get_logger(__name__)


class MCPServer:
    """
    MCP (Model Context Protocol) Server for H200 Mug Positioning.
    
    This server implements the A2A protocol for exposing mug positioning
    capabilities to external systems. It supports:
    
    - Multiple authentication methods (JWT, API Key)
    - Rate limiting per client
    - Async tool execution
    - Batch requests
    - WebSocket streaming
    - Tool discovery
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        enable_auth: bool = True,
        enable_websocket: bool = True,
        enable_http: bool = True
    ):
        """Initialize MCP server."""
        self.host = host
        self.port = port
        self.enable_auth = enable_auth
        self.enable_websocket = enable_websocket
        self.enable_http = enable_http
        
        # Core components
        self.authenticator = MCPAuthenticator() if enable_auth else None
        self.tools: Dict[str, BaseMCPTool] = {}
        
        # Shared components
        self.analyzer = None
        self.rule_engine = None
        self.positioning_engine = None
        
        # Server state
        self._initialized = False
        self._running = False
        self._websocket_server = None
        self._http_app = None
        self._connected_clients: Set[WebSocketServerProtocol] = set()
        
        # Statistics
        self._stats = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "active_connections": 0,
            "start_time": datetime.utcnow()
        }
    
    async def initialize(self):
        """Initialize server and components."""
        if self._initialized:
            return
        
        logger.info("Initializing MCP server", host=self.host, port=self.port)
        
        # Setup logging
        setup_structured_logging()
        
        # Initialize shared components
        self.analyzer = ImageAnalyzer()
        await self.analyzer.initialize()
        
        self.rule_engine = RuleEngine()
        await self.rule_engine.initialize()
        
        self.positioning_engine = MugPositioningEngine()
        await self.positioning_engine.initialize()
        
        # Register tools
        await self._register_tools()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._initialized = True
        logger.info("MCP server initialized", num_tools=len(self.tools))
    
    async def _register_tools(self):
        """Register available tools."""
        # Create tool instances with shared components
        tool_instances = {
            "analyze_image": AnalyzeImageTool(
                analyzer=self.analyzer,
                authenticator=self.authenticator
            ),
            "apply_rules": ApplyRulesTool(
                rule_engine=self.rule_engine,
                authenticator=self.authenticator
            ),
            "get_suggestions": GetSuggestionsTool(
                positioning_engine=self.positioning_engine,
                authenticator=self.authenticator
            ),
            "update_positioning": UpdatePositioningTool(
                positioning_engine=self.positioning_engine,
                authenticator=self.authenticator
            )
        }
        
        # Initialize and register tools
        for name, tool in tool_instances.items():
            await tool.initialize()
            self.tools[name] = tool
            logger.info("Registered tool", tool=name)
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals."""
        logger.info("Received shutdown signal", signal=sig)
        asyncio.create_task(self.shutdown())
    
    async def start(self):
        """Start the MCP server."""
        if not self._initialized:
            await self.initialize()
        
        self._running = True
        logger.info("Starting MCP server")
        
        tasks = []
        
        # Start WebSocket server
        if self.enable_websocket:
            tasks.append(self._start_websocket_server())
        
        # Start HTTP server
        if self.enable_http:
            tasks.append(self._start_http_server())
        
        # Start statistics reporter
        tasks.append(self._report_statistics())
        
        # Wait for all tasks
        await asyncio.gather(*tasks)
    
    async def _start_websocket_server(self):
        """Start WebSocket server for streaming connections."""
        async def handler(websocket: WebSocketServerProtocol, path: str):
            await self._handle_websocket_connection(websocket, path)
        
        self._websocket_server = await websockets.serve(
            handler,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        )
        
        logger.info(
            "WebSocket server started",
            host=self.host,
            port=self.port
        )
        
        # Keep server running
        await asyncio.Future()
    
    async def _handle_websocket_connection(
        self,
        websocket: WebSocketServerProtocol,
        path: str
    ):
        """Handle individual WebSocket connection."""
        client_id = str(uuid.uuid4())
        self._connected_clients.add(websocket)
        self._stats["active_connections"] += 1
        
        logger.info(
            "WebSocket client connected",
            client_id=client_id,
            path=path,
            remote_address=websocket.remote_address
        )
        
        try:
            async for message in websocket:
                try:
                    # Parse request
                    data = json.loads(message)
                    
                    # Handle different message types
                    if data.get("type") == "capabilities":
                        # Send capabilities
                        capabilities = self.get_capabilities()
                        await websocket.send(json.dumps({
                            "type": "capabilities",
                            "data": capabilities.dict()
                        }))
                    
                    elif data.get("type") == "request":
                        # Handle tool request
                        request = MCPRequest(**data.get("request", {}))
                        response = await self._handle_request(request)
                        
                        await websocket.send(json.dumps({
                            "type": "response",
                            "data": response.dict()
                        }))
                    
                    elif data.get("type") == "batch":
                        # Handle batch request
                        batch_request = MCPBatchRequest(**data.get("batch", {}))
                        batch_response = await self._handle_batch_request(batch_request)
                        
                        await websocket.send(json.dumps({
                            "type": "batch_response",
                            "data": batch_response.dict()
                        }))
                    
                    elif data.get("type") == "ping":
                        # Respond to ping
                        await websocket.send(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        }))
                    
                except json.JSONDecodeError as e:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": {
                            "code": "INVALID_JSON",
                            "message": f"Invalid JSON: {str(e)}"
                        }
                    }))
                
                except Exception as e:
                    logger.error(
                        "WebSocket message handling error",
                        error=str(e),
                        client_id=client_id
                    )
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": "Internal server error"
                        }
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket client disconnected", client_id=client_id)
        
        finally:
            self._connected_clients.discard(websocket)
            self._stats["active_connections"] -= 1
    
    async def _start_http_server(self):
        """Start HTTP server for REST API."""
        from aiohttp import web
        
        app = web.Application()
        
        # Add routes
        app.router.add_get("/mcp/v1/capabilities", self._handle_capabilities)
        app.router.add_post("/mcp/v1/execute", self._handle_execute)
        app.router.add_post("/mcp/v1/batch", self._handle_batch)
        app.router.add_get("/mcp/v1/health", self._handle_health)
        app.router.add_get("/mcp/v1/stats", self._handle_stats)
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port + 1)
        await site.start()
        
        self._http_app = app
        
        logger.info(
            "HTTP server started",
            host=self.host,
            port=self.port + 1
        )
        
        # Keep server running
        await asyncio.Future()
    
    async def _handle_capabilities(self, request):
        """Handle capabilities request."""
        from aiohttp import web
        
        capabilities = self.get_capabilities()
        return web.json_response(capabilities.dict())
    
    async def _handle_execute(self, request):
        """Handle tool execution request."""
        from aiohttp import web
        
        try:
            data = await request.json()
            mcp_request = MCPRequest(**data)
            response = await self._handle_request(mcp_request)
            return web.json_response(response.dict())
        
        except Exception as e:
            return web.json_response({
                "error": {
                    "code": "REQUEST_ERROR",
                    "message": str(e)
                }
            }, status=400)
    
    async def _handle_batch(self, request):
        """Handle batch request."""
        from aiohttp import web
        
        try:
            data = await request.json()
            batch_request = MCPBatchRequest(**data)
            batch_response = await self._handle_batch_request(batch_request)
            return web.json_response(batch_response.dict())
        
        except Exception as e:
            return web.json_response({
                "error": {
                    "code": "BATCH_ERROR",
                    "message": str(e)
                }
            }, status=400)
    
    async def _handle_health(self, request):
        """Handle health check."""
        from aiohttp import web
        
        return web.json_response({
            "status": "healthy",
            "version": MCPVersion.V1_0.value,
            "uptime": (datetime.utcnow() - self._stats["start_time"]).total_seconds(),
            "active_connections": self._stats["active_connections"]
        })
    
    async def _handle_stats(self, request):
        """Handle statistics request."""
        from aiohttp import web
        
        # Check authorization
        auth_header = request.headers.get("Authorization")
        if self.enable_auth and not auth_header:
            return web.json_response({
                "error": "Authorization required"
            }, status=401)
        
        return web.json_response(self._stats)
    
    async def _handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle individual MCP request."""
        self._stats["requests_total"] += 1
        start_time = time.time()
        
        try:
            # Validate tool exists
            if request.tool not in self.tools:
                self._stats["requests_failed"] += 1
                return MCPResponse(
                    id=str(uuid.uuid4()),
                    request_id=request.id,
                    result=MCPToolResult(
                        success=False,
                        error=MCPError(
                            code="UNKNOWN_TOOL",
                            message=f"Unknown tool: {request.tool}"
                        ),
                        execution_time_ms=(time.time() - start_time) * 1000
                    )
                )
            
            # Execute tool
            tool = self.tools[request.tool]
            response = await tool.execute(request)
            
            if response.result.success:
                self._stats["requests_success"] += 1
            else:
                self._stats["requests_failed"] += 1
            
            return response
            
        except Exception as e:
            logger.error(
                "Request handling error",
                error=str(e),
                request_id=request.id,
                tool=request.tool
            )
            self._stats["requests_failed"] += 1
            
            return MCPResponse(
                id=str(uuid.uuid4()),
                request_id=request.id,
                result=MCPToolResult(
                    success=False,
                    error=MCPError(
                        code="EXECUTION_ERROR",
                        message=f"Tool execution failed: {str(e)}"
                    ),
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            )
    
    async def _handle_batch_request(
        self,
        batch_request: MCPBatchRequest
    ) -> MCPBatchResponse:
        """Handle batch request."""
        responses = []
        
        if batch_request.parallel:
            # Execute requests in parallel
            tasks = [
                self._handle_request(req)
                for req in batch_request.requests
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error responses
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    responses[i] = MCPResponse(
                        id=str(uuid.uuid4()),
                        request_id=batch_request.requests[i].id,
                        result=MCPToolResult(
                            success=False,
                            error=MCPError(
                                code="BATCH_ITEM_ERROR",
                                message=str(response)
                            ),
                            execution_time_ms=0
                        )
                    )
        else:
            # Execute requests sequentially
            for req in batch_request.requests:
                response = await self._handle_request(req)
                responses.append(response)
                
                # Stop on error if requested
                if batch_request.stop_on_error and not response.result.success:
                    break
        
        # Calculate summary
        successful = sum(1 for r in responses if r.result.success)
        failed = len(responses) - successful
        
        return MCPBatchResponse(
            id=str(uuid.uuid4()),
            batch_id=batch_request.id,
            responses=responses,
            summary={
                "total": len(responses),
                "successful": successful,
                "failed": failed
            }
        )
    
    def get_capabilities(self) -> MCPCapabilities:
        """Get server capabilities."""
        tool_definitions = [
            tool.get_definition()
            for tool in self.tools.values()
        ]
        
        return MCPCapabilities(
            version=MCPVersion.V1_0,
            tools=tool_definitions,
            auth_types=[MCPAuthType.JWT, MCPAuthType.API_KEY] if self.enable_auth else [],
            features=[
                "async",
                "batch",
                "streaming" if self.enable_websocket else None,
                "rate_limiting" if self.enable_auth else None
            ],
            limits={
                "max_request_size": 10 * 1024 * 1024,  # 10MB
                "max_batch_size": 100,
                "max_websocket_connections": 1000,
                "timeout_seconds": 300
            }
        )
    
    async def _report_statistics(self):
        """Periodically report server statistics."""
        while self._running:
            try:
                logger.info(
                    "Server statistics",
                    **self._stats,
                    uptime_seconds=(
                        datetime.utcnow() - self._stats["start_time"]
                    ).total_seconds()
                )
                
                # Report every 60 seconds
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error("Statistics reporting error", error=str(e))
                await asyncio.sleep(60)
    
    async def broadcast_to_clients(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSocket clients."""
        if not self._connected_clients:
            return
        
        message_str = json.dumps(message)
        
        # Send to all clients
        disconnected = set()
        for client in self._connected_clients:
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(
                    "Broadcast error",
                    error=str(e),
                    client=client.remote_address
                )
        
        # Remove disconnected clients
        self._connected_clients -= disconnected
    
    async def shutdown(self):
        """Gracefully shutdown the server."""
        logger.info("Shutting down MCP server")
        self._running = False
        
        # Close WebSocket connections
        if self._connected_clients:
            close_tasks = [
                client.close()
                for client in self._connected_clients
            ]
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        # Stop WebSocket server
        if self._websocket_server:
            self._websocket_server.close()
            await self._websocket_server.wait_closed()
        
        # Stop HTTP server
        if self._http_app:
            await self._http_app.shutdown()
            await self._http_app.cleanup()
        
        logger.info("MCP server shutdown complete")


# Convenience function for running the server
async def run_mcp_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    **kwargs
):
    """Run the MCP server."""
    server = MCPServer(host=host, port=port, **kwargs)
    await server.initialize()
    await server.start()


if __name__ == "__main__":
    # Run server directly
    asyncio.run(run_mcp_server())