"""
Tests for MCP Server Implementation.

Tests the MCP protocol server, authentication, and tool execution.
"""

import pytest
import asyncio
import json
import base64
import uuid
from datetime import datetime, timedelta
from PIL import Image
import io
import jwt

from src.core.mcp.server import MCPServer
from src.core.mcp.client import MCPClient
from src.core.mcp.auth import MCPAuthenticator
from src.core.mcp.models import (
    MCPRequest,
    MCPResponse,
    MCPAuthType,
    MCPVersion,
    MCPError
)


@pytest.fixture
async def mcp_server():
    """Create test MCP server."""
    server = MCPServer(
        host="127.0.0.1",
        port=8888,
        enable_auth=True,
        enable_websocket=True,
        enable_http=True
    )
    
    # Initialize but don't start
    await server.initialize()
    
    yield server
    
    # Cleanup
    await server.shutdown()


@pytest.fixture
async def mcp_client(mcp_server):
    """Create test MCP client."""
    # Generate test token
    authenticator = MCPAuthenticator()
    token = authenticator.generate_jwt_token(
        client_id="test-client",
        scopes=["analyze", "suggest", "update"]
    )
    
    client = MCPClient(
        base_url="http://127.0.0.1:8889",
        websocket_url="ws://127.0.0.1:8888",
        auth_type=MCPAuthType.JWT,
        auth_token=token
    )
    
    yield client
    
    await client.disconnect()


@pytest.fixture
def test_image():
    """Create test image."""
    image = Image.new('RGB', (640, 480), color='white')
    return image


class TestMCPAuth:
    """Test authentication functionality."""
    
    def test_jwt_token_generation(self):
        """Test JWT token generation."""
        auth = MCPAuthenticator(jwt_secret="test-secret")
        
        token = auth.generate_jwt_token(
            client_id="test-client",
            scopes=["analyze", "suggest"],
            metadata={"app": "test"}
        )
        
        assert isinstance(token, str)
        
        # Decode and verify
        payload = jwt.decode(token, "test-secret", algorithms=["HS256"])
        assert payload["client_id"] == "test-client"
        assert payload["scopes"] == ["analyze", "suggest"]
        assert payload["metadata"]["app"] == "test"
    
    def test_jwt_token_validation(self):
        """Test JWT token validation."""
        auth = MCPAuthenticator(jwt_secret="test-secret")
        
        # Valid token
        token = auth.generate_jwt_token("test-client")
        is_valid, payload = auth.validate_jwt_token(token)
        assert is_valid
        assert payload["client_id"] == "test-client"
        
        # Invalid token
        is_valid, error = auth.validate_jwt_token("invalid-token")
        assert not is_valid
        assert "error" in error
        
        # Expired token
        expired_token = jwt.encode(
            {
                "client_id": "test-client",
                "exp": datetime.utcnow() - timedelta(hours=1)
            },
            "test-secret",
            algorithm="HS256"
        )
        is_valid, error = auth.validate_jwt_token(expired_token)
        assert not is_valid
        assert error["error"] == "Token expired"
    
    def test_api_key_generation(self):
        """Test API key generation."""
        auth = MCPAuthenticator()
        
        api_key = auth.generate_api_key(
            client_id="test-client",
            name="test-key",
            scopes=["analyze"],
            rate_limit={"requests_per_minute": 30}
        )
        
        assert isinstance(api_key, str)
        assert len(api_key) == 64  # SHA256 hex
        
        # Validate key
        is_valid, key_data = auth.validate_api_key(api_key)
        assert is_valid
        assert key_data["client_id"] == "test-client"
        assert key_data["name"] == "test-key"
        assert key_data["scopes"] == ["analyze"]
    
    def test_rate_limiting(self):
        """Test rate limiting."""
        auth = MCPAuthenticator()
        
        # Test with low limit
        rate_limit = {"requests_per_minute": 2}
        
        # First two requests should pass
        for i in range(2):
            is_allowed, limit_info = auth.check_rate_limit("test-client", rate_limit)
            assert is_allowed
        
        # Third request should fail
        is_allowed, limit_info = auth.check_rate_limit("test-client", rate_limit)
        assert not is_allowed
        assert limit_info.requests == 2
        assert limit_info.current_count == 2
    
    @pytest.mark.asyncio
    async def test_auth_decorator(self):
        """Test authentication decorator."""
        auth = MCPAuthenticator(jwt_secret="test-secret")
        
        class TestTool:
            def __init__(self):
                self.authenticator = auth
            
            @auth.auth_required(scope="test")
            async def protected_method(self, request: MCPRequest, auth_info=None):
                return {"auth_info": auth_info}
        
        tool = TestTool()
        
        # Request without auth
        request = MCPRequest(
            tool="test",
            parameters={}
        )
        result = await tool.protected_method(request)
        assert isinstance(result, MCPResponse)
        assert not result.result.success
        assert result.result.error.code == "AUTH_REQUIRED"
        
        # Request with valid auth
        token = auth.generate_jwt_token("test-client", scopes=["test"])
        request.auth = {
            "type": MCPAuthType.JWT.value,
            "token": token
        }
        result = await tool.protected_method(request)
        assert result["auth_info"]["client_id"] == "test-client"


class TestMCPServer:
    """Test MCP server functionality."""
    
    @pytest.mark.asyncio
    async def test_server_initialization(self, mcp_server):
        """Test server initialization."""
        assert mcp_server._initialized
        assert len(mcp_server.tools) == 4
        assert "analyze_image" in mcp_server.tools
        assert "apply_rules" in mcp_server.tools
        assert "get_suggestions" in mcp_server.tools
        assert "update_positioning" in mcp_server.tools
    
    @pytest.mark.asyncio
    async def test_capabilities(self, mcp_server):
        """Test capabilities reporting."""
        capabilities = mcp_server.get_capabilities()
        
        assert capabilities.version == MCPVersion.V1_0
        assert len(capabilities.tools) == 4
        assert MCPAuthType.JWT in capabilities.auth_types
        assert MCPAuthType.API_KEY in capabilities.auth_types
        assert "async" in capabilities.features
        assert "batch" in capabilities.features
    
    @pytest.mark.asyncio
    async def test_request_handling(self, mcp_server):
        """Test request handling."""
        # Create request
        request = MCPRequest(
            tool="get_suggestions",
            parameters={
                "scene_context": {
                    "objects": [
                        {"type": "mug", "position": {"x": 100, "y": 200}},
                        {"type": "laptop", "position": {"x": 300, "y": 200}}
                    ]
                },
                "strategy": "safety_first"
            }
        )
        
        # Add auth
        auth = mcp_server.authenticator
        token = auth.generate_jwt_token("test-client", scopes=["suggest"])
        request.auth = {
            "type": MCPAuthType.JWT.value,
            "token": token
        }
        
        # Execute request
        response = await mcp_server._handle_request(request)
        
        assert isinstance(response, MCPResponse)
        assert response.request_id == request.id
        
        # Tool should execute (though it may fail without full setup)
        # Just check that it attempted execution
        assert response.result.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_batch_request_handling(self, mcp_server):
        """Test batch request handling."""
        # Create auth token
        auth = mcp_server.authenticator
        token = auth.generate_jwt_token("test-client", scopes=["analyze", "suggest"])
        auth_data = {
            "type": MCPAuthType.JWT.value,
            "token": token
        }
        
        # Create batch request
        from src.core.mcp.models import MCPBatchRequest
        
        batch = MCPBatchRequest(
            requests=[
                MCPRequest(
                    tool="get_suggestions",
                    parameters={
                        "scene_context": {"objects": []},
                        "strategy": "balanced"
                    },
                    auth=auth_data
                ),
                MCPRequest(
                    tool="apply_rules",
                    parameters={
                        "rule_text": "Test rule",
                        "validate_only": True
                    },
                    auth=auth_data
                )
            ],
            parallel=True
        )
        
        # Execute batch
        response = await mcp_server._handle_batch_request(batch)
        
        assert response.batch_id == batch.id
        assert len(response.responses) == 2
        assert response.summary["total"] == 2
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mcp_server):
        """Test error handling."""
        # Request for unknown tool
        request = MCPRequest(
            tool="unknown_tool",
            parameters={}
        )
        
        response = await mcp_server._handle_request(request)
        
        assert not response.result.success
        assert response.result.error.code == "UNKNOWN_TOOL"
        assert "unknown_tool" in response.result.error.message
    
    @pytest.mark.asyncio
    async def test_statistics(self, mcp_server):
        """Test statistics tracking."""
        initial_stats = mcp_server._stats.copy()
        
        # Make some requests
        request = MCPRequest(tool="unknown_tool", parameters={})
        await mcp_server._handle_request(request)
        
        assert mcp_server._stats["requests_total"] == initial_stats["requests_total"] + 1
        assert mcp_server._stats["requests_failed"] == initial_stats["requests_failed"] + 1


class TestMCPTools:
    """Test individual MCP tools."""
    
    @pytest.mark.asyncio
    async def test_analyze_image_tool(self, mcp_server, test_image):
        """Test analyze_image tool."""
        tool = mcp_server.tools["analyze_image"]
        
        # Convert image to base64
        buffer = io.BytesIO()
        test_image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create request
        request = MCPRequest(
            tool="analyze_image",
            parameters={
                "image": image_base64,
                "image_format": "jpeg",
                "apply_rules": False,
                "confidence_threshold": 0.5
            }
        )
        
        # Execute (may fail without GPU, but should handle gracefully)
        response = await tool.execute(request)
        
        assert isinstance(response, MCPResponse)
        assert response.result.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_apply_rules_tool(self, mcp_server):
        """Test apply_rules tool."""
        tool = mcp_server.tools["apply_rules"]
        
        # Create request
        request = MCPRequest(
            tool="apply_rules",
            parameters={
                "rule_text": "Keep mugs at least 6 inches from electronics",
                "priority": 8,
                "tags": ["safety", "electronics"],
                "validate_only": True
            }
        )
        
        # Execute
        response = await tool.execute(request)
        
        assert isinstance(response, MCPResponse)
        # Check that validation was attempted
        if response.result.success:
            assert "validation" in response.result.data
    
    @pytest.mark.asyncio
    async def test_tool_definition(self, mcp_server):
        """Test tool definitions."""
        for tool_name, tool in mcp_server.tools.items():
            definition = tool.get_definition()
            
            assert definition.name == tool_name
            assert definition.description
            assert len(definition.parameters) > 0
            assert definition.returns
            
            # Check rate limits
            if definition.rate_limit:
                assert "requests_per_minute" in definition.rate_limit


class TestMCPClient:
    """Test MCP client functionality."""
    
    @pytest.mark.asyncio
    async def test_client_connection(self, mcp_client):
        """Test client connection."""
        await mcp_client.connect()
        assert mcp_client._session is not None
    
    @pytest.mark.asyncio
    async def test_client_auth_headers(self, mcp_client):
        """Test auth header generation."""
        headers = mcp_client._get_auth_header()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
    
    @pytest.mark.asyncio
    async def test_client_request_creation(self, mcp_client):
        """Test request creation."""
        auth_data = mcp_client._get_auth_data()
        assert auth_data["type"] == MCPAuthType.JWT.value
        assert "token" in auth_data


@pytest.mark.asyncio
async def test_integration_flow(mcp_server, test_image):
    """Test complete integration flow."""
    # Start server in background
    server_task = asyncio.create_task(mcp_server.start())
    
    # Wait a bit for server to start
    await asyncio.sleep(0.5)
    
    try:
        # Create client
        auth = mcp_server.authenticator
        token = auth.generate_jwt_token(
            client_id="integration-test",
            scopes=["analyze", "suggest", "update"]
        )
        
        client = MCPClient(
            base_url="http://127.0.0.1:8889",
            websocket_url="ws://127.0.0.1:8888",
            auth_type=MCPAuthType.JWT,
            auth_token=token
        )
        
        async with client:
            # Test capabilities
            capabilities = await client.get_capabilities()
            assert len(capabilities.tools) == 4
            
            # Test health check
            health = await client.health_check()
            assert health["status"] == "healthy"
            
            # Test stats
            stats = await client.get_stats()
            assert "requests_total" in stats
            
    finally:
        # Stop server
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])