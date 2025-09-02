# MCP (Model Context Protocol) Server

The MCP server provides an embedded A2A (Application-to-Application) protocol implementation that exposes the H200 mug positioning capabilities as tools for external systems.

## Features

- **Protocol Support**: Implements MCP v1.0 with full A2A capabilities
- **Multiple Transport**: Supports both HTTP REST and WebSocket connections
- **Authentication**: JWT tokens and API key authentication with rate limiting
- **Async Operations**: All tools support async execution with configurable timeouts
- **Batch Processing**: Execute multiple tool calls in parallel or sequentially
- **Tool Discovery**: Automatic capability reporting and tool documentation
- **Streaming**: Real-time updates via WebSocket connections

## Available Tools

### 1. analyze_image
Analyze an image to detect mugs and suggest optimal positions.

**Parameters:**
- `image` (string, required): Base64-encoded image data
- `image_format` (string): Image format - jpeg, png, webp (default: jpeg)
- `apply_rules` (boolean): Apply natural language rules (default: true)
- `return_embeddings` (boolean): Return CLIP embeddings (default: false)
- `confidence_threshold` (number): Minimum detection confidence 0-1 (default: 0.5)

**Returns:**
- Analysis ID, detections, suggestions, and processing time

### 2. apply_rules
Apply natural language positioning rules.

**Parameters:**
- `rule_text` (string, required): Natural language rule
- `priority` (integer): Rule priority 1-10 (default: 5)
- `tags` (array): Tags for categorization (default: [])
- `validate_only` (boolean): Only validate without saving (default: false)

**Returns:**
- Rule ID, parsed rule, and validation results

### 3. get_suggestions
Get mug positioning suggestions based on scene analysis.

**Parameters:**
- `scene_context` (object, required): Scene context with detected objects
- `strategy` (string): Positioning strategy - safety_first, balanced, efficiency, aesthetic (default: balanced)
- `constraints` (object): Additional constraints (default: {})

**Returns:**
- Positioning suggestions with confidence scores

### 4. update_positioning
Update mug positions based on user feedback.

**Parameters:**
- `analysis_id` (string, required): Original analysis ID
- `feedback` (object, required): User feedback data
- `update_model` (boolean): Update model with feedback (default: false)

**Returns:**
- Feedback ID, updated positions, and improvement score

## Usage

### Starting the Server

```python
from src.core.mcp import MCPServer

# Create and start server
server = MCPServer(
    host="0.0.0.0",
    port=8765,
    enable_auth=True,
    enable_websocket=True,
    enable_http=True
)

await server.initialize()
await server.start()
```

### Using the Client

```python
from src.core.mcp import MCPClient
from src.core.mcp.models import MCPAuthType

# Connect to server
async with MCPClient(
    auth_type=MCPAuthType.JWT,
    auth_token="your-token"
) as client:
    
    # Analyze image
    result = await client.analyze_image(image)
    
    # Apply rule
    rule = await client.apply_rule(
        "Keep mugs away from keyboard"
    )
    
    # Get suggestions
    suggestions = await client.get_suggestions(
        scene_context=result['detections']
    )
```

### WebSocket Streaming

```python
# Connect for streaming updates
await client.connect_websocket()

async def handle_update(data):
    print(f"Received update: {data}")

await client.stream_updates(handle_update)
```

## Authentication

### JWT Tokens

```python
# Generate JWT token
authenticator = MCPAuthenticator()
token = authenticator.generate_jwt_token(
    client_id="my-app",
    scopes=["analyze", "suggest"],
    metadata={"app_version": "1.0"}
)
```

### API Keys

```python
# Generate API key
api_key = authenticator.generate_api_key(
    client_id="my-app",
    name="production-key",
    scopes=["analyze", "suggest"],
    rate_limit={"requests_per_minute": 100}
)
```

## Configuration

### Environment Variables

- `MCP_JWT_SECRET`: Secret for JWT token signing
- `MCP_SERVER_HOST`: Server host (default: 0.0.0.0)
- `MCP_SERVER_PORT`: Server port (default: 8765)
- `MCP_ENABLE_AUTH`: Enable authentication (default: true)
- `MCP_ENABLE_WEBSOCKET`: Enable WebSocket support (default: true)

### Rate Limiting

Default rate limits:
- 60 requests per minute
- 1000 requests per hour

Custom rate limits can be set per API key:

```python
api_key = authenticator.generate_api_key(
    client_id="premium-app",
    name="premium-key",
    rate_limit={
        "requests_per_minute": 300,
        "requests_per_hour": 5000
    }
)
```

## Integration Examples

### With FastAPI

```python
from fastapi import FastAPI, Depends
from src.core.mcp import MCPClient

app = FastAPI()
mcp_client = MCPClient()

@app.post("/analyze")
async def analyze_endpoint(image: UploadFile):
    # Use MCP client to analyze
    result = await mcp_client.analyze_image(
        Image.open(image.file)
    )
    return result
```

### With Celery

```python
from celery import Celery
from src.core.mcp import MCPClient

app = Celery('tasks')

@app.task
async def analyze_image_task(image_path):
    async with MCPClient() as client:
        image = Image.open(image_path)
        return await client.analyze_image(image)
```

### With External Systems

```python
# External system can connect via standard protocols
import requests

# Get capabilities
response = requests.get(
    "http://h200-server:8766/mcp/v1/capabilities",
    headers={"Authorization": "Bearer <token>"}
)

# Execute tool
response = requests.post(
    "http://h200-server:8766/mcp/v1/execute",
    json={
        "tool": "analyze_image",
        "parameters": {
            "image": base64_image,
            "apply_rules": True
        }
    },
    headers={"Authorization": "Bearer <token>"}
)
```

## Monitoring

### Health Check

```bash
curl http://localhost:8766/mcp/v1/health
```

### Statistics

```bash
curl -H "Authorization: Bearer <token>" \
     http://localhost:8766/mcp/v1/stats
```

### Prometheus Metrics

The server exposes metrics compatible with Prometheus:
- `mcp_requests_total`: Total requests by tool
- `mcp_request_duration_seconds`: Request duration histogram
- `mcp_active_connections`: Current WebSocket connections
- `mcp_auth_failures_total`: Authentication failures

## Security

1. **Authentication**: All requests require valid JWT or API key
2. **Rate Limiting**: Configurable per-client limits
3. **Input Validation**: All parameters validated with Pydantic
4. **TLS Support**: Configure with SSL certificates
5. **Scope-based Access**: Fine-grained permissions per token

## Performance

- **Cold Start**: 500ms-2s with FlashBoot
- **Tool Execution**: <500ms for most operations
- **Concurrent Requests**: Up to 1000 via WebSocket
- **Batch Processing**: Up to 100 requests per batch
- **Memory Usage**: ~500MB base + model memory

## Troubleshooting

### Connection Refused
- Check server is running: `ps aux | grep mcp`
- Verify port is open: `netstat -an | grep 8765`
- Check firewall rules

### Authentication Failed
- Verify token/API key is valid
- Check token expiration
- Ensure correct auth type is used

### Rate Limit Exceeded
- Check current limits in response headers
- Wait for reset time
- Request higher limits if needed

### Tool Execution Failed
- Check tool parameters match schema
- Verify image format is supported
- Check server logs for details