# API Endpoints Reference

This document provides detailed information about all API endpoints in the H200 Intelligent Mug Positioning System.

> **In Simple Terms**: This API lets you send pictures to our AI system, and it tells you where coffee mugs are located and if they're placed safely. Think of it like a smart assistant that looks at your desk and warns you if your coffee is too close to your laptop!

## Image Analysis Endpoints

### Analyze Image with Feedback

**`POST /api/v1/analyze/with-feedback`**

Analyzes an uploaded image to detect mugs and provide positioning feedback based on configured rules.

**What it does**: You upload a photo, and the system finds all coffee mugs in it, then tells you if they're positioned safely or if they need to be moved.

#### Example CURL Request

```bash
# For Timed Pod deployment
POD_URL="http://your-pod-id-8000.proxy.runpod.net"
JWT_TOKEN="your-jwt-token-here"

curl -X POST ${POD_URL}/api/v1/analyze/with-feedback \
  -H "Authorization: Bearer ${JWT_TOKEN}" \
  -F "image=@coffee-mug.jpg" \
  -F "include_feedback=true" \
  -F "confidence_threshold=0.7"
```

#### Request

**Content-Type**: `multipart/form-data`

| Parameter | Type | Required | Description | In Simple Terms |
|-----------|------|----------|-------------|-----------------|
| `image` | File | Yes | Image file (JPG, PNG, WebP) | The photo you want to analyze |
| `include_feedback` | Boolean | No | Include positioning feedback (default: true) | Whether you want advice on better mug placement |
| `rules_context` | String | No | Natural language rules context | Special rules like "keep mugs 6 inches from laptops" |
| `calibration_mm_per_pixel` | Float | No | Calibration factor for measurements | Helps convert pixels to real-world distances |
| `confidence_threshold` | Float | No | Detection confidence threshold (default: 0.7) | How sure the AI needs to be (0.7 = 70% sure) |

#### Response

**What you get back**: A detailed report about all mugs found, their positions, and safety recommendations.

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",  // Unique ID for this analysis
  "timestamp": "2025-09-02T10:30:00Z",                   // When the analysis was done
  "processing_time_ms": 450.5,                           // How long it took (milliseconds)
  
  "detections": [                                        // All mugs found in the image
    {
      "id": "det_001",                                   // Unique ID for this mug
      "bbox": {                                          // Where the mug is located
        "top_left": {"x": 100, "y": 50},               // Top-left corner coordinates
        "bottom_right": {"x": 200, "y": 150},          // Bottom-right corner coordinates
        "confidence": 0.95                               // 95% sure this is a mug
      },
      "attributes": {                                    // What the mug looks like
        "color": "white",
        "size": "medium"
      }
    }
  ],
  
  "positioning": {                                       // How well the mug is positioned
    "position": "slightly off-center",                  // Description of placement
    "confidence": 0.87,                                  // 87% confidence in this assessment
    "offset_pixels": {"x": 15, "y": -5},               // How far from ideal position (in pixels)
    "offset_mm": {"x": 7.5, "y": -2.5},                // Same distance in millimeters
    "rule_violations": []                                // Any safety rules broken
  },
  
  "feedback": "Mug detected successfully. Position is good with minor offset.",
  "suggestions": [                                       // What to do for better placement
    "Move mug left by 15 pixels to center perfectly"
  ],
  
  "metadata": {                                          // Extra technical info
    "user_id": "user123",
    "image_size": [1920, 1080],
    "model_version": "yolov8n-1.0"
  }
}
```

#### Error Responses

- `400 Bad Request`: Invalid image format or parameters
- `413 Payload Too Large`: Image file too large (max 10MB)
- `422 Unprocessable Entity`: Invalid request data
- `503 Service Unavailable`: Model not initialized

### Submit Analysis Feedback

**`POST /api/v1/analyze/feedback`**

Submit feedback on the accuracy of an analysis result to improve future predictions.

#### Request

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "is_correct": true,
  "feedback_type": "positioning",
  "comments": "Detection was accurate, positioning suggestion was helpful",
  "suggested_improvements": [
    "Consider ambient lighting in analysis"
  ]
}
```

#### Response

```json
{
  "success": true,
  "message": "Feedback submitted successfully"
}
```

## Rules Management Endpoints

### Create Rule from Natural Language

**`POST /api/v1/rules/natural-language`**

Creates positioning rules from natural language descriptions using LangChain processing.

#### Example CURL Request

```bash
# Create a positioning rule from natural language
curl -X POST ${POD_URL}/api/v1/rules/natural-language \
  -H "Authorization: Bearer ${JWT_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The mug should be centered on the coaster with at least 1 inch clearance from edges",
    "context": "coffee_shop_setup",
    "auto_enable": true
  }'
```

#### Request

```json
{
  "text": "The mug should be centered on the coaster with at least 1 inch clearance from edges",
  "context": "coffee_shop_setup",
  "auto_enable": true
}
```

#### Response

```json
{
  "rule": {
    "id": "rule_123",
    "name": "Center mug on coaster",
    "description": "The mug should be centered on the coaster with at least 1 inch clearance from edges",
    "type": "positioning",
    "priority": "medium",
    "conditions": [
      {
        "field": "distance_from_center",
        "operator": "less_than",
        "value": 0.5,
        "unit": "inches"
      }
    ],
    "actions": [
      {
        "type": "alert",
        "parameters": {
          "message": "Mug should be centered on coaster"
        }
      }
    ],
    "enabled": true,
    "created_at": "2025-09-02T10:30:00Z",
    "metadata": {
      "created_by": "user123",
      "natural_language_source": "The mug should be centered...",
      "parser_confidence": 0.92
    }
  },
  "interpretation": "Created positioning rule: Center mug on coaster with 1 condition(s)",
  "confidence": 0.92,
  "warnings": []
}
```

### List Rules

**`GET /api/v1/rules`**

List all positioning rules with optional filtering.

#### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled_only` | Boolean | Return only enabled rules |
| `rule_type` | String | Filter by rule type |

#### Response

```json
[
  {
    "id": "rule_123",
    "name": "Center mug on coaster",
    "description": "Mug positioning rule",
    "type": "positioning",
    "priority": "medium",
    "conditions": [...],
    "actions": [...],
    "enabled": true,
    "created_at": "2025-09-02T10:30:00Z",
    "updated_at": "2025-09-02T10:30:00Z",
    "metadata": {...}
  }
]
```

### Get Specific Rule

**`GET /api/v1/rules/{rule_id}`**

Retrieve detailed information about a specific rule.

### Create Rule

**`POST /api/v1/rules`**

Create a new positioning rule with explicit conditions and actions.

### Update Rule

**`PATCH /api/v1/rules/{rule_id}`**

Update an existing rule. Only provided fields will be updated.

### Delete Rule

**`DELETE /api/v1/rules/{rule_id}`**

Permanently delete a positioning rule.

### Evaluate Rules

**`POST /api/v1/rules/evaluate`**

Test rules against positioning data without affecting the database.

## Dashboard Endpoints

### Get Dashboard Data

**`GET /api/v1/dashboard`**

Retrieve comprehensive system status and metrics.

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_metrics` | Boolean | true | Include performance metrics |
| `include_health` | Boolean | true | Include service health status |
| `include_resources` | Boolean | true | Include resource usage |
| `include_costs` | Boolean | true | Include cost tracking |
| `include_activity` | Boolean | true | Include recent activity |
| `activity_limit` | Integer | 50 | Number of activity entries |

#### Response

```json
{
  "timestamp": "2025-09-02T10:30:00Z",
  "overall_health": "healthy",
  "services": [
    {
      "name": "MongoDB",
      "status": "healthy",
      "last_check": "2025-09-02T10:29:45Z",
      "uptime_seconds": 86400,
      "details": {
        "version": "7.0.0",
        "uptime": 86400
      }
    }
  ],
  "performance": {
    "cold_start_ms": 1500.0,
    "warm_start_ms": 50.0,
    "image_processing_ms": 450.0,
    "api_latency_p95_ms": 180.0,
    "gpu_utilization_percent": 75.5,
    "cache_hit_rate": 87.3,
    "requests_per_second": 25.5
  },
  "resources": {
    "cpu_percent": 45.2,
    "memory_used_mb": 2048,
    "memory_total_mb": 8192,
    "disk_used_gb": 15.5,
    "disk_total_gb": 100,
    "gpu_memory_used_mb": 12000,
    "gpu_memory_total_mb": 20480
  },
  "costs": {
    "period": "daily",
    "compute_cost": 84.0,
    "storage_cost": 0.08,
    "network_cost": 0.90,
    "total_cost": 84.98,
    "cost_per_request": 0.0085,
    "breakdown": {
      "gpu_hours": 24,
      "storage_gb": 100,
      "network_gb": 10
    }
  },
  "recent_activity": [
    {
      "timestamp": "2025-09-02T10:25:00Z",
      "type": "analysis",
      "user_id": "user123",
      "action": "Image analysis completed",
      "details": {
        "request_id": "req_456"
      },
      "duration_ms": 450
    }
  ],
  "summary": {
    "total_requests_today": 12345,
    "average_response_time_ms": 180,
    "error_rate_percent": 0.5,
    "active_users": 42,
    "active_servers": 2,
    "total_servers": 3
  },
  "metrics": [
    {
      "name": "api_requests_total",
      "type": "counter",
      "value": 12345,
      "timestamp": "2025-09-02T10:30:00Z"
    }
  ]
}
```

### Get Metric History

**`GET /api/v1/dashboard/metrics/{metric_name}`**

Retrieve historical data for a specific metric.

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_range` | String | "1h" | Time range: 1h, 6h, 1d, 7d, 30d |

## Server Control Endpoints

### Control Server Lifecycle

**`POST /api/v1/servers/{server_type}/control`**

Control server operations (start, stop, restart, scale).

#### Path Parameters

- `server_type`: `serverless` or `timed`

#### Request

```json
{
  "action": "start",
  "force": false,
  "config": {
    "min_instances": 0,
    "max_instances": 3,
    "idle_timeout": 600
  }
}
```

#### Response

```json
{
  "success": true,
  "server": {
    "id": "srv_123",
    "type": "serverless",
    "state": "running",
    "endpoint": "http://pod-123-8000.proxy.runpod.net",
    "health": "healthy",
    "created_at": "2025-09-02T10:00:00Z",
    "config": {...},
    "metrics": {...}
  },
  "message": "Serverless server started successfully",
  "duration_seconds": 5.2,
  "warnings": []
}
```

### List Servers

**`GET /api/v1/servers`**

List all configured servers with their current status.

### Deploy New Server

**`POST /api/v1/servers/deploy`**

Deploy a new server instance to RunPod infrastructure.

### Get Server Metrics

**`GET /api/v1/servers/{server_id}/metrics`**

Get real-time performance metrics for a specific server.

### Get Server Logs

**`POST /api/v1/servers/{server_id}/logs`**

Retrieve logs from a specific server.

## WebSocket API

### Control Plane WebSocket

**`WS /ws/control-plane?token=YOUR_JWT_TOKEN`**

Real-time updates for system monitoring and control. When deployed on RunPod, use the pod proxy URL:

**RunPod WebSocket URL**: `ws://[POD_ID]-8000.proxy.runpod.net/ws/control-plane?token=YOUR_JWT_TOKEN`

#### Available Topics

- `metrics`: Performance metrics updates (every 5s)
- `logs`: Live log streaming (every 10s)
- `alerts`: System alerts and notifications
- `activity`: User activity updates

#### Protocol

1. **Connect**: Include JWT token in query parameters
2. **Subscribe**: Send subscription message
3. **Receive**: Get real-time updates for subscribed topics

#### Example Messages

**Subscribe to metrics:**
```json
{
  "action": "subscribe",
  "topic": "metrics"
}
```

**Metrics update:**
```json
{
  "type": "metrics",
  "timestamp": "2025-09-02T10:30:00Z",
  "data": {
    "gpu_utilization": 75.5,
    "requests_per_second": 25.5,
    "average_latency_ms": 145,
    "cache_hit_rate": 0.87
  }
}
```

## Response Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 413 | Payload Too Large | File upload too large |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

## Serverless Endpoint Examples

### Analyze Image (Serverless)

For serverless deployments, use the RunPod serverless API:

```bash
# Set your endpoint ID and API key
ENDPOINT_ID="your-endpoint-id"
RUNPOD_API_KEY="your-runpod-api-key"

# Option 1: Analyze with image URL
curl -X POST https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_url": "https://example.com/coffee-mug.jpg",
      "apply_rules": true,
      "confidence_threshold": 0.7
    }
  }'

# Option 2: Analyze with base64 image
IMAGE_BASE64=$(base64 -i coffee-mug.jpg | tr -d '\n')
curl -X POST https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"image\": \"${IMAGE_BASE64}\",
      \"apply_rules\": true
    }
  }"
```

## Health Check

**`GET /api/health`**

Public endpoint for service health monitoring (no authentication required).

```bash
# Check service health (Timed Pod only)
curl ${POD_URL}/api/health
```

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "mongodb": true,
    "redis": true,
    "models": true,
    "orchestrator": true
  },
  "orchestrator": {
    "servers": 2,
    "auto_shutdown": true,
    "notifications": 5
  }
}
```

## Performance Considerations

- **Cold Start**: 500ms-2s with FlashBoot enabled on RunPod
- **Image Size**: Recommend max 1920x1080 for optimal processing
- **Batch Processing**: Use batch endpoints for multiple images
- **Caching**: Results cached for 5 minutes by default
- **Rate Limits**: Adjust request frequency based on limits
- **GPU Memory**: Large images may require more GPU memory on H200

## Security Notes

- All endpoints require authentication except `/api/health` and docs
- JWT tokens expire after 24 hours
- Rate limiting applies per authenticated user
- File uploads are validated for type and size
- All requests logged for audit purposes

## Next Steps

- **[Request/Response Models](./models.md)** - Detailed data structures
- **[Client Libraries](./client-libraries.md)** - SDKs and examples
- **[Authentication Guide](../user-guides/authentication.md)** - Auth setup