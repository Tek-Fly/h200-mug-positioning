# API Documentation

The H200 Intelligent Mug Positioning System provides a comprehensive REST API with WebSocket support for real-time updates.

## Base URLs

- **Production**: `https://your-domain.com/api/v1`
- **Local Development**: `http://localhost:8000/api/v1`
- **RunPod**: `http://[POD_ID]-8000.proxy.runpod.net/api/v1`

## Authentication

All API endpoints require JWT token authentication except for health checks and documentation endpoints.

### Getting a Token

```bash
curl -X POST "/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

### Using the Token

Include the token in the `Authorization` header:

```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" "/api/v1/dashboard"
```

## Endpoints Overview

| Category | Endpoint | Description |
|----------|----------|-------------|
| **Analysis** | `POST /analyze/with-feedback` | Analyze image and get positioning feedback |
| **Analysis** | `POST /analyze/feedback` | Submit feedback for analysis results |
| **Analysis** | `POST /analyze/batch` | Batch process multiple images |
| **Rules** | `POST /rules/natural-language` | Create rules from natural language |
| **Rules** | `GET /rules` | List all positioning rules |
| **Rules** | `POST /rules` | Create a new rule |
| **Rules** | `PATCH /rules/{id}` | Update existing rule |
| **Rules** | `DELETE /rules/{id}` | Delete a rule |
| **Rules** | `POST /rules/evaluate` | Evaluate rules against data |
| **Dashboard** | `GET /dashboard` | Get complete dashboard data |
| **Dashboard** | `GET /dashboard/metrics/{name}` | Get specific metric history |
| **Servers** | `POST /servers/{type}/control` | Control server lifecycle |
| **Servers** | `GET /servers` | List all servers |
| **Servers** | `POST /servers/deploy` | Deploy new server |
| **Servers** | `GET /servers/{id}/metrics` | Get server metrics |
| **Servers** | `POST /servers/{id}/logs` | Get server logs |
| **WebSocket** | `WS /ws/control-plane` | Real-time updates |

## Quick Examples

### Analyze an Image

```bash
curl -X POST "/api/v1/analyze/with-feedback" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@mug_photo.jpg" \
  -F "include_feedback=true" \
  -F "confidence_threshold=0.8"
```

### Create a Rule from Natural Language

```bash
curl -X POST "/api/v1/rules/natural-language" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The mug should be centered on the coaster",
    "auto_enable": true
  }'
```

### Get System Dashboard

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" "/api/v1/dashboard"
```

## Rate Limiting

- **Default**: 100 requests per minute per user
- **Burst**: Up to 200 requests in 30 seconds
- **Headers**: Rate limit info included in response headers

## Error Handling

All errors follow a consistent format:

```json
{
  "error": {
    "code": 400,
    "message": "Validation error",
    "type": "validation_error",
    "details": [...]
  }
}
```

## WebSocket Real-Time Updates

Connect to `/ws/control-plane` for real-time system updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/control-plane?token=YOUR_TOKEN');

// Subscribe to metrics updates
ws.send(JSON.stringify({
  action: 'subscribe',
  topic: 'metrics'
}));
```

## Interactive Documentation

- **Swagger UI**: `/api/docs`
- **ReDoc**: `/api/redoc`
- **OpenAPI Spec**: `/api/openapi.json`

## Next Steps

- **[Detailed Endpoints](./endpoints.md)** - Complete endpoint documentation
- **[Request/Response Models](./models.md)** - Data structure specifications
- **[Error Codes](../reference/error-codes.md)** - All possible error responses
- **[Client Libraries](./client-libraries.md)** - SDKs and code examples