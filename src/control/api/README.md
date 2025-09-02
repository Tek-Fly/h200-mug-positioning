# H200 FastAPI Control Plane

This is the FastAPI application for the H200 Intelligent Mug Positioning System control plane.

## Features

- **Image Analysis API** - Analyze mug positions with feedback
- **Natural Language Rules** - Create positioning rules from plain English
- **Real-time Dashboard** - System metrics and monitoring
- **Server Control** - Manage RunPod deployments
- **WebSocket Support** - Real-time updates and notifications

## API Endpoints

### Authentication
All endpoints (except health and docs) require a JWT token in the Authorization header:
```
Authorization: Bearer <token>
```

### Core Endpoints

#### Image Analysis
```
POST /api/v1/analyze/with-feedback
Content-Type: multipart/form-data

Parameters:
- image: Image file (required)
- include_feedback: Boolean (default: true)
- rules_context: String (optional) - Natural language rules
- calibration_mm_per_pixel: Float (optional)
- confidence_threshold: Float (default: 0.7)

Response: AnalysisResponse with detections, positioning, and feedback
```

#### Natural Language Rules
```
POST /api/v1/rules/natural-language
Content-Type: application/json

Body:
{
  "text": "The mug should be centered on the coaster",
  "context": "optional additional context",
  "auto_enable": false
}

Response: Created rule with interpretation
```

#### Dashboard
```
GET /api/v1/dashboard?include_metrics=true&include_health=true

Response: Complete system dashboard with:
- Service health status
- Performance metrics
- Resource usage
- Cost tracking
- Recent activity
```

#### Server Control
```
POST /api/v1/servers/{server_type}/control
Content-Type: application/json

Body:
{
  "action": "start|stop|restart|scale",
  "config": {...},  // Optional for scale action
  "force": false,
  "wait_for_completion": true
}

Response: Server control result
```

### WebSocket Connection
```
ws://localhost:8000/ws/control-plane?token=<jwt_token>

Subscribe to topics:
{
  "action": "subscribe",
  "topic": "metrics|logs|alerts|activity"
}
```

## Quick Start

1. **Set Environment Variables**
```bash
export SECRET_KEY="your-secret-key"
export MONGODB_ATLAS_URI="mongodb+srv://..."
export REDIS_URL="redis://..."
# See .env.example for all variables
```

2. **Run the API**
```bash
# Development mode with auto-reload
uvicorn src.control.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.control.api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

3. **Access Documentation**
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc
- OpenAPI JSON: http://localhost:8000/api/openapi.json

## Configuration

The API is configured via environment variables or a `.env` file. Key settings:

- `SECRET_KEY` - JWT signing key (required)
- `DEBUG` - Enable debug mode
- `CORS_ORIGINS` - Comma-separated allowed origins
- `RATE_LIMIT_CALLS` - Rate limit per period (default: 100)
- `RATE_LIMIT_PERIOD` - Period in seconds (default: 60)

## Middleware

The API includes several middleware components:

1. **Authentication** - JWT-based auth for all protected endpoints
2. **Rate Limiting** - Configurable per-client rate limits
3. **Logging** - Structured JSON logging with request IDs
4. **CORS** - Configurable cross-origin support
5. **Prometheus Metrics** - Available at `/api/metrics`

## Error Handling

All errors follow a consistent JSON structure:
```json
{
  "error": {
    "code": 400,
    "message": "Detailed error message",
    "type": "validation_error",
    "details": {...}  // Optional additional details
  }
}
```

## Development

### Running Tests
```bash
# Test API initialization
python src/control/api/test_api.py

# Run full test suite
pytest tests/control/api/ -v
```

### Code Quality
```bash
# Linting
pylint src/control/

# Type checking
mypy src/control/ --strict

# Format code
black src/control/
```

## Docker Support

Build and run with Docker:
```bash
# Build image
docker build -t h200-api -f docker/api.Dockerfile .

# Run container
docker run -p 8000:8000 --env-file .env h200-api
```

## Performance Targets

- Cold start: 500ms-2s (with FlashBoot)
- Warm API response: <200ms (p95)
- WebSocket latency: <50ms
- Concurrent connections: 1000+
- Requests per second: 100+

## Security

- JWT tokens with configurable expiration
- Rate limiting per user/IP
- Input validation on all endpoints
- CORS protection
- TLS termination at load balancer
- No sensitive data in logs

## Monitoring

- Health check: `/api/health`
- Prometheus metrics: `/api/metrics`
- Structured JSON logs
- Request ID tracking
- Performance timing headers