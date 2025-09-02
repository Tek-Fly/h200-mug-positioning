# H200 Intelligent Mug Positioning System - User Manual

## Table of Contents
1. [Quick Start](#quick-start)
2. [Complete Setup Guide](#complete-setup-guide)
3. [API Usage Guide](#api-usage-guide)
4. [Dashboard Guide](#dashboard-guide)
5. [Rule Management](#rule-management)
6. [Deployment Guide](#deployment-guide)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Python 3.11+ (for local development)
- RunPod account with API key
- MongoDB Atlas account
- Cloudflare R2 account
- Redis instance (or use Docker)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/h200-mug-positioning.git
cd h200-mug-positioning

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

### 2. Start Local Development

```bash
# Start all services with Docker
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs
docker-compose logs -f
```

### 3. Access the System

- **Dashboard**: http://localhost:3000
- **API Docs**: http://localhost:8000/api/docs
- **Monitoring**: http://localhost:3000/monitoring (admin/admin123)

---

## Complete Setup Guide

### Step 1: Environment Configuration

Create your `.env` file with the following values:

```bash
# MongoDB Atlas
MONGODB_ATLAS_URI=mongodb+srv://username:password@cluster.mongodb.net/h200_production

# RunPod (Get from https://runpod.io/console/settings)
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_SERVERLESS_ENDPOINT_ID=auto_generated
RUNPOD_TIMED_POD_ID=auto_generated
RUNPOD_NETWORK_VOLUME_ID=auto_generated

# Cloudflare R2 (Get from Cloudflare dashboard)
R2_ENDPOINT_URL=https://account-id.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
R2_BUCKET_NAME=h200-backup

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your_secure_redis_password
REDIS_DB=0

# Google Cloud (Optional, for secret management)
GCP_PROJECT_ID=your_project_id
GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-key.json

# Security
JWT_SECRET=your_very_secure_jwt_secret_key_here

# Docker Hub (for image registry)
DOCKER_USERNAME=your_docker_username

# OpenAI (for rule processing)
OPENAI_API_KEY=your_openai_api_key

# Webhooks (Optional)
WEBHOOK_URL=https://your-webhook-endpoint.com/webhook
N8N_WEBHOOK_URL=https://your-n8n-instance.com/webhook/abc123
TEMPLATED_API_KEY=your_templated_api_key
```

### Step 2: Install Dependencies (Local Development)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Step 3: Database Setup

```bash
# MongoDB Atlas Setup:
# 1. Create a cluster at https://cloud.mongodb.com
# 2. Add your IP to Network Access
# 3. Create a database user
# 4. Get your connection string

# Redis Setup (if not using Docker):
# Option 1: Use Redis Cloud (https://redis.com/cloud/)
# Option 2: Install locally: brew install redis (macOS)
```

### Step 4: Build and Deploy

```bash
# Build Docker images
./scripts/deploy/build_and_push.sh

# Deploy to RunPod
python scripts/deploy/deploy_to_runpod.py both
```

---

## API Usage Guide

### Authentication

All API requests require a JWT token. First, get your token:

```bash
# Get JWT token (replace with your credentials)
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your_password"}'

# Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

### Image Analysis

#### Analyze a Single Image

```bash
# Analyze image for mug positioning
curl -X POST http://localhost:8000/api/v1/analyze/with-feedback \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "image=@/path/to/your/image.jpg" \
  -F "confidence_threshold=0.7" \
  -F "include_feedback=true"

# Response:
{
  "analysis_id": "ana_123456",
  "timestamp": "2025-01-09T12:34:56Z",
  "detections": [
    {
      "class": "mug",
      "confidence": 0.95,
      "bbox": [100, 200, 150, 250],
      "position": {
        "x": 125,
        "y": 225,
        "distance_to_edge": 50
      }
    }
  ],
  "positioning": {
    "optimal_position": {
      "x": 150,
      "y": 225
    },
    "adjustments_needed": [
      {
        "type": "move_right",
        "distance_mm": 25,
        "reason": "Too close to laptop"
      }
    ],
    "safety_score": 0.85
  },
  "feedback": {
    "summary": "Mug detected but positioned too close to electronic device",
    "recommendations": [
      "Move mug 25mm to the right",
      "Consider using a coaster"
    ]
  },
  "processing_time_ms": 156
}
```

#### Batch Analysis

```bash
# Analyze multiple images
curl -X POST http://localhost:8000/api/v1/analyze/batch \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "images=@image3.jpg"
```

### Rule Management

#### Create a Natural Language Rule

```bash
# Create a rule using natural language
curl -X POST http://localhost:8000/api/v1/rules/natural-language \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Keep mugs at least 6 inches away from laptops",
    "context": "office",
    "auto_enable": true
  }'

# Response:
{
  "rule_id": "rule_789",
  "name": "Laptop Safety Distance",
  "type": "distance",
  "conditions": [
    {
      "object1": "mug",
      "object2": "laptop",
      "operator": "min_distance",
      "value": 152.4,
      "unit": "mm"
    }
  ],
  "actions": [
    {
      "type": "maintain_distance",
      "parameters": {
        "distance": 152.4,
        "unit": "mm"
      }
    }
  ],
  "priority": 8,
  "enabled": true,
  "created_at": "2025-01-09T12:35:00Z"
}
```

#### List Active Rules

```bash
# Get all active rules
curl -X GET http://localhost:8000/api/v1/rules?enabled=true \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### Update Rule

```bash
# Enable/disable a rule
curl -X PATCH http://localhost:8000/api/v1/rules/rule_789 \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

### Server Management

#### Control Servers

```bash
# Start serverless endpoint
curl -X POST http://localhost:8000/api/v1/servers/serverless/control \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action": "start"}'

# Stop timed GPU
curl -X POST http://localhost:8000/api/v1/servers/timed/control \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action": "stop"}'

# Get server status
curl -X GET http://localhost:8000/api/v1/servers/status \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Dashboard Data

```bash
# Get complete dashboard metrics
curl -X GET http://localhost:8000/api/v1/dashboard \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Response includes:
# - System health status
# - GPU metrics (utilization, memory, temperature)
# - Performance metrics (latency, throughput)
# - Cost tracking
# - Recent activity
```

### WebSocket Real-time Updates

```javascript
// Connect to WebSocket for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws/control-plane');

ws.onopen = () => {
  // Authenticate
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'YOUR_JWT_TOKEN'
  }));
  
  // Subscribe to topics
  ws.send(JSON.stringify({
    type: 'subscribe',
    topics: ['metrics', 'alerts', 'activity']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
  // Handle real-time updates
};
```

---

## Dashboard Guide

### Accessing the Dashboard

1. Open http://localhost:3000 in your browser
2. Login with your credentials
3. You'll see the main dashboard with real-time metrics

### Dashboard Features

#### 1. System Overview
- **Health Status**: Green/Yellow/Red indicators for all services
- **GPU Metrics**: Real-time GPU utilization, memory, and temperature
- **Performance**: Request latency, throughput, and cache hit rates
- **Cost Tracking**: Current usage costs and projections

#### 2. Image Analysis
- **Upload Images**: Drag and drop or click to upload
- **View Results**: Interactive visualization of mug positions
- **Analysis History**: View and export past analyses
- **Batch Processing**: Upload multiple images at once

#### 3. Rule Management
- **Create Rules**: Use natural language to define positioning rules
- **View Active Rules**: See all enabled rules with performance stats
- **Test Rules**: Test rules against sample images
- **Rule History**: Track rule changes and effectiveness

#### 4. Server Control
- **Server Status**: View status of serverless and timed instances
- **Start/Stop**: Control server lifecycle with one click
- **Auto-shutdown**: Configure idle timeout settings
- **Logs**: View real-time server logs

#### 5. Settings
- **Theme**: Switch between light/dark/system themes
- **Notifications**: Configure alert preferences
- **API Keys**: Manage API access (admin only)
- **Export Data**: Download analysis history and metrics

---

## Rule Management

### Understanding Rules

Rules define how mugs should be positioned. The system supports:

1. **Distance Rules**: Minimum/maximum distance between objects
2. **Placement Rules**: Where mugs should be placed
3. **Safety Rules**: Keep mugs away from hazards
4. **Context Rules**: Different rules for different environments

### Creating Rules

#### Natural Language Examples

```text
"Keep mugs at least 6 inches from laptops"
"Place all beverages on coasters when available"
"Center the mug on the desk mat"
"Keep hot beverages away from the keyboard"
"Group multiple mugs together"
"Avoid placing mugs near the edge of the table"
```

#### Rule Priority

- **1-3**: Low priority (suggestions)
- **4-6**: Medium priority (recommendations)
- **7-9**: High priority (warnings)
- **10**: Critical (safety alerts)

### Managing Rules via API

```python
import requests

# Create a rule
response = requests.post(
    "http://localhost:8000/api/v1/rules/natural-language",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "text": "Keep mugs centered on coasters",
        "context": "office",
        "priority": 7
    }
)
rule = response.json()

# Test a rule
response = requests.post(
    f"http://localhost:8000/api/v1/rules/{rule['rule_id']}/test",
    headers={"Authorization": f"Bearer {token}"},
    files={"image": open("test_image.jpg", "rb")}
)
```

---

## Deployment Guide

### Local Development

```bash
# Start development environment
docker-compose up -d

# Watch logs
docker-compose logs -f h200-timed

# Run tests
make test

# Stop services
docker-compose down
```

### Production Deployment

#### 1. Build and Push Images

```bash
# Build multi-platform images
./scripts/deploy/build_and_push.sh --platforms linux/amd64,linux/arm64

# Verify images
docker images | grep h200
```

#### 2. Deploy to RunPod

```bash
# Deploy both serverless and timed instances
python scripts/deploy/deploy_to_runpod.py both --gpu-type H100

# Deploy only serverless
python scripts/deploy/deploy_to_runpod.py serverless

# Deploy with custom configuration
python scripts/deploy/deploy_to_runpod.py timed \
  --gpu-type A100 \
  --gpu-count 1 \
  --idle-timeout 600
```

#### 3. Verify Deployment

```bash
# Check deployment status
python scripts/deploy/deploy_to_runpod.py status

# Run health checks
./scripts/deploy/health_check.sh --endpoint https://your-endpoint.runpod.io
```

### Deployment Strategies

#### Blue-Green Deployment

```bash
# Deploy new version alongside current
python scripts/deploy/deploy_to_runpod.py both \
  --strategy blue-green \
  --health-check-url http://localhost:8000/health
```

#### Canary Deployment

```bash
# Gradually roll out to 20% of traffic
python scripts/deploy/deploy_to_runpod.py both \
  --strategy canary \
  --canary-percent 20 \
  --canary-duration 3600
```

---

## Monitoring & Maintenance

### Accessing Monitoring

1. **Grafana**: http://localhost:3000/monitoring
   - Username: admin
   - Password: admin123 (change on first login)

2. **Prometheus**: http://localhost:9091

3. **AlertManager**: http://localhost:9093

### Key Metrics to Monitor

#### System Health
- **API Latency**: Should be <200ms (p95)
- **GPU Utilization**: Target 70-80%
- **Cache Hit Rate**: Should be >85%
- **Error Rate**: Should be <1%

#### Cost Optimization
- **Idle Time**: Monitor for auto-shutdown
- **Request Rate**: Adjust scaling based on load
- **GPU Memory**: Optimize batch sizes

### Maintenance Tasks

#### Daily
- Check system health dashboard
- Review error logs
- Monitor cost trends

#### Weekly
- Review and optimize rules
- Clean up old analysis data
- Update monitoring thresholds

#### Monthly
- Rotate API keys and secrets
- Review and update documentation
- Performance optimization review

### Backup and Recovery

```bash
# Backup MongoDB data
mongodump --uri="$MONGODB_ATLAS_URI" --out=backup/

# Backup Redis data
redis-cli --rdb redis-backup.rdb

# Backup R2 data
aws s3 sync s3://h200-backup backup/r2/ --endpoint-url=$R2_ENDPOINT_URL
```

---

## Troubleshooting

### Common Issues

#### 1. GPU Out of Memory

**Symptoms**: "CUDA out of memory" errors

**Solutions**:
```bash
# Reduce batch size
export MAX_BATCH_SIZE=16

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Restart services
docker-compose restart h200-timed
```

#### 2. Slow Cold Starts

**Symptoms**: First request takes >2s

**Solutions**:
```bash
# Ensure FlashBoot is enabled
export FLASHBOOT_ENABLED=true

# Pre-warm models
python docker/preload_models.py

# Check Redis cache
redis-cli ping
```

#### 3. Connection Errors

**MongoDB Connection Failed**:
```bash
# Check connection string
echo $MONGODB_ATLAS_URI

# Test connection
mongosh "$MONGODB_ATLAS_URI" --eval "db.adminCommand('ping')"

# Check IP whitelist in Atlas
```

**Redis Connection Failed**:
```bash
# Check Redis is running
docker-compose ps redis

# Test connection
redis-cli -h localhost -p 6379 ping

# Check password
redis-cli -h localhost -p 6379 -a $REDIS_PASSWORD ping
```

#### 4. API Authentication Issues

**"Unauthorized" errors**:
```bash
# Verify JWT secret is set
echo $JWT_SECRET

# Get new token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your_password"}'
```

#### 5. RunPod Deployment Issues

**Deployment fails**:
```bash
# Check API key
echo $RUNPOD_API_KEY

# Verify RunPod CLI
runpod whoami

# Check available GPUs
python scripts/deploy/deploy_to_runpod.py list-gpus
```

### Debug Mode

Enable debug logging:
```bash
# Set debug level
export LOG_LEVEL=DEBUG

# Enable FastAPI debug mode
export FASTAPI_DEBUG=true

# Restart services
docker-compose restart
```

### Getting Help

1. Check logs: `docker-compose logs -f [service_name]`
2. Review documentation: `/docs/troubleshooting.md`
3. Check monitoring dashboards
4. Review error codes: `/docs/reference/error-codes.md`

### Support Channels

- GitHub Issues: Report bugs and feature requests
- Documentation: Comprehensive guides in `/docs/`
- API Reference: Interactive docs at `/api/docs`

---

## Appendix

### Performance Benchmarks

| Metric | Target | Typical |
|--------|--------|---------|
| Cold Start | 500ms-2s | 1.2s |
| Warm Start | <100ms | 45ms |
| Image Processing | <500ms | 250ms |
| API Latency (p95) | <200ms | 150ms |
| Cache Hit Rate | >85% | 92% |
| GPU Utilization | >70% | 75% |

### Cost Estimates

| Component | Usage | Monthly Cost |
|-----------|-------|--------------|
| RunPod H100 | 100 hrs | ~$400 |
| RunPod Serverless | 10K requests | ~$50 |
| MongoDB Atlas | M10 cluster | $60 |
| Cloudflare R2 | 100GB storage | $1.50 |
| Redis Cloud | 2GB RAM | $30 |
| **Total** | Moderate usage | ~$541.50 |

### Security Best Practices

1. **Rotate secrets regularly** (every 90 days)
2. **Use environment variables** for all credentials
3. **Enable TLS/HTTPS** in production
4. **Implement rate limiting** on all endpoints
5. **Monitor for suspicious activity**
6. **Keep dependencies updated**
7. **Use least-privilege access**

### Useful Commands Reference

```bash
# Docker commands
docker-compose up -d              # Start all services
docker-compose logs -f            # View logs
docker-compose ps                 # Check status
docker-compose down              # Stop all services

# Testing commands
make test                        # Run all tests
make test-unit                   # Unit tests only
make test-integration            # Integration tests
make coverage                    # Generate coverage report

# Deployment commands
./scripts/deploy/build_and_push.sh    # Build images
python scripts/deploy/deploy_to_runpod.py both  # Deploy all
python scripts/deploy/deploy_to_runpod.py status # Check status

# Monitoring commands
curl http://localhost:9091/metrics    # Prometheus metrics
docker-compose logs grafana           # Grafana logs

# Database commands
mongosh $MONGODB_ATLAS_URI           # MongoDB shell
redis-cli -h localhost -p 6379       # Redis CLI
```

---

This manual provides comprehensive guidance for using the H200 Intelligent Mug Positioning System. For additional details, refer to the technical documentation in the `/docs/` directory.