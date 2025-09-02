# Quick Start Guide

Get up and running with the H200 Intelligent Mug Positioning System in minutes.

## Overview

The H200 Intelligent Mug Positioning System analyzes images to detect mugs and provides intelligent positioning feedback based on configurable rules. This guide will help you get started quickly.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.11+** installed
- **Docker Desktop** running
- **Git** for cloning the repository
- Access to the following cloud services:
  - **MongoDB Atlas** account
  - **RunPod** account (for GPU deployment)
  - **Cloudflare** account (for R2 storage)
  - **Google Cloud** account (for Secret Manager)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tekfly/h200-mug-positioning.git
cd h200-mug-positioning
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Configure Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

Required environment variables:

```env
# Database Configuration
MONGODB_ATLAS_URI=mongodb+srv://user:pass@cluster.mongodb.net/h200?retryWrites=true&w=majority
REDIS_HOST=your-redis-host
REDIS_PASSWORD=your-redis-password

# RunPod Configuration
RUNPOD_API_KEY=your-runpod-api-key

# Cloudflare R2 Configuration
R2_ACCESS_KEY_ID=your-r2-access-key
R2_SECRET_ACCESS_KEY=your-r2-secret-key
R2_BUCKET_NAME=h200-storage

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Security
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
```

### 4. Start Local Development

```bash
# Start services with Docker Compose
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 5. Test the Installation

```bash
# Run health check
curl http://localhost:8000/api/health

# Expected response:
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "mongodb": true,
    "redis": true,
    "models": true
  }
}
```

## First Steps

### 1. Access the Dashboard

Open your browser and navigate to:
- **Local**: http://localhost:3000
- **Production**: https://your-domain.com

### 2. Analyze Your First Image

**Using the Web Interface:**

1. Go to the Analysis page
2. Upload an image containing mugs
3. Click "Analyze Image"
4. Review the positioning feedback

**Using the API:**

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/with-feedback" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@your-mug-image.jpg" \
  -F "include_feedback=true" \
  -F "confidence_threshold=0.8"
```

### 3. Create a Positioning Rule

**Natural Language Rule Creation:**

```bash
curl -X POST "http://localhost:8000/api/v1/rules/natural-language" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The mug should be centered on the coaster",
    "auto_enable": true
  }'
```

## Development Workflow

### 1. Local Development

```bash
# Start API server with hot reload
uvicorn src.control.api.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend development server
cd dashboard
npm run dev
```

### 2. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 3. Code Quality Checks

```bash
# Lint code
python -m pylint src/ --rcfile=.pylintrc

# Type checking
python -m mypy src/ --strict

# Format code
python -m black src/ tests/
```

## Production Deployment

### 1. Build Docker Images

```bash
# Build all images
docker-compose -f docker-compose.production.yml build

# Push to registry
docker-compose -f docker-compose.production.yml push
```

### 2. Deploy to RunPod

```bash
# Deploy serverless endpoint
python scripts/deploy_to_runpod.py --mode=serverless

# Deploy timed instance
python scripts/deploy_to_runpod.py --mode=timed
```

### 3. Monitor Deployment

```bash
# Check deployment status
python scripts/check_deployment_status.py

# View logs
docker-compose -f docker-compose.production.yml logs -f
```

## Common Configuration

### GPU Memory Optimization

For H100/H200 GPUs with limited memory:

```env
# In .env
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
MODEL_CACHE_SIZE=2048
```

### Performance Tuning

For high-throughput scenarios:

```env
# Enable model caching
ENABLE_MODEL_CACHE=true
CACHE_TTL_SECONDS=300

# Batch processing
MAX_BATCH_SIZE=8
BATCH_TIMEOUT_MS=100

# Connection pooling
DB_POOL_SIZE=20
REDIS_POOL_SIZE=10
```

### Security Hardening

For production deployments:

```env
# Enable security features
ENABLE_RATE_LIMITING=true
RATE_LIMIT_CALLS=100
RATE_LIMIT_PERIOD=60

# CORS configuration
CORS_ORIGINS=["https://your-domain.com"]
ALLOWED_HOSTS=["your-domain.com"]

# JWT security
JWT_EXPIRY_HOURS=24
REQUIRE_HTTPS=true
```

## Monitoring Setup

### 1. Enable Monitoring

```bash
# Start monitoring stack
docker-compose -f configs/monitoring/docker-compose.monitoring.yml up -d

# Access monitoring dashboards
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
```

### 2. Configure Alerts

```bash
# Edit alert rules
nano configs/monitoring/alert_rules.yml

# Restart Prometheus to apply changes
docker-compose -f configs/monitoring/docker-compose.monitoring.yml restart prometheus
```

## Next Steps

Now that you have the system running:

1. **[User Manual](./user-manual.md)** - Complete feature guide
2. **[Dashboard Guide](./dashboard-guide.md)** - Navigate the web interface  
3. **[Rules Management](./rules-management.md)** - Create and manage rules
4. **[API Documentation](../api/README.md)** - Integrate with your applications
5. **[Troubleshooting](../operations/troubleshooting.md)** - Solve common issues

## Getting Help

If you encounter issues:

1. **Check the logs**: `docker-compose logs -f`
2. **Verify health**: `curl http://localhost:8000/api/health`
3. **Review documentation**: Check the relevant guide sections
4. **Ask for help**: Open an issue on GitHub or contact support

## Performance Expectations

With proper setup, you should see:

- **Cold Start**: 500ms-2s (FlashBoot enabled)
- **Warm Start**: <100ms
- **Image Processing**: <500ms for 1080p images
- **API Latency**: p95 <200ms
- **Cache Hit Rate**: >85%

## Cleanup

To remove all containers and data:

```bash
# Stop and remove containers
docker-compose down -v

# Remove images
docker rmi $(docker images -q "tekfly/h200*")

# Deactivate virtual environment
deactivate
```

Congratulations! You now have the H200 Intelligent Mug Positioning System running. Continue with the [User Manual](./user-manual.md) to learn about all available features.