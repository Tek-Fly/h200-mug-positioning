# H200 Intelligent Mug Positioning System - Claude Development Guide

## Quick Reference

### Critical Performance Update
**FlashBoot cold start: 500ms-2s** (not sub-200ms) - All performance targets updated accordingly

## Implementation Overview

This guide provides comprehensive instructions for Claude or any AI assistant working on the H200 Intelligent Mug Positioning System. Follow these guidelines to maintain consistency and quality across all development work.

### Implementation Phases

#### Phase 1: Infrastructure Setup
- Configure Cloudflare R2 replacing all S3 references
- Deploy Redis with clustering and LRU eviction
- Setup MongoDB Atlas with vector search indexes
- Implement Google Secret Manager integration

#### Phase 2: Core System Development  
- Build H200ImageAnalyzer with dual caching (Redis L1 + H200 L2)
- Implement YOLOv8 + CLIP models with Redis state management
- Create embedded MCP server with A2A protocol compliance
- Develop LangChain rule management system

#### Phase 3: Control Plane & Monitoring
- Build FastAPI control plane with WebSocket support
- Create Vue.js dashboard with real-time metrics
- Implement cost monitoring with progressive alerts
- Setup comprehensive logging and audit trails

#### Phase 4: Deployment & Integration
- Configure GitHub Actions with Docker Build Cloud
- Setup RunPod deployment automation
- Implement N8N workflow integration examples
- Complete Templated.io API integration

### Success Criteria
- FlashBoot cold starts: 500ms-2s consistently
- Redis cache hit rate: >85%
- H200 GPU utilization: >70%
- R2 backup cost savings: $15-25/month vs S3
- Complete dual deployment: serverless + timed modes

### Key Commands
```bash
# Lint and typecheck (run after code changes)
python -m pylint src/ --rcfile=.pylintrc
python -m mypy src/ --strict

# Run tests
pytest tests/ -v --cov=src --cov-report=term-missing

# Build Docker images
docker-compose -f docker-compose.production.yml build

# Start services locally
docker-compose up -d

# Deploy to RunPod
python scripts/deploy_to_runpod.py --mode=serverless
python scripts/deploy_to_runpod.py --mode=timed

# Check Redis cache stats
redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD INFO stats

# Monitor GPU usage
nvidia-smi -l 1

# View logs
docker-compose logs -f h200-serverless
docker-compose logs -f h200-timed
```

### Project Structure
```
src/
├── core/           # Image analysis, models, MCP server
├── control/        # FastAPI, WebSocket, control plane
├── database/       # MongoDB, Redis, R2 connections
├── deployment/     # RunPod deployment logic
├── integrations/   # External service clients
└── utils/          # Secrets, logging, common utilities
```

### Environment Setup
1. Copy `.env.example` to `.env`
2. Fill in all required credentials
3. Ensure Docker and Docker Compose are installed
4. Install Python 3.11+ and create virtual environment
5. Install dependencies: `pip install -r requirements.txt`

### Development Workflow
1. Always update COMPREHENSIVE_HANDOVER_DOCUMENT.md when completing work
2. Run linting and type checking before committing
3. Ensure all tests pass before marking task complete
4. Use async/await for all I/O operations
5. Log at appropriate levels (DEBUG in dev, INFO in prod)

### Common Issues & Solutions

#### Redis Connection Failed
- Check REDIS_HOST and REDIS_PASSWORD in .env
- Ensure Redis container is running: `docker-compose ps`
- Verify Redis memory limit: should be 2GB

#### MongoDB Connection Timeout
- Verify MONGODB_ATLAS_URI includes all parameters
- Check network connectivity to MongoDB Atlas
- Ensure IP is whitelisted in Atlas

#### GPU Out of Memory
- Reduce batch size in model configuration
- Clear GPU cache between operations
- Monitor with `nvidia-smi`

#### FlashBoot Performance Issues
- Ensure models are pre-loaded in Redis
- Check Redis cache hit rate (target >85%)
- Verify GPU warm-up is working

### API Endpoints Reference
```
POST /api/v1/analyze/with-feedback
  - Main image analysis endpoint
  - Accepts: multipart/form-data with image
  - Returns: positioning data with confidence

POST /api/v1/rules/natural-language
  - Update positioning rules via NLP
  - Accepts: JSON with rule text
  - Returns: rule ID and validation status

GET /api/v1/dashboard
  - Complete system metrics
  - Returns: JSON with all dashboard data

POST /api/v1/servers/{type}/control
  - Control server lifecycle
  - Types: serverless, timed
  - Actions: start, stop, restart

WebSocket /ws/control-plane
  - Real-time system updates
  - Subscribe to: metrics, logs, alerts
```

### Performance Metrics
- **Cold Start**: 500ms-2s (FlashBoot enabled)
- **Warm Start**: <100ms
- **Image Processing**: <500ms for 1080p
- **Cache Hit Rate**: >85%
- **GPU Utilization**: >70% during processing
- **API Latency**: p95 <200ms

### Security Checklist
- [ ] All secrets in Google Secret Manager
- [ ] JWT tokens for API authentication
- [ ] TLS for all external connections
- [ ] Input validation on all endpoints
- [ ] Rate limiting configured
- [ ] Docker Scout scanning enabled
- [ ] No hardcoded credentials

### Monitoring & Debugging
```bash
# View structured logs
docker-compose logs h200-timed | jq .

# Check Prometheus metrics
curl http://localhost:9090/metrics

# Monitor Redis operations
redis-cli MONITOR

# GPU debugging
CUDA_LAUNCH_BLOCKING=1 python src/core/analyzer.py

# API debugging
FASTAPI_DEBUG=true uvicorn src.control.api.main:app --reload
```

### Cost Optimization Tips
1. Use Cloudflare R2 for all backup storage (zero egress fees)
2. Enable auto-shutdown after 10 minutes idle
3. Batch process images when possible
4. Monitor cache hit rates to reduce GPU usage
5. Use serverless mode for sporadic workloads

### Integration Testing
```bash
# Test MongoDB connection
python -c "from src.database.mongodb import test_connection; test_connection()"

# Test Redis caching
python -c "from src.database.redis import test_cache; test_cache()"

# Test R2 storage
python -c "from src.database.r2 import test_upload; test_upload()"

# Test model loading
python -c "from src.core.models import test_models; test_models()"
```

### Deployment Checklist
- [ ] All environment variables set
- [ ] Docker images built and pushed
- [ ] RunPod resources allocated
- [ ] Monitoring configured
- [ ] Logs aggregation working
- [ ] Health checks passing
- [ ] Auto-shutdown tested
- [ ] Backup storage verified

### Contact & Resources
- API Documentation: /docs/api/endpoints.md
- Monitoring Dashboard: http://localhost:3000
- RunPod Console: https://runpod.io/console
- MongoDB Atlas: https://cloud.mongodb.com