# Deployment Guide

Complete guide to deploying the H200 Intelligent Mug Positioning System across different environments.

## Overview

The H200 System supports multiple deployment scenarios:
- **Local Development**: Docker Compose setup
- **Cloud Development**: Single-instance cloud deployment
- **Production**: Scalable RunPod GPU deployment
- **Enterprise**: Multi-region, high-availability setup

## Prerequisites

### Required Accounts and Services

**Cloud Infrastructure:**
- **RunPod Account**: GPU compute instances
- **MongoDB Atlas**: Managed database service
- **Cloudflare Account**: R2 object storage
- **Google Cloud**: Secret Manager for credentials
- **Docker Hub**: Container registry

**Development Tools:**
- Docker Desktop (v4.0+)
- Python 3.11+
- Git
- kubectl (for Kubernetes deployments)

### Required Credentials

Create accounts and obtain the following credentials:

```env
# RunPod
RUNPOD_API_KEY=your_runpod_api_key

# MongoDB Atlas  
MONGODB_ATLAS_URI=mongodb+srv://user:pass@cluster.mongodb.net/h200

# Redis (RunPod or external)
REDIS_HOST=your_redis_host
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Cloudflare R2
R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
R2_BUCKET_NAME=h200-storage

# Google Cloud
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json

# Security
SECRET_KEY=your_256_bit_secret_key
JWT_ALGORITHM=HS256
```

## Local Development Deployment

### Quick Setup

```bash
# 1. Clone repository
git clone https://github.com/tekfly/h200-mug-positioning.git
cd h200-mug-positioning

# 2. Environment setup
cp .env.example .env
# Edit .env with your credentials

# 3. Start services
docker-compose up -d

# 4. Verify deployment
curl http://localhost:8000/api/health
```

### Development Docker Compose

**File**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  # Main API service
  h200-api:
    build: 
      context: .
      dockerfile: docker/Dockerfile.development
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
    volumes:
      - ./src:/app/src
      - ./models:/app/models
    depends_on:
      - redis-local
      - mongo-local
    restart: unless-stopped

  # Frontend dashboard
  h200-dashboard:
    build:
      context: ./dashboard
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    environment:
      - API_BASE_URL=http://localhost:8000
    restart: unless-stopped

  # Local Redis for development
  redis-local:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --requirepass devpassword
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Local MongoDB for development  
  mongo-local:
    image: mongo:7
    ports:
      - "27017:27017"
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=devpassword
      - MONGO_INITDB_DATABASE=h200
    volumes:
      - mongo_data:/data/db
    restart: unless-stopped

volumes:
  redis_data:
  mongo_data:
```

### Development Features

**Hot Reload:**
- API server restarts on code changes
- Frontend rebuilds automatically
- Database schemas update dynamically

**Debug Tools:**
- Interactive API documentation at `/api/docs`
- Debug logs with detailed tracing
- Performance profiling endpoints

**Development Database:**
- Pre-populated with test data
- Easy reset and migration
- Local backup and restore

## Cloud Development Deployment

### Single-Instance Cloud Setup

For testing cloud integrations without production complexity:

```bash
# 1. Build and push images
docker build -f docker/Dockerfile.cloud-dev -t tekfly/h200:cloud-dev .
docker push tekfly/h200:cloud-dev

# 2. Deploy to RunPod
python scripts/deploy/deploy_to_runpod.py \
  --mode=development \
  --image=tekfly/h200:cloud-dev \
  --gpu-type=H100 \
  --enable-ssh=true
```

**Cloud Development Configuration:**

```python
CLOUD_DEV_CONFIG = {
    "pod_type": "INTERRUPTIBLE",  # Cost-effective
    "image_name": "tekfly/h200:cloud-dev",
    "gpu_type": "H100",
    "gpu_count": 1,
    "vcpus": 8,
    "memory": 32,
    "container_disk": 50,
    "volume_size": 100,
    "ports": [
        {"internal": 8000, "external": 8000, "type": "http"},
        {"internal": 3000, "external": 3000, "type": "http"},
        {"internal": 22, "external": 22, "type": "tcp"}  # SSH
    ],
    "environment_variables": {
        "ENVIRONMENT": "cloud-development",
        "DEBUG": "true",
        "ENABLE_SSH": "true"
    }
}
```

## Production Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                            │
│                  (Cloudflare/RunPod)                        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 API Gateway Instances                       │
│            (Multiple RunPod Serverless)                     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                GPU Processing Layer                         │
│        (H200 Serverless + Timed Instances)                 │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Managed Services                          │
│   MongoDB Atlas │ Redis Cloud │ Cloudflare R2             │
└─────────────────────────────────────────────────────────────┘
```

### Production Docker Images

**Serverless Optimized:**

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS serverless-base

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python optimization
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install minimal dependencies for fast cold starts
COPY requirements-serverless.txt .
RUN pip install --no-cache-dir -r requirements-serverless.txt

# Copy application code
COPY src/ ./src/
COPY docker/preload_models.py .

# Pre-optimize models for inference
RUN python preload_models.py --target=serverless --optimize=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000
CMD ["python", "-m", "gunicorn", "src.control.api.main:app", \
     "-w", "1", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "60", \
     "--preload"]
```

**Timed Instance Optimized:**

```dockerfile  
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS timed-base

# Full development and debugging tools
RUN apt-get update && apt-get install -y \
    python3.11-dev \
    python3-pip \
    build-essential \
    nvidia-ml-py3 \
    htop \
    nvtop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install full dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development tools for timed instances
RUN pip install --no-cache-dir \
    jupyter \
    tensorboard \
    wandb \
    pytest

# Copy application and tools
COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY dashboard/dist/ ./dashboard/

# Multi-service startup script
COPY docker/start-timed.sh .
RUN chmod +x start-timed.sh

EXPOSE 8000 8888 6006
CMD ["./start-timed.sh"]
```

### Production Configuration

**Serverless Deployment:**

```python
SERVERLESS_PROD_CONFIG = {
    "name": "h200-serverless-prod",
    "image_name": "tekfly/h200:serverless-v1.0.0",
    "gpu_type": "H200",
    "gpu_count": 1,
    "container_disk_size": 20,
    "volume_size": 50,
    "network_volume_id": "shared-models-volume",
    
    # Scaling configuration
    "scaling": {
        "min_instances": 0,
        "max_instances": 10,
        "idle_timeout": 300,  # 5 minutes
        "scale_up_threshold": 0.8,
        "scale_down_threshold": 0.2,
        "concurrent_requests": 1,
    },
    
    # Environment variables from secrets
    "environment_variables": {
        "ENVIRONMENT": "production",
        "LOG_LEVEL": "INFO",
        "ENABLE_METRICS": "true",
        "MONGODB_URI": "${SECRET:MONGODB_PROD_URI}",
        "REDIS_URL": "${SECRET:REDIS_PROD_URL}",
        "R2_CREDENTIALS": "${SECRET:R2_PROD_CREDENTIALS}",
        "JWT_SECRET": "${SECRET:JWT_PROD_SECRET}"
    },
    
    # Resource limits
    "limits": {
        "cpu": "8000m",    # 8 CPU cores
        "memory": "32Gi",  # 32GB RAM
        "gpu_memory": "80Gi"  # Full H200 memory
    }
}
```

**Timed Instance Configuration:**

```python
TIMED_PROD_CONFIG = {
    "name": "h200-timed-prod", 
    "image_name": "tekfly/h200:timed-v1.0.0",
    "gpu_type": "H200",
    "gpu_count": 1,
    "vcpus": 16,
    "memory": 64,  # GB
    "container_disk_size": 100,
    "volume_size": 200,
    
    # Always-on configuration  
    "bid_per_gpu": 2.50,  # Maximum bid
    "country_code": "US",
    "min_upload": 100,    # Mbps
    "min_download": 100,  # Mbps
    
    # High availability
    "restart_policy": "always",
    "health_check": {
        "path": "/api/health",
        "interval": 30,
        "timeout": 10,
        "retries": 3
    }
}
```

### Deployment Scripts

**Automated Production Deployment:**

```python
#!/usr/bin/env python3
"""
Production deployment automation script
"""
import asyncio
import logging
from typing import Dict, Any

from scripts.deploy.runpod_client import RunPodClient
from scripts.deploy.health_checker import HealthChecker
from scripts.deploy.metrics_validator import MetricsValidator

logger = logging.getLogger(__name__)

class ProductionDeployer:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.runpod = RunPodClient(api_key=self.config["runpod_api_key"])
        self.health_checker = HealthChecker()
        self.metrics_validator = MetricsValidator()
    
    async def deploy(self) -> Dict[str, Any]:
        """Execute full production deployment"""
        deployment_id = f"prod-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        try:
            # 1. Pre-deployment validation
            await self.validate_prerequisites()
            
            # 2. Build and push images
            await self.build_and_push_images()
            
            # 3. Deploy serverless endpoints
            serverless_info = await self.deploy_serverless()
            
            # 4. Deploy timed instances
            timed_info = await self.deploy_timed_instances()
            
            # 5. Configure load balancing
            lb_info = await self.configure_load_balancer(
                serverless_info, timed_info
            )
            
            # 6. Health checks and validation
            await self.validate_deployment(deployment_id)
            
            # 7. Enable monitoring and alerting
            await self.setup_monitoring(deployment_id)
            
            return {
                "deployment_id": deployment_id,
                "status": "success",
                "endpoints": {
                    "serverless": serverless_info,
                    "timed": timed_info,
                    "load_balancer": lb_info
                }
            }
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            await self.rollback_deployment(deployment_id)
            raise
    
    async def validate_prerequisites(self):
        """Validate all required services and credentials"""
        checks = [
            self.check_mongodb_connection(),
            self.check_redis_connection(),
            self.check_r2_access(),
            self.check_secret_manager(),
            self.check_docker_registry()
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        failed_checks = [
            f"Check {i}: {result}" 
            for i, result in enumerate(results) 
            if isinstance(result, Exception)
        ]
        
        if failed_checks:
            raise RuntimeError(f"Prerequisites failed: {failed_checks}")
    
    async def deploy_serverless(self) -> Dict[str, Any]:
        """Deploy serverless endpoints with health validation"""
        logger.info("Deploying serverless endpoints...")
        
        # Create serverless endpoint
        endpoint = await self.runpod.create_serverless_endpoint(
            SERVERLESS_PROD_CONFIG
        )
        
        # Wait for deployment
        await self.wait_for_endpoint_ready(endpoint["id"])
        
        # Validate with test requests
        await self.validate_endpoint_performance(endpoint["url"])
        
        return {
            "id": endpoint["id"],
            "url": endpoint["url"],
            "status": "deployed"
        }
    
    async def validate_deployment(self, deployment_id: str):
        """Comprehensive deployment validation"""
        logger.info(f"Validating deployment {deployment_id}")
        
        # Health checks
        health_results = await self.health_checker.check_all_endpoints()
        if not all(health_results.values()):
            raise RuntimeError(f"Health checks failed: {health_results}")
        
        # Performance validation
        perf_results = await self.metrics_validator.validate_performance()
        if not perf_results["meets_sla"]:
            raise RuntimeError(f"Performance validation failed: {perf_results}")
        
        # Security validation
        security_results = await self.validate_security_configuration()
        if not security_results["secure"]:
            raise RuntimeError(f"Security validation failed: {security_results}")
        
        logger.info("Deployment validation completed successfully")

# Usage
if __name__ == "__main__":
    deployer = ProductionDeployer("config/production.yaml")
    result = asyncio.run(deployer.deploy())
    print(f"Deployment completed: {result}")
```

### Environment-Specific Configurations

**Staging Environment:**

```yaml
# config/staging.yaml
environment: staging
debug: false
log_level: INFO

database:
  mongodb_uri: ${SECRET:MONGODB_STAGING_URI}
  redis_url: ${SECRET:REDIS_STAGING_URL}

storage:
  r2_bucket: h200-staging-storage
  model_cache_size: 4096

scaling:
  serverless:
    min_instances: 0
    max_instances: 3
    idle_timeout: 300
  
  timed:
    instance_count: 1
    auto_shutdown: true

monitoring:
  enable_tracing: true
  metrics_retention: 7d
  alert_channels: ["slack-staging"]
```

**Production Environment:**

```yaml
# config/production.yaml
environment: production
debug: false
log_level: WARN

database:
  mongodb_uri: ${SECRET:MONGODB_PROD_URI}
  redis_url: ${SECRET:REDIS_PROD_URL}
  connection_pool_size: 20

storage:
  r2_bucket: h200-prod-storage
  model_cache_size: 8192
  backup_enabled: true

scaling:
  serverless:
    min_instances: 1  # Always warm
    max_instances: 20
    idle_timeout: 600  # 10 minutes
  
  timed:
    instance_count: 3
    auto_shutdown: false  # Always on
    high_availability: true

security:
  tls_required: true
  rate_limiting: true
  ip_whitelist_enabled: true

monitoring:
  enable_tracing: true
  metrics_retention: 30d
  alert_channels: ["pagerduty", "slack-alerts", "email-oncall"]
  
performance:
  target_latency_p95: 200ms
  target_availability: 99.9%
  target_error_rate: 0.1%
```

## Multi-Region Deployment

### Global Architecture

```
Region: US-East
├── RunPod Serverless (Primary)
├── RunPod Timed (Primary)
└── Redis Cache (Primary)

Region: Europe
├── RunPod Serverless (Secondary)  
├── RunPod Timed (Secondary)
└── Redis Cache (Secondary)

Global Services:
├── MongoDB Atlas (Multi-region)
├── Cloudflare R2 (Global CDN)
└── Cloudflare Load Balancer
```

### Region Configuration

```python
MULTI_REGION_CONFIG = {
    "regions": {
        "us-east": {
            "primary": True,
            "runpod_region": "US-East",
            "mongodb_region": "US-EAST-1",
            "redis_cluster": "us-east-redis",
            "failover_priority": 1
        },
        "europe": {
            "primary": False,
            "runpod_region": "EU-RO-1", 
            "mongodb_region": "EUROPE-WEST1",
            "redis_cluster": "eu-west-redis",
            "failover_priority": 2
        }
    },
    
    "load_balancing": {
        "strategy": "geographic",
        "health_check_interval": 30,
        "failover_timeout": 10
    },
    
    "data_replication": {
        "mongodb_read_preference": "primaryPreferred",
        "redis_replication": "async",
        "r2_replication": "global"
    }
}
```

## Monitoring and Health Checks

### Health Check Endpoints

```python
@app.get("/api/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "database": await check_database(),
            "cache": await check_cache(),
            "gpu": await check_gpu(),
            "storage": await check_storage()
        },
        "metrics": {
            "response_time_ms": 45,
            "memory_usage_percent": 67,
            "gpu_utilization_percent": 23
        }
    }

@app.get("/api/health/deep")
async def deep_health_check():
    """Detailed health check for monitoring systems"""
    return {
        "services": await run_all_service_checks(),
        "performance": await collect_performance_metrics(),
        "security": await validate_security_status(),
        "dependencies": await check_external_dependencies()
    }
```

### Monitoring Setup

**Prometheus Configuration:**

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'h200-api'
    static_configs:
      - targets: ['h200-api:8000']
    metrics_path: '/api/metrics'
    
  - job_name: 'h200-gpu'
    static_configs:
      - targets: ['h200-gpu:8000']
    metrics_path: '/gpu/metrics'

alertmanager:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

## Backup and Recovery

### Database Backup Strategy

```python
class BackupManager:
    def __init__(self):
        self.mongodb_client = MongoDBClient()
        self.r2_client = R2Client()
        
    async def create_full_backup(self):
        """Create complete system backup"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_id = f"backup_{timestamp}"
        
        # Database backup
        db_backup = await self.backup_mongodb(backup_id)
        
        # Configuration backup
        config_backup = await self.backup_configurations(backup_id)
        
        # Model backup (if changed)
        model_backup = await self.backup_models(backup_id)
        
        # Store backup manifest
        manifest = {
            "backup_id": backup_id,
            "timestamp": timestamp,
            "components": {
                "database": db_backup,
                "config": config_backup,
                "models": model_backup
            }
        }
        
        await self.r2_client.upload_json(
            f"backups/{backup_id}/manifest.json", 
            manifest
        )
        
        return backup_id
    
    async def restore_from_backup(self, backup_id: str):
        """Restore system from backup"""
        # Download manifest
        manifest = await self.r2_client.download_json(
            f"backups/{backup_id}/manifest.json"
        )
        
        # Restore in order: models -> database -> config
        await self.restore_models(manifest["components"]["models"])
        await self.restore_database(manifest["components"]["database"])
        await self.restore_config(manifest["components"]["config"])
        
        return {"status": "restored", "backup_id": backup_id}
```

### Disaster Recovery Procedures

**Recovery Time Objectives (RTO):**
- Database restoration: < 15 minutes
- Service restart: < 5 minutes  
- Full system recovery: < 30 minutes

**Recovery Point Objectives (RPO):**
- Database data loss: < 5 minutes
- Configuration loss: < 1 hour
- Model updates: < 24 hours

## Troubleshooting Deployments

### Common Issues

**Image Build Failures:**
```bash
# Check Docker build context
docker build --no-cache -t debug-build .

# Inspect failed layer
docker run --rm -it <failed_layer_id> /bin/bash

# Check available space
df -h /var/lib/docker
```

**RunPod Deployment Failures:**
```python
# Check pod logs
logs = await runpod_client.get_pod_logs(pod_id)
print(f"Error logs: {logs}")

# Validate environment variables
env_vars = await runpod_client.get_pod_environment(pod_id)
for key, value in env_vars.items():
    if not value:
        print(f"Missing environment variable: {key}")
```

**Performance Issues:**
```bash
# GPU memory check
nvidia-smi

# Container resource usage
docker stats h200-container

# Network connectivity
curl -w "@curl-format.txt" -s -o /dev/null http://api-endpoint/health
```

### Rollback Procedures

**Automated Rollback:**
```python
async def rollback_deployment(deployment_id: str, target_version: str):
    """Rollback to previous working version"""
    
    # 1. Stop new version
    await stop_deployment(deployment_id)
    
    # 2. Restore previous image
    await deploy_version(target_version)
    
    # 3. Restore database state if needed
    if requires_db_rollback(deployment_id):
        await restore_database_backup(get_pre_deployment_backup())
    
    # 4. Validate rollback
    await validate_deployment_health()
    
    # 5. Update load balancer
    await update_load_balancer_targets()
    
    return {"status": "rolled_back", "version": target_version}
```

This deployment guide provides comprehensive coverage of all deployment scenarios from local development to multi-region production setups. The automated scripts and monitoring ensure reliable, scalable deployments with proper observability and recovery procedures.