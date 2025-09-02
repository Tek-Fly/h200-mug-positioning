# H200 Deployment Scripts

This directory contains scripts for building and deploying the H200 Intelligent Mug Positioning System.

## Scripts Overview

### build_and_push.sh
Builds and pushes Docker images to a container registry with multi-platform support.

**Features:**
- Multi-stage builds with caching
- Support for AMD64 and ARM64 architectures
- Docker Build Cloud integration
- Automatic tagging with timestamps
- Image verification

**Usage:**
```bash
# Basic usage (requires DOCKER_USERNAME env var)
./build_and_push.sh

# With custom platforms
./build_and_push.sh --platforms linux/amd64

# With custom registry
./build_and_push.sh --registry ghcr.io

# Prune build cache after build
./build_and_push.sh --prune-cache
```

### deploy_to_runpod.py
Deploys the system to RunPod for GPU-accelerated processing.

**Features:**
- Serverless deployment with FlashBoot (500ms-2s cold start)
- Timed GPU deployment with auto-shutdown
- Automatic environment variable configuration
- Health check verification
- Deployment status monitoring

**Usage:**
```bash
# Deploy serverless endpoint
./deploy_to_runpod.py serverless

# Deploy timed GPU pod
./deploy_to_runpod.py timed

# Deploy both modes
./deploy_to_runpod.py both --gpu-type H100 --gpu-count 1

# With custom configuration
./deploy_to_runpod.py serverless \
    --min-workers 1 \
    --max-workers 5 \
    --gpu-type A100 \
    --tag v1.0.0
```

### health_check.sh
Performs health checks on deployed services.

**Usage:**
```bash
# Check local deployment
./health_check.sh

# Check remote endpoint
./health_check.sh --endpoint https://api.example.com/health

# With custom retry settings
./health_check.sh --retries 5 --interval 10 --timeout 30
```

## Environment Variables

Required environment variables (set in .env file):

```bash
# Docker Registry
DOCKER_USERNAME=your_username
DOCKER_PASSWORD=your_password  # Optional, for automated builds
DOCKER_BUILD_CLOUD_TOKEN=your_token  # Optional, for Build Cloud

# RunPod
RUNPOD_API_KEY=your_api_key

# MongoDB Atlas
MONGODB_ATLAS_URI=mongodb+srv://...

# Cloudflare R2
R2_ENDPOINT_URL=https://...
R2_ACCESS_KEY_ID=your_key
R2_SECRET_ACCESS_KEY=your_secret

# Redis
REDIS_HOST=your_redis_host
REDIS_PASSWORD=your_password

# Other required vars
GCP_PROJECT_ID=your_project
JWT_SECRET=your_secret
```

## Deployment Flow

1. **Build Images**
   ```bash
   ./build_and_push.sh
   ```

2. **Deploy to RunPod**
   ```bash
   # For production
   ./deploy_to_runpod.py both --gpu-type H100
   
   # For testing
   ./deploy_to_runpod.py serverless --gpu-type RTX4090
   ```

3. **Verify Deployment**
   ```bash
   # Check health
   ./health_check.sh --endpoint https://your-endpoint.runpod.io/health
   ```

## Docker Images

The build process creates three images:

1. **base**: Python 3.11 with CUDA support and all dependencies
2. **serverless**: Optimized for RunPod serverless with FlashBoot
3. **timed**: Full API server with control plane for timed deployments

## Troubleshooting

### Build Issues
- Ensure Docker buildx is installed: `docker buildx version`
- Check available builders: `docker buildx ls`
- Clear build cache: `docker buildx prune -f`

### Deployment Issues
- Verify RunPod API key: `echo $RUNPOD_API_KEY`
- Check RunPod quota and available GPUs
- Review deployment logs in RunPod dashboard

### Health Check Failures
- Ensure service ports are accessible
- Check firewall/security group settings
- Review application logs for errors

## Cost Optimization

- Use serverless for sporadic workloads
- Configure auto-shutdown for timed deployments
- Monitor GPU utilization and adjust instance types
- Use spot instances when available