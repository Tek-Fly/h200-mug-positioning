# H200 Intelligent Mug Positioning System - RunPod Native Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the H200 Mug Positioning System using RunPod's native template-based architecture. No Docker required - deploy directly with Python on H200 GPUs.

## Prerequisites Checklist

Before deploying, ensure you have:

- [ ] RunPod account with H200 GPU access (verify at https://runpod.io/console/gpu-cloud)
- [ ] MongoDB Atlas cluster (M10 or higher recommended for production)
- [ ] Redis instance (2GB memory minimum, Redis Cloud or AWS ElastiCache recommended)
- [ ] Cloudflare R2 bucket (for zero-egress-fee storage)
- [ ] Google Cloud project with Secret Manager API enabled
- [ ] OpenAI API key for LangChain features (from platform.openai.com)
- [ ] Python 3.11+ installed locally

## Step 1: Clone Repository

```bash
git clone https://github.com/Tek-Fly/h200-mug-positioning.git
cd h200-mug-positioning
```

## Step 2: Setup Environment

```bash
# Copy environment template
cp .env.runpod .env

# Edit with your actual credentials
nano .env  # or use your preferred editor
```

### Required Environment Variables

Edit the `.env` file and update these required values:

```env
# RunPod (get from https://runpod.io/console/user/settings)
RUNPOD_API_KEY=your_actual_runpod_api_key

# MongoDB Atlas (from your cluster connection string)
MONGODB_ATLAS_URI=mongodb+srv://your_username:your_password@your-cluster.mongodb.net/h200_production?retryWrites=true&w=majority

# Redis (from your Redis provider)
REDIS_HOST=your-redis-endpoint.com
REDIS_PASSWORD=your_redis_password

# Cloudflare R2 (from R2 dashboard)
R2_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key

# Google Cloud (from GCP console)
GCP_PROJECT_ID=your-gcp-project-id

# OpenAI (from platform.openai.com)
OPENAI_API_KEY=sk-your_openai_api_key

# Security (generate a secure key)
JWT_SECRET=your_secure_jwt_secret_minimum_32_characters
```

## Step 3: Install Dependencies

```bash
# Create virtual environment (Python 3.11+ required)
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip to latest version
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Install RunPod CLI tools
pip install runpod
```

## Step 4: Download AI Models

```bash
# This downloads YOLOv8 and CLIP models to the models/ directory
python scripts/setup_models.py

# Verify models were downloaded correctly (should show ~2GB of model files)
ls -lah models/

# Optional: Test model loading locally
python scripts/setup_models.py --verify-only
```

## Step 5: Deploy to RunPod

### Option A: Deploy Both Modes (Recommended for Production)
```bash
# Deploy both serverless and timed modes with production settings
python scripts/deploy_runpod.py --mode both --environment production
```

### Option B: Deploy Serverless Mode Only
```bash
# For variable/sporadic workloads
python scripts/deploy_runpod.py --mode serverless --environment production
```

### Option C: Deploy Timed Mode Only
```bash
# For continuous workloads or development
python scripts/deploy_runpod.py --mode timed --environment production
```

The deployment script will:
1. Create RunPod templates with your configuration
2. Upload your code and models
3. Configure H200 GPU instances
4. Set up networking and endpoints
5. Return your endpoint URLs

## Step 6: Validate Deployment

```bash
# Run validation checks
python scripts/validate_deployment.py
```

## Step 7: Monitor Deployment

```bash
# Start real-time monitoring
python scripts/monitor_runpod.py

# Monitor specific mode
python scripts/monitor_runpod.py --mode serverless --interval 5
```

## Deployment Modes

### Serverless Mode
- **Best for**: Sporadic workloads, cost optimization
- **Cold start**: 500ms-2s with FlashBoot
- **Cost**: ~$0.0002 per request
- **Auto-scales**: 0-10 workers

### Timed Mode
- **Best for**: Continuous workloads, development
- **Always on**: With 10-minute idle shutdown
- **Cost**: ~$3.50 per hour
- **Features**: Full API + Dashboard

## Post-Deployment

### Access Points

After deployment, you'll have:

1. **Serverless API Endpoint**:
   ```
   https://api.runpod.ai/v2/{endpoint_id}/runsync
   ```

2. **Timed Pod API**:
   ```
   http://{pod_ip}:8000/api/v1/
   ```

3. **Dashboard** (Timed mode only):
   ```
   http://{pod_ip}:3000
   ```

### Test the Deployment

#### Test Serverless Endpoint
```bash
# Get your endpoint ID from deployment output
ENDPOINT_ID="your-endpoint-id-here"
RUNPOD_API_KEY="your-runpod-api-key-here"

# Test with image URL
curl -X POST https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_url": "https://example.com/test-mug-image.jpg",
      "apply_rules": true,
      "confidence_threshold": 0.7
    }
  }'

# Test with base64 encoded image
IMAGE_BASE64=$(base64 -i test-image.jpg)
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

#### Test Timed Pod API
```bash
# Get your pod URL from deployment output
POD_URL="http://your-pod-id-8000.proxy.runpod.net"

# Test health endpoint (no auth required)
curl ${POD_URL}/api/health

# Test analysis endpoint with file upload
curl -X POST ${POD_URL}/api/v1/analyze/with-feedback \
  -H "Authorization: Bearer your-jwt-token" \
  -F "image=@test-image.jpg" \
  -F "include_feedback=true" \
  -F "confidence_threshold=0.7"

# Access the dashboard (in browser)
open ${POD_URL/:8000/:3000}  # Opens dashboard on port 3000
```

## Troubleshooting

### Common Issues

1. **MongoDB Connection Failed**
   - Check if your IP is whitelisted in MongoDB Atlas
   - Verify connection string format
   - Ensure database user has correct permissions

2. **Redis Connection Timeout**
   - Verify Redis host is accessible
   - Check if Redis password is correct
   - Ensure Redis has sufficient memory (2GB)

3. **RunPod Deployment Failed**
   - Verify RunPod API key is valid
   - Check if you have H200 GPU quota
   - Ensure account has sufficient credits

4. **Model Download Failed**
   - Check internet connectivity
   - Verify disk space (need ~5GB)
   - Try running with sudo if permission issues

### Getting Help

- Check logs: `runpod logs {pod_id} --tail 100`
- Monitor GPU: `python scripts/monitor_runpod.py`
- Review docs: `/docs/operations/troubleshooting.md`

## Cost Optimization Tips

1. **Use Serverless for Variable Loads**
   - Scales to zero when idle
   - Pay only for actual usage

2. **Enable Auto-Shutdown**
   - Timed pods shut down after 10 minutes idle
   - Prevents runaway costs

3. **Monitor Usage**
   - Use monitoring script to track costs
   - Set up billing alerts in RunPod

4. **Batch Processing**
   - Group requests to minimize cold starts
   - Use async processing for bulk operations

## Security Notes

- Never commit `.env` file to git
- Rotate JWT secrets regularly
- Use Google Secret Manager for production
- Enable audit logging for compliance
- Review `/docs/operations/security.md` for full guidelines

## RunPod Template Details

### Template Configuration

The system uses RunPod's native template system with:
- **Base Image**: RunPod PyTorch (includes CUDA 12.1)
- **Python Version**: 3.11
- **GPU**: H200 (80GB VRAM)
- **Persistent Storage**: 50GB NVMe volume for models

### Startup Sequence

1. **Git Clone**: Pulls latest code from repository
2. **Dependency Install**: Installs Python packages
3. **Model Loading**: Loads from persistent volume or downloads if missing
4. **Redis Warming**: Pre-caches models in Redis
5. **Service Start**: Launches API server or serverless handler

### Environment Variables

All secrets are automatically injected from your `.env` file during deployment:
- No manual configuration needed on RunPod console
- Secrets stored securely in RunPod's environment
- Automatic rotation support via deployment script

## Production Best Practices

### High Availability
```bash
# Deploy with redundancy
python scripts/deploy_runpod.py \
  --mode serverless \
  --min-workers 2 \
  --max-workers 10 \
  --environment production
```

### Monitoring Setup
```bash
# Enable comprehensive monitoring
python scripts/deploy_runpod.py \
  --mode both \
  --enable-monitoring \
  --alert-email your-email@example.com
```

### Cost Controls
```bash
# Set spending limits
python scripts/deploy_runpod.py \
  --mode timed \
  --max-hourly-cost 10.00 \
  --auto-shutdown-minutes 10
```

## Next Steps

1. **Monitor Performance**: Use `python scripts/monitor_runpod.py` to track metrics
2. **Configure Alerts**: Set up cost and performance alerts
3. **Implement Rules**: Create custom positioning rules via API
4. **Scale as Needed**: Adjust worker counts based on load
5. **Review Documentation**: 
   - API Reference: `/docs/api/endpoints.md`
   - Operations Guide: `/docs/operations/`
   - User Manual: `/docs/user-guides/user-manual.md`

## Support Resources

- **RunPod Console**: https://runpod.io/console
- **System Status**: Check health at `{POD_URL}/api/health`
- **Logs**: View with `runpod logs {POD_ID} --tail 100`
- **SSH Access**: `ssh root@{POD_IP} -p {SSH_PORT}`

For detailed documentation, see:
- Architecture: `/docs/developer-guides/architecture.md`
- API Reference: `/docs/api/endpoints.md`
- Troubleshooting: `/docs/operations/troubleshooting.md`