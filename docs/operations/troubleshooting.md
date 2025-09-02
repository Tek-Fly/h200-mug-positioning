# Troubleshooting Guide

Comprehensive guide to diagnosing and resolving common issues in the H200 Intelligent Mug Positioning System.

## Quick Diagnosis

### System Health Check

Start troubleshooting with these commands:

```bash
# Check overall system health
curl http://localhost:8000/api/health

# Check Docker containers
docker-compose ps

# Check logs for errors
docker-compose logs --tail=50

# Check resource usage
docker stats --no-stream

# Check GPU status (if available)
nvidia-smi
```

### Common Symptoms and Quick Fixes

| Symptom | Quick Check | Quick Fix |
|---------|-------------|-----------|
| API not responding | `curl localhost:8000/api/health` | `docker-compose restart h200-api` |
| Dashboard blank | Check browser console | Clear cache, reload page |
| Slow analysis | Check GPU usage | Restart containers, check memory |
| Authentication fails | Check JWT token | Regenerate token |
| Database errors | MongoDB connection | Check credentials, restart MongoDB |

## API and Backend Issues

### 1. API Server Not Starting

**Symptoms:**
- Container exits immediately
- Port binding errors
- Import errors in logs

**Diagnosis:**
```bash
# Check container logs
docker-compose logs h200-api

# Check port conflicts
lsof -i :8000

# Test Python imports
docker-compose exec h200-api python -c "from src.control.api.main import app"
```

**Solutions:**

**Port Already in Use:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill process or change port
kill -9 <PID>
# Or edit docker-compose.yml to use different port
```

**Python Import Errors:**
```bash
# Check Python path
docker-compose exec h200-api python -c "import sys; print(sys.path)"

# Verify dependencies
docker-compose exec h200-api pip list

# Reinstall dependencies
docker-compose exec h200-api pip install -r requirements.txt
```

**Environment Variable Issues:**
```bash
# Check environment variables
docker-compose exec h200-api env | grep -E "(MONGODB|REDIS|SECRET)"

# Validate .env file
cat .env | grep -v "^#" | grep -v "^$"
```

### 2. Database Connection Issues

#### MongoDB Connection Failures

**Symptoms:**
- "Connection timeout" errors
- "Authentication failed" messages
- Health check shows MongoDB as unhealthy

**Diagnosis:**
```bash
# Test MongoDB connection directly
mongosh "mongodb+srv://user:pass@cluster.mongodb.net/h200"

# Check MongoDB Atlas network access
curl -I https://cluster0.mongodb.net

# Test connection from container
docker-compose exec h200-api python -c "
from motor.motor_asyncio import AsyncIOMotorClient
client = AsyncIOMotorClient('your_mongodb_uri')
print(await client.admin.command('ping'))
"
```

**Solutions:**

**Network Access:**
```bash
# Check IP whitelist in MongoDB Atlas
# Add current IP or use 0.0.0.0/0 for development

# Test network connectivity
nslookup cluster0.mongodb.net
ping cluster0.mongodb.net
```

**Authentication Issues:**
```bash
# Verify credentials
echo "mongodb+srv://user:pass@cluster.mongodb.net" | base64

# Test with escaped special characters
# If password contains special characters, URL encode them
```

**Connection String Issues:**
```bash
# Correct format:
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/database?retryWrites=true&w=majority

# Common mistakes:
# Missing authSource for custom databases
# Wrong database name
# Missing connection parameters
```

#### Redis Connection Issues

**Symptoms:**
- Cache misses for all requests
- "Connection refused" errors
- High API latency

**Diagnosis:**
```bash
# Test Redis connection
redis-cli -h localhost -p 6379 -a password ping

# Check Redis from container
docker-compose exec h200-api python -c "
import asyncio
import aioredis
redis = aioredis.from_url('redis://localhost:6379')
print(await redis.ping())
"

# Check Redis memory and performance
redis-cli -h localhost -p 6379 -a password info memory
redis-cli -h localhost -p 6379 -a password info stats
```

**Solutions:**

**Connection Configuration:**
```bash
# Check Redis URL format
REDIS_URL=redis://password@localhost:6379/0

# For Redis clusters:
REDIS_URL=redis://password@cluster-endpoint:6379/0?ssl=true

# Test different connection methods
redis-cli -u redis://password@localhost:6379/0 ping
```

**Memory Issues:**
```bash
# Check Redis memory usage
redis-cli info memory | grep used_memory_human

# Clear cache if needed (development only)
redis-cli flushall

# Increase memory limit
redis-cli config set maxmemory 2gb
```

### 3. GPU and Model Issues

#### GPU Not Detected

**Symptoms:**
- "CUDA not available" warnings
- Models falling back to CPU
- Slow analysis performance

**Diagnosis:**
```bash
# Check GPU availability
nvidia-smi

# Check CUDA installation
nvcc --version

# Test PyTorch GPU access
docker-compose exec h200-api python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

**Solutions:**

**NVIDIA Docker Runtime:**
```bash
# Install NVIDIA Docker runtime
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Update docker-compose.yml to use GPU
services:
  h200-api:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

**Driver Issues:**
```bash
# Check NVIDIA driver version
nvidia-smi

# Update drivers if needed
sudo apt update
sudo apt install nvidia-driver-525  # Or latest version

# Reboot after driver update
sudo reboot
```

#### Model Loading Failures

**Symptoms:**
- "Model not found" errors
- Long loading times
- Out of memory errors

**Diagnosis:**
```bash
# Check model files
ls -la models/
du -sh models/*

# Test model loading
docker-compose exec h200-api python -c "
from src.core.models.manager import ModelManager
manager = ModelManager()
await manager.load_model('yolo_v8n')
"

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**Solutions:**

**Model Download Issues:**
```bash
# Download models manually
python scripts/download_models.py --force

# Check download location
ls -la ~/.cache/torch/hub/
ls -la models/

# Verify model file integrity
python -c "
import torch
model = torch.load('models/yolo_v8n.pt')
print('Model loaded successfully')
"
```

**GPU Memory Issues:**
```bash
# Clear GPU cache
docker-compose exec h200-api python -c "
import torch
torch.cuda.empty_cache()
print('GPU cache cleared')
"

# Reduce model size in config
# Edit .env:
MODEL_PRECISION=fp16  # Use half precision
BATCH_SIZE=1         # Reduce batch size
```

### 4. Performance Issues

#### Slow API Response Times

**Symptoms:**
- Response times > 2 seconds
- Timeouts on requests
- High CPU/memory usage

**Diagnosis:**
```bash
# Check API performance
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8000/api/health

# Check container resources
docker stats h200-api

# Profile API requests
# Add profiling to specific endpoints
```

**curl-format.txt:**
```
     time_namelookup:  %{time_namelookup}s
        time_connect:  %{time_connect}s
     time_appconnect:  %{time_appconnect}s
    time_pretransfer:  %{time_pretransfer}s
       time_redirect:  %{time_redirect}s
  time_starttransfer:  %{time_starttransfer}s
                     ──────────────
          time_total:  %{time_total}s
```

**Solutions:**

**Resource Optimization:**
```bash
# Increase container memory
# Edit docker-compose.yml:
services:
  h200-api:
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

**Cache Optimization:**
```bash
# Check cache hit rates
redis-cli info stats | grep keyspace

# Tune cache settings
redis-cli config set maxmemory-policy allkeys-lru
redis-cli config set maxmemory 4gb
```

**Model Optimization:**
```python
# Enable model compilation for speed
# In model configuration:
TORCH_COMPILE=true
TORCH_COMPILE_MODE=reduce-overhead

# Use quantization
MODEL_QUANTIZATION=int8

# Enable GPU memory pre-allocation
CUDA_MEMORY_PREALLOC=true
```

#### High Memory Usage

**Symptoms:**
- Container being killed (OOMKilled)
- Swap usage increasing
- System becoming unresponsive

**Diagnosis:**
```bash
# Monitor memory usage
top -p $(pgrep -f "python.*main:app")

# Check memory leaks
docker-compose exec h200-api python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024**2:.1f} MB')
"

# Check for memory leaks in models
nvidia-smi --query-gpu=memory.used --format=csv --loop=5
```

**Solutions:**

**Memory Leak Detection:**
```python
# Add memory monitoring to problematic functions
import tracemalloc

async def analyze_with_memory_tracking(image_data: bytes):
    tracemalloc.start()
    
    try:
        result = await analyze_image(image_data)
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        logger.info(f"Memory usage: {current / 1024**2:.1f} MB, "
                   f"Peak: {peak / 1024**2:.1f} MB")
    
    return result
```

**Garbage Collection Tuning:**
```python
# Explicit garbage collection after heavy operations
import gc
import torch

async def analyze_image_with_cleanup(image_data: bytes):
    try:
        result = await analyze_image(image_data)
        return result
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
```

## Frontend Issues

### 1. Dashboard Not Loading

**Symptoms:**
- Blank white page
- JavaScript errors in console
- Network request failures

**Diagnosis:**
```bash
# Check frontend container
docker-compose ps dashboard

# Check frontend logs
docker-compose logs dashboard

# Check browser console (F12)
# Look for JavaScript errors or network failures

# Test API connectivity
curl http://localhost:8000/api/health
```

**Solutions:**

**Build Issues:**
```bash
# Rebuild frontend
cd dashboard
npm run build

# Check build output
ls -la dist/

# Restart container
docker-compose restart dashboard
```

**API Connection Issues:**
```bash
# Check API base URL in frontend config
# dashboard/src/api/client.ts
grep -n "baseURL" dashboard/src/api/client.ts

# Update API endpoint if needed
export const API_BASE_URL = process.env.VUE_APP_API_URL || 'http://localhost:8000'
```

### 2. WebSocket Connection Issues

**Symptoms:**
- Real-time updates not working
- "WebSocket connection failed" errors
- Metrics not refreshing

**Diagnosis:**
```bash
# Test WebSocket connection manually
wscat -c ws://localhost:8000/ws/control-plane?token=YOUR_TOKEN

# Check browser network tab
# Look for WebSocket connection status

# Check server WebSocket logs
docker-compose logs h200-api | grep -i websocket
```

**Solutions:**

**Token Issues:**
```javascript
// Ensure valid JWT token in WebSocket connection
const token = localStorage.getItem('auth_token');
const ws = new WebSocket(`ws://localhost:8000/ws/control-plane?token=${token}`);

// Handle token expiration
ws.onerror = (error) => {
  if (error.code === 1008) { // Policy violation (invalid token)
    // Refresh token and reconnect
    refreshAuthToken().then(newToken => {
      connectWebSocket(newToken);
    });
  }
};
```

**Proxy Configuration:**
```nginx
# If using reverse proxy, configure WebSocket properly
# nginx.conf
location /ws/ {
    proxy_pass http://backend;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_cache_bypass $http_upgrade;
}
```

## Deployment Issues

### 1. RunPod Deployment Failures

#### Pod Creation Failures

**Symptoms:**
- "Pod creation failed" errors
- Pods stuck in pending state
- Resource allocation errors

**Diagnosis:**
```python
# Check RunPod API status
import aiohttp

async def check_runpod_status():
    async with aiohttp.ClientSession() as session:
        async with session.get(
            'https://api.runpod.ai/graphql',
            headers={'Authorization': f'Bearer {RUNPOD_API_KEY}'}
        ) as response:
            print(f"RunPod API status: {response.status}")

# Check pod logs
from src.deployment.client import RunPodClient

client = RunPodClient(api_key=RUNPOD_API_KEY)
logs = await client.get_pod_logs(pod_id)
print(logs)
```

**Solutions:**

**Resource Availability:**
```python
# Check available GPUs
available_gpus = await runpod_client.get_available_gpus()
print(f"Available H100 GPUs: {available_gpus['H100']['available']}")

# Use different GPU type if H100/H200 unavailable
FALLBACK_CONFIG = {
    "gpu_type": "A100",  # Fallback option
    "gpu_count": 1,
    "bid_per_gpu": 1.50  # Competitive bid
}
```

**Image Issues:**
```bash
# Test Docker image locally
docker run -it --rm tekfly/h200:serverless-latest /bin/bash

# Check image exists in registry
docker pull tekfly/h200:serverless-latest

# Rebuild and push if needed
docker build -f docker/Dockerfile.serverless -t tekfly/h200:serverless-latest .
docker push tekfly/h200:serverless-latest
```

#### Pod Performance Issues

**Symptoms:**
- High latency responses
- GPU underutilization
- Memory errors

**Diagnosis:**
```python
# Monitor pod performance
metrics = await runpod_client.get_pod_metrics(pod_id)
print(f"GPU utilization: {metrics['gpu_utilization']}%")
print(f"Memory usage: {metrics['memory_usage']}%")

# Check model loading time
start_time = time.time()
await model_manager.load_model('yolo_v8n')
load_time = time.time() - start_time
print(f"Model load time: {load_time:.2f}s")
```

**Solutions:**

**GPU Memory Optimization:**
```python
# Optimize model loading
OPTIMIZED_CONFIG = {
    "model_precision": "fp16",        # Half precision
    "enable_trt": True,               # TensorRT optimization
    "max_batch_size": 4,              # Smaller batches
    "enable_compilation": True,       # PyTorch compilation
}

# Implement model sharing between requests
class SharedModelManager:
    def __init__(self):
        self._loaded_models = {}
        self._model_lock = asyncio.Lock()
    
    async def get_model(self, model_name: str):
        async with self._model_lock:
            if model_name not in self._loaded_models:
                self._loaded_models[model_name] = await self.load_model(model_name)
            return self._loaded_models[model_name]
```

### 2. Auto-Scaling Issues

#### Serverless Scaling Problems

**Symptoms:**
- Instances not scaling up under load
- Instances scaling down too quickly
- Inconsistent performance

**Diagnosis:**
```python
# Check scaling metrics
scaling_data = await runpod_client.get_scaling_metrics(endpoint_id)
print(f"Active instances: {scaling_data['active_instances']}")
print(f"Queue length: {scaling_data['queue_length']}")
print(f"Average response time: {scaling_data['avg_response_time']}")

# Check scaling configuration
config = await runpod_client.get_endpoint_config(endpoint_id)
print(f"Min instances: {config['scaling']['min_instances']}")
print(f"Max instances: {config['scaling']['max_instances']}")
print(f"Scale threshold: {config['scaling']['scale_up_threshold']}")
```

**Solutions:**

**Scaling Configuration Tuning:**
```python
OPTIMIZED_SCALING_CONFIG = {
    "min_instances": 1,              # Keep one warm instance
    "max_instances": 10,             # Increase for high load
    "idle_timeout": 600,             # 10 minutes instead of 5
    "scale_up_threshold": 0.6,       # Scale up at 60% capacity
    "scale_down_threshold": 0.2,     # Scale down at 20% capacity
    "cold_start_boost": True,        # Priority scheduling for cold starts
}
```

**Load Balancing:**
```python
# Implement request distribution
class LoadBalancer:
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.current_loads = {ep: 0 for ep in endpoints}
    
    async def get_best_endpoint(self) -> str:
        """Select endpoint with lowest load."""
        loads = await asyncio.gather(*[
            self.check_endpoint_load(ep) for ep in self.endpoints
        ])
        
        min_load_endpoint = min(
            zip(self.endpoints, loads),
            key=lambda x: x[1]
        )[0]
        
        return min_load_endpoint
```

## Storage and Data Issues

### 1. Cloudflare R2 Issues

#### Upload Failures

**Symptoms:**
- "Upload failed" errors
- Slow upload speeds
- Authentication errors

**Diagnosis:**
```python
# Test R2 connection
from src.database.r2_storage import R2Client

r2_client = R2Client()
test_result = await r2_client.test_connection()
print(f"R2 connection test: {test_result}")

# Check R2 credentials
import boto3
s3_client = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY
)

try:
    response = s3_client.list_buckets()
    print("R2 connection successful")
except Exception as e:
    print(f"R2 connection failed: {e}")
```

**Solutions:**

**Credential Issues:**
```bash
# Verify R2 credentials format
echo $R2_ACCESS_KEY_ID | wc -c    # Should be 32 characters
echo $R2_SECRET_ACCESS_KEY | wc -c # Should be 43 characters

# Test with AWS CLI
aws s3 ls s3://your-bucket-name \
  --endpoint-url https://your-account-id.r2.cloudflarestorage.com \
  --profile r2
```

**Upload Performance:**
```python
# Implement upload retry logic
import backoff

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    max_time=30
)
async def upload_with_retry(file_path: str, content: bytes):
    """Upload with exponential backoff retry."""
    return await r2_client.upload(file_path, content)

# Use multipart upload for large files
async def upload_large_file(file_path: str, content: bytes):
    """Upload large files in chunks."""
    if len(content) > 5 * 1024 * 1024:  # > 5MB
        return await r2_client.multipart_upload(file_path, content)
    else:
        return await r2_client.single_upload(file_path, content)
```

### 2. Cache Performance Issues

#### Low Cache Hit Rates

**Symptoms:**
- Cache hit rate < 70%
- Repeated expensive operations
- High API latency

**Diagnosis:**
```bash
# Check cache statistics
redis-cli info stats | grep -E "(keyspace_hits|keyspace_misses)"

# Check cache key patterns
redis-cli keys "*" | head -20

# Monitor cache operations
redis-cli monitor | head -50
```

**Solutions:**

**Cache Key Optimization:**
```python
# Improve cache key design
class CacheKeyManager:
    @staticmethod
    def analysis_key(image_hash: str, config: dict) -> str:
        """Generate consistent cache key for analysis results."""
        config_hash = hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        return f"analysis:{image_hash}:{config_hash}"
    
    @staticmethod
    def model_key(model_name: str, version: str) -> str:
        """Generate cache key for model data."""
        return f"model:{model_name}:{version}"

# Implement cache warming
async def warm_cache():
    """Pre-load frequently used data into cache."""
    # Pre-load models
    await cache.set("model:yolo_v8n", model_data, ttl=3600)
    
    # Pre-load common rules
    rules = await rule_engine.get_active_rules()
    await cache.set("rules:active", rules, ttl=1800)
```

**TTL Optimization:**
```python
# Dynamic TTL based on data volatility
TTL_CONFIG = {
    "models": 3600,           # 1 hour (stable)
    "analysis_results": 300,  # 5 minutes (frequent changes)
    "rules": 1800,           # 30 minutes (occasional changes)
    "user_sessions": 86400,   # 24 hours (daily rotation)
    "system_metrics": 60,     # 1 minute (real-time data)
}

async def set_with_smart_ttl(key: str, value: Any, data_type: str):
    """Set cache value with appropriate TTL."""
    ttl = TTL_CONFIG.get(data_type, 300)  # Default 5 minutes
    await cache.set(key, value, ttl=ttl)
```

## Monitoring and Alerting Issues

### 1. Metrics Collection Problems

#### Missing Metrics

**Symptoms:**
- Empty Grafana dashboards
- Prometheus showing no data
- Alerting not triggering

**Diagnosis:**
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check metric endpoints
curl http://localhost:8000/api/metrics

# Verify Prometheus configuration
docker-compose -f configs/monitoring/docker-compose.monitoring.yml config
```

**Solutions:**

**Metric Endpoint Issues:**
```python
# Ensure metrics are properly exported
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Create metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'Request duration')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    
    return response

# Metrics endpoint
@app.get("/api/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

**Prometheus Configuration:**
```yaml
# configs/monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'h200-api'
    static_configs:
      - targets: ['h200-api:8000']
    metrics_path: '/api/metrics'
    scrape_interval: 10s
    
  - job_name: 'h200-gpu'
    static_configs:
      - targets: ['h200-api:8000']  
    metrics_path: '/gpu/metrics'
    scrape_interval: 5s  # More frequent for GPU metrics
```

### 2. Alert Configuration Issues

#### Alerts Not Firing

**Diagnosis:**
```bash
# Check alert rules syntax
promtool check rules configs/monitoring/alert_rules.yml

# Check Alertmanager configuration
curl http://localhost:9093/api/v1/status

# Test alert delivery
curl -X POST http://localhost:9093/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[{
    "labels": {"alertname": "TestAlert", "severity": "warning"},
    "annotations": {"summary": "Test alert"}
  }]'
```

**Solutions:**

**Alert Rule Debugging:**
```yaml
# configs/monitoring/alert_rules.yml
groups:
  - name: h200_alerts
    rules:
      - alert: HighAPILatency
        expr: histogram_quantile(0.95, http_request_duration_seconds) > 0.2
        for: 2m
        labels:
          severity: warning
          service: h200-api
        annotations:
          summary: "High API latency detected"
          description: "95th percentile latency is {{ $value }}s"
          
      - alert: GPUMemoryHigh
        expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.9
        for: 1m
        labels:
          severity: critical
          service: h200-gpu
        annotations:
          summary: "GPU memory usage critical"
          description: "GPU memory usage is {{ $value | humanizePercentage }}"
```

## Security Issues

### 1. Authentication Problems

#### JWT Token Issues

**Symptoms:**
- "Invalid token" errors
- Frequent re-authentication required
- Token validation failures

**Diagnosis:**
```bash
# Decode JWT token to check expiration
python -c "
import jwt
import json
token = 'your_jwt_token_here'
decoded = jwt.decode(token, options={'verify_signature': False})
print(json.dumps(decoded, indent=2, default=str))
"

# Check token expiration
date -d @$(python -c "
import jwt
token = 'your_jwt_token_here'
decoded = jwt.decode(token, options={'verify_signature': False})
print(decoded['exp'])
")
```

**Solutions:**

**Token Refresh Implementation:**
```python
class TokenManager:
    def __init__(self, api_client):
        self.api_client = api_client
        self.token = None
        self.refresh_token = None
        self.expires_at = None
    
    async def get_valid_token(self) -> str:
        """Get valid token, refreshing if necessary."""
        if self.token and datetime.utcnow() < self.expires_at - timedelta(minutes=5):
            return self.token
        
        # Refresh token
        await self.refresh_authentication()
        return self.token
    
    async def refresh_authentication(self):
        """Refresh expired token."""
        if self.refresh_token:
            # Use refresh token
            auth_response = await self.api_client.refresh_token(self.refresh_token)
        else:
            # Re-authenticate
            auth_response = await self.api_client.authenticate(
                username=self.username,
                password=self.password
            )
        
        self.token = auth_response['access_token']
        self.refresh_token = auth_response.get('refresh_token')
        self.expires_at = datetime.utcnow() + timedelta(seconds=auth_response['expires_in'])
```

### 2. SSL/TLS Issues

#### Certificate Problems

**Symptoms:**
- "SSL certificate verify failed" errors
- Browser security warnings
- API connection failures

**Diagnosis:**
```bash
# Check certificate validity
openssl s_client -connect your-domain.com:443 -servername your-domain.com

# Test certificate chain
curl -I https://your-domain.com/api/health

# Check certificate expiration
echo | openssl s_client -connect your-domain.com:443 2>/dev/null | openssl x509 -noout -dates
```

**Solutions:**

**Certificate Renewal:**
```bash
# Renew Let's Encrypt certificate
certbot renew --dry-run

# Update certificate in deployment
kubectl create secret tls h200-tls \
  --cert=/etc/letsencrypt/live/your-domain.com/fullchain.pem \
  --key=/etc/letsencrypt/live/your-domain.com/privkey.pem
```

## Performance Troubleshooting

### 1. GPU Performance Issues

#### Suboptimal GPU Utilization

**Symptoms:**
- GPU utilization < 70%
- Analysis taking longer than expected
- Inconsistent performance

**Diagnosis:**
```bash
# Monitor GPU usage during analysis
nvidia-smi dmon -s pucvmet -d 1

# Check GPU memory usage patterns
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# Profile GPU kernels
nsys profile python analyze_sample.py
```

**Solutions:**

**Batch Processing Optimization:**
```python
class OptimizedAnalyzer:
    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size
        self.pending_requests = []
        self.batch_processor = None
    
    async def analyze_image(self, image_data: bytes) -> AnalysisResult:
        """Analyze image with intelligent batching."""
        request = AnalysisRequest(image_data)
        
        # Add to batch queue
        future = asyncio.Future()
        self.pending_requests.append((request, future))
        
        # Trigger batch processing if needed
        if len(self.pending_requests) >= self.max_batch_size:
            await self.process_batch()
        elif not self.batch_processor:
            # Start timer for partial batch
            self.batch_processor = asyncio.create_task(
                self.process_batch_after_delay(0.1)  # 100ms delay
            )
        
        return await future
    
    async def process_batch(self):
        """Process accumulated requests as batch."""
        if not self.pending_requests:
            return
        
        requests, futures = zip(*self.pending_requests)
        self.pending_requests.clear()
        
        try:
            # Batch GPU processing
            results = await self.gpu_batch_process(requests)
            
            # Return results to futures
            for future, result in zip(futures, results):
                future.set_result(result)
                
        except Exception as e:
            # Handle batch failure
            for future in futures:
                future.set_exception(e)
```

### 2. Memory Issues

#### Memory Leaks

**Symptoms:**
- Memory usage continuously increasing
- Out of memory errors
- System becoming unresponsive

**Diagnosis:**
```python
# Memory profiling
import tracemalloc
import psutil

class MemoryProfiler:
    def __init__(self):
        self.snapshots = []
    
    def start_profiling(self):
        """Start memory profiling."""
        tracemalloc.start()
        self.snapshots = []
    
    def take_snapshot(self, label: str):
        """Take memory snapshot with label."""
        snapshot = tracemalloc.take_snapshot()
        process = psutil.Process()
        memory_info = process.memory_info()
        
        self.snapshots.append({
            'label': label,
            'tracemalloc': snapshot,
            'rss': memory_info.rss,
            'vms': memory_info.vms,
            'timestamp': datetime.utcnow()
        })
    
    def analyze_memory_growth(self):
        """Analyze memory growth between snapshots."""
        if len(self.snapshots) < 2:
            return
        
        for i in range(1, len(self.snapshots)):
            current = self.snapshots[i]
            previous = self.snapshots[i-1]
            
            # Compare tracemalloc snapshots
            top_stats = current['tracemalloc'].compare_to(
                previous['tracemalloc'], 'lineno'
            )
            
            print(f"\nMemory changes from {previous['label']} to {current['label']}:")
            for stat in top_stats[:10]:
                print(stat)
```

**Solutions:**

**Memory Management:**
```python
# Implement proper cleanup
import weakref
from contextlib import asynccontextmanager

class ResourceManager:
    def __init__(self):
        self._resources = weakref.WeakSet()
    
    @asynccontextmanager
    async def managed_resource(self, resource_factory):
        """Context manager for automatic resource cleanup."""
        resource = await resource_factory()
        self._resources.add(resource)
        
        try:
            yield resource
        finally:
            if hasattr(resource, 'cleanup'):
                await resource.cleanup()
            elif hasattr(resource, 'close'):
                await resource.close()

# Usage
async def analyze_with_cleanup(image_data: bytes):
    async with ResourceManager().managed_resource(lambda: ModelManager()) as models:
        result = await models.analyze_image(image_data)
        return result
```

## Emergency Procedures

### System Recovery

#### Complete System Failure

**Immediate Actions:**
1. Check overall system health
2. Identify failing components
3. Attempt service restart
4. Escalate if restart fails

```bash
#!/bin/bash
# emergency_recovery.sh

echo "=== H200 System Emergency Recovery ==="

# 1. Check system status
echo "Checking system status..."
curl -f http://localhost:8000/api/health || echo "API unhealthy"

# 2. Check all containers
echo "Container status:"
docker-compose ps

# 3. Check resource usage
echo "Resource usage:"
docker stats --no-stream

# 4. Attempt restart
echo "Attempting service restart..."
docker-compose restart

# 5. Wait and verify
sleep 30
curl -f http://localhost:8000/api/health && echo "Recovery successful" || echo "Recovery failed"

# 6. If failed, gather diagnostic information
if [ $? -ne 0 ]; then
    echo "Gathering diagnostic information..."
    docker-compose logs --tail=100 > emergency_logs.txt
    docker system df > disk_usage.txt
    nvidia-smi > gpu_status.txt 2>/dev/null || echo "No GPU" > gpu_status.txt
    
    echo "Diagnostic files created. Contact support with:"
    echo "- emergency_logs.txt"
    echo "- disk_usage.txt" 
    echo "- gpu_status.txt"
fi
```

#### Data Recovery

**Database Recovery:**
```python
# Restore from latest backup
async def emergency_database_restore():
    """Restore database from latest backup."""
    
    # 1. Stop all services
    await stop_all_services()
    
    # 2. Find latest backup
    backup_manager = BackupManager()
    latest_backup = await backup_manager.find_latest_backup()
    
    # 3. Restore database
    await backup_manager.restore_database(latest_backup['backup_id'])
    
    # 4. Verify data integrity
    await verify_database_integrity()
    
    # 5. Restart services
    await start_all_services()
    
    # 6. Run health checks
    health_status = await run_comprehensive_health_check()
    
    return {
        "status": "completed",
        "backup_used": latest_backup['backup_id'],
        "data_loss_minutes": calculate_data_loss(latest_backup['timestamp']),
        "health_status": health_status
    }
```

### Contact Information

For issues requiring immediate assistance:

**Emergency Contact:**
- Email: emergency@tekfly.co.uk
- Phone: +44 (0) 20 XXXX XXXX (24/7 for critical issues)

**Support Levels:**
- **Critical**: System down, data loss, security breach
- **High**: Major functionality impaired
- **Medium**: Minor functionality issues
- **Low**: Questions, feature requests

**When Contacting Support:**
Include the following information:
1. System status (output of health check)
2. Error messages and logs
3. Steps taken to resolve the issue
4. Impact on users/business
5. Environment details (production, staging, development)

This troubleshooting guide covers the most common issues and their resolutions. For issues not covered here, please consult the specific component documentation or contact support.