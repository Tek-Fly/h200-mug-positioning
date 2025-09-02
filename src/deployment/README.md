# RunPod Deployment Module

This module provides comprehensive deployment management for RunPod GPU instances, including serverless and timed deployments, network volume management, and integration with the control plane.

## Architecture

The deployment module is organized into several components:

### Core Components

- **`client.py`**: Async RunPod API client with retry logic and error handling
- **`config.py`**: Deployment configuration models and constants
- **`deployer.py`**: Core deployment functionality for creating, updating, and managing deployments
- **`manager.py`**: High-level deployment management with state tracking and monitoring
- **`orchestrator.py`**: Complex deployment strategies (blue-green, canary, rolling)
- **`validator.py`**: Configuration validation and deployment health checks
- **`volume.py`**: Network volume management for persistent model storage

## Usage Examples

### Basic Deployment

```python
from src.deployment import DeploymentManager, DeploymentConfig, GPUType, DeploymentMode

# Initialize manager
manager = DeploymentManager(api_key="your-runpod-api-key")
await manager.initialize()

# Create deployment config
config = DeploymentConfig(
    name="h200-mug-positioning",
    mode=DeploymentMode.SERVERLESS,
    docker_image="your-docker-image",
    gpu=GPUConfig(type=GPUType.H100, count=1),
    env_vars={
        "MONGODB_ATLAS_URI": "mongodb+srv://...",
        "REDIS_HOST": "redis.example.com",
        # ... other environment variables
    }
)

# Deploy
state = await manager.deploy(config)
print(f"Deployed: {state.deployment_id}")
```

### Using Deployment Orchestrator

```python
from src.deployment import DeploymentOrchestrator, DeploymentStrategy

# Initialize orchestrator
orchestrator = DeploymentOrchestrator(manager)

# Deploy with blue-green strategy
result = await orchestrator.deploy_application(
    app_name="mug-positioning",
    config=config,
    strategy=DeploymentStrategy.BLUE_GREEN
)

# Deploy with canary strategy
result = await orchestrator.deploy_application(
    app_name="mug-positioning",
    config=config,
    strategy=DeploymentStrategy.CANARY,
    canary_percentage=10,
    validation_duration_minutes=30
)
```

### Managing Deployments

```python
# List deployments
deployments = await manager.list_deployments(mode=DeploymentMode.SERVERLESS)

# Get deployment status
state = await manager.get_deployment("deployment-id")

# Stop deployment
await manager.stop_deployment("deployment-id")

# Start deployment
await manager.start_deployment("deployment-id")

# Terminate deployment
await manager.terminate_deployment("deployment-id")

# Get deployment metrics
metrics = await manager.get_deployment_metrics("deployment-id")

# Get deployment logs
logs = await manager.get_deployment_logs("deployment-id", lines=100)

# Scale serverless deployment
await manager.scale_deployment("deployment-id", min_workers=1, max_workers=10)
```

### Volume Management

```python
from src.deployment import VolumeManager

# Initialize volume manager
volume_manager = VolumeManager(client)

# Ensure volume exists
volume_id = await volume_manager.ensure_volume(
    name="model-storage",
    size_gb=100,
    region="us-east-1"
)

# Attach volume to deployment
await volume_manager.attach_to_deployment(
    volume_id=volume_id,
    deployment_id="deployment-id",
    mount_path="/models"
)

# Get volume metrics
metrics = await volume_manager.get_volume_metrics(volume_id)

# Cleanup unused volumes
cleaned = await volume_manager.cleanup_unused_volumes(days_unused=30)
```

### Configuration Validation

```python
from src.deployment import DeploymentValidator

# Initialize validator
validator = DeploymentValidator()

# Validate configuration
result = await validator.validate_config(config)
if not result["valid"]:
    print(f"Validation errors: {result['errors']}")

# Validate deployment health
health = await validator.validate_deployment_health(
    client=client,
    deployment_id="deployment-id",
    mode=DeploymentMode.SERVERLESS,
    checks=["status", "connectivity", "resources", "performance"]
)
```

## Deployment Strategies

### Recreate
- Terminates existing deployment and creates new one
- Simplest strategy with downtime
- Good for development environments

### Blue-Green
- Deploys new version alongside existing
- Switches traffic after health checks pass
- Zero-downtime deployment
- Automatic rollback on failure

### Canary
- Gradually rolls out to percentage of traffic
- Monitors metrics during validation period
- Promotes to full deployment if successful
- Automatic rollback on errors

### Rolling
- Updates deployments in batches
- Maintains service availability
- Good for multiple instance deployments

## Environment Presets

The module includes predefined configurations for different environments:

```python
from src.deployment.config import DEPLOYMENT_PRESETS

# Production preset
prod_config = DEPLOYMENT_PRESETS["production"]
# - H100 GPU with auto-scaling
# - Enhanced security and monitoring
# - Optimized for performance

# Staging preset
staging_config = DEPLOYMENT_PRESETS["staging"]
# - A100 GPU with fixed resources
# - Moderate security settings
# - Good for testing

# Development preset
dev_config = DEPLOYMENT_PRESETS["development"]
# - RTX3090 GPU with minimal resources
# - Relaxed security for debugging
# - Cost-optimized
```

## Integration with Control Plane

The deployment module integrates seamlessly with the control plane:

```python
# In control plane API
from src.deployment import DeploymentManager

# Initialize in FastAPI startup
@app.on_event("startup")
async def startup():
    app.state.deployment_manager = DeploymentManager(
        api_key=os.getenv("RUNPOD_API_KEY")
    )
    await app.state.deployment_manager.initialize()

# Use in endpoints
@app.post("/api/v1/deployments")
async def create_deployment(config: DeploymentConfig):
    state = await app.state.deployment_manager.deploy(config)
    return {"deployment_id": state.deployment_id, "status": state.status}

# WebSocket updates
@app.websocket("/ws/deployments/{deployment_id}")
async def deployment_updates(websocket: WebSocket, deployment_id: str):
    await websocket.accept()
    
    while True:
        metrics = await app.state.deployment_manager.get_deployment_metrics(deployment_id)
        await websocket.send_json(metrics)
        await asyncio.sleep(5)
```

## Best Practices

1. **Always validate configurations before deployment**
   ```python
   validation = await validator.validate_config(config)
   if not validation["valid"]:
       raise ValueError(f"Invalid config: {validation['errors']}")
   ```

2. **Use appropriate deployment strategies**
   - Production: Canary or Blue-Green
   - Staging: Blue-Green or Recreate
   - Development: Recreate

3. **Monitor deployment health**
   - Set up health checks
   - Monitor metrics
   - Configure alerts

4. **Manage costs**
   - Use auto-shutdown for idle deployments
   - Scale down during off-hours
   - Monitor usage and costs

5. **Handle failures gracefully**
   - Implement retry logic
   - Plan rollback strategies
   - Log all operations

## Error Handling

The module provides comprehensive error handling:

```python
from src.deployment.client import RunPodAPIError

try:
    state = await manager.deploy(config)
except RunPodAPIError as e:
    print(f"API Error: {e.message}")
    print(f"Status Code: {e.status_code}")
    print(f"Response: {e.response}")
except ValueError as e:
    print(f"Configuration Error: {e}")
except Exception as e:
    print(f"Unexpected Error: {e}")
```

## Monitoring and Metrics

The module provides detailed metrics:

- **Deployment Metrics**: Status, uptime, resource usage
- **Performance Metrics**: Latency, throughput, error rates
- **Cost Metrics**: GPU hours, storage usage, estimated costs
- **Health Metrics**: Service availability, resource health

## Security Considerations

1. **API Keys**: Store RunPod API keys securely
2. **Environment Variables**: Use secrets management
3. **Network Security**: Configure proper firewall rules
4. **Access Control**: Implement proper authentication
5. **Audit Logging**: Track all deployment operations

## Troubleshooting

Common issues and solutions:

1. **Deployment Stuck in Provisioning**
   - Check GPU availability in region
   - Verify Docker image accessibility
   - Check RunPod account limits

2. **Health Checks Failing**
   - Review application logs
   - Check resource utilization
   - Verify network connectivity

3. **High Error Rates**
   - Check application performance
   - Review scaling configuration
   - Monitor GPU memory usage

4. **Volume Mount Issues**
   - Ensure volume exists in same region
   - Check mount permissions
   - Verify volume isn't attached elsewhere