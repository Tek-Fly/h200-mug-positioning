# Control Plane Manager

The control plane manager provides comprehensive management and monitoring for H200 GPU servers on RunPod, including automatic shutdown of idle instances to save costs.

## Components

### 1. **ControlPlaneOrchestrator** (`orchestrator.py`)
The main orchestrator that coordinates all control plane components:
- Manages server lifecycle (deploy, start, stop, restart, scale)
- Coordinates monitoring and metrics collection
- Handles auto-shutdown based on idle time
- Provides unified dashboard data
- Manages real-time notifications via WebSocket

### 2. **ServerManager** (`server_manager.py`)
Manages server lifecycle on RunPod:
- Deploy new serverless and timed GPU instances
- Start/stop/restart servers
- Scale serverless endpoints
- Query server status from RunPod API
- Track server costs

### 3. **ResourceMonitor** (`resource_monitor.py`)
Monitors GPU and system resources:
- GPU utilization, memory, temperature, power draw
- CPU and system memory usage
- Disk and network statistics
- Resource alerts and thresholds
- nvidia-smi integration

### 4. **AutoShutdownScheduler** (`auto_shutdown.py`)
Automatically shuts down idle servers to save costs:
- Configurable idle timeout (default: 10 minutes)
- Tracks server activity and requests
- Sends warnings before shutdown
- Calculates cost savings
- Server protection from auto-shutdown

### 5. **MetricsCollector** (`metrics.py`)
Collects and aggregates performance metrics:
- Request latency and throughput
- GPU and model performance
- Cache hit rates
- Cost tracking
- Prometheus export support

### 6. **StatusTracker** (`status_tracker.py`)
Tracks deployment and server health:
- Health checks (endpoint, metrics, resources)
- Alert management
- Uptime tracking
- Error monitoring

### 7. **WebSocketNotifier** (`notifier.py`)
Real-time notifications via WebSocket:
- Metrics updates
- System alerts
- Activity logs
- Server events
- Batched notifications for efficiency

## Usage

### Basic Setup

```python
from src.control.manager.orchestrator import ControlPlaneOrchestrator

# Create orchestrator
orchestrator = ControlPlaneOrchestrator(
    runpod_api_key="your-api-key",
    idle_timeout_seconds=600,  # 10 minutes
    enable_auto_shutdown=True,
)

# Start orchestrator
await orchestrator.start()

# Deploy a serverless endpoint
server = await orchestrator.deploy_server(
    server_type=ServerType.SERVERLESS,
    config=ServerConfig(
        docker_image="h200-mug-positioning:latest",
        min_instances=0,
        max_instances=10,
        idle_timeout_seconds=600,
    ),
    auto_start=True,
)

# Stop orchestrator when done
await orchestrator.stop()
```

### API Integration

The control plane integrates with FastAPI through the `integration.py` module:

```python
from src.control.manager.integration import setup_control_plane

# In your FastAPI app
setup_control_plane(app)
```

This automatically:
- Initializes the orchestrator on startup
- Tracks requests for auto-shutdown
- Records metrics for all API calls
- Provides control plane endpoints

### Auto-Shutdown Configuration

Configure idle timeout and protection:

```python
# Set idle timeout
orchestrator.auto_shutdown.idle_timeout_seconds = 300  # 5 minutes

# Protect a server from auto-shutdown
orchestrator.auto_shutdown.protect_server("server-id")

# Remove protection
orchestrator.auto_shutdown.unprotect_server("server-id")
```

### Monitoring and Metrics

Access real-time metrics:

```python
# Get dashboard data
dashboard = await orchestrator.get_dashboard_data()

# Get specific server metrics
metrics = orchestrator.metrics_collector.get_server_metrics("server-id")

# Export Prometheus metrics
prometheus_data = orchestrator.metrics_collector.export_metrics("prometheus")
```

### WebSocket Notifications

Subscribe to real-time updates:

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/control-plane?token=YOUR_TOKEN');

// Subscribe to topics
ws.send(JSON.stringify({
    action: 'subscribe',
    topic: 'metrics'
}));

// Receive updates
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Update:', data);
};
```

## API Endpoints

### Server Control
- `POST /api/v1/servers/{type}/control` - Start/stop/restart/scale servers
- `GET /api/v1/servers` - List all servers
- `POST /api/v1/servers/deploy` - Deploy new server
- `DELETE /api/v1/servers/{id}` - Delete server

### Monitoring
- `GET /api/v1/servers/{id}/metrics` - Get server metrics
- `GET /api/v1/servers/{id}/status` - Get detailed status
- `GET /api/v1/servers/{id}/logs` - Get server logs

### Protection
- `POST /api/v1/servers/{id}/protect` - Protect from auto-shutdown
- `DELETE /api/v1/servers/{id}/protect` - Remove protection

### Dashboard
- `GET /api/v1/dashboard` - Get comprehensive dashboard data

## Environment Variables

```bash
# RunPod configuration
RUNPOD_API_KEY=your-api-key

# Control plane settings
IDLE_TIMEOUT_SECONDS=600
ENABLE_AUTO_SHUTDOWN=true
HEALTH_CHECK_INTERVAL=30
RESOURCE_MONITOR_INTERVAL=5
```

## Cost Optimization

The control plane helps optimize costs through:

1. **Auto-shutdown**: Automatically stops idle servers after configurable timeout
2. **Serverless scaling**: Scales to 0 when not in use
3. **Cost tracking**: Monitors and reports costs per server
4. **Resource monitoring**: Identifies underutilized resources
5. **Smart scheduling**: Optimizes server usage patterns

## Troubleshooting

### Server not auto-shutting down
- Check if server is protected: `orchestrator.auto_shutdown._protected_servers`
- Verify idle timeout: `orchestrator.auto_shutdown.idle_timeout_seconds`
- Check recent activity: `orchestrator.idle_tracker.get_idle_time("server-id")`

### Metrics not updating
- Ensure orchestrator is running: `orchestrator.is_running`
- Check resource monitor: `orchestrator.resource_monitor.is_monitoring`
- Verify GPU availability: `orchestrator.resource_monitor._gpu_available`

### WebSocket connection issues
- Check authentication token
- Verify WebSocket endpoint is accessible
- Check notification statistics: `orchestrator.notifier.get_statistics()`

## Testing

Run the test script to verify functionality:

```bash
python src/control/manager/test_control_plane.py
```

This tests all major components including:
- Resource monitoring
- Metrics collection
- Status tracking
- WebSocket notifications
- Auto-shutdown scheduler
- Dashboard data aggregation