# Monitoring & Alerting Guide

Comprehensive guide to monitoring and alerting for the H200 Intelligent Mug Positioning System.

## Monitoring Overview

The H200 System implements comprehensive observability with multiple monitoring layers:

- **Application Metrics**: Business logic and performance KPIs
- **Infrastructure Metrics**: System resources and health
- **GPU Metrics**: Specialized GPU utilization and performance
- **User Metrics**: Usage patterns and satisfaction
- **Cost Metrics**: Resource utilization and optimization

## Monitoring Stack

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Collection                          │
├─────────────────────────────────────────────────────────────┤
│ App Metrics │ GPU Stats │ System Metrics │ Logs           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Storage & Processing                       │
├─────────────────────────────────────────────────────────────┤
│ Prometheus │ Grafana │ Loki │ AlertManager               │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Visualization                             │
├─────────────────────────────────────────────────────────────┤
│ Dashboards │ Alerts │ Reports │ Mobile Notifications      │
└─────────────────────────────────────────────────────────────┘
```

### Component Setup

#### Prometheus Configuration

```yaml
# configs/monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'h200-production'
    region: 'us-east'

rule_files:
  - "alert_rules.yml"
  - "recording_rules.yml"

scrape_configs:
  # H200 API metrics
  - job_name: 'h200-api'
    static_configs:
      - targets: ['h200-api:8000']
    metrics_path: '/api/metrics'
    scrape_interval: 10s
    scrape_timeout: 5s
    
  # GPU metrics
  - job_name: 'h200-gpu'
    static_configs:
      - targets: ['h200-api:8000']
    metrics_path: '/gpu/metrics'
    scrape_interval: 5s  # More frequent for GPU
    
  # System metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    
  # Redis metrics
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    
  # Container metrics
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
      path_prefix: '/'
      scheme: 'http'
```

#### Grafana Dashboards

**H200 Overview Dashboard:**

```json
{
  "dashboard": {
    "id": null,
    "title": "H200 System Overview",
    "tags": ["h200", "overview"],
    "timezone": "UTC",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 50},
                {"color": "red", "value": 100}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Response Time",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "max": 2.0
          }
        }
      },
      {
        "id": 3,
        "title": "GPU Utilization",
        "type": "gauge",
        "targets": [
          {
            "expr": "gpu_utilization_percent",
            "legendFormat": "GPU %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 30},
                {"color": "green", "value": 70}
              ]
            }
          }
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

## Application Metrics

### 1. Business Metrics

#### Analysis Performance Tracking

```python
from prometheus_client import Counter, Histogram, Gauge, Summary

# Business metrics
ANALYSIS_REQUESTS = Counter(
    'analysis_requests_total',
    'Total analysis requests',
    ['user_id', 'status', 'has_feedback']
)

ANALYSIS_DURATION = Histogram(
    'analysis_duration_seconds',
    'Time spent analyzing images',
    ['model_version', 'image_size_category'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

RULE_EVALUATIONS = Counter(
    'rule_evaluations_total',
    'Total rule evaluations',
    ['rule_id', 'rule_type', 'matched']
)

USER_SATISFACTION = Gauge(
    'user_satisfaction_score',
    'Average user satisfaction score',
    ['time_period']
)

# GPU specific metrics
GPU_UTILIZATION = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id', 'operation_type']
)

GPU_MEMORY_USAGE = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory usage in bytes',
    ['gpu_id']
)

MODEL_INFERENCE_TIME = Histogram(
    'model_inference_duration_seconds',
    'Model inference time',
    ['model_name', 'batch_size'],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
)
```

#### Custom Metrics Collection

```python
class BusinessMetricsCollector:
    def __init__(self, db, redis):
        self.db = db
        self.redis = redis
        
    async def collect_daily_metrics(self):
        """Collect daily business metrics."""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)
        
        # Analysis metrics
        analysis_stats = await self.db.analysis_results.aggregate([
            {
                "$match": {
                    "timestamp": {"$gte": today, "$lt": tomorrow}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_requests": {"$sum": 1},
                    "avg_processing_time": {"$avg": "$processing_time_ms"},
                    "success_rate": {
                        "$avg": {"$cond": [{"$ifNull": ["$error", False]}, 0, 1]}
                    },
                    "unique_users": {"$addToSet": "$user_id"}
                }
            }
        ]).to_list(length=1)
        
        # User satisfaction metrics
        satisfaction_stats = await self.db.analysis_feedback.aggregate([
            {
                "$match": {
                    "timestamp": {"$gte": today, "$lt": tomorrow}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avg_satisfaction": {"$avg": "$satisfaction_score"},
                    "total_feedback": {"$sum": 1}
                }
            }
        ]).to_list(length=1)
        
        # Update Prometheus metrics
        if analysis_stats:
            stats = analysis_stats[0]
            USER_SATISFACTION.labels(time_period='daily').set(
                satisfaction_stats[0]['avg_satisfaction'] if satisfaction_stats else 0
            )
        
        return {
            'analysis': analysis_stats[0] if analysis_stats else {},
            'satisfaction': satisfaction_stats[0] if satisfaction_stats else {},
            'date': today.isoformat()
        }
```

### 2. Performance Metrics

#### Detailed Performance Tracking

```python
import time
import functools
from typing import Dict, List

class PerformanceTracker:
    def __init__(self):
        self.operation_times = {}
        self.resource_usage = {}
        
    def track_operation(self, operation_name: str):
        """Decorator to track operation performance."""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                try:
                    result = await func(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    raise
                finally:
                    end_time = time.time()
                    end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    # Record metrics
                    duration = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    # Update Prometheus metrics
                    ANALYSIS_DURATION.labels(
                        model_version="v1.0",
                        image_size_category="medium"
                    ).observe(duration)
                    
                    # Store detailed metrics
                    await self.record_performance_data(
                        operation_name=operation_name,
                        duration=duration,
                        memory_used=memory_delta,
                        success=success
                    )
                
                return result
            return wrapper
        return decorator
    
    async def record_performance_data(
        self,
        operation_name: str,
        duration: float,
        memory_used: int,
        success: bool
    ):
        """Record detailed performance data."""
        
        performance_data = {
            'timestamp': datetime.utcnow(),
            'operation': operation_name,
            'duration_ms': duration * 1000,
            'memory_used_mb': memory_used / (1024 * 1024),
            'success': success,
            'gpu_utilization': self.get_current_gpu_utilization(),
            'system_load': psutil.cpu_percent()
        }
        
        # Store in time-series database or cache
        await self.redis.lpush(
            f"performance:{operation_name}",
            json.dumps(performance_data)
        )
        
        # Keep only recent data (last 1000 entries)
        await self.redis.ltrim(f"performance:{operation_name}", 0, 999)

# Usage
tracker = PerformanceTracker()

@tracker.track_operation("image_analysis")
async def analyze_image_tracked(image_data: bytes):
    return await analyze_image(image_data)
```

## GPU Monitoring

### 1. NVIDIA GPU Metrics

#### GPU Metrics Collection

```python
import pynvml
from prometheus_client import Gauge

class GPUMetricsCollector:
    def __init__(self):
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        
        # Prometheus metrics
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id', 'gpu_name']
        )
        
        self.gpu_memory_used = Gauge(
            'gpu_memory_used_bytes',
            'GPU memory used in bytes',
            ['gpu_id', 'gpu_name']
        )
        
        self.gpu_temperature = Gauge(
            'gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['gpu_id', 'gpu_name']
        )
        
        self.gpu_power_usage = Gauge(
            'gpu_power_usage_watts',
            'GPU power usage in watts',
            ['gpu_id', 'gpu_name']
        )
    
    async def collect_gpu_metrics(self):
        """Collect comprehensive GPU metrics."""
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get device info
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self.gpu_utilization.labels(gpu_id=str(i), gpu_name=name).set(util.gpu)
            
            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.gpu_memory_used.labels(gpu_id=str(i), gpu_name=name).set(mem_info.used)
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            self.gpu_temperature.labels(gpu_id=str(i), gpu_name=name).set(temp)
            
            # Power
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                self.gpu_power_usage.labels(gpu_id=str(i), gpu_name=name).set(power)
            except pynvml.NVMLError:
                # Power monitoring not supported on all GPUs
                pass
    
    async def get_detailed_gpu_info(self) -> List[Dict]:
        """Get detailed GPU information for monitoring."""
        gpu_info = []
        
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            info = {
                'gpu_id': i,
                'name': pynvml.nvmlDeviceGetName(handle).decode('utf-8'),
                'driver_version': pynvml.nvmlSystemGetDriverVersion().decode('utf-8'),
                'cuda_version': pynvml.nvmlSystemGetCudaDriverVersion(),
                'utilization': pynvml.nvmlDeviceGetUtilizationRates(handle)._asdict(),
                'memory': pynvml.nvmlDeviceGetMemoryInfo(handle)._asdict(),
                'temperature': pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                'performance_state': pynvml.nvmlDeviceGetPerformanceState(handle),
                'processes': []
            }
            
            # Get running processes
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in processes:
                    info['processes'].append({
                        'pid': proc.pid,
                        'used_memory': proc.usedGpuMemory
                    })
            except pynvml.NVMLError:
                pass
            
            gpu_info.append(info)
        
        return gpu_info
```

### 2. Model Performance Monitoring

#### Model Inference Tracking

```python
class ModelPerformanceMonitor:
    def __init__(self):
        self.inference_times = {}
        self.accuracy_metrics = {}
        
    async def track_model_inference(
        self,
        model_name: str,
        input_data: torch.Tensor,
        inference_func: callable
    ):
        """Track model inference performance."""
        
        # Pre-inference metrics
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated()
        
        # Run inference
        with torch.no_grad():
            result = await inference_func(input_data)
        
        # Post-inference metrics
        torch.cuda.synchronize()  # Ensure GPU operations complete
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated()
        
        # Calculate metrics
        inference_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Record metrics
        MODEL_INFERENCE_TIME.labels(
            model_name=model_name,
            batch_size=str(input_data.shape[0])
        ).observe(inference_time)
        
        # Store detailed metrics
        await self.store_inference_metrics({
            'model_name': model_name,
            'inference_time_ms': inference_time * 1000,
            'memory_used_mb': memory_used / (1024 * 1024),
            'batch_size': input_data.shape[0],
            'input_shape': list(input_data.shape),
            'timestamp': datetime.utcnow()
        })
        
        return result
    
    async def monitor_model_accuracy(
        self,
        model_name: str,
        predictions: List[Any],
        ground_truth: List[Any]
    ):
        """Monitor model accuracy over time."""
        
        # Calculate accuracy metrics
        accuracy = calculate_accuracy(predictions, ground_truth)
        precision = calculate_precision(predictions, ground_truth)
        recall = calculate_recall(predictions, ground_truth)
        f1_score = calculate_f1_score(predictions, ground_truth)
        
        # Store in time-series data
        accuracy_data = {
            'model_name': model_name,
            'timestamp': datetime.utcnow(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'sample_count': len(predictions)
        }
        
        await self.store_accuracy_metrics(accuracy_data)
        
        # Alert if accuracy drops significantly
        if accuracy < 0.8:  # Threshold
            await self.trigger_accuracy_alert(model_name, accuracy)
```

## Alerting System

### 1. Alert Rules Configuration

#### Critical System Alerts

```yaml
# configs/monitoring/alert_rules.yml
groups:
  - name: h200_critical
    rules:
      - alert: SystemDown
        expr: up{job="h200-api"} == 0
        for: 30s
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "H200 API is down"
          description: "H200 API has been down for more than 30 seconds"
          runbook_url: "https://docs.tekfly.co.uk/runbooks/system-down"
          
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"
          
      - alert: GPUMemoryExhausted
        expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.95
        for: 1m
        labels:
          severity: critical
          team: ml-ops
        annotations:
          summary: "GPU memory critically high"
          description: "GPU memory usage is {{ $value | humanizePercentage }}"
          action: "Clear GPU cache or restart service"

  - name: h200_performance
    rules:
      - alert: SlowAPIResponse
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "API response time degraded"
          description: "95th percentile response time is {{ $value }}s"
          
      - alert: LowCacheHitRate
        expr: cache_hit_rate < 0.7
        for: 10m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "Cache hit rate below threshold"
          description: "Cache hit rate is {{ $value | humanizePercentage }}"
          
      - alert: HighGPUUtilization
        expr: gpu_utilization_percent > 90
        for: 5m
        labels:
          severity: warning
          team: ml-ops
        annotations:
          summary: "GPU utilization very high"
          description: "GPU utilization is {{ $value }}% for 5+ minutes"

  - name: h200_business
    rules:
      - alert: NoAnalysisRequests
        expr: rate(analysis_requests_total[1h]) == 0
        for: 30m
        labels:
          severity: warning
          team: business
        annotations:
          summary: "No analysis requests received"
          description: "No analysis requests in the last hour"
          
      - alert: LowUserSatisfaction
        expr: user_satisfaction_score < 3.5
        for: 1h
        labels:
          severity: warning
          team: product
        annotations:
          summary: "User satisfaction below threshold"
          description: "Average satisfaction is {{ $value }}/5.0"
```

### 2. AlertManager Configuration

#### Routing and Notification

```yaml
# configs/monitoring/alertmanager/alertmanager.yml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@tekfly.co.uk'
  smtp_auth_username: 'alerts@tekfly.co.uk'
  smtp_auth_password_file: '/etc/alertmanager/smtp_password'

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'default'
  
  routes:
    # Critical alerts go to PagerDuty
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
      group_wait: 5s
      repeat_interval: 5m
      
    # Platform alerts go to Slack
    - match:
        team: platform
      receiver: 'slack-platform'
      
    # ML Ops alerts
    - match:
        team: ml-ops
      receiver: 'slack-ml-ops'
      
    # Business alerts  
    - match:
        team: business
      receiver: 'email-business'

receivers:
  - name: 'default'
    email_configs:
      - to: 'team@tekfly.co.uk'
        subject: '[H200] {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Severity: {{ .Labels.severity }}
          Time: {{ .StartsAt }}
          {{ end }}

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
        description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'
        
  - name: 'slack-platform'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#h200-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Severity:* {{ .Labels.severity }}
          {{ if .Annotations.runbook_url }}*Runbook:* {{ .Annotations.runbook_url }}{{ end }}
          {{ end }}
```

### 3. Advanced Alerting

#### Intelligent Alert Correlation

```python
class IntelligentAlerting:
    def __init__(self, alert_history_db):
        self.alert_history = alert_history_db
        self.correlation_rules = self.load_correlation_rules()
        
    async def process_alert(self, alert: Dict):
        """Process alert with correlation and suppression."""
        
        # 1. Store alert
        await self.store_alert(alert)
        
        # 2. Check for correlations
        correlated_alerts = await self.find_correlated_alerts(alert)
        
        # 3. Apply suppression rules
        if await self.should_suppress_alert(alert, correlated_alerts):
            logger.info(f"Suppressing correlated alert: {alert['alertname']}")
            return
        
        # 4. Enhance alert with context
        enhanced_alert = await self.enhance_alert_context(alert)
        
        # 5. Route to appropriate channels
        await self.route_alert(enhanced_alert)
    
    async def find_correlated_alerts(self, alert: Dict) -> List[Dict]:
        """Find correlated alerts in recent history."""
        
        # Look for related alerts in last 10 minutes
        time_window = datetime.utcnow() - timedelta(minutes=10)
        
        related_alerts = await self.alert_history.find({
            'timestamp': {'$gte': time_window},
            'labels.service': alert['labels'].get('service'),
            'resolved': False
        }).to_list(length=100)
        
        # Apply correlation rules
        correlated = []
        for rule in self.correlation_rules:
            if rule['condition'](alert, related_alerts):
                correlated.extend(rule['correlated_alerts'](alert, related_alerts))
        
        return correlated
    
    async def enhance_alert_context(self, alert: Dict) -> Dict:
        """Add contextual information to alert."""
        
        enhanced = alert.copy()
        
        # Add system context
        enhanced['context'] = {
            'recent_deployments': await self.get_recent_deployments(),
            'current_load': await self.get_current_system_load(),
            'related_metrics': await self.get_related_metrics(alert),
            'suggested_actions': await self.get_suggested_actions(alert)
        }
        
        return enhanced
```

#### Custom Alert Actions

```python
class AlertActionExecutor:
    def __init__(self):
        self.action_handlers = {
            'auto_restart': self.auto_restart_service,
            'scale_up': self.auto_scale_up,
            'clear_cache': self.auto_clear_cache,
            'emergency_contact': self.emergency_escalation
        }
    
    async def execute_alert_actions(self, alert: Dict):
        """Execute automated actions for alerts."""
        
        actions = alert.get('annotations', {}).get('actions', '').split(',')
        
        for action in actions:
            action = action.strip()
            if action in self.action_handlers:
                try:
                    await self.action_handlers[action](alert)
                    logger.info(f"Executed alert action: {action}")
                except Exception as e:
                    logger.error(f"Failed to execute action {action}: {e}")
    
    async def auto_restart_service(self, alert: Dict):
        """Automatically restart service if safe to do so."""
        service_name = alert['labels'].get('service')
        
        if not service_name:
            return
        
        # Check if restart is safe
        if await self.is_safe_to_restart(service_name):
            # Perform graceful restart
            await self.graceful_restart_service(service_name)
            
            # Notify of action taken
            await self.send_notification(
                f"Automatically restarted {service_name} due to alert: {alert['alertname']}"
            )
    
    async def auto_scale_up(self, alert: Dict):
        """Automatically scale up resources."""
        if 'serverless' in alert['labels'].get('deployment_type', ''):
            # Scale up serverless instances
            current_max = await self.get_current_max_instances()
            new_max = min(current_max * 2, 20)  # Don't exceed 20 instances
            
            await self.update_scaling_config(max_instances=new_max)
            
            await self.send_notification(
                f"Auto-scaled serverless instances to {new_max} due to load"
            )
```

## Log Management

### 1. Structured Logging

#### Log Configuration

```python
import structlog
from pythonjsonlogger import jsonlogger

class StructuredLogging:
    def __init__(self):
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.add_logger_name,
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.dev.ConsoleRenderer() if DEBUG else structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(30),  # INFO level
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    def get_logger(self, name: str):
        """Get structured logger for component."""
        return structlog.get_logger(name)

# Usage
logger = StructuredLogging().get_logger(__name__)

async def analyze_image_with_logging(image_data: bytes, user_id: str):
    """Analyze image with comprehensive logging."""
    
    # Bind context for all log messages
    log = logger.bind(
        operation="image_analysis",
        user_id=user_id,
        image_size=len(image_data),
        request_id=str(uuid4())
    )
    
    log.info("Starting image analysis")
    
    try:
        # Analysis logic
        start_time = time.time()
        result = await perform_analysis(image_data)
        duration = time.time() - start_time
        
        log.info(
            "Analysis completed successfully",
            duration_ms=duration * 1000,
            detections_count=len(result.detections),
            confidence_avg=result.average_confidence
        )
        
        return result
        
    except Exception as e:
        log.error(
            "Analysis failed",
            error=str(e),
            error_type=type(e).__name__
        )
        raise
```

### 2. Log Aggregation

#### Centralized Logging with Loki

```yaml
# configs/monitoring/loki/loki-config.yml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: false
  retention_period: 0s
```

#### Log Processing Pipeline

```python
class LogProcessor:
    def __init__(self, loki_client):
        self.loki = loki_client
        
    async def process_application_logs(self, log_entry: Dict):
        """Process and forward application logs."""
        
        # Enrich log entry
        enriched_log = {
            **log_entry,
            'service': 'h200-api',
            'version': '1.0.0',
            'environment': os.getenv('ENVIRONMENT', 'unknown'),
            'instance_id': os.getenv('HOSTNAME', 'unknown')
        }
        
        # Add performance context if available
        if 'operation' in log_entry:
            enriched_log['performance_metrics'] = await self.get_operation_metrics(
                log_entry['operation']
            )
        
        # Forward to Loki
        await self.loki.push_log(enriched_log)
        
        # Check for error patterns
        if log_entry.get('level') == 'ERROR':
            await self.analyze_error_pattern(enriched_log)
    
    async def analyze_error_pattern(self, error_log: Dict):
        """Analyze error patterns for automated remediation."""
        
        error_signature = self.create_error_signature(error_log)
        
        # Check for known error patterns
        pattern_match = await self.find_matching_pattern(error_signature)
        
        if pattern_match:
            # Execute remediation action
            await self.execute_remediation_action(pattern_match['action'])
            
            logger.info(
                f"Executed automatic remediation for error pattern: {pattern_match['name']}"
            )
```

## Cost Monitoring

### 1. Resource Cost Tracking

#### Cost Calculation

```python
class CostMonitor:
    def __init__(self):
        self.pricing = {
            'runpod': {
                'H100': {'hourly': 1.69, 'serverless_per_second': 0.00047},
                'H200': {'hourly': 3.50, 'serverless_per_second': 0.00097},
                'A100': {'hourly': 1.25, 'serverless_per_second': 0.00035}
            },
            'storage': {
                'r2_per_gb_month': 0.015,
                'mongodb_atlas_m10': 57.00,  # Monthly
                'redis_cloud_1gb': 15.00     # Monthly
            },
            'network': {
                'cloudflare_r2_egress': 0.00,  # Free
                'runpod_egress_per_gb': 0.10
            }
        }
    
    async def calculate_daily_cost(self, date: datetime) -> Dict:
        """Calculate comprehensive daily costs."""
        
        # GPU compute costs
        gpu_usage = await self.get_gpu_usage_for_date(date)
        compute_cost = 0
        
        for usage in gpu_usage:
            if usage['deployment_type'] == 'serverless':
                # Serverless billing
                seconds_used = usage['total_inference_seconds']
                rate = self.pricing['runpod'][usage['gpu_type']]['serverless_per_second']
                compute_cost += seconds_used * rate
            else:
                # Timed instance billing
                hours_used = usage['total_hours']
                rate = self.pricing['runpod'][usage['gpu_type']]['hourly']
                compute_cost += hours_used * rate
        
        # Storage costs (prorated daily)
        storage_cost = await self.calculate_storage_cost(date)
        
        # Network costs
        network_cost = await self.calculate_network_cost(date)
        
        # Additional service costs
        service_cost = await self.calculate_service_cost(date)
        
        total_cost = compute_cost + storage_cost + network_cost + service_cost
        
        return {
            'date': date.isoformat(),
            'compute_cost': compute_cost,
            'storage_cost': storage_cost,
            'network_cost': network_cost,
            'service_cost': service_cost,
            'total_cost': total_cost,
            'breakdown': {
                'gpu_usage': gpu_usage,
                'request_count': await self.get_request_count_for_date(date),
                'data_transfer_gb': await self.get_data_transfer_for_date(date)
            }
        }
    
    async def cost_optimization_analysis(self) -> Dict:
        """Analyze costs and suggest optimizations."""
        
        # Get recent cost data
        recent_costs = await self.get_costs_last_30_days()
        
        optimizations = []
        
        # Check for idle GPU time
        idle_percentage = await self.calculate_idle_gpu_percentage()
        if idle_percentage > 50:
            optimizations.append({
                'type': 'idle_reduction',
                'description': f'GPU idle {idle_percentage:.1f}% of time',
                'potential_savings': recent_costs['avg_daily'] * 0.3,
                'action': 'Enable aggressive auto-shutdown'
            })
        
        # Check cache efficiency
        cache_hit_rate = await self.get_average_cache_hit_rate()
        if cache_hit_rate < 0.8:
            potential_reduction = recent_costs['compute_cost'] * (0.8 - cache_hit_rate)
            optimizations.append({
                'type': 'cache_optimization',
                'description': f'Cache hit rate only {cache_hit_rate:.1%}',
                'potential_savings': potential_reduction,
                'action': 'Increase cache size and TTL'
            })
        
        return {
            'current_monthly_cost': recent_costs['monthly_projection'],
            'optimization_opportunities': optimizations,
            'total_potential_savings': sum(opt['potential_savings'] for opt in optimizations)
        }
```

### 2. Budget Alerting

#### Cost-Based Alerts

```yaml
# Cost monitoring alerts
groups:
  - name: h200_cost
    rules:
      - alert: DailyCostExceeded
        expr: daily_cost_usd > 100
        for: 1h
        labels:
          severity: warning
          team: finance
        annotations:
          summary: "Daily cost budget exceeded"
          description: "Daily cost is ${{ $value }} (budget: $100)"
          
      - alert: MonthlyBudgetProjection
        expr: monthly_cost_projection_usd > 2500
        for: 1d
        labels:
          severity: warning
          team: finance
        annotations:
          summary: "Monthly budget projection exceeded"
          description: "Projected monthly cost: ${{ $value }}"
          
      - alert: UnusualCostSpike
        expr: increase(daily_cost_usd[1h]) > 50
        for: 30m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "Unusual cost increase detected"
          description: "Cost increased by ${{ $value }} in 1 hour"
          action: "Investigate for runaway processes"
```

## Health Monitoring

### 1. Service Health Checks

#### Comprehensive Health Monitoring

```python
class HealthMonitor:
    def __init__(self):
        self.health_checks = {
            'api': self.check_api_health,
            'database': self.check_database_health,
            'cache': self.check_cache_health,
            'gpu': self.check_gpu_health,
            'storage': self.check_storage_health,
            'external_services': self.check_external_services
        }
    
    async def run_comprehensive_health_check(self) -> Dict:
        """Run all health checks and return status."""
        
        results = {}
        overall_healthy = True
        
        # Run all checks concurrently
        check_tasks = {
            name: asyncio.create_task(check_func())
            for name, check_func in self.health_checks.items()
        }
        
        # Wait for all checks to complete
        for name, task in check_tasks.items():
            try:
                result = await asyncio.wait_for(task, timeout=10.0)
                results[name] = result
                
                if not result['healthy']:
                    overall_healthy = False
                    
            except asyncio.TimeoutError:
                results[name] = {
                    'healthy': False,
                    'error': 'Health check timeout',
                    'response_time_ms': 10000
                }
                overall_healthy = False
            except Exception as e:
                results[name] = {
                    'healthy': False,
                    'error': str(e),
                    'response_time_ms': None
                }
                overall_healthy = False
        
        return {
            'overall_healthy': overall_healthy,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': results,
            'summary': self.generate_health_summary(results)
        }
    
    async def check_api_health(self) -> Dict:
        """Check API service health."""
        start_time = time.time()
        
        try:
            # Test basic endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8000/api/health') as response:
                    if response.status == 200:
                        data = await response.json()
                        response_time = (time.time() - start_time) * 1000
                        
                        return {
                            'healthy': data.get('status') == 'healthy',
                            'response_time_ms': response_time,
                            'version': data.get('version'),
                            'details': data.get('services', {})
                        }
                    else:
                        return {
                            'healthy': False,
                            'error': f'HTTP {response.status}',
                            'response_time_ms': (time.time() - start_time) * 1000
                        }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'response_time_ms': (time.time() - start_time) * 1000
            }
    
    async def check_gpu_health(self) -> Dict:
        """Check GPU health and availability."""
        try:
            if not torch.cuda.is_available():
                return {
                    'healthy': False,
                    'error': 'CUDA not available',
                    'gpu_count': 0
                }
            
            gpu_count = torch.cuda.device_count()
            gpu_details = []
            
            for i in range(gpu_count):
                # Check if GPU is accessible
                try:
                    device_props = torch.cuda.get_device_properties(i)
                    memory_info = torch.cuda.mem_get_info(i)
                    
                    gpu_details.append({
                        'gpu_id': i,
                        'name': device_props.name,
                        'total_memory_gb': device_props.total_memory / (1024**3),
                        'free_memory_gb': memory_info[0] / (1024**3),
                        'used_memory_gb': (memory_info[1] - memory_info[0]) / (1024**3),
                        'healthy': True
                    })
                except Exception as e:
                    gpu_details.append({
                        'gpu_id': i,
                        'healthy': False,
                        'error': str(e)
                    })
            
            all_healthy = all(gpu['healthy'] for gpu in gpu_details)
            
            return {
                'healthy': all_healthy,
                'gpu_count': gpu_count,
                'gpus': gpu_details
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'gpu_count': 0
            }
```

### 2. Automated Recovery

#### Self-Healing System

```python
class SelfHealingSystem:
    def __init__(self):
        self.recovery_actions = {
            'api_unresponsive': self.restart_api_service,
            'gpu_memory_full': self.clear_gpu_memory,
            'database_connection_failed': self.reconnect_database,
            'cache_unavailable': self.restart_cache_service,
            'high_error_rate': self.investigate_errors
        }
        
    async def monitor_and_heal(self):
        """Continuous monitoring with automated healing."""
        
        while True:
            try:
                # Run health checks
                health_status = await self.run_health_checks()
                
                # Identify issues
                issues = self.identify_issues(health_status)
                
                # Attempt automated recovery
                for issue in issues:
                    if issue['type'] in self.recovery_actions:
                        await self.attempt_recovery(issue)
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Longer wait on error
    
    async def attempt_recovery(self, issue: Dict):
        """Attempt automated recovery for identified issue."""
        
        recovery_action = self.recovery_actions[issue['type']]
        
        logger.info(f"Attempting recovery for issue: {issue['description']}")
        
        try:
            # Execute recovery action
            result = await recovery_action(issue)
            
            if result['success']:
                logger.info(f"Recovery successful: {result['message']}")
                
                # Verify recovery
                await asyncio.sleep(10)  # Wait for recovery
                verification = await self.verify_recovery(issue)
                
                if verification['recovered']:
                    logger.info("Recovery verified successfully")
                else:
                    logger.warning("Recovery action completed but issue persists")
            else:
                logger.error(f"Recovery failed: {result['error']}")
                
        except Exception as e:
            logger.error(f"Recovery action failed: {e}")
    
    async def clear_gpu_memory(self, issue: Dict) -> Dict:
        """Clear GPU memory to resolve memory issues."""
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Clear application-level caches
            if hasattr(app.state, 'model_manager'):
                await app.state.model_manager.clear_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            return {
                'success': True,
                'message': 'GPU memory cleared successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
```

This comprehensive monitoring and alerting setup ensures the H200 System maintains optimal performance and reliability through proactive monitoring, intelligent alerting, and automated recovery capabilities.