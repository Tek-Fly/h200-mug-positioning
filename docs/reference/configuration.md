# Configuration Reference

Complete reference for all configuration options in the H200 Intelligent Mug Positioning System.

## Environment Variables

### Core System Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENVIRONMENT` | String | `development` | Deployment environment (development, staging, production) |
| `DEBUG` | Boolean | `false` | Enable debug mode with verbose logging |
| `LOG_LEVEL` | String | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `SECRET_KEY` | String | **Required** | Secret key for JWT token signing (256-bit recommended) |
| `JWT_ALGORITHM` | String | `HS256` | JWT signing algorithm |
| `JWT_EXPIRY_HOURS` | Integer | `24` | JWT token expiration time in hours |

### Database Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MONGODB_URI` | String | **Required** | MongoDB connection string |
| `MONGODB_DATABASE` | String | `h200` | Database name |
| `MONGODB_POOL_SIZE` | Integer | `20` | Connection pool size |
| `MONGODB_TIMEOUT_MS` | Integer | `5000` | Connection timeout in milliseconds |
| `REDIS_URL` | String | **Required** | Redis connection URL |
| `REDIS_POOL_SIZE` | Integer | `10` | Redis connection pool size |
| `REDIS_TIMEOUT_SECONDS` | Integer | `5` | Redis operation timeout |

### GPU and Model Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | String | `0` | GPU devices to use (comma-separated) |
| `GPU_MEMORY_FRACTION` | Float | `0.8` | Fraction of GPU memory to use |
| `MODEL_CACHE_SIZE` | Integer | `2048` | Model cache size in MB |
| `ENABLE_MODEL_COMPILATION` | Boolean | `true` | Enable PyTorch model compilation |
| `MODEL_PRECISION` | String | `fp32` | Model precision (fp16, fp32) |
| `BATCH_SIZE` | Integer | `1` | Default batch size for inference |
| `ENABLE_TENSORRT` | Boolean | `false` | Enable TensorRT optimization |

### Storage Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `R2_ENDPOINT_URL` | String | **Required** | Cloudflare R2 endpoint URL |
| `R2_ACCESS_KEY_ID` | String | **Required** | R2 access key |
| `R2_SECRET_ACCESS_KEY` | String | **Required** | R2 secret key |
| `R2_BUCKET_NAME` | String | **Required** | R2 bucket name |
| `R2_REGION` | String | `auto` | R2 region |
| `BACKUP_ENABLED` | Boolean | `true` | Enable automatic backups |
| `BACKUP_RETENTION_DAYS` | Integer | `30` | Backup retention period |

### RunPod Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RUNPOD_API_KEY` | String | **Required** | RunPod API key |
| `RUNPOD_DEFAULT_GPU` | String | `H100` | Default GPU type for deployments |
| `SERVERLESS_MIN_INSTANCES` | Integer | `0` | Minimum serverless instances |
| `SERVERLESS_MAX_INSTANCES` | Integer | `10` | Maximum serverless instances |
| `IDLE_TIMEOUT_SECONDS` | Integer | `300` | Idle timeout before auto-shutdown |
| `ENABLE_AUTO_SHUTDOWN` | Boolean | `true` | Enable automatic resource shutdown |

### API and Network Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `API_HOST` | String | `0.0.0.0` | API server host |
| `API_PORT` | Integer | `8000` | API server port |
| `CORS_ORIGINS` | String | `*` | Allowed CORS origins (comma-separated) |
| `ALLOWED_HOSTS` | String | `*` | Allowed host headers |
| `RATE_LIMIT_CALLS` | Integer | `100` | Rate limit: calls per period |
| `RATE_LIMIT_PERIOD` | Integer | `60` | Rate limit period in seconds |
| `MAX_UPLOAD_SIZE` | Integer | `10485760` | Maximum upload size in bytes (10MB) |

### Security Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_HTTPS` | Boolean | `false` | Require HTTPS connections |
| `SSL_CERT_PATH` | String | `null` | Path to SSL certificate |
| `SSL_KEY_PATH` | String | `null` | Path to SSL private key |
| `ENABLE_RATE_LIMITING` | Boolean | `true` | Enable API rate limiting |
| `ENABLE_AUTH_MIDDLEWARE` | Boolean | `true` | Enable authentication middleware |
| `TRUSTED_PROXIES` | String | `null` | Trusted proxy IP addresses |

### Monitoring Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_METRICS` | Boolean | `true` | Enable Prometheus metrics |
| `METRICS_PORT` | Integer | `8000` | Metrics endpoint port |
| `ENABLE_TRACING` | Boolean | `false` | Enable distributed tracing |
| `JAEGER_ENDPOINT` | String | `null` | Jaeger tracing endpoint |
| `LOG_FORMAT` | String | `json` | Log format (json, text) |
| `ENABLE_REQUEST_LOGGING` | Boolean | `true` | Log all HTTP requests |

## Configuration Files

### 1. Application Configuration

#### Main Configuration Schema

```python
# src/control/api/config.py
from pydantic import BaseSettings, Field, validator
from typing import List, Optional

class Settings(BaseSettings):
    """Application configuration with validation."""
    
    # Core settings
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    secret_key: str = Field(..., env="SECRET_KEY")
    
    # Database settings
    mongodb_uri: str = Field(..., env="MONGODB_URI")
    mongodb_database: str = Field(default="h200", env="MONGODB_DATABASE")
    redis_url: str = Field(..., env="REDIS_URL")
    
    # GPU settings
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    gpu_memory_fraction: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")
    model_cache_size: int = Field(default=2048, env="MODEL_CACHE_SIZE")
    
    # API settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Security settings
    enable_https: bool = Field(default=False, env="ENABLE_HTTPS")
    rate_limit_calls: int = Field(default=100, env="RATE_LIMIT_CALLS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()
    
    @validator('gpu_memory_fraction')
    def validate_gpu_memory_fraction(cls, v):
        """Validate GPU memory fraction."""
        if not 0.1 <= v <= 1.0:
            raise ValueError('GPU memory fraction must be between 0.1 and 1.0')
        return v
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
```

### 2. Model Configuration

#### Model Management Settings

```yaml
# configs/models.yml
models:
  detection:
    yolo_v8n:
      path: "models/yolo_v8n.pt"
      input_size: [640, 640]
      confidence_threshold: 0.7
      nms_threshold: 0.5
      max_detections: 100
      warmup_iterations: 3
      
    yolo_v8s:
      path: "models/yolo_v8s.pt"  
      input_size: [640, 640]
      confidence_threshold: 0.8
      nms_threshold: 0.5
      max_detections: 100
      warmup_iterations: 3
      
  classification:
    clip_vit_b32:
      path: "models/clip_vit_b32.pt"
      input_size: [224, 224]
      embedding_dim: 512
      batch_size: 32
      
  positioning:
    custom_cnn:
      path: "models/positioning_cnn.pt"
      input_size: [256, 256]
      output_classes: 10
      confidence_threshold: 0.6

# Model loading configuration
loading:
  preload_models: ["yolo_v8n", "clip_vit_b32"]
  lazy_loading: true
  memory_management:
    max_models_in_memory: 3
    lru_eviction: true
    memory_limit_mb: 8192
    
# Optimization settings
optimization:
  enable_compilation: true
  compilation_mode: "reduce-overhead"
  enable_quantization: false
  quantization_dtype: "int8"
  enable_tensorrt: false
  tensorrt_precision: "fp16"
```

### 3. Deployment Configuration

#### RunPod Deployment Templates

```yaml
# configs/deployment/serverless.yml
deployment:
  type: serverless
  name: h200-serverless
  
image:
  name: tekfly/h200:serverless-latest
  registry_auth:
    username: ${DOCKER_USERNAME}
    password: ${DOCKER_PASSWORD}

resources:
  gpu_type: H100
  gpu_count: 1
  vcpus: 8
  memory_gb: 32
  container_disk_gb: 20
  volume_gb: 50

scaling:
  min_instances: 0
  max_instances: 10
  idle_timeout_seconds: 300
  scale_up_threshold: 0.8
  scale_down_threshold: 0.2
  cold_start_boost: true

environment:
  ENVIRONMENT: production
  LOG_LEVEL: INFO
  ENABLE_METRICS: true
  MODEL_CACHE_SIZE: 4096
  
secrets:
  - name: MONGODB_URI
    source: google_secret_manager
    key: mongodb-prod-uri
    
  - name: REDIS_URL
    source: google_secret_manager  
    key: redis-prod-url
    
  - name: SECRET_KEY
    source: google_secret_manager
    key: jwt-secret-key

networking:
  ports:
    - internal: 8000
      external: 8000
      protocol: http
      
health_check:
  path: "/api/health"
  initial_delay_seconds: 30
  period_seconds: 30
  timeout_seconds: 10
  failure_threshold: 3
```

```yaml
# configs/deployment/timed.yml  
deployment:
  type: timed
  name: h200-timed
  
image:
  name: tekfly/h200:timed-latest
  
resources:
  gpu_type: H200
  gpu_count: 1
  vcpus: 16
  memory_gb: 64
  container_disk_gb: 100
  volume_gb: 200

instance:
  bid_per_gpu: 2.50
  country_code: "US"
  min_download_mbps: 100
  min_upload_mbps: 100
  
runtime:
  max_runtime_hours: 24
  auto_terminate: false
  restart_policy: always

ports:
  - internal: 8000
    external: 8000
    type: http
  - internal: 8888  
    external: 8888
    type: http  # Jupyter
  - internal: 6006
    external: 6006
    type: http  # TensorBoard
```

### 4. Monitoring Configuration

#### Grafana Dashboard Configuration

```yaml
# configs/monitoring/dashboards/h200-overview.yml
dashboard:
  title: "H200 System Overview"
  uid: "h200-overview"
  tags: ["h200", "overview", "production"]
  
  time:
    from: "now-1h"
    to: "now"
  refresh: "5s"
  
  variables:
    - name: "instance"
      type: "query"
      query: "label_values(up{job='h200-api'}, instance)"
      multi: true
      includeAll: true
      
    - name: "interval"
      type: "interval"
      options: ["1m", "5m", "15m", "1h"]
      current: "5m"

  panels:
    - title: "System Health"
      type: "singlestat"
      span: 3
      targets:
        - expr: "up{job='h200-api'}"
          legendFormat: "API Status"
      thresholds: [0.5, 1]
      colors: ["red", "yellow", "green"]
      
    - title: "Request Rate"  
      type: "graph"
      span: 6
      targets:
        - expr: "rate(http_requests_total[$interval])"
          legendFormat: "{{method}} {{endpoint}}"
      yAxes:
        - label: "requests/sec"
          min: 0
          
    - title: "Response Time"
      type: "graph" 
      span: 6
      targets:
        - expr: "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[$interval]))"
          legendFormat: "P50"
        - expr: "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[$interval]))"
          legendFormat: "P95"
        - expr: "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[$interval]))"
          legendFormat: "P99"
      yAxes:
        - label: "seconds"
          min: 0
          max: 2
```

## Performance Configuration

### 1. GPU Performance Tuning

```python
# GPU optimization configuration
GPU_CONFIG = {
    # Memory management
    'memory_fraction': 0.8,
    'allow_growth': True,
    'memory_limit_mb': None,  # Auto-detect
    
    # Performance optimization
    'enable_cudnn': True,
    'cudnn_benchmark': True,
    'cudnn_deterministic': False,
    
    # Model optimization
    'compile_models': True,
    'compilation_mode': 'reduce-overhead',
    'enable_tensorrt': False,
    'tensorrt_precision': 'fp16',
    
    # Batch processing
    'max_batch_size': 8,
    'adaptive_batching': True,
    'batch_timeout_ms': 100,
    
    # Memory cleanup
    'auto_clear_cache': True,
    'cache_clear_threshold': 0.9,
    'garbage_collection_frequency': 100  # Every 100 operations
}
```

### 2. Cache Configuration

```python
# Multi-level cache configuration
CACHE_CONFIG = {
    'l1_memory': {
        'enabled': True,
        'max_size_mb': 1024,
        'ttl_seconds': 300,
        'eviction_policy': 'lru'
    },
    
    'l2_redis': {
        'enabled': True,
        'max_size_mb': 4096,
        'ttl_seconds': 1800,
        'compression': True,
        'connection_pool_size': 20
    },
    
    'l3_storage': {
        'enabled': True,
        'backend': 'r2',
        'ttl_seconds': 86400,
        'compression': True,
        'async_write': True
    },
    
    'strategies': {
        'models': {
            'levels': ['l1', 'l2'],
            'ttl_override': 3600,
            'preload': True
        },
        'analysis_results': {
            'levels': ['l1', 'l2', 'l3'],
            'ttl_override': None,
            'preload': False
        },
        'user_data': {
            'levels': ['l1', 'l2'],
            'ttl_override': 1800,
            'preload': False
        }
    }
}
```

### 3. Scaling Configuration

```yaml
# configs/scaling.yml
scaling:
  serverless:
    # Instance scaling
    min_instances: 0
    max_instances: 20
    target_utilization: 70
    scale_up_cooldown: 60  # seconds
    scale_down_cooldown: 300  # seconds
    
    # Request-based scaling
    requests_per_instance: 10
    queue_length_threshold: 5
    response_time_threshold: 1.0  # seconds
    
    # Cost optimization
    enable_spot_instances: true
    max_spot_price: 2.0
    
  timed:
    # Instance management
    default_runtime_hours: 1
    max_runtime_hours: 24
    auto_terminate: true
    
    # Resource allocation
    resource_profiles:
      small:
        gpu_type: H100
        vcpus: 8
        memory_gb: 32
      medium:
        gpu_type: H100
        vcpus: 16
        memory_gb: 64
      large:
        gpu_type: H200
        vcpus: 32
        memory_gb: 128

  auto_shutdown:
    enabled: true
    idle_threshold_minutes: 10
    check_interval_seconds: 60
    grace_period_seconds: 30
    protected_hours: []  # Hours to never auto-shutdown
    
  load_balancing:
    strategy: round_robin  # round_robin, least_connections, weighted
    health_check_interval: 30
    unhealthy_threshold: 3
    recovery_check_interval: 60
```

## Rules Engine Configuration

### 1. Rule Processing Configuration

```yaml
# configs/rules.yml
rules_engine:
  # Processing configuration
  max_rules_per_evaluation: 50
  evaluation_timeout_seconds: 5
  parallel_evaluation: true
  max_parallel_rules: 10
  
  # Natural language processing
  nlp:
    model: "gpt-3.5-turbo"
    confidence_threshold: 0.7
    max_tokens: 500
    temperature: 0.2
    
  # Rule compilation
  compilation:
    enable_caching: true
    cache_compiled_rules: true
    recompile_on_change: true
    optimization_level: 2
    
  # Performance
  metrics:
    track_execution_time: true
    track_accuracy: true
    track_user_feedback: true
    
  # Storage
  storage:
    backend: mongodb
    collection_name: "rules"
    enable_versioning: true
    max_versions_per_rule: 10
```

### 2. Rule Validation Configuration

```python
# Rule validation configuration
RULE_VALIDATION_CONFIG = {
    'conditions': {
        'max_conditions_per_rule': 10,
        'max_nesting_depth': 3,
        'allowed_operators': [
            'equals', 'not_equals',
            'greater_than', 'greater_than_or_equal',
            'less_than', 'less_than_or_equal',
            'contains', 'not_contains',
            'in_range', 'outside_range'
        ],
        'allowed_fields': [
            'distance_from_center',
            'distance_to_edge', 
            'distance_between_mugs',
            'mug_count',
            'surface_coverage',
            'handle_orientation'
        ]
    },
    
    'actions': {
        'max_actions_per_rule': 5,
        'allowed_action_types': [
            'alert', 'log', 'notification',
            'adjustment', 'webhook'
        ],
        'rate_limits': {
            'notification': {'calls': 10, 'period': 3600},
            'webhook': {'calls': 100, 'period': 3600}
        }
    },
    
    'validation': {
        'require_test_cases': True,
        'min_test_cases': 3,
        'accuracy_threshold': 0.8,
        'performance_limit_ms': 100
    }
}
```

## Development Configuration

### 1. Development Environment

```yaml
# configs/development.yml
development:
  # Hot reload
  enable_hot_reload: true
  watch_directories: ["src/", "tests/"]
  ignore_patterns: ["**/__pycache__/**", "**/*.pyc"]
  
  # Debug features
  enable_debug_endpoints: true
  enable_profiling: true
  detailed_error_messages: true
  
  # Testing
  test_database: h200_test
  mock_external_services: true
  fake_gpu_for_testing: true
  
  # Development tools
  jupyter:
    enabled: true
    port: 8888
    password: "development"
    
  tensorboard:
    enabled: true
    port: 6006
    log_dir: "logs/tensorboard"
    
  # Local services
  use_local_redis: true
  use_local_mongodb: true
  use_local_storage: true  # MinIO instead of R2
```

### 2. Testing Configuration

```python
# Test configuration
TEST_CONFIG = {
    'databases': {
        'mongodb': {
            'uri': 'mongodb://localhost:27017/h200_test',
            'drop_on_start': True,
            'seed_data': True
        },
        'redis': {
            'url': 'redis://localhost:6379/1',  # Different DB for tests
            'flush_on_start': True
        }
    },
    
    'fixtures': {
        'image_data_dir': 'tests/fixtures/images/',
        'sample_rules_file': 'tests/fixtures/sample_rules.json',
        'user_data_file': 'tests/fixtures/users.json'
    },
    
    'performance': {
        'benchmark_iterations': 100,
        'timeout_seconds': 30,
        'performance_regression_threshold': 1.2,  # 20% slower = fail
        'memory_leak_threshold_mb': 100
    },
    
    'mocking': {
        'mock_gpu': True,
        'mock_external_apis': True,
        'mock_file_uploads': True,
        'deterministic_results': True
    }
}
```

## Production Configuration

### 1. Production Environment

```yaml
# configs/production.yml
production:
  # Security
  security:
    require_https: true
    enable_rate_limiting: true
    enable_ip_whitelist: false
    enable_audit_logging: true
    session_timeout_hours: 8
    
  # Performance  
  performance:
    enable_compression: true
    enable_caching: true
    connection_pool_size: 50
    worker_processes: 4
    
  # Reliability
  reliability:
    enable_health_checks: true
    health_check_interval: 30
    enable_circuit_breaker: true
    retry_attempts: 3
    timeout_seconds: 30
    
  # Monitoring
  monitoring:
    metrics_enabled: true
    tracing_enabled: true
    log_level: WARNING
    enable_profiling: false
    
  # Resource management
  resources:
    cpu_limit: "8000m"
    memory_limit: "32Gi"
    gpu_memory_limit: "80Gi"
    disk_limit: "100Gi"
```

### 2. High Availability Configuration

```yaml
# configs/ha.yml
high_availability:
  # Multi-region setup
  regions:
    primary:
      name: "us-east"
      weight: 70
      endpoints:
        - "https://us-east-1.h200.tekfly.co.uk"
        - "https://us-east-2.h200.tekfly.co.uk"
        
    secondary:
      name: "eu-west"
      weight: 30
      endpoints:
        - "https://eu-west-1.h200.tekfly.co.uk"
        
  # Failover configuration
  failover:
    health_check_interval: 10
    failure_threshold: 3
    recovery_threshold: 2
    automatic_failover: true
    failback_delay_minutes: 5
    
  # Load balancing
  load_balancing:
    algorithm: "least_connections"
    sticky_sessions: false
    health_check_path: "/api/health"
    
  # Data replication
  replication:
    mongodb_read_preference: "primaryPreferred"
    redis_replication_mode: "async"
    backup_frequency_hours: 6
    cross_region_backup: true
```

## Configuration Validation

### 1. Startup Validation

```python
class ConfigurationValidator:
    def __init__(self, config: Settings):
        self.config = config
        
    async def validate_all(self) -> Dict[str, Any]:
        """Validate all configuration settings."""
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }
        
        # Run all validation checks
        checks = [
            ('database', self.validate_database_config),
            ('gpu', self.validate_gpu_config),
            ('storage', self.validate_storage_config),
            ('security', self.validate_security_config),
            ('performance', self.validate_performance_config)
        ]
        
        for check_name, check_func in checks:
            try:
                result = await check_func()
                validation_results['checks'][check_name] = result
                
                if not result['valid']:
                    validation_results['valid'] = False
                    validation_results['errors'].extend(result.get('errors', []))
                
                validation_results['warnings'].extend(result.get('warnings', []))
                
            except Exception as e:
                validation_results['valid'] = False
                validation_results['errors'].append(f"{check_name} validation failed: {e}")
        
        return validation_results
    
    async def validate_database_config(self) -> Dict:
        """Validate database configuration."""
        errors = []
        warnings = []
        
        # Test MongoDB connection
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            client = AsyncIOMotorClient(self.config.mongodb_uri)
            await client.admin.command('ping')
            await client.close()
        except Exception as e:
            errors.append(f"MongoDB connection failed: {e}")
        
        # Test Redis connection
        try:
            import aioredis
            redis = aioredis.from_url(self.config.redis_url)
            await redis.ping()
            await redis.close()
        except Exception as e:
            errors.append(f"Redis connection failed: {e}")
        
        # Check connection pool sizes
        if self.config.mongodb_pool_size < 5:
            warnings.append("MongoDB pool size very small, may impact performance")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    async def validate_gpu_config(self) -> Dict:
        """Validate GPU configuration."""
        errors = []
        warnings = []
        
        # Check CUDA availability
        import torch
        if not torch.cuda.is_available():
            if self.config.environment == 'production':
                errors.append("CUDA not available in production environment")
            else:
                warnings.append("CUDA not available, falling back to CPU")
        
        # Check GPU memory configuration
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            requested_memory = total_memory * self.config.gpu_memory_fraction
            
            if requested_memory < 2 * 1024**3:  # < 2GB
                warnings.append("Low GPU memory allocation may impact performance")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
```

This configuration reference provides comprehensive documentation for all system settings, enabling users and administrators to properly configure the H200 System for their specific requirements and environments.