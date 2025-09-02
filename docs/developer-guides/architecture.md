# System Architecture

Comprehensive overview of the H200 Intelligent Mug Positioning System architecture, components, and design decisions.

## Architecture Overview

The H200 System is built on a modern, cloud-native microservices architecture designed for scalability, reliability, and performance optimization with GPU acceleration.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Layer                            │
├─────────────────────────────────────────────────────────────┤
│ Web Dashboard │ Mobile App │ API Clients │ CLI Tools       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway                              │
├─────────────────────────────────────────────────────────────┤
│ FastAPI │ Auth │ Rate Limiting │ Load Balancing           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 Application Layer                           │
├─────────────────────────────────────────────────────────────┤
│ Analysis Service │ Rules Engine │ Control Plane            │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   GPU Processing                            │
├─────────────────────────────────────────────────────────────┤
│ Model Manager │ H200 GPU │ Cache Layer │ Pipeline          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Data Layer                                │
├─────────────────────────────────────────────────────────────┤
│ MongoDB Atlas │ Redis Cache │ Cloudflare R2 │ Secrets      │
└─────────────────────────────────────────────────────────────┘
```

### Core Principles

**1. Microservices Design**
- Loosely coupled, independently deployable services
- Clear service boundaries and responsibilities
- Event-driven communication patterns

**2. GPU-First Architecture**
- Optimized for H200 GPU processing
- Efficient memory management and model loading
- Smart caching strategies for performance

**3. Cloud-Native Patterns**
- Container-based deployment with Docker
- Horizontal scaling with Kubernetes compatibility
- Infrastructure as Code principles

**4. Performance Optimization**
- Sub-second response times for image analysis
- Intelligent caching at multiple layers
- Asynchronous processing patterns

## System Components

### 1. API Gateway (FastAPI)

**Location**: `src/control/api/`

**Responsibilities:**
- HTTP/WebSocket endpoint management
- Authentication and authorization
- Rate limiting and throttling
- Request routing and load balancing
- Error handling and logging

**Key Features:**
```python
# Main application configuration
app = FastAPI(
    title="H200 Intelligent Mug Positioning System",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Middleware stack
app.add_middleware(CORSMiddleware)
app.add_middleware(RateLimitMiddleware) 
app.add_middleware(LoggingMiddleware)
app.add_middleware(AuthMiddleware)
```

**Endpoints:**
- `/api/v1/analyze/*` - Image analysis operations
- `/api/v1/rules/*` - Rules management
- `/api/v1/dashboard/*` - System monitoring
- `/api/v1/servers/*` - Server control
- `/ws/control-plane` - Real-time WebSocket

### 2. Analysis Service

**Location**: `src/core/`

**Responsibilities:**
- Image preprocessing and validation
- ML model orchestration
- Detection and positioning analysis
- Result formatting and caching

**Architecture:**
```python
class H200ImageAnalyzer:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.positioning_engine = PositioningEngine()
        self.cache = CacheManager()
    
    async def analyze_image(self, image, confidence_threshold=0.7):
        # 1. Preprocess image
        processed_image = await self.preprocess(image)
        
        # 2. Run detection models
        detections = await self.detect_objects(processed_image)
        
        # 3. Analyze positioning
        positioning = await self.analyze_positioning(detections)
        
        # 4. Cache results
        await self.cache.store_result(result)
        
        return AnalysisResult(detections, positioning)
```

**Processing Pipeline:**
```
Raw Image → Validation → Preprocessing → GPU Analysis →
Positioning Analysis → Rule Evaluation → Result Caching → Response
```

### 3. Model Management System

**Location**: `src/core/models/`

**Responsibilities:**
- ML model lifecycle management
- GPU memory optimization
- Model versioning and deployment
- Performance monitoring

**Model Architecture:**
```python
class ModelManager:
    def __init__(self):
        self.models = {}
        self.gpu_memory_manager = GPUMemoryManager()
        self.model_cache = ModelCache()
    
    async def load_model(self, model_name: str, model_version: str):
        """Load model into GPU memory with optimization"""
        if model_name in self.models:
            return self.models[model_name]
        
        # Optimize GPU memory
        await self.gpu_memory_manager.prepare_memory(model_name)
        
        # Load and cache model
        model = await self.load_from_storage(model_name, model_version)
        self.models[model_name] = model
        
        return model
```

**Supported Models:**
- **YOLO v8n**: Object detection (mugs, coasters, surfaces)
- **CLIP**: Semantic understanding and attributes
- **Custom CNN**: Positioning quality assessment
- **Depth Estimation**: 3D spatial understanding

### 4. Rules Engine

**Location**: `src/core/rules/`

**Responsibilities:**
- Rule parsing and compilation
- Natural language processing
- Rule evaluation and execution
- Performance optimization

**Engine Architecture:**
```python
class RuleEngine:
    def __init__(self, db: Database):
        self.db = db
        self.parser = NaturalLanguageParser()
        self.executor = RuleExecutor()
        self.compiled_rules = {}
    
    async def evaluate_rules(self, analysis_data, context=None):
        """Evaluate all applicable rules against analysis data"""
        applicable_rules = await self.get_applicable_rules(context)
        
        results = []
        for rule in applicable_rules:
            result = await self.executor.evaluate(rule, analysis_data)
            results.append(result)
        
        return RuleEvaluationResults(results)
```

**Rule Processing Flow:**
```
Natural Language → Parser → AST → Compiler → Executable Rule →
Evaluation Engine → Action Executor → Result Storage
```

### 5. Control Plane

**Location**: `src/control/manager/`

**Responsibilities:**
- Server lifecycle management
- Resource monitoring and optimization
- Auto-scaling and shutdown
- Cost tracking and optimization

**Orchestrator Architecture:**
```python
class ControlPlaneOrchestrator:
    def __init__(self):
        self.server_manager = ServerManager()
        self.metrics_collector = MetricsCollector()
        self.auto_shutdown = AutoShutdownManager()
        self.notifier = NotificationManager()
    
    async def start_server(self, server_type: ServerType):
        """Start a server with full lifecycle management"""
        server = await self.server_manager.create_server(server_type)
        
        # Start monitoring
        await self.metrics_collector.start_monitoring(server.id)
        
        # Configure auto-shutdown
        self.auto_shutdown.register_server(server.id)
        
        return server
```

### 6. Data Layer

#### MongoDB Atlas (Primary Database)

**Collections:**
```javascript
// Analysis results
analysis_results: {
  _id: ObjectId,
  request_id: String,
  user_id: String,
  timestamp: Date,
  image_metadata: Object,
  detections: Array,
  positioning: Object,
  rules_applied: Array,
  processing_time_ms: Number
}

// Rules definitions
rules: {
  _id: ObjectId,
  name: String,
  description: String,
  type: String,
  conditions: Array,
  actions: Array,
  enabled: Boolean,
  metadata: Object
}

// User activity logs
activity_logs: {
  _id: ObjectId,
  user_id: String,
  action: String,
  timestamp: Date,
  details: Object,
  ip_address: String
}
```

#### Redis Cache (Performance Layer)

**Caching Strategy:**
```python
# Multi-level caching
CACHE_LEVELS = {
    'models': {
        'ttl': 3600,  # 1 hour
        'max_size': '8GB',  # GPU memory limit
        'strategy': 'LRU'
    },
    'analysis_results': {
        'ttl': 300,   # 5 minutes
        'max_size': '2GB',
        'strategy': 'LRU'
    },
    'rules': {
        'ttl': 1800,  # 30 minutes  
        'max_size': '100MB',
        'strategy': 'LRU'
    }
}
```

**Cache Patterns:**
- **Write-Through**: Critical data (rules, configurations)
- **Write-Behind**: Performance data (metrics, logs)
- **Cache-Aside**: Analysis results and user data

#### Cloudflare R2 (Object Storage)

**Storage Organization:**
```
h200-storage/
├── models/
│   ├── yolo/v8n/
│   ├── clip/vit-b-32/
│   └── custom/positioning/
├── analysis-results/
│   ├── {year}/{month}/{day}/
│   └── thumbnails/
├── user-uploads/
│   └── {user_id}/{session_id}/
└── backups/
    ├── mongodb/
    └── config/
```

## Deployment Architecture

### Container Strategy

#### Multi-Stage Docker Builds

**Base Image:**
```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS base

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    nvidia-ml-py3 \
    && rm -rf /var/lib/apt/lists/*

# Python environment
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
```

**Serverless Image:**
```dockerfile
FROM base AS serverless

# Optimized for fast cold starts
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Pre-compile models for faster loading
COPY preload_models.py .
RUN python preload_models.py --optimize-for-inference

EXPOSE 8000
CMD ["uvicorn", "src.control.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Timed Instance Image:**
```dockerfile
FROM base AS timed

# Full feature set with all dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development and debugging tools
RUN pip install --no-cache-dir \
    jupyter \
    tensorboard \
    wandb

EXPOSE 8000 8888
CMD ["uvicorn", "src.control.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### RunPod Deployment

#### Serverless Configuration

```python
SERVERLESS_CONFIG = {
    "name": "h200-serverless",
    "image_name": "tekfly/h200:serverless-latest",
    "gpu_type": "H100",
    "gpu_count": 1,
    "container_disk_size": 20,  # GB
    "volume_size": 50,  # GB
    "environment_variables": {
        "MODEL_CACHE_SIZE": "8192",
        "REDIS_URL": "${REDIS_URL}",
        "MONGODB_URI": "${MONGODB_URI}"
    },
    "scaling": {
        "min_instances": 0,
        "max_instances": 3,
        "idle_timeout": 300,  # 5 minutes
        "scale_up_threshold": 0.8,
        "scale_down_threshold": 0.2
    }
}
```

#### Timed Instance Configuration

```python
TIMED_CONFIG = {
    "name": "h200-timed",
    "image_name": "tekfly/h200:timed-latest", 
    "gpu_type": "H200",
    "gpu_count": 1,
    "vcpus": 8,
    "memory": 32,  # GB
    "container_disk_size": 50,  # GB
    "volume_size": 100,  # GB
    "network_storage": {
        "size": 200,  # GB
        "type": "network"
    }
}
```

## Performance Architecture

### GPU Optimization

#### Memory Management

```python
class GPUMemoryManager:
    def __init__(self):
        self.allocated_memory = {}
        self.max_memory = self.get_gpu_memory_limit()
        self.fragmentation_threshold = 0.2
    
    async def allocate_model_memory(self, model_name: str, size_mb: int):
        """Smart GPU memory allocation with defragmentation"""
        if self.get_available_memory() < size_mb:
            await self.defragment_memory()
        
        if self.get_available_memory() < size_mb:
            # Free least recently used models
            await self.free_lru_models(size_mb)
        
        # Allocate memory
        memory_address = torch.cuda.memory.allocate(size_mb * 1024 * 1024)
        self.allocated_memory[model_name] = {
            'address': memory_address,
            'size': size_mb,
            'last_access': datetime.utcnow()
        }
        
        return memory_address
```

#### Model Loading Optimization

```python
class OptimizedModelLoader:
    def __init__(self):
        self.compiled_models = {}
        self.quantized_models = {}
    
    async def load_with_optimization(self, model_path: str):
        """Load model with TensorRT optimization"""
        # 1. Try loading compiled model
        if model_path in self.compiled_models:
            return self.compiled_models[model_path]
        
        # 2. Load and optimize model
        model = torch.load(model_path)
        
        # 3. Apply quantization for speed
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # 4. Compile for inference
        compiled_model = torch.compile(
            quantized_model, 
            mode="reduce-overhead"
        )
        
        # 5. Cache optimized model
        self.compiled_models[model_path] = compiled_model
        
        return compiled_model
```

### Caching Architecture

#### Multi-Level Cache Strategy

```python
class CacheManager:
    def __init__(self):
        self.l1_cache = {}  # In-memory (fastest)
        self.l2_cache = RedisClient()  # Network cache (fast)
        self.l3_cache = R2Storage()  # Object storage (slower)
    
    async def get(self, key: str):
        """Smart cache retrieval with fallback"""
        # L1: In-memory cache
        if key in self.l1_cache:
            self.update_access_time(key, 'l1')
            return self.l1_cache[key]
        
        # L2: Redis cache
        value = await self.l2_cache.get(key)
        if value:
            # Promote to L1
            self.l1_cache[key] = value
            return value
        
        # L3: Object storage
        value = await self.l3_cache.get(key)
        if value:
            # Promote to L2 and L1
            await self.l2_cache.set(key, value, ttl=300)
            self.l1_cache[key] = value
            return value
        
        return None
```

### Request Processing Pipeline

#### Asynchronous Processing

```python
class AsyncProcessingPipeline:
    def __init__(self):
        self.preprocessing_queue = asyncio.Queue(maxsize=100)
        self.analysis_queue = asyncio.Queue(maxsize=50)
        self.postprocessing_queue = asyncio.Queue(maxsize=200)
    
    async def process_request(self, request: AnalysisRequest):
        """Async pipeline processing"""
        # Stage 1: Preprocessing
        preprocessed = await self.preprocess_stage(request)
        
        # Stage 2: GPU Analysis (with batching)
        analysis_result = await self.analysis_stage(preprocessed)
        
        # Stage 3: Postprocessing
        final_result = await self.postprocess_stage(analysis_result)
        
        return final_result
    
    async def batch_analysis(self, requests: List[AnalysisRequest]):
        """Batch multiple requests for GPU efficiency"""
        batch_size = min(len(requests), 8)  # Optimal batch size
        
        batches = [
            requests[i:i + batch_size] 
            for i in range(0, len(requests), batch_size)
        ]
        
        results = []
        for batch in batches:
            batch_result = await self.process_batch_gpu(batch)
            results.extend(batch_result)
        
        return results
```

## Security Architecture

### Authentication and Authorization

#### JWT-Based Authentication

```python
class JWTHandler:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = timedelta(hours=24)
    
    def generate_token(self, user_data: dict) -> str:
        """Generate JWT with user claims"""
        payload = {
            "user_id": user_data["id"],
            "username": user_data["username"],
            "permissions": user_data["permissions"],
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow(),
            "iss": "h200-system"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT"""
        try:
            payload = jwt.decode(
                token, self.secret_key, 
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
```

#### Role-Based Access Control

```python
class RBACManager:
    def __init__(self):
        self.permissions = {
            'viewer': ['read:analysis', 'read:dashboard'],
            'analyst': ['read:*', 'write:analysis', 'write:feedback'],
            'manager': ['read:*', 'write:*', 'delete:rules'],
            'admin': ['*']
        }
    
    def check_permission(self, user_role: str, action: str, resource: str) -> bool:
        """Check if user has permission for action"""
        user_permissions = self.permissions.get(user_role, [])
        
        # Check for wildcard permissions
        if '*' in user_permissions:
            return True
        
        # Check for specific permission
        required_permission = f"{action}:{resource}"
        if required_permission in user_permissions:
            return True
        
        # Check for wildcard resource permissions
        wildcard_permission = f"{action}:*"
        if wildcard_permission in user_permissions:
            return True
        
        return False
```

### Data Encryption

#### At-Rest Encryption

```python
class EncryptionManager:
    def __init__(self, encryption_key: bytes):
        self.fernet = Fernet(encryption_key)
    
    def encrypt_sensitive_data(self, data: dict) -> str:
        """Encrypt sensitive data before storage"""
        json_data = json.dumps(data).encode()
        encrypted_data = self.fernet.encrypt(json_data)
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> dict:
        """Decrypt sensitive data after retrieval"""
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted_data = self.fernet.decrypt(encrypted_bytes)
        return json.loads(decrypted_data.decode())
```

#### In-Transit Encryption

All communications use TLS 1.3:
- Client ↔ API Gateway: HTTPS
- API Gateway ↔ Services: mTLS
- Service ↔ Database: TLS
- Cache connections: TLS/SSL

## Monitoring and Observability

### Metrics Collection

#### Prometheus Integration

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# GPU metrics  
GPU_UTILIZATION = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage'
)

GPU_MEMORY = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory usage in bytes'
)

# Business metrics
ANALYSIS_COUNT = Counter(
    'analysis_requests_total',
    'Total analysis requests',
    ['user_id', 'status']
)
```

#### Custom Metrics Collection

```python
class MetricsCollector:
    def __init__(self):
        self.metrics_buffer = []
        self.collection_interval = 5  # seconds
    
    async def collect_system_metrics(self):
        """Collect comprehensive system metrics"""
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory()._asdict(),
            'gpu_stats': self.get_gpu_stats(),
            'cache_stats': await self.get_cache_stats(),
            'request_stats': self.get_request_stats(),
        }
        
        self.metrics_buffer.append(metrics)
        
        # Flush buffer periodically
        if len(self.metrics_buffer) >= 100:
            await self.flush_metrics()
    
    def get_gpu_stats(self) -> dict:
        """Collect GPU-specific metrics"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            return {
                'memory_used': mem_info.used,
                'memory_total': mem_info.total,
                'memory_free': mem_info.free,
                'gpu_utilization': util.gpu,
                'memory_utilization': util.memory,
                'temperature': pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            }
        except Exception as e:
            logger.error(f"Failed to collect GPU stats: {e}")
            return {}
```

### Distributed Tracing

#### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Initialize tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Trace analysis requests
@tracer.start_as_current_span("analyze_image")
async def analyze_image(image_data: bytes):
    with tracer.start_as_current_span("preprocess_image") as span:
        span.set_attribute("image_size", len(image_data))
        processed_image = await preprocess_image(image_data)
    
    with tracer.start_as_current_span("gpu_inference") as span:
        span.set_attribute("model_type", "yolo")
        result = await run_inference(processed_image)
    
    return result
```

This architecture provides a solid foundation for scalable, performant, and maintainable image analysis with intelligent positioning capabilities. The design emphasizes GPU optimization, caching efficiency, and cloud-native deployment patterns.