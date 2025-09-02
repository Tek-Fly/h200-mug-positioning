# Performance Tuning Guide

Comprehensive guide to optimizing performance in the H200 Intelligent Mug Positioning System.

## Performance Overview

The H200 System is designed for high-performance image analysis with the following targets:

### Performance Targets

**Response Times:**
- **Cold Start**: 500ms - 2s (FlashBoot enabled)
- **Warm Start**: < 100ms  
- **Image Processing**: < 500ms for 1080p images
- **API Latency**: p95 < 200ms

**Throughput:**
- **Single Instance**: 100+ requests/minute
- **Batch Processing**: 500+ images/minute
- **Concurrent Users**: 50+ simultaneous users

**Resource Efficiency:**
- **GPU Utilization**: > 70% during processing
- **Cache Hit Rate**: > 85%
- **Memory Efficiency**: < 8GB GPU memory per instance

## GPU Performance Optimization

### 1. Model Optimization

#### Model Compilation and Quantization

```python
# Optimize models for inference speed
import torch
from torch.quantization import quantize_dynamic

class OptimizedModelManager:
    def __init__(self):
        self.compiled_models = {}
        self.quantized_models = {}
    
    async def load_optimized_model(self, model_path: str):
        """Load model with all optimizations applied."""
        
        # 1. Load base model
        model = torch.load(model_path, map_location='cuda')
        model.eval()
        
        # 2. Apply quantization for speed
        quantized_model = quantize_dynamic(
            model, 
            {torch.nn.Linear, torch.nn.Conv2d}, 
            dtype=torch.qint8
        )
        
        # 3. Compile for inference (PyTorch 2.0+)
        compiled_model = torch.compile(
            quantized_model,
            mode="reduce-overhead",  # Optimize for inference
            fullgraph=True
        )
        
        # 4. Warm up model
        dummy_input = torch.randn(1, 3, 640, 640).cuda()
        with torch.no_grad():
            _ = compiled_model(dummy_input)
        
        return compiled_model
```

#### TensorRT Integration

```python
# TensorRT optimization for maximum performance
import tensorrt as trt
import torch_tensorrt

class TensorRTOptimizer:
    def __init__(self):
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
    
    def optimize_model(self, model, input_shape):
        """Convert PyTorch model to TensorRT for maximum speed."""
        
        # Configure TensorRT
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(
                shape=input_shape,
                dtype=torch.float16  # Use half precision
            )],
            enabled_precisions={torch.float16},
            workspace_size=1 << 30,  # 1GB workspace
            max_batch_size=8,
            use_python_runtime=False  # Use C++ runtime for speed
        )
        
        return trt_model

# Performance comparison
def benchmark_models():
    """Compare model performance across optimizations."""
    models = {
        'baseline': load_baseline_model(),
        'quantized': load_quantized_model(), 
        'compiled': load_compiled_model(),
        'tensorrt': load_tensorrt_model()
    }
    
    results = {}
    test_input = torch.randn(1, 3, 640, 640).cuda()
    
    for name, model in models.items():
        times = []
        for _ in range(100):
            start = time.time()
            with torch.no_grad():
                _ = model(test_input)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        results[name] = {
            'avg_time_ms': statistics.mean(times) * 1000,
            'p95_time_ms': sorted(times)[95] * 1000,
            'throughput_fps': 1.0 / statistics.mean(times)
        }
    
    return results
```

### 2. Memory Management

#### GPU Memory Optimization

```python
class GPUMemoryOptimizer:
    def __init__(self):
        self.memory_pool = torch.cuda.memory.MemoryPool()
        self.allocation_strategy = 'defragment'
    
    @contextmanager
    def optimized_memory_context(self):
        """Context manager for optimized GPU memory usage."""
        # Clear cache before operation
        torch.cuda.empty_cache()
        
        # Set memory fraction if needed
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        try:
            yield
        finally:
            # Cleanup after operation
            torch.cuda.empty_cache()
    
    async def analyze_with_memory_optimization(self, image_data: bytes):
        """Analyze image with optimal memory usage."""
        with self.optimized_memory_context():
            # Enable memory efficient attention if available
            with torch.backends.cuda.sdp_kernel(enable_memory_efficient=True):
                result = await self.analyze_image(image_data)
        
        return result

# Memory monitoring and alerts
class MemoryMonitor:
    def __init__(self, warning_threshold: float = 0.8):
        self.warning_threshold = warning_threshold
        self.critical_threshold = 0.95
    
    def check_gpu_memory(self):
        """Monitor GPU memory usage."""
        if not torch.cuda.is_available():
            return
        
        memory_used = torch.cuda.memory_allocated()
        memory_total = torch.cuda.get_device_properties(0).total_memory
        usage_ratio = memory_used / memory_total
        
        if usage_ratio > self.critical_threshold:
            logger.critical(f"GPU memory critical: {usage_ratio:.1%}")
            # Force cleanup
            torch.cuda.empty_cache()
            gc.collect()
        elif usage_ratio > self.warning_threshold:
            logger.warning(f"GPU memory high: {usage_ratio:.1%}")
```

#### Memory Pool Management

```python
class AdvancedMemoryManager:
    def __init__(self):
        self.memory_pools = {
            'models': torch.cuda.memory.MemoryPool(),
            'inference': torch.cuda.memory.MemoryPool(),
            'cache': torch.cuda.memory.MemoryPool()
        }
    
    @contextmanager
    def pooled_memory(self, pool_name: str):
        """Use specific memory pool for operation."""
        old_pool = torch.cuda.memory.set_per_process_memory_pool(
            self.memory_pools[pool_name]
        )
        
        try:
            yield
        finally:
            torch.cuda.memory.set_per_process_memory_pool(old_pool)
    
    async def analyze_with_pooled_memory(self, image_data: bytes):
        """Analyze using dedicated inference memory pool."""
        with self.pooled_memory('inference'):
            return await self.analyze_image(image_data)
```

### 3. Caching Optimization

#### Multi-Level Cache Strategy

```python
class AdvancedCacheManager:
    def __init__(self):
        self.l1_cache = {}  # In-memory (fastest)
        self.l2_cache = RedisClient()  # Network (fast)
        self.l3_cache = R2Storage()  # Storage (slower)
        
        self.cache_policies = {
            'models': {'ttl': 3600, 'levels': ['l1', 'l2']},
            'analysis': {'ttl': 300, 'levels': ['l1', 'l2', 'l3']},
            'rules': {'ttl': 1800, 'levels': ['l1', 'l2']},
            'metrics': {'ttl': 60, 'levels': ['l1']}
        }
    
    async def get_with_promotion(self, key: str, data_type: str):
        """Get value with automatic cache level promotion."""
        policy = self.cache_policies[data_type]
        
        # Try each cache level
        for level in policy['levels']:
            cache = getattr(self, f'{level}_cache')
            value = await cache.get(key)
            
            if value is not None:
                # Promote to higher levels
                await self.promote_to_higher_levels(key, value, level, policy)
                return value
        
        return None
    
    async def set_with_distribution(self, key: str, value: Any, data_type: str):
        """Set value across appropriate cache levels."""
        policy = self.cache_policies[data_type]
        
        # Store in all configured levels
        tasks = []
        for level in policy['levels']:
            cache = getattr(self, f'{level}_cache')
            tasks.append(cache.set(key, value, ttl=policy['ttl']))
        
        await asyncio.gather(*tasks, return_exceptions=True)
```

#### Cache Warming Strategies

```python
class CacheWarmer:
    def __init__(self, cache_manager: AdvancedCacheManager):
        self.cache = cache_manager
        self.warming_schedule = {}
    
    async def warm_critical_data(self):
        """Pre-load critical data into cache."""
        warming_tasks = [
            self.warm_models(),
            self.warm_active_rules(),
            self.warm_user_sessions(),
            self.warm_system_config()
        ]
        
        results = await asyncio.gather(*warming_tasks, return_exceptions=True)
        
        # Log warming results
        for task_name, result in zip(
            ['models', 'rules', 'sessions', 'config'], 
            results
        ):
            if isinstance(result, Exception):
                logger.error(f"Failed to warm {task_name}: {result}")
            else:
                logger.info(f"Successfully warmed {task_name}: {result}")
    
    async def warm_models(self):
        """Pre-load ML models into cache."""
        model_configs = [
            {'name': 'yolo_v8n', 'priority': 'high'},
            {'name': 'clip_vit_b32', 'priority': 'medium'},
            {'name': 'positioning_cnn', 'priority': 'high'}
        ]
        
        # Load high priority models first
        high_priority = [m for m in model_configs if m['priority'] == 'high']
        medium_priority = [m for m in model_configs if m['priority'] == 'medium']
        
        # Sequential loading to avoid memory issues
        for model_config in high_priority + medium_priority:
            model_data = await load_model(model_config['name'])
            await self.cache.set_with_distribution(
                f"model:{model_config['name']}", 
                model_data, 
                'models'
            )
    
    async def intelligent_cache_warming(self):
        """Warm cache based on usage patterns."""
        # Analyze recent usage patterns
        usage_stats = await self.analyze_usage_patterns()
        
        # Prioritize most-used data
        for item in usage_stats['most_accessed']:
            if item['hit_rate'] < 0.8:  # Low hit rate items
                await self.preload_item(item['key'], item['type'])
```

## Database Performance

### 1. MongoDB Optimization

#### Index Optimization

```python
class DatabaseOptimizer:
    def __init__(self, db):
        self.db = db
    
    async def optimize_indexes(self):
        """Create optimal indexes for query patterns."""
        
        # Analysis results collection
        await self.db.analysis_results.create_index([
            ("user_id", 1),
            ("timestamp", -1)
        ], background=True)
        
        await self.db.analysis_results.create_index([
            ("timestamp", -1),
            ("processing_time_ms", 1)
        ], background=True)
        
        # Compound index for dashboard queries
        await self.db.analysis_results.create_index([
            ("timestamp", -1),
            ("positioning.confidence", -1),
            ("user_id", 1)
        ], background=True)
        
        # Rules collection
        await self.db.rules.create_index([
            ("enabled", 1),
            ("priority", -1),
            ("type", 1)
        ], background=True)
        
        # Activity logs with TTL
        await self.db.activity_logs.create_index(
            [("timestamp", 1)],
            expireAfterSeconds=7776000  # 90 days
        )
    
    async def analyze_query_performance(self):
        """Analyze slow queries and suggest optimizations."""
        
        # Enable profiling
        await self.db.set_profiling_level(2)  # Profile all operations
        
        # Run for some time, then analyze
        await asyncio.sleep(300)  # 5 minutes
        
        # Get slow operations
        slow_ops = await self.db.system.profile.find({
            "millis": {"$gt": 100}  # Queries taking > 100ms
        }).sort("millis", -1).limit(10).to_list(length=10)
        
        for op in slow_ops:
            logger.warning(f"Slow query: {op['command']} took {op['millis']}ms")
            
            # Suggest index if not available
            if 'planSummary' in op and 'COLLSCAN' in op['planSummary']:
                logger.warning(f"Collection scan detected for: {op['command']}")
```

#### Connection Pool Optimization

```python
from motor.motor_asyncio import AsyncIOMotorClient

class OptimizedDatabaseClient:
    def __init__(self, uri: str):
        self.client = AsyncIOMotorClient(
            uri,
            # Connection pool settings
            maxPoolSize=50,        # Max connections
            minPoolSize=10,        # Min connections  
            maxIdleTimeMS=30000,   # 30 seconds idle timeout
            waitQueueTimeoutMS=5000,  # 5 seconds wait timeout
            
            # Performance settings
            retryWrites=True,
            w='majority',
            readPreference='primaryPreferred',
            readConcernLevel='local',
            
            # Compression
            compressors='snappy,zlib',
            
            # Connection monitoring
            heartbeatFrequencyMS=10000,  # 10 seconds
            serverSelectionTimeoutMS=5000,  # 5 seconds
        )
    
    async def get_database(self, db_name: str):
        """Get database with optimized settings."""
        db = self.client[db_name]
        
        # Configure read/write concerns
        db = db.with_options(
            read_concern=ReadConcern('local'),
            write_concern=WriteConcern(w='majority', j=True),
            read_preference=ReadPreference.PRIMARY_PREFERRED
        )
        
        return db
```

### 2. Redis Performance Tuning

#### Configuration Optimization

```redis
# redis.conf optimizations for H200 System

# Memory management
maxmemory 8gb
maxmemory-policy allkeys-lru
maxmemory-samples 10

# Persistence (adjust based on requirements)
save 900 1      # Save after 900 sec if at least 1 key changed
save 300 10     # Save after 300 sec if at least 10 keys changed
save 60 10000   # Save after 60 sec if at least 10000 keys changed

# Network optimization
tcp-keepalive 300
timeout 300

# Performance tuning
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64

# Disable slow operations in production
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command KEYS ""
rename-command CONFIG ""
```

#### Advanced Caching Patterns

```python
class AdvancedCachePatterns:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.cache_stats = {}
    
    async def cache_with_tags(self, key: str, value: Any, tags: List[str], ttl: int):
        """Cache with tag-based invalidation."""
        # Store main value
        await self.redis.setex(key, ttl, json.dumps(value))
        
        # Store tags for invalidation
        for tag in tags:
            await self.redis.sadd(f"tag:{tag}", key)
            await self.redis.expire(f"tag:{tag}", ttl + 3600)  # Tags live longer
    
    async def invalidate_by_tag(self, tag: str):
        """Invalidate all cache entries with specific tag."""
        keys = await self.redis.smembers(f"tag:{tag}")
        
        if keys:
            # Delete all tagged keys
            await self.redis.delete(*keys)
            # Delete tag set
            await self.redis.delete(f"tag:{tag}")
    
    async def cache_with_refresh_ahead(
        self, 
        key: str, 
        value_factory: Callable,
        ttl: int,
        refresh_threshold: float = 0.8
    ):
        """Cache with proactive refresh before expiration."""
        
        # Check if key exists and when it expires
        ttl_remaining = await self.redis.ttl(key)
        
        if ttl_remaining > 0:
            # Check if we need to refresh
            refresh_time = ttl * refresh_threshold
            if ttl_remaining < (ttl - refresh_time):
                # Trigger background refresh
                asyncio.create_task(self.refresh_cache_entry(key, value_factory, ttl))
            
            # Return existing value
            cached_value = await self.redis.get(key)
            if cached_value:
                return json.loads(cached_value)
        
        # Generate new value
        new_value = await value_factory()
        await self.redis.setex(key, ttl, json.dumps(new_value))
        
        return new_value
```

## API Performance Optimization

### 1. Request Processing

#### Async Optimization

```python
import asyncio
from asyncio import Semaphore
from contextlib import asynccontextmanager

class OptimizedRequestProcessor:
    def __init__(self):
        # Limit concurrent GPU operations
        self.gpu_semaphore = Semaphore(2)  # Max 2 concurrent GPU operations
        self.db_semaphore = Semaphore(20)  # Max 20 concurrent DB operations
        self.request_queue = asyncio.Queue(maxsize=100)
    
    @asynccontextmanager
    async def gpu_context(self):
        """Context manager for GPU resource limiting."""
        async with self.gpu_semaphore:
            yield
    
    @asynccontextmanager  
    async def db_context(self):
        """Context manager for database resource limiting."""
        async with self.db_semaphore:
            yield
    
    async def process_analysis_request(self, request: AnalysisRequest):
        """Process analysis with resource management."""
        
        # Database operations with concurrency limit
        async with self.db_context():
            user_rules = await self.get_user_rules(request.user_id)
        
        # GPU operations with concurrency limit
        async with self.gpu_context():
            # Use memory-optimized processing
            analysis_result = await self.analyze_image_optimized(
                request.image_data,
                rules=user_rules
            )
        
        # Concurrent result storage (non-blocking)
        storage_tasks = [
            self.store_analysis_result(analysis_result),
            self.update_user_metrics(request.user_id),
            self.cache_result(analysis_result)
        ]
        
        # Don't wait for storage to complete
        asyncio.create_task(asyncio.gather(*storage_tasks))
        
        return analysis_result
```

#### Request Batching

```python
class RequestBatcher:
    def __init__(self, max_batch_size: int = 8, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.batch_timer = None
    
    async def add_request(self, request: AnalysisRequest) -> AnalysisResult:
        """Add request to batch queue."""
        future = asyncio.Future()
        self.pending_requests.append((request, future))
        
        # Process immediately if batch is full
        if len(self.pending_requests) >= self.max_batch_size:
            await self.process_batch()
        # Otherwise start timer for partial batch
        elif not self.batch_timer:
            self.batch_timer = asyncio.create_task(
                self.process_batch_after_delay()
            )
        
        return await future
    
    async def process_batch_after_delay(self):
        """Process partial batch after delay."""
        await asyncio.sleep(self.max_wait_time)
        await self.process_batch()
    
    async def process_batch(self):
        """Process accumulated requests as a batch."""
        if not self.pending_requests:
            return
        
        requests, futures = zip(*self.pending_requests)
        self.pending_requests.clear()
        self.batch_timer = None
        
        try:
            # Batch GPU processing
            batch_results = await self.gpu_batch_process(requests)
            
            # Return results to waiting requests
            for future, result in zip(futures, batch_results):
                future.set_result(result)
                
        except Exception as e:
            # Handle batch failure
            for future in futures:
                future.set_exception(e)
```

### 2. API Response Optimization

#### Response Compression

```python
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Use faster JSON serialization
class OptimizedResponse(ORJSONResponse):
    def render(self, content: Any) -> bytes:
        # Use orjson for faster serialization
        return orjson.dumps(
            content,
            option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY
        )

# Apply to all routes
@app.get("/api/v1/analyze", response_class=OptimizedResponse)
async def analyze_endpoint(...):
    return result
```

#### Streaming Responses

```python
from fastapi.responses import StreamingResponse

@app.post("/api/v1/analyze/stream")
async def streaming_analysis(
    images: List[UploadFile],
    user: str = Depends(get_current_user)
):
    """Stream analysis results as they complete."""
    
    async def generate_results():
        """Generate streaming analysis results."""
        for i, image in enumerate(images):
            try:
                # Process image
                result = await analyze_image(await image.read())
                
                # Stream result
                response_data = {
                    "index": i,
                    "status": "completed", 
                    "result": result.model_dump()
                }
                
                yield f"data: {json.dumps(response_data)}\n\n"
                
            except Exception as e:
                # Stream error
                error_data = {
                    "index": i,
                    "status": "error",
                    "error": str(e)
                }
                
                yield f"data: {json.dumps(error_data)}\n\n"
        
        # Signal completion
        yield f"data: {json.dumps({'status': 'complete'})}\n\n"
    
    return StreamingResponse(
        generate_results(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )
```

## Frontend Performance

### 1. Vue.js Optimization

#### Component Performance

```vue
<!-- Optimized Vue component -->
<template>
  <div class="analysis-results">
    <!-- Use v-show for frequent toggles -->
    <div v-show="showDetails" class="details-panel">
      <!-- Lazy load expensive components -->
      <Suspense>
        <template #default>
          <AnalysisChart :data="chartData" />
        </template>
        <template #fallback>
          <div>Loading chart...</div>
        </template>
      </Suspense>
    </div>
    
    <!-- Virtual scrolling for large lists -->
    <VirtualList
      :items="analysisHistory"
      :item-height="60"
      :visible-count="10"
    >
      <template #item="{ item }">
        <AnalysisItem :analysis="item" />
      </template>
    </VirtualList>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watchEffect, defineAsyncComponent } from 'vue'

// Lazy load heavy components
const AnalysisChart = defineAsyncComponent(() => import('./AnalysisChart.vue'))

// Use reactive refs efficiently
const analysisHistory = ref<AnalysisResult[]>([])
const showDetails = ref(false)

// Computed values with caching
const chartData = computed(() => {
  // Expensive computation cached automatically
  return processAnalysisData(analysisHistory.value)
})

// Debounced search
import { debounce } from 'lodash-es'

const searchTerm = ref('')
const debouncedSearch = debounce(async (term: string) => {
  if (term.length > 2) {
    analysisHistory.value = await searchAnalysisHistory(term)
  }
}, 300)

watchEffect(() => {
  debouncedSearch(searchTerm.value)
})
</script>
```

#### Bundle Optimization

```typescript
// vite.config.ts - Optimized build configuration
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  
  build: {
    // Chunk splitting for better caching
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['vue', 'vue-router', 'pinia'],
          'charts': ['chart.js', 'vue-chartjs'],
          'ui': ['@headlessui/vue', '@heroicons/vue'],
          'utils': ['lodash-es', 'date-fns']
        }
      }
    },
    
    // Minification
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,  // Remove console.log in production
        drop_debugger: true
      }
    },
    
    // Source maps for debugging
    sourcemap: process.env.NODE_ENV === 'development'
  },
  
  // Development optimizations
  server: {
    hmr: {
      overlay: false  // Disable error overlay for performance
    }
  }
})
```

### 2. Asset Optimization

#### Image Optimization

```javascript
// Optimize uploaded images on client side
class ImageOptimizer {
  static async optimizeForAnalysis(file: File): Promise<File> {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      const img = new Image()
      
      img.onload = () => {
        // Calculate optimal dimensions
        const maxWidth = 1920
        const maxHeight = 1080
        
        let { width, height } = img
        
        if (width > maxWidth || height > maxHeight) {
          const ratio = Math.min(maxWidth / width, maxHeight / height)
          width *= ratio
          height *= ratio
        }
        
        // Resize image
        canvas.width = width
        canvas.height = height
        ctx.drawImage(img, 0, 0, width, height)
        
        // Convert to optimized blob
        canvas.toBlob((blob) => {
          const optimizedFile = new File([blob], file.name, {
            type: 'image/jpeg',
            lastModified: Date.now()
          })
          resolve(optimizedFile)
        }, 'image/jpeg', 0.9)  // 90% quality
      }
      
      img.src = URL.createObjectURL(file)
    })
  }
  
  static async generateThumbnail(file: File, size: number = 150): Promise<string> {
    """Generate thumbnail for preview."""
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      const img = new Image()
      
      img.onload = () => {
        canvas.width = size
        canvas.height = size
        
        // Calculate crop area for square thumbnail
        const minDim = Math.min(img.width, img.height)
        const cropX = (img.width - minDim) / 2
        const cropY = (img.height - minDim) / 2
        
        ctx.drawImage(
          img, cropX, cropY, minDim, minDim,
          0, 0, size, size
        )
        
        resolve(canvas.toDataURL('image/jpeg', 0.8))
      }
      
      img.src = URL.createObjectURL(file)
    })
  }
}
```

## Network Performance

### 1. CDN and Edge Optimization

#### Cloudflare Configuration

```javascript
// Cloudflare Workers for edge optimization
export default {
  async fetch(request, env) {
    const url = new URL(request.url)
    
    // Cache static assets aggressively
    if (url.pathname.startsWith('/static/')) {
      const response = await fetch(request)
      const newResponse = new Response(response.body, response)
      
      newResponse.headers.set('Cache-Control', 'public, max-age=31536000') // 1 year
      newResponse.headers.set('CDN-Cache-Control', 'public, max-age=31536000')
      
      return newResponse
    }
    
    // Cache API responses briefly
    if (url.pathname.startsWith('/api/')) {
      const cacheKey = new Request(url.toString(), request)
      const cache = caches.default
      
      // Try cache first
      let response = await cache.match(cacheKey)
      
      if (!response) {
        response = await fetch(request)
        
        // Cache successful responses
        if (response.status === 200) {
          const cacheResponse = response.clone()
          cacheResponse.headers.set('Cache-Control', 'public, max-age=60') // 1 minute
          await cache.put(cacheKey, cacheResponse)
        }
      }
      
      return response
    }
    
    return fetch(request)
  }
}
```

### 2. Connection Optimization

#### HTTP/2 and Connection Pooling

```python
import aiohttp
from aiohttp import TCPConnector

class OptimizedHTTPClient:
    def __init__(self):
        # Optimized connector
        connector = TCPConnector(
            limit=100,              # Total connection pool size
            limit_per_host=20,      # Per-host connection limit
            keepalive_timeout=30,   # Keep connections alive
            enable_cleanup_closed=True,
            use_dns_cache=True,
            ttl_dns_cache=300       # DNS cache TTL
        )
        
        # Optimized timeout
        timeout = aiohttp.ClientTimeout(
            total=30,      # Total request timeout
            connect=5,     # Connection timeout
            sock_read=10   # Socket read timeout
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            # HTTP/2 support
            http_version=aiohttp.HttpVersion11,  # Or HttpVersion20
            # Compression
            auto_decompress=True,
            # Headers optimization
            headers={
                'User-Agent': 'H200-System/1.0',
                'Accept-Encoding': 'gzip, deflate, br'
            }
        )
    
    async def make_optimized_request(self, method: str, url: str, **kwargs):
        """Make HTTP request with optimization."""
        
        # Add compression headers
        headers = kwargs.get('headers', {})
        headers.update({
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        })
        kwargs['headers'] = headers
        
        # Make request with retry logic
        for attempt in range(3):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    return await response.json()
            except aiohttp.ClientError as e:
                if attempt == 2:  # Last attempt
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
```

## Monitoring Performance

### 1. Performance Metrics

#### Key Performance Indicators

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'gpu_utilization': [],
            'cache_hit_rates': [],
            'memory_usage': [],
            'error_rates': []
        }
    
    async def collect_performance_metrics(self):
        """Collect comprehensive performance metrics."""
        
        # API response times
        api_metrics = await self.collect_api_metrics()
        
        # GPU utilization
        gpu_metrics = await self.collect_gpu_metrics()
        
        # Cache performance
        cache_metrics = await self.collect_cache_metrics()
        
        # System resources
        system_metrics = await self.collect_system_metrics()
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'api': api_metrics,
            'gpu': gpu_metrics,
            'cache': cache_metrics,
            'system': system_metrics
        }
    
    async def analyze_performance_trends(self, time_window: timedelta):
        """Analyze performance trends over time."""
        
        # Get historical data
        end_time = datetime.utcnow()
        start_time = end_time - time_window
        
        metrics = await self.get_metrics_in_range(start_time, end_time)
        
        # Calculate trends
        trends = {
            'response_time_trend': self.calculate_trend(metrics['response_times']),
            'gpu_utilization_trend': self.calculate_trend(metrics['gpu_utilization']),
            'cache_hit_trend': self.calculate_trend(metrics['cache_hit_rates']),
            'error_rate_trend': self.calculate_trend(metrics['error_rates'])
        }
        
        # Generate recommendations
        recommendations = self.generate_performance_recommendations(trends)
        
        return {
            'trends': trends,
            'recommendations': recommendations,
            'time_window': str(time_window),
            'sample_count': len(metrics['response_times'])
        }
```

### 2. Automated Performance Optimization

#### Self-Tuning System

```python
class PerformanceOptimizer:
    def __init__(self):
        self.optimization_history = []
        self.current_config = self.load_default_config()
    
    async def auto_optimize(self):
        """Automatically optimize system based on performance data."""
        
        # Collect current performance
        current_perf = await self.measure_current_performance()
        
        # Identify optimization opportunities
        opportunities = self.identify_bottlenecks(current_perf)
        
        for opportunity in opportunities:
            # Test optimization
            test_config = self.generate_test_config(opportunity)
            test_performance = await self.test_configuration(test_config)
            
            # Apply if improvement is significant
            if self.is_significant_improvement(current_perf, test_performance):
                await self.apply_configuration(test_config)
                logger.info(f"Applied optimization: {opportunity['description']}")
                
                self.optimization_history.append({
                    'timestamp': datetime.utcnow(),
                    'optimization': opportunity,
                    'performance_before': current_perf,
                    'performance_after': test_performance
                })
    
    def identify_bottlenecks(self, performance_data: dict) -> List[dict]:
        """Identify performance bottlenecks and suggest optimizations."""
        opportunities = []
        
        # High GPU memory usage
        if performance_data['gpu_memory_usage'] > 0.9:
            opportunities.append({
                'type': 'gpu_memory',
                'description': 'Reduce model batch size',
                'config_changes': {'batch_size': performance_data['batch_size'] // 2}
            })
        
        # Low cache hit rate
        if performance_data['cache_hit_rate'] < 0.7:
            opportunities.append({
                'type': 'cache_tuning',
                'description': 'Increase cache TTL',
                'config_changes': {'cache_ttl': performance_data['cache_ttl'] * 1.5}
            })
        
        # High API latency
        if performance_data['api_latency_p95'] > 200:
            opportunities.append({
                'type': 'concurrency',
                'description': 'Increase worker processes',
                'config_changes': {'worker_count': performance_data['worker_count'] + 1}
            })
        
        return opportunities
```

## Performance Testing and Benchmarking

### Load Testing

```python
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor

class LoadTester:
    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url
        self.auth_token = auth_token
        self.results = []
    
    async def run_load_test(
        self,
        endpoint: str,
        concurrent_users: int,
        requests_per_user: int,
        ramp_up_time: float = 0
    ):
        """Run comprehensive load test."""
        
        # Create session for load testing
        connector = aiohttp.TCPConnector(limit=concurrent_users * 2)
        session = aiohttp.ClientSession(connector=connector)
        
        try:
            # Create user tasks
            user_tasks = []
            for user_id in range(concurrent_users):
                task = asyncio.create_task(
                    self.simulate_user(session, endpoint, requests_per_user, user_id)
                )
                user_tasks.append(task)
                
                # Ramp up gradually
                if ramp_up_time > 0:
                    await asyncio.sleep(ramp_up_time / concurrent_users)
            
            # Wait for all users to complete
            results = await asyncio.gather(*user_tasks, return_exceptions=True)
            
            # Analyze results
            return self.analyze_load_test_results(results)
            
        finally:
            await session.close()
    
    async def simulate_user(
        self, 
        session: aiohttp.ClientSession,
        endpoint: str,
        request_count: int,
        user_id: int
    ):
        """Simulate single user load."""
        user_results = []
        
        for request_id in range(request_count):
            start_time = time.time()
            
            try:
                # Make request with authentication
                async with session.post(
                    f"{self.base_url}{endpoint}",
                    headers={'Authorization': f'Bearer {self.auth_token}'},
                    data=self.generate_test_data()
                ) as response:
                    
                    end_time = time.time()
                    
                    user_results.append({
                        'user_id': user_id,
                        'request_id': request_id,
                        'response_time': end_time - start_time,
                        'status_code': response.status,
                        'success': response.status < 400
                    })
                    
            except Exception as e:
                user_results.append({
                    'user_id': user_id,
                    'request_id': request_id,
                    'error': str(e),
                    'success': False
                })
            
            # Think time between requests
            await asyncio.sleep(0.1)
        
        return user_results
```

### Benchmarking Scripts

```bash
#!/bin/bash
# scripts/benchmark.sh

echo "=== H200 System Performance Benchmark ==="

# Configuration
API_URL="http://localhost:8000"
CONCURRENT_USERS=10
REQUESTS_PER_USER=50
TEST_IMAGE="tests/fixtures/sample_mug.jpg"

# 1. Warm up system
echo "Warming up system..."
for i in {1..5}; do
    curl -s -X POST "$API_URL/api/v1/analyze/with-feedback" \
         -H "Authorization: Bearer $AUTH_TOKEN" \
         -F "image=@$TEST_IMAGE" > /dev/null
done

# 2. Run performance test
echo "Running performance test..."
python scripts/performance/load_test.py \
    --url "$API_URL" \
    --endpoint "/api/v1/analyze/with-feedback" \
    --users "$CONCURRENT_USERS" \
    --requests "$REQUESTS_PER_USER" \
    --image "$TEST_IMAGE"

# 3. GPU benchmark
echo "GPU benchmark..."
python scripts/performance/gpu_benchmark.py \
    --model-path "models/yolo_v8n.pt" \
    --batch-sizes "1,2,4,8" \
    --image-sizes "640,800,1024"

# 4. Cache performance test
echo "Cache performance test..."
python scripts/performance/cache_benchmark.py \
    --operations 10000 \
    --key-pattern "test:performance:{i}" \
    --value-size-kb 10

echo "Benchmark completed. Check results in benchmarks/"
```

This performance guide provides comprehensive strategies for optimizing every aspect of the H200 System, from GPU utilization to frontend responsiveness and network efficiency.