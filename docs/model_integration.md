# Model Integration Guide - H200 Intelligent Mug Positioning System

## Overview

The H200 system integrates three main AI models:
1. **YOLOv8** - Object detection for identifying mugs and scene objects
2. **CLIP** - Vision-language model for semantic understanding and optimal positioning
3. **MugPositioningEngine** - Advanced positioning algorithms combining rule-based and AI-guided strategies

## Architecture

```
H200ImageAnalyzer
├── ModelManager (handles model lifecycle)
│   ├── YOLOv8Model (detection)
│   └── CLIPVisionModel (embeddings)
└── MugPositioningEngine (positioning logic)
    ├── Rule-based positioning
    ├── CLIP-guided positioning
    └── Hybrid strategies
```

## Model Integration Status

### ✅ Completed
- YOLOv8 model implementation with mug-specific detection
- CLIP model for embedding generation and positioning scoring
- MugPositioningEngine with multiple strategies
- ModelManager with automatic downloading from R2/HuggingFace/GitHub LFS
- Integration with H200ImageAnalyzer
- Batch processing support
- Performance monitoring and caching

### 🔄 Configuration Options

```python
# Initialize analyzer with custom models
analyzer = H200ImageAnalyzer(
    yolo_model_size="s",         # Options: n, s, m, l
    clip_model_name="ViT-B/32",  # Options: ViT-B/32, ViT-B/16, ViT-L/14
    positioning_strategy=PositioningStrategy.HYBRID  # RULE_BASED, CLIP_GUIDED, HYBRID, etc.
)
```

## Model Performance Profiles

### YOLO Models
- **yolov8n** (Nano): Fastest (7ms), lowest memory (2GB), good for edge devices
- **yolov8s** (Small): Fast (11ms), balanced (4GB), recommended default
- **yolov8m** (Medium): Moderate (20ms), good accuracy (6GB)
- **yolov8l** (Large): Slower (35ms), high accuracy (8GB)

### CLIP Models
- **ViT-B/32**: Fast (15ms), good for general use (4GB)
- **ViT-B/16**: Better accuracy (25ms), higher resolution (6GB)
- **ViT-L/14**: Best accuracy (45ms), large model (10GB)

## Key Features

### 1. Automatic Model Downloading
Models are automatically downloaded from multiple sources:
- Primary: Cloudflare R2 (configured via R2_* env vars)
- Fallback 1: HuggingFace Hub
- Fallback 2: GitHub LFS
- Fallback 3: Direct HTTP URLs

### 2. FlashBoot Optimization
Models are preloaded during initialization for fast cold starts:
- Models cached in GPU memory
- Warmup iterations performed
- Target: 500ms-2s cold start time

### 3. Advanced Positioning
The positioning engine provides:
- **Rule-based**: Edge clearance, stable surfaces, electronics avoidance
- **CLIP-guided**: Semantic understanding of optimal placement
- **Hybrid**: Combines both approaches
- **Context-aware**: Adapts to office/kitchen/dining scenarios

### 4. Batch Processing
Efficient GPU utilization with:
- Configurable batch sizes
- Parallel processing
- Automatic caching

## Usage Example

```python
import asyncio
from PIL import Image
from src.core.analyzer import H200ImageAnalyzer

async def analyze_mug_image():
    # Create analyzer
    analyzer = H200ImageAnalyzer()
    
    # Initialize (loads models)
    await analyzer.initialize()
    
    # Load image
    image = Image.open("mug_on_table.jpg")
    
    # Analyze
    result = await analyzer.analyze_image(image)
    
    # Access results
    print(f"Detections: {len(result.detections)}")
    print(f"Mug positions: {result.mug_positions}")
    print(f"Confidence: {result.confidence_scores}")
    
    # Cleanup
    await analyzer.cleanup()

# Run
asyncio.run(analyze_mug_image())
```

## MongoDB Vector Search

The system supports vector similarity search for finding similar mug positioning scenarios:

```python
# Search for similar images
similar = await analyzer.search_similar_images(
    query_embedding=result.embeddings,
    limit=10,
    min_score=0.7
)
```

**Note**: This requires MongoDB Atlas with a vector search index configured on the `embeddings` field.

## Environment Variables

Required for model downloading:
```bash
# R2 Storage (primary source)
R2_ENDPOINT_URL=https://your-account.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=h200-models

# Optional: HuggingFace token for private models
HUGGINGFACE_TOKEN=your_token
```

## Testing

Run the integration test:
```bash
python tests/test_model_integration.py
```

This will:
1. Load all models
2. Create a test image with simulated objects
3. Run object detection
4. Calculate mug positions
5. Generate embeddings
6. Test caching
7. Test batch processing
8. Display performance statistics

## Performance Monitoring

The system tracks:
- Model load times
- Inference times per model
- GPU memory usage
- Cache hit rates
- Batch processing efficiency

Access stats via:
```python
stats = analyzer.get_performance_stats()
model_info = analyzer.model_manager.get_registry_info()
```

## Troubleshooting

### Models not downloading
1. Check R2 credentials in .env
2. Ensure internet connectivity for fallback sources
3. Check logs for specific download errors

### GPU out of memory
1. Use smaller models (yolov8n, ViT-B/32)
2. Reduce batch size
3. Enable automatic mixed precision (AMP)

### Slow inference
1. Ensure models are warmed up
2. Check GPU utilization
3. Consider using smaller models
4. Enable caching for repeated images

## Next Steps

1. Configure MongoDB Atlas vector search index
2. Fine-tune positioning rules for specific use cases
3. Benchmark different model combinations
4. Implement custom positioning strategies
5. Add model quantization for edge deployment