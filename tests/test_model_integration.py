"""
Test script for model integration in H200 Intelligent Mug Positioning System.

This script tests the integration of YOLO, CLIP, and positioning models
with the H200ImageAnalyzer.
"""

import asyncio
import numpy as np
from PIL import Image
import structlog

from src.core.analyzer import H200ImageAnalyzer
from src.core.cache import DualLayerCache
from src.core.positioning import PositioningStrategy

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def test_model_integration():
    """Test the complete model integration."""
    
    logger.info("Starting model integration test...")
    
    # Create cache
    cache = DualLayerCache()
    
    # Create analyzer with different model configurations
    analyzer = H200ImageAnalyzer(
        cache=cache,
        yolo_model_size="s",  # Small model for testing
        clip_model_name="ViT-B/32",
        positioning_strategy=PositioningStrategy.HYBRID,
        enable_performance_logging=True
    )
    
    try:
        # Initialize analyzer (this will load models)
        logger.info("Initializing analyzer and loading models...")
        await analyzer.initialize()
        
        # Create a test image (640x640 RGB)
        test_image = Image.new('RGB', (640, 640), color='white')
        
        # Add some rectangles to simulate objects
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_image)
        
        # Draw a "mug" (rectangle)
        draw.rectangle([100, 100, 200, 250], fill='brown', outline='black')
        draw.text((120, 260), "Mug", fill='black')
        
        # Draw a "table" (larger rectangle)
        draw.rectangle([50, 300, 590, 400], fill='tan', outline='black')
        draw.text((300, 350), "Table", fill='black')
        
        # Draw a "laptop" (rectangle)
        draw.rectangle([350, 150, 500, 250], fill='gray', outline='black')
        draw.text((400, 260), "Laptop", fill='black')
        
        logger.info("Running image analysis...")
        
        # Analyze the test image
        result = await analyzer.analyze_image(test_image)
        
        # Print results
        logger.info(
            "Analysis completed",
            analysis_id=result.analysis_id,
            processing_time_ms=result.processing_time_ms,
            num_detections=len(result.detections),
            num_mug_positions=len(result.mug_positions),
            confidence_scores=result.confidence_scores,
            gpu_memory_mb=result.gpu_memory_mb,
            cached=result.cached
        )
        
        # Test object detection
        if result.detections:
            logger.info("Detections found:")
            for det in result.detections:
                logger.info(
                    "  Detection",
                    class_name=det.get('class'),
                    confidence=det.get('confidence'),
                    is_mug_related=det.get('is_mug_related'),
                    bbox=det.get('bbox')
                )
        
        # Test mug positioning
        if result.mug_positions:
            logger.info("Mug positions calculated:")
            for pos in result.mug_positions:
                logger.info(
                    "  Position",
                    x=pos.get('x'),
                    y=pos.get('y'),
                    confidence=pos.get('confidence'),
                    strategy=pos.get('strategy'),
                    reasoning=pos.get('reasoning')
                )
        
        # Test embeddings
        logger.info(
            "Embeddings generated",
            shape=result.embeddings.shape,
            dtype=result.embeddings.dtype,
            non_zero=np.count_nonzero(result.embeddings)
        )
        
        # Test caching
        logger.info("Testing cache...")
        cached_result = await analyzer.analyze_image(test_image)
        logger.info(
            "Cache test",
            cached=cached_result.cached,
            processing_time_ms=cached_result.processing_time_ms
        )
        
        # Test batch processing
        logger.info("Testing batch processing...")
        batch_images = [test_image] * 3
        batch_result = await analyzer.analyze_batch(batch_images)
        logger.info(
            "Batch processing completed",
            batch_id=batch_result.batch_id,
            num_results=len(batch_result.results),
            total_time_ms=batch_result.total_processing_time_ms,
            avg_time_per_image_ms=batch_result.average_time_per_image_ms,
            cache_hit_rate=batch_result.cache_hit_rate
        )
        
        # Get performance stats
        stats = analyzer.get_performance_stats()
        logger.info("Performance statistics", **stats)
        
        # Test model manager info
        registry_info = analyzer.model_manager.get_registry_info()
        logger.info("Available models:")
        for name, info in registry_info.items():
            logger.info(f"  {name}: {info['description']} (loaded: {info['loaded']})")
        
        logger.info("Model integration test completed successfully!")
        
    except Exception as e:
        logger.error("Model integration test failed", error=str(e), exc_info=True)
        raise
    
    finally:
        # Cleanup
        await analyzer.cleanup()
        logger.info("Cleanup completed")


if __name__ == "__main__":
    asyncio.run(test_model_integration())