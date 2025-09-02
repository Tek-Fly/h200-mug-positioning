#!/usr/bin/env python3
"""
Preload models for FlashBoot optimization
This script loads AI models into memory to reduce cold start times
"""

import os
import sys
import torch
import clip
from ultralytics import YOLO
import asyncio
import logging
import time
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelPreloader:
    """Handles model preloading and warming"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model: Optional[YOLO] = None
        self.clip_model = None
        self.clip_preprocess = None
        
    async def load_yolo_model(self) -> bool:
        """Load and warm up YOLOv8 model"""
        try:
            start_time = time.time()
            logger.info("Loading YOLOv8 model...")
            
            # Load model
            self.yolo_model = YOLO('yolov8n.pt')
            self.yolo_model.to(self.device)
            
            # Warm up with dummy inference
            logger.info("Warming up YOLOv8...")
            dummy_image = torch.randn(1, 3, 640, 640).to(self.device)
            
            # Run inference multiple times to ensure GPU kernels are compiled
            for i in range(3):
                _ = self.yolo_model.predict(
                    source=dummy_image,
                    verbose=False,
                    imgsz=640,
                    conf=0.25,
                    iou=0.45
                )
            
            load_time = time.time() - start_time
            logger.info(f"YOLOv8 loaded and warmed up in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            return False
    
    async def load_clip_model(self) -> bool:
        """Load and warm up CLIP model"""
        try:
            start_time = time.time()
            logger.info("Loading CLIP model...")
            
            # Load model
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            
            # Warm up with dummy inference
            logger.info("Warming up CLIP...")
            dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_text = clip.tokenize(["a photo of a mug"]).to(self.device)
            
            # Run inference multiple times
            with torch.no_grad():
                for i in range(3):
                    _ = self.clip_model.encode_image(dummy_image)
                    _ = self.clip_model.encode_text(dummy_text)
            
            load_time = time.time() - start_time
            logger.info(f"CLIP loaded and warmed up in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            return False
    
    async def verify_gpu(self) -> bool:
        """Verify GPU is available and working"""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available, using CPU")
                return True
            
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Test GPU computation
            test_tensor = torch.randn(1000, 1000).to(self.device)
            result = torch.matmul(test_tensor, test_tensor)
            torch.cuda.synchronize()
            
            logger.info("GPU verification successful")
            return True
            
        except Exception as e:
            logger.error(f"GPU verification failed: {e}")
            return False
    
    async def optimize_memory(self) -> None:
        """Optimize GPU memory settings"""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            
            # Set memory fraction if needed
            if os.getenv("PYTORCH_CUDA_ALLOC_CONF"):
                logger.info(f"CUDA allocation config: {os.getenv('PYTORCH_CUDA_ALLOC_CONF')}")
            
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            logger.info("GPU memory optimized")
    
    async def preload_all(self) -> bool:
        """Preload all models"""
        try:
            total_start = time.time()
            logger.info("Starting model preloading...")
            
            # Verify GPU
            if not await self.verify_gpu():
                return False
            
            # Optimize memory
            await self.optimize_memory()
            
            # Load models in parallel
            tasks = [
                self.load_yolo_model(),
                self.load_clip_model()
            ]
            
            results = await asyncio.gather(*tasks)
            
            if all(results):
                total_time = time.time() - total_start
                logger.info(f"All models preloaded successfully in {total_time:.2f}s")
                
                # Log memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    logger.info(f"GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
                
                return True
            else:
                logger.error("Some models failed to load")
                return False
                
        except Exception as e:
            logger.error(f"Model preloading failed: {e}")
            return False

async def main():
    """Main preloading function"""
    preloader = ModelPreloader()
    success = await preloader.preload_all()
    
    if success:
        logger.info("Model preloading completed successfully")
        logger.info("Ready for FlashBoot deployment")
        return 0
    else:
        logger.error("Model preloading failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)