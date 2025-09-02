"""
Base model class for AI models in H200 Intelligent Mug Positioning System.

This module provides abstract base classes for model management,
state persistence, and GPU allocation strategies.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

import structlog
import torch
import numpy as np

from src.database.r2_storage import get_r2_client
from src.core.cache import DualLayerCache

# Initialize structured logger
logger = structlog.get_logger(__name__)


class ModelState(Enum):
    """Model lifecycle states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    WARMING = "warming"
    READY = "ready"
    ERROR = "error"


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    version: str
    model_type: str
    model_path: Optional[str] = None
    r2_key: Optional[str] = None
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    max_batch_size: int = 32
    warmup_iterations: int = 3
    cache_model: bool = True
    preload_on_startup: bool = True
    gpu_memory_fraction: float = 0.9
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Ensure device is available
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "CUDA not available, falling back to CPU",
                requested_device=self.device
            )
            self.device = "cpu"
            self.dtype = torch.float32


class BaseModel(ABC):
    """
    Abstract base class for AI models with state management and GPU allocation.
    
    Provides common functionality for:
    - Model loading and unloading
    - State persistence
    - GPU memory management
    - Warmup procedures
    - Performance monitoring
    """
    
    def __init__(
        self,
        config: ModelConfig,
        cache: Optional[DualLayerCache] = None
    ):
        """
        Initialize base model.
        
        Args:
            config: Model configuration
            cache: Optional cache instance
        """
        self.config = config
        self.cache = cache
        self.state = ModelState.UNLOADED
        self.model: Optional[torch.nn.Module] = None
        
        # Performance tracking
        self._load_time: Optional[float] = None
        self._inference_times: List[float] = []
        self._batch_sizes: List[int] = []
        self._gpu_memory_usage: List[float] = []
        
        # State persistence
        self._state_file = Path(f"/tmp/h200_model_state_{config.name}_{config.version}.json")
        
        logger.info(
            "BaseModel initialized",
            name=config.name,
            version=config.version,
            device=config.device
        )
    
    @abstractmethod
    async def _load_model_impl(self) -> torch.nn.Module:
        """
        Implementation-specific model loading.
        
        Returns:
            Loaded PyTorch model
        """
        pass
    
    @abstractmethod
    async def preprocess(self, inputs: Any) -> Any:
        """
        Preprocess inputs for model.
        
        Args:
            inputs: Raw inputs
            
        Returns:
            Preprocessed inputs ready for model
        """
        pass
    
    @abstractmethod
    async def forward(self, inputs: Any) -> Any:
        """
        Run model inference.
        
        Args:
            inputs: Preprocessed inputs
            
        Returns:
            Model outputs
        """
        pass
    
    @abstractmethod
    async def postprocess(self, outputs: Any) -> Any:
        """
        Postprocess model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Processed outputs
        """
        pass
    
    async def load(self) -> None:
        """Load model with state management and caching."""
        if self.state == ModelState.READY:
            logger.info("Model already loaded", name=self.config.name)
            return
        
        self.state = ModelState.LOADING
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            if self.cache and self.config.cache_model:
                cached_model = await self.cache.get(
                    f"model:{self.config.name}:{self.config.version}"
                )
                if cached_model:
                    self.model = cached_model
                    self.state = ModelState.LOADED
                    logger.info(
                        "Model loaded from cache",
                        name=self.config.name,
                        load_time_ms=(time.perf_counter() - start_time) * 1000
                    )
                    await self._warmup()
                    return
            
            # Load from implementation
            self.model = await self._load_model_impl()
            
            # Move to device and set dtype
            if self.model:
                self.model = self.model.to(
                    device=self.config.device,
                    dtype=self.config.dtype
                )
                self.model.eval()
            
            self.state = ModelState.LOADED
            self._load_time = time.perf_counter() - start_time
            
            # Cache if configured
            if self.cache and self.config.cache_model:
                await self.cache.set(
                    f"model:{self.config.name}:{self.config.version}",
                    self.model,
                    cache_level="l2"  # GPU cache only
                )
            
            logger.info(
                "Model loaded successfully",
                name=self.config.name,
                device=self.config.device,
                load_time_ms=self._load_time * 1000
            )
            
            # Perform warmup
            await self._warmup()
            
        except Exception as e:
            self.state = ModelState.ERROR
            logger.error(
                "Model loading failed",
                name=self.config.name,
                error=str(e)
            )
            raise
    
    async def unload(self) -> None:
        """Unload model and free GPU memory."""
        if self.state == ModelState.UNLOADED:
            return
        
        # Save state before unloading
        await self._save_state()
        
        # Clear model
        if self.model:
            del self.model
            self.model = None
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.state = ModelState.UNLOADED
        
        logger.info("Model unloaded", name=self.config.name)
    
    async def _warmup(self) -> None:
        """Perform model warmup for optimal performance."""
        if not self.model:
            return
        
        self.state = ModelState.WARMING
        logger.info(
            "Starting model warmup",
            name=self.config.name,
            iterations=self.config.warmup_iterations
        )
        
        try:
            # Create dummy input
            dummy_input = await self._create_dummy_input()
            
            # Run warmup iterations
            warmup_times = []
            for i in range(self.config.warmup_iterations):
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    if self.config.device.startswith("cuda"):
                        with torch.cuda.amp.autocast():
                            _ = await self.forward(dummy_input)
                    else:
                        _ = await self.forward(dummy_input)
                
                warmup_time = (time.perf_counter() - start_time) * 1000
                warmup_times.append(warmup_time)
                
                # Sync GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            avg_warmup_time = np.mean(warmup_times)
            
            self.state = ModelState.READY
            logger.info(
                "Model warmup completed",
                name=self.config.name,
                avg_warmup_time_ms=avg_warmup_time
            )
            
        except Exception as e:
            self.state = ModelState.ERROR
            logger.error(
                "Model warmup failed",
                name=self.config.name,
                error=str(e)
            )
            raise
    
    async def _create_dummy_input(self) -> Any:
        """Create dummy input for warmup (override if needed)."""
        # Default: create random tensor
        return torch.randn(
            1, 3, 224, 224,
            device=self.config.device,
            dtype=self.config.dtype
        )
    
    async def predict(
        self,
        inputs: Any,
        batch_size: Optional[int] = None
    ) -> Any:
        """
        High-level prediction interface with pre/post processing.
        
        Args:
            inputs: Raw inputs
            batch_size: Optional batch size override
            
        Returns:
            Processed predictions
        """
        if self.state != ModelState.READY:
            await self.load()
        
        # Track performance
        start_time = time.perf_counter()
        
        # Preprocess
        processed_inputs = await self.preprocess(inputs)
        
        # Run inference
        with torch.no_grad():
            if self.config.device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    outputs = await self.forward(processed_inputs)
            else:
                outputs = await self.forward(processed_inputs)
        
        # Postprocess
        results = await self.postprocess(outputs)
        
        # Track metrics
        inference_time = (time.perf_counter() - start_time) * 1000
        self._inference_times.append(inference_time)
        
        if hasattr(processed_inputs, 'shape'):
            batch_size = processed_inputs.shape[0]
            self._batch_sizes.append(batch_size)
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self._gpu_memory_usage.append(gpu_memory)
        
        return results
    
    async def download_from_r2(self, local_path: Optional[str] = None) -> str:
        """
        Download model from R2 storage.
        
        Args:
            local_path: Local path to save model
            
        Returns:
            Path to downloaded model
        """
        if not self.config.r2_key:
            raise ValueError("No R2 key configured for model")
        
        r2_client = await get_r2_client()
        
        # Default local path
        if not local_path:
            local_path = f"/tmp/models/{self.config.name}_{self.config.version}.pt"
        
        # Ensure directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Download model
        await r2_client.download_file(
            self.config.r2_key,
            local_path
        )
        
        logger.info(
            "Model downloaded from R2",
            name=self.config.name,
            r2_key=self.config.r2_key,
            local_path=local_path
        )
        
        return local_path
    
    async def upload_to_r2(self, local_path: str) -> str:
        """
        Upload model to R2 storage.
        
        Args:
            local_path: Local model path
            
        Returns:
            R2 key
        """
        r2_client = await get_r2_client()
        
        # Generate R2 key
        r2_key = f"models/{self.config.name}/{self.config.version}/model.pt"
        
        # Upload model
        await r2_client.upload_file(
            local_path,
            r2_key,
            metadata={
                'model_name': self.config.name,
                'model_version': self.config.version,
                'model_type': self.config.model_type,
                'uploaded_at': time.time()
            }
        )
        
        logger.info(
            "Model uploaded to R2",
            name=self.config.name,
            r2_key=r2_key
        )
        
        return r2_key
    
    def get_gpu_allocation(self) -> Dict[str, Any]:
        """Get current GPU memory allocation."""
        if not torch.cuda.is_available():
            return {
                'device': 'cpu',
                'allocated_mb': 0,
                'reserved_mb': 0,
                'free_mb': 0
            }
        
        device_idx = int(self.config.device.split(':')[1]) if ':' in self.config.device else 0
        
        return {
            'device': self.config.device,
            'allocated_mb': torch.cuda.memory_allocated(device_idx) / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved(device_idx) / 1024 / 1024,
            'free_mb': (torch.cuda.get_device_properties(device_idx).total_memory - 
                       torch.cuda.memory_allocated(device_idx)) / 1024 / 1024
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get model performance statistics."""
        stats = {
            'state': self.state.value,
            'load_time_ms': self._load_time * 1000 if self._load_time else None,
            'inference_count': len(self._inference_times),
            'gpu_allocation': self.get_gpu_allocation()
        }
        
        if self._inference_times:
            stats.update({
                'avg_inference_time_ms': np.mean(self._inference_times),
                'median_inference_time_ms': np.median(self._inference_times),
                'p95_inference_time_ms': np.percentile(self._inference_times, 95),
                'avg_batch_size': np.mean(self._batch_sizes) if self._batch_sizes else 0,
                'avg_gpu_memory_mb': np.mean(self._gpu_memory_usage) if self._gpu_memory_usage else 0
            })
        
        return stats
    
    async def _save_state(self) -> None:
        """Save model state to disk."""
        state_data = {
            'name': self.config.name,
            'version': self.config.version,
            'state': self.state.value,
            'performance_stats': self.get_performance_stats(),
            'timestamp': time.time()
        }
        
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    async def _load_state(self) -> Optional[Dict[str, Any]]:
        """Load model state from disk."""
        if not self._state_file.exists():
            return None
        
        try:
            with open(self._state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(
                "Failed to load model state",
                name=self.config.name,
                error=str(e)
            )
            return None