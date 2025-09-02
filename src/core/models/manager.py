"""
Model manager for H200 Intelligent Mug Positioning System.

This module provides model registry, versioning, automatic downloading,
caching, and performance benchmarking.
"""

# Standard library imports
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

# Third-party imports
import aiohttp
import structlog
import torch
from huggingface_hub import hf_hub_download

# First-party imports
from src.core.cache import DualLayerCache
from src.core.models.base import BaseModel, ModelConfig, ModelState
from src.core.models.clip import CLIPVisionModel
from src.core.models.yolo import YOLOv8Model
from src.database.r2_storage import get_r2_client

# Initialize structured logger
logger = structlog.get_logger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata for registry."""

    model_class: Type[BaseModel]
    default_config: ModelConfig
    description: str
    requirements: Dict[str, Any]
    performance_profile: Dict[str, float]


@dataclass
class BenchmarkResult:
    """Model benchmark results."""

    model_name: str
    model_version: str
    timestamp: datetime
    load_time_ms: float
    warmup_time_ms: float
    inference_times_ms: List[float]
    batch_sizes: List[int]
    memory_usage_mb: float
    throughput_fps: float
    device: str
    success: bool
    error: Optional[str] = None


class ModelManager:
    """
    Centralized model management with registry, versioning, and benchmarking.

    Features:
    - Model registry with metadata
    - Automatic model downloading from multiple sources
    - Model versioning and compatibility checking
    - Performance benchmarking
    - State persistence
    - Resource management
    """

    # Model registry
    REGISTRY: Dict[str, ModelMetadata] = {
        "yolov8n": ModelMetadata(
            model_class=YOLOv8Model,
            default_config=ModelConfig(
                name="yolov8n",
                version="8.0.196",
                model_type="detection",
                max_batch_size=64,
                gpu_memory_fraction=0.3,
            ),
            description="YOLOv8 Nano - Fastest, lowest accuracy",
            requirements={"gpu_memory_gb": 2, "inference_ms": 7},
            performance_profile={"speed": 10, "accuracy": 6, "memory": 10},
        ),
        "yolov8s": ModelMetadata(
            model_class=YOLOv8Model,
            default_config=ModelConfig(
                name="yolov8s",
                version="8.0.196",
                model_type="detection",
                max_batch_size=32,
                gpu_memory_fraction=0.4,
            ),
            description="YOLOv8 Small - Fast, good accuracy",
            requirements={"gpu_memory_gb": 4, "inference_ms": 11},
            performance_profile={"speed": 8, "accuracy": 7, "memory": 8},
        ),
        "yolov8m": ModelMetadata(
            model_class=YOLOv8Model,
            default_config=ModelConfig(
                name="yolov8m",
                version="8.0.196",
                model_type="detection",
                max_batch_size=16,
                gpu_memory_fraction=0.5,
            ),
            description="YOLOv8 Medium - Balanced",
            requirements={"gpu_memory_gb": 6, "inference_ms": 20},
            performance_profile={"speed": 6, "accuracy": 8, "memory": 6},
        ),
        "yolov8l": ModelMetadata(
            model_class=YOLOv8Model,
            default_config=ModelConfig(
                name="yolov8l",
                version="8.0.196",
                model_type="detection",
                max_batch_size=8,
                gpu_memory_fraction=0.6,
            ),
            description="YOLOv8 Large - Slower, high accuracy",
            requirements={"gpu_memory_gb": 8, "inference_ms": 35},
            performance_profile={"speed": 4, "accuracy": 9, "memory": 4},
        ),
        "clip_vit_b32": ModelMetadata(
            model_class=CLIPVisionModel,
            default_config=ModelConfig(
                name="clip_vit_b32",
                version="1.0",
                model_type="vision_language",
                max_batch_size=32,
                gpu_memory_fraction=0.4,
            ),
            description="CLIP ViT-B/32 - Fast, good for general use",
            requirements={"gpu_memory_gb": 4, "inference_ms": 15},
            performance_profile={"speed": 8, "accuracy": 7, "memory": 8},
        ),
        "clip_vit_b16": ModelMetadata(
            model_class=CLIPVisionModel,
            default_config=ModelConfig(
                name="clip_vit_b16",
                version="1.0",
                model_type="vision_language",
                max_batch_size=16,
                gpu_memory_fraction=0.5,
            ),
            description="CLIP ViT-B/16 - Higher resolution, better accuracy",
            requirements={"gpu_memory_gb": 6, "inference_ms": 25},
            performance_profile={"speed": 6, "accuracy": 8, "memory": 6},
        ),
        "clip_vit_l14": ModelMetadata(
            model_class=CLIPVisionModel,
            default_config=ModelConfig(
                name="clip_vit_l14",
                version="1.0",
                model_type="vision_language",
                max_batch_size=8,
                gpu_memory_fraction=0.7,
            ),
            description="CLIP ViT-L/14 - Large model, best accuracy",
            requirements={"gpu_memory_gb": 10, "inference_ms": 45},
            performance_profile={"speed": 3, "accuracy": 10, "memory": 3},
        ),
    }

    def __init__(
        self,
        cache: Optional[DualLayerCache] = None,
        models_dir: str = "/models",
        auto_download: bool = True,
        enable_benchmarking: bool = True,
    ):
        """
        Initialize model manager.

        Args:
            cache: Optional cache instance
            models_dir: Directory for model storage
            auto_download: Automatically download missing models
            enable_benchmarking: Enable performance benchmarking
        """
        self.cache = cache or DualLayerCache()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.auto_download = auto_download
        self.enable_benchmarking = enable_benchmarking

        # Active models
        self._models: Dict[str, BaseModel] = {}

        # Model states
        self._model_states: Dict[str, Dict[str, Any]] = {}

        # Benchmark results
        self._benchmarks: List[BenchmarkResult] = []

        # Download sources
        self.download_sources = {
            "r2": self._download_from_r2,
            "huggingface": self._download_from_huggingface,
            "github_lfs": self._download_from_github_lfs,
            "http": self._download_from_http,
        }

        logger.info(
            "ModelManager initialized",
            models_dir=str(self.models_dir),
            registry_size=len(self.REGISTRY),
            auto_download=auto_download,
        )

    async def initialize(self) -> None:
        """Initialize model manager and cache."""
        await self.cache.initialize()

        # Load saved states
        await self._load_states()

        # Verify model files
        await self._verify_models()

        logger.info("ModelManager initialization complete")

    async def get_model(
        self, model_name: str, load_on_demand: bool = True, **kwargs
    ) -> BaseModel:
        """
        Get model instance by name.

        Args:
            model_name: Model name from registry
            load_on_demand: Load model if not already loaded
            **kwargs: Additional model initialization parameters

        Returns:
            Model instance
        """
        if model_name not in self.REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(self.REGISTRY.keys())}"
            )

        # Check if already loaded
        if model_name in self._models:
            model = self._models[model_name]
            if model.state == ModelState.READY:
                return model

        # Create new instance
        metadata = self.REGISTRY[model_name]

        # Prepare initialization kwargs
        init_kwargs = {"cache": self.cache}

        # Handle model-specific parameters
        if model_name.startswith("yolov8"):
            model_size = model_name[-1]  # Extract size (n, s, m, l)
            init_kwargs["model_size"] = model_size
        elif model_name.startswith("clip"):
            # Extract CLIP model variant
            if "vit_b32" in model_name:
                init_kwargs["model_name"] = "ViT-B/32"
            elif "vit_b16" in model_name:
                init_kwargs["model_name"] = "ViT-B/16"
            elif "vit_l14" in model_name:
                init_kwargs["model_name"] = "ViT-L/14"

        # Override with user kwargs
        init_kwargs.update(kwargs)

        # Create model instance
        model = metadata.model_class(**init_kwargs)

        # Store instance
        self._models[model_name] = model

        # Load if requested
        if load_on_demand:
            await self.load_model(model_name)

        return model

    async def load_model(self, model_name: str) -> BaseModel:
        """
        Load model and ensure it's ready.

        Args:
            model_name: Model name

        Returns:
            Loaded model
        """
        model = await self.get_model(model_name, load_on_demand=False)

        if model.state != ModelState.READY:
            # Check if model files exist
            model_path = self.models_dir / f"{model_name}.pt"

            if not model_path.exists() and self.auto_download:
                await self._download_model(model_name)

            # Load model
            await model.load()

            # Update state
            self._model_states[model_name] = {
                "state": model.state.value,
                "loaded_at": datetime.utcnow().isoformat(),
                "performance_stats": model.get_performance_stats(),
            }

            # Save states
            await self._save_states()

        return model

    async def unload_model(self, model_name: str) -> None:
        """Unload model to free resources."""
        if model_name in self._models:
            model = self._models[model_name]
            await model.unload()

            # Update state
            self._model_states[model_name]["state"] = ModelState.UNLOADED.value
            await self._save_states()

    async def preload_models(self, model_names: List[str]) -> None:
        """
        Preload multiple models for FlashBoot optimization.

        Args:
            model_names: List of models to preload
        """
        logger.info(f"Preloading {len(model_names)} models...")

        # Load models in parallel
        tasks = [self.load_model(name) for name in model_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Report results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(
            f"Preloading complete: {successful}/{len(model_names)} successful",
            failed=[
                name
                for name, r in zip(model_names, results)
                if isinstance(r, Exception)
            ],
        )

    async def benchmark_model(
        self,
        model_name: str,
        num_iterations: int = 100,
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
    ) -> BenchmarkResult:
        """
        Benchmark model performance.

        Args:
            model_name: Model to benchmark
            num_iterations: Number of inference iterations
            batch_sizes: Batch sizes to test

        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking {model_name}...")

        result = BenchmarkResult(
            model_name=model_name,
            model_version=self.REGISTRY[model_name].default_config.version,
            timestamp=datetime.utcnow(),
            load_time_ms=0,
            warmup_time_ms=0,
            inference_times_ms=[],
            batch_sizes=[],
            memory_usage_mb=0,
            throughput_fps=0,
            device="cpu",
            success=False,
        )

        try:
            # Load model
            load_start = time.perf_counter()
            model = await self.load_model(model_name)
            result.load_time_ms = (time.perf_counter() - load_start) * 1000

            result.device = model.config.device

            # Test different batch sizes
            all_times = []

            for batch_size in batch_sizes:
                if batch_size > model.config.max_batch_size:
                    continue

                # Create dummy input
                dummy_input = await model._create_dummy_input()

                # Adjust for batch size
                if isinstance(dummy_input, list):
                    dummy_input = dummy_input[:batch_size]
                elif isinstance(dummy_input, torch.Tensor):
                    dummy_input = dummy_input[:batch_size]

                # Run iterations
                batch_times = []

                for _ in range(num_iterations):
                    start_time = time.perf_counter()

                    with torch.no_grad():
                        _ = await model.predict(dummy_input)

                    inference_time = (time.perf_counter() - start_time) * 1000
                    batch_times.append(inference_time)
                    all_times.append(inference_time)

                result.batch_sizes.append(batch_size)

                logger.info(
                    f"Batch size {batch_size}: avg={np.mean(batch_times):.2f}ms, "
                    f"median={np.median(batch_times):.2f}ms"
                )

            # Calculate metrics
            result.inference_times_ms = all_times
            result.memory_usage_mb = model.get_gpu_allocation()["allocated_mb"]

            # Calculate throughput (images per second)
            avg_time_ms = np.mean(all_times)
            avg_batch_size = np.mean(result.batch_sizes)
            result.throughput_fps = (avg_batch_size / avg_time_ms) * 1000

            result.success = True

            # Store benchmark
            self._benchmarks.append(result)

            logger.info(
                f"Benchmark complete for {model_name}",
                avg_inference_ms=avg_time_ms,
                throughput_fps=result.throughput_fps,
                memory_mb=result.memory_usage_mb,
            )

        except Exception as e:
            result.error = str(e)
            logger.error(f"Benchmark failed for {model_name}: {e}")

        return result

    async def benchmark_all_models(self) -> List[BenchmarkResult]:
        """Benchmark all registered models."""
        results = []

        for model_name in self.REGISTRY:
            result = await self.benchmark_model(model_name)
            results.append(result)

        # Save benchmark results
        await self._save_benchmarks()

        return results

    async def get_optimal_model(
        self, model_type: str, constraints: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Get optimal model based on constraints.

        Args:
            model_type: Type of model (detection, vision_language)
            constraints: Optional constraints (e.g., max_memory_gb, max_inference_ms)

        Returns:
            Optimal model name
        """
        candidates = []

        for name, metadata in self.REGISTRY.items():
            if metadata.default_config.model_type != model_type:
                continue

            # Check constraints
            if constraints:
                reqs = metadata.requirements

                if "max_memory_gb" in constraints:
                    if reqs.get("gpu_memory_gb", 0) > constraints["max_memory_gb"]:
                        continue

                if "max_inference_ms" in constraints:
                    if reqs.get("inference_ms", 0) > constraints["max_inference_ms"]:
                        continue

            candidates.append((name, metadata))

        if not candidates:
            raise ValueError(f"No models found matching constraints: {constraints}")

        # Sort by performance profile (prefer balanced models)
        candidates.sort(
            key=lambda x: sum(x[1].performance_profile.values()), reverse=True
        )

        return candidates[0][0]

    async def _download_model(self, model_name: str) -> str:
        """Download model from available sources."""
        metadata = self.REGISTRY[model_name]
        model_path = self.models_dir / f"{model_name}.pt"

        # Try download sources in order
        for source_name, download_func in self.download_sources.items():
            try:
                logger.info(f"Attempting to download {model_name} from {source_name}")
                await download_func(model_name, model_path)

                if model_path.exists():
                    logger.info(
                        f"Successfully downloaded {model_name} from {source_name}"
                    )
                    return str(model_path)

            except Exception as e:
                logger.warning(f"Failed to download from {source_name}: {e}")
                continue

        raise RuntimeError(f"Failed to download model {model_name} from any source")

    async def _download_from_r2(self, model_name: str, target_path: Path) -> None:
        """Download model from R2 storage."""
        metadata = self.REGISTRY[model_name]
        r2_key = metadata.default_config.r2_key

        if not r2_key:
            raise ValueError(f"No R2 key configured for {model_name}")

        r2_client = await get_r2_client()
        await r2_client.download_file(r2_key, str(target_path))

    async def _download_from_huggingface(
        self, model_name: str, target_path: Path
    ) -> None:
        """Download model from HuggingFace Hub."""
        # Map to HuggingFace model IDs
        hf_models = {
            "yolov8n": "ultralyticsplus/yolov8n",
            "yolov8s": "ultralyticsplus/yolov8s",
            "yolov8m": "ultralyticsplus/yolov8m",
            "yolov8l": "ultralyticsplus/yolov8l",
            "clip_vit_b32": "openai/clip-vit-base-patch32",
            "clip_vit_b16": "openai/clip-vit-base-patch16",
            "clip_vit_l14": "openai/clip-vit-large-patch14",
        }

        if model_name not in hf_models:
            raise ValueError(f"No HuggingFace mapping for {model_name}")

        # Download using HF Hub
        downloaded_path = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: hf_hub_download(
                repo_id=hf_models[model_name],
                filename="pytorch_model.bin",
                cache_dir=str(self.models_dir / ".cache"),
            ),
        )

        # Move to target path
        Path(downloaded_path).rename(target_path)

    async def _download_from_github_lfs(
        self, model_name: str, target_path: Path
    ) -> None:
        """Download model from GitHub LFS."""
        # GitHub LFS URLs for models
        github_urls = {
            "yolov8n": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            "yolov8s": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
            "yolov8m": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
            "yolov8l": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
        }

        if model_name not in github_urls:
            raise ValueError(f"No GitHub LFS URL for {model_name}")

        await self._download_from_http(model_name, target_path, github_urls[model_name])

    async def _download_from_http(
        self, model_name: str, target_path: Path, url: Optional[str] = None
    ) -> None:
        """Download model from HTTP URL."""
        if not url:
            # Use default URLs
            default_urls = {
                "yolov8n": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                "yolov8s": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
                "yolov8m": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
                "yolov8l": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
            }

            if model_name not in default_urls:
                raise ValueError(f"No download URL for {model_name}")

            url = default_urls[model_name]

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()

                with open(target_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)

    async def _verify_models(self) -> None:
        """Verify model files and checksums."""
        for model_name in self.REGISTRY:
            model_path = self.models_dir / f"{model_name}.pt"

            if model_path.exists():
                # Calculate checksum
                with open(model_path, "rb") as f:
                    checksum = hashlib.sha256(f.read()).hexdigest()

                logger.info(
                    f"Model file verified: {model_name}",
                    path=str(model_path),
                    size_mb=model_path.stat().st_size / 1024 / 1024,
                    checksum=checksum[:8],
                )
            else:
                logger.warning(f"Model file not found: {model_name}")

    async def _save_states(self) -> None:
        """Save model states to disk."""
        state_file = self.models_dir / "model_states.json"

        with open(state_file, "w") as f:
            json.dump(self._model_states, f, indent=2)

    async def _load_states(self) -> None:
        """Load model states from disk."""
        state_file = self.models_dir / "model_states.json"

        if state_file.exists():
            with open(state_file, "r") as f:
                self._model_states = json.load(f)

    async def _save_benchmarks(self) -> None:
        """Save benchmark results."""
        benchmark_file = self.models_dir / "benchmarks.json"

        # Convert to serializable format
        benchmarks_data = []
        for b in self._benchmarks:
            data = {
                "model_name": b.model_name,
                "model_version": b.model_version,
                "timestamp": b.timestamp.isoformat(),
                "load_time_ms": b.load_time_ms,
                "warmup_time_ms": b.warmup_time_ms,
                "avg_inference_ms": (
                    np.mean(b.inference_times_ms) if b.inference_times_ms else 0
                ),
                "memory_usage_mb": b.memory_usage_mb,
                "throughput_fps": b.throughput_fps,
                "device": b.device,
                "success": b.success,
                "error": b.error,
            }
            benchmarks_data.append(data)

        with open(benchmark_file, "w") as f:
            json.dump(benchmarks_data, f, indent=2)

    def get_registry_info(self) -> Dict[str, Any]:
        """Get information about registered models."""
        info = {}

        for name, metadata in self.REGISTRY.items():
            info[name] = {
                "description": metadata.description,
                "type": metadata.default_config.model_type,
                "requirements": metadata.requirements,
                "performance": metadata.performance_profile,
                "loaded": name in self._models
                and self._models[name].state == ModelState.READY,
            }

        return info

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Unload all models
        for model_name in list(self._models.keys()):
            await self.unload_model(model_name)

        # Save final states
        await self._save_states()

        logger.info("ModelManager cleanup complete")
