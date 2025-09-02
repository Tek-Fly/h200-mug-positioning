"""
H200 Image Analyzer - Main image processing pipeline for mug positioning.

This module provides the core async image processing pipeline with
GPU acceleration, batch processing, and performance monitoring.
"""

# Standard library imports
import asyncio
import base64
import hashlib
import io
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import structlog
import torch
from PIL import Image
from torch.cuda.amp import autocast

# First-party imports
from src.core.cache import DualLayerCache
from src.core.models.clip import CLIPVisionModel
from src.core.models.manager import ModelManager
from src.core.models.yolo import YOLOv8Model
from src.core.positioning import MugPositioningEngine, PositioningStrategy
from src.core.utils import (
    calculate_gpu_memory_usage,
    generate_analysis_id,
    log_performance_metrics,
    preprocess_image,
)
from src.database.mongodb import get_mongodb_client
from src.database.r2_storage import get_r2_client
from src.database.redis_client import get_redis_client

# Initialize structured logger
logger = structlog.get_logger(__name__)


@dataclass
class AnalysisResult:
    """Result of image analysis."""

    analysis_id: str
    timestamp: datetime
    image_hash: str
    detections: List[Dict[str, Any]]  # YOLO detections
    embeddings: np.ndarray  # CLIP embeddings
    mug_positions: List[Dict[str, float]]  # Calculated positions
    confidence_scores: Dict[str, float]
    processing_time_ms: float
    gpu_memory_mb: float
    cached: bool = False


@dataclass
class BatchResult:
    """Result of batch processing."""

    batch_id: str
    results: List[AnalysisResult]
    total_processing_time_ms: float
    average_time_per_image_ms: float
    gpu_peak_memory_mb: float
    cache_hit_rate: float


class H200ImageAnalyzer:
    """
    Main image analysis pipeline with GPU acceleration and async processing.

    This class orchestrates the complete image analysis workflow including:
    - Image preprocessing and validation
    - YOLO object detection
    - CLIP embedding generation
    - Mug position calculation
    - Result caching and storage
    - Performance monitoring
    """

    def __init__(
        self,
        cache: Optional[DualLayerCache] = None,
        batch_size: int = 32,
        max_image_size: Tuple[int, int] = (1920, 1080),
        gpu_device: str = "cuda:0",
        enable_amp: bool = True,
        cache_ttl: int = 3600,
        enable_performance_logging: bool = True,
        yolo_model_size: str = "s",  # Options: n, s, m, l
        clip_model_name: str = "ViT-B/32",  # Options: ViT-B/32, ViT-B/16, ViT-L/14
        positioning_strategy: PositioningStrategy = PositioningStrategy.HYBRID,
    ):
        """
        Initialize the H200 Image Analyzer.

        Args:
            cache: Dual-layer cache instance (creates new if None)
            batch_size: Maximum batch size for GPU processing
            max_image_size: Maximum image dimensions (width, height)
            gpu_device: GPU device to use
            enable_amp: Enable automatic mixed precision
            cache_ttl: Cache TTL in seconds
            enable_performance_logging: Enable performance metrics logging
            yolo_model_size: YOLO model size (n=nano, s=small, m=medium, l=large)
            clip_model_name: CLIP model variant
            positioning_strategy: Positioning strategy to use
        """
        self.cache = cache or DualLayerCache()
        self.batch_size = batch_size
        self.max_image_size = max_image_size
        self.gpu_device = gpu_device if torch.cuda.is_available() else "cpu"
        self.enable_amp = enable_amp and torch.cuda.is_available()
        self.cache_ttl = cache_ttl
        self.enable_performance_logging = enable_performance_logging

        # Initialize model manager
        self.model_manager = ModelManager(
            cache=self.cache, auto_download=True, enable_benchmarking=True
        )

        # Model configuration
        self.yolo_model_name = f"yolov8{yolo_model_size}"
        self.clip_model_name = clip_model_name
        self.positioning_strategy = positioning_strategy

        # Model instances (loaded lazily)
        self.yolo_model: Optional[YOLOv8Model] = None
        self.clip_model: Optional[CLIPVisionModel] = None
        self.positioning_engine: Optional[MugPositioningEngine] = None

        # Performance tracking
        self._processing_times: List[float] = []
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info(
            "H200ImageAnalyzer initialized",
            batch_size=batch_size,
            gpu_device=self.gpu_device,
            enable_amp=enable_amp,
            cache_ttl=cache_ttl,
            yolo_model=self.yolo_model_name,
            clip_model=clip_model_name,
            positioning_strategy=positioning_strategy.value,
        )

    async def initialize(self) -> None:
        """
        Initialize async components and warm up cache.

        This should be called before processing any images.
        """
        # Initialize cache
        await self.cache.initialize()

        # Initialize model manager
        await self.model_manager.initialize()

        # Get database clients
        self.mongodb = await get_mongodb_client()
        self.redis = await get_redis_client()
        self.r2 = await get_r2_client()

        # Ensure R2 bucket exists
        await self.r2.create_bucket_if_not_exists()

        # Create MongoDB indexes
        await self._create_mongodb_indexes()

        # Load models for FlashBoot optimization
        await self._load_models()

        logger.info("H200ImageAnalyzer initialization complete")

    async def _create_mongodb_indexes(self) -> None:
        """Create required MongoDB indexes for efficient queries."""
        indexes = [
            {"keys": [("analysis_id", 1)], "options": {"unique": True}},
            {"keys": [("image_hash", 1)], "options": {}},
            {"keys": [("timestamp", -1)], "options": {}},
            {"keys": [("embeddings", "2dsphere")], "options": {}},
        ]

        await self.mongodb.create_indexes("image_analyses", indexes)

    async def _load_models(self) -> None:
        """Load AI models for FlashBoot optimization."""
        logger.info("Loading models for FlashBoot...")

        # Load YOLO model
        self.yolo_model = await self.model_manager.get_model(self.yolo_model_name)

        # Load CLIP model - map model names to registry keys
        clip_registry_map = {
            "ViT-B/32": "clip_vit_b32",
            "ViT-B/16": "clip_vit_b16",
            "ViT-L/14": "clip_vit_l14",
        }
        clip_registry_name = clip_registry_map.get(self.clip_model_name, "clip_vit_b32")
        self.clip_model = await self.model_manager.get_model(clip_registry_name)

        # Initialize positioning engine
        self.positioning_engine = MugPositioningEngine(
            yolo_model=self.yolo_model,
            clip_model=self.clip_model,
            strategy=self.positioning_strategy,
        )

        logger.info("Models loaded successfully")

    async def analyze_image(
        self,
        image_data: Union[bytes, str, Image.Image],
        metadata: Optional[Dict[str, Any]] = None,
        skip_cache: bool = False,
    ) -> AnalysisResult:
        """
        Analyze a single image for mug positioning.

        Args:
            image_data: Image as bytes, base64 string, or PIL Image
            metadata: Optional metadata to store with results
            skip_cache: Skip cache lookup

        Returns:
            AnalysisResult with detection and positioning data
        """
        start_time = time.perf_counter()

        # Convert image to standardized format
        image, image_bytes = await self._prepare_image(image_data)
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        # Check cache unless skipped
        if not skip_cache:
            cached_result = await self._get_cached_result(image_hash)
            if cached_result:
                self._cache_hits += 1
                cached_result.cached = True
                return cached_result

        self._cache_misses += 1

        # Process image
        analysis_id = generate_analysis_id()

        # GPU processing with AMP
        if self.enable_amp:
            with autocast():
                detections = await self._run_object_detection(image)
                embeddings = await self._generate_embeddings(image)
        else:
            detections = await self._run_object_detection(image)
            embeddings = await self._generate_embeddings(image)

        # Calculate mug positions
        mug_positions = await self._calculate_mug_positions(detections, image)

        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(detections, mug_positions)

        # Measure GPU memory
        gpu_memory_mb = calculate_gpu_memory_usage() if torch.cuda.is_available() else 0

        # Create result
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        result = AnalysisResult(
            analysis_id=analysis_id,
            timestamp=datetime.utcnow(),
            image_hash=image_hash,
            detections=detections,
            embeddings=embeddings,
            mug_positions=mug_positions,
            confidence_scores=confidence_scores,
            processing_time_ms=processing_time_ms,
            gpu_memory_mb=gpu_memory_mb,
        )

        # Store results asynchronously
        await asyncio.gather(
            self._store_result(result, image_bytes, metadata),
            self._cache_result(result),
            return_exceptions=True,
        )

        # Log performance metrics
        if self.enable_performance_logging:
            self._processing_times.append(processing_time_ms)
            log_performance_metrics(
                {
                    "analysis_id": analysis_id,
                    "processing_time_ms": processing_time_ms,
                    "gpu_memory_mb": gpu_memory_mb,
                    "cache_hit": False,
                    "num_detections": len(detections),
                    "num_mug_positions": len(mug_positions),
                }
            )

        return result

    async def analyze_batch(
        self,
        images: List[Union[bytes, str, Image.Image]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        skip_cache: bool = False,
    ) -> BatchResult:
        """
        Analyze a batch of images for improved GPU utilization.

        Args:
            images: List of images to process
            metadata_list: Optional metadata for each image
            skip_cache: Skip cache lookup

        Returns:
            BatchResult with all analysis results
        """
        batch_start_time = time.perf_counter()
        batch_id = generate_analysis_id("batch")

        # Process in chunks based on batch_size
        results = []
        cache_hits = 0

        for i in range(0, len(images), self.batch_size):
            chunk = images[i : i + self.batch_size]
            chunk_metadata = (
                metadata_list[i : i + self.batch_size]
                if metadata_list
                else [None] * len(chunk)
            )

            # Process chunk in parallel
            chunk_results = await asyncio.gather(
                *[
                    self.analyze_image(img, meta, skip_cache)
                    for img, meta in zip(chunk, chunk_metadata)
                ],
                return_exceptions=True,
            )

            # Filter out exceptions and count cache hits
            for result in chunk_results:
                if isinstance(result, AnalysisResult):
                    results.append(result)
                    if result.cached:
                        cache_hits += 1
                else:
                    logger.error("Batch processing error", error=str(result))

        # Calculate batch metrics
        total_processing_time_ms = (time.perf_counter() - batch_start_time) * 1000
        avg_time_per_image = total_processing_time_ms / len(images) if images else 0
        gpu_peak_memory = max(r.gpu_memory_mb for r in results) if results else 0
        cache_hit_rate = cache_hits / len(images) if images else 0

        batch_result = BatchResult(
            batch_id=batch_id,
            results=results,
            total_processing_time_ms=total_processing_time_ms,
            average_time_per_image_ms=avg_time_per_image,
            gpu_peak_memory_mb=gpu_peak_memory,
            cache_hit_rate=cache_hit_rate,
        )

        # Log batch performance
        if self.enable_performance_logging:
            log_performance_metrics(
                {
                    "batch_id": batch_id,
                    "batch_size": len(images),
                    "total_processing_time_ms": total_processing_time_ms,
                    "average_time_per_image_ms": avg_time_per_image,
                    "gpu_peak_memory_mb": gpu_peak_memory,
                    "cache_hit_rate": cache_hit_rate,
                }
            )

        return batch_result

    async def _prepare_image(
        self, image_data: Union[bytes, str, Image.Image]
    ) -> Tuple[Image.Image, bytes]:
        """Convert image data to PIL Image and bytes."""
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
            image_bytes = image_data
        elif isinstance(image_data, str):
            # Assume base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        elif isinstance(image_data, Image.Image):
            image = image_data
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        else:
            raise ValueError(f"Unsupported image data type: {type(image_data)}")

        # Resize if needed
        if (
            image.size[0] > self.max_image_size[0]
            or image.size[1] > self.max_image_size[1]
        ):
            image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

        return image, image_bytes

    async def _run_object_detection(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Run YOLO object detection."""
        if not self.yolo_model:
            logger.error("YOLO model not loaded")
            return []

        try:
            # Run detection (returns list of detections per image)
            detections_batch = await self.yolo_model.detect_mugs(
                image, mug_only=False  # Get all objects for scene context
            )

            # Since we're processing a single image, get first result
            detections = detections_batch[0] if detections_batch else []

            logger.debug(
                "Object detection completed",
                num_detections=len(detections),
                num_mugs=len([d for d in detections if d.get("is_mug_related", False)]),
            )

            return detections

        except Exception as e:
            logger.error("Object detection failed", error=str(e))
            return []

    async def _generate_embeddings(self, image: Image.Image) -> np.ndarray:
        """Generate CLIP embeddings."""
        if not self.clip_model:
            logger.error("CLIP model not loaded")
            # Return zero embeddings with expected dimension
            return np.zeros(512, dtype=np.float32)

        try:
            # Generate embeddings for the image
            embeddings = await self.clip_model.encode_images(image)

            # Ensure it's a 1D array (single image)
            if embeddings.ndim > 1:
                embeddings = embeddings[0]

            logger.debug(
                "Embedding generation completed", embedding_dim=embeddings.shape[0]
            )

            return embeddings.astype(np.float32)

        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            # Return zero embeddings as fallback
            return np.zeros(512, dtype=np.float32)

    async def _calculate_mug_positions(
        self, detections: List[Dict[str, Any]], image: Optional[Image.Image] = None
    ) -> List[Dict[str, float]]:
        """Calculate optimal mug positions based on detections."""
        # If we have a positioning engine and image, use advanced positioning
        if self.positioning_engine and image:
            try:
                # Get full positioning analysis
                positioning_result = await self.positioning_engine.calculate_positions(
                    image=image, detections=detections
                )

                # Convert MugPosition objects to dict format
                mug_positions = []
                for pos in positioning_result.positions:
                    position = {
                        "x": pos.x,
                        "y": pos.y,
                        "width": pos.width,
                        "height": pos.height,
                        "confidence": pos.confidence,
                        "strategy": pos.strategy,
                        "reasoning": pos.reasoning,
                        "safe_zone": pos.safe_zone,
                        "conflicts": pos.conflicts,
                    }
                    mug_positions.append(position)

                # Add recommendations to first position if available
                if mug_positions and positioning_result.recommendations:
                    mug_positions[0][
                        "recommendations"
                    ] = positioning_result.recommendations

                logger.debug(
                    "Advanced positioning completed",
                    num_positions=len(mug_positions),
                    strategy=self.positioning_strategy.value,
                    overall_confidence=positioning_result.overall_confidence,
                )

                return mug_positions

            except Exception as e:
                logger.error(
                    "Advanced positioning failed, falling back to basic", error=str(e)
                )

        # Fallback to basic positioning
        mug_positions = []

        # Filter for mug/cup detections
        mug_detections = [d for d in detections if d.get("is_mug_related", False)]

        for detection in mug_detections:
            bbox = detection.get("bbox", {})
            position = {
                "x": (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2,
                "y": (bbox.get("y1", 0) + bbox.get("y2", 0)) / 2,
                "width": bbox.get("x2", 0) - bbox.get("x1", 0),
                "height": bbox.get("y2", 0) - bbox.get("y1", 0),
                "confidence": detection.get("confidence", 0),
                "class": detection.get("class", "unknown"),
                "strategy": "basic",
                "reasoning": "Fallback positioning",
            }
            mug_positions.append(position)

        return mug_positions

    def _calculate_confidence_scores(
        self, detections: List[Dict[str, Any]], mug_positions: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate various confidence scores."""
        if not detections:
            return {
                "overall": 0.0,
                "detection": 0.0,
                "positioning": 0.0,
                "stability": 0.0,
            }

        # Average detection confidence
        detection_conf = np.mean([d.get("confidence", 0) for d in detections])

        # Positioning confidence based on mug positions
        positioning_conf = len(mug_positions) / max(len(detections), 1)

        # Stability confidence (placeholder)
        stability_conf = 0.8 if mug_positions else 0.0

        # Overall confidence
        overall_conf = np.mean([detection_conf, positioning_conf, stability_conf])

        return {
            "overall": float(overall_conf),
            "detection": float(detection_conf),
            "positioning": float(positioning_conf),
            "stability": float(stability_conf),
        }

    async def _get_cached_result(self, image_hash: str) -> Optional[AnalysisResult]:
        """Get cached analysis result."""
        cached_data = await self.cache.get(f"analysis:{image_hash}")
        if cached_data:
            # Reconstruct AnalysisResult from cached data
            return AnalysisResult(
                analysis_id=cached_data["analysis_id"],
                timestamp=datetime.fromisoformat(cached_data["timestamp"]),
                image_hash=cached_data["image_hash"],
                detections=cached_data["detections"],
                embeddings=np.array(cached_data["embeddings"]),
                mug_positions=cached_data["mug_positions"],
                confidence_scores=cached_data["confidence_scores"],
                processing_time_ms=cached_data["processing_time_ms"],
                gpu_memory_mb=cached_data["gpu_memory_mb"],
            )
        return None

    async def _cache_result(self, result: AnalysisResult) -> None:
        """Cache analysis result."""
        cache_data = {
            "analysis_id": result.analysis_id,
            "timestamp": result.timestamp.isoformat(),
            "image_hash": result.image_hash,
            "detections": result.detections,
            "embeddings": result.embeddings.tolist(),
            "mug_positions": result.mug_positions,
            "confidence_scores": result.confidence_scores,
            "processing_time_ms": result.processing_time_ms,
            "gpu_memory_mb": result.gpu_memory_mb,
        }

        await self.cache.set(
            f"analysis:{result.image_hash}", cache_data, ttl=self.cache_ttl
        )

    async def _store_result(
        self,
        result: AnalysisResult,
        image_bytes: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store analysis result in MongoDB and image in R2."""
        # Store image in R2
        image_key = f"images/{result.analysis_id}.png"
        await self.r2.upload_bytes(
            image_bytes,
            image_key,
            metadata={
                "analysis_id": result.analysis_id,
                "image_hash": result.image_hash,
                "timestamp": result.timestamp.isoformat(),
            },
            content_type="image/png",
        )

        # Store analysis in MongoDB
        doc = {
            "analysis_id": result.analysis_id,
            "timestamp": result.timestamp,
            "image_hash": result.image_hash,
            "image_key": image_key,
            "detections": result.detections,
            "embeddings": result.embeddings.tolist(),
            "mug_positions": result.mug_positions,
            "confidence_scores": result.confidence_scores,
            "processing_time_ms": result.processing_time_ms,
            "gpu_memory_mb": result.gpu_memory_mb,
            "metadata": metadata or {},
        }

        collection = self.mongodb.get_collection("image_analyses")
        await collection.insert_one(doc)

    async def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis result by ID."""
        collection = self.mongodb.get_collection("image_analyses")
        return await collection.find_one({"analysis_id": analysis_id})

    async def get_recent_analyses(
        self, limit: int = 100, skip: int = 0
    ) -> List[Dict[str, Any]]:
        """Get recent analysis results."""
        collection = self.mongodb.get_collection("image_analyses")
        cursor = collection.find().sort("timestamp", -1).skip(skip).limit(limit)
        return await cursor.to_list(length=limit)

    async def search_similar_images(
        self, query_embedding: np.ndarray, limit: int = 10, min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar images using vector similarity.

        Note: This requires MongoDB Atlas with vector search configured.
        For now, returns empty list if not configured.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            min_score: Minimum similarity score

        Returns:
            List of similar image analyses
        """
        try:
            collection = self.mongodb.get_collection("image_analyses")

            # MongoDB Atlas Vector Search query
            # This requires setting up a search index in Atlas
            pipeline = [
                {
                    "$search": {
                        "index": "embedding_index",
                        "knnBeta": {
                            "vector": query_embedding.tolist(),
                            "path": "embeddings",
                            "k": limit,
                            "filter": {},
                        },
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "analysis_id": 1,
                        "image_hash": 1,
                        "timestamp": 1,
                        "mug_positions": 1,
                        "confidence_scores": 1,
                        "score": {"$meta": "searchScore"},
                    }
                },
                {"$match": {"score": {"$gte": min_score}}},
            ]

            results = []
            async for doc in collection.aggregate(pipeline):
                results.append(doc)

            logger.info(
                "Vector search completed", num_results=len(results), min_score=min_score
            )

            return results

        except Exception as e:
            logger.warning(
                "Vector search failed (index may not be configured)", error=str(e)
            )
            return []

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self._processing_times:
            return {
                "cache_hit_rate": 0.0,
                "average_processing_time_ms": 0.0,
                "median_processing_time_ms": 0.0,
                "total_processed": 0,
                "cache_hits": 0,
                "cache_misses": 0,
            }

        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_hit_rate": cache_hit_rate,
            "average_processing_time_ms": np.mean(self._processing_times),
            "median_processing_time_ms": np.median(self._processing_times),
            "min_processing_time_ms": np.min(self._processing_times),
            "max_processing_time_ms": np.max(self._processing_times),
            "total_processed": total_requests,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Cleanup models
        if self.model_manager:
            await self.model_manager.cleanup()

        # Cleanup cache
        await self.cache.cleanup()

        logger.info("H200ImageAnalyzer cleanup complete")
