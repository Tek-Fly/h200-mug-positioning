"""
YOLOv8 model implementation for object detection in H200 Intelligent Mug Positioning System.

This module provides YOLOv8 integration with ultralytics for efficient
object detection with focus on mug/cup detection.
"""

# Standard library imports
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import structlog
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

# First-party imports
from src.core.models.base import BaseModel, ModelConfig, ModelState

# Initialize structured logger
logger = structlog.get_logger(__name__)


class YOLOv8Model(BaseModel):
    """
    YOLOv8 model for object detection with mug-specific optimizations.

    Supports multiple model sizes (n, s, m, l, x) and provides
    efficient batch inference with GPU acceleration.
    """

    # Mug-related class names we're interested in
    MUG_CLASSES = ["cup", "mug", "glass", "bottle", "bowl"]

    # Model size configurations
    MODEL_SIZES = {
        "n": {"params": 3.2e6, "flops": 8.7e9},  # Nano - fastest
        "s": {"params": 11.2e6, "flops": 28.6e9},  # Small
        "m": {"params": 25.9e6, "flops": 78.9e9},  # Medium
        "l": {"params": 43.7e6, "flops": 165.2e9},  # Large
        "x": {"params": 68.2e6, "flops": 257.8e9},  # Extra Large - most accurate
    }

    def __init__(
        self,
        model_size: str = "s",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        agnostic_nms: bool = False,
        cache: Optional[Any] = None,
    ):
        """
        Initialize YOLOv8 model.

        Args:
            model_size: Model size (n, s, m, l, x)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
            max_detections: Maximum number of detections
            agnostic_nms: Class-agnostic NMS
            cache: Optional cache instance
        """
        if model_size not in self.MODEL_SIZES:
            raise ValueError(
                f"Invalid model size: {model_size}. Must be one of {list(self.MODEL_SIZES.keys())}"
            )

        # Create model config
        config = ModelConfig(
            name=f"yolov8{model_size}",
            version="8.0.196",
            model_type="detection",
            model_path=f"yolov8{model_size}.pt",
            r2_key=f"models/yolo/yolov8{model_size}_v8.0.196.pt",
            max_batch_size=32,
            warmup_iterations=3,
            metadata={
                "model_size": model_size,
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold,
                "max_detections": max_detections,
                "agnostic_nms": agnostic_nms,
            },
        )

        super().__init__(config, cache)

        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.agnostic_nms = agnostic_nms

        # Class name mapping
        self._class_names: Optional[List[str]] = None

    async def _load_model_impl(self) -> torch.nn.Module:
        """Load YOLOv8 model from disk or download."""
        model_path = Path(self.config.model_path)

        # Check if model exists locally
        if not model_path.exists():
            # Try to download from R2
            try:
                logger.info(f"Downloading model from R2: {self.config.r2_key}")
                model_path = await self.download_from_r2(str(model_path))
            except Exception as e:
                # Fall back to ultralytics download
                logger.warning(
                    f"R2 download failed: {e}. Downloading from ultralytics..."
                )
                model_path = self.config.model_path

        # Load model with ultralytics
        model = YOLO(str(model_path))

        # Store class names
        self._class_names = model.names if hasattr(model, "names") else []

        logger.info(
            "YOLOv8 model loaded",
            model_size=self.model_size,
            num_classes=len(self._class_names),
            model_path=str(model_path),
        )

        return model

    async def preprocess(
        self, inputs: Union[Image.Image, List[Image.Image], np.ndarray]
    ) -> torch.Tensor:
        """
        Preprocess images for YOLOv8.

        Args:
            inputs: Single image, list of images, or numpy array

        Returns:
            Preprocessed tensor ready for model
        """
        # Convert single image to list
        if isinstance(inputs, Image.Image):
            inputs = [inputs]
        elif isinstance(inputs, np.ndarray):
            # Assume it's a single image array
            inputs = [Image.fromarray(inputs)]

        # YOLOv8 handles preprocessing internally, but we need to ensure correct format
        # For batch processing, we'll convert to numpy arrays
        processed_images = []

        for img in inputs:
            if isinstance(img, Image.Image):
                # Convert PIL to numpy
                img_array = np.array(img)
            else:
                img_array = img

            # Ensure correct shape (H, W, C)
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.ndim == 3 and img_array.shape[0] == 3:
                img_array = img_array.transpose(1, 2, 0)

            processed_images.append(img_array)

        return processed_images

    async def forward(self, inputs: List[np.ndarray]) -> List[Results]:
        """
        Run YOLOv8 inference.

        Args:
            inputs: Preprocessed images

        Returns:
            List of detection results
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Run inference with configured parameters
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.model(
                inputs,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                agnostic=self.agnostic_nms,
                device=self.config.device,
                verbose=False,
            ),
        )

        return results

    async def postprocess(self, outputs: List[Results]) -> List[List[Dict[str, Any]]]:
        """
        Postprocess YOLOv8 outputs to standard format.

        Args:
            outputs: YOLOv8 Results objects

        Returns:
            List of detections per image
        """
        all_detections = []

        for result in outputs:
            detections = []

            if result.boxes is not None:
                boxes = result.boxes

                # Extract detection information
                for i in range(len(boxes)):
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

                    # Get class and confidence
                    cls_idx = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])

                    # Get class name
                    class_name = (
                        self._class_names[cls_idx]
                        if cls_idx < len(self._class_names)
                        else f"class_{cls_idx}"
                    )

                    # Check if it's a mug-related class
                    is_mug_related = class_name.lower() in [
                        c.lower() for c in self.MUG_CLASSES
                    ]

                    detection = {
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                        },
                        "class": class_name,
                        "class_id": cls_idx,
                        "confidence": confidence,
                        "is_mug_related": is_mug_related,
                        "area": float((x2 - x1) * (y2 - y1)),
                        "center": {
                            "x": float((x1 + x2) / 2),
                            "y": float((y1 + y2) / 2),
                        },
                    }

                    detections.append(detection)

            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x["confidence"], reverse=True)

            all_detections.append(detections)

        return all_detections

    async def detect_mugs(
        self, images: Union[Image.Image, List[Image.Image]], mug_only: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Specialized method for mug detection.

        Args:
            images: Input images
            mug_only: Filter to only mug-related objects

        Returns:
            List of mug detections per image
        """
        # Ensure model is loaded
        if self.state != ModelState.READY:
            await self.load()

        # Process images
        preprocessed = await self.preprocess(images)
        results = await self.forward(preprocessed)
        detections = await self.postprocess(results)

        # Filter for mug-related objects if requested
        if mug_only:
            filtered_detections = []
            for image_detections in detections:
                mug_detections = [d for d in image_detections if d["is_mug_related"]]
                filtered_detections.append(mug_detections)
            detections = filtered_detections

        return detections

    async def _create_dummy_input(self) -> List[np.ndarray]:
        """Create dummy input for warmup."""
        # Create a batch of dummy images
        batch_size = min(4, self.config.max_batch_size)
        dummy_images = []

        for _ in range(batch_size):
            # Random image with typical input size
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            dummy_images.append(dummy_img)

        return dummy_images

    def analyze_detections(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze detections for mug positioning insights.

        Args:
            detections: List of detections for a single image

        Returns:
            Analysis results
        """
        mug_detections = [d for d in detections if d["is_mug_related"]]

        analysis = {
            "total_objects": len(detections),
            "mug_count": len(mug_detections),
            "mug_types": {},
            "spatial_distribution": {
                "clustering_score": 0.0,
                "average_distance": 0.0,
                "coverage_ratio": 0.0,
            },
            "quality_metrics": {
                "average_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
            },
        }

        if mug_detections:
            # Count mug types
            for det in mug_detections:
                mug_type = det["class"]
                analysis["mug_types"][mug_type] = (
                    analysis["mug_types"].get(mug_type, 0) + 1
                )

            # Calculate quality metrics
            confidences = [d["confidence"] for d in mug_detections]
            analysis["quality_metrics"]["average_confidence"] = float(
                np.mean(confidences)
            )
            analysis["quality_metrics"]["min_confidence"] = float(np.min(confidences))
            analysis["quality_metrics"]["max_confidence"] = float(np.max(confidences))

            # Calculate spatial distribution
            if len(mug_detections) > 1:
                centers = [d["center"] for d in mug_detections]

                # Calculate pairwise distances
                distances = []
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dx = centers[i]["x"] - centers[j]["x"]
                        dy = centers[i]["y"] - centers[j]["y"]
                        distance = np.sqrt(dx**2 + dy**2)
                        distances.append(distance)

                analysis["spatial_distribution"]["average_distance"] = float(
                    np.mean(distances)
                )

                # Simple clustering score (inverse of average distance normalized)
                max_distance = np.sqrt(640**2 + 640**2)  # Diagonal of typical image
                analysis["spatial_distribution"]["clustering_score"] = 1.0 - (
                    analysis["spatial_distribution"]["average_distance"] / max_distance
                )

            # Coverage ratio (total mug area / image area)
            total_area = sum(d["area"] for d in mug_detections)
            image_area = 640 * 640  # Typical YOLO input size
            analysis["spatial_distribution"]["coverage_ratio"] = min(
                total_area / image_area, 1.0
            )

        return analysis

    def filter_overlapping_detections(
        self, detections: List[Dict[str, Any]], iou_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Filter overlapping detections using custom logic.

        Args:
            detections: List of detections
            iou_threshold: IOU threshold for filtering

        Returns:
            Filtered detections
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence
        sorted_detections = sorted(
            detections, key=lambda x: x["confidence"], reverse=True
        )
        keep = []

        for i, det in enumerate(sorted_detections):
            should_keep = True

            # Check against already kept detections
            for kept_det in keep:
                iou = self._calculate_iou(det["bbox"], kept_det["bbox"])
                if iou > iou_threshold:
                    should_keep = False
                    break

            if should_keep:
                keep.append(det)

        return keep

    def _calculate_iou(self, bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
        """Calculate IoU between two bounding boxes."""
        # Calculate intersection
        x1_inter = max(bbox1["x1"], bbox2["x1"])
        y1_inter = max(bbox1["y1"], bbox2["y1"])
        x2_inter = min(bbox1["x2"], bbox2["x2"])
        y2_inter = min(bbox1["y2"], bbox2["y2"])

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        # Calculate areas
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        bbox1_area = (bbox1["x2"] - bbox1["x1"]) * (bbox1["y2"] - bbox1["y1"])
        bbox2_area = (bbox2["x2"] - bbox2["x1"]) * (bbox2["y2"] - bbox2["y1"])

        # Calculate union
        union_area = bbox1_area + bbox2_area - inter_area

        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0.0

        return iou
