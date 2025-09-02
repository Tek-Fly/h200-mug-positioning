"""
Core utilities for H200 Intelligent Mug Positioning System.

This module provides common utility functions for image preprocessing,
performance monitoring, and GPU management.
"""

# Standard library imports
import base64
import hashlib
import io
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import GPUtil
import numpy as np
import psutil
import structlog
import torch
from PIL import Image

# Initialize structured logger
logger = structlog.get_logger(__name__)


def generate_analysis_id(prefix: str = "analysis") -> str:
    """
    Generate a unique analysis ID.

    Args:
        prefix: ID prefix

    Returns:
        Unique ID string
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{unique_id}"


def calculate_image_hash(image_data: Union[bytes, np.ndarray, Image.Image]) -> str:
    """
    Calculate SHA256 hash of image data.

    Args:
        image_data: Image data in various formats

    Returns:
        Hex digest of image hash
    """
    if isinstance(image_data, bytes):
        return hashlib.sha256(image_data).hexdigest()
    elif isinstance(image_data, np.ndarray):
        return hashlib.sha256(image_data.tobytes()).hexdigest()
    elif isinstance(image_data, Image.Image):
        buffer = io.BytesIO()
        image_data.save(buffer, format="PNG")
        return hashlib.sha256(buffer.getvalue()).hexdigest()
    else:
        raise ValueError(f"Unsupported image data type: {type(image_data)}")


def preprocess_image(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (640, 640),
    normalize: bool = True,
    to_tensor: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Preprocess image for model input.

    Args:
        image: Input image
        target_size: Target size (width, height)
        normalize: Apply normalization
        to_tensor: Convert to PyTorch tensor

    Returns:
        Preprocessed image
    """
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Resize with aspect ratio preservation
    image.thumbnail(target_size, Image.Resampling.LANCZOS)

    # Pad to target size
    width, height = image.size
    pad_width = target_size[0] - width
    pad_height = target_size[1] - height

    if pad_width > 0 or pad_height > 0:
        padded = Image.new("RGB", target_size, (0, 0, 0))
        padded.paste(image, (pad_width // 2, pad_height // 2))
        image = padded

    # Convert to array
    img_array = np.array(image).astype(np.float32)

    # Normalize if requested
    if normalize:
        img_array = img_array / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std

    # Convert to tensor if requested
    if to_tensor:
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
        return img_tensor.unsqueeze(0)  # Add batch dimension

    return img_array


def encode_image_base64(image: Union[Image.Image, np.ndarray, bytes]) -> str:
    """
    Encode image to base64 string.

    Args:
        image: Image in various formats

    Returns:
        Base64 encoded string
    """
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    raise ValueError(f"Unsupported image type: {type(image)}")


def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decode base64 string to PIL Image.

    Args:
        base64_string: Base64 encoded image

    Returns:
        PIL Image
    """
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


def calculate_gpu_memory_usage() -> float:
    """
    Calculate current GPU memory usage in MB.

    Returns:
        GPU memory usage in MB
    """
    if not torch.cuda.is_available():
        return 0.0

    return torch.cuda.memory_allocated() / 1024 / 1024


def get_gpu_stats() -> Dict[str, Any]:
    """
    Get comprehensive GPU statistics.

    Returns:
        Dictionary with GPU stats
    """
    stats = {"cuda_available": torch.cuda.is_available(), "gpus": []}

    if not torch.cuda.is_available():
        return stats

    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_info = {
                "id": gpu.id,
                "name": gpu.name,
                "memory_total_mb": gpu.memoryTotal,
                "memory_used_mb": gpu.memoryUsed,
                "memory_free_mb": gpu.memoryFree,
                "memory_util_percent": gpu.memoryUtil * 100,
                "gpu_util_percent": gpu.load * 100,
                "temperature": gpu.temperature,
            }
            stats["gpus"].append(gpu_info)
    except:
        # Fallback to torch stats
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info = {
                "id": i,
                "name": props.name,
                "memory_total_mb": props.total_memory / 1024 / 1024,
                "memory_allocated_mb": torch.cuda.memory_allocated(i) / 1024 / 1024,
                "memory_reserved_mb": torch.cuda.memory_reserved(i) / 1024 / 1024,
            }
            stats["gpus"].append(gpu_info)

    return stats


def get_system_stats() -> Dict[str, Any]:
    """
    Get system resource statistics.

    Returns:
        Dictionary with system stats
    """
    cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
    memory = psutil.virtual_memory()

    return {
        "cpu": {
            "count": psutil.cpu_count(),
            "percent": np.mean(cpu_percent),
            "per_cpu": cpu_percent,
        },
        "memory": {
            "total_mb": memory.total / 1024 / 1024,
            "available_mb": memory.available / 1024 / 1024,
            "used_mb": memory.used / 1024 / 1024,
            "percent": memory.percent,
        },
        "disk": {
            "total_gb": psutil.disk_usage("/").total / 1024 / 1024 / 1024,
            "used_gb": psutil.disk_usage("/").used / 1024 / 1024 / 1024,
            "free_gb": psutil.disk_usage("/").free / 1024 / 1024 / 1024,
            "percent": psutil.disk_usage("/").percent,
        },
    }


def log_performance_metrics(metrics: Dict[str, Any]) -> None:
    """
    Log performance metrics in structured format.

    Args:
        metrics: Dictionary of metrics to log
    """
    # Add timestamp
    metrics["timestamp"] = datetime.utcnow().isoformat()

    # Add system stats
    metrics["system"] = get_system_stats()
    metrics["gpu"] = get_gpu_stats()

    # Log with appropriate level based on performance
    if "processing_time_ms" in metrics:
        if metrics["processing_time_ms"] > 2000:
            logger.warning("Slow processing detected", **metrics)
        else:
            logger.info("Performance metrics", **metrics)
    else:
        logger.info("Performance metrics", **metrics)


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, operation_name: str, log_metrics: bool = True):
        """
        Initialize performance timer.

        Args:
            operation_name: Name of operation being timed
            log_metrics: Whether to log metrics on exit
        """
        self.operation_name = operation_name
        self.log_metrics = log_metrics
        self.start_time = None
        self.end_time = None
        self.duration_ms = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and optionally log."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000

        if self.log_metrics:
            log_performance_metrics(
                {
                    "operation": self.operation_name,
                    "duration_ms": self.duration_ms,
                    "success": exc_type is None,
                }
            )


def validate_image_format(
    image_data: Union[bytes, str, Image.Image],
    allowed_formats: List[str] = ["JPEG", "PNG", "WEBP"],
) -> Tuple[bool, Optional[str]]:
    """
    Validate image format.

    Args:
        image_data: Image data to validate
        allowed_formats: List of allowed formats

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if isinstance(image_data, str):
            # Assume base64
            image_data = base64.b64decode(image_data)

        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data

        if image.format not in allowed_formats:
            return False, f"Invalid format: {image.format}. Allowed: {allowed_formats}"

        # Check image size
        if image.width > 4096 or image.height > 4096:
            return (
                False,
                f"Image too large: {image.width}x{image.height}. Max: 4096x4096",
            )

        return True, None

    except Exception as e:
        return False, f"Invalid image data: {str(e)}"


def batch_images(
    images: List[Union[Image.Image, np.ndarray, torch.Tensor]], batch_size: int = 32
) -> List[List[Any]]:
    """
    Batch images for processing.

    Args:
        images: List of images
        batch_size: Maximum batch size

    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        batches.append(batch)

    return batches


def estimate_processing_time(
    num_images: int,
    avg_time_per_image_ms: float = 50,
    batch_size: int = 32,
    overhead_ms: float = 10,
) -> float:
    """
    Estimate total processing time for images.

    Args:
        num_images: Number of images to process
        avg_time_per_image_ms: Average time per image
        batch_size: Batch size for processing
        overhead_ms: Overhead per batch

    Returns:
        Estimated time in milliseconds
    """
    num_batches = (num_images + batch_size - 1) // batch_size
    batch_time = avg_time_per_image_ms * min(batch_size, num_images)
    total_time = (batch_time + overhead_ms) * num_batches

    return total_time


def format_duration(milliseconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        milliseconds: Duration in milliseconds

    Returns:
        Formatted string
    """
    if milliseconds < 1000:
        return f"{milliseconds:.1f}ms"
    elif milliseconds < 60000:
        return f"{milliseconds/1000:.1f}s"
    else:
        minutes = int(milliseconds / 60000)
        seconds = (milliseconds % 60000) / 1000
        return f"{minutes}m {seconds:.1f}s"
