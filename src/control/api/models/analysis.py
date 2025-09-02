"""Models for image analysis endpoints."""

# Standard library imports
from datetime import datetime
from typing import Any, Dict, List, Optional

# Third-party imports
from pydantic import BaseModel, ConfigDict, Field


class Point2D(BaseModel):
    """2D point coordinates."""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class BoundingBox(BaseModel):
    """Bounding box for detected objects."""

    top_left: Point2D = Field(..., description="Top-left corner")
    bottom_right: Point2D = Field(..., description="Bottom-right corner")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")


class MugDetection(BaseModel):
    """Detected mug information."""

    id: str = Field(..., description="Unique mug ID")
    bbox: BoundingBox = Field(..., description="Bounding box")
    class_name: str = Field(default="mug", description="Object class")
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Additional attributes"
    )


class PositioningResult(BaseModel):
    """Positioning analysis result."""

    position: str = Field(
        ..., description="Position description (e.g., 'left edge', 'center')"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Position confidence")
    offset_pixels: Point2D = Field(
        ..., description="Offset from ideal position in pixels"
    )
    offset_mm: Optional[Point2D] = Field(
        None, description="Offset in millimeters if calibrated"
    )
    rule_violations: List[str] = Field(
        default_factory=list, description="Violated rules"
    )


class AnalysisRequest(BaseModel):
    """Image analysis request."""

    include_feedback: bool = Field(
        default=True, description="Include positioning feedback"
    )
    rules_context: Optional[str] = Field(
        None, description="Natural language rules context"
    )
    calibration_mm_per_pixel: Optional[float] = Field(
        None, description="Calibration factor"
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Detection confidence threshold"
    )


class AnalysisResponse(BaseModel):
    """Image analysis response."""

    model_config = ConfigDict(from_attributes=True)

    request_id: str = Field(..., description="Unique request ID")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )

    detections: List[MugDetection] = Field(..., description="Detected mugs")
    positioning: PositioningResult = Field(..., description="Positioning analysis")

    feedback: Optional[str] = Field(None, description="Human-readable feedback")
    suggestions: List[str] = Field(
        default_factory=list, description="Positioning suggestions"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class AnalysisFeedback(BaseModel):
    """User feedback for analysis result."""

    request_id: str = Field(..., description="Original request ID")
    is_correct: bool = Field(..., description="Whether the analysis was correct")
    correct_position: Optional[str] = Field(
        None, description="Correct position if different"
    )
    comments: Optional[str] = Field(None, description="Additional comments")


class BatchAnalysisRequest(BaseModel):
    """Batch analysis request."""

    images: List[str] = Field(..., description="List of image URLs or base64 data")
    settings: AnalysisRequest = Field(
        default_factory=AnalysisRequest, description="Analysis settings"
    )
    parallel: bool = Field(default=True, description="Process images in parallel")


class BatchAnalysisResponse(BaseModel):
    """Batch analysis response."""

    request_id: str = Field(..., description="Batch request ID")
    total_images: int = Field(..., description="Total number of images")
    successful: int = Field(..., description="Number of successful analyses")
    failed: int = Field(..., description="Number of failed analyses")
    results: List[AnalysisResponse] = Field(
        ..., description="Individual analysis results"
    )
    errors: List[Dict[str, str]] = Field(
        default_factory=list, description="Error details for failed images"
    )
