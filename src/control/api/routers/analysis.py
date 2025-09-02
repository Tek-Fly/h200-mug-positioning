"""Image analysis API endpoints."""

import io
import logging
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from PIL import Image

from src.control.api.middleware.auth import get_current_user
from src.control.api.models.analysis import (
    AnalysisFeedback,
    AnalysisRequest,
    AnalysisResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    BoundingBox,
    MugDetection,
    Point2D,
    PositioningResult,
)
from src.core.analyzer import H200ImageAnalyzer
from src.core.positioning import PositioningEngine
from src.core.rules.engine import RuleEngine

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_analyzer(request: Request) -> H200ImageAnalyzer:
    """Get image analyzer from app state."""
    if not hasattr(request.app.state, "model_manager"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized",
        )
    
    # Create analyzer with model manager
    analyzer = H200ImageAnalyzer(model_manager=request.app.state.model_manager)
    return analyzer


async def get_positioning_engine(request: Request) -> PositioningEngine:
    """Get positioning engine."""
    redis_client = request.app.state.redis
    positioning_engine = PositioningEngine(redis_client=redis_client)
    return positioning_engine


async def get_rule_engine(request: Request) -> RuleEngine:
    """Get rule engine."""
    mongodb = request.app.state.mongodb
    rule_engine = RuleEngine(db=mongodb)
    await rule_engine.initialize()
    return rule_engine


@router.post("/analyze/with-feedback", response_model=AnalysisResponse)
async def analyze_image_with_feedback(
    request: Request,
    image: UploadFile = File(..., description="Image file to analyze"),
    include_feedback: bool = Form(default=True, description="Include positioning feedback"),
    rules_context: Optional[str] = Form(None, description="Natural language rules context"),
    calibration_mm_per_pixel: Optional[float] = Form(None, description="Calibration factor"),
    confidence_threshold: float = Form(default=0.7, description="Detection confidence threshold"),
    current_user: str = Depends(get_current_user),
) -> AnalysisResponse:
    """
    Analyze an image and provide positioning feedback.
    
    This endpoint:
    1. Detects mugs in the uploaded image
    2. Analyzes their positions relative to defined rules
    3. Provides feedback and suggestions for improvement
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Validate image
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid content type: {image.content_type}",
            )
        
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Get services
        analyzer = await get_analyzer(request)
        positioning_engine = await get_positioning_engine(request)
        rule_engine = await get_rule_engine(request)
        
        # Analyze image
        logger.info(f"Analyzing image for request {request_id}")
        analysis_result = await analyzer.analyze_image(
            image=pil_image,
            confidence_threshold=confidence_threshold,
        )
        
        # Convert detections
        detections = []
        for detection in analysis_result.mugs:
            mug = MugDetection(
                id=str(uuid.uuid4()),
                bbox=BoundingBox(
                    top_left=Point2D(x=detection.bbox[0], y=detection.bbox[1]),
                    bottom_right=Point2D(x=detection.bbox[2], y=detection.bbox[3]),
                    confidence=detection.confidence,
                ),
                attributes=detection.metadata,
            )
            detections.append(mug)
        
        # Analyze positioning
        positioning_data = {
            "detections": detections,
            "image_width": pil_image.width,
            "image_height": pil_image.height,
            "calibration_mm_per_pixel": calibration_mm_per_pixel,
        }
        
        # Apply rules if context provided
        rule_violations = []
        if rules_context:
            # Parse natural language rules
            parsed_rules = await rule_engine.parse_natural_language(rules_context)
            
            # Evaluate rules
            for rule in parsed_rules:
                result = await rule_engine.evaluate_rule(rule.id, positioning_data)
                if result.matched:
                    rule_violations.extend([
                        f"Rule '{rule.name}' violated: {action.parameters.get('message', 'No details')}"
                        for action in result.actions_taken
                    ])
        
        # Calculate positioning
        position_info = await positioning_engine.calculate_position(
            detections=detections,
            image_size=(pil_image.width, pil_image.height),
        )
        
        positioning = PositioningResult(
            position=position_info["description"],
            confidence=position_info["confidence"],
            offset_pixels=Point2D(
                x=position_info["offset"]["x"],
                y=position_info["offset"]["y"],
            ),
            offset_mm=Point2D(
                x=position_info["offset"]["x"] * calibration_mm_per_pixel,
                y=position_info["offset"]["y"] * calibration_mm_per_pixel,
            ) if calibration_mm_per_pixel else None,
            rule_violations=rule_violations,
        )
        
        # Generate feedback
        feedback = None
        suggestions = []
        
        if include_feedback:
            feedback = analysis_result.summary
            
            # Add suggestions based on positioning
            if positioning.offset_pixels.x > 50:
                suggestions.append("Move mug left to center it better")
            elif positioning.offset_pixels.x < -50:
                suggestions.append("Move mug right to center it better")
            
            if rule_violations:
                suggestions.append("Adjust positioning to comply with active rules")
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Store analysis result
        if hasattr(request.app.state, "mongodb"):
            analysis_doc = {
                "request_id": request_id,
                "user_id": current_user,
                "timestamp": datetime.utcnow(),
                "processing_time_ms": processing_time_ms,
                "detections": [d.model_dump() for d in detections],
                "positioning": positioning.model_dump(),
                "feedback": feedback,
                "suggestions": suggestions,
            }
            
            await request.app.state.mongodb.analysis_results.insert_one(analysis_doc)
        
        # Return response
        return AnalysisResponse(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            processing_time_ms=processing_time_ms,
            detections=detections,
            positioning=positioning,
            feedback=feedback,
            suggestions=suggestions,
            metadata={
                "user_id": current_user,
                "image_size": [pil_image.width, pil_image.height],
                "model_version": analysis_result.model_info.get("version", "unknown"),
            },
        )
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze image: {str(e)}",
        )


@router.post("/analyze/feedback")
async def submit_analysis_feedback(
    request: Request,
    feedback: AnalysisFeedback,
    current_user: str = Depends(get_current_user),
) -> dict:
    """Submit feedback for an analysis result."""
    try:
        # Store feedback
        if hasattr(request.app.state, "mongodb"):
            feedback_doc = {
                **feedback.model_dump(),
                "user_id": current_user,
                "timestamp": datetime.utcnow(),
            }
            
            await request.app.state.mongodb.analysis_feedback.insert_one(feedback_doc)
            
            # Update analysis result with feedback
            await request.app.state.mongodb.analysis_results.update_one(
                {"request_id": feedback.request_id},
                {
                    "$set": {
                        "has_feedback": True,
                        "feedback_correct": feedback.is_correct,
                    }
                },
            )
        
        return {
            "success": True,
            "message": "Feedback submitted successfully",
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}",
        )


@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    request: Request,
    batch_request: BatchAnalysisRequest,
    current_user: str = Depends(get_current_user),
) -> BatchAnalysisResponse:
    """Analyze multiple images in batch."""
    # This is a placeholder for batch processing
    # In production, this would process images in parallel
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Batch analysis not yet implemented",
    )