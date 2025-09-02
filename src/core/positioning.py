"""Positioning analysis engine for mug placement."""

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PositioningEngine:
    """Engine for analyzing mug positioning."""
    
    def __init__(self, redis_client=None):
        """Initialize positioning engine."""
        self.redis_client = redis_client
        self.position_labels = {
            "center": "Centered on surface",
            "left_edge": "Near left edge",
            "right_edge": "Near right edge", 
            "top_edge": "Near top edge",
            "bottom_edge": "Near bottom edge",
            "top_left": "Top-left corner",
            "top_right": "Top-right corner",
            "bottom_left": "Bottom-left corner",
            "bottom_right": "Bottom-right corner",
        }
    
    async def calculate_position(
        self,
        detections: List[Any],
        image_size: Tuple[int, int],
        reference_point: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate position information for detected mugs.
        
        Args:
            detections: List of mug detections
            image_size: (width, height) of the image
            reference_point: Optional reference point for position calculation
            
        Returns:
            Position information including description and offset
        """
        if not detections:
            return {
                "description": "No mugs detected",
                "confidence": 0.0,
                "offset": {"x": 0.0, "y": 0.0},
            }
        
        # Use first detection for now (can be extended for multiple mugs)
        detection = detections[0]
        
        # Get bounding box center
        bbox_center_x = (detection.bbox.top_left.x + detection.bbox.bottom_right.x) / 2
        bbox_center_y = (detection.bbox.top_left.y + detection.bbox.bottom_right.y) / 2
        
        # Image dimensions
        img_width, img_height = image_size
        
        # Calculate relative position (0-1 range)
        rel_x = bbox_center_x / img_width
        rel_y = bbox_center_y / img_height
        
        # Determine position description
        position_desc = self._get_position_description(rel_x, rel_y)
        
        # Calculate offset from center or reference point
        if reference_point:
            ref_x, ref_y = reference_point
        else:
            # Default to image center
            ref_x, ref_y = img_width / 2, img_height / 2
        
        offset_x = bbox_center_x - ref_x
        offset_y = bbox_center_y - ref_y
        
        # Calculate confidence based on detection confidence and position clarity
        position_confidence = detection.bbox.confidence
        
        # Adjust confidence based on how clearly defined the position is
        if position_desc == "center":
            position_confidence *= 0.95  # High confidence for center
        elif "edge" in position_desc:
            position_confidence *= 0.85  # Medium confidence for edges
        else:
            position_confidence *= 0.8   # Lower confidence for corners
        
        result = {
            "description": self.position_labels.get(position_desc, position_desc),
            "position_key": position_desc,
            "confidence": float(position_confidence),
            "offset": {
                "x": float(offset_x),
                "y": float(offset_y),
            },
            "relative_position": {
                "x": float(rel_x),
                "y": float(rel_y),
            },
            "bbox_center": {
                "x": float(bbox_center_x),
                "y": float(bbox_center_y),
            },
        }
        
        # Cache result if Redis available
        if self.redis_client:
            try:
                cache_key = f"position:{img_width}x{img_height}:{bbox_center_x:.0f},{bbox_center_y:.0f}"
                await self.redis_client.setex(
                    cache_key,
                    300,  # 5 minute TTL
                    json.dumps(result),
                )
            except Exception as e:
                logger.warning(f"Failed to cache position result: {e}")
        
        return result
    
    def _get_position_description(self, rel_x: float, rel_y: float) -> str:
        """
        Get position description based on relative coordinates.
        
        Args:
            rel_x: Relative X position (0-1)
            rel_y: Relative Y position (0-1)
            
        Returns:
            Position description key
        """
        # Define thresholds
        edge_threshold = 0.2
        center_threshold = 0.3
        
        # Check for center position
        if (abs(rel_x - 0.5) < center_threshold and 
            abs(rel_y - 0.5) < center_threshold):
            return "center"
        
        # Check edges and corners
        is_left = rel_x < edge_threshold
        is_right = rel_x > (1 - edge_threshold)
        is_top = rel_y < edge_threshold
        is_bottom = rel_y > (1 - edge_threshold)
        
        # Corners
        if is_top and is_left:
            return "top_left"
        elif is_top and is_right:
            return "top_right"
        elif is_bottom and is_left:
            return "bottom_left"
        elif is_bottom and is_right:
            return "bottom_right"
        
        # Edges
        elif is_left:
            return "left_edge"
        elif is_right:
            return "right_edge"
        elif is_top:
            return "top_edge"
        elif is_bottom:
            return "bottom_edge"
        
        # Default to center if no clear edge/corner
        return "center"
    
    async def analyze_spacing(
        self,
        detections: List[Any],
        min_spacing_pixels: float = 50.0,
    ) -> Dict[str, Any]:
        """
        Analyze spacing between multiple mugs.
        
        Args:
            detections: List of mug detections
            min_spacing_pixels: Minimum required spacing in pixels
            
        Returns:
            Spacing analysis results
        """
        if len(detections) < 2:
            return {
                "has_multiple": False,
                "violations": [],
                "average_spacing": None,
            }
        
        violations = []
        spacings = []
        
        # Check spacing between all pairs
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                mug1 = detections[i]
                mug2 = detections[j]
                
                # Calculate distance between centers
                center1_x = (mug1.bbox.top_left.x + mug1.bbox.bottom_right.x) / 2
                center1_y = (mug1.bbox.top_left.y + mug1.bbox.bottom_right.y) / 2
                center2_x = (mug2.bbox.top_left.x + mug2.bbox.bottom_right.x) / 2
                center2_y = (mug2.bbox.top_left.y + mug2.bbox.bottom_right.y) / 2
                
                distance = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
                spacings.append(distance)
                
                # Check for minimum spacing violation
                if distance < min_spacing_pixels:
                    violations.append({
                        "mug1_id": mug1.id,
                        "mug2_id": mug2.id,
                        "distance": float(distance),
                        "required": float(min_spacing_pixels),
                        "shortage": float(min_spacing_pixels - distance),
                    })
        
        return {
            "has_multiple": True,
            "mug_count": len(detections),
            "violations": violations,
            "average_spacing": float(np.mean(spacings)) if spacings else None,
            "min_spacing": float(min(spacings)) if spacings else None,
            "max_spacing": float(max(spacings)) if spacings else None,
        }
    
    async def check_alignment(
        self,
        detections: List[Any],
        alignment_threshold_pixels: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Check alignment of multiple mugs.
        
        Args:
            detections: List of mug detections
            alignment_threshold_pixels: Threshold for considering mugs aligned
            
        Returns:
            Alignment analysis results
        """
        if len(detections) < 2:
            return {
                "has_multiple": False,
                "is_aligned": None,
                "alignment_type": None,
            }
        
        # Extract centers
        centers = []
        for mug in detections:
            center_x = (mug.bbox.top_left.x + mug.bbox.bottom_right.x) / 2
            center_y = (mug.bbox.top_left.y + mug.bbox.bottom_right.y) / 2
            centers.append((center_x, center_y))
        
        # Check horizontal alignment
        y_coords = [c[1] for c in centers]
        y_variance = np.var(y_coords)
        is_horizontal = y_variance < alignment_threshold_pixels**2
        
        # Check vertical alignment
        x_coords = [c[0] for c in centers]
        x_variance = np.var(x_coords)
        is_vertical = x_variance < alignment_threshold_pixels**2
        
        # Determine alignment type
        if is_horizontal and is_vertical:
            alignment_type = "clustered"
        elif is_horizontal:
            alignment_type = "horizontal"
        elif is_vertical:
            alignment_type = "vertical"
        else:
            alignment_type = "none"
        
        return {
            "has_multiple": True,
            "is_aligned": is_horizontal or is_vertical,
            "alignment_type": alignment_type,
            "horizontal_variance": float(y_variance),
            "vertical_variance": float(x_variance),
            "threshold": float(alignment_threshold_pixels),
        }


# Import json for caching
import json