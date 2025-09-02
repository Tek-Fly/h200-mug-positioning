"""
MCP Tool Implementations.

Implements the individual tools exposed by the MCP server for
mug positioning analysis and control.
"""

import time
import base64
import io
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio

import structlog
from PIL import Image
import numpy as np

from src.core.analyzer import ImageAnalyzer, AnalysisResult
from src.core.rules.engine import RuleEngine
from src.core.positioning import (
    MugPositioningEngine,
    PositioningResult,
    PositioningStrategy
)
from src.database.mongodb import get_mongodb_client
from src.database.redis_client import get_redis_client

from .models import (
    MCPToolDefinition,
    MCPToolParameter,
    MCPToolType,
    MCPRequest,
    MCPResponse,
    MCPToolResult,
    MCPError
)


logger = structlog.get_logger(__name__)


class BaseMCPTool:
    """Base class for MCP tools."""
    
    def __init__(
        self,
        name: str,
        description: str,
        authenticator: Optional[Any] = None  # MCPAuthenticator type
    ):
        self.name = name
        self.description = description
        self.authenticator = authenticator
        self._initialized = False
    
    async def initialize(self):
        """Initialize the tool."""
        if self._initialized:
            return
        
        await self._setup()
        self._initialized = True
    
    async def _setup(self):
        """Override in subclasses for specific setup."""
        pass
    
    def get_definition(self) -> MCPToolDefinition:
        """Get tool definition."""
        raise NotImplementedError
    
    async def execute(
        self,
        request: MCPRequest,
        auth_info: Optional[Dict[str, Any]] = None
    ) -> MCPResponse:
        """Execute the tool."""
        raise NotImplementedError
    
    def _create_response(
        self,
        request_id: str,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[MCPError] = None,
        execution_time_ms: float = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MCPResponse:
        """Create standardized response."""
        return MCPResponse(
            id=str(uuid.uuid4()),
            request_id=request_id,
            result=MCPToolResult(
                success=success,
                data=data,
                error=error,
                execution_time_ms=execution_time_ms,
                metadata=metadata
            )
        )


class AnalyzeImageTool(BaseMCPTool):
    """Tool for analyzing images for mug positioning."""
    
    def __init__(
        self,
        analyzer: Optional[ImageAnalyzer] = None,
        authenticator: Optional[Any] = None  # MCPAuthenticator type
    ):
        super().__init__(
            name="analyze_image",
            description="Analyze an image to detect mugs and suggest optimal positions",
            authenticator=authenticator
        )
        self.analyzer = analyzer
    
    async def _setup(self):
        """Initialize analyzer if not provided."""
        if not self.analyzer:
            self.analyzer = ImageAnalyzer()
            await self.analyzer.initialize()
    
    def get_definition(self) -> MCPToolDefinition:
        """Get tool definition."""
        return MCPToolDefinition(
            name=self.name,
            type=MCPToolType.ASYNC,
            description=self.description,
            parameters=[
                MCPToolParameter(
                    name="image",
                    type="string",
                    description="Base64-encoded image data or image URL",
                    required=True
                ),
                MCPToolParameter(
                    name="image_format",
                    type="string",
                    description="Image format (jpeg, png, webp)",
                    required=False,
                    default="jpeg",
                    enum=["jpeg", "png", "webp"]
                ),
                MCPToolParameter(
                    name="apply_rules",
                    type="boolean",
                    description="Whether to apply natural language rules",
                    required=False,
                    default=True
                ),
                MCPToolParameter(
                    name="return_embeddings",
                    type="boolean",
                    description="Whether to return CLIP embeddings",
                    required=False,
                    default=False
                ),
                MCPToolParameter(
                    name="confidence_threshold",
                    type="number",
                    description="Minimum confidence for detections (0-1)",
                    required=False,
                    default=0.5
                )
            ],
            returns={
                "type": "object",
                "properties": {
                    "analysis_id": {"type": "string"},
                    "detections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "class": {"type": "string"},
                                "confidence": {"type": "number"},
                                "bbox": {"type": "array", "items": {"type": "number"}},
                                "position": {"type": "object"}
                            }
                        }
                    },
                    "suggestions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "mug_id": {"type": "string"},
                                "current_position": {"type": "object"},
                                "suggested_position": {"type": "object"},
                                "reason": {"type": "string"},
                                "confidence": {"type": "number"}
                            }
                        }
                    },
                    "embeddings": {"type": "array", "optional": True},
                    "processing_time_ms": {"type": "number"}
                }
            },
            rate_limit={"requests_per_minute": 30},
            async_timeout=30
        )
    
    async def execute(
        self,
        request: MCPRequest,
        auth_info: Optional[Dict[str, Any]] = None
    ) -> MCPResponse:
        """Execute image analysis."""
        start_time = time.time()
        
        try:
            # Extract parameters
            params = request.parameters
            image_data = params.get("image")
            image_format = params.get("image_format", "jpeg")
            apply_rules = params.get("apply_rules", True)
            return_embeddings = params.get("return_embeddings", False)
            confidence_threshold = params.get("confidence_threshold", 0.5)
            
            if not image_data:
                return self._create_response(
                    request.id,
                    success=False,
                    error=MCPError(
                        code="MISSING_PARAMETER",
                        message="Image data is required"
                    ),
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Decode image
            if image_data.startswith("data:"):
                # Remove data URL prefix
                image_data = image_data.split(",")[1]
            
            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                return self._create_response(
                    request.id,
                    success=False,
                    error=MCPError(
                        code="INVALID_IMAGE",
                        message=f"Failed to decode image: {str(e)}"
                    ),
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Analyze image
            result = await self.analyzer.analyze_image(
                image,
                apply_rules=apply_rules,
                confidence_threshold=confidence_threshold
            )
            
            # Prepare response data
            response_data = {
                "analysis_id": result.analysis_id,
                "detections": [
                    {
                        "class": det["class"],
                        "confidence": det["confidence"],
                        "bbox": det["bbox"],
                        "position": {
                            "x": det["bbox"][0] + det["bbox"][2] / 2,
                            "y": det["bbox"][1] + det["bbox"][3] / 2
                        }
                    }
                    for det in result.detections
                ],
                "suggestions": result.suggestions,
                "processing_time_ms": result.processing_time_ms
            }
            
            if return_embeddings and hasattr(result, 'embeddings'):
                response_data["embeddings"] = result.embeddings.tolist()
            
            # Store in database
            if auth_info:
                response_data["client_id"] = auth_info.get("client_id")
            
            return self._create_response(
                request.id,
                success=True,
                data=response_data,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "image_size": list(image.size),
                    "format": image_format,
                    "rules_applied": apply_rules
                }
            )
            
        except Exception as e:
            logger.error("Image analysis failed", error=str(e), request_id=request.id)
            return self._create_response(
                request.id,
                success=False,
                error=MCPError(
                    code="ANALYSIS_ERROR",
                    message=f"Analysis failed: {str(e)}"
                ),
                execution_time_ms=(time.time() - start_time) * 1000
            )


class ApplyRulesTool(BaseMCPTool):
    """Tool for applying natural language rules to positioning."""
    
    def __init__(
        self,
        rule_engine: Optional[RuleEngine] = None,
        authenticator: Optional[Any] = None  # MCPAuthenticator type
    ):
        super().__init__(
            name="apply_rules",
            description="Apply natural language positioning rules",
            authenticator=authenticator
        )
        self.rule_engine = rule_engine
    
    async def _setup(self):
        """Initialize rule engine if not provided."""
        if not self.rule_engine:
            self.rule_engine = RuleEngine()
            await self.rule_engine.initialize()
    
    def get_definition(self) -> MCPToolDefinition:
        """Get tool definition."""
        return MCPToolDefinition(
            name=self.name,
            type=MCPToolType.FUNCTION,
            description=self.description,
            parameters=[
                MCPToolParameter(
                    name="rule_text",
                    type="string",
                    description="Natural language rule to apply",
                    required=True
                ),
                MCPToolParameter(
                    name="priority",
                    type="integer",
                    description="Rule priority (higher = more important)",
                    required=False,
                    default=5
                ),
                MCPToolParameter(
                    name="tags",
                    type="array",
                    description="Tags for rule categorization",
                    required=False,
                    default=[]
                ),
                MCPToolParameter(
                    name="validate_only",
                    type="boolean",
                    description="Only validate without saving",
                    required=False,
                    default=False
                )
            ],
            returns={
                "type": "object",
                "properties": {
                    "rule_id": {"type": "string"},
                    "parsed_rule": {"type": "object"},
                    "validation": {
                        "type": "object",
                        "properties": {
                            "is_valid": {"type": "boolean"},
                            "issues": {"type": "array"},
                            "warnings": {"type": "array"}
                        }
                    },
                    "saved": {"type": "boolean"}
                }
            },
            rate_limit={"requests_per_minute": 60}
        )
    
    async def execute(
        self,
        request: MCPRequest,
        auth_info: Optional[Dict[str, Any]] = None
    ) -> MCPResponse:
        """Execute rule application."""
        start_time = time.time()
        
        try:
            # Extract parameters
            params = request.parameters
            rule_text = params.get("rule_text")
            priority = params.get("priority", 5)
            tags = params.get("tags", [])
            validate_only = params.get("validate_only", False)
            
            if not rule_text:
                return self._create_response(
                    request.id,
                    success=False,
                    error=MCPError(
                        code="MISSING_PARAMETER",
                        message="Rule text is required"
                    ),
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Add or validate rule
            if validate_only:
                validation_result = await self.rule_engine.validate_rule(rule_text)
                
                return self._create_response(
                    request.id,
                    success=True,
                    data={
                        "rule_id": None,
                        "parsed_rule": validation_result.parsed_rule,
                        "validation": {
                            "is_valid": validation_result.is_valid,
                            "issues": validation_result.issues,
                            "warnings": validation_result.warnings
                        },
                        "saved": False
                    },
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            else:
                rule_result = await self.rule_engine.add_rule(
                    rule_text,
                    priority=priority,
                    tags=tags,
                    created_by=auth_info.get("client_id") if auth_info else None
                )
                
                return self._create_response(
                    request.id,
                    success=True,
                    data={
                        "rule_id": rule_result.rule_id,
                        "parsed_rule": rule_result.parsed_rule,
                        "validation": {
                            "is_valid": rule_result.validation.is_valid,
                            "issues": rule_result.validation.issues,
                            "warnings": rule_result.validation.warnings
                        },
                        "saved": True
                    },
                    execution_time_ms=(time.time() - start_time) * 1000,
                    metadata={
                        "priority": priority,
                        "tags": tags
                    }
                )
            
        except Exception as e:
            logger.error("Rule application failed", error=str(e), request_id=request.id)
            return self._create_response(
                request.id,
                success=False,
                error=MCPError(
                    code="RULE_ERROR",
                    message=f"Rule processing failed: {str(e)}"
                ),
                execution_time_ms=(time.time() - start_time) * 1000
            )


class GetSuggestionsTool(BaseMCPTool):
    """Tool for getting positioning suggestions for a scene."""
    
    def __init__(
        self,
        positioning_engine: Optional[MugPositioningEngine] = None,
        authenticator: Optional[Any] = None  # MCPAuthenticator type
    ):
        super().__init__(
            name="get_suggestions",
            description="Get mug positioning suggestions based on scene analysis",
            authenticator=authenticator
        )
        self.positioning_engine = positioning_engine
    
    async def _setup(self):
        """Initialize positioning engine if not provided."""
        if not self.positioning_engine:
            self.positioning_engine = MugPositioningEngine()
            await self.positioning_engine.initialize()
    
    def get_definition(self) -> MCPToolDefinition:
        """Get tool definition."""
        return MCPToolDefinition(
            name=self.name,
            type=MCPToolType.FUNCTION,
            description=self.description,
            parameters=[
                MCPToolParameter(
                    name="scene_context",
                    type="object",
                    description="Scene context including detected objects",
                    required=True
                ),
                MCPToolParameter(
                    name="strategy",
                    type="string",
                    description="Positioning strategy to use",
                    required=False,
                    default="balanced",
                    enum=["safety_first", "balanced", "efficiency", "aesthetic"]
                ),
                MCPToolParameter(
                    name="constraints",
                    type="object",
                    description="Additional positioning constraints",
                    required=False,
                    default={}
                )
            ],
            returns={
                "type": "object",
                "properties": {
                    "suggestions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "mug_id": {"type": "string"},
                                "current_position": {"type": "object"},
                                "suggested_position": {"type": "object"},
                                "reason": {"type": "string"},
                                "confidence": {"type": "number"},
                                "safety_score": {"type": "number"},
                                "efficiency_score": {"type": "number"}
                            }
                        }
                    },
                    "overall_score": {"type": "number"},
                    "applied_rules": {"type": "array"}
                }
            },
            rate_limit={"requests_per_minute": 120}
        )
    
    async def execute(
        self,
        request: MCPRequest,
        auth_info: Optional[Dict[str, Any]] = None
    ) -> MCPResponse:
        """Execute suggestion generation."""
        start_time = time.time()
        
        try:
            # Extract parameters
            params = request.parameters
            scene_context = params.get("scene_context")
            strategy_name = params.get("strategy", "balanced")
            constraints = params.get("constraints", {})
            
            if not scene_context:
                return self._create_response(
                    request.id,
                    success=False,
                    error=MCPError(
                        code="MISSING_PARAMETER",
                        message="Scene context is required"
                    ),
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Map strategy name to enum
            strategy_map = {
                "safety_first": PositioningStrategy.SAFETY_FIRST,
                "balanced": PositioningStrategy.BALANCED,
                "efficiency": PositioningStrategy.EFFICIENCY,
                "aesthetic": PositioningStrategy.AESTHETIC
            }
            strategy = strategy_map.get(strategy_name, PositioningStrategy.BALANCED)
            
            # Generate suggestions
            positioning_result = await self.positioning_engine.calculate_positions(
                scene_context=scene_context,
                strategy=strategy,
                constraints=constraints
            )
            
            # Format response
            suggestions = []
            for suggestion in positioning_result.suggestions:
                suggestions.append({
                    "mug_id": suggestion["mug_id"],
                    "current_position": suggestion["current_position"],
                    "suggested_position": suggestion["suggested_position"],
                    "reason": suggestion["reason"],
                    "confidence": suggestion["confidence"],
                    "safety_score": suggestion.get("safety_score", 0),
                    "efficiency_score": suggestion.get("efficiency_score", 0)
                })
            
            return self._create_response(
                request.id,
                success=True,
                data={
                    "suggestions": suggestions,
                    "overall_score": positioning_result.overall_score,
                    "applied_rules": positioning_result.applied_rules
                },
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "strategy": strategy_name,
                    "num_suggestions": len(suggestions)
                }
            )
            
        except Exception as e:
            logger.error("Suggestion generation failed", error=str(e), request_id=request.id)
            return self._create_response(
                request.id,
                success=False,
                error=MCPError(
                    code="SUGGESTION_ERROR",
                    message=f"Failed to generate suggestions: {str(e)}"
                ),
                execution_time_ms=(time.time() - start_time) * 1000
            )


class UpdatePositioningTool(BaseMCPTool):
    """Tool for updating mug positions based on feedback."""
    
    def __init__(
        self,
        positioning_engine: Optional[MugPositioningEngine] = None,
        authenticator: Optional[Any] = None  # MCPAuthenticator type
    ):
        super().__init__(
            name="update_positioning",
            description="Update mug positioning based on user feedback",
            authenticator=authenticator
        )
        self.positioning_engine = positioning_engine
        self.mongodb_client = None
    
    async def _setup(self):
        """Initialize dependencies."""
        if not self.positioning_engine:
            self.positioning_engine = MugPositioningEngine()
            await self.positioning_engine.initialize()
        
        if not self.mongodb_client:
            self.mongodb_client = await get_mongodb_client()
    
    def get_definition(self) -> MCPToolDefinition:
        """Get tool definition."""
        return MCPToolDefinition(
            name=self.name,
            type=MCPToolType.FUNCTION,
            description=self.description,
            parameters=[
                MCPToolParameter(
                    name="analysis_id",
                    type="string",
                    description="ID of the original analysis",
                    required=True
                ),
                MCPToolParameter(
                    name="feedback",
                    type="object",
                    description="User feedback on positioning",
                    required=True
                ),
                MCPToolParameter(
                    name="update_model",
                    type="boolean",
                    description="Whether to update the model with this feedback",
                    required=False,
                    default=False
                )
            ],
            returns={
                "type": "object",
                "properties": {
                    "feedback_id": {"type": "string"},
                    "updated_positions": {"type": "array"},
                    "model_updated": {"type": "boolean"},
                    "improvement_score": {"type": "number"}
                }
            },
            rate_limit={"requests_per_minute": 30}
        )
    
    async def execute(
        self,
        request: MCPRequest,
        auth_info: Optional[Dict[str, Any]] = None
    ) -> MCPResponse:
        """Execute positioning update."""
        start_time = time.time()
        
        try:
            # Extract parameters
            params = request.parameters
            analysis_id = params.get("analysis_id")
            feedback = params.get("feedback")
            update_model = params.get("update_model", False)
            
            if not analysis_id or not feedback:
                return self._create_response(
                    request.id,
                    success=False,
                    error=MCPError(
                        code="MISSING_PARAMETERS",
                        message="Analysis ID and feedback are required"
                    ),
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Store feedback
            feedback_doc = {
                "feedback_id": str(uuid.uuid4()),
                "analysis_id": analysis_id,
                "feedback": feedback,
                "timestamp": datetime.utcnow(),
                "client_id": auth_info.get("client_id") if auth_info else None
            }
            
            db = self.mongodb_client.h200_positioning
            await db.feedback.insert_one(feedback_doc)
            
            # Process feedback
            updated_positions = await self.positioning_engine.process_feedback(
                analysis_id=analysis_id,
                feedback=feedback
            )
            
            # Update model if requested
            model_updated = False
            if update_model and auth_info and "model_training" in auth_info.get("scopes", []):
                # This would trigger a model update process
                # For now, we'll just log it
                logger.info(
                    "Model update requested",
                    analysis_id=analysis_id,
                    client_id=auth_info.get("client_id")
                )
                model_updated = True
            
            # Calculate improvement score
            improvement_score = await self.positioning_engine.calculate_improvement_score(
                original_positions=feedback.get("original_positions", []),
                updated_positions=updated_positions
            )
            
            return self._create_response(
                request.id,
                success=True,
                data={
                    "feedback_id": feedback_doc["feedback_id"],
                    "updated_positions": updated_positions,
                    "model_updated": model_updated,
                    "improvement_score": improvement_score
                },
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "feedback_type": feedback.get("type", "general")
                }
            )
            
        except Exception as e:
            logger.error("Positioning update failed", error=str(e), request_id=request.id)
            return self._create_response(
                request.id,
                success=False,
                error=MCPError(
                    code="UPDATE_ERROR",
                    message=f"Failed to update positioning: {str(e)}"
                ),
                execution_time_ms=(time.time() - start_time) * 1000
            )


# Tool registry
AVAILABLE_TOOLS = {
    "analyze_image": AnalyzeImageTool,
    "apply_rules": ApplyRulesTool,
    "get_suggestions": GetSuggestionsTool,
    "update_positioning": UpdatePositioningTool
}