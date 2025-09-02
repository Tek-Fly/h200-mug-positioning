"""Models for rules management endpoints."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, ConfigDict


class RuleType(str, Enum):
    """Rule types."""
    POSITION = "position"
    SPACING = "spacing"
    ALIGNMENT = "alignment"
    CUSTOM = "custom"


class RulePriority(str, Enum):
    """Rule priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RuleCondition(BaseModel):
    """Rule condition specification."""
    field: str = Field(..., description="Field to check (e.g., 'position.x')")
    operator: str = Field(..., description="Comparison operator (e.g., '>', '==', 'in')")
    value: Any = Field(..., description="Value to compare against")
    

class RuleAction(BaseModel):
    """Rule action specification."""
    type: str = Field(..., description="Action type (e.g., 'adjust', 'alert', 'reject')")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")


class Rule(BaseModel):
    """Positioning rule."""
    model_config = ConfigDict(from_attributes=True)
    
    id: str = Field(..., description="Unique rule ID")
    name: str = Field(..., description="Rule name")
    description: Optional[str] = Field(None, description="Rule description")
    type: RuleType = Field(..., description="Rule type")
    priority: RulePriority = Field(default=RulePriority.MEDIUM, description="Rule priority")
    
    conditions: List[RuleCondition] = Field(..., description="Rule conditions")
    actions: List[RuleAction] = Field(..., description="Actions to take when rule matches")
    
    enabled: bool = Field(default=True, description="Whether rule is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class NaturalLanguageRuleRequest(BaseModel):
    """Request to create/update rule from natural language."""
    text: str = Field(..., description="Natural language rule description")
    context: Optional[str] = Field(None, description="Additional context")
    auto_enable: bool = Field(default=False, description="Automatically enable the rule")


class NaturalLanguageRuleResponse(BaseModel):
    """Response for natural language rule creation."""
    rule: Rule = Field(..., description="Created/updated rule")
    interpretation: str = Field(..., description="How the system interpreted the request")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Interpretation confidence")
    warnings: List[str] = Field(default_factory=list, description="Any warnings or ambiguities")


class RuleCreateRequest(BaseModel):
    """Request to create a rule."""
    name: str = Field(..., description="Rule name")
    description: Optional[str] = Field(None, description="Rule description")
    type: RuleType = Field(..., description="Rule type")
    priority: RulePriority = Field(default=RulePriority.MEDIUM, description="Rule priority")
    conditions: List[RuleCondition] = Field(..., description="Rule conditions")
    actions: List[RuleAction] = Field(..., description="Actions to take")
    enabled: bool = Field(default=True, description="Whether to enable immediately")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RuleUpdateRequest(BaseModel):
    """Request to update a rule."""
    name: Optional[str] = Field(None, description="New rule name")
    description: Optional[str] = Field(None, description="New description")
    priority: Optional[RulePriority] = Field(None, description="New priority")
    conditions: Optional[List[RuleCondition]] = Field(None, description="New conditions")
    actions: Optional[List[RuleAction]] = Field(None, description="New actions")
    enabled: Optional[bool] = Field(None, description="Enable/disable rule")
    metadata: Optional[Dict[str, Any]] = Field(None, description="New metadata")


class RuleEvaluationRequest(BaseModel):
    """Request to evaluate rules against data."""
    data: Dict[str, Any] = Field(..., description="Data to evaluate against rules")
    rule_ids: Optional[List[str]] = Field(None, description="Specific rules to evaluate")
    include_disabled: bool = Field(default=False, description="Include disabled rules")


class RuleEvaluationResult(BaseModel):
    """Result of rule evaluation."""
    rule_id: str = Field(..., description="Rule ID")
    rule_name: str = Field(..., description="Rule name")
    matched: bool = Field(..., description="Whether rule conditions matched")
    actions_taken: List[RuleAction] = Field(..., description="Actions that were/would be taken")
    details: Dict[str, Any] = Field(default_factory=dict, description="Evaluation details")


class RuleEvaluationResponse(BaseModel):
    """Response for rule evaluation."""
    evaluated_count: int = Field(..., description="Number of rules evaluated")
    matched_count: int = Field(..., description="Number of rules that matched")
    results: List[RuleEvaluationResult] = Field(..., description="Individual evaluation results")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics")