"""
Data models for the rule engine.

Defines rule structures, conditions, and actions for positioning rules.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from bson import ObjectId


class RuleType(Enum):
    """Types of positioning rules."""
    DISTANCE = "distance"           # Min/max distance between objects
    PLACEMENT = "placement"         # Where to place objects
    AVOIDANCE = "avoidance"        # Areas/objects to avoid
    GROUPING = "grouping"          # How to group multiple items
    PREFERENCE = "preference"       # User preferences
    SAFETY = "safety"              # Safety-related rules
    CONTEXT = "context"            # Context-dependent rules


class ConditionOperator(Enum):
    """Operators for rule conditions."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    NEAR = "near"
    FAR_FROM = "far_from"
    ON = "on"
    WITHIN = "within"
    OUTSIDE = "outside"


class ActionType(Enum):
    """Types of actions a rule can trigger."""
    MOVE = "move"                  # Move object to location
    MAINTAIN_DISTANCE = "maintain_distance"
    GROUP = "group"                # Group objects together
    SEPARATE = "separate"          # Separate objects
    PLACE_ON = "place_on"          # Place on specific surface
    AVOID_AREA = "avoid_area"      # Avoid specific area
    ROTATE = "rotate"              # Rotate object
    ALERT = "alert"                # Alert user
    SUGGEST = "suggest"            # Suggest position
    ENFORCE = "enforce"            # Enforce position


@dataclass
class RuleCondition:
    """Condition for rule activation."""
    subject: str                   # What to check (e.g., "mug", "laptop")
    operator: ConditionOperator    # How to check
    target: Any                    # What to check against
    value: Optional[Any] = None    # Additional value (e.g., distance)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'subject': self.subject,
            'operator': self.operator.value,
            'target': self.target,
            'value': self.value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuleCondition':
        """Create from dictionary."""
        return cls(
            subject=data['subject'],
            operator=ConditionOperator(data['operator']),
            target=data['target'],
            value=data.get('value'),
            metadata=data.get('metadata', {})
        )


@dataclass
class RuleAction:
    """Action to take when rule conditions are met."""
    type: ActionType               # What to do
    target: str                    # What to act on
    parameters: Dict[str, Any]     # Action parameters
    priority: int = 5              # Action priority (1-10)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'type': self.type.value,
            'target': self.target,
            'parameters': self.parameters,
            'priority': self.priority,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuleAction':
        """Create from dictionary."""
        return cls(
            type=ActionType(data['type']),
            target=data['target'],
            parameters=data['parameters'],
            priority=data.get('priority', 5),
            metadata=data.get('metadata', {})
        )


@dataclass
class Rule:
    """Complete positioning rule."""
    id: Optional[str] = None
    name: str = ""
    description: str = ""
    natural_language: str = ""     # Original NL input
    type: RuleType = RuleType.PLACEMENT
    conditions: List[RuleCondition] = field(default_factory=list)
    actions: List[RuleAction] = field(default_factory=list)
    priority: int = 5              # Rule priority (1-10)
    enabled: bool = True
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        data = {
            'name': self.name,
            'description': self.description,
            'natural_language': self.natural_language,
            'type': self.type.value,
            'conditions': [c.to_dict() for c in self.conditions],
            'actions': [a.to_dict() for a in self.actions],
            'priority': self.priority,
            'enabled': self.enabled,
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'created_by': self.created_by,
            'metadata': self.metadata
        }
        
        if self.id:
            data['_id'] = ObjectId(self.id) if isinstance(self.id, str) else self.id
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Rule':
        """Create from MongoDB document."""
        rule = cls(
            id=str(data.get('_id', '')) if data.get('_id') else None,
            name=data['name'],
            description=data['description'],
            natural_language=data['natural_language'],
            type=RuleType(data['type']),
            conditions=[RuleCondition.from_dict(c) for c in data['conditions']],
            actions=[RuleAction.from_dict(a) for a in data['actions']],
            priority=data['priority'],
            enabled=data['enabled'],
            version=data['version'],
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            created_by=data['created_by'],
            metadata=data.get('metadata', {})
        )
        
        return rule
    
    def matches_conditions(self, context: Dict[str, Any]) -> bool:
        """Check if all conditions are met in given context."""
        if not self.conditions:
            return True
            
        for condition in self.conditions:
            if not self._evaluate_condition(condition, context):
                return False
                
        return True
    
    def _evaluate_condition(
        self,
        condition: RuleCondition,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate a single condition."""
        # This is a simplified evaluation
        # In production, this would be more sophisticated
        
        subject_value = context.get(condition.subject)
        if subject_value is None:
            return False
            
        operator = condition.operator
        target = condition.target
        value = condition.value
        
        if operator == ConditionOperator.EQUALS:
            return subject_value == target
        elif operator == ConditionOperator.NOT_EQUALS:
            return subject_value != target
        elif operator == ConditionOperator.GREATER_THAN:
            return subject_value > target
        elif operator == ConditionOperator.LESS_THAN:
            return subject_value < target
        elif operator == ConditionOperator.CONTAINS:
            return target in subject_value
        elif operator == ConditionOperator.NEAR:
            # Check if subject is near target
            # This would use spatial calculations
            return True  # Placeholder
        elif operator == ConditionOperator.ON:
            # Check if subject is on target surface
            return True  # Placeholder
            
        return False


@dataclass
class RuleValidationResult:
    """Result of rule validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    parsed_rule: Optional[Rule] = None


@dataclass
class RuleExecutionResult:
    """Result of rule execution."""
    rule_id: str
    executed: bool
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    confidence: float = 1.0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)