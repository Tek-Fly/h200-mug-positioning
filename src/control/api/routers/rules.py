"""Rules management API endpoints."""

import logging
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.control.api.middleware.auth import get_current_user, require_permission
from src.control.api.models.rules import (
    NaturalLanguageRuleRequest,
    NaturalLanguageRuleResponse,
    Rule,
    RuleCreateRequest,
    RuleEvaluationRequest,
    RuleEvaluationResponse,
    RuleUpdateRequest,
)
from src.core.rules.engine import RuleEngine
from src.core.rules.parser import NaturalLanguageParser

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_rule_engine(request: Request) -> RuleEngine:
    """Get rule engine from app state."""
    if not hasattr(request.app.state, "mongodb"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    
    rule_engine = RuleEngine(db=request.app.state.mongodb)
    await rule_engine.initialize()
    return rule_engine


@router.post("/rules/natural-language", response_model=NaturalLanguageRuleResponse)
async def create_rule_from_natural_language(
    request: Request,
    nl_request: NaturalLanguageRuleRequest,
    current_user: str = Depends(get_current_user),
) -> NaturalLanguageRuleResponse:
    """
    Create or update a positioning rule from natural language description.
    
    Examples:
    - "The mug should be centered on the coaster"
    - "Keep at least 2 inches between mugs"
    - "Alert if mug is too close to the edge"
    """
    try:
        rule_engine = await get_rule_engine(request)
        parser = NaturalLanguageParser()
        
        # Parse natural language
        logger.info(f"Parsing natural language rule: {nl_request.text}")
        parsed_rule = parser.parse(nl_request.text, context=nl_request.context)
        
        # Create rule object
        rule = Rule(
            id=str(uuid4()),
            name=parsed_rule["name"],
            description=parsed_rule.get("description", nl_request.text),
            type=parsed_rule["type"],
            priority=parsed_rule.get("priority", "medium"),
            conditions=parsed_rule["conditions"],
            actions=parsed_rule["actions"],
            enabled=nl_request.auto_enable,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={
                "created_by": current_user,
                "natural_language_source": nl_request.text,
                "parser_confidence": parsed_rule.get("confidence", 0.8),
            },
        )
        
        # Store rule
        await rule_engine.create_rule(rule.model_dump())
        
        # Generate interpretation
        interpretation = f"Created {rule.type} rule: {rule.name}"
        if rule.conditions:
            interpretation += f" with {len(rule.conditions)} condition(s)"
        
        # Check for warnings
        warnings = []
        if parsed_rule.get("confidence", 1.0) < 0.7:
            warnings.append("Low confidence in interpretation, please verify the rule")
        
        if not rule.conditions:
            warnings.append("No specific conditions were identified")
        
        return NaturalLanguageRuleResponse(
            rule=rule,
            interpretation=interpretation,
            confidence=parsed_rule.get("confidence", 0.8),
            warnings=warnings,
        )
        
    except Exception as e:
        logger.error(f"Error creating rule from natural language: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create rule: {str(e)}",
        )


@router.get("/rules", response_model=List[Rule])
async def list_rules(
    request: Request,
    enabled_only: bool = False,
    rule_type: Optional[str] = None,
    current_user: str = Depends(get_current_user),
) -> List[Rule]:
    """List all positioning rules."""
    try:
        rule_engine = await get_rule_engine(request)
        
        # Build filter
        filter_dict = {}
        if enabled_only:
            filter_dict["enabled"] = True
        if rule_type:
            filter_dict["type"] = rule_type
        
        # Get rules
        rules_data = await rule_engine.list_rules(filter_dict)
        
        # Convert to models
        rules = [Rule(**rule_data) for rule_data in rules_data]
        
        return rules
        
    except Exception as e:
        logger.error(f"Error listing rules: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list rules: {str(e)}",
        )


@router.get("/rules/{rule_id}", response_model=Rule)
async def get_rule(
    request: Request,
    rule_id: str,
    current_user: str = Depends(get_current_user),
) -> Rule:
    """Get a specific rule by ID."""
    try:
        rule_engine = await get_rule_engine(request)
        rule_data = await rule_engine.get_rule(rule_id)
        
        if not rule_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Rule {rule_id} not found",
            )
        
        return Rule(**rule_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting rule: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get rule: {str(e)}",
        )


@router.post("/rules", response_model=Rule)
async def create_rule(
    request: Request,
    rule_request: RuleCreateRequest,
    current_user: str = Depends(get_current_user),
) -> Rule:
    """Create a new positioning rule."""
    try:
        rule_engine = await get_rule_engine(request)
        
        # Create rule
        rule = Rule(
            id=str(uuid4()),
            name=rule_request.name,
            description=rule_request.description,
            type=rule_request.type,
            priority=rule_request.priority,
            conditions=rule_request.conditions,
            actions=rule_request.actions,
            enabled=rule_request.enabled,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={
                **rule_request.metadata,
                "created_by": current_user,
            },
        )
        
        # Store rule
        await rule_engine.create_rule(rule.model_dump())
        
        return rule
        
    except Exception as e:
        logger.error(f"Error creating rule: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create rule: {str(e)}",
        )


@router.patch("/rules/{rule_id}", response_model=Rule)
async def update_rule(
    request: Request,
    rule_id: str,
    update_request: RuleUpdateRequest,
    current_user: str = Depends(get_current_user),
) -> Rule:
    """Update an existing rule."""
    try:
        rule_engine = await get_rule_engine(request)
        
        # Get existing rule
        existing_rule = await rule_engine.get_rule(rule_id)
        if not existing_rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Rule {rule_id} not found",
            )
        
        # Build update
        update_data = update_request.model_dump(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        if "metadata" in update_data:
            update_data["metadata"]["updated_by"] = current_user
        
        # Update rule
        updated_rule = await rule_engine.update_rule(rule_id, update_data)
        
        return Rule(**updated_rule)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating rule: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update rule: {str(e)}",
        )


@router.delete("/rules/{rule_id}")
async def delete_rule(
    request: Request,
    rule_id: str,
    current_user: str = Depends(get_current_user),
) -> dict:
    """Delete a rule."""
    try:
        rule_engine = await get_rule_engine(request)
        
        # Check if rule exists
        existing_rule = await rule_engine.get_rule(rule_id)
        if not existing_rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Rule {rule_id} not found",
            )
        
        # Delete rule
        await rule_engine.delete_rule(rule_id)
        
        return {
            "success": True,
            "message": f"Rule {rule_id} deleted successfully",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting rule: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete rule: {str(e)}",
        )


@router.post("/rules/evaluate", response_model=RuleEvaluationResponse)
async def evaluate_rules(
    request: Request,
    evaluation_request: RuleEvaluationRequest,
    current_user: str = Depends(get_current_user),
) -> RuleEvaluationResponse:
    """Evaluate rules against provided data."""
    try:
        rule_engine = await get_rule_engine(request)
        
        # Get rules to evaluate
        if evaluation_request.rule_ids:
            rules = []
            for rule_id in evaluation_request.rule_ids:
                rule = await rule_engine.get_rule(rule_id)
                if rule:
                    rules.append(rule)
        else:
            # Get all rules
            filter_dict = {} if evaluation_request.include_disabled else {"enabled": True}
            rules = await rule_engine.list_rules(filter_dict)
        
        # Evaluate each rule
        results = []
        matched_count = 0
        
        for rule in rules:
            result = await rule_engine.evaluate_rule(rule["id"], evaluation_request.data)
            results.append(result)
            
            if result.matched:
                matched_count += 1
        
        # Build summary
        summary = {
            "total_rules": len(rules),
            "matched_rules": matched_count,
            "match_rate": matched_count / len(rules) if rules else 0,
        }
        
        return RuleEvaluationResponse(
            evaluated_count=len(rules),
            matched_count=matched_count,
            results=results,
            summary=summary,
        )
        
    except Exception as e:
        logger.error(f"Error evaluating rules: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate rules: {str(e)}",
        )