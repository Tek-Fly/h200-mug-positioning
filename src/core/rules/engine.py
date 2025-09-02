"""
Main rule engine orchestrator.

Coordinates parsing, storage, and execution of positioning rules.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import structlog

from src.core.positioning import PositioningResult, MugPositioningEngine
from .parser import NaturalLanguageRuleParser
from .executor import RuleExecutor
from .storage import RuleStorage
from .models import Rule, RuleValidationResult, RuleExecutionResult, RuleType

# Initialize logger
logger = structlog.get_logger(__name__)


class RuleEngine:
    """
    Main rule engine for natural language positioning rules.
    
    This engine orchestrates:
    - Natural language rule parsing
    - Rule validation and storage
    - Rule execution during positioning
    - Rule management and versioning
    
    Example usage:
        engine = RuleEngine()
        await engine.initialize()
        
        # Add a rule
        result = await engine.add_rule("Keep mugs at least 6 inches from laptops")
        
        # Apply rules to positioning
        positioning_result = await engine.apply_rules(positioning_result, scene_context)
    """
    
    def __init__(
        self,
        positioning_engine: Optional[MugPositioningEngine] = None,
        openai_api_key: Optional[str] = None,
        conflict_resolution: str = "priority"
    ):
        """
        Initialize rule engine.
        
        Args:
            positioning_engine: Mug positioning engine instance
            openai_api_key: OpenAI API key for LangChain
            conflict_resolution: Strategy for resolving rule conflicts
        """
        self.positioning_engine = positioning_engine
        self.parser = NaturalLanguageRuleParser(api_key=openai_api_key)
        self.executor = RuleExecutor(conflict_resolution_strategy=conflict_resolution)
        self.storage = RuleStorage()
        self.rule_cache = {}
        self.initialized = False
        
        logger.info(
            "RuleEngine initialized",
            conflict_resolution=conflict_resolution
        )
    
    async def initialize(self):
        """Initialize all components."""
        if not self.initialized:
            await self.storage.initialize()
            await self._load_active_rules()
            self.initialized = True
            
            logger.info(
                "RuleEngine fully initialized",
                active_rules=len(self.rule_cache)
            )
    
    async def add_rule(
        self,
        natural_language: str,
        user_id: str = "system",
        priority: Optional[int] = None
    ) -> RuleValidationResult:
        """
        Add a new rule from natural language.
        
        Args:
            natural_language: Natural language rule description
            user_id: ID of user adding the rule
            priority: Optional priority override
            
        Returns:
            Validation result with rule ID if successful
        """
        await self.initialize()
        
        try:
            # Parse the rule
            validation_result = await self.parser.parse_rule(natural_language)
            
            if not validation_result.valid:
                return validation_result
            
            rule = validation_result.parsed_rule
            
            # Override priority if specified
            if priority is not None:
                rule.priority = max(1, min(10, priority))
            
            # Store the rule
            rule_id = await self.storage.create_rule(rule, user_id)
            rule.id = rule_id
            
            # Add to cache
            self.rule_cache[rule_id] = rule
            
            # Update validation result
            validation_result.parsed_rule = rule
            
            logger.info(
                "Rule added successfully",
                rule_id=rule_id,
                rule_name=rule.name,
                natural_language=natural_language[:100]
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(
                "Failed to add rule",
                natural_language=natural_language[:100],
                error=str(e)
            )
            return RuleValidationResult(
                valid=False,
                errors=[f"Failed to add rule: {str(e)}"]
            )
    
    async def update_rule(
        self,
        rule_id: str,
        natural_language: str,
        user_id: str = "system"
    ) -> RuleValidationResult:
        """
        Update an existing rule with new natural language.
        
        Args:
            rule_id: ID of rule to update
            natural_language: New natural language description
            user_id: ID of user updating the rule
            
        Returns:
            Validation result
        """
        await self.initialize()
        
        # Parse the new rule
        validation_result = await self.parser.parse_rule(natural_language)
        
        if not validation_result.valid:
            return validation_result
        
        new_rule = validation_result.parsed_rule
        
        # Get existing rule for comparison
        existing_rule = await self.storage.get_rule(rule_id)
        if not existing_rule:
            return RuleValidationResult(
                valid=False,
                errors=["Rule not found"]
            )
        
        # Update fields
        updates = {
            "name": new_rule.name,
            "description": new_rule.description,
            "natural_language": natural_language,
            "type": new_rule.type.value,
            "conditions": [c.to_dict() for c in new_rule.conditions],
            "actions": [a.to_dict() for a in new_rule.actions],
            "priority": new_rule.priority
        }
        
        # Store update
        success = await self.storage.update_rule(rule_id, updates, user_id)
        
        if success:
            # Update cache
            new_rule.id = rule_id
            new_rule.version = existing_rule.version + 1
            self.rule_cache[rule_id] = new_rule
            
            logger.info(
                "Rule updated successfully",
                rule_id=rule_id,
                version=new_rule.version
            )
        
        return validation_result
    
    async def delete_rule(
        self,
        rule_id: str,
        user_id: str = "system",
        permanent: bool = False
    ) -> bool:
        """
        Delete a rule.
        
        Args:
            rule_id: ID of rule to delete
            user_id: ID of user deleting the rule
            permanent: If True, permanently delete (vs soft delete)
            
        Returns:
            True if successful
        """
        await self.initialize()
        
        success = await self.storage.delete_rule(
            rule_id,
            user_id,
            soft_delete=not permanent
        )
        
        if success:
            # Remove from cache
            self.rule_cache.pop(rule_id, None)
            
            logger.info(
                "Rule deleted",
                rule_id=rule_id,
                permanent=permanent
            )
        
        return success
    
    async def get_rule(self, rule_id: str) -> Optional[Rule]:
        """
        Get a rule by ID.
        
        Args:
            rule_id: Rule ID
            
        Returns:
            Rule object or None
        """
        await self.initialize()
        
        # Check cache first
        if rule_id in self.rule_cache:
            return self.rule_cache[rule_id]
        
        # Load from storage
        rule = await self.storage.get_rule(rule_id)
        if rule:
            self.rule_cache[rule_id] = rule
        
        return rule
    
    async def list_rules(
        self,
        rule_type: Optional[RuleType] = None,
        enabled_only: bool = True,
        limit: int = 100
    ) -> List[Rule]:
        """
        List rules with optional filtering.
        
        Args:
            rule_type: Filter by rule type
            enabled_only: Only return enabled rules
            limit: Maximum number of rules
            
        Returns:
            List of rules
        """
        await self.initialize()
        
        filter_dict = {}
        if rule_type:
            filter_dict["type"] = rule_type.value
        if enabled_only:
            filter_dict["enabled"] = True
        
        return await self.storage.list_rules(
            filter_dict=filter_dict,
            limit=limit
        )
    
    async def apply_rules(
        self,
        positioning_result: PositioningResult,
        scene_context: Dict[str, Any],
        image: Image.Image,
        rule_ids: Optional[List[str]] = None
    ) -> Tuple[PositioningResult, List[RuleExecutionResult]]:
        """
        Apply rules to positioning result.
        
        Args:
            positioning_result: Current positioning result
            scene_context: Scene analysis context
            image: Original image
            rule_ids: Specific rule IDs to apply (None = all active)
            
        Returns:
            Updated positioning result and execution results
        """
        await self.initialize()
        
        # Get rules to apply
        if rule_ids:
            rules = [
                await self.get_rule(rid)
                for rid in rule_ids
            ]
            rules = [r for r in rules if r]  # Filter None values
        else:
            # Use all active rules from cache
            rules = [
                rule for rule in self.rule_cache.values()
                if rule.enabled
            ]
        
        if not rules:
            logger.warning("No rules to apply")
            return positioning_result, []
        
        # Execute rules
        updated_result, execution_results = await self.executor.execute_rules(
            rules,
            positioning_result,
            scene_context,
            image.size
        )
        
        # Log summary
        executed_count = sum(1 for r in execution_results if r.executed)
        logger.info(
            "Rules applied",
            total_rules=len(rules),
            executed_rules=executed_count,
            confidence=updated_result.overall_confidence
        )
        
        return updated_result, execution_results
    
    async def validate_rule(
        self,
        natural_language: str
    ) -> RuleValidationResult:
        """
        Validate a rule without storing it.
        
        Args:
            natural_language: Natural language rule
            
        Returns:
            Validation result
        """
        return await self.parser.parse_rule(natural_language)
    
    async def batch_add_rules(
        self,
        rule_texts: List[str],
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Add multiple rules in batch.
        
        Args:
            rule_texts: List of natural language rules
            user_id: ID of user adding rules
            
        Returns:
            Summary of batch operation
        """
        await self.initialize()
        
        results = {
            "total": len(rule_texts),
            "successful": 0,
            "failed": 0,
            "rules": [],
            "errors": []
        }
        
        # Parse all rules in parallel
        validation_results = await self.parser.parse_multiple_rules(rule_texts)
        
        # Process results
        for i, (rule_text, validation) in enumerate(zip(rule_texts, validation_results)):
            if validation.valid:
                try:
                    rule_id = await self.storage.create_rule(
                        validation.parsed_rule,
                        user_id
                    )
                    validation.parsed_rule.id = rule_id
                    self.rule_cache[rule_id] = validation.parsed_rule
                    
                    results["successful"] += 1
                    results["rules"].append({
                        "index": i,
                        "rule_id": rule_id,
                        "name": validation.parsed_rule.name
                    })
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append({
                        "index": i,
                        "text": rule_text[:100],
                        "error": str(e)
                    })
            else:
                results["failed"] += 1
                results["errors"].append({
                    "index": i,
                    "text": rule_text[:100],
                    "error": validation.errors[0] if validation.errors else "Unknown error"
                })
        
        return results
    
    async def get_rule_suggestions(
        self,
        scene_context: Dict[str, Any]
    ) -> List[str]:
        """
        Get rule suggestions based on scene context.
        
        Args:
            scene_context: Current scene analysis
            
        Returns:
            List of suggested rules in natural language
        """
        suggestions = []
        
        # Check for common scenarios
        objects = scene_context.get('detected_objects', {})
        scene_type = scene_context.get('scene_type', 'unknown')
        
        # Office scenario
        if scene_type == 'office' or 'laptop' in objects:
            suggestions.extend([
                "Keep beverages at least 6 inches away from electronic devices",
                "Place mugs on coasters when available",
                "Position drinks on the right side of the workspace"
            ])
        
        # Kitchen scenario
        if scene_type == 'kitchen' or 'stove' in objects:
            suggestions.extend([
                "Keep mugs away from hot surfaces and cooking areas",
                "Place beverages on designated counter areas",
                "Avoid placing drinks near the sink splash zone"
            ])
        
        # Multiple mugs
        if objects.get('mug', 0) > 1:
            suggestions.extend([
                "Group all beverages together for easier access",
                "Maintain at least 4 inches between each mug",
                "Arrange mugs in a single row when space is limited"
            ])
        
        # Edge detection
        if any('edge' in zone.get('type', '') for zone in scene_context.get('danger_zones', [])):
            suggestions.append("Keep all beverages at least 3 inches from table edges")
        
        return suggestions
    
    async def export_rules(
        self,
        format: str = "json",
        include_disabled: bool = False
    ) -> Any:
        """
        Export rules in specified format.
        
        Args:
            format: Export format (json, yaml, etc.)
            include_disabled: Include disabled rules
            
        Returns:
            Exported rules data
        """
        await self.initialize()
        
        filter_dict = {} if include_disabled else {"enabled": True}
        rules_data = await self.storage.export_rules(filter_dict)
        
        if format == "json":
            return rules_data
        else:
            # Add other formats as needed
            raise ValueError(f"Unsupported export format: {format}")
    
    async def import_rules(
        self,
        rules_data: List[Dict[str, Any]],
        user_id: str = "system",
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Import rules from exported data.
        
        Args:
            rules_data: Exported rules data
            user_id: ID of user importing rules
            overwrite: Whether to overwrite existing rules
            
        Returns:
            Import summary
        """
        await self.initialize()
        
        result = await self.storage.import_rules(
            rules_data,
            user_id,
            overwrite
        )
        
        # Reload cache
        await self._load_active_rules()
        
        return result
    
    async def _load_active_rules(self):
        """Load active rules into cache."""
        active_rules = await self.storage.get_active_rules()
        
        self.rule_cache = {
            rule.id: rule
            for rule in active_rules
            if rule.id
        }
        
        logger.info(
            "Active rules loaded",
            count=len(self.rule_cache)
        )
    
    async def analyze_rule_performance(
        self,
        rule_id: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze performance metrics for a rule.
        
        Args:
            rule_id: Rule ID to analyze
            limit: Number of recent executions to analyze
            
        Returns:
            Performance analysis
        """
        # This would connect to execution logs in production
        # For now, return mock analysis
        return {
            "rule_id": rule_id,
            "execution_count": 0,
            "success_rate": 0.0,
            "average_execution_time": 0.0,
            "common_conflicts": [],
            "recommendation": "Insufficient data for analysis"
        }