"""
Rule executor for applying positioning rules.

Executes parsed rules against positioning results and scene context.
"""

# Standard library imports
import time
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import structlog

# First-party imports
from src.core.positioning import MugPosition, PositioningResult

# Local imports
from .models import ActionType, ConditionOperator, Rule, RuleAction, RuleExecutionResult

# Initialize logger
logger = structlog.get_logger(__name__)


class RuleExecutor:
    """
    Executes positioning rules against detected objects and scene context.

    Features:
    - Condition evaluation with spatial reasoning
    - Action execution with conflict resolution
    - Priority-based rule ordering
    - Performance tracking
    """

    def __init__(self, conflict_resolution_strategy: str = "priority"):
        """
        Initialize rule executor.

        Args:
            conflict_resolution_strategy: How to resolve conflicting rules
                - "priority": Higher priority rules override
                - "combine": Try to satisfy all rules
                - "user": Ask user for preference
        """
        self.conflict_resolution_strategy = conflict_resolution_strategy
        self.execution_cache = {}

        logger.info("RuleExecutor initialized", strategy=conflict_resolution_strategy)

    async def execute_rules(
        self,
        rules: List[Rule],
        positioning_result: PositioningResult,
        scene_context: Dict[str, Any],
        image_size: Tuple[int, int],
    ) -> Tuple[PositioningResult, List[RuleExecutionResult]]:
        """
        Execute rules against positioning result.

        Args:
            rules: List of rules to execute
            positioning_result: Current positioning result
            scene_context: Scene analysis context
            image_size: Image dimensions (width, height)

        Returns:
            Updated positioning result and execution results
        """
        start_time = time.time()
        execution_results = []

        # Filter enabled rules and sort by priority
        active_rules = [r for r in rules if r.enabled]
        active_rules.sort(key=lambda r: r.priority, reverse=True)

        # Build execution context
        context = self._build_execution_context(
            positioning_result, scene_context, image_size
        )

        # Track modifications
        modified_positions = list(positioning_result.positions)
        applied_rules = list(positioning_result.applied_rules)

        # Execute each rule
        for rule in active_rules:
            try:
                # Check if rule conditions are met
                if self._evaluate_rule_conditions(rule, context):
                    # Execute rule actions
                    result = await self._execute_rule_actions(
                        rule, modified_positions, context, image_size
                    )

                    if result.executed:
                        applied_rules.append(rule.name)

                    execution_results.append(result)

                    # Update context with modifications
                    context = self._update_context(context, modified_positions)

                else:
                    # Rule conditions not met
                    execution_results.append(
                        RuleExecutionResult(
                            rule_id=rule.id or rule.name,
                            executed=False,
                            metadata={"reason": "conditions_not_met"},
                        )
                    )

            except Exception as e:
                logger.error(
                    "Failed to execute rule",
                    rule_id=rule.id,
                    rule_name=rule.name,
                    error=str(e),
                )
                execution_results.append(
                    RuleExecutionResult(
                        rule_id=rule.id or rule.name,
                        executed=False,
                        metadata={"error": str(e)},
                    )
                )

        # Resolve conflicts if any
        if self.conflict_resolution_strategy != "combine":
            modified_positions = self._resolve_conflicts(
                modified_positions, execution_results
            )

        # Create updated positioning result
        updated_result = PositioningResult(
            positions=modified_positions,
            scene_analysis=positioning_result.scene_analysis,
            applied_rules=applied_rules,
            conflicts_resolved=positioning_result.conflicts_resolved,
            overall_confidence=self._calculate_confidence(modified_positions),
            recommendations=self._generate_recommendations(
                modified_positions, execution_results
            ),
        )

        # Log execution summary
        execution_time = time.time() - start_time
        logger.info(
            "Rules executed",
            num_rules=len(active_rules),
            num_executed=sum(1 for r in execution_results if r.executed),
            execution_time=execution_time,
        )

        return updated_result, execution_results

    def _build_execution_context(
        self,
        positioning_result: PositioningResult,
        scene_context: Dict[str, Any],
        image_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Build context for rule execution."""
        context = {
            "positions": positioning_result.positions,
            "scene": scene_context,
            "image_width": image_size[0],
            "image_height": image_size[1],
            "objects": {},
            "surfaces": [],
            "relationships": {},
        }

        # Extract objects by type
        for pos in positioning_result.positions:
            obj_type = pos.metadata.get("detection", {}).get("class", "mug")
            if obj_type not in context["objects"]:
                context["objects"][obj_type] = []
            context["objects"][obj_type].append(pos)

        # Extract surfaces
        for surface in scene_context.get("surface_areas", []):
            context["surfaces"].append(
                {
                    "type": surface["type"],
                    "bbox": surface["bbox"],
                    "confidence": surface["confidence"],
                }
            )

        # Calculate spatial relationships
        context["relationships"] = self._calculate_relationships(
            positioning_result.positions, scene_context
        )

        return context

    def _evaluate_rule_conditions(self, rule: Rule, context: Dict[str, Any]) -> bool:
        """Evaluate if rule conditions are met."""
        if not rule.conditions:
            return True

        for condition in rule.conditions:
            if not self._evaluate_single_condition(condition, context):
                return False

        return True

    def _evaluate_single_condition(
        self, condition: Any, context: Dict[str, Any]
    ) -> bool:
        """Evaluate a single condition."""
        subject = condition.subject.lower()
        operator = condition.operator
        target = condition.target
        value = condition.value

        # Get subject objects
        subject_objects = context["objects"].get(subject, [])
        if not subject_objects and subject in ["mug", "cup", "beverage", "drink"]:
            # Try common aliases
            for alias in ["mug", "cup", "beverage", "drink"]:
                if alias in context["objects"]:
                    subject_objects = context["objects"][alias]
                    break

        # Handle different operators
        if operator == ConditionOperator.NEAR:
            # Check if any subject is near target
            target_objects = context["objects"].get(target.lower(), [])
            if not target_objects:
                return False

            for subj in subject_objects:
                for targ in target_objects:
                    distance = self._calculate_distance(subj, targ)
                    if distance < (value or 200):  # Default 200 pixels
                        return True
            return False

        elif operator == ConditionOperator.FAR_FROM:
            # Check if all subjects are far from target
            target_objects = context["objects"].get(target.lower(), [])
            if not target_objects:
                return True  # No target means condition is met

            for subj in subject_objects:
                for targ in target_objects:
                    distance = self._calculate_distance(subj, targ)
                    if distance < (value or 200):
                        return False
            return True

        elif operator == ConditionOperator.ON:
            # Check if subject is on target surface
            surfaces = [s for s in context["surfaces"] if s["type"] == target.lower()]
            if not surfaces:
                return False

            for subj in subject_objects:
                for surface in surfaces:
                    if self._is_on_surface(subj, surface):
                        return True
            return False

        elif operator == ConditionOperator.CONTAINS:
            # Check if scene contains target
            return target.lower() in context["objects"]

        elif operator == ConditionOperator.EQUALS:
            # Check equality
            if subject == "scene_type":
                return context["scene"].get("scene_type") == target
            elif subject == "num_mugs":
                return len(subject_objects) == int(target)

        elif operator == ConditionOperator.GREATER_THAN:
            # Numeric comparison
            if subject == "num_mugs":
                return len(subject_objects) > int(target)

        return False

    async def _execute_rule_actions(
        self,
        rule: Rule,
        positions: List[MugPosition],
        context: Dict[str, Any],
        image_size: Tuple[int, int],
    ) -> RuleExecutionResult:
        """Execute rule actions."""
        start_time = time.time()
        actions_taken = []
        conflicts = []

        try:
            for action in rule.actions:
                # Find target positions
                target_positions = self._find_target_positions(
                    action.target, positions, context
                )

                if not target_positions:
                    continue

                # Execute action based on type
                if action.type == ActionType.MAINTAIN_DISTANCE:
                    success = self._execute_maintain_distance(
                        target_positions, action.parameters, context, image_size
                    )

                elif action.type == ActionType.PLACE_ON:
                    success = self._execute_place_on(
                        target_positions, action.parameters, context
                    )

                elif action.type == ActionType.MOVE:
                    success = self._execute_move(
                        target_positions, action.parameters, image_size
                    )

                elif action.type == ActionType.GROUP:
                    success = self._execute_group(
                        target_positions, action.parameters, context
                    )

                elif action.type == ActionType.AVOID_AREA:
                    success = self._execute_avoid_area(
                        target_positions, action.parameters, context, image_size
                    )

                else:
                    success = False
                    logger.warning(
                        "Unsupported action type", action_type=action.type.value
                    )

                if success:
                    actions_taken.append(
                        {
                            "type": action.type.value,
                            "target": action.target,
                            "parameters": action.parameters,
                        }
                    )

        except Exception as e:
            logger.error("Error executing rule actions", rule_id=rule.id, error=str(e))
            conflicts.append(f"Execution error: {str(e)}")

        execution_time = time.time() - start_time

        return RuleExecutionResult(
            rule_id=rule.id or rule.name,
            executed=len(actions_taken) > 0,
            actions_taken=actions_taken,
            conflicts=conflicts,
            confidence=0.9 if actions_taken else 0.0,
            execution_time=execution_time,
            metadata={"rule_name": rule.name},
        )

    def _execute_maintain_distance(
        self,
        positions: List[MugPosition],
        parameters: Dict[str, Any],
        context: Dict[str, Any],
        image_size: Tuple[int, int],
    ) -> bool:
        """Execute maintain distance action."""
        min_distance = parameters.get("distance", 200)
        from_object = parameters.get("from", "laptop")

        # Find objects to maintain distance from
        avoid_objects = context["objects"].get(from_object.lower(), [])
        if not avoid_objects:
            return True  # No objects to avoid

        modified = False
        for pos in positions:
            for avoid in avoid_objects:
                distance = self._calculate_distance(pos, avoid)

                if distance < min_distance:
                    # Move position away
                    direction = self._calculate_direction(pos, avoid)
                    move_distance = min_distance - distance + 20  # Add buffer

                    # Calculate new position
                    new_x = pos.x + direction[0] * move_distance
                    new_y = pos.y + direction[1] * move_distance

                    # Ensure within bounds
                    margin = 50
                    new_x = max(margin, min(new_x, image_size[0] - margin))
                    new_y = max(margin, min(new_y, image_size[1] - margin))

                    pos.x = new_x
                    pos.y = new_y
                    pos.reasoning += (
                        f" | Moved {move_distance:.0f}px from {from_object}"
                    )
                    modified = True

        return modified

    def _execute_place_on(
        self,
        positions: List[MugPosition],
        parameters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> bool:
        """Execute place on surface action."""
        surface_type = parameters.get("surface", "coaster")

        # Find target surfaces
        surfaces = [
            s for s in context["surfaces"] if s["type"].lower() == surface_type.lower()
        ]

        if not surfaces:
            # Check if we have detected objects that could be surfaces
            surface_objects = context["objects"].get(surface_type.lower(), [])
            if surface_objects:
                surfaces = [
                    {
                        "bbox": {
                            "x1": obj.x - obj.width / 2,
                            "y1": obj.y - obj.height / 2,
                            "x2": obj.x + obj.width / 2,
                            "y2": obj.y + obj.height / 2,
                        }
                    }
                    for obj in surface_objects
                ]

        if not surfaces:
            return False

        modified = False
        for i, pos in enumerate(positions):
            # Find nearest suitable surface
            best_surface = min(
                surfaces, key=lambda s: self._distance_to_surface(pos, s)
            )

            # Center on surface
            bbox = best_surface["bbox"]
            center_x = (bbox["x1"] + bbox["x2"]) / 2
            center_y = (bbox["y1"] + bbox["y2"]) / 2

            if abs(pos.x - center_x) > 10 or abs(pos.y - center_y) > 10:
                pos.x = center_x
                pos.y = center_y
                pos.reasoning += f" | Placed on {surface_type}"
                modified = True

        return modified

    def _execute_move(
        self,
        positions: List[MugPosition],
        parameters: Dict[str, Any],
        image_size: Tuple[int, int],
    ) -> bool:
        """Execute move action."""
        direction = parameters.get("direction", "right")
        distance = parameters.get("distance", 100)

        move_vectors = {
            "left": (-1, 0),
            "right": (1, 0),
            "up": (0, -1),
            "down": (0, 1),
            "top": (0, -1),
            "bottom": (0, 1),
        }

        if direction not in move_vectors:
            return False

        vec = move_vectors[direction]

        for pos in positions:
            new_x = pos.x + vec[0] * distance
            new_y = pos.y + vec[1] * distance

            # Ensure within bounds
            margin = 50
            new_x = max(margin, min(new_x, image_size[0] - margin))
            new_y = max(margin, min(new_y, image_size[1] - margin))

            pos.x = new_x
            pos.y = new_y
            pos.reasoning += f" | Moved {direction} by {distance}px"

        return True

    def _execute_group(
        self,
        positions: List[MugPosition],
        parameters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> bool:
        """Execute grouping action."""
        if len(positions) < 2:
            return False

        group_type = parameters.get("type", "tight")
        side = parameters.get("side", "center")

        # Calculate target area based on side
        image_width = context["image_width"]
        image_height = context["image_height"]

        if side == "left":
            target_x = image_width * 0.25
            target_y = image_height * 0.5
        elif side == "right":
            target_x = image_width * 0.75
            target_y = image_height * 0.5
        elif side == "top":
            target_x = image_width * 0.5
            target_y = image_height * 0.25
        elif side == "bottom":
            target_x = image_width * 0.5
            target_y = image_height * 0.75
        else:  # center
            target_x = image_width * 0.5
            target_y = image_height * 0.5

        # Arrange positions around target
        if group_type == "tight":
            spacing = 100
        elif group_type == "loose":
            spacing = 200
        else:
            spacing = 150

        # Simple grid arrangement
        num_cols = int(np.ceil(np.sqrt(len(positions))))

        for i, pos in enumerate(positions):
            row = i // num_cols
            col = i % num_cols

            offset_x = (col - num_cols / 2) * spacing
            offset_y = (row - len(positions) / num_cols / 2) * spacing

            pos.x = target_x + offset_x
            pos.y = target_y + offset_y
            pos.reasoning += f" | Grouped {group_type} on {side}"

        return True

    def _execute_avoid_area(
        self,
        positions: List[MugPosition],
        parameters: Dict[str, Any],
        context: Dict[str, Any],
        image_size: Tuple[int, int],
    ) -> bool:
        """Execute avoid area action."""
        area_type = parameters.get("area", "edge")
        margin = parameters.get("margin", 100)

        modified = False

        for pos in positions:
            if area_type == "edge":
                # Keep away from edges
                old_x, old_y = pos.x, pos.y

                pos.x = max(margin, min(pos.x, image_size[0] - margin))
                pos.y = max(margin, min(pos.y, image_size[1] - margin))

                if old_x != pos.x or old_y != pos.y:
                    pos.reasoning += f" | Moved away from edge"
                    modified = True

            elif area_type == "danger":
                # Keep away from danger zones
                for zone in context["scene"].get("danger_zones", []):
                    if self._position_in_bbox(pos, zone["bbox"]):
                        # Move out of danger zone
                        self._move_out_of_bbox(pos, zone["bbox"], margin)
                        pos.reasoning += f" | Moved out of danger zone"
                        modified = True

        return modified

    def _find_target_positions(
        self, target: str, positions: List[MugPosition], context: Dict[str, Any]
    ) -> List[MugPosition]:
        """Find positions matching target description."""
        target_lower = target.lower()

        # Handle special targets
        if target_lower in ["all", "any", "*"]:
            return positions

        if target_lower in ["mug", "mugs", "cup", "cups", "beverage", "beverages"]:
            # Return all mug positions
            return positions

        # Filter by metadata
        filtered = []
        for pos in positions:
            obj_class = pos.metadata.get("detection", {}).get("class", "mug")
            if obj_class.lower() == target_lower:
                filtered.append(pos)

        return filtered if filtered else positions

    def _calculate_distance(self, pos1: MugPosition, pos2: MugPosition) -> float:
        """Calculate distance between two positions."""
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        return np.sqrt(dx**2 + dy**2)

    def _calculate_direction(
        self, from_pos: MugPosition, to_pos: MugPosition
    ) -> Tuple[float, float]:
        """Calculate normalized direction vector."""
        dx = from_pos.x - to_pos.x
        dy = from_pos.y - to_pos.y
        distance = np.sqrt(dx**2 + dy**2)

        if distance == 0:
            return (1, 0)  # Default right

        return (dx / distance, dy / distance)

    def _is_on_surface(self, position: MugPosition, surface: Dict[str, Any]) -> bool:
        """Check if position is on surface."""
        bbox = surface["bbox"]
        return (
            bbox["x1"] <= position.x <= bbox["x2"]
            and bbox["y1"] <= position.y <= bbox["y2"]
        )

    def _distance_to_surface(
        self, position: MugPosition, surface: Dict[str, Any]
    ) -> float:
        """Calculate distance from position to surface center."""
        bbox = surface["bbox"]
        center_x = (bbox["x1"] + bbox["x2"]) / 2
        center_y = (bbox["y1"] + bbox["y2"]) / 2

        dx = position.x - center_x
        dy = position.y - center_y
        return np.sqrt(dx**2 + dy**2)

    def _position_in_bbox(self, position: MugPosition, bbox: Dict[str, float]) -> bool:
        """Check if position is within bbox."""
        return (
            bbox["x1"] <= position.x <= bbox["x2"]
            and bbox["y1"] <= position.y <= bbox["y2"]
        )

    def _move_out_of_bbox(
        self, position: MugPosition, bbox: Dict[str, float], margin: float
    ):
        """Move position outside of bbox."""
        # Find closest edge
        distances = [
            (position.x - bbox["x1"], "left"),
            (bbox["x2"] - position.x, "right"),
            (position.y - bbox["y1"], "top"),
            (bbox["y2"] - position.y, "bottom"),
        ]

        min_dist, direction = min(distances, key=lambda x: x[0])

        if direction == "left":
            position.x = bbox["x1"] - margin
        elif direction == "right":
            position.x = bbox["x2"] + margin
        elif direction == "top":
            position.y = bbox["y1"] - margin
        elif direction == "bottom":
            position.y = bbox["y2"] + margin

    def _calculate_relationships(
        self, positions: List[MugPosition], scene_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate spatial relationships between objects."""
        relationships = {"distances": {}, "nearest_neighbors": {}, "clusters": []}

        # Calculate pairwise distances
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions[i + 1 :], i + 1):
                distance = self._calculate_distance(pos1, pos2)
                relationships["distances"][(i, j)] = distance

        # Find nearest neighbors
        for i, pos in enumerate(positions):
            if i < len(positions) - 1:
                nearest_idx = min(
                    range(len(positions)),
                    key=lambda j: (
                        relationships["distances"].get(
                            (min(i, j), max(i, j)), float("inf")
                        )
                        if i != j
                        else float("inf")
                    ),
                )
                if nearest_idx != i:
                    relationships["nearest_neighbors"][i] = nearest_idx

        return relationships

    def _update_context(
        self, context: Dict[str, Any], positions: List[MugPosition]
    ) -> Dict[str, Any]:
        """Update context with modified positions."""
        context["positions"] = positions

        # Re-extract objects by type
        context["objects"] = {}
        for pos in positions:
            obj_type = pos.metadata.get("detection", {}).get("class", "mug")
            if obj_type not in context["objects"]:
                context["objects"][obj_type] = []
            context["objects"][obj_type].append(pos)

        # Recalculate relationships
        context["relationships"] = self._calculate_relationships(
            positions, context["scene"]
        )

        return context

    def _resolve_conflicts(
        self, positions: List[MugPosition], execution_results: List[RuleExecutionResult]
    ) -> List[MugPosition]:
        """Resolve conflicts between rule executions."""
        if self.conflict_resolution_strategy == "priority":
            # Already handled by priority ordering
            return positions

        # Additional conflict resolution can be added here
        return positions

    def _calculate_confidence(self, positions: List[MugPosition]) -> float:
        """Calculate overall confidence score."""
        if not positions:
            return 0.0

        confidences = [p.confidence for p in positions]
        return float(np.mean(confidences))

    def _generate_recommendations(
        self, positions: List[MugPosition], execution_results: List[RuleExecutionResult]
    ) -> List[str]:
        """Generate recommendations based on execution results."""
        recommendations = []

        # Check for failed executions
        failed_rules = [
            r
            for r in execution_results
            if not r.executed and r.metadata.get("reason") != "conditions_not_met"
        ]

        if failed_rules:
            recommendations.append(
                f"{len(failed_rules)} rule(s) failed to execute. Check rule configuration."
            )

        # Check for conflicts
        total_conflicts = sum(len(r.conflicts) for r in execution_results)
        if total_conflicts > 0:
            recommendations.append(f"Resolved {total_conflicts} positioning conflicts.")

        # Check position confidence
        low_confidence = [p for p in positions if p.confidence < 0.5]
        if low_confidence:
            recommendations.append(
                f"{len(low_confidence)} position(s) have low confidence. Manual review recommended."
            )

        return recommendations
