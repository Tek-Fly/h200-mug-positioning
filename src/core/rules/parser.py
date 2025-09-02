"""Natural language parser for positioning rules."""

# Standard library imports
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NaturalLanguageParser:
    """Parse natural language into positioning rules."""

    def __init__(self):
        """Initialize parser with patterns."""
        self.position_patterns = {
            r"center(?:ed)?": "center",
            r"left(?: edge)?": "left_edge",
            r"right(?: edge)?": "right_edge",
            r"top(?: edge)?": "top_edge",
            r"bottom(?: edge)?": "bottom_edge",
            r"corner": "corner",
            r"edge": "edge",
        }

        self.spacing_patterns = {
            r"(\d+)\s*(?:inches?|in|\")": "inches",
            r"(\d+)\s*(?:millimeters?|mm)": "mm",
            r"(\d+)\s*(?:centimeters?|cm)": "cm",
            r"(\d+)\s*(?:pixels?|px)?": "pixels",
        }

        self.action_patterns = {
            r"alert|notify|warn": "alert",
            r"reject|fail|error": "reject",
            r"adjust|move|reposition": "adjust",
        }

        self.priority_patterns = {
            r"critical|urgent|high priority": "critical",
            r"high|important": "high",
            r"medium|normal": "medium",
            r"low|minor": "low",
        }

    def parse(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse natural language text into rule specification.

        Args:
            text: Natural language rule description
            context: Additional context for parsing

        Returns:
            Parsed rule specification
        """
        text = text.lower().strip()

        # Determine rule type
        rule_type = self._determine_rule_type(text)

        # Parse based on type
        if rule_type == "position":
            return self._parse_position_rule(text)
        elif rule_type == "spacing":
            return self._parse_spacing_rule(text)
        elif rule_type == "alignment":
            return self._parse_alignment_rule(text)
        else:
            return self._parse_custom_rule(text)

    def _determine_rule_type(self, text: str) -> str:
        """Determine the type of rule from text."""
        if any(word in text for word in ["center", "edge", "corner", "position"]):
            return "position"
        elif any(word in text for word in ["space", "spacing", "distance", "between"]):
            return "spacing"
        elif any(word in text for word in ["align", "line up", "row", "column"]):
            return "alignment"
        else:
            return "custom"

    def _parse_position_rule(self, text: str) -> Dict[str, Any]:
        """Parse position-based rule."""
        # Extract position
        position = None
        for pattern, pos_type in self.position_patterns.items():
            if re.search(pattern, text):
                position = pos_type
                break

        # Extract action
        action = "alert"  # default
        for pattern, act_type in self.action_patterns.items():
            if re.search(pattern, text):
                action = act_type
                break

        # Extract priority
        priority = "medium"  # default
        for pattern, pri_type in self.priority_patterns.items():
            if re.search(pattern, text):
                priority = pri_type
                break

        # Build conditions
        conditions = []
        if position:
            if position == "center":
                conditions.append(
                    {
                        "field": "positioning.position_key",
                        "operator": "==",
                        "value": "center",
                    }
                )
            elif "edge" in position:
                conditions.append(
                    {
                        "field": "positioning.position_key",
                        "operator": "contains",
                        "value": position,
                    }
                )

        # Build actions
        actions = []
        if action == "alert":
            actions.append(
                {
                    "type": "alert",
                    "parameters": {
                        "message": f"Mug should be {position or 'properly positioned'}",
                        "severity": priority,
                    },
                }
            )
        elif action == "reject":
            actions.append(
                {
                    "type": "reject",
                    "parameters": {
                        "message": f"Mug is not {position or 'properly positioned'}",
                        "reason": "position_violation",
                    },
                }
            )

        return {
            "name": f"Position Rule: {position or 'custom'}",
            "type": "position",
            "priority": priority,
            "conditions": conditions,
            "actions": actions,
            "confidence": 0.8 if position else 0.6,
            "description": text,
        }

    def _parse_spacing_rule(self, text: str) -> Dict[str, Any]:
        """Parse spacing-based rule."""
        # Extract spacing value
        spacing_value = None
        spacing_unit = "pixels"

        for pattern, unit in self.spacing_patterns.items():
            match = re.search(pattern, text)
            if match:
                spacing_value = float(match.group(1))
                spacing_unit = unit
                break

        # Convert to pixels if needed (approximate conversions)
        if spacing_value:
            if spacing_unit == "inches":
                spacing_value *= 96  # Assuming 96 DPI
            elif spacing_unit == "mm":
                spacing_value *= 3.78  # Approximate mm to pixels
            elif spacing_unit == "cm":
                spacing_value *= 37.8  # Approximate cm to pixels

        # Extract comparison type
        if "at least" in text or "minimum" in text:
            operator = ">="
        elif "at most" in text or "maximum" in text:
            operator = "<="
        else:
            operator = ">"

        # Build conditions
        conditions = []
        if spacing_value:
            conditions.append(
                {
                    "field": "spacing.min_spacing",
                    "operator": operator,
                    "value": spacing_value,
                }
            )

        # Build actions
        actions = [
            {
                "type": "alert",
                "parameters": {
                    "message": f"Mugs should have {text}",
                    "severity": "medium",
                },
            }
        ]

        return {
            "name": f"Spacing Rule: {spacing_value or 'custom'} {spacing_unit}",
            "type": "spacing",
            "priority": "medium",
            "conditions": conditions,
            "actions": actions,
            "confidence": 0.85 if spacing_value else 0.5,
            "description": text,
        }

    def _parse_alignment_rule(self, text: str) -> Dict[str, Any]:
        """Parse alignment-based rule."""
        # Determine alignment type
        if "horizontal" in text or "row" in text:
            alignment = "horizontal"
        elif "vertical" in text or "column" in text:
            alignment = "vertical"
        else:
            alignment = "any"

        # Build conditions
        conditions = [
            {
                "field": "alignment.is_aligned",
                "operator": "==",
                "value": True,
            }
        ]

        if alignment != "any":
            conditions.append(
                {
                    "field": "alignment.alignment_type",
                    "operator": "==",
                    "value": alignment,
                }
            )

        # Build actions
        actions = [
            {
                "type": "alert",
                "parameters": {
                    "message": f"Mugs should be aligned {alignment if alignment != 'any' else 'properly'}",
                    "severity": "low",
                },
            }
        ]

        return {
            "name": f"Alignment Rule: {alignment}",
            "type": "alignment",
            "priority": "low",
            "conditions": conditions,
            "actions": actions,
            "confidence": 0.75,
            "description": text,
        }

    def _parse_custom_rule(self, text: str) -> Dict[str, Any]:
        """Parse custom rule with best-effort interpretation."""
        # Try to extract any meaningful patterns
        priority = "medium"
        for pattern, pri_type in self.priority_patterns.items():
            if re.search(pattern, text):
                priority = pri_type
                break

        return {
            "name": "Custom Rule",
            "type": "custom",
            "priority": priority,
            "conditions": [],
            "actions": [
                {
                    "type": "alert",
                    "parameters": {
                        "message": text,
                        "severity": priority,
                    },
                }
            ],
            "confidence": 0.4,
            "description": text,
        }
