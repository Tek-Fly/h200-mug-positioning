"""
Example usage of the rule engine.

Demonstrates how to create, manage, and apply positioning rules.
"""

# Standard library imports
import asyncio
from typing import List

# Third-party imports
from PIL import Image

# First-party imports
from src.core.models.manager import ModelManager
from src.core.positioning import MugPositioningEngine, PositioningStrategy
from src.core.rules import RuleEngine


async def example_rule_creation():
    """Example of creating rules from natural language."""
    # Initialize rule engine
    engine = RuleEngine()
    await engine.initialize()

    # Example rules to add
    example_rules = [
        # Distance rules
        "Keep mugs at least 6 inches away from laptops",
        "Maintain a minimum distance of 8 inches between beverages and keyboards",
        "Ensure drinks are no closer than 4 inches to the edge of the table",
        # Placement rules
        "Place all beverages on coasters when available",
        "Position mugs on the right side of the workspace",
        "Keep hot beverages on heat-resistant surfaces",
        # Grouping rules
        "Group all drinks together in the designated beverage area",
        "Cluster mugs within 12 inches of each other for easy access",
        # Safety rules
        "Avoid placing liquids near electronic devices",
        "Keep beverages away from paper documents",
        "Ensure mugs are on stable, flat surfaces",
        # Context-specific rules
        "In office settings, place mugs on the non-dominant side",
        "In kitchen environments, keep drinks away from cooking areas",
        "During meetings, position beverages at least 10 inches from shared documents",
    ]

    print("Adding example rules...")

    for rule_text in example_rules:
        result = await engine.add_rule(rule_text)

        if result.valid:
            rule = result.parsed_rule
            print(f"✓ Added: {rule.name}")
            print(f"  Type: {rule.type.value}")
            print(f"  Priority: {rule.priority}")
            print(f"  Conditions: {len(rule.conditions)}")
            print(f"  Actions: {len(rule.actions)}")
        else:
            print(f"✗ Failed: {rule_text[:50]}...")
            print(f"  Errors: {', '.join(result.errors)}")

        print()

    # List all active rules
    print("\nActive Rules:")
    rules = await engine.list_rules()
    for rule in rules:
        print(f"- {rule.name} (Priority: {rule.priority})")


async def example_rule_application():
    """Example of applying rules to positioning results."""
    # Initialize components
    model_manager = ModelManager()
    await model_manager.initialize()

    positioning_engine = MugPositioningEngine(
        yolo_model=model_manager.yolo_model,
        clip_model=model_manager.clip_model,
        strategy=PositioningStrategy.HYBRID,
    )

    rule_engine = RuleEngine(positioning_engine=positioning_engine)
    await rule_engine.initialize()

    # Add some rules
    await rule_engine.add_rule("Keep mugs at least 6 inches from laptops", priority=9)
    await rule_engine.add_rule("Place beverages on coasters when available", priority=7)
    await rule_engine.add_rule(
        "Group all drinks together on the right side", priority=5
    )

    # Load test image (you would use a real image here)
    # image = Image.open("test_image.jpg")
    # For demo, create a dummy image
    image = Image.new("RGB", (1920, 1080), color="white")

    # Get initial positioning
    print("\nCalculating initial positions...")
    positioning_result = await positioning_engine.calculate_positions(image)

    print(f"Initial positions: {len(positioning_result.positions)}")
    print(f"Initial confidence: {positioning_result.overall_confidence:.2f}")

    # Mock scene context for demo
    scene_context = {
        "detected_objects": {"mug": 2, "laptop": 1, "coaster": 1, "desk": 1},
        "scene_type": "office",
        "surface_areas": [
            {
                "type": "desk",
                "bbox": {"x1": 0, "y1": 500, "x2": 1920, "y2": 1080},
                "confidence": 0.95,
            }
        ],
        "danger_zones": [
            {
                "type": "electronics",
                "object": "laptop",
                "bbox": {"x1": 800, "y1": 600, "x2": 1200, "y2": 800},
            }
        ],
    }

    # Apply rules
    print("\nApplying positioning rules...")
    updated_result, execution_results = await rule_engine.apply_rules(
        positioning_result, scene_context, image
    )

    print(f"\nRule Execution Results:")
    for exec_result in execution_results:
        print(f"- Rule: {exec_result.metadata.get('rule_name', 'Unknown')}")
        print(f"  Executed: {exec_result.executed}")
        if exec_result.actions_taken:
            print(f"  Actions: {[a['type'] for a in exec_result.actions_taken]}")
        if exec_result.conflicts:
            print(f"  Conflicts: {exec_result.conflicts}")

    print(f"\nUpdated positions: {len(updated_result.positions)}")
    print(f"Updated confidence: {updated_result.overall_confidence:.2f}")
    print(f"Applied rules: {updated_result.applied_rules}")
    print(f"Recommendations: {updated_result.recommendations}")


async def example_rule_management():
    """Example of rule management operations."""
    engine = RuleEngine()
    await engine.initialize()

    # Add a rule
    result = await engine.add_rule(
        "Keep beverages at least 10 inches from electronic devices", priority=8
    )

    if result.valid:
        rule_id = result.parsed_rule.id
        print(f"Created rule: {rule_id}")

        # Get rule details
        rule = await engine.get_rule(rule_id)
        print(f"\nRule Details:")
        print(f"Name: {rule.name}")
        print(f"Description: {rule.description}")
        print(f"Natural Language: {rule.natural_language}")
        print(f"Type: {rule.type.value}")
        print(f"Priority: {rule.priority}")
        print(f"Version: {rule.version}")

        # Update the rule
        print("\nUpdating rule...")
        update_result = await engine.update_rule(
            rule_id,
            "Keep beverages at least 12 inches from all electronic devices including phones",
        )

        if update_result.valid:
            updated_rule = await engine.get_rule(rule_id)
            print(f"Updated version: {updated_rule.version}")
            print(f"New description: {updated_rule.description}")

        # Get rule history
        history = await engine.storage.get_rule_history(rule_id)
        print(f"\nRule History ({len(history)} entries):")
        for entry in history[:3]:
            print(
                f"- Version {entry['version']}: {entry['action']} by {entry['user_id']}"
            )

        # Disable the rule
        print("\nDisabling rule...")
        await engine.delete_rule(rule_id, permanent=False)

        # List rules by type
        print("\nDistance Rules:")
        distance_rules = await engine.list_rules(rule_type="distance")
        for rule in distance_rules:
            print(f"- {rule.name}")


async def example_batch_operations():
    """Example of batch rule operations."""
    engine = RuleEngine()
    await engine.initialize()

    # Batch add rules
    office_rules = [
        "Keep mugs on the right side of the monitor",
        "Maintain 6 inch clearance from keyboard",
        "Use coasters on wooden desks",
        "Group beverages near the desk lamp",
        "Avoid placing drinks on mouse pads",
    ]

    print("Batch adding office rules...")
    results = await engine.batch_add_rules(office_rules, user_id="office_admin")

    print(f"\nBatch Results:")
    print(f"Total: {results['total']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")

    if results["errors"]:
        print("\nErrors:")
        for error in results["errors"]:
            print(f"- Rule {error['index']}: {error['error']}")

    # Export rules
    print("\nExporting rules...")
    exported = await engine.export_rules(format="json")
    print(f"Exported {len(exported)} rules")

    # Get suggestions based on scene
    scene_context = {
        "scene_type": "office",
        "detected_objects": {"mug": 3, "laptop": 1, "monitor": 2},
    }

    suggestions = await engine.get_rule_suggestions(scene_context)
    print("\nSuggested Rules for Office Scene:")
    for suggestion in suggestions:
        print(f"- {suggestion}")


async def example_custom_rule_patterns():
    """Example of complex rule patterns."""
    engine = RuleEngine()
    await engine.initialize()

    # Complex conditional rules
    complex_rules = [
        # Time-based rule (would need time context)
        "During video calls, keep beverages at least 12 inches from the webcam",
        # Multi-object rule
        "When both coffee and water are present, place coffee closer to the dominant hand",
        # Quantity-based rule
        "If more than 3 mugs are detected, arrange them in a grid pattern",
        # Surface-specific rule
        "On glass surfaces, always use coasters or protective mats",
        # Priority-based rule
        "Prioritize placing hot beverages on heat-resistant surfaces over other placement rules",
    ]

    print("Adding complex rules...")
    for rule_text in complex_rules:
        result = await engine.add_rule(rule_text)
        if result.valid:
            print(f"✓ {result.parsed_rule.name}")

            # Show parsed conditions and actions
            rule = result.parsed_rule
            print(f"  Conditions:")
            for cond in rule.conditions:
                print(f"    - {cond.subject} {cond.operator.value} {cond.target}")
            print(f"  Actions:")
            for action in rule.actions:
                print(f"    - {action.type.value} {action.target}")
        else:
            print(f"✗ Failed to parse: {rule_text[:50]}...")
        print()


# Main execution
if __name__ == "__main__":

    async def main():
        print("=== Rule Engine Examples ===\n")

        print("1. Rule Creation Example")
        print("-" * 50)
        await example_rule_creation()

        print("\n\n2. Rule Application Example")
        print("-" * 50)
        # await example_rule_application()  # Commented as it needs models

        print("\n\n3. Rule Management Example")
        print("-" * 50)
        await example_rule_management()

        print("\n\n4. Batch Operations Example")
        print("-" * 50)
        await example_batch_operations()

        print("\n\n5. Custom Rule Patterns Example")
        print("-" * 50)
        await example_custom_rule_patterns()

    asyncio.run(main())
