# Rule Engine for H200 Intelligent Mug Positioning System

The rule engine provides natural language processing capabilities for defining and executing positioning rules. Users can create rules in plain English that are automatically parsed, validated, and applied during mug positioning.

## Features

- **Natural Language Parsing**: Convert plain English rules into structured positioning logic using LangChain
- **Rule Types**: Support for distance, placement, avoidance, grouping, preference, safety, and context rules
- **Version Control**: Full version history tracking for all rules with rollback capabilities
- **Conflict Resolution**: Smart conflict resolution between competing rules based on priority
- **Dynamic Updates**: Rules can be added, updated, or removed in real-time
- **MongoDB Storage**: Persistent storage with indexing for fast retrieval
- **Batch Operations**: Import/export and batch processing of rules

## Architecture

```
rules/
├── engine.py       # Main orchestrator
├── parser.py       # LangChain-based NLP parser
├── executor.py     # Rule execution logic
├── storage.py      # MongoDB persistence
├── models.py       # Data models
└── examples.py     # Usage examples
```

## Quick Start

```python
from src.core.rules import RuleEngine

# Initialize engine
engine = RuleEngine()
await engine.initialize()

# Add a rule from natural language
result = await engine.add_rule(
    "Keep mugs at least 6 inches from laptops",
    priority=9
)

# Apply rules to positioning
updated_positions, execution_results = await engine.apply_rules(
    positioning_result,
    scene_context,
    image
)
```

## Rule Types

### Distance Rules
Control minimum/maximum distances between objects:
```
"Keep mugs at least 6 inches from laptops"
"Maintain 10cm distance between beverages"
"Ensure drinks are no closer than 4 inches to edges"
```

### Placement Rules
Specify where objects should be placed:
```
"Place all beverages on coasters when available"
"Position mugs on the right side of the workspace"
"Keep drinks on designated beverage mats"
```

### Avoidance Rules
Define areas or objects to avoid:
```
"Keep liquids away from electronic devices"
"Avoid placing mugs near paper documents"
"Don't place beverages on mouse pads"
```

### Grouping Rules
Control how multiple items are arranged:
```
"Group all drinks together for easy access"
"Cluster beverages within 12 inches of each other"
"Arrange mugs in a single row when space is limited"
```

### Safety Rules
Enforce safety-related positioning:
```
"Ensure mugs are on stable surfaces"
"Keep hot beverages on heat-resistant surfaces"
"Maintain safe distance from table edges"
```

### Context Rules
Apply rules based on environment:
```
"In office settings, place mugs on the non-dominant side"
"During meetings, keep beverages away from shared documents"
"In kitchen environments, avoid cooking areas"
```

## Rule Structure

Each rule consists of:

1. **Conditions**: When the rule should apply
   - Subject (what to check)
   - Operator (how to check)
   - Target (what to check against)
   - Value (optional numeric value)

2. **Actions**: What to do when conditions are met
   - Type (move, maintain_distance, place_on, etc.)
   - Target (what to act on)
   - Parameters (action-specific settings)

3. **Metadata**:
   - Priority (1-10, higher executes first)
   - Version tracking
   - Creation/update timestamps
   - User attribution

## API Reference

### RuleEngine

```python
# Add a rule
result = await engine.add_rule(
    natural_language="Keep mugs 6 inches from laptops",
    user_id="john_doe",
    priority=8
)

# Update a rule
result = await engine.update_rule(
    rule_id="rule_123",
    natural_language="Keep mugs 8 inches from all electronics",
    user_id="john_doe"
)

# Delete a rule
success = await engine.delete_rule(
    rule_id="rule_123",
    permanent=False  # Soft delete by default
)

# List rules
rules = await engine.list_rules(
    rule_type=RuleType.DISTANCE,
    enabled_only=True,
    limit=50
)

# Apply rules to positioning
updated_result, execution_results = await engine.apply_rules(
    positioning_result,
    scene_context,
    image,
    rule_ids=["rule_1", "rule_2"]  # Optional: specific rules
)

# Validate without storing
validation = await engine.validate_rule(
    "Place mugs on coasters"
)

# Batch operations
results = await engine.batch_add_rules([
    "Rule 1 text",
    "Rule 2 text"
])

# Get suggestions
suggestions = await engine.get_rule_suggestions(scene_context)

# Export/Import
exported = await engine.export_rules(format="json")
imported = await engine.import_rules(rules_data)
```

### Rule Storage

```python
# Direct storage operations
storage = RuleStorage()
await storage.initialize()

# Get rule history
history = await storage.get_rule_history(
    rule_id="rule_123",
    limit=10
)

# Restore previous version
success = await storage.restore_rule_version(
    rule_id="rule_123",
    version=3
)

# Search rules
results = await storage.search_rules(
    query="laptop distance",
    rule_type=RuleType.DISTANCE
)
```

## Configuration

### Environment Variables

```bash
# OpenAI API key for LangChain
OPENAI_API_KEY=your_api_key

# MongoDB connection
MONGODB_ATLAS_URI=mongodb+srv://...

# Rule engine settings
RULE_CONFLICT_RESOLUTION=priority  # priority, combine, or user
RULE_CACHE_SIZE=1000
RULE_EXECUTION_TIMEOUT=5000  # milliseconds
```

### Rule Priorities

- 10: Critical safety rules
- 8-9: Important placement rules
- 5-7: Standard positioning rules
- 3-4: Preference rules
- 1-2: Suggestions

## Examples

### Office Setup
```python
office_rules = [
    "Keep beverages at least 6 inches from keyboards and mice",
    "Place mugs on the right side of the monitor for right-handed users",
    "Use coasters on wooden desk surfaces",
    "Group multiple beverages together near the desk lamp",
    "Avoid placing drinks directly behind the laptop screen"
]

for rule in office_rules:
    await engine.add_rule(rule, user_id="office_admin")
```

### Kitchen Safety
```python
kitchen_rules = [
    "Keep all beverages at least 12 inches from the stove",
    "Place mugs on designated counter areas only",
    "Avoid positioning drinks near the sink splash zone",
    "Ensure hot beverages are on heat-resistant surfaces",
    "Maintain clear pathways by keeping mugs away from counter edges"
]

results = await engine.batch_add_rules(
    kitchen_rules,
    user_id="kitchen_safety"
)
```

### Meeting Room
```python
meeting_rules = [
    "Position beverages at least 10 inches from shared documents",
    "Place drinks on individual coasters during meetings",
    "Keep beverages on the outer edge of the conference table",
    "Group refreshments in the designated beverage area",
    "Ensure each participant's drink is within their reach"
]
```

## Performance Considerations

1. **Rule Caching**: Active rules are cached in memory for fast access
2. **Indexed Storage**: MongoDB indexes on common query fields
3. **Parallel Parsing**: Batch rule parsing uses async operations
4. **Conflict Resolution**: O(n log n) complexity for priority-based resolution
5. **Spatial Calculations**: Optimized distance and overlap calculations

## Troubleshooting

### Common Issues

1. **Rule Not Parsing**
   - Check natural language is clear and specific
   - Verify measurements include units (inches, cm)
   - Ensure objects are recognizable (mug, laptop, etc.)

2. **Rules Not Applying**
   - Verify rule is enabled
   - Check rule conditions match scene context
   - Ensure priority is appropriate

3. **Conflicting Rules**
   - Review rule priorities
   - Check conflict resolution strategy
   - Consider combining related rules

### Debug Mode

```python
# Enable debug logging
import structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG)
)

# Get detailed execution info
execution_results = await engine.apply_rules(
    positioning_result,
    scene_context,
    image
)

for result in execution_results:
    print(f"Rule: {result.rule_id}")
    print(f"Executed: {result.executed}")
    print(f"Actions: {result.actions_taken}")
    print(f"Conflicts: {result.conflicts}")
    print(f"Time: {result.execution_time}ms")
```

## Best Practices

1. **Rule Design**
   - Be specific with measurements and units
   - Use clear object names (avoid ambiguity)
   - Set appropriate priorities
   - Group related rules

2. **Performance**
   - Limit active rules to necessary ones
   - Use rule types for efficient filtering
   - Cache frequently used rules
   - Monitor execution times

3. **Maintenance**
   - Regular review of rule effectiveness
   - Version control for important rules
   - Document complex rule logic
   - Test rules in various scenarios

## Future Enhancements

- [ ] Machine learning for rule effectiveness
- [ ] Visual rule builder interface
- [ ] Rule templates for common scenarios
- [ ] A/B testing for rule variations
- [ ] Performance analytics dashboard
- [ ] Multi-language rule support
- [ ] Rule chaining and dependencies
- [ ] Conditional rule activation based on time/user/context