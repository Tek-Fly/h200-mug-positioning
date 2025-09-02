# Rules Management Guide

Comprehensive guide to creating, managing, and optimizing positioning rules in the H200 System.

## Introduction to Rules

Rules in the H200 System define the criteria for good mug positioning and provide automated feedback when analyzing images. They enable consistent quality control and automated guidance for positioning optimization.

### What Rules Do

**Primary Functions:**
- **Evaluate Positioning**: Check if mugs meet defined criteria
- **Provide Feedback**: Give specific suggestions for improvement
- **Automate Quality Control**: Consistent standards across all analyses
- **Generate Alerts**: Notify when violations occur

**Rule Applications:**
- Coffee shop setup standards
- Photography positioning guidelines
- Quality control in production
- Training material validation
- Automated workflow checks

## Rule Concepts

### Rule Components

Every rule consists of four main components:

#### 1. Conditions
Define what to check and measure:
```
Distance from center < 2cm
AND
Distance to nearest edge > 1cm
```

#### 2. Actions
Specify what happens when conditions are met:
```
Show alert: "Mug positioning needs adjustment"
Log event: "Rule violation detected"
Send notification: Email to supervisor
```

#### 3. Priority
Determine rule importance:
- **Critical**: Must be addressed immediately
- **High**: Important violations requiring attention
- **Medium**: Standard positioning guidelines
- **Low**: Optional optimizations

#### 4. Context
Define when rules apply:
- **Universal**: All analyses
- **Contextual**: Specific scenarios (coffee shop, dining, photography)
- **Conditional**: Based on image characteristics or user settings

### Rule Types

#### Positioning Rules
Focus on mug placement relative to surfaces and objects:

**Center Alignment:**
```
Rule: "Mug should be centered on coaster"
Condition: distance_from_center < 1.5cm
Action: Alert if offset > threshold
```

**Orientation Rules:**
```
Rule: "Mug handle should face right"
Condition: handle_angle between 0-45 degrees
Action: Suggest handle repositioning
```

#### Spacing Rules
Manage distances between multiple mugs:

**Minimum Separation:**
```
Rule: "Mugs should be at least 5cm apart"
Condition: min_distance_between_mugs >= 5cm
Action: Alert if too close together
```

**Grid Alignment:**
```
Rule: "Mugs should align in rows"
Condition: vertical_alignment_tolerance < 2cm
Action: Suggest alignment adjustment
```

#### Safety Rules
Ensure stable and safe positioning:

**Edge Distance:**
```
Rule: "Keep 3cm clearance from table edge"
Condition: distance_to_edge >= 3cm
Action: Critical alert if too close
```

**Stability Check:**
```
Rule: "Mug must be fully on surface"
Condition: surface_coverage > 90%
Action: Warning if unstable
```

#### Quality Rules
Maintain visual and functional standards:

**Size Consistency:**
```
Rule: "All mugs should be similar size"
Condition: size_variance < 20%
Action: Note inconsistency
```

**Orientation Consistency:**
```
Rule: "All handles face same direction"
Condition: handle_direction_variance < 30°
Action: Suggest realignment
```

## Creating Rules with Natural Language

The easiest way to create rules is using natural language descriptions.

### Natural Language Interface

#### Basic Rule Creation

**Step 1: Describe the Rule**
```
Input: "The mug should be centered on the coaster with at least 
       2 inches of clearance from the table edge"

System Interpretation:
- Rule Type: Positioning + Safety
- Conditions: 
  • Center offset < 1cm (interpreted from "centered")
  • Edge distance > 5cm (converted from 2 inches)
- Confidence: 94%
```

**Step 2: Review and Refine**
The system shows its interpretation and confidence level. If confidence is low (<70%), consider rephrasing:

```
❌ Low Confidence (65%):
"The mug should be somewhere nice on the table"

✅ High Confidence (95%):
"The mug should be centered on the coaster"
```

**Step 3: Add Context**
Specify when the rule should apply:
```
Context Options:
- Coffee shop setup
- Formal dining
- Photography session
- Training environment
- Custom: "Café morning service"
```

#### Advanced Natural Language Features

**Compound Rules:**
```
"The mug should be centered on the coaster AND at least 
3 inches from other mugs AND the handle should point 
toward the customer"

Parsed as:
- Condition 1: Center alignment
- Condition 2: Minimum spacing
- Condition 3: Handle orientation
```

**Conditional Rules:**
```
"IF there are multiple mugs THEN they should be 
evenly spaced, OTHERWISE the single mug should 
be centered"

Parsed as:
- Multi-mug scenario: Even spacing rule
- Single mug scenario: Center alignment rule
```

**Measurement Specifications:**
```
Supported Units:
- Distance: mm, cm, inches, pixels
- Angles: degrees, radians
- Percentages: coverage, variance
- Counts: number of mugs, violations

Examples:
"Keep 2 inches from edge" → 5.08cm
"Rotate 90 degrees" → π/2 radians  
"Cover 95% of coaster" → 0.95 coverage
```

### Natural Language Examples

#### Coffee Shop Rules
```
"Mugs should be centered on their coasters with handles 
pointing toward customers. Keep at least 4 inches between 
mugs and 3 inches from table edges."

Generated Rules:
1. Center alignment (priority: high)
2. Handle orientation (priority: medium)  
3. Mug spacing (priority: medium)
4. Edge clearance (priority: critical)
```

#### Photography Rules
```
"For product photography, mugs should be perfectly 
centered with handles at 45-degree angles. No shadows 
should fall on the product area."

Generated Rules:
1. Precise center alignment (priority: critical)
2. Handle angle specification (priority: high)
3. Shadow detection (priority: high)
```

#### Training Rules
```
"Training setups should have mugs slightly off-center 
so students can practice corrections. Keep safe distances 
from edges but allow positioning mistakes."

Generated Rules:
1. Intentional offset allowance (priority: low)
2. Safety edge distance (priority: critical)
3. Learning tolerance zones (priority: medium)
```

### Improving Natural Language Processing

#### Tips for Better Interpretation

**Be Specific:**
```
❌ Vague: "Mug should look good"
✅ Specific: "Mug should be centered with 2cm tolerance"
```

**Use Standard Units:**
```
❌ Ambiguous: "Keep some distance"
✅ Clear: "Keep 5 centimeters clearance"
```

**Specify Context:**
```
❌ Generic: "Arrange mugs properly"
✅ Contextual: "For coffee service, arrange mugs with handles facing customers"
```

**Break Down Complex Rules:**
```
❌ Complex: "Mugs should be arranged nicely with good spacing and proper orientation while maintaining safety distances and visual appeal"

✅ Simplified: 
- "Mugs should be evenly spaced 6cm apart"
- "Handles should point toward customers"  
- "Keep 4cm from table edges"
- "Maintain visual symmetry"
```

## Manual Rule Creation

For precise control, create rules manually using the advanced rule builder.

### Condition Builder

#### Available Fields

**Geometric Fields:**
- `distance_from_center`: Distance from center point (cm)
- `distance_to_edge`: Minimum distance to any edge (cm)
- `distance_between_mugs`: Distance between mug centers (cm)
- `angle_from_reference`: Angle from reference direction (degrees)
- `surface_coverage`: Percentage of mug on surface (0-1)

**Appearance Fields:**
- `mug_size`: Detected mug size category (small/medium/large)
- `mug_color`: Dominant color classification
- `handle_visible`: Whether handle is clearly visible (true/false)
- `handle_orientation`: Handle direction in degrees (0-360)

**Contextual Fields:**
- `mug_count`: Number of mugs detected
- `image_quality`: Overall image quality score (0-1)
- `lighting_quality`: Lighting assessment score (0-1)
- `background_contrast`: Contrast with background (0-1)

#### Operators

**Comparison Operators:**
- `equals`, `not_equals`: Exact matching
- `greater_than`, `greater_than_or_equal`: Minimum thresholds
- `less_than`, `less_than_or_equal`: Maximum limits
- `between`: Range checking (inclusive)
- `outside_range`: Exclusion range

**String Operators:**
- `contains`, `not_contains`: Substring matching
- `starts_with`, `ends_with`: Pattern matching
- `matches_regex`: Regular expression matching

**Array Operators:**
- `in_list`, `not_in_list`: Membership testing
- `all_match`, `any_match`: Multi-condition evaluation

#### Complex Conditions

**Multiple Conditions:**
```json
{
  "logic": "AND",
  "conditions": [
    {
      "field": "distance_from_center",
      "operator": "less_than",
      "value": 2.0,
      "unit": "cm"
    },
    {
      "field": "distance_to_edge", 
      "operator": "greater_than",
      "value": 3.0,
      "unit": "cm"
    }
  ]
}
```

**Nested Logic:**
```json
{
  "logic": "OR",
  "conditions": [
    {
      "logic": "AND",
      "conditions": [
        {"field": "mug_count", "operator": "equals", "value": 1},
        {"field": "distance_from_center", "operator": "less_than", "value": 1.0}
      ]
    },
    {
      "logic": "AND", 
      "conditions": [
        {"field": "mug_count", "operator": "greater_than", "value": 1},
        {"field": "distance_between_mugs", "operator": "greater_than", "value": 5.0}
      ]
    }
  ]
}
```

### Action Configuration

#### Action Types

**Alert Actions:**
```json
{
  "type": "alert",
  "parameters": {
    "message": "Mug positioning requires attention",
    "severity": "warning",
    "category": "positioning"
  }
}
```

**Notification Actions:**
```json
{
  "type": "notification", 
  "parameters": {
    "channels": ["email", "webhook"],
    "recipients": ["supervisor@company.com"],
    "template": "positioning_violation"
  }
}
```

**Adjustment Actions:**
```json
{
  "type": "adjustment",
  "parameters": {
    "suggestion": "Move mug 2cm toward center",
    "direction": "center",
    "distance_cm": 2.0,
    "confidence": 0.9
  }
}
```

**Logging Actions:**
```json
{
  "type": "log",
  "parameters": {
    "level": "INFO",
    "message": "Rule evaluation completed",
    "include_details": true,
    "retention_days": 30
  }
}
```

#### Action Parameters

**Message Formatting:**
```json
{
  "message": "Mug at ({x}, {y}) is {distance}cm from center (max: {threshold}cm)",
  "variables": {
    "x": "detection.center.x",
    "y": "detection.center.y", 
    "distance": "calculation.distance_from_center",
    "threshold": "rule.condition.value"
  }
}
```

**Conditional Actions:**
```json
{
  "type": "conditional_alert",
  "conditions": {
    "severity_high": {
      "condition": "distance > 5.0",
      "action": {
        "message": "CRITICAL: Mug positioning violation",
        "color": "red",
        "urgent": true
      }
    },
    "severity_medium": {
      "condition": "distance > 2.0",
      "action": {
        "message": "WARNING: Minor positioning issue", 
        "color": "yellow",
        "urgent": false
      }
    }
  }
}
```

## Rule Testing and Validation

### Test Environment

#### Test Image Collection

**Prepare Test Sets:**
```
Test Categories:
├── positive_cases/     # Images that should pass rules
├── negative_cases/     # Images that should trigger rules  
├── edge_cases/        # Boundary conditions
└── performance_cases/ # Complex scenarios
```

**Image Requirements:**
- Diverse lighting conditions
- Various mug types and colors
- Different positioning scenarios
- Multiple mug configurations
- Edge case scenarios

#### Testing Interface

**Test Configuration:**
```
┌─────────────────────────────────────────────────────────────┐
│ 🧪 Rule Testing                                            │
├─────────────────────────────────────────────────────────────┤
│ Rules to Test: [✓] Center alignment                        │
│               [✓] Edge clearance                           │
│               [ ] Handle orientation                       │
│                                                             │
│ Test Images: 25 selected                                   │
│ Expected Results: 15 pass, 10 fail                        │
│                                                             │
│ [Run Test] [Save Configuration] [Load Previous]            │
└─────────────────────────────────────────────────────────────┘
```

### Test Results Analysis

#### Accuracy Metrics

**Confusion Matrix:**
```
                 Predicted
                Pass  Fail
Actual  Pass     12    3    (15 total)
        Fail      2    8    (10 total)
                ─────────
                 14   11    (25 total)

Accuracy: 80% (20/25)
Precision: 72.7% (8/11) 
Recall: 80% (8/10)
F1-Score: 76.2%
```

**Performance by Category:**
```
Rule Category          Accuracy  Precision  Recall
─────────────────────  ────────  ─────────  ──────
Center alignment         95%       92%       98%
Edge clearance          85%       78%       95%
Handle orientation      70%       65%       82%
Multi-mug spacing       88%       85%       91%
```

#### False Positive Analysis

**Common False Positives:**
1. **Lighting Effects**: Shadows causing detection errors
2. **Background Interference**: Similar objects confused with mugs
3. **Partial Occlusion**: Hidden parts affecting measurements
4. **Calibration Issues**: Incorrect pixel-to-distance conversion

**Mitigation Strategies:**
```
Issue: Shadow-based false positives
Solution: Add lighting quality condition:
- lighting_quality > 0.7 AND shadow_area < 0.1
```

#### False Negative Analysis

**Common False Negatives:**
1. **Threshold Too Strict**: Minor violations not detected
2. **Edge Cases**: Unusual but valid positioning ignored
3. **Measurement Errors**: Inaccurate distance calculations
4. **Context Mismatches**: Rules applied inappropriately

**Optimization Approaches:**
```
Issue: Missed minor violations
Solution: Adjust thresholds:
- distance_from_center < 1.5cm → < 2.0cm
- Add warning level for 1.5-2.0cm range
```

### Rule Performance Optimization

#### Threshold Tuning

**Sensitivity Analysis:**
```
Threshold Value    True Positives    False Positives    Accuracy
─────────────────  ──────────────    ───────────────    ────────
1.0cm                    8                 2             80%
1.5cm                   10                 4             75%
2.0cm                   12                 6             70%
2.5cm                   14                 9             65%

Optimal: 1.5cm (balance of precision and recall)
```

**ROC Curve Analysis:**
Generate receiver operating characteristic curves to find optimal thresholds for different use cases:
- **High Precision**: Strict thresholds, fewer false alerts
- **High Recall**: Loose thresholds, catch more violations
- **Balanced**: Optimize F1-score for general use

#### Conditional Logic Optimization

**Context-Aware Rules:**
```
IF image_quality < 0.7 THEN
  increase_tolerance_by(0.5cm)
ELSE IF lighting_quality < 0.6 THEN
  disable_shadow_sensitive_rules()
ELSE
  use_standard_thresholds()
```

**Adaptive Thresholds:**
```
Base threshold: 2.0cm
Adjustments:
- High-quality image: -0.2cm (stricter)
- Poor lighting: +0.3cm (more tolerant)
- Multiple mugs: +0.1cm (account for complexity)
- Training mode: +0.5cm (more forgiving)
```

## Rule Management Operations

### Rule Organization

#### Categorization System

**Hierarchical Structure:**
```
Rules/
├── Critical/
│   ├── Safety/
│   │   ├── Edge_clearance
│   │   └── Stability_check
│   └── Quality/
│       └── Surface_coverage
├── Standard/
│   ├── Positioning/
│   │   ├── Center_alignment
│   │   └── Handle_orientation
│   └── Spacing/
│       └── Multi_mug_distance
└── Optional/
    ├── Aesthetics/
    └── Training/
```

**Tagging System:**
```
Tags: coffee_shop, photography, training, automated, manual_review
Priority: critical, high, medium, low
Context: universal, conditional, specialized
Status: active, inactive, testing, archived
```

#### Rule Sets

**Predefined Collections:**
```
Coffee Shop Service:
- Center alignment (high priority)
- Handle toward customer (medium)
- Edge clearance (critical)
- Consistent spacing (medium)

Photography Setup:
- Perfect center alignment (critical)
- Handle 45° angle (high) 
- No shadows (high)
- Color consistency (medium)

Training Environment:
- Relaxed tolerances (low priority)
- Safety rules only (critical)
- Learning feedback (medium)
```

### Rule Lifecycle Management

#### Rule States

**Development Lifecycle:**
```
Draft → Testing → Review → Active → Deprecated → Archived

Draft:     Under development, not evaluated
Testing:   Being validated with test data
Review:    Pending approval for production use
Active:    Live and evaluating images
Deprecated: Still active but scheduled for removal
Archived:  Historical record, no longer active
```

#### Version Control

**Rule Versioning:**
```
Rule: Center_alignment
├── v1.0: Initial implementation
├── v1.1: Added tolerance adjustment  
├── v1.2: Improved accuracy (current)
└── v2.0: Complete rewrite (in testing)

Version History:
- Performance improvements
- Bug fixes and corrections
- Parameter adjustments
- Logic enhancements
```

**Change Management:**
```
Change Request Process:
1. Identify need for rule modification
2. Document proposed changes
3. Test changes with validation data
4. Review impact assessment
5. Approve and deploy to production
6. Monitor performance post-deployment
```

### Rule Performance Monitoring

#### Real-Time Monitoring

**Rule Execution Metrics:**
```
┌─────────────────────────────────────────────────────────────┐
│ 📊 Rule Performance Dashboard                               │
├─────────────────────────────────────────────────────────────┤
│ Center Alignment:                                           │
│   Evaluations: 1,247 today                                │
│   Matches: 89 (7.1%)                                      │
│   Avg execution: 15ms                                     │
│   User feedback: 4.2/5.0 ⭐                               │
│                                                             │
│ Edge Clearance:                                            │
│   Evaluations: 1,247 today                                │
│   Matches: 23 (1.8%)                                      │
│   Avg execution: 8ms                                      │
│   User feedback: 4.7/5.0 ⭐                               │
└─────────────────────────────────────────────────────────────┘
```

**Performance Trends:**
- Execution time trends
- Match rate changes
- User satisfaction scores
- Error rate monitoring

#### User Feedback Integration

**Feedback Collection:**
```
Rule Evaluation Result:
"Mug should be moved 2cm toward center"

User Options:
☑ This suggestion was helpful
☐ This suggestion was incorrect
☐ This suggestion was unclear

Additional Comments:
"The measurement was accurate but the 
suggestion could be more specific about 
which direction to move."
```

**Feedback Analysis:**
```
Rule: Center_alignment
Feedback Summary (last 30 days):
- Helpful: 78% (156/200 responses)
- Incorrect: 12% (24/200 responses) 
- Unclear: 10% (20/200 responses)

Common Issues:
- Direction ambiguity (15 comments)
- Distance unit confusion (8 comments)
- Multiple mug scenarios (12 comments)
```

## Advanced Rule Features

### Dynamic Rule Parameters

#### Adaptive Thresholds

**Context-Sensitive Adjustments:**
```python
def adjust_threshold_for_context(base_threshold, context):
    adjustments = {
        'coffee_shop': {'tolerance': 0.5, 'precision': 'medium'},
        'photography': {'tolerance': 0.1, 'precision': 'high'},
        'training': {'tolerance': 1.0, 'precision': 'low'}
    }
    
    adjustment = adjustments.get(context, {'tolerance': 0.0, 'precision': 'medium'})
    return base_threshold + adjustment['tolerance']
```

**Image Quality Compensation:**
```python
def compensate_for_image_quality(threshold, image_metrics):
    quality_score = image_metrics['overall_quality']
    lighting_score = image_metrics['lighting_quality']
    
    if quality_score < 0.7:
        threshold *= 1.2  # More tolerant for poor quality images
    
    if lighting_score < 0.6:
        threshold *= 1.1  # Account for lighting issues
    
    return threshold
```

#### Learning Rules

**Performance-Based Adjustment:**
```python
class AdaptiveRule:
    def __init__(self, base_threshold):
        self.base_threshold = base_threshold
        self.performance_history = []
        self.adjustment_factor = 1.0
    
    def update_from_feedback(self, feedback_score):
        self.performance_history.append(feedback_score)
        
        # Adjust if recent performance is poor
        recent_avg = sum(self.performance_history[-10:]) / 10
        if recent_avg < 3.5:  # Poor performance
            self.adjustment_factor *= 1.05  # Make more tolerant
        elif recent_avg > 4.5:  # Excellent performance  
            self.adjustment_factor *= 0.98  # Make more strict
    
    def get_current_threshold(self):
        return self.base_threshold * self.adjustment_factor
```

### Custom Rule Extensions

#### Plugin Architecture

**Custom Condition Functions:**
```python
@rule_condition
def advanced_symmetry_check(detections, parameters):
    """Check for symmetrical mug arrangement"""
    if len(detections) < 2:
        return False
    
    centers = [d.center for d in detections]
    symmetry_score = calculate_symmetry(centers)
    
    return symmetry_score > parameters.get('min_symmetry', 0.8)

@rule_action  
def custom_notification(violation_data, parameters):
    """Send custom notification format"""
    message = format_custom_message(violation_data, parameters['template'])
    send_to_external_system(message, parameters['endpoint'])
```

**Integration Examples:**
```python
# Integrate with external quality systems
@rule_action
def quality_system_integration(violation_data, parameters):
    qms_client = QualityManagementClient(parameters['qms_endpoint'])
    qms_client.report_violation({
        'timestamp': violation_data['timestamp'],
        'severity': violation_data['severity'],
        'details': violation_data['rule_details'],
        'image_id': violation_data['image_reference']
    })
```

### Rule Debugging and Diagnostics

#### Debug Mode

**Detailed Execution Trace:**
```
Rule Evaluation: Center_alignment_v1.2
├── Condition 1: distance_from_center
│   ├── Calculated value: 2.3cm
│   ├── Threshold: 2.0cm  
│   ├── Result: FAIL (2.3 > 2.0)
│   └── Execution time: 12ms
├── Condition 2: surface_coverage
│   ├── Calculated value: 0.95
│   ├── Threshold: 0.90
│   ├── Result: PASS (0.95 > 0.90)
│   └── Execution time: 8ms
├── Overall Result: FAIL (AND logic, condition 1 failed)
└── Actions Triggered:
    └── Alert: "Move mug 0.3cm toward center"
```

#### Performance Profiling

**Execution Analysis:**
```
Rule Performance Profile:
┌─────────────────────┬──────────┬──────────┬──────────┐
│ Component          │ Min (ms) │ Avg (ms) │ Max (ms) │
├─────────────────────┼──────────┼──────────┼──────────┤
│ Condition parsing   │    1.2   │    2.1   │    4.3   │
│ Distance calculation│    5.4   │    8.7   │   15.2   │
│ Threshold comparison│    0.1   │    0.2   │    0.5   │
│ Action execution   │    2.1   │    4.5   │   12.8   │
│ Total rule time    │    8.8   │   15.5   │   32.8   │
└─────────────────────┴──────────┴──────────┴──────────┘
```

This comprehensive rules management guide provides everything needed to create, test, optimize, and maintain positioning rules effectively. The combination of natural language processing and manual configuration options ensures flexibility for all use cases while maintaining system reliability and performance.