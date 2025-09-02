# User Manual

Complete guide to using the H200 Intelligent Mug Positioning System.

## Table of Contents

1. [System Overview](#system-overview)
2. [Getting Started](#getting-started)
3. [Image Analysis](#image-analysis)
4. [Rules Management](#rules-management)
5. [Dashboard and Monitoring](#dashboard-and-monitoring)
6. [Server Management](#server-management)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)

## System Overview

The H200 Intelligent Mug Positioning System is an AI-powered solution that analyzes images to detect mugs and provides intelligent positioning feedback based on configurable rules.

### Key Features

- **AI-Powered Detection**: Advanced computer vision models detect mugs with high accuracy
- **Intelligent Positioning**: Analyzes spatial relationships and positioning quality
- **Dynamic Rules**: Create positioning rules using natural language
- **Real-Time Feedback**: Instant analysis results with actionable suggestions
- **Dual GPU Modes**: Serverless auto-scaling and timed instances
- **Comprehensive Monitoring**: Full system visibility and performance tracking

### Use Cases

- **Quality Control**: Ensure consistent mug positioning in photography
- **Training**: Help staff learn proper mug placement techniques
- **Automation**: Integrate positioning checks into production workflows
- **Research**: Analyze positioning patterns and optimize layouts

## Getting Started

### System Requirements

**Minimum:**
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection for cloud deployment
- Valid user account with appropriate permissions

**For Local Development:**
- Python 3.11+
- Docker Desktop
- 8GB RAM minimum, 16GB recommended
- GPU with 8GB VRAM for optimal performance

### User Roles and Permissions

The system supports role-based access control:

- **Viewer**: View analysis results and dashboard
- **Analyst**: Perform image analysis and view results
- **Manager**: Create/edit rules and manage system settings
- **Administrator**: Full system access including server management

### First Login

1. Navigate to the system URL
2. Enter your credentials
3. Complete any required setup steps
4. Review the dashboard overview

## Image Analysis

### Supported Image Formats

- **JPEG** (.jpg, .jpeg) - Recommended for photographs
- **PNG** (.png) - Good for high-quality images
- **WebP** (.webp) - Modern format with good compression

### Image Requirements

- **Resolution**: 640x480 minimum, 1920x1080 recommended
- **File Size**: Maximum 10MB per image
- **Quality**: Well-lit images with clear mug visibility
- **Angle**: Front-facing or slightly angled views work best

### Basic Image Analysis

#### Using the Web Interface

1. **Navigate to Analysis Page**
   - Click "Analysis" in the main navigation
   - Select "Upload Image" tab

2. **Upload Your Image**
   - Drag and drop image or click "Choose File"
   - Wait for upload confirmation
   - Preview will appear automatically

3. **Configure Analysis Settings**
   - **Confidence Threshold**: 0.7 (default) - 0.9 (strict)
   - **Include Feedback**: Enable for positioning suggestions
   - **Rules Context**: Optional context for rule evaluation

4. **Run Analysis**
   - Click "Analyze Image"
   - Wait for processing (typically 1-3 seconds)
   - Review results in the analysis panel

#### Analysis Results

The analysis provides comprehensive information:

**Detection Results:**
- Number of mugs detected
- Confidence scores for each detection
- Bounding box coordinates
- Mug attributes (color, size, material if detectable)

**Positioning Analysis:**
- Position quality assessment
- Offset from ideal position (pixels and millimeters)
- Confidence in positioning analysis
- Rule violations (if rules are active)

**Feedback and Suggestions:**
- Detailed explanation of analysis
- Specific improvement suggestions
- Action items for better positioning

### Advanced Analysis Options

#### Calibration

For accurate measurements, calibrate the system:

1. **Automatic Calibration**
   - Include a reference object (coin, ruler) in the image
   - System will attempt automatic calibration
   - Verify calibration accuracy in results

2. **Manual Calibration**
   - Measure a known distance in your setup
   - Enter mm per pixel ratio in analysis settings
   - Use consistent camera position for accuracy

#### Batch Processing

Process multiple images efficiently:

1. **Prepare Images**
   - Name files consistently
   - Ensure consistent lighting and angles
   - Keep file sizes reasonable

2. **Upload Batch**
   - Use the batch analysis interface
   - Select multiple images (up to 10)
   - Configure shared settings

3. **Review Results**
   - Results appear as batch completes
   - Export results to CSV or JSON
   - Compare results across images

### Using the API

For programmatic access:

```python
import requests

# Analyze single image
with open('mug_image.jpg', 'rb') as image_file:
    response = requests.post(
        'http://localhost:8000/api/v1/analyze/with-feedback',
        headers={'Authorization': 'Bearer YOUR_TOKEN'},
        files={'image': image_file},
        data={
            'include_feedback': True,
            'confidence_threshold': 0.8
        }
    )
    
    result = response.json()
    print(f"Detected {len(result['detections'])} mugs")
    print(f"Position quality: {result['positioning']['position']}")
```

## Rules Management

Rules define positioning criteria and provide automated feedback.

### Understanding Rules

**Rule Components:**
- **Conditions**: What to check (distance, alignment, etc.)
- **Actions**: What to do when conditions are met
- **Priority**: Rule importance (low, medium, high, critical)
- **Context**: When to apply the rule

### Creating Rules with Natural Language

The easiest way to create rules:

1. **Navigate to Rules**
   - Go to Rules section in navigation
   - Click "Create Rule" button
   - Select "Natural Language" tab

2. **Describe Your Rule**
   ```
   "The mug should be centered on the coaster with at least 1 inch clearance from all edges"
   ```

3. **Add Context (Optional)**
   - Specify when rule applies
   - Examples: "coffee shop setup", "formal dining", "photography"

4. **Review Interpretation**
   - System shows how it understood your rule
   - Confidence score indicates interpretation quality
   - Edit if necessary

5. **Enable Rule**
   - Toggle "Auto-enable" for immediate use
   - Rules can be enabled/disabled later

### Advanced Rule Creation

For precise control, create rules manually:

#### Positioning Rules

**Center Alignment:**
```json
{
  "name": "Center mug on coaster",
  "conditions": [
    {
      "field": "distance_from_center",
      "operator": "less_than",
      "value": 1.0,
      "unit": "cm"
    }
  ],
  "actions": [
    {
      "type": "alert",
      "parameters": {
        "message": "Mug should be centered on coaster"
      }
    }
  ]
}
```

**Edge Distance:**
```json
{
  "name": "Minimum edge clearance",
  "conditions": [
    {
      "field": "distance_to_edge",
      "operator": "greater_than",
      "value": 2.5,
      "unit": "cm"
    }
  ],
  "actions": [
    {
      "type": "alert",
      "parameters": {
        "message": "Mug too close to edge",
        "severity": "warning"
      }
    }
  ]
}
```

#### Spacing Rules

**Multi-Mug Spacing:**
```json
{
  "name": "Minimum mug separation",
  "conditions": [
    {
      "field": "minimum_distance_between_mugs",
      "operator": "greater_than",
      "value": 5.0,
      "unit": "cm"
    }
  ],
  "actions": [
    {
      "type": "alert",
      "parameters": {
        "message": "Mugs should be at least 5cm apart"
      }
    }
  ]
}
```

### Rule Management

#### Listing Rules

View all rules with filtering:
- **Status**: Active, Inactive, All
- **Type**: Positioning, Spacing, Safety
- **Priority**: Filter by importance level

#### Editing Rules

1. **Select Rule**: Click on rule name
2. **Modify Settings**: Edit name, conditions, actions
3. **Test Changes**: Use "Test Rule" feature
4. **Save Updates**: Confirm changes

#### Rule Testing

Before applying rules to production:

1. **Test Data**: Use representative test images
2. **Dry Run**: Run analysis with test rules
3. **Review Results**: Check for false positives/negatives
4. **Adjust**: Fine-tune conditions and thresholds

### Rule Evaluation

Understanding how rules are evaluated:

#### Evaluation Process

1. **Image Analysis**: Detect mugs and extract features
2. **Rule Loading**: Load applicable rules for context
3. **Condition Checking**: Evaluate each rule condition
4. **Action Execution**: Trigger actions for matched rules
5. **Result Compilation**: Generate feedback and suggestions

#### Rule Priority

When multiple rules match:
- **Critical**: Always shown, may block operations
- **High**: Prominent display in results
- **Medium**: Standard feedback inclusion
- **Low**: Background validation

## Dashboard and Monitoring

The dashboard provides comprehensive system visibility.

### Dashboard Overview

#### Main Sections

1. **System Health**
   - Service status indicators
   - Overall system health score
   - Recent alerts and warnings

2. **Performance Metrics**
   - Processing times and throughput
   - GPU utilization and efficiency
   - Cache hit rates and optimization

3. **Activity Summary**
   - Recent analysis requests
   - User activity patterns
   - System usage trends

4. **Cost Tracking**
   - Resource utilization costs
   - Optimization opportunities
   - Budget tracking and alerts

### Key Metrics

#### Performance Indicators

- **Cold Start Time**: Initial request processing time
- **Warm Start Time**: Subsequent request processing time
- **Image Processing Time**: Core analysis duration
- **API Latency**: End-to-end response time
- **GPU Utilization**: Graphics processing efficiency
- **Cache Hit Rate**: Data caching effectiveness

#### System Health

- **Service Status**: Individual component health
- **Resource Usage**: CPU, memory, disk utilization
- **Error Rates**: Request failure percentages
- **Uptime**: System availability metrics

### Real-Time Monitoring

#### Live Updates

Connect to real-time data streams:

1. **Enable Auto-Refresh**: Updates every 5 seconds
2. **WebSocket Connection**: Live metric streaming
3. **Alert Notifications**: Immediate issue alerts

#### Custom Views

Create personalized monitoring views:

1. **Metric Selection**: Choose relevant KPIs
2. **Time Ranges**: Adjust monitoring periods
3. **Threshold Setting**: Configure alert levels
4. **Export Options**: Save data for analysis

### Alerting System

#### Alert Categories

- **System Alerts**: Service failures, resource exhaustion
- **Performance Alerts**: Latency spikes, efficiency drops
- **Security Alerts**: Authentication failures, suspicious activity
- **Business Alerts**: Usage limits, cost thresholds

#### Alert Management

1. **Configuration**: Set thresholds and conditions
2. **Channels**: Email, webhook, dashboard notifications
3. **Escalation**: Multi-level alert routing
4. **Suppression**: Temporary alert silencing

## Server Management

Control and monitor GPU server deployments.

### Server Types

#### Serverless Deployment
- **Auto-scaling**: Scales from 0 to configured maximum
- **Cost-efficient**: Pay only for actual usage
- **Cold starts**: 500ms-2s initial request delay
- **Best for**: Variable workloads, development, testing

#### Timed Instance
- **Always-on**: Dedicated GPU instance
- **Consistent performance**: No cold start delays
- **Fixed cost**: Hourly billing regardless of usage  
- **Best for**: Production workloads, continuous processing

### Server Control

#### Starting Servers

1. **Select Server Type**: Serverless or Timed
2. **Configure Resources**: GPU type, memory, instances
3. **Deploy**: Initiate server creation
4. **Monitor**: Watch deployment progress
5. **Verify**: Test server health and performance

#### Server Configuration

**Serverless Settings:**
- **Min Instances**: 0 (scales to zero when idle)
- **Max Instances**: 1-10 (based on expected load)
- **Idle Timeout**: 5-30 minutes before scaling down
- **GPU Type**: H100, H200, A100 options

**Timed Instance Settings:**
- **GPU Type**: Select based on workload requirements
- **Duration**: 1-24 hour increments
- **Auto-renewal**: Extend automatically
- **Resource allocation**: CPU, memory, storage

#### Monitoring Servers

Track server performance and health:

**Key Metrics:**
- **Uptime**: Server availability percentage
- **Request Count**: Number of processed requests
- **Response Time**: Average processing latency
- **Error Rate**: Failed request percentage
- **GPU Utilization**: Processing efficiency
- **Memory Usage**: Resource consumption

**Health Checks:**
- **Endpoint Status**: API availability
- **Model Loading**: AI model readiness
- **Database Connection**: Backend connectivity
- **Cache Performance**: Caching system status

### Cost Optimization

#### Auto-Shutdown

Reduce costs with intelligent shutdown:

1. **Idle Detection**: Monitor request activity
2. **Grace Period**: Wait before shutdown
3. **Safe Shutdown**: Complete in-progress requests
4. **Quick Restart**: Fast recovery when needed

#### Resource Sizing

Choose appropriate resources:

- **Development**: Smaller instances, shorter timeouts
- **Testing**: Medium instances, moderate scaling
- **Production**: Larger instances, higher availability

#### Usage Monitoring

Track costs and optimization opportunities:

1. **Resource Utilization**: Identify underused capacity
2. **Peak Analysis**: Understand usage patterns
3. **Cost Attribution**: Track expenses by team/project
4. **Optimization Suggestions**: Automated recommendations

## Advanced Features

### Integration Options

#### Webhook Notifications

Receive real-time updates:

```json
{
  "event": "analysis_complete",
  "timestamp": "2025-09-02T10:30:00Z",
  "data": {
    "request_id": "req_123",
    "user_id": "user_456",
    "detections": 2,
    "rule_violations": 0,
    "processing_time_ms": 450
  },
  "callback_url": "https://your-app.com/webhooks/h200"
}
```

#### API Integration

Embed analysis in your applications:

```javascript
// JavaScript example
async function analyzeMugImage(imageFile) {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('include_feedback', 'true');
    
    const response = await fetch('/api/v1/analyze/with-feedback', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`
        },
        body: formData
    });
    
    return await response.json();
}
```

#### MCP Protocol Support

For agentic AI integration:

```python
# MCP client example
from src.core.mcp.client import H200MCPClient

async def analyze_with_mcp():
    client = H200MCPClient()
    
    # Analyze image through MCP
    result = await client.analyze_image(
        image_path="mug_photo.jpg",
        rules_context="coffee_shop"
    )
    
    return result.positioning_feedback
```

### Customization Options

#### Custom Models

Deploy specialized detection models:

1. **Model Training**: Train on your specific use case
2. **Model Registration**: Add to model manager
3. **Model Testing**: Validate performance
4. **Model Deployment**: Activate for production

#### Custom Rules Engine

Extend rule capabilities:

1. **Custom Conditions**: Add new evaluation criteria
2. **Custom Actions**: Create specialized responses
3. **Custom Contexts**: Define application-specific scenarios
4. **Custom Validators**: Add business logic validation

### Performance Optimization

#### Caching Strategy

Optimize response times:

- **Model Caching**: Keep models loaded in GPU memory
- **Result Caching**: Cache analysis results for identical images
- **Rule Caching**: Cache compiled rule evaluations
- **Asset Caching**: Cache static resources

#### Batch Processing

Process multiple images efficiently:

1. **Batch Collection**: Group related images
2. **Parallel Processing**: Utilize multiple GPU streams
3. **Result Aggregation**: Combine and summarize results
4. **Optimized Scheduling**: Balance load across resources

## Best Practices

### Image Quality Guidelines

**Lighting:**
- Use consistent, well-distributed lighting
- Avoid harsh shadows or backlighting  
- Natural light or balanced artificial lighting works best

**Camera Position:**
- Position camera perpendicular to surface
- Maintain consistent distance and angle
- Use tripod for stability in repeated setups

**Background:**
- Use neutral, contrasting backgrounds
- Avoid busy patterns or competing objects
- Ensure sufficient contrast between mug and background

### Rule Design Principles

**Start Simple:**
- Begin with basic positioning rules
- Add complexity gradually based on results
- Test thoroughly before enabling in production

**Be Specific:**
- Use precise measurements and thresholds
- Provide clear, actionable feedback messages
- Consider edge cases and exceptions

**Monitor Performance:**
- Track rule effectiveness and accuracy
- Adjust thresholds based on real-world data
- Remove or modify underperforming rules

### System Maintenance

#### Regular Tasks

**Daily:**
- Monitor system health and alerts
- Review performance metrics
- Check error logs for issues

**Weekly:**
- Analyze usage patterns and trends
- Review and optimize active rules
- Update system documentation

**Monthly:**
- Perform cost optimization review
- Evaluate system capacity and scaling
- Plan feature updates and improvements

#### Backup and Recovery

**Data Backup:**
- Automated daily database backups
- Rule configurations stored in version control
- Image analysis results archived regularly

**Disaster Recovery:**
- Multi-region deployment for high availability
- Automated failover procedures
- Regular recovery testing and validation

### Security Considerations

**Access Control:**
- Use strong authentication methods
- Implement role-based access control
- Regularly review and update permissions

**Data Protection:**
- Encrypt sensitive data at rest and in transit
- Implement data retention policies
- Comply with privacy regulations (GDPR, CCPA)

**Network Security:**
- Use TLS for all communications
- Implement proper firewall rules
- Monitor for suspicious activity

This completes the comprehensive user manual. For specific technical details, refer to the [API Documentation](../api/README.md) or [Developer Guides](../developer-guides/README.md).