# H200 External Integrations

This module provides integration clients for external services used by the H200 Intelligent Mug Positioning System.

## Available Integrations

### 1. Templated.io Client

The `TemplatedClient` provides integration with Templated.io for design rendering based on mug positioning results.

**Features:**
- Render designs using positioning data
- Template management
- Batch rendering support
- Automatic retry logic

**Configuration:**
- `TEMPLATED_API_KEY`: API key for Templated.io

**Example Usage:**
```python
from src.integrations import TemplatedClient

async with TemplatedClient() as client:
    # Render design based on positioning
    result = await client.render_design(
        template_id="mug-template-1",
        positioning_data={
            "mug_position": {"x": 100, "y": 200},
            "confidence": 0.95,
            "template_params": {
                "text": "Perfect Position!",
                "color": "#00FF00"
            }
        }
    )
```

### 2. N8N Client

The `N8NClient` provides webhook integration for workflow automation with N8N.

**Features:**
- Send events to N8N workflows
- Support for different event types
- Batch event sending
- Event queuing with auto-flush

**Configuration:**
- `N8N_WEBHOOK_URL`: N8N webhook endpoint URL

**Event Types:**
- `MUG_DETECTED`: When mugs are detected in an image
- `POSITIONING_ANALYZED`: When positioning analysis is complete
- `RULE_TRIGGERED`: When a positioning rule is triggered
- `ERROR_OCCURRED`: When an error occurs
- `SYSTEM_ALERT`: For system-level alerts
- `BATCH_COMPLETED`: When batch processing is done
- `MODEL_UPDATED`: When ML models are updated

**Example Usage:**
```python
from src.integrations import N8NClient
from src.integrations.n8n import N8NEventType

async with N8NClient() as client:
    # Send mug detection event
    await client.send_mug_detection(
        image_id="img_123",
        mug_count=3,
        positions=[...],
        confidence_scores=[0.95, 0.92, 0.88],
        processing_time=0.245
    )
    
    # Queue events for batch sending
    await client.queue_event(
        N8NEventType.SYSTEM_ALERT,
        {"type": "performance", "metric": "gpu_usage", "value": 85.5}
    )
    await client.flush_events()
```

### 3. Notification Client

The `NotificationClient` provides multi-channel notifications for alerts and updates.

**Features:**
- Multiple notification channels (Email, Webhook, Slack, Discord)
- Priority-based routing
- Rich formatting for each channel
- Batch notification support

**Configuration:**
- Email:
  - `SMTP_HOST`: SMTP server host
  - `SMTP_USERNAME`: SMTP username
  - `SMTP_PASSWORD`: SMTP password
  - `EMAIL_FROM`: Sender email address
  - `EMAIL_TO`: Comma-separated recipient emails
- Webhook:
  - `WEBHOOK_URL`: Generic webhook URL
- Slack:
  - `SLACK_WEBHOOK_URL`: Slack incoming webhook URL
- Discord:
  - `DISCORD_WEBHOOK_URL`: Discord webhook URL

**Notification Types:**
- `INFO`: Informational messages
- `WARNING`: Warning messages
- `ERROR`: Error alerts
- `CRITICAL`: Critical system alerts
- `SUCCESS`: Success notifications

**Example Usage:**
```python
from src.integrations import NotificationClient
from src.integrations.notifications import NotificationType, NotificationChannel

async with NotificationClient() as client:
    # Send critical alert to all channels
    await client.send_notification(
        title="Critical Error",
        message="GPU memory exceeded threshold",
        notification_type=NotificationType.CRITICAL,
        data={"gpu_usage": 95.5, "threshold": 90}
    )
    
    # Send to specific channels
    await client.send_notification(
        title="Analysis Complete",
        message="Batch processing finished",
        notification_type=NotificationType.SUCCESS,
        channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
    )
```

## Integration with Other Components

### Using in API Endpoints
```python
from fastapi import BackgroundTasks
from src.integrations import N8NClient, NotificationClient

@app.post("/analyze")
async def analyze_image(background_tasks: BackgroundTasks):
    # Perform analysis...
    
    # Send event to N8N in background
    background_tasks.add_task(
        send_n8n_event,
        event_type="POSITIONING_ANALYZED",
        data=analysis_result
    )
    
    # Send notification if critical
    if analysis_result.confidence < 0.5:
        background_tasks.add_task(
            send_notification,
            "Low Confidence Detection",
            f"Analysis confidence is {analysis_result.confidence}"
        )
```

### Using in Control Plane
```python
from src.integrations import NotificationClient
from src.integrations.notifications import NotificationType

class SystemMonitor:
    def __init__(self):
        self.notification_client = NotificationClient()
    
    async def check_system_health(self):
        metrics = await self.get_metrics()
        
        # Check GPU usage
        if metrics.gpu_usage > 90:
            await self.notification_client.send_performance_alert(
                metric_name="GPU Usage",
                current_value=metrics.gpu_usage,
                threshold=90
            )
```

### Using with Rules Engine
```python
from src.integrations import TemplatedClient
from src.core.rules import RuleEngine

class PositioningPipeline:
    def __init__(self):
        self.templated_client = TemplatedClient()
        self.rule_engine = RuleEngine()
    
    async def process_and_render(self, image_data):
        # Apply rules
        positioning = await self.rule_engine.analyze(image_data)
        
        # Render visualization
        if positioning.needs_visualization:
            render_result = await self.templated_client.render_design(
                template_id="positioning-overlay",
                positioning_data=positioning.to_dict()
            )
            
        return {
            "positioning": positioning,
            "visualization_url": render_result.get("render_url")
        }
```

## Error Handling

All integration clients include:
- Automatic retry with exponential backoff
- Comprehensive error logging
- Graceful degradation when services are unavailable

```python
from src.integrations import NotificationClient
from src.integrations.notifications import NotificationError

try:
    async with NotificationClient() as client:
        await client.send_notification(...)
except NotificationError as e:
    logger.error(f"Failed to send notification: {e}")
    # Continue operation without blocking
```

## Testing

Each integration includes example usage that can be run directly:

```bash
# Test Templated.io integration
python -m src.integrations.templated

# Test N8N integration
python -m src.integrations.n8n

# Test Notification system
python -m src.integrations.notifications
```

## Performance Considerations

1. **Connection Pooling**: All clients use aiohttp session pooling
2. **Async Operations**: All operations are fully async
3. **Timeout Configuration**: Configurable timeouts for all requests
4. **Batch Support**: Batch operations where applicable

## Security

1. **API Keys**: All sensitive credentials loaded from environment
2. **TLS**: All external connections use HTTPS/TLS
3. **Input Validation**: All inputs are validated before sending
4. **No Credential Logging**: Credentials are never logged