# API Data Models

This document describes all data structures used in the H200 Intelligent Mug Positioning System API.

## Analysis Models

### AnalysisRequest

Request model for image analysis.

```json
{
  "image": "file",
  "include_feedback": true,
  "rules_context": "coffee_shop_setup",
  "calibration_mm_per_pixel": 0.5,
  "confidence_threshold": 0.7
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | File | Yes | Image file (JPG, PNG, WebP, max 10MB) |
| `include_feedback` | Boolean | No | Include positioning feedback (default: true) |
| `rules_context` | String | No | Context for rule evaluation |
| `calibration_mm_per_pixel` | Float | No | Pixel to millimeter conversion factor |
| `confidence_threshold` | Float | No | Detection confidence threshold (0.0-1.0, default: 0.7) |

### AnalysisResponse

Response from image analysis.

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-09-02T10:30:00Z",
  "processing_time_ms": 450.5,
  "detections": [MugDetection],
  "positioning": PositioningResult,
  "feedback": "Detailed analysis feedback",
  "suggestions": ["List of improvement suggestions"],
  "metadata": {
    "user_id": "user123",
    "image_size": [1920, 1080],
    "model_version": "yolov8n-1.0"
  }
}
```

### MugDetection

Individual mug detection result.

```json
{
  "id": "det_001",
  "bbox": BoundingBox,
  "attributes": {
    "color": "white",
    "size": "medium",
    "material": "ceramic",
    "confidence": 0.95
  }
}
```

### BoundingBox

Bounding box coordinates with confidence.

```json
{
  "top_left": Point2D,
  "bottom_right": Point2D,
  "confidence": 0.95
}
```

### Point2D

2D coordinate point.

```json
{
  "x": 100.5,
  "y": 75.2
}
```

### PositioningResult

Positioning analysis result.

```json
{
  "position": "slightly off-center",
  "confidence": 0.87,
  "offset_pixels": Point2D,
  "offset_mm": Point2D,
  "rule_violations": [
    "Mug too close to edge (2.3cm < 5cm required)"
  ]
}
```

### AnalysisFeedback

Feedback submission for analysis results.

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "is_correct": true,
  "feedback_type": "positioning",
  "comments": "Detection was accurate",
  "suggested_improvements": [
    "Consider ambient lighting conditions"
  ]
}
```

## Rules Models

### Rule

Complete rule definition.

```json
{
  "id": "rule_123",
  "name": "Center mug on coaster",
  "description": "Mug should be centered on coaster",
  "type": "positioning",
  "priority": "medium",
  "conditions": [RuleCondition],
  "actions": [RuleAction],
  "enabled": true,
  "created_at": "2025-09-02T10:30:00Z",
  "updated_at": "2025-09-02T10:30:00Z",
  "metadata": {
    "created_by": "user123",
    "tags": ["coffee", "positioning"]
  }
}
```

### RuleCondition

Rule evaluation condition.

```json
{
  "field": "distance_from_center",
  "operator": "less_than",
  "value": 2.5,
  "unit": "cm",
  "description": "Distance from coaster center"
}
```

#### Supported Operators

- `equals` / `not_equals`
- `greater_than` / `greater_than_or_equal`
- `less_than` / `less_than_or_equal`
- `contains` / `not_contains`
- `in_range` / `outside_range`

### RuleAction

Action to take when rule condition is met.

```json
{
  "type": "alert",
  "parameters": {
    "message": "Mug positioning violation detected",
    "severity": "warning",
    "notify_user": true
  }
}
```

#### Action Types

- `alert`: Show alert message
- `adjust`: Suggest positioning adjustment
- `log`: Log event
- `webhook`: Send webhook notification
- `email`: Send email notification

### NaturalLanguageRuleRequest

Request to create rule from natural language.

```json
{
  "text": "The mug should be centered on the coaster with 1 inch clearance",
  "context": "coffee_shop_setup",
  "auto_enable": true
}
```

### NaturalLanguageRuleResponse

Response from natural language rule creation.

```json
{
  "rule": Rule,
  "interpretation": "Created positioning rule with 2 conditions",
  "confidence": 0.92,
  "warnings": [
    "Low confidence in distance measurement interpretation"
  ]
}
```

### RuleEvaluationRequest

Request to evaluate rules against data.

```json
{
  "data": {
    "detections": [MugDetection],
    "image_size": [1920, 1080],
    "calibration_factor": 0.5
  },
  "rule_ids": ["rule_123", "rule_456"],
  "include_disabled": false
}
```

### RuleEvaluationResponse

Result of rule evaluation.

```json
{
  "evaluated_count": 5,
  "matched_count": 2,
  "results": [RuleEvaluationResult],
  "summary": {
    "total_rules": 5,
    "matched_rules": 2,
    "match_rate": 0.4
  }
}
```

### RuleEvaluationResult

Individual rule evaluation result.

```json
{
  "rule_id": "rule_123",
  "matched": true,
  "confidence": 0.89,
  "conditions_met": 2,
  "total_conditions": 2,
  "actions_taken": [RuleAction],
  "execution_time_ms": 15.2,
  "details": {
    "distance_from_center": 1.2,
    "threshold": 2.5
  }
}
```

## Dashboard Models

### DashboardResponse

Complete dashboard data.

```json
{
  "timestamp": "2025-09-02T10:30:00Z",
  "overall_health": "healthy",
  "services": [ServiceHealth],
  "performance": PerformanceMetrics,
  "resources": ResourceUsage,
  "costs": CostMetrics,
  "recent_activity": [ActivityLog],
  "summary": {
    "total_requests_today": 12345,
    "average_response_time_ms": 180,
    "error_rate_percent": 0.5,
    "active_users": 42
  },
  "metrics": [SystemMetric]
}
```

### ServiceHealth

Health status of individual service.

```json
{
  "name": "MongoDB",
  "status": "healthy",
  "last_check": "2025-09-02T10:29:45Z",
  "uptime_seconds": 86400,
  "details": {
    "version": "7.0.0",
    "connections": 25,
    "memory_mb": 512
  }
}
```

#### Service Status Values

- `healthy`: Service operating normally
- `degraded`: Service operational with issues
- `unhealthy`: Service not responding
- `unknown`: Status cannot be determined

### PerformanceMetrics

System performance metrics.

```json
{
  "cold_start_ms": 1500.0,
  "warm_start_ms": 50.0,
  "image_processing_ms": 450.0,
  "api_latency_p95_ms": 180.0,
  "gpu_utilization_percent": 75.5,
  "cache_hit_rate": 87.3,
  "requests_per_second": 25.5
}
```

### ResourceUsage

Current system resource usage.

```json
{
  "cpu_percent": 45.2,
  "memory_used_mb": 2048,
  "memory_total_mb": 8192,
  "disk_used_gb": 15.5,
  "disk_total_gb": 100,
  "gpu_memory_used_mb": 12000,
  "gpu_memory_total_mb": 20480
}
```

### CostMetrics

Cost tracking and breakdown.

```json
{
  "period": "daily",
  "compute_cost": 84.0,
  "storage_cost": 0.08,
  "network_cost": 0.90,
  "total_cost": 84.98,
  "cost_per_request": 0.0085,
  "breakdown": {
    "gpu_hours": 24,
    "storage_gb": 100,
    "network_gb": 10,
    "api_calls": 10000
  }
}
```

### ActivityLog

System activity log entry.

```json
{
  "timestamp": "2025-09-02T10:25:00Z",
  "type": "analysis",
  "user_id": "user123",
  "action": "Image analysis completed",
  "details": {
    "request_id": "req_456",
    "processing_time_ms": 450,
    "detections": 2
  },
  "duration_ms": 450
}
```

### SystemMetric

Individual system metric.

```json
{
  "name": "api_requests_total",
  "type": "counter",
  "value": 12345,
  "timestamp": "2025-09-02T10:30:00Z",
  "labels": {
    "endpoint": "/api/v1/analyze",
    "status_code": "200"
  }
}
```

#### Metric Types

- `counter`: Incrementing value
- `gauge`: Current value
- `histogram`: Distribution of values
- `summary`: Statistical summary

## Server Models

### ServerInfo

Server configuration and status.

```json
{
  "id": "srv_123",
  "type": "serverless",
  "state": "running",
  "endpoint": "https://api-123.runpod.ai",
  "health": "healthy",
  "created_at": "2025-09-02T10:00:00Z",
  "last_activity": "2025-09-02T10:29:00Z",
  "config": ServerConfig,
  "metrics": ServerMetrics
}
```

### ServerConfig

Server configuration settings.

```json
{
  "image": "tekfly/h200:serverless-latest",
  "gpu_type": "H100",
  "min_instances": 0,
  "max_instances": 3,
  "idle_timeout": 600,
  "environment": {
    "CUDA_VISIBLE_DEVICES": "0"
  },
  "resources": {
    "cpu_cores": 8,
    "memory_gb": 32,
    "gpu_memory_gb": 80
  }
}
```

### ServerMetrics

Real-time server metrics.

```json
{
  "timestamp": "2025-09-02T10:30:00Z",
  "uptime_seconds": 1800,
  "requests_total": 245,
  "requests_per_minute": 15.2,
  "average_latency_ms": 125.5,
  "error_rate_percent": 0.8,
  "gpu_utilization_percent": 72.3,
  "memory_usage_percent": 45.1,
  "active_connections": 3
}
```

### ServerControlRequest

Server control action request.

```json
{
  "action": "start",
  "force": false,
  "config": ServerConfig
}
```

#### Available Actions

- `start`: Start stopped server
- `stop`: Stop running server  
- `restart`: Restart server
- `scale`: Update configuration (serverless only)

### ServerControlResponse

Server control action response.

```json
{
  "success": true,
  "server": ServerInfo,
  "message": "Server started successfully",
  "duration_seconds": 5.2,
  "warnings": [
    "Server took longer than expected to start"
  ]
}
```

### DeploymentRequest

New server deployment request.

```json
{
  "type": "serverless",
  "config": ServerConfig,
  "auto_start": true,
  "tags": {
    "environment": "production",
    "team": "ai-ops"
  }
}
```

### DeploymentResponse

Server deployment response.

```json
{
  "deployment_id": "dep_789",
  "server": ServerInfo,
  "deployment_time_seconds": 45.8,
  "message": "Server deployed successfully"
}
```

## WebSocket Models

### WebSocket Message

Base structure for all WebSocket messages.

```json
{
  "type": "metrics",
  "timestamp": "2025-09-02T10:30:00Z",
  "data": {}
}
```

### Subscription Message

Client subscription request.

```json
{
  "action": "subscribe",
  "topic": "metrics"
}
```

### Metrics Update

Real-time metrics update.

```json
{
  "type": "metrics",
  "timestamp": "2025-09-02T10:30:00Z",
  "data": {
    "gpu_utilization": 75.5,
    "requests_per_second": 25.5,
    "average_latency_ms": 145,
    "cache_hit_rate": 0.87
  }
}
```

### Log Entry

Live log stream entry.

```json
{
  "type": "log",
  "timestamp": "2025-09-02T10:30:00Z",
  "data": {
    "level": "INFO",
    "message": "Image analysis completed",
    "context": {
      "service": "api",
      "request_id": "req_123",
      "user_id": "user456"
    }
  }
}
```

### Alert Message

System alert notification.

```json
{
  "type": "alert",
  "timestamp": "2025-09-02T10:30:00Z",
  "data": {
    "severity": "warning",
    "title": "High GPU utilization",
    "message": "GPU utilization above 90% for 5 minutes",
    "source": "resource_monitor",
    "tags": ["gpu", "performance"]
  }
}
```

## Error Models

### Error Response

Standard error response format.

```json
{
  "error": {
    "code": 400,
    "message": "Validation error",
    "type": "validation_error",
    "details": [
      {
        "field": "confidence_threshold",
        "message": "Must be between 0.0 and 1.0",
        "value": 1.5
      }
    ],
    "request_id": "req_123",
    "timestamp": "2025-09-02T10:30:00Z"
  }
}
```

### Validation Error Detail

Individual field validation error.

```json
{
  "field": "confidence_threshold",
  "message": "Must be between 0.0 and 1.0",
  "value": 1.5,
  "constraint": "range",
  "constraint_params": {
    "min": 0.0,
    "max": 1.0
  }
}
```

## Authentication Models

### Login Request

User authentication request.

```json
{
  "username": "user@example.com",
  "password": "secure_password"
}
```

### Login Response

Authentication response with JWT token.

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "user123",
    "username": "user@example.com",
    "permissions": ["analyze", "rules:read", "rules:write"]
  }
}
```

## Enums and Constants

### ServerType
- `serverless`: Auto-scaling serverless deployment
- `timed`: Fixed-time GPU instance

### ServerState
- `pending`: Being created
- `starting`: Starting up
- `running`: Active and ready
- `stopping`: Shutting down
- `stopped`: Inactive
- `error`: Error state

### RulePriority
- `low`: Low priority rule
- `medium`: Medium priority rule
- `high`: High priority rule
- `critical`: Critical rule

### ActivityType
- `analysis`: Image analysis activity
- `rule_creation`: Rule management activity
- `server_control`: Server control activity
- `user_login`: Authentication activity

This completes the comprehensive data model documentation for the H200 Intelligent Mug Positioning System API.