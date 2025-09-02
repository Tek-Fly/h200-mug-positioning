# Error Codes Reference

Complete reference for all error codes and messages in the H200 Intelligent Mug Positioning System.

## Error Format

All API errors follow a consistent format:

```json
{
  "error": {
    "code": 400,
    "message": "Invalid request parameters",
    "type": "validation_error", 
    "details": [
      {
        "field": "confidence_threshold",
        "message": "Must be between 0.0 and 1.0",
        "value": 1.5
      }
    ],
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-09-02T10:30:00Z",
    "documentation_url": "https://docs.tekfly.co.uk/errors/validation_error"
  }
}
```

## HTTP Status Codes

### 2xx Success Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 202 | Accepted | Request accepted for processing |
| 204 | No Content | Request successful, no content returned |

### 4xx Client Error Codes

| Code | Status | Description | Action |
|------|--------|-------------|---------|
| 400 | Bad Request | Invalid request format or parameters | Check request format and parameters |
| 401 | Unauthorized | Missing or invalid authentication | Provide valid JWT token |
| 403 | Forbidden | Insufficient permissions | Check user permissions |
| 404 | Not Found | Resource not found | Verify resource exists |
| 405 | Method Not Allowed | HTTP method not supported | Use correct HTTP method |
| 409 | Conflict | Resource conflict (duplicate, etc.) | Resolve resource conflict |
| 413 | Payload Too Large | Request body too large | Reduce payload size |
| 415 | Unsupported Media Type | Invalid content type | Use supported content type |
| 422 | Unprocessable Entity | Valid format but semantic errors | Fix semantic issues |
| 429 | Too Many Requests | Rate limit exceeded | Reduce request frequency |

### 5xx Server Error Codes

| Code | Status | Description | Action |
|------|--------|-------------|---------|
| 500 | Internal Server Error | Unexpected server error | Check server logs, contact support |
| 502 | Bad Gateway | Upstream service error | Check service dependencies |
| 503 | Service Unavailable | Service temporarily unavailable | Retry after delay |
| 504 | Gateway Timeout | Request timeout | Increase timeout or retry |

## Application Error Types

### Authentication Errors (AUTH_*)

#### AUTH_001: Invalid Token
```json
{
  "code": 401,
  "type": "auth_invalid_token",
  "message": "JWT token is invalid or malformed",
  "details": {
    "token_error": "Invalid signature",
    "suggestion": "Generate a new token using /api/auth/login"
  }
}
```

#### AUTH_002: Token Expired
```json
{
  "code": 401,
  "type": "auth_token_expired", 
  "message": "JWT token has expired",
  "details": {
    "expired_at": "2025-09-01T10:30:00Z",
    "suggestion": "Refresh token or re-authenticate"
  }
}
```

#### AUTH_003: Insufficient Permissions
```json
{
  "code": 403,
  "type": "auth_insufficient_permissions",
  "message": "User lacks required permissions for this operation",
  "details": {
    "required_permission": "rules:write",
    "user_permissions": ["rules:read", "analysis:read"]
  }
}
```

#### AUTH_004: Invalid Credentials
```json
{
  "code": 401,
  "type": "auth_invalid_credentials",
  "message": "Username or password is incorrect"
}
```

### Validation Errors (VALIDATION_*)

#### VALIDATION_001: Required Field Missing
```json
{
  "code": 422,
  "type": "validation_required_field",
  "message": "Required field is missing",
  "details": {
    "field": "image",
    "type": "file",
    "description": "Image file to analyze"
  }
}
```

#### VALIDATION_002: Invalid Field Value
```json
{
  "code": 422,
  "type": "validation_invalid_value",
  "message": "Field value is invalid",
  "details": {
    "field": "confidence_threshold",
    "value": 1.5,
    "expected": "Number between 0.0 and 1.0",
    "constraint": "0.0 <= value <= 1.0"
  }
}
```

#### VALIDATION_003: Invalid File Type
```json
{
  "code": 422,
  "type": "validation_invalid_file_type",
  "message": "Uploaded file type is not supported",
  "details": {
    "received_type": "text/plain",
    "supported_types": ["image/jpeg", "image/png", "image/webp"],
    "suggestion": "Upload a valid image file"
  }
}
```

#### VALIDATION_004: File Too Large
```json
{
  "code": 413,
  "type": "validation_file_too_large",
  "message": "Uploaded file exceeds size limit",
  "details": {
    "file_size_bytes": 15728640,
    "max_size_bytes": 10485760,
    "max_size_human": "10 MB"
  }
}
```

#### VALIDATION_005: Invalid Image Content
```json
{
  "code": 422,
  "type": "validation_invalid_image",
  "message": "Image file is corrupted or invalid",
  "details": {
    "validation_error": "Cannot identify image format",
    "suggestion": "Upload a valid, uncorrupted image file"
  }
}
```

### Analysis Errors (ANALYSIS_*)

#### ANALYSIS_001: Model Not Available
```json
{
  "code": 503,
  "type": "analysis_model_unavailable",
  "message": "Analysis model is not available",
  "details": {
    "model_name": "yolo_v8n",
    "status": "loading",
    "suggestion": "Wait for model to finish loading and retry"
  }
}
```

#### ANALYSIS_002: GPU Memory Error
```json
{
  "code": 500,
  "type": "analysis_gpu_memory_error",
  "message": "Insufficient GPU memory for analysis",
  "details": {
    "required_memory_mb": 2048,
    "available_memory_mb": 1024,
    "suggestion": "Reduce batch size or clear GPU cache"
  }
}
```

#### ANALYSIS_003: Processing Timeout
```json
{
  "code": 504,
  "type": "analysis_processing_timeout", 
  "message": "Image analysis timed out",
  "details": {
    "timeout_seconds": 30,
    "processing_time_seconds": 35,
    "suggestion": "Try with a smaller image or contact support"
  }
}
```

#### ANALYSIS_004: No Detections Found
```json
{
  "code": 200,
  "type": "analysis_no_detections",
  "message": "No mugs detected in the image",
  "details": {
    "confidence_threshold": 0.7,
    "suggestion": "Lower confidence threshold or use clearer image"
  }
}
```

#### ANALYSIS_005: Analysis Failed
```json
{
  "code": 500,
  "type": "analysis_processing_failed",
  "message": "Analysis processing failed",
  "details": {
    "stage": "object_detection",
    "error": "CUDA out of memory",
    "suggestion": "Retry request or contact support if issue persists"
  }
}
```

### Rules Engine Errors (RULES_*)

#### RULES_001: Rule Not Found
```json
{
  "code": 404,
  "type": "rules_not_found",
  "message": "Rule with specified ID not found",
  "details": {
    "rule_id": "550e8400-e29b-41d4-a716-446655440000",
    "suggestion": "Check rule ID and ensure rule exists"
  }
}
```

#### RULES_002: Invalid Rule Definition
```json
{
  "code": 422,
  "type": "rules_invalid_definition",
  "message": "Rule definition contains errors",
  "details": {
    "validation_errors": [
      {
        "path": "conditions[0].operator",
        "error": "Invalid operator 'equals_exactly'",
        "valid_operators": ["equals", "not_equals", "greater_than"]
      }
    ]
  }
}
```

#### RULES_003: Rule Compilation Failed
```json
{
  "code": 500,
  "type": "rules_compilation_failed",
  "message": "Failed to compile rule for execution",
  "details": {
    "rule_id": "rule_123",
    "compilation_error": "Circular dependency detected",
    "suggestion": "Simplify rule conditions or contact support"
  }
}
```

#### RULES_004: Natural Language Parse Error
```json
{
  "code": 422,
  "type": "rules_nlp_parse_error",
  "message": "Could not parse natural language rule description",
  "details": {
    "input_text": "The mug should be something somewhere",
    "confidence": 0.2,
    "suggestion": "Provide more specific rule description with clear conditions"
  }
}
```

#### RULES_005: Rule Evaluation Error
```json
{
  "code": 500,
  "type": "rules_evaluation_error",
  "message": "Error occurred while evaluating rule",
  "details": {
    "rule_id": "rule_456",
    "evaluation_stage": "condition_check",
    "error": "Division by zero in distance calculation"
  }
}
```

### Database Errors (DB_*)

#### DB_001: Connection Failed
```json
{
  "code": 503,
  "type": "db_connection_failed",
  "message": "Database connection failed",
  "details": {
    "database": "mongodb",
    "error": "Connection timeout",
    "suggestion": "Check database connectivity and credentials"
  }
}
```

#### DB_002: Query Failed
```json
{
  "code": 500,
  "type": "db_query_failed",
  "message": "Database query execution failed",
  "details": {
    "operation": "find",
    "collection": "analysis_results",
    "error": "Index not found",
    "suggestion": "Contact support if issue persists"
  }
}
```

#### DB_003: Duplicate Key Error
```json
{
  "code": 409,
  "type": "db_duplicate_key",
  "message": "Resource already exists with this identifier",
  "details": {
    "field": "request_id",
    "value": "req_123",
    "suggestion": "Use unique identifier or update existing resource"
  }
}
```

#### DB_004: Document Not Found
```json
{
  "code": 404,
  "type": "db_document_not_found",
  "message": "Requested document not found in database",
  "details": {
    "collection": "users",
    "query": {"user_id": "user_456"},
    "suggestion": "Verify document ID and ensure it exists"
  }
}
```

### Cache Errors (CACHE_*)

#### CACHE_001: Connection Failed
```json
{
  "code": 503,
  "type": "cache_connection_failed",
  "message": "Cache server connection failed",
  "details": {
    "cache_type": "redis",
    "error": "Connection refused",
    "impact": "Performance may be degraded"
  }
}
```

#### CACHE_002: Memory Full
```json
{
  "code": 507,
  "type": "cache_memory_full",
  "message": "Cache memory limit exceeded",
  "details": {
    "used_memory_mb": 4096,
    "max_memory_mb": 4096,
    "suggestion": "Clear cache or increase memory limit"
  }
}
```

### Storage Errors (STORAGE_*)

#### STORAGE_001: Upload Failed
```json
{
  "code": 500,
  "type": "storage_upload_failed",
  "message": "Failed to upload file to storage",
  "details": {
    "storage_backend": "cloudflare_r2",
    "error": "Access denied",
    "suggestion": "Check storage credentials and permissions"
  }
}
```

#### STORAGE_002: File Not Found
```json
{
  "code": 404,
  "type": "storage_file_not_found",
  "message": "Requested file not found in storage",
  "details": {
    "file_path": "analysis-results/2025/09/02/result_123.json",
    "storage_backend": "cloudflare_r2"
  }
}
```

#### STORAGE_003: Insufficient Space
```json
{
  "code": 507,
  "type": "storage_insufficient_space",
  "message": "Insufficient storage space available",
  "details": {
    "required_space_mb": 500,
    "available_space_mb": 100,
    "suggestion": "Clean up old files or increase storage quota"
  }
}
```

### Server Management Errors (SERVER_*)

#### SERVER_001: Server Not Found
```json
{
  "code": 404,
  "type": "server_not_found",
  "message": "Server with specified ID not found",
  "details": {
    "server_id": "srv_123",
    "server_type": "serverless",
    "suggestion": "Check server ID and ensure server exists"
  }
}
```

#### SERVER_002: Server Creation Failed
```json
{
  "code": 500,
  "type": "server_creation_failed",
  "message": "Failed to create server instance",
  "details": {
    "server_type": "timed",
    "gpu_type": "H200",
    "error": "No available GPU resources",
    "suggestion": "Try different GPU type or wait for resources"
  }
}
```

#### SERVER_003: Server Operation Failed
```json
{
  "code": 500,
  "type": "server_operation_failed",
  "message": "Server operation could not be completed",
  "details": {
    "server_id": "srv_456",
    "operation": "start",
    "error": "Server in invalid state",
    "current_state": "error"
  }
}
```

#### SERVER_004: Resource Limit Exceeded
```json
{
  "code": 429,
  "type": "server_resource_limit",
  "message": "Server resource limit exceeded",
  "details": {
    "limit_type": "concurrent_instances",
    "current_count": 10,
    "max_allowed": 10,
    "suggestion": "Stop unused instances or contact support for limit increase"
  }
}
```

## Rate Limiting Errors

### Rate Limit Headers

When rate limits are applied, the following headers are included:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 23
X-RateLimit-Reset: 1693737600
X-RateLimit-Window: 60
```

### RATE_001: Rate Limit Exceeded
```json
{
  "code": 429,
  "type": "rate_limit_exceeded",
  "message": "API rate limit exceeded",
  "details": {
    "limit": 100,
    "window_seconds": 60,
    "retry_after_seconds": 45,
    "suggestion": "Reduce request frequency or contact support for higher limits"
  }
}
```

## System Status Errors

### SYSTEM_001: Service Unavailable
```json
{
  "code": 503,
  "type": "system_service_unavailable",
  "message": "Service is temporarily unavailable",
  "details": {
    "service": "analysis_engine",
    "status": "maintenance",
    "estimated_recovery": "2025-09-02T11:00:00Z",
    "suggestion": "Wait for maintenance to complete and retry"
  }
}
```

### SYSTEM_002: Overloaded
```json
{
  "code": 503,
  "type": "system_overloaded",
  "message": "System is currently overloaded",
  "details": {
    "current_load": 95,
    "max_load": 80,
    "queue_length": 50,
    "suggestion": "Retry request after a short delay"
  }
}
```

## Error Handling Best Practices

### 1. Client-Side Error Handling

```javascript
// JavaScript example
async function handleApiRequest(url, options) {
  try {
    const response = await fetch(url, options);
    
    if (!response.ok) {
      const error = await response.json();
      throw new ApiError(error);
    }
    
    return await response.json();
    
  } catch (error) {
    if (error instanceof ApiError) {
      handleApiError(error);
    } else {
      handleNetworkError(error);
    }
    throw error;
  }
}

function handleApiError(error) {
  switch (error.type) {
    case 'auth_token_expired':
      // Redirect to login
      window.location.href = '/login';
      break;
      
    case 'validation_invalid_value':
      // Show field validation error
      showFieldError(error.details.field, error.details.message);
      break;
      
    case 'rate_limit_exceeded':
      // Show rate limit message and retry after delay
      showRateLimitWarning(error.details.retry_after_seconds);
      break;
      
    case 'analysis_model_unavailable':
      // Show loading message and retry
      showLoadingMessage('Models are starting up...');
      setTimeout(() => retryRequest(), 5000);
      break;
      
    default:
      showGenericError(error.message);
  }
}
```

### 2. Server-Side Error Handling

```python
# Python example
from fastapi import HTTPException
from typing import Dict, Any, Optional

class H200Exception(Exception):
    """Base exception for H200 system."""
    
    def __init__(
        self,
        message: str,
        error_type: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

class ValidationError(H200Exception):
    """Validation error."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        details = {}
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = value
            
        super().__init__(
            message=message,
            error_type='validation_invalid_value',
            status_code=422,
            details=details
        )

# Error handler
@app.exception_handler(H200Exception)
async def h200_exception_handler(request: Request, exc: H200Exception):
    """Handle custom H200 exceptions."""
    
    error_response = {
        'error': {
            'code': exc.status_code,
            'message': exc.message,
            'type': exc.error_type,
            'details': exc.details,
            'request_id': getattr(request.state, 'request_id', None),
            'timestamp': datetime.utcnow().isoformat(),
            'documentation_url': f'https://docs.tekfly.co.uk/errors/{exc.error_type}'
        }
    }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )
```

### 3. Retry Logic

```python
# Exponential backoff retry logic
import asyncio
import random
from typing import Callable, Any

async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_errors: tuple = (ConnectionError, TimeoutError)
) -> Any:
    """Retry function with exponential backoff."""
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
            
        except Exception as e:
            if attempt == max_retries or not isinstance(e, retryable_errors):
                raise
                
            # Calculate delay with jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)  # 10% jitter
            
            await asyncio.sleep(delay + jitter)
    
    raise RuntimeError(f"Max retries ({max_retries}) exceeded")

# Usage
try:
    result = await retry_with_backoff(
        lambda: analyze_image(image_data),
        max_retries=3,
        retryable_errors=(AnalysisError, GPUMemoryError)
    )
except Exception as e:
    handle_final_error(e)
```

## Error Monitoring and Alerting

### Error Tracking

Monitor these error patterns for system health:

- **High error rates**: > 5% of requests failing
- **Specific error spikes**: Sudden increase in specific error types
- **Critical errors**: GPU memory errors, database connection failures
- **User impact**: Authentication and validation errors affecting users

### Alert Thresholds

```yaml
# Error-based alerts
error_alerts:
  high_error_rate:
    threshold: 0.05  # 5%
    window: "5m"
    severity: critical
    
  gpu_memory_errors:
    threshold: 10  # 10 errors
    window: "1m" 
    severity: critical
    
  authentication_failures:
    threshold: 100  # 100 failures
    window: "5m"
    severity: warning
    
  validation_errors:
    threshold: 50  # 50 errors
    window: "5m"
    severity: warning
```

This error codes reference provides comprehensive information for troubleshooting and handling all types of errors that may occur in the H200 System.