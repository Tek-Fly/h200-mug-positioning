"""Logging middleware for request/response tracking."""

import json
import logging
import time
import uuid
from typing import Dict, Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process the request and log details."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        request_log = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }
        
        # Add user info if available
        if hasattr(request.state, "user_id"):
            request_log["user_id"] = request.state.user_id
        
        logger.info(f"Request started: {json.dumps(request_log)}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            response_log = {
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_seconds": round(duration, 3),
            }
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(duration, 3))
            
            logger.info(f"Request completed: {json.dumps(response_log)}")
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            error_log = {
                "request_id": request_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_seconds": round(duration, 3),
            }
            
            logger.error(f"Request failed: {json.dumps(error_log)}", exc_info=True)
            
            # Re-raise the exception
            raise


def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", "unknown")