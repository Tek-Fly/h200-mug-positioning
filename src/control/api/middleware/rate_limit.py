"""Rate limiting middleware."""

# Standard library imports
import logging
import time
from collections import defaultdict
from typing import Dict, Tuple

# Third-party imports
from fastapi import status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""

    def __init__(self, app, calls: int = 100, period: int = 60):
        """
        Initialize rate limiter.

        Args:
            app: The FastAPI application
            calls: Number of calls allowed per period
            period: Time period in seconds
        """
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients: Dict[str, Tuple[int, float]] = defaultdict(lambda: (0, 0))

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Use user ID if authenticated
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"

        # Otherwise use IP address
        if request.client:
            return f"ip:{request.client.host}"

        return "unknown"

    def _is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited."""
        current_time = time.time()
        calls, window_start = self.clients[client_id]

        # Reset window if period has passed
        if current_time - window_start > self.period:
            self.clients[client_id] = (1, current_time)
            return False

        # Check if limit exceeded
        if calls >= self.calls:
            return True

        # Increment call count
        self.clients[client_id] = (calls + 1, window_start)
        return False

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process the request with rate limiting."""
        # Skip rate limiting for certain paths
        skip_paths = [
            "/api/health",
            "/api/metrics",
            "/api/docs",
            "/api/redoc",
            "/api/openapi.json",
        ]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return await call_next(request)

        # Get client ID
        client_id = self._get_client_id(request)

        # Check rate limit
        if self._is_rate_limited(client_id):
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "code": status.HTTP_429_TOO_MANY_REQUESTS,
                        "message": f"Rate limit exceeded. Maximum {self.calls} requests per {self.period} seconds.",
                        "type": "rate_limit_error",
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(self.calls),
                    "X-RateLimit-Period": str(self.period),
                    "Retry-After": str(self.period),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        calls, _ = self.clients[client_id]
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.calls - calls))
        response.headers["X-RateLimit-Period"] = str(self.period)

        return response
