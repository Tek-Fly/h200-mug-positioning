"""Authentication middleware using JWT tokens."""

# Standard library imports
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

# Third-party imports
import jwt
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# First-party imports
from src.control.api.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

security = HTTPBearer()


class JWTHandler:
    """JWT token handler."""

    def __init__(
        self, secret_key: str, algorithm: str = "HS256", expiration_hours: int = 24
    ):
        """Initialize JWT handler."""
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expiration_hours = expiration_hours

    def create_token(
        self, user_id: str, permissions: Optional[List[str]] = None
    ) -> str:
        """Create a JWT token."""
        payload = {
            "user_id": user_id,
            "permissions": permissions or [],
            "exp": datetime.now(timezone.utc) + timedelta(hours=self.expiration_hours),
            "iat": datetime.now(timezone.utc),
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> dict:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware."""

    def __init__(self, app, public_paths: Optional[List[str]] = None):
        """Initialize auth middleware."""
        super().__init__(app)
        self.public_paths = public_paths or []
        self.jwt_handler = JWTHandler(
            secret_key=settings.secret_key.get_secret_value(),
            algorithm=settings.jwt_algorithm,
            expiration_hours=settings.jwt_expiration_hours,
        )

    async def dispatch(self, request: Request, call_next):
        """Process the request."""
        # Check if path is public
        if any(request.url.path.startswith(path) for path in self.public_paths):
            return await call_next(request)

        # Extract token from header
        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "code": status.HTTP_401_UNAUTHORIZED,
                        "message": "Missing or invalid authorization header",
                        "type": "authentication_error",
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = authorization.split(" ")[1]

        try:
            # Verify token
            payload = self.jwt_handler.verify_token(token)

            # Add user info to request state
            request.state.user_id = payload.get("user_id")
            request.state.permissions = payload.get("permissions", [])

            # Process request
            response = await call_next(request)
            return response

        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": {
                        "code": e.status_code,
                        "message": e.detail,
                        "type": "authentication_error",
                    }
                },
                headers=e.headers,
            )
        except Exception as e:
            logger.error(f"Authentication error: {e}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                        "message": "Internal authentication error",
                        "type": "authentication_error",
                    }
                },
            )


def get_current_user(request: Request) -> str:
    """Get current user from request state."""
    if not hasattr(request.state, "user_id"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return request.state.user_id


def require_permission(permission: str):
    """Decorator to require specific permission."""

    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            if not hasattr(request.state, "permissions"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                )

            if permission not in request.state.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required",
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
