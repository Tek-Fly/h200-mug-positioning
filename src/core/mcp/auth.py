"""
MCP Authentication and Authorization.

Implements authentication mechanisms for the MCP server including
JWT tokens, API keys, and rate limiting.
"""

import time
import jwt
import hashlib
import uuid
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime, timedelta
from functools import wraps
import asyncio
from collections import defaultdict

import structlog
from pydantic import BaseModel

from src.utils.secrets import get_secret
from .models import MCPAuthType, MCPError


logger = structlog.get_logger(__name__)


class AuthToken(BaseModel):
    """Authentication token model."""
    token: str
    type: MCPAuthType
    expires_at: Optional[datetime] = None
    scopes: List[str] = []
    metadata: Dict[str, Any] = {}


class RateLimitInfo(BaseModel):
    """Rate limit information."""
    requests: int
    window_seconds: int
    current_count: int
    reset_at: datetime


class MCPAuthenticator:
    """
    Handles authentication and authorization for MCP requests.
    
    Supports:
    - JWT token validation
    - API key validation
    - Rate limiting per client
    - Scope-based authorization
    """
    
    def __init__(
        self,
        jwt_secret: Optional[str] = None,
        jwt_algorithm: str = "HS256",
        token_expiry_hours: int = 24,
        enable_rate_limiting: bool = True,
        default_rate_limit: Dict[str, int] = None
    ):
        """Initialize authenticator."""
        self.jwt_secret = jwt_secret or get_secret("MCP_JWT_SECRET")
        self.jwt_algorithm = jwt_algorithm
        self.token_expiry_hours = token_expiry_hours
        self.enable_rate_limiting = enable_rate_limiting
        self.default_rate_limit = default_rate_limit or {
            "requests_per_minute": 60,
            "requests_per_hour": 1000
        }
        
        # Rate limiting storage
        self._rate_limits: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # API key storage (in production, this would be in a database)
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_rate_limits())
    
    async def _cleanup_rate_limits(self):
        """Periodically clean up old rate limit entries."""
        while True:
            try:
                current_time = time.time()
                
                for client_id in list(self._rate_limits.keys()):
                    for window, timestamps in list(self._rate_limits[client_id].items()):
                        # Remove timestamps older than the window
                        window_seconds = self._get_window_seconds(window)
                        cutoff = current_time - window_seconds
                        
                        self._rate_limits[client_id][window] = [
                            ts for ts in timestamps if ts > cutoff
                        ]
                        
                        # Remove empty entries
                        if not self._rate_limits[client_id][window]:
                            del self._rate_limits[client_id][window]
                    
                    if not self._rate_limits[client_id]:
                        del self._rate_limits[client_id]
                
                # Run cleanup every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error("Rate limit cleanup error", error=str(e))
                await asyncio.sleep(60)
    
    def _get_window_seconds(self, window: str) -> int:
        """Convert window name to seconds."""
        if window == "requests_per_minute":
            return 60
        elif window == "requests_per_hour":
            return 3600
        elif window == "requests_per_day":
            return 86400
        return 60  # Default to minute
    
    def generate_jwt_token(
        self,
        client_id: str,
        scopes: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Generate a JWT token."""
        payload = {
            "client_id": client_id,
            "scopes": scopes or ["analyze", "suggest"],
            "metadata": metadata or {},
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours)
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        logger.info(
            "Generated JWT token",
            client_id=client_id,
            scopes=scopes,
            expires_in_hours=self.token_expiry_hours
        )
        
        return token
    
    def validate_jwt_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate a JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            
            # Check expiration
            if "exp" in payload:
                exp_time = datetime.fromtimestamp(payload["exp"])
                if exp_time < datetime.utcnow():
                    return False, {"error": "Token expired"}
            
            return True, payload
            
        except jwt.ExpiredSignatureError:
            return False, {"error": "Token expired"}
        except jwt.InvalidTokenError as e:
            return False, {"error": f"Invalid token: {str(e)}"}
        except Exception as e:
            logger.error("JWT validation error", error=str(e))
            return False, {"error": "Token validation failed"}
    
    def generate_api_key(
        self,
        client_id: str,
        name: str,
        scopes: List[str] = None,
        rate_limit: Dict[str, int] = None
    ) -> str:
        """Generate an API key."""
        # Generate a secure random key
        key_data = f"{client_id}:{name}:{time.time()}"
        api_key = hashlib.sha256(key_data.encode()).hexdigest()
        
        # Store key metadata
        self._api_keys[api_key] = {
            "client_id": client_id,
            "name": name,
            "scopes": scopes or ["analyze", "suggest"],
            "rate_limit": rate_limit or self.default_rate_limit,
            "created_at": datetime.utcnow(),
            "last_used": None,
            "usage_count": 0
        }
        
        logger.info(
            "Generated API key",
            client_id=client_id,
            name=name,
            scopes=scopes
        )
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate an API key."""
        if api_key not in self._api_keys:
            return False, {"error": "Invalid API key"}
        
        key_data = self._api_keys[api_key]
        
        # Update usage stats
        key_data["last_used"] = datetime.utcnow()
        key_data["usage_count"] += 1
        
        return True, key_data
    
    async def authenticate(
        self,
        auth_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[MCPError]]:
        """
        Authenticate a request.
        
        Returns: (is_authenticated, auth_info, error)
        """
        if not auth_data:
            return False, None, MCPError(
                code="AUTH_REQUIRED",
                message="Authentication required"
            )
        
        auth_type = auth_data.get("type")
        
        if auth_type == MCPAuthType.JWT.value:
            token = auth_data.get("token")
            if not token:
                return False, None, MCPError(
                    code="MISSING_TOKEN",
                    message="JWT token required"
                )
            
            is_valid, payload = self.validate_jwt_token(token)
            if not is_valid:
                return False, None, MCPError(
                    code="INVALID_TOKEN",
                    message=payload.get("error", "Invalid JWT token")
                )
            
            return True, payload, None
        
        elif auth_type == MCPAuthType.API_KEY.value:
            api_key = auth_data.get("key")
            if not api_key:
                return False, None, MCPError(
                    code="MISSING_API_KEY",
                    message="API key required"
                )
            
            is_valid, key_data = self.validate_api_key(api_key)
            if not is_valid:
                return False, None, MCPError(
                    code="INVALID_API_KEY",
                    message=key_data.get("error", "Invalid API key")
                )
            
            return True, key_data, None
        
        else:
            return False, None, MCPError(
                code="UNSUPPORTED_AUTH_TYPE",
                message=f"Unsupported authentication type: {auth_type}"
            )
    
    def check_rate_limit(
        self,
        client_id: str,
        rate_limit: Optional[Dict[str, int]] = None
    ) -> Tuple[bool, Optional[RateLimitInfo]]:
        """Check if client has exceeded rate limit."""
        if not self.enable_rate_limiting:
            return True, None
        
        rate_limit = rate_limit or self.default_rate_limit
        current_time = time.time()
        
        for window, limit in rate_limit.items():
            window_seconds = self._get_window_seconds(window)
            cutoff = current_time - window_seconds
            
            # Get timestamps within window
            timestamps = self._rate_limits[client_id][window]
            valid_timestamps = [ts for ts in timestamps if ts > cutoff]
            
            if len(valid_timestamps) >= limit:
                # Rate limit exceeded
                reset_at = datetime.fromtimestamp(min(valid_timestamps) + window_seconds)
                
                return False, RateLimitInfo(
                    requests=limit,
                    window_seconds=window_seconds,
                    current_count=len(valid_timestamps),
                    reset_at=reset_at
                )
        
        # Add current request
        for window in rate_limit:
            self._rate_limits[client_id][window].append(current_time)
        
        return True, None
    
    def check_scope(
        self,
        required_scope: str,
        auth_info: Dict[str, Any]
    ) -> bool:
        """Check if authenticated client has required scope."""
        scopes = auth_info.get("scopes", [])
        
        # Check for wildcard scope
        if "*" in scopes or "admin" in scopes:
            return True
        
        return required_scope in scopes
    
    def auth_required(
        self,
        scope: Optional[str] = None,
        check_rate_limit: bool = True
    ):
        """
        Decorator for protecting MCP tool methods.
        
        Usage:
            @auth_required(scope="analyze")
            async def analyze_image(self, request: MCPRequest) -> MCPResponse:
                ...
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(self, request: MCPRequest, *args, **kwargs):
                # Authenticate
                is_auth, auth_info, error = await self.authenticate(request.auth)
                if not is_auth:
                    return MCPResponse(
                        id=str(uuid.uuid4()),
                        request_id=request.id,
                        result=MCPToolResult(
                            success=False,
                            error=error,
                            execution_time_ms=0
                        )
                    )
                
                # Check scope
                if scope and not self.check_scope(scope, auth_info):
                    return MCPResponse(
                        id=str(uuid.uuid4()),
                        request_id=request.id,
                        result=MCPToolResult(
                            success=False,
                            error=MCPError(
                                code="INSUFFICIENT_SCOPE",
                                message=f"Required scope: {scope}"
                            ),
                            execution_time_ms=0
                        )
                    )
                
                # Check rate limit
                if check_rate_limit:
                    client_id = auth_info.get("client_id", "unknown")
                    rate_limit = auth_info.get("rate_limit")
                    
                    is_allowed, limit_info = self.check_rate_limit(client_id, rate_limit)
                    if not is_allowed:
                        return MCPResponse(
                            id=str(uuid.uuid4()),
                            request_id=request.id,
                            result=MCPToolResult(
                                success=False,
                                error=MCPError(
                                    code="RATE_LIMIT_EXCEEDED",
                                    message="Rate limit exceeded",
                                    details={
                                        "limit": limit_info.requests,
                                        "window": limit_info.window_seconds,
                                        "reset_at": limit_info.reset_at.isoformat()
                                    },
                                    retry_after=int(
                                        (limit_info.reset_at - datetime.utcnow()).total_seconds()
                                    )
                                ),
                                execution_time_ms=0
                            )
                        )
                
                # Call the actual function
                return await func(self, request, auth_info=auth_info, *args, **kwargs)
            
            return wrapper
        return decorator