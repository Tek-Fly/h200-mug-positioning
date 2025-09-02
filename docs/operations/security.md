# Security Guide

Comprehensive security guide for the H200 Intelligent Mug Positioning System covering deployment, configuration, and operational security best practices.

## Security Overview

The H200 System implements enterprise-grade security with multiple layers of protection:

- **Authentication**: JWT-based with configurable expiration
- **Authorization**: Role-based access control (RBAC)
- **Encryption**: TLS 1.3 for transit, AES-256 for storage
- **Secret Management**: Google Secret Manager integration
- **Network Security**: VPC isolation and firewall rules
- **Audit Logging**: Comprehensive security event logging
- **Vulnerability Management**: Regular security scanning and updates

## Authentication and Authorization

### 1. JWT Authentication

#### Token Configuration

```python
# Secure JWT configuration
JWT_CONFIG = {
    'algorithm': 'RS256',  # Use RSA for production
    'expiry_hours': 8,     # Shorter expiry for security
    'refresh_token_hours': 168,  # 7 days
    'issuer': 'h200-system',
    'audience': 'h200-api',
    'require_exp': True,
    'require_iat': True,
    'require_nbf': True,
    'leeway_seconds': 30,  # Clock skew tolerance
}

# Generate secure key pair
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048
)

private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

public_key = private_key.public_key()
public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)
```

#### Token Validation

```python
import jwt
from datetime import datetime, timezone
from typing import Dict, Optional

class SecureJWTHandler:
    def __init__(self, private_key: str, public_key: str):
        self.private_key = private_key
        self.public_key = public_key
        self.algorithm = 'RS256'
    
    def generate_token(self, user_data: Dict, expires_in: int = 28800) -> str:
        """Generate secure JWT token."""
        now = datetime.now(timezone.utc)
        
        payload = {
            # Standard claims
            'iss': 'h200-system',
            'aud': 'h200-api', 
            'sub': str(user_data['user_id']),
            'iat': now.timestamp(),
            'exp': (now + timedelta(seconds=expires_in)).timestamp(),
            'nbf': now.timestamp(),
            'jti': str(uuid4()),  # Unique token ID
            
            # Custom claims
            'user_id': str(user_data['user_id']),
            'username': user_data['username'],
            'roles': user_data['roles'],
            'permissions': user_data['permissions'],
            'session_id': str(uuid4()),
            
            # Security context
            'ip_address': user_data.get('ip_address'),
            'user_agent': user_data.get('user_agent'),
            'login_method': user_data.get('login_method', 'password')
        }
        
        return jwt.encode(payload, self.private_key, algorithm=self.algorithm)
    
    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate JWT token with comprehensive checks."""
        try:
            # Decode and verify token
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[self.algorithm],
                issuer='h200-system',
                audience='h200-api',
                options={
                    'require_exp': True,
                    'require_iat': True,
                    'require_nbf': True,
                    'verify_signature': True,
                    'verify_exp': True,
                    'verify_nbf': True,
                    'verify_iat': True,
                    'verify_aud': True,
                    'verify_iss': True
                }
            )
            
            # Additional security checks
            if not self.is_token_valid(payload):
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def is_token_valid(self, payload: Dict) -> bool:
        """Additional token validation checks."""
        
        # Check token age (not just expiration)
        max_age_hours = 24
        issued_at = datetime.fromtimestamp(payload['iat'], timezone.utc)
        if datetime.now(timezone.utc) - issued_at > timedelta(hours=max_age_hours):
            return False
        
        # Check for required claims
        required_claims = ['user_id', 'username', 'roles', 'permissions']
        if not all(claim in payload for claim in required_claims):
            return False
        
        # Validate user still exists and is active
        return self.validate_user_status(payload['user_id'])
```

### 2. Role-Based Access Control

#### Permission System

```python
from enum import Enum
from typing import Set, Dict, List

class Permission(Enum):
    # Analysis permissions
    ANALYSIS_READ = "analysis:read"
    ANALYSIS_WRITE = "analysis:write"
    ANALYSIS_DELETE = "analysis:delete"
    
    # Rules permissions
    RULES_READ = "rules:read" 
    RULES_WRITE = "rules:write"
    RULES_DELETE = "rules:delete"
    RULES_EXECUTE = "rules:execute"
    
    # Server management
    SERVERS_READ = "servers:read"
    SERVERS_CONTROL = "servers:control"
    SERVERS_DEPLOY = "servers:deploy"
    SERVERS_DELETE = "servers:delete"
    
    # System administration
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_AUDIT = "admin:audit"

class Role:
    def __init__(self, name: str, permissions: Set[Permission]):
        self.name = name
        self.permissions = permissions

# Define roles
ROLES = {
    'viewer': Role('viewer', {
        Permission.ANALYSIS_READ,
        Permission.RULES_READ,
        Permission.SERVERS_READ
    }),
    
    'analyst': Role('analyst', {
        Permission.ANALYSIS_READ,
        Permission.ANALYSIS_WRITE,
        Permission.RULES_READ,
        Permission.RULES_EXECUTE,
        Permission.SERVERS_READ
    }),
    
    'manager': Role('manager', {
        Permission.ANALYSIS_READ,
        Permission.ANALYSIS_WRITE,
        Permission.ANALYSIS_DELETE,
        Permission.RULES_READ,
        Permission.RULES_WRITE,
        Permission.RULES_DELETE,
        Permission.RULES_EXECUTE,
        Permission.SERVERS_READ,
        Permission.SERVERS_CONTROL
    }),
    
    'admin': Role('admin', set(Permission))  # All permissions
}

class RBACManager:
    def __init__(self, roles: Dict[str, Role]):
        self.roles = roles
    
    def check_permission(
        self,
        user_roles: List[str],
        required_permission: Permission
    ) -> bool:
        """Check if user has required permission."""
        
        user_permissions = set()
        for role_name in user_roles:
            if role_name in self.roles:
                user_permissions.update(self.roles[role_name].permissions)
        
        return required_permission in user_permissions
    
    def get_user_permissions(self, user_roles: List[str]) -> Set[Permission]:
        """Get all permissions for user roles."""
        
        permissions = set()
        for role_name in user_roles:
            if role_name in self.roles:
                permissions.update(self.roles[role_name].permissions)
        
        return permissions

# Usage in FastAPI
from fastapi import Depends, HTTPException, status

def require_permission(permission: Permission):
    """Decorator to require specific permission."""
    
    def permission_checker(current_user: dict = Depends(get_current_user)):
        rbac = RBACManager(ROLES)
        
        if not rbac.check_permission(current_user['roles'], permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions: {permission.value} required"
            )
        
        return current_user
    
    return permission_checker

# Apply to endpoints
@app.post("/api/v1/rules")
async def create_rule(
    rule_data: RuleCreateRequest,
    user: dict = Depends(require_permission(Permission.RULES_WRITE))
):
    # Implementation
    pass
```

## Encryption and Data Protection

### 1. Data Encryption

#### At-Rest Encryption

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class DataEncryption:
    def __init__(self, password: bytes, salt: bytes = None):
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # OWASP recommended minimum
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.cipher_suite = Fernet(key)
        self.salt = salt
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_dict(self, data: dict) -> str:
        """Encrypt dictionary data."""
        json_data = json.dumps(data, sort_keys=True)
        return self.encrypt(json_data)
    
    def decrypt_dict(self, encrypted_data: str) -> dict:
        """Decrypt dictionary data."""
        json_data = self.decrypt(encrypted_data)
        return json.loads(json_data)

# Usage for sensitive data
class SecureUserData:
    def __init__(self, encryption_key: bytes):
        self.encryption = DataEncryption(encryption_key)
    
    async def store_user_data(self, user_id: str, sensitive_data: dict):
        """Store encrypted user data."""
        encrypted_data = self.encryption.encrypt_dict(sensitive_data)
        
        await self.db.user_data.update_one(
            {"user_id": user_id},
            {"$set": {
                "encrypted_data": encrypted_data,
                "encryption_version": "v1",
                "updated_at": datetime.utcnow()
            }},
            upsert=True
        )
    
    async def retrieve_user_data(self, user_id: str) -> dict:
        """Retrieve and decrypt user data."""
        doc = await self.db.user_data.find_one({"user_id": user_id})
        
        if not doc or "encrypted_data" not in doc:
            return {}
        
        return self.encryption.decrypt_dict(doc["encrypted_data"])
```

#### In-Transit Encryption

```nginx
# nginx.conf - TLS Configuration
server {
    listen 443 ssl http2;
    server_name h200.tekfly.co.uk;
    
    # TLS Configuration
    ssl_certificate /etc/ssl/certs/h200.crt;
    ssl_certificate_key /etc/ssl/private/h200.key;
    
    # Modern TLS configuration
    ssl_protocols TLSv1.3 TLSv1.2;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;
    
    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    
    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:;" always;
    
    location /api/ {
        proxy_pass http://h200-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Security headers for API
        proxy_hide_header X-Powered-By;
        add_header X-API-Version "1.0.0" always;
    }
}
```

### 2. Secret Management

#### Google Secret Manager Integration

```python
from google.cloud import secretmanager
from typing import Dict, Optional
import logging

class SecretManager:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()
        self.cache = {}  # Local cache for performance
        self.cache_ttl = 300  # 5 minutes
        
    async def get_secret(self, secret_name: str, version: str = "latest") -> str:
        """Retrieve secret from Google Secret Manager."""
        
        # Check cache first
        cache_key = f"{secret_name}:{version}"
        cached = self.cache.get(cache_key)
        
        if cached and cached['expires'] > datetime.utcnow():
            return cached['value']
        
        try:
            # Build the resource name
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/{version}"
            
            # Access the secret version
            response = self.client.access_secret_version(request={"name": name})
            secret_value = response.payload.data.decode("UTF-8")
            
            # Cache the secret
            self.cache[cache_key] = {
                'value': secret_value,
                'expires': datetime.utcnow() + timedelta(seconds=self.cache_ttl)
            }
            
            return secret_value
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            raise SecretManagerError(f"Cannot retrieve secret: {secret_name}")
    
    async def create_secret(self, secret_name: str, secret_value: str) -> str:
        """Create new secret in Secret Manager."""
        
        try:
            parent = f"projects/{self.project_id}"
            
            # Create the secret
            secret = self.client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": secret_name,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
            
            # Add the secret version
            response = self.client.add_secret_version(
                request={
                    "parent": secret.name,
                    "payload": {"data": secret_value.encode("UTF-8")},
                }
            )
            
            return response.name
            
        except Exception as e:
            logger.error(f"Failed to create secret {secret_name}: {e}")
            raise
    
    async def rotate_secret(self, secret_name: str, new_value: str):
        """Rotate secret with new value."""
        
        try:
            parent = f"projects/{self.project_id}/secrets/{secret_name}"
            
            # Add new version
            response = self.client.add_secret_version(
                request={
                    "parent": parent,
                    "payload": {"data": new_value.encode("UTF-8")},
                }
            )
            
            # Clear cache
            for key in list(self.cache.keys()):
                if key.startswith(f"{secret_name}:"):
                    del self.cache[key]
            
            logger.info(f"Secret {secret_name} rotated successfully")
            return response.name
            
        except Exception as e:
            logger.error(f"Failed to rotate secret {secret_name}: {e}")
            raise

# Environment-specific secret management
class EnvironmentSecrets:
    def __init__(self, environment: str):
        self.environment = environment
        self.secret_manager = SecretManager(os.getenv('GOOGLE_CLOUD_PROJECT'))
        
    async def get_database_url(self) -> str:
        """Get environment-specific database URL."""
        return await self.secret_manager.get_secret(f"mongodb-{self.environment}-uri")
    
    async def get_jwt_keys(self) -> tuple[str, str]:
        """Get JWT private and public keys."""
        private_key = await self.secret_manager.get_secret(f"jwt-private-key-{self.environment}")
        public_key = await self.secret_manager.get_secret(f"jwt-public-key-{self.environment}")
        return private_key, public_key
    
    async def get_encryption_key(self) -> str:
        """Get data encryption key."""
        return await self.secret_manager.get_secret(f"data-encryption-key-{self.environment}")
```

## Network Security

### 1. VPC and Firewall Configuration

#### RunPod Network Security

```python
# RunPod network security configuration
NETWORK_SECURITY_CONFIG = {
    'firewall_rules': {
        'inbound': [
            {
                'protocol': 'tcp',
                'port': 8000,
                'source': 'load_balancer',
                'description': 'API traffic from load balancer'
            },
            {
                'protocol': 'tcp', 
                'port': 22,
                'source': 'admin_ips',
                'description': 'SSH access for administrators'
            }
        ],
        'outbound': [
            {
                'protocol': 'tcp',
                'port': 443,
                'destination': 'any',
                'description': 'HTTPS outbound for API calls'
            },
            {
                'protocol': 'tcp',
                'port': 27017,
                'destination': 'mongodb_atlas',
                'description': 'MongoDB Atlas connection'
            },
            {
                'protocol': 'tcp',
                'port': 6379,
                'destination': 'redis_cluster',
                'description': 'Redis cluster connection'
            }
        ]
    },
    
    'network_isolation': {
        'enable_vpc': True,
        'subnet_cidr': '10.0.0.0/16',
        'enable_nat_gateway': True,
        'enable_flow_logs': True
    },
    
    'ddos_protection': {
        'enable_rate_limiting': True,
        'requests_per_second': 100,
        'burst_size': 200,
        'block_duration_minutes': 10
    }
}
```

### 2. API Security

#### Input Validation and Sanitization

```python
from pydantic import BaseModel, validator, Field
from typing import Any, Dict, List
import re
import html

class SecureRequest(BaseModel):
    """Base class for secure request validation."""
    
    class Config:
        # Validate assignment to prevent injection
        validate_assignment = True
        # Allow population by field name for API compatibility
        allow_population_by_field_name = True
        # Strict validation
        extra = 'forbid'
    
    @validator('*', pre=True)
    def sanitize_strings(cls, v):
        """Sanitize string inputs."""
        if isinstance(v, str):
            # HTML escape
            v = html.escape(v)
            # Remove null bytes
            v = v.replace('\x00', '')
            # Limit length
            if len(v) > 10000:  # Configurable limit
                raise ValueError('Input too long')
        return v

class SecureAnalysisRequest(SecureRequest):
    """Secure analysis request with comprehensive validation."""
    
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold"
    )
    
    rules_context: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
        regex=r'^[a-zA-Z0-9_\-\s]+$',  # Alphanumeric, underscore, hyphen, space only
        description="Rules context identifier"
    )
    
    include_feedback: bool = Field(
        default=True,
        description="Include positioning feedback"
    )
    
    @validator('rules_context')
    def validate_rules_context(cls, v):
        """Additional validation for rules context."""
        if v is not None:
            # Check for SQL injection patterns
            sql_patterns = [
                r'(union|select|insert|update|delete|drop|create|alter)',
                r'(--|\#|/\*|\*/)',
                r'(\;|\||&)'
            ]
            
            for pattern in sql_patterns:
                if re.search(pattern, v.lower()):
                    raise ValueError('Invalid characters in rules context')
        
        return v

# File upload security
import magic
from PIL import Image

class SecureFileUpload:
    def __init__(self):
        self.allowed_mime_types = {
            'image/jpeg',
            'image/png', 
            'image/webp'
        }
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.max_dimensions = (4096, 4096)  # Max width/height
        
    async def validate_uploaded_file(self, file: UploadFile) -> bool:
        """Validate uploaded file for security."""
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset
        
        if size > self.max_file_size:
            raise ValidationError(f"File too large: {size} bytes (max: {self.max_file_size})")
        
        # Read file content for validation
        content = await file.read()
        await file.seek(0)  # Reset for later use
        
        # Validate MIME type using python-magic
        detected_type = magic.from_buffer(content, mime=True)
        if detected_type not in self.allowed_mime_types:
            raise ValidationError(f"Invalid file type: {detected_type}")
        
        # Additional image validation
        try:
            image = Image.open(io.BytesIO(content))
            
            # Check dimensions
            if image.width > self.max_dimensions[0] or image.height > self.max_dimensions[1]:
                raise ValidationError(f"Image too large: {image.width}x{image.height}")
            
            # Verify image integrity
            image.verify()
            
        except Exception as e:
            raise ValidationError(f"Invalid image file: {e}")
        
        return True
```

## Security Monitoring and Auditing

### 1. Security Event Logging

```python
import structlog
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional

class SecurityEventType(Enum):
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    TOKEN_EXPIRED = "token_expired"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    ADMIN_ACTION = "admin_action"
    CONFIGURATION_CHANGE = "configuration_change"

class SecurityAuditLogger:
    def __init__(self):
        self.logger = structlog.get_logger("security_audit")
    
    async def log_security_event(
        self,
        event_type: SecurityEventType,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "INFO"
    ):
        """Log security event with comprehensive context."""
        
        event_data = {
            "event_type": event_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,
            "user_id": user_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "session_id": self.get_current_session_id(),
            "request_id": self.get_current_request_id(),
            "details": details or {}
        }
        
        # Add geolocation if available
        if ip_address:
            event_data["geolocation"] = await self.get_ip_geolocation(ip_address)
        
        # Log event
        if severity == "CRITICAL":
            self.logger.critical("Security event", **event_data)
        elif severity == "WARNING":
            self.logger.warning("Security event", **event_data)
        else:
            self.logger.info("Security event", **event_data)
        
        # Store in security audit collection
        await self.store_audit_event(event_data)
        
        # Trigger alerts for critical events
        if severity == "CRITICAL":
            await self.trigger_security_alert(event_data)
    
    async def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        ip_address: str
    ):
        """Log data access events for compliance."""
        
        await self.log_security_event(
            event_type=SecurityEventType.DATA_ACCESS,
            user_id=user_id,
            ip_address=ip_address,
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                "action": action,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# Usage in API endpoints
security_logger = SecurityAuditLogger()

@app.middleware("http")
async def security_audit_middleware(request: Request, call_next):
    """Middleware to log security-relevant events."""
    
    start_time = time.time()
    
    # Extract client info
    ip_address = request.headers.get("X-Forwarded-For", request.client.host)
    user_agent = request.headers.get("User-Agent", "")
    
    # Process request
    response = await call_next(request)
    
    # Log security events based on response
    if response.status_code == 401:
        await security_logger.log_security_event(
            event_type=SecurityEventType.LOGIN_FAILURE,
            ip_address=ip_address,
            user_agent=user_agent,
            details={"endpoint": str(request.url.path)},
            severity="WARNING"
        )
    elif response.status_code == 403:
        user_id = getattr(request.state, 'user_id', None)
        await security_logger.log_security_event(
            event_type=SecurityEventType.PERMISSION_DENIED,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details={"endpoint": str(request.url.path)},
            severity="WARNING"
        )
    
    return response
```

### 2. Intrusion Detection

```python
from collections import defaultdict, deque
from datetime import datetime, timedelta
import asyncio

class IntrusionDetectionSystem:
    def __init__(self):
        self.failed_attempts = defaultdict(deque)  # IP -> failed attempts
        self.request_rates = defaultdict(deque)    # IP -> request timestamps
        self.suspicious_patterns = defaultdict(int) # IP -> pattern count
        
        # Thresholds
        self.max_failed_attempts = 5
        self.max_requests_per_minute = 100
        self.time_window = timedelta(minutes=5)
        
    async def analyze_request(
        self,
        ip_address: str,
        endpoint: str,
        user_agent: str,
        status_code: int
    ) -> Dict[str, Any]:
        """Analyze request for suspicious patterns."""
        
        now = datetime.utcnow()
        analysis_result = {
            "suspicious": False,
            "risk_level": "low",
            "reasons": [],
            "recommended_action": "allow"
        }
        
        # Track failed authentication attempts
        if status_code in [401, 403]:
            self.failed_attempts[ip_address].append(now)
            
            # Remove old attempts
            cutoff = now - self.time_window
            while (self.failed_attempts[ip_address] and 
                   self.failed_attempts[ip_address][0] < cutoff):
                self.failed_attempts[ip_address].popleft()
            
            # Check if threshold exceeded
            if len(self.failed_attempts[ip_address]) >= self.max_failed_attempts:
                analysis_result.update({
                    "suspicious": True,
                    "risk_level": "high",
                    "reasons": ["Multiple failed authentication attempts"],
                    "recommended_action": "block"
                })
        
        # Track request rates
        self.request_rates[ip_address].append(now)
        
        # Remove old requests
        cutoff = now - timedelta(minutes=1)
        while (self.request_rates[ip_address] and 
               self.request_rates[ip_address][0] < cutoff):
            self.request_rates[ip_address].popleft()
        
        # Check request rate
        requests_per_minute = len(self.request_rates[ip_address])
        if requests_per_minute > self.max_requests_per_minute:
            analysis_result.update({
                "suspicious": True,
                "risk_level": "medium",
                "reasons": analysis_result["reasons"] + [f"High request rate: {requests_per_minute}/min"],
                "recommended_action": "rate_limit"
            })
        
        # Check for suspicious patterns
        suspicious_patterns = self.detect_suspicious_patterns(endpoint, user_agent)
        if suspicious_patterns:
            analysis_result.update({
                "suspicious": True,
                "risk_level": "medium",
                "reasons": analysis_result["reasons"] + suspicious_patterns,
                "recommended_action": "monitor"
            })
        
        return analysis_result
    
    def detect_suspicious_patterns(self, endpoint: str, user_agent: str) -> List[str]:
        """Detect suspicious request patterns."""
        
        suspicious = []
        
        # Check for common attack patterns in endpoints
        attack_patterns = [
            r'\.\./',           # Directory traversal
            r'<script',         # XSS attempts
            r'union.*select',   # SQL injection
            r'eval\s*\(',      # Code injection
            r'exec\s*\(',      # Command injection
        ]
        
        for pattern in attack_patterns:
            if re.search(pattern, endpoint, re.IGNORECASE):
                suspicious.append(f"Suspicious endpoint pattern: {pattern}")
        
        # Check user agent
        if not user_agent or len(user_agent) < 10:
            suspicious.append("Missing or suspicious user agent")
        
        # Check for automated tools
        bot_patterns = [
            r'curl', r'wget', r'python', r'bot', r'crawler',
            r'scanner', r'nikto', r'sqlmap', r'nmap'
        ]
        
        for pattern in bot_patterns:
            if re.search(pattern, user_agent, re.IGNORECASE):
                suspicious.append(f"Automated tool detected: {pattern}")
        
        return suspicious
```

## Vulnerability Management

### 1. Security Scanning

```bash
#!/bin/bash
# Security scanning script

echo "=== H200 Security Scanner ==="

# 1. Docker image vulnerability scanning
echo "Scanning Docker images..."
docker scout cves tekfly/h200:latest --format json > docker-scan-results.json

# 2. Dependency vulnerability scanning  
echo "Scanning Python dependencies..."
pip-audit --requirement requirements.txt --format json --output pip-audit-results.json

# 3. Static code analysis
echo "Running static code analysis..."
bandit -r src/ -f json -o bandit-results.json

# 4. Configuration security check
echo "Checking configuration security..."
python scripts/security/config_security_check.py

# 5. SSL/TLS configuration test
echo "Testing SSL/TLS configuration..."
testssl.sh --jsonfile ssl-test-results.json https://h200.tekfly.co.uk

# 6. Generate security report
echo "Generating security report..."
python scripts/security/generate_security_report.py

echo "Security scan completed. Check security-report.html for results."
```

### 2. Automated Security Updates

```python
import subprocess
import json
from datetime import datetime
from typing import List, Dict

class SecurityUpdateManager:
    def __init__(self):
        self.vulnerability_db = {}
        self.update_log = []
        
    async def scan_vulnerabilities(self) -> Dict:
        """Scan for security vulnerabilities."""
        
        scan_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "docker_vulnerabilities": await self.scan_docker_images(),
            "python_vulnerabilities": await self.scan_python_dependencies(),
            "code_vulnerabilities": await self.scan_source_code(),
        }
        
        return scan_results
    
    async def scan_docker_images(self) -> List[Dict]:
        """Scan Docker images for vulnerabilities."""
        
        try:
            result = subprocess.run([
                "docker", "scout", "cves", "tekfly/h200:latest", 
                "--format", "json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                logger.error(f"Docker scan failed: {result.stderr}")
                return []
                
        except Exception as e:
            logger.error(f"Docker scan error: {e}")
            return []
    
    async def apply_security_updates(self, scan_results: Dict) -> Dict:
        """Apply available security updates."""
        
        update_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "applied_updates": [],
            "failed_updates": [],
            "reboot_required": False
        }
        
        # Update Python dependencies
        python_updates = await self.update_python_dependencies(
            scan_results["python_vulnerabilities"]
        )
        update_results["applied_updates"].extend(python_updates)
        
        # Update base Docker images
        docker_updates = await self.update_docker_images(
            scan_results["docker_vulnerabilities"]  
        )
        update_results["applied_updates"].extend(docker_updates)
        
        # Log updates
        self.update_log.append(update_results)
        
        return update_results
    
    async def create_security_patch(self, vulnerability: Dict) -> str:
        """Create security patch for identified vulnerability."""
        
        patch_info = {
            "vulnerability_id": vulnerability["id"],
            "severity": vulnerability["severity"],
            "component": vulnerability["component"],
            "patch_created": datetime.utcnow().isoformat(),
            "patch_type": "automatic"
        }
        
        # Generate patch based on vulnerability type
        if vulnerability["type"] == "dependency":
            patch_content = await self.create_dependency_patch(vulnerability)
        elif vulnerability["type"] == "configuration":
            patch_content = await self.create_config_patch(vulnerability)
        else:
            patch_content = await self.create_code_patch(vulnerability)
        
        # Store patch for review
        patch_id = f"patch_{vulnerability['id']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        await self.store_security_patch(patch_id, patch_info, patch_content)
        
        return patch_id
```

## Compliance and Governance

### 1. Data Privacy Compliance

```python
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class PersonalDataType(Enum):
    EMAIL = "email"
    IP_ADDRESS = "ip_address"
    USER_ID = "user_id"
    BIOMETRIC = "biometric"
    LOCATION = "location"

class DataPrivacyManager:
    def __init__(self):
        self.retention_policies = {
            DataClassification.PUBLIC: timedelta(days=365),
            DataClassification.INTERNAL: timedelta(days=180),
            DataClassification.CONFIDENTIAL: timedelta(days=90),
            DataClassification.RESTRICTED: timedelta(days=30)
        }
        
    async def classify_data(self, data: Dict) -> DataClassification:
        """Classify data based on content and context."""
        
        # Check for personal data
        personal_data_found = []
        
        for key, value in data.items():
            if self.contains_personal_data(key, value):
                personal_data_found.append(key)
        
        # Classify based on sensitivity
        if any(self.is_sensitive_personal_data(pd) for pd in personal_data_found):
            return DataClassification.RESTRICTED
        elif personal_data_found:
            return DataClassification.CONFIDENTIAL
        elif self.contains_business_data(data):
            return DataClassification.INTERNAL
        else:
            return DataClassification.PUBLIC
    
    def contains_personal_data(self, key: str, value: Any) -> bool:
        """Check if field contains personal data."""
        
        personal_data_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b'
        }
        
        key_lower = key.lower()
        
        # Check key name
        if any(pd in key_lower for pd in ['email', 'phone', 'ssn', 'address', 'name']):
            return True
        
        # Check value patterns
        if isinstance(value, str):
            for pattern in personal_data_patterns.values():
                if re.search(pattern, value):
                    return True
        
        return False
    
    async def apply_data_retention(self, collection_name: str):
        """Apply data retention policies."""
        
        # Get retention policy for collection
        classification = await self.get_collection_classification(collection_name)
        retention_period = self.retention_policies[classification]
        cutoff_date = datetime.utcnow() - retention_period
        
        # Find expired records
        expired_records = await self.db[collection_name].find({
            "created_at": {"$lt": cutoff_date}
        }).to_list(length=None)
        
        # Handle personal data specially
        for record in expired_records:
            if await self.contains_personal_data_record(record):
                await self.anonymize_personal_data(collection_name, record["_id"])
            else:
                await self.delete_record(collection_name, record["_id"])
        
        logger.info(f"Applied retention policy to {collection_name}: {len(expired_records)} records processed")
    
    async def handle_gdpr_request(self, request_type: str, user_id: str) -> Dict:
        """Handle GDPR data subject requests."""
        
        if request_type == "access":
            return await self.export_user_data(user_id)
        elif request_type == "deletion":
            return await self.delete_user_data(user_id)
        elif request_type == "rectification":
            return await self.correct_user_data(user_id)
        elif request_type == "portability":
            return await self.export_user_data_portable(user_id)
        else:
            raise ValueError(f"Unknown GDPR request type: {request_type}")
```

### 2. Security Governance

```python
class SecurityGovernanceFramework:
    def __init__(self):
        self.policies = {}
        self.controls = {}
        self.compliance_status = {}
        
    async def implement_security_controls(self):
        """Implement comprehensive security controls."""
        
        controls = {
            # Access Controls
            "AC-1": {
                "name": "Access Control Policy",
                "implementation": self.implement_access_control_policy,
                "status": "implemented"
            },
            "AC-2": {
                "name": "Account Management", 
                "implementation": self.implement_account_management,
                "status": "implemented"
            },
            
            # Audit and Accountability
            "AU-1": {
                "name": "Audit Policy",
                "implementation": self.implement_audit_policy,
                "status": "implemented"
            },
            "AU-2": {
                "name": "Event Logging",
                "implementation": self.implement_event_logging,
                "status": "implemented"
            },
            
            # Configuration Management
            "CM-1": {
                "name": "Configuration Management Policy",
                "implementation": self.implement_cm_policy,
                "status": "implemented"
            },
            
            # System and Communications Protection
            "SC-1": {
                "name": "System and Communications Protection Policy", 
                "implementation": self.implement_sc_policy,
                "status": "implemented"
            },
            "SC-8": {
                "name": "Transmission Confidentiality",
                "implementation": self.implement_transmission_security,
                "status": "implemented"
            }
        }
        
        self.controls = controls
        
        # Implement each control
        for control_id, control in controls.items():
            try:
                await control["implementation"]()
                logger.info(f"Implemented security control {control_id}: {control['name']}")
            except Exception as e:
                logger.error(f"Failed to implement control {control_id}: {e}")
                control["status"] = "failed"
    
    async def generate_compliance_report(self) -> Dict:
        """Generate comprehensive compliance report."""
        
        report = {
            "report_date": datetime.utcnow().isoformat(),
            "security_controls": self.controls,
            "compliance_frameworks": {
                "SOC2": await self.assess_soc2_compliance(),
                "ISO27001": await self.assess_iso27001_compliance(),
                "GDPR": await self.assess_gdpr_compliance()
            },
            "risk_assessment": await self.perform_risk_assessment(),
            "recommendations": await self.generate_security_recommendations()
        }
        
        return report
```

This comprehensive security guide provides the foundation for implementing and maintaining enterprise-grade security across all aspects of the H200 System, from development through production operations.