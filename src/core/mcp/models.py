"""
MCP Protocol Models and Data Structures.

Defines the data models for MCP requests, responses, and tool definitions.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class MCPVersion(str, Enum):
    """Supported MCP protocol versions."""
    V1_0 = "1.0"
    V1_1 = "1.1"


class MCPToolType(str, Enum):
    """Types of MCP tools."""
    FUNCTION = "function"
    STREAMING = "streaming"
    ASYNC = "async"


class MCPAuthType(str, Enum):
    """Authentication types."""
    JWT = "jwt"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"


class MCPToolParameter(BaseModel):
    """Definition of a tool parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    
    @validator('type')
    def validate_type(cls, v):
        valid_types = {'string', 'number', 'integer', 'boolean', 'array', 'object', 'file'}
        if v not in valid_types:
            raise ValueError(f"Invalid parameter type: {v}")
        return v


class MCPToolDefinition(BaseModel):
    """Definition of an MCP tool."""
    name: str
    type: MCPToolType = MCPToolType.FUNCTION
    description: str
    parameters: List[MCPToolParameter]
    returns: Dict[str, Any]
    examples: Optional[List[Dict[str, Any]]] = None
    rate_limit: Optional[Dict[str, int]] = None  # e.g., {"requests_per_minute": 60}
    requires_auth: bool = True
    async_timeout: Optional[int] = 30  # seconds
    
    class Config:
        json_encoders = {
            MCPToolType: lambda v: v.value
        }


class MCPRequest(BaseModel):
    """MCP protocol request."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: MCPVersion = MCPVersion.V1_0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tool: str
    parameters: Dict[str, Any]
    auth: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('parameters')
    def validate_parameters(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Parameters must be a dictionary")
        return v


class MCPError(BaseModel):
    """MCP error response."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    retry_after: Optional[int] = None  # seconds


class MCPToolResult(BaseModel):
    """Result from a tool execution."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[MCPError] = None
    execution_time_ms: float
    metadata: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """MCP protocol response."""
    id: str
    request_id: str
    version: MCPVersion = MCPVersion.V1_0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    result: MCPToolResult
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            MCPVersion: lambda v: v.value
        }


class MCPCapabilities(BaseModel):
    """Server capabilities response."""
    version: MCPVersion
    tools: List[MCPToolDefinition]
    auth_types: List[MCPAuthType]
    features: List[str]  # e.g., ["streaming", "batch", "async"]
    limits: Dict[str, Any]  # e.g., {"max_request_size": 10485760}
    
    class Config:
        json_encoders = {
            MCPVersion: lambda v: v.value,
            MCPAuthType: lambda v: v.value,
            MCPToolType: lambda v: v.value
        }


class MCPStreamChunk(BaseModel):
    """Chunk of streaming response."""
    request_id: str
    sequence: int
    data: Dict[str, Any]
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None


class MCPBatchRequest(BaseModel):
    """Batch request containing multiple tool calls."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: MCPVersion = MCPVersion.V1_0
    requests: List[MCPRequest]
    parallel: bool = True
    stop_on_error: bool = False
    
    @validator('requests')
    def validate_requests(cls, v):
        if not v:
            raise ValueError("Batch must contain at least one request")
        if len(v) > 100:
            raise ValueError("Batch cannot contain more than 100 requests")
        return v


class MCPBatchResponse(BaseModel):
    """Batch response containing multiple results."""
    id: str
    batch_id: str
    version: MCPVersion = MCPVersion.V1_0
    responses: List[MCPResponse]
    summary: Dict[str, Any]  # e.g., {"total": 10, "successful": 8, "failed": 2}