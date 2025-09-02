"""
MCP (Model Context Protocol) Server for H200 Mug Positioning System.

This module implements an embedded MCP server that exposes the H200
mug positioning capabilities as tools for external systems.
"""

# Local imports
from .auth import MCPAuthenticator
from .models import MCPRequest, MCPResponse, MCPToolDefinition, MCPToolResult
from .server import MCPServer
from .tools import (
    AnalyzeImageTool,
    ApplyRulesTool,
    GetSuggestionsTool,
    UpdatePositioningTool,
)

__all__ = [
    "MCPServer",
    "AnalyzeImageTool",
    "ApplyRulesTool",
    "GetSuggestionsTool",
    "UpdatePositioningTool",
    "MCPAuthenticator",
    "MCPRequest",
    "MCPResponse",
    "MCPToolDefinition",
    "MCPToolResult",
]
