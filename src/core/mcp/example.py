#!/usr/bin/env python3
"""
Example MCP Server Usage.

Demonstrates how to start the MCP server and use it with a client.
"""

# Standard library imports
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Third-party imports
import structlog
from PIL import Image

# First-party imports
from src.core.mcp import MCPClient, MCPServer
from src.core.mcp.auth import MCPAuthenticator
from src.core.mcp.models import MCPAuthType
from src.utils.logging import setup_structured_logging

logger = structlog.get_logger(__name__)


async def start_server():
    """Start the MCP server."""
    logger.info("Starting MCP server...")

    server = MCPServer(
        host="0.0.0.0",
        port=8765,
        enable_auth=True,
        enable_websocket=True,
        enable_http=True,
    )

    await server.initialize()

    # Generate some test credentials
    auth = server.authenticator

    # Generate JWT token
    jwt_token = auth.generate_jwt_token(
        client_id="demo-client",
        scopes=["analyze", "suggest", "update"],
        metadata={"demo": True},
    )
    logger.info("Generated JWT token", token=jwt_token[:20] + "...")

    # Generate API key
    api_key = auth.generate_api_key(
        client_id="demo-client",
        name="demo-key",
        scopes=["analyze", "suggest"],
        rate_limit={"requests_per_minute": 100},
    )
    logger.info("Generated API key", key=api_key[:20] + "...")

    logger.info("MCP server ready on ports 8765 (WebSocket) and 8766 (HTTP)")
    logger.info("Press Ctrl+C to stop")

    # Start server
    await server.start()


async def demo_client():
    """Demonstrate client usage."""
    logger.info("Starting MCP client demo...")

    # Wait for server to start
    await asyncio.sleep(2)

    # Create authenticator to generate token
    auth = MCPAuthenticator(jwt_secret=os.getenv("MCP_JWT_SECRET", "demo-secret"))
    token = auth.generate_jwt_token(
        client_id="demo-client", scopes=["analyze", "suggest", "update"]
    )

    # Create client
    client = MCPClient(
        base_url="http://localhost:8766",
        websocket_url="ws://localhost:8765",
        auth_type=MCPAuthType.JWT,
        auth_token=token,
    )

    async with client:
        # Check capabilities
        logger.info("Getting server capabilities...")
        capabilities = await client.get_capabilities()
        logger.info(
            "Server capabilities",
            version=capabilities.version,
            tools=[t.name for t in capabilities.tools],
            features=capabilities.features,
        )

        # Check health
        health = await client.health_check()
        logger.info("Server health", status=health["status"])

        # Apply a rule
        logger.info("Applying positioning rule...")
        rule_result = await client.apply_rule(
            "Keep mugs at least 6 inches away from electronic devices for safety",
            priority=8,
            tags=["safety", "electronics"],
            validate_only=True,
        )
        logger.info(
            "Rule validation result",
            is_valid=rule_result["validation"]["is_valid"],
            issues=rule_result["validation"]["issues"],
        )

        # Get suggestions for a mock scene
        logger.info("Getting positioning suggestions...")
        scene_context = {
            "objects": [
                {
                    "type": "mug",
                    "id": "mug1",
                    "position": {"x": 100, "y": 200},
                    "bbox": [90, 190, 20, 20],
                },
                {
                    "type": "laptop",
                    "id": "laptop1",
                    "position": {"x": 150, "y": 200},
                    "bbox": [120, 180, 60, 40],
                },
            ],
            "environment": {"type": "office", "desk_size": "medium"},
        }

        suggestions = await client.get_suggestions(
            scene_context=scene_context, strategy="safety_first"
        )
        logger.info(
            "Positioning suggestions",
            num_suggestions=len(suggestions.get("suggestions", [])),
            overall_score=suggestions.get("overall_score"),
        )

        # Create a test image and analyze it
        if False:  # Set to True if you want to test image analysis
            logger.info("Creating and analyzing test image...")
            test_image = Image.new("RGB", (640, 480), color="white")

            try:
                analysis = await client.analyze_image(
                    test_image, apply_rules=True, confidence_threshold=0.5
                )
                logger.info(
                    "Image analysis complete",
                    analysis_id=analysis["analysis_id"],
                    num_detections=len(analysis["detections"]),
                )
            except Exception as e:
                logger.warning(
                    "Image analysis failed (expected without GPU)", error=str(e)
                )

        # Get server stats
        stats = await client.get_stats()
        logger.info(
            "Server statistics",
            total_requests=stats["requests_total"],
            success_rate=(
                stats["requests_success"] / stats["requests_total"]
                if stats["requests_total"] > 0
                else 0
            ),
        )


async def main():
    """Main entry point."""
    setup_structured_logging()

    # Check command line argument
    if len(sys.argv) > 1:
        if sys.argv[1] == "server":
            await start_server()
        elif sys.argv[1] == "client":
            await demo_client()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python example.py [server|client]")
    else:
        # Run both in parallel for demo
        logger.info("Starting server and client demo...")

        # Start server in background
        server_task = asyncio.create_task(start_server())

        # Run client demo
        try:
            await demo_client()
        except KeyboardInterrupt:
            logger.info("Demo interrupted")
        finally:
            # Cancel server
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
