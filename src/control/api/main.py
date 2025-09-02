"""Main FastAPI application for the H200 Intelligent Mug Positioning System."""

# Standard library imports
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

# Third-party imports
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.exceptions import HTTPException

# First-party imports
from src.control.api.config import get_settings
from src.control.api.middleware.auth import AuthMiddleware
from src.control.api.middleware.logging import LoggingMiddleware
from src.control.api.middleware.rate_limit import RateLimitMiddleware
from src.control.api.routers import analysis, dashboard, rules, servers, websocket
from src.control.manager.integration import (
    get_orchestrator,
    init_orchestrator,
    shutdown_orchestrator,
)
from src.core.models.manager import ModelManager
from src.database.get_db import get_mongodb, get_redis
from src.utils.logging_config import setup_logging

# Setup logging
logger = setup_logging(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting H200 API server...")

    try:
        # Initialize database connections
        app.state.mongodb = await get_mongodb()
        app.state.redis = await get_redis()

        # Initialize model manager
        app.state.model_manager = ModelManager()
        await app.state.model_manager.initialize()

        # Initialize control plane orchestrator
        app.state.orchestrator = await init_orchestrator(
            runpod_api_key=settings.runpod_api_key,
            idle_timeout_seconds=settings.idle_timeout_seconds,
            enable_auto_shutdown=settings.enable_auto_shutdown,
        )

        # Store settings in app state
        app.state.settings = settings

        logger.info("All services initialized successfully")

        yield

    finally:
        # Shutdown
        logger.info("Shutting down H200 API server...")

        # Shutdown orchestrator
        await shutdown_orchestrator()

        # Cleanup model manager
        if hasattr(app.state, "model_manager"):
            await app.state.model_manager.cleanup()

        # Close database connections
        if hasattr(app.state, "mongodb"):
            app.state.mongodb.close()

        if hasattr(app.state, "redis"):
            await app.state.redis.close()

        logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="H200 Intelligent Mug Positioning System",
    description="API for analyzing mug positions and managing positioning rules",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"],
)

# Add trusted host middleware
if settings.allowed_hosts:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts,
    )

# Add custom middleware
app.add_middleware(
    RateLimitMiddleware,
    calls=settings.rate_limit_calls,
    period=settings.rate_limit_period,
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    AuthMiddleware,
    public_paths=["/api/health", "/api/docs", "/api/redoc", "/api/openapi.json"],
)


# Add request tracking middleware for control plane
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track requests for auto-shutdown and metrics."""
    # Standard library imports
    import time

    start_time = time.time()

    # Store start time for later use
    request.state.start_time = start_time

    # Process request
    response = await call_next(request)

    # Calculate processing time
    request.state.end_time = time.time()

    # Track in orchestrator if available
    if hasattr(request.app.state, "orchestrator"):
        orchestrator = request.app.state.orchestrator

        # Track activity for auto-shutdown
        if request.url.path.startswith("/api/v1/analyze"):
            server_id = request.headers.get("X-Server-ID", "default")
            orchestrator.auto_shutdown.record_activity(server_id)

        # Record metrics
        latency_ms = (request.state.end_time - request.state.start_time) * 1000
        server_id = request.headers.get("X-Server-ID", "default")

        orchestrator.record_request(
            server_id=server_id,
            latency_ms=latency_ms,
            success=response.status_code < 400,
            endpoint=str(request.url.path),
        )

    return response


# Add Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, endpoint="/api/metrics")


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "http_error",
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "message": "Validation error",
                "type": "validation_error",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": "Internal server error",
                "type": "internal_error",
            }
        },
    )


# Health check endpoint
@app.get("/api/health", tags=["health"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "version": app.version,
        "services": {
            "mongodb": False,
            "redis": False,
            "models": False,
        },
    }

    try:
        # Check MongoDB
        if hasattr(app.state, "mongodb"):
            await app.state.mongodb.admin.command("ping")
            health_status["services"]["mongodb"] = True
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")

    try:
        # Check Redis
        if hasattr(app.state, "redis"):
            await app.state.redis.ping()
            health_status["services"]["redis"] = True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")

    try:
        # Check models
        if (
            hasattr(app.state, "model_manager")
            and app.state.model_manager.is_initialized
        ):
            health_status["services"]["models"] = True
    except Exception as e:
        logger.error(f"Model health check failed: {e}")

    try:
        # Check orchestrator
        if hasattr(app.state, "orchestrator") and app.state.orchestrator.is_running:
            health_status["services"]["orchestrator"] = True

            # Add orchestrator details
            orchestrator = app.state.orchestrator
            health_status["orchestrator"] = {
                "servers": len(await orchestrator.server_manager.list_servers()),
                "auto_shutdown": orchestrator.enable_auto_shutdown,
                "notifications": orchestrator.notifier.get_statistics()[
                    "connected_clients"
                ],
            }
    except Exception as e:
        logger.error(f"Orchestrator health check failed: {e}")
        health_status["services"]["orchestrator"] = False

    # Determine overall health
    all_healthy = all(health_status["services"].values())
    if not all_healthy:
        health_status["status"] = "degraded"

    return health_status


# Include routers
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
app.include_router(rules.router, prefix="/api/v1", tags=["rules"])
app.include_router(dashboard.router, prefix="/api/v1", tags=["dashboard"])
app.include_router(servers.router, prefix="/api/v1", tags=["servers"])
app.include_router(websocket.router, tags=["websocket"])


# Root endpoint
@app.get("/", include_in_schema=False)
async def root() -> Dict[str, str]:
    """Root endpoint redirect to docs."""
    return {
        "message": "H200 Intelligent Mug Positioning System API",
        "docs": "/api/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    # Third-party imports
    import uvicorn

    uvicorn.run(
        "src.control.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
