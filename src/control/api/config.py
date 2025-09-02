"""Configuration settings for the FastAPI application."""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application settings
    app_name: str = "H200 Intelligent Mug Positioning System"
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="production", description="Environment name")
    
    # Security settings
    secret_key: SecretStr = Field(..., description="Secret key for JWT tokens")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, description="JWT token expiration in hours")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )
    
    # Host settings
    allowed_hosts: Optional[List[str]] = Field(
        default=None,
        description="Allowed hosts for TrustedHostMiddleware"
    )
    
    # Rate limiting
    rate_limit_calls: int = Field(default=100, description="Number of calls allowed per period")
    rate_limit_period: int = Field(default=60, description="Rate limit period in seconds")
    
    # Database settings
    mongodb_uri: str = Field(..., description="MongoDB connection URI")
    mongodb_database: str = Field(default="h200_positioning", description="MongoDB database name")
    
    redis_url: str = Field(..., description="Redis connection URL")
    redis_password: Optional[SecretStr] = Field(default=None, description="Redis password")
    
    # Model settings
    model_cache_ttl: int = Field(default=3600, description="Model cache TTL in seconds")
    model_warm_up: bool = Field(default=True, description="Warm up models on startup")
    
    # RunPod settings
    runpod_api_key: Optional[SecretStr] = Field(default=None, description="RunPod API key")
    runpod_serverless_id: Optional[str] = Field(default=None, description="RunPod serverless endpoint ID")
    runpod_timed_id: Optional[str] = Field(default=None, description="RunPod timed endpoint ID")
    
    # Control plane settings
    idle_timeout_seconds: int = Field(default=600, description="Idle timeout before auto-shutdown (seconds)")
    enable_auto_shutdown: bool = Field(default=True, description="Enable automatic shutdown of idle servers")
    health_check_interval: int = Field(default=30, description="Health check interval (seconds)")
    resource_monitor_interval: int = Field(default=5, description="Resource monitoring interval (seconds)")
    
    # Google Cloud settings
    google_cloud_project: Optional[str] = Field(default=None, description="Google Cloud project ID")
    google_secret_manager_enabled: bool = Field(default=False, description="Enable Google Secret Manager")
    
    # Cloudflare R2 settings
    r2_access_key_id: Optional[SecretStr] = Field(default=None, description="R2 access key ID")
    r2_secret_access_key: Optional[SecretStr] = Field(default=None, description="R2 secret access key")
    r2_endpoint_url: Optional[str] = Field(default=None, description="R2 endpoint URL")
    r2_bucket_name: Optional[str] = Field(default=None, description="R2 bucket name")
    
    # Monitoring settings
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    log_level: str = Field(default="INFO", description="Logging level")
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v):
        """Parse allowed hosts from comma-separated string."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Field mappings for environment variables
        fields = {
            "secret_key": {"env": "SECRET_KEY"},
            "mongodb_uri": {"env": "MONGODB_ATLAS_URI"},
            "redis_url": {"env": "REDIS_URL"},
            "redis_password": {"env": "REDIS_PASSWORD"},
            "runpod_api_key": {"env": "RUNPOD_API_KEY"},
            "runpod_serverless_id": {"env": "RUNPOD_SERVERLESS_ID"},
            "runpod_timed_id": {"env": "RUNPOD_TIMED_ID"},
            "google_cloud_project": {"env": "GOOGLE_CLOUD_PROJECT"},
            "r2_access_key_id": {"env": "R2_ACCESS_KEY_ID"},
            "r2_secret_access_key": {"env": "R2_SECRET_ACCESS_KEY"},
            "r2_endpoint_url": {"env": "R2_ENDPOINT_URL"},
            "r2_bucket_name": {"env": "R2_BUCKET_NAME"},
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()