"""
Google Secret Manager integration for H200 Intelligent Mug Positioning System.

This module provides secure secret management using Google Secret Manager
with caching, retry logic, and comprehensive error handling.
"""

# Standard library imports
import asyncio
import json
import os
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

# Third-party imports
import structlog
from dotenv import load_dotenv
from google.api_core import exceptions as gcp_exceptions
from google.api_core import retry
from google.cloud import secretmanager
from google.cloud.secretmanager_v1 import SecretVersion
from google.oauth2 import service_account

# Load environment variables
load_dotenv()

# Initialize structured logger
logger = structlog.get_logger(__name__)


class SecretManager:
    """
    Google Secret Manager client for secure secret storage and retrieval.

    Provides async interface for managing secrets with automatic caching,
    retry logic, and comprehensive error handling.

    Attributes:
        project_id: GCP project ID
        credentials_path: Path to service account JSON
        client: Secret Manager client instance
        cache_ttl: Cache time-to-live in seconds
        retry_attempts: Number of retry attempts
        retry_delay: Delay between retries in seconds
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        cache_ttl: int = 300,  # 5 minutes
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Google Secret Manager client.

        Args:
            project_id: GCP project ID (defaults to env variable)
            credentials_path: Path to service account JSON (defaults to env variable)
            cache_ttl: Cache time-to-live in seconds
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        if not self.project_id:
            raise ValueError("GCP project ID not provided or found in environment")

        self.credentials_path = credentials_path or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        self.cache_ttl = cache_ttl
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Initialize client
        self._client: Optional[secretmanager.SecretManagerServiceClient] = None
        self._credentials = None
        self._cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "Secret Manager initialized",
            project_id=self.project_id,
            cache_ttl=cache_ttl,
        )

    def _get_client(self) -> secretmanager.SecretManagerServiceClient:
        """
        Get or create Secret Manager client.

        Returns:
            Secret Manager client instance
        """
        if self._client is None:
            if self.credentials_path and os.path.exists(self.credentials_path):
                self._credentials = (
                    service_account.Credentials.from_service_account_file(
                        self.credentials_path
                    )
                )
                self._client = secretmanager.SecretManagerServiceClient(
                    credentials=self._credentials
                )
            else:
                # Use default credentials (e.g., from environment)
                self._client = secretmanager.SecretManagerServiceClient()

            logger.info("Secret Manager client created")

        return self._client

    def _get_cache_key(self, secret_id: str, version: str = "latest") -> str:
        """Generate cache key for a secret."""
        return f"{self.project_id}:{secret_id}:{version}"

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if not cache_entry:
            return False

        expires_at = cache_entry.get("expires_at")
        if not expires_at:
            return False

        return datetime.now() < expires_at

    async def get_secret(
        self,
        secret_id: str,
        version: str = "latest",
        decode: bool = True,
        use_cache: bool = True,
    ) -> Union[str, bytes]:
        """
        Retrieve a secret from Google Secret Manager.

        Args:
            secret_id: Secret identifier
            version: Secret version (default: "latest")
            decode: Whether to decode bytes to string
            use_cache: Whether to use cache

        Returns:
            Secret value as string or bytes

        Raises:
            Exception: If secret cannot be retrieved after retries
        """
        cache_key = self._get_cache_key(secret_id, version)

        # Check cache
        if use_cache and cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.info(
                    "Secret retrieved from cache", secret_id=secret_id, version=version
                )
                return cache_entry["value"]

        # Retrieve from Secret Manager
        client = self._get_client()
        name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"

        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                # Access the secret version
                response = await asyncio.get_event_loop().run_in_executor(
                    None, client.access_secret_version, {"name": name}
                )

                # Get the secret value
                payload = response.payload.data
                value = payload.decode("UTF-8") if decode else payload

                # Cache the secret
                if use_cache:
                    self._cache[cache_key] = {
                        "value": value,
                        "expires_at": datetime.now()
                        + timedelta(seconds=self.cache_ttl),
                        "version": response.name.split("/")[-1],
                    }

                logger.info(
                    "Secret retrieved successfully",
                    secret_id=secret_id,
                    version=version,
                    attempt=attempt + 1,
                )

                return value

            except gcp_exceptions.NotFound:
                logger.error("Secret not found", secret_id=secret_id, version=version)
                raise

            except gcp_exceptions.PermissionDenied:
                logger.error(
                    "Permission denied accessing secret",
                    secret_id=secret_id,
                    version=version,
                )
                raise

            except Exception as e:
                last_error = e
                logger.warning(
                    "Failed to retrieve secret",
                    secret_id=secret_id,
                    version=version,
                    error=str(e),
                    attempt=attempt + 1,
                )

                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        logger.error(
            "Failed to retrieve secret after all retries",
            secret_id=secret_id,
            attempts=self.retry_attempts,
        )
        raise last_error

    async def create_secret(
        self,
        secret_id: str,
        secret_value: Union[str, bytes],
        labels: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create a new secret in Google Secret Manager.

        Args:
            secret_id: Secret identifier
            secret_value: Secret value
            labels: Optional labels for the secret

        Returns:
            Created secret resource name
        """
        client = self._get_client()
        parent = f"projects/{self.project_id}"

        # Create the secret
        secret = {"replication": {"automatic": {}}}

        if labels:
            secret["labels"] = labels

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                client.create_secret,
                {
                    "parent": parent,
                    "secret_id": secret_id,
                    "secret": secret,
                },
            )

            # Add initial version
            version_parent = response.name
            payload = (
                secret_value.encode("UTF-8")
                if isinstance(secret_value, str)
                else secret_value
            )

            version_response = await asyncio.get_event_loop().run_in_executor(
                None,
                client.add_secret_version,
                {
                    "parent": version_parent,
                    "payload": {"data": payload},
                },
            )

            logger.info(
                "Secret created successfully",
                secret_id=secret_id,
                version=version_response.name.split("/")[-1],
            )

            return response.name

        except gcp_exceptions.AlreadyExists:
            logger.warning("Secret already exists", secret_id=secret_id)
            raise
        except Exception as e:
            logger.error("Failed to create secret", secret_id=secret_id, error=str(e))
            raise

    async def update_secret(
        self,
        secret_id: str,
        secret_value: Union[str, bytes],
    ) -> str:
        """
        Update a secret by adding a new version.

        Args:
            secret_id: Secret identifier
            secret_value: New secret value

        Returns:
            New version resource name
        """
        client = self._get_client()
        parent = f"projects/{self.project_id}/secrets/{secret_id}"

        payload = (
            secret_value.encode("UTF-8")
            if isinstance(secret_value, str)
            else secret_value
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                client.add_secret_version,
                {
                    "parent": parent,
                    "payload": {"data": payload},
                },
            )

            # Invalidate cache for this secret
            cache_pattern = f"{self.project_id}:{secret_id}:"
            keys_to_remove = [
                k for k in self._cache.keys() if k.startswith(cache_pattern)
            ]
            for key in keys_to_remove:
                del self._cache[key]

            logger.info(
                "Secret updated successfully",
                secret_id=secret_id,
                version=response.name.split("/")[-1],
            )

            return response.name

        except Exception as e:
            logger.error("Failed to update secret", secret_id=secret_id, error=str(e))
            raise

    async def delete_secret(self, secret_id: str) -> None:
        """
        Delete a secret from Google Secret Manager.

        Args:
            secret_id: Secret identifier
        """
        client = self._get_client()
        name = f"projects/{self.project_id}/secrets/{secret_id}"

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, client.delete_secret, {"name": name}
            )

            # Remove from cache
            cache_pattern = f"{self.project_id}:{secret_id}:"
            keys_to_remove = [
                k for k in self._cache.keys() if k.startswith(cache_pattern)
            ]
            for key in keys_to_remove:
                del self._cache[key]

            logger.info("Secret deleted successfully", secret_id=secret_id)

        except gcp_exceptions.NotFound:
            logger.warning("Secret not found for deletion", secret_id=secret_id)
        except Exception as e:
            logger.error("Failed to delete secret", secret_id=secret_id, error=str(e))
            raise

    async def list_secrets(
        self,
        filter_string: Optional[str] = None,
        page_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List secrets in the project.

        Args:
            filter_string: Optional filter string
            page_size: Number of results per page

        Returns:
            List of secret metadata
        """
        client = self._get_client()
        parent = f"projects/{self.project_id}"

        secrets = []

        try:
            request = {
                "parent": parent,
                "page_size": page_size,
            }

            if filter_string:
                request["filter"] = filter_string

            # List secrets
            page_result = await asyncio.get_event_loop().run_in_executor(
                None, client.list_secrets, request
            )

            for secret in page_result:
                secrets.append(
                    {
                        "name": secret.name,
                        "secret_id": secret.name.split("/")[-1],
                        "create_time": secret.create_time,
                        "labels": dict(secret.labels) if secret.labels else {},
                    }
                )

            logger.info("Listed secrets successfully", count=len(secrets))

            return secrets

        except Exception as e:
            logger.error("Failed to list secrets", error=str(e))
            raise

    async def get_json_secret(
        self,
        secret_id: str,
        version: str = "latest",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieve and parse a JSON secret.

        Args:
            secret_id: Secret identifier
            version: Secret version
            use_cache: Whether to use cache

        Returns:
            Parsed JSON object
        """
        secret_value = await self.get_secret(
            secret_id, version=version, decode=True, use_cache=use_cache
        )

        try:
            return json.loads(secret_value)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse JSON secret", secret_id=secret_id, error=str(e)
            )
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Secret Manager connection.

        Returns:
            Dict containing health status
        """
        health_status = {
            "connected": False,
            "project_id": self.project_id,
            "error": None,
            "secret_count": 0,
            "cache_size": len(self._cache),
        }

        try:
            # Try to list secrets
            secrets = await self.list_secrets(page_size=1)
            health_status["connected"] = True

            # Get total count (approximate)
            client = self._get_client()
            parent = f"projects/{self.project_id}"

            count = 0
            page_result = client.list_secrets(request={"parent": parent})
            for _ in page_result:
                count += 1
                if count > 100:  # Limit counting for performance
                    break

            health_status["secret_count"] = count

            logger.info("Secret Manager health check passed")

        except Exception as e:
            health_status["error"] = str(e)
            logger.error("Secret Manager health check failed", error=str(e))

        return health_status

    def clear_cache(self) -> None:
        """Clear the secret cache."""
        self._cache.clear()
        logger.info("Secret cache cleared")


# Global secret manager instance
_secret_manager: Optional[SecretManager] = None


async def get_secret_manager() -> SecretManager:
    """
    Get or create the global Secret Manager instance.

    Returns:
        SecretManager instance
    """
    global _secret_manager

    if _secret_manager is None:
        _secret_manager = SecretManager()

    return _secret_manager


# Convenience functions for common operations


async def get_secret(secret_id: str, version: str = "latest") -> str:
    """
    Get a secret value.

    Args:
        secret_id: Secret identifier
        version: Secret version

    Returns:
        Secret value as string
    """
    manager = await get_secret_manager()
    return await manager.get_secret(secret_id, version)


async def get_json_secret(secret_id: str, version: str = "latest") -> Dict[str, Any]:
    """
    Get and parse a JSON secret.

    Args:
        secret_id: Secret identifier
        version: Secret version

    Returns:
        Parsed JSON object
    """
    manager = await get_secret_manager()
    return await manager.get_json_secret(secret_id, version)
