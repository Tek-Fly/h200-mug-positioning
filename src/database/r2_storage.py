"""
Cloudflare R2 storage client for H200 Intelligent Mug Positioning System.

This module provides an async S3-compatible client for Cloudflare R2 storage
with connection pooling, retry logic, and comprehensive error handling.
"""

# Standard library imports
import asyncio
import io
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, BinaryIO, Dict, List, Optional, Union

# Third-party imports
import aioboto3
import structlog
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
)
from botocore.exceptions import ConnectionError as BotoConnectionError
from botocore.exceptions import (
    EndpointConnectionError,
    ReadTimeoutError,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize structured logger
logger = structlog.get_logger(__name__)


class R2StorageClient:
    """
    Async Cloudflare R2 storage client with S3-compatible interface.

    Provides high-performance object storage operations with automatic
    retry logic, connection pooling, and comprehensive monitoring.

    Attributes:
        endpoint_url: R2 endpoint URL
        access_key_id: R2 access key ID
        secret_access_key: R2 secret access key
        bucket_name: Default bucket name
        region_name: R2 region (auto for Cloudflare)
        max_pool_connections: Maximum connections in pool
        retry_attempts: Number of retry attempts
        retry_delay: Delay between retries in seconds
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        region_name: str = "auto",
        max_pool_connections: int = 50,
        connect_timeout: int = 10,
        read_timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Cloudflare R2 storage client.

        Args:
            endpoint_url: R2 endpoint URL (defaults to env variable)
            access_key_id: R2 access key (defaults to env variable)
            secret_access_key: R2 secret key (defaults to env variable)
            bucket_name: Default bucket name (defaults to env variable)
            region_name: R2 region (auto for Cloudflare)
            max_pool_connections: Maximum connections in pool
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries
        """
        self.endpoint_url = endpoint_url or os.getenv("R2_ENDPOINT_URL")
        if not self.endpoint_url:
            raise ValueError("R2 endpoint URL not provided or found in environment")

        self.access_key_id = access_key_id or os.getenv("R2_ACCESS_KEY_ID")
        if not self.access_key_id:
            raise ValueError("R2 access key ID not provided or found in environment")

        self.secret_access_key = secret_access_key or os.getenv("R2_SECRET_ACCESS_KEY")
        if not self.secret_access_key:
            raise ValueError(
                "R2 secret access key not provided or found in environment"
            )

        self.bucket_name = bucket_name or os.getenv("R2_BUCKET_NAME", "h200-backup")
        self.region_name = region_name
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Configure boto3 client
        self.config = Config(
            max_pool_connections=max_pool_connections,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retries={"max_attempts": retry_attempts, "mode": "adaptive"},
        )

        self._session: Optional[aioboto3.Session] = None

        logger.info(
            "R2 storage client initialized",
            endpoint=self.endpoint_url,
            bucket=self.bucket_name,
            max_connections=max_pool_connections,
        )

    @asynccontextmanager
    async def get_client(self):
        """
        Get an async S3 client for R2 operations.

        Yields:
            Async S3 client instance
        """
        if self._session is None:
            self._session = aioboto3.Session()

        async with self._session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name,
            config=self.config,
        ) as client:
            yield client

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on R2 storage connection.

        Returns:
            Dict containing health status and storage details
        """
        health_status = {
            "connected": False,
            "bucket": self.bucket_name,
            "endpoint": self.endpoint_url,
            "error": None,
            "bucket_exists": False,
            "stats": {},
        }

        try:
            async with self.get_client() as client:
                # Check if bucket exists
                try:
                    await client.head_bucket(Bucket=self.bucket_name)
                    health_status["bucket_exists"] = True
                except ClientError as e:
                    if e.response["Error"]["Code"] == "404":
                        health_status["bucket_exists"] = False
                    else:
                        raise

                # Get bucket location if exists
                if health_status["bucket_exists"]:
                    try:
                        location = await client.get_bucket_location(
                            Bucket=self.bucket_name
                        )
                        health_status["bucket_location"] = location.get(
                            "LocationConstraint", "us-east-1"
                        )
                    except:
                        pass

                # List buckets to verify credentials
                response = await client.list_buckets()
                health_status["connected"] = True
                health_status["stats"]["total_buckets"] = len(
                    response.get("Buckets", [])
                )

                logger.info(
                    "R2 storage health check passed",
                    bucket_exists=health_status["bucket_exists"],
                )

        except Exception as e:
            health_status["error"] = str(e)
            logger.error("R2 storage health check failed", error=str(e))

        return health_status

    async def create_bucket_if_not_exists(
        self, bucket_name: Optional[str] = None
    ) -> bool:
        """
        Create bucket if it doesn't exist.

        Args:
            bucket_name: Bucket name (defaults to configured bucket)

        Returns:
            True if bucket was created or already exists
        """
        bucket = bucket_name or self.bucket_name

        try:
            async with self.get_client() as client:
                # Check if bucket exists
                try:
                    await client.head_bucket(Bucket=bucket)
                    logger.info("Bucket already exists", bucket=bucket)
                    return True
                except ClientError as e:
                    if e.response["Error"]["Code"] != "404":
                        raise

                # Create bucket
                await client.create_bucket(Bucket=bucket)
                logger.info("Bucket created successfully", bucket=bucket)
                return True

        except Exception as e:
            logger.error("Failed to create bucket", bucket=bucket, error=str(e))
            return False

    async def upload_file(
        self,
        file_path: str,
        key: str,
        bucket_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload a file to R2 storage.

        Args:
            file_path: Local file path to upload
            key: Object key in R2
            bucket_name: Bucket name (defaults to configured bucket)
            metadata: Optional metadata to attach
            content_type: Optional content type

        Returns:
            Dict with upload details including ETag
        """
        bucket = bucket_name or self.bucket_name

        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata
            if content_type:
                extra_args["ContentType"] = content_type

            async with self.get_client() as client:
                with open(file_path, "rb") as file:
                    response = await client.put_object(
                        Bucket=bucket, Key=key, Body=file, **extra_args
                    )

                logger.info(
                    "File uploaded successfully",
                    bucket=bucket,
                    key=key,
                    etag=response.get("ETag"),
                )

                return {
                    "bucket": bucket,
                    "key": key,
                    "etag": response.get("ETag"),
                    "version_id": response.get("VersionId"),
                }

        except Exception as e:
            logger.error("Failed to upload file", bucket=bucket, key=key, error=str(e))
            raise

    async def upload_bytes(
        self,
        data: Union[bytes, BinaryIO],
        key: str,
        bucket_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload bytes/stream to R2 storage.

        Args:
            data: Bytes or file-like object to upload
            key: Object key in R2
            bucket_name: Bucket name (defaults to configured bucket)
            metadata: Optional metadata to attach
            content_type: Optional content type

        Returns:
            Dict with upload details including ETag
        """
        bucket = bucket_name or self.bucket_name

        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata
            if content_type:
                extra_args["ContentType"] = content_type

            async with self.get_client() as client:
                response = await client.put_object(
                    Bucket=bucket, Key=key, Body=data, **extra_args
                )

                logger.info(
                    "Bytes uploaded successfully",
                    bucket=bucket,
                    key=key,
                    etag=response.get("ETag"),
                )

                return {
                    "bucket": bucket,
                    "key": key,
                    "etag": response.get("ETag"),
                    "version_id": response.get("VersionId"),
                }

        except Exception as e:
            logger.error("Failed to upload bytes", bucket=bucket, key=key, error=str(e))
            raise

    async def download_file(
        self,
        key: str,
        file_path: str,
        bucket_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Download a file from R2 storage.

        Args:
            key: Object key in R2
            file_path: Local file path to save to
            bucket_name: Bucket name (defaults to configured bucket)

        Returns:
            Dict with download details
        """
        bucket = bucket_name or self.bucket_name

        try:
            async with self.get_client() as client:
                response = await client.get_object(Bucket=bucket, Key=key)

                # Create directory if needed
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Write to file
                with open(file_path, "wb") as file:
                    async for chunk in response["Body"]:
                        file.write(chunk)

                logger.info(
                    "File downloaded successfully",
                    bucket=bucket,
                    key=key,
                    file_path=file_path,
                )

                return {
                    "bucket": bucket,
                    "key": key,
                    "file_path": file_path,
                    "content_length": response.get("ContentLength"),
                    "etag": response.get("ETag"),
                    "last_modified": response.get("LastModified"),
                }

        except Exception as e:
            logger.error(
                "Failed to download file", bucket=bucket, key=key, error=str(e)
            )
            raise

    async def download_bytes(
        self,
        key: str,
        bucket_name: Optional[str] = None,
    ) -> bytes:
        """
        Download an object as bytes from R2 storage.

        Args:
            key: Object key in R2
            bucket_name: Bucket name (defaults to configured bucket)

        Returns:
            Object content as bytes
        """
        bucket = bucket_name or self.bucket_name

        try:
            async with self.get_client() as client:
                response = await client.get_object(Bucket=bucket, Key=key)

                # Read all content
                content = b""
                async for chunk in response["Body"]:
                    content += chunk

                logger.info(
                    "Bytes downloaded successfully",
                    bucket=bucket,
                    key=key,
                    size=len(content),
                )

                return content

        except Exception as e:
            logger.error(
                "Failed to download bytes", bucket=bucket, key=key, error=str(e)
            )
            raise

    async def delete_object(
        self,
        key: str,
        bucket_name: Optional[str] = None,
    ) -> bool:
        """
        Delete an object from R2 storage.

        Args:
            key: Object key in R2
            bucket_name: Bucket name (defaults to configured bucket)

        Returns:
            True if deleted successfully
        """
        bucket = bucket_name or self.bucket_name

        try:
            async with self.get_client() as client:
                await client.delete_object(Bucket=bucket, Key=key)

                logger.info("Object deleted successfully", bucket=bucket, key=key)
                return True

        except Exception as e:
            logger.error(
                "Failed to delete object", bucket=bucket, key=key, error=str(e)
            )
            return False

    async def list_objects(
        self,
        prefix: Optional[str] = None,
        bucket_name: Optional[str] = None,
        max_keys: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        List objects in R2 storage.

        Args:
            prefix: Object key prefix to filter by
            bucket_name: Bucket name (defaults to configured bucket)
            max_keys: Maximum number of keys to return

        Returns:
            List of object metadata
        """
        bucket = bucket_name or self.bucket_name
        objects = []

        try:
            async with self.get_client() as client:
                paginator = client.get_paginator("list_objects_v2")

                params = {
                    "Bucket": bucket,
                    "MaxKeys": max_keys,
                }
                if prefix:
                    params["Prefix"] = prefix

                async for page in paginator.paginate(**params):
                    for obj in page.get("Contents", []):
                        objects.append(
                            {
                                "key": obj["Key"],
                                "size": obj["Size"],
                                "last_modified": obj["LastModified"],
                                "etag": obj["ETag"],
                                "storage_class": obj.get("StorageClass", "STANDARD"),
                            }
                        )

                logger.info(
                    "Listed objects successfully",
                    bucket=bucket,
                    prefix=prefix,
                    count=len(objects),
                )

                return objects

        except Exception as e:
            logger.error(
                "Failed to list objects", bucket=bucket, prefix=prefix, error=str(e)
            )
            raise

    async def object_exists(
        self,
        key: str,
        bucket_name: Optional[str] = None,
    ) -> bool:
        """
        Check if an object exists in R2 storage.

        Args:
            key: Object key in R2
            bucket_name: Bucket name (defaults to configured bucket)

        Returns:
            True if object exists
        """
        bucket = bucket_name or self.bucket_name

        try:
            async with self.get_client() as client:
                await client.head_object(Bucket=bucket, Key=key)
                return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    async def get_object_metadata(
        self,
        key: str,
        bucket_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get object metadata from R2 storage.

        Args:
            key: Object key in R2
            bucket_name: Bucket name (defaults to configured bucket)

        Returns:
            Object metadata or None if not found
        """
        bucket = bucket_name or self.bucket_name

        try:
            async with self.get_client() as client:
                response = await client.head_object(Bucket=bucket, Key=key)

                return {
                    "content_length": response.get("ContentLength"),
                    "content_type": response.get("ContentType"),
                    "etag": response.get("ETag"),
                    "last_modified": response.get("LastModified"),
                    "metadata": response.get("Metadata", {}),
                    "version_id": response.get("VersionId"),
                }

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return None
            raise

    async def generate_presigned_url(
        self,
        key: str,
        operation: str = "get_object",
        expires_in: int = 3600,
        bucket_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate a presigned URL for an object.

        Args:
            key: Object key in R2
            operation: S3 operation (get_object, put_object)
            expires_in: URL expiration time in seconds
            bucket_name: Bucket name (defaults to configured bucket)
            **kwargs: Additional parameters for the operation

        Returns:
            Presigned URL
        """
        bucket = bucket_name or self.bucket_name

        try:
            async with self.get_client() as client:
                url = await client.generate_presigned_url(
                    ClientMethod=operation,
                    Params={"Bucket": bucket, "Key": key, **kwargs},
                    ExpiresIn=expires_in,
                )

                logger.info(
                    "Generated presigned URL",
                    bucket=bucket,
                    key=key,
                    operation=operation,
                    expires_in=expires_in,
                )

                return url

        except Exception as e:
            logger.error(
                "Failed to generate presigned URL", bucket=bucket, key=key, error=str(e)
            )
            raise


# Global client instance
_r2_client: Optional[R2StorageClient] = None


async def get_r2_client() -> R2StorageClient:
    """
    Get or create the global R2 storage client instance.

    Returns:
        R2StorageClient instance
    """
    global _r2_client

    if _r2_client is None:
        _r2_client = R2StorageClient()

    return _r2_client
