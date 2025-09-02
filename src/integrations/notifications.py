"""Notification system for alerts and updates via multiple channels."""

# Standard library imports
import asyncio
import json
import logging
import os
import smtplib
from dataclasses import asdict, dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Third-party imports
import aiohttp
import aiosmtplib
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Local imports
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class NotificationType(str, Enum):
    """Notification types."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SUCCESS = "success"


class NotificationChannel(str, Enum):
    """Notification channels."""

    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"


@dataclass
class NotificationConfig:
    """Notification configuration."""

    email_enabled: bool = False
    email_smtp_host: Optional[str] = None
    email_smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: List[str] = None

    webhook_enabled: bool = True
    webhook_url: Optional[str] = None

    slack_enabled: bool = False
    slack_webhook_url: Optional[str] = None

    discord_enabled: bool = False
    discord_webhook_url: Optional[str] = None

    def __post_init__(self):
        """Load from environment variables if not provided."""
        if self.email_smtp_host is None:
            self.email_smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        if self.email_username is None:
            self.email_username = os.getenv("SMTP_USERNAME")
        if self.email_password is None:
            self.email_password = os.getenv("SMTP_PASSWORD")
        if self.email_from is None:
            self.email_from = os.getenv("EMAIL_FROM")
        if self.email_to is None:
            email_to = os.getenv("EMAIL_TO", "")
            self.email_to = [e.strip() for e in email_to.split(",") if e.strip()]

        if self.webhook_url is None:
            self.webhook_url = os.getenv("WEBHOOK_URL")
        if self.slack_webhook_url is None:
            self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if self.discord_webhook_url is None:
            self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

        # Enable channels based on configuration
        self.email_enabled = bool(
            self.email_username and self.email_password and self.email_to
        )
        self.webhook_enabled = bool(self.webhook_url)
        self.slack_enabled = bool(self.slack_webhook_url)
        self.discord_enabled = bool(self.discord_webhook_url)


class NotificationError(Exception):
    """Notification error."""

    pass


class NotificationClient:
    """
    Multi-channel notification client.

    Features:
    - Send notifications via email, webhooks, Slack, Discord
    - Priority-based routing
    - Template support for formatted messages
    - Async operations with retry logic
    - Batch notification support

    Example:
        ```python
        client = NotificationClient()

        # Send critical alert to all channels
        await client.send_notification(
            title="Critical Error",
            message="GPU memory exceeded threshold",
            notification_type=NotificationType.CRITICAL,
            data={"gpu_usage": 95.5, "threshold": 90}
        )

        # Send to specific channels
        await client.send_notification(
            title="Analysis Complete",
            message="Batch processing finished",
            notification_type=NotificationType.SUCCESS,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
        )
        ```
    """

    MAX_RETRIES = 3
    DEFAULT_TIMEOUT = 10

    def __init__(self, config: Optional[NotificationConfig] = None):
        """
        Initialize notification client.

        Args:
            config: Notification configuration
        """
        self.config = config or NotificationConfig()
        self.session: Optional[aiohttp.ClientSession] = None

        # Log enabled channels
        enabled_channels = []
        if self.config.email_enabled:
            enabled_channels.append("email")
        if self.config.webhook_enabled:
            enabled_channels.append("webhook")
        if self.config.slack_enabled:
            enabled_channels.append("Slack")
        if self.config.discord_enabled:
            enabled_channels.append("Discord")

        logger.info(
            f"Initialized notification client with channels: {enabled_channels}"
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    def _get_channels_for_type(
        self, notification_type: NotificationType
    ) -> List[NotificationChannel]:
        """
        Get appropriate channels based on notification type.

        Args:
            notification_type: Type of notification

        Returns:
            List of channels to use
        """
        channels = []

        # Critical notifications go to all channels
        if notification_type == NotificationType.CRITICAL:
            if self.config.email_enabled:
                channels.append(NotificationChannel.EMAIL)
            if self.config.webhook_enabled:
                channels.append(NotificationChannel.WEBHOOK)
            if self.config.slack_enabled:
                channels.append(NotificationChannel.SLACK)
            if self.config.discord_enabled:
                channels.append(NotificationChannel.DISCORD)

        # Errors go to email and Slack
        elif notification_type == NotificationType.ERROR:
            if self.config.email_enabled:
                channels.append(NotificationChannel.EMAIL)
            if self.config.slack_enabled:
                channels.append(NotificationChannel.SLACK)
            if self.config.webhook_enabled:
                channels.append(NotificationChannel.WEBHOOK)

        # Warnings and info go to webhook and Discord
        else:
            if self.config.webhook_enabled:
                channels.append(NotificationChannel.WEBHOOK)
            if self.config.discord_enabled:
                channels.append(NotificationChannel.DISCORD)

        return channels

    async def send_notification(
        self,
        title: str,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        data: Optional[Dict[str, Any]] = None,
        channels: Optional[List[NotificationChannel]] = None,
    ) -> Dict[str, Any]:
        """
        Send notification to specified channels.

        Args:
            title: Notification title
            message: Notification message
            notification_type: Type of notification
            data: Additional data
            channels: Specific channels to use (if None, auto-select based on type)

        Returns:
            Results from each channel
        """
        # Auto-select channels if not specified
        if channels is None:
            channels = self._get_channels_for_type(notification_type)

        # Prepare notification data
        notification_data = {
            "title": title,
            "message": message,
            "type": notification_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data or {},
        }

        results = {}

        # Send to each channel
        for channel in channels:
            try:
                if channel == NotificationChannel.EMAIL:
                    results["email"] = await self._send_email(notification_data)
                elif channel == NotificationChannel.WEBHOOK:
                    results["webhook"] = await self._send_webhook(notification_data)
                elif channel == NotificationChannel.SLACK:
                    results["slack"] = await self._send_slack(notification_data)
                elif channel == NotificationChannel.DISCORD:
                    results["discord"] = await self._send_discord(notification_data)
            except Exception as e:
                logger.error(f"Failed to send to {channel}: {e}")
                results[channel.value] = {"success": False, "error": str(e)}

        return results

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _send_email(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send email notification."""
        if not self.config.email_enabled:
            raise NotificationError("Email notifications not configured")

        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{data['type'].upper()}] {data['title']}"
        msg["From"] = self.config.email_from
        msg["To"] = ", ".join(self.config.email_to)

        # Create HTML content
        html_content = self._format_email_html(data)
        msg.attach(MIMEText(html_content, "html"))

        # Send email
        async with aiosmtplib.SMTP(
            hostname=self.config.email_smtp_host,
            port=self.config.email_smtp_port,
            use_tls=True,
        ) as smtp:
            await smtp.login(self.config.email_username, self.config.email_password)
            await smtp.send_message(msg)

        logger.info(f"Email sent to {len(self.config.email_to)} recipients")
        return {"success": True, "recipients": len(self.config.email_to)}

    def _format_email_html(self, data: Dict[str, Any]) -> str:
        """Format email HTML content."""
        type_colors = {
            "info": "#2196F3",
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#f44336",
            "critical": "#9C27B0",
        }

        color = type_colors.get(data["type"], "#2196F3")

        html = f"""
        <html>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <div style="max-width: 600px; margin: 0 auto;">
                    <div style="background-color: {color}; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
                        <h2 style="margin: 0;">{data['title']}</h2>
                    </div>
                    <div style="background-color: #f5f5f5; padding: 20px; border: 1px solid #ddd; border-top: none;">
                        <p style="font-size: 16px; line-height: 1.5;">{data['message']}</p>
                        <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
                        <p style="color: #666; font-size: 14px;">
                            <strong>Timestamp:</strong> {data['timestamp']}<br>
                            <strong>Type:</strong> {data['type'].upper()}
                        </p>
        """

        if data.get("data"):
            html += """
                        <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
                        <h3>Additional Data:</h3>
                        <pre style="background-color: #fff; padding: 10px; border: 1px solid #ddd; border-radius: 3px; overflow-x: auto;">"""
            html += json.dumps(data["data"], indent=2)
            html += "</pre>"

        html += """
                    </div>
                    <div style="text-align: center; padding: 10px; color: #999; font-size: 12px;">
                        Sent by H200 Intelligent Mug Positioning System
                    </div>
                </div>
            </body>
        </html>
        """

        return html

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _send_webhook(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send webhook notification."""
        if not self.config.webhook_enabled:
            raise NotificationError("Webhook notifications not configured")

        await self._ensure_session()

        timeout = aiohttp.ClientTimeout(total=self.DEFAULT_TIMEOUT)
        async with self.session.post(
            self.config.webhook_url, json=data, timeout=timeout
        ) as response:
            response.raise_for_status()
            return {"success": True, "status": response.status}

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _send_slack(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send Slack notification."""
        if not self.config.slack_enabled:
            raise NotificationError("Slack notifications not configured")

        await self._ensure_session()

        # Format Slack message
        slack_data = {
            "text": data["title"],
            "attachments": [
                {
                    "color": self._get_slack_color(data["type"]),
                    "fields": [
                        {"title": "Message", "value": data["message"], "short": False},
                        {"title": "Type", "value": data["type"].upper(), "short": True},
                        {
                            "title": "Timestamp",
                            "value": data["timestamp"],
                            "short": True,
                        },
                    ],
                    "footer": "H200 Mug Positioning System",
                }
            ],
        }

        # Add data fields if present
        if data.get("data"):
            for key, value in data["data"].items():
                slack_data["attachments"][0]["fields"].append(
                    {
                        "title": key.replace("_", " ").title(),
                        "value": str(value),
                        "short": True,
                    }
                )

        timeout = aiohttp.ClientTimeout(total=self.DEFAULT_TIMEOUT)
        async with self.session.post(
            self.config.slack_webhook_url, json=slack_data, timeout=timeout
        ) as response:
            response.raise_for_status()
            return {"success": True, "status": response.status}

    def _get_slack_color(self, notification_type: str) -> str:
        """Get Slack attachment color based on notification type."""
        colors = {
            "info": "#2196F3",
            "success": "good",
            "warning": "warning",
            "error": "danger",
            "critical": "#9C27B0",
        }
        return colors.get(notification_type, "#2196F3")

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _send_discord(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send Discord notification."""
        if not self.config.discord_enabled:
            raise NotificationError("Discord notifications not configured")

        await self._ensure_session()

        # Format Discord embed
        discord_data = {
            "embeds": [
                {
                    "title": data["title"],
                    "description": data["message"],
                    "color": self._get_discord_color(data["type"]),
                    "fields": [
                        {"name": "Type", "value": data["type"].upper(), "inline": True},
                        {
                            "name": "Timestamp",
                            "value": data["timestamp"],
                            "inline": True,
                        },
                    ],
                    "footer": {"text": "H200 Mug Positioning System"},
                }
            ]
        }

        # Add data fields if present
        if data.get("data"):
            for key, value in data["data"].items():
                discord_data["embeds"][0]["fields"].append(
                    {
                        "name": key.replace("_", " ").title(),
                        "value": str(value),
                        "inline": True,
                    }
                )

        timeout = aiohttp.ClientTimeout(total=self.DEFAULT_TIMEOUT)
        async with self.session.post(
            self.config.discord_webhook_url, json=discord_data, timeout=timeout
        ) as response:
            response.raise_for_status()
            return {"success": True, "status": response.status}

    def _get_discord_color(self, notification_type: str) -> int:
        """Get Discord embed color based on notification type."""
        colors = {
            "info": 0x2196F3,
            "success": 0x4CAF50,
            "warning": 0xFF9800,
            "error": 0xF44336,
            "critical": 0x9C27B0,
        }
        return colors.get(notification_type, 0x2196F3)

    async def send_batch_notifications(
        self, notifications: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Send multiple notifications.

        Args:
            notifications: List of notification configurations

        Returns:
            List of results
        """
        results = []

        for notification in notifications:
            try:
                result = await self.send_notification(**notification)
                results.append({"success": True, "result": result})
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
                results.append({"success": False, "error": str(e)})

        return results

    async def send_system_startup(self):
        """Send system startup notification."""
        await self.send_notification(
            title="System Started",
            message="H200 Intelligent Mug Positioning System has started successfully",
            notification_type=NotificationType.INFO,
            data={
                "version": "1.0",
                "mode": os.getenv("DEPLOYMENT_MODE", "development"),
            },
        )

    async def send_error_alert(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Send error alert."""
        await self.send_notification(
            title=f"Error: {error_type}",
            message=error_message,
            notification_type=NotificationType.ERROR,
            data=context or {},
        )

    async def send_performance_alert(
        self, metric_name: str, current_value: float, threshold: float
    ):
        """Send performance alert."""
        severity = NotificationType.WARNING
        if current_value > threshold * 1.5:
            severity = NotificationType.CRITICAL

        await self.send_notification(
            title=f"Performance Alert: {metric_name}",
            message=f"{metric_name} has exceeded threshold: {current_value:.2f} (threshold: {threshold:.2f})",
            notification_type=severity,
            data={
                "metric": metric_name,
                "current_value": current_value,
                "threshold": threshold,
                "exceeded_by": f"{((current_value / threshold - 1) * 100):.1f}%",
            },
        )


# Usage example
async def main():
    """Example usage of NotificationClient."""
    # Initialize client
    async with NotificationClient() as client:
        # Example 1: Send critical alert
        await client.send_notification(
            title="Critical GPU Error",
            message="GPU memory usage exceeded 95%, system may become unstable",
            notification_type=NotificationType.CRITICAL,
            data={
                "gpu_usage": 95.5,
                "available_memory": "500MB",
                "total_memory": "16GB",
            },
        )

        # Example 2: Send success notification to specific channels
        await client.send_notification(
            title="Batch Processing Complete",
            message="Successfully processed 1000 images",
            notification_type=NotificationType.SUCCESS,
            channels=[NotificationChannel.WEBHOOK, NotificationChannel.SLACK],
            data={
                "images_processed": 1000,
                "processing_time": "45.3s",
                "average_confidence": 0.92,
            },
        )

        # Example 3: Send system startup notification
        await client.send_system_startup()

        # Example 4: Send performance alert
        await client.send_performance_alert(
            metric_name="API Response Time", current_value=850, threshold=500
        )

        # Example 5: Batch notifications
        batch_notifications = [
            {
                "title": "Model Updated",
                "message": "YOLO model updated to v1.2",
                "notification_type": NotificationType.INFO,
            },
            {
                "title": "Cache Cleared",
                "message": "Redis cache has been cleared",
                "notification_type": NotificationType.INFO,
            },
        ]

        results = await client.send_batch_notifications(batch_notifications)
        logger.info(f"Sent {len(results)} notifications")


if __name__ == "__main__":
    asyncio.run(main())
