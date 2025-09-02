"""N8N webhook integration for workflow automation."""

# Standard library imports
import asyncio
import json
import logging
import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Third-party imports
import aiohttp
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


class N8NEventType(str, Enum):
    """N8N webhook event types."""

    MUG_DETECTED = "mug_detected"
    POSITIONING_ANALYZED = "positioning_analyzed"
    RULE_TRIGGERED = "rule_triggered"
    ERROR_OCCURRED = "error_occurred"
    SYSTEM_ALERT = "system_alert"
    BATCH_COMPLETED = "batch_completed"
    MODEL_UPDATED = "model_updated"


class N8NWebhookError(Exception):
    """N8N webhook error."""

    pass


class N8NClient:
    """
    Async client for N8N webhook integration.

    Features:
    - Send events to N8N workflows
    - Support for different event types
    - Batch event sending
    - Async operations with retry logic
    - Event validation and enrichment

    Example:
        ```python
        client = N8NClient()

        # Send positioning analysis event
        await client.send_event(
            event_type=N8NEventType.POSITIONING_ANALYZED,
            data={
                "image_id": "img_123",
                "mug_position": {"x": 100, "y": 200},
                "confidence": 0.95,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        # Send batch of events
        events = [
            {"type": N8NEventType.MUG_DETECTED, "data": {...}},
            {"type": N8NEventType.RULE_TRIGGERED, "data": {...}}
        ]
        await client.send_batch_events(events)
        ```
    """

    DEFAULT_TIMEOUT = 10
    MAX_RETRIES = 3
    BATCH_SIZE_LIMIT = 100

    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize N8N client.

        Args:
            webhook_url: N8N webhook URL. If not provided, reads from N8N_WEBHOOK_URL env var.
        """
        self.webhook_url = webhook_url or os.getenv("N8N_WEBHOOK_URL")
        if not self.webhook_url:
            raise ValueError("N8N webhook URL not provided")

        self.session: Optional[aiohttp.ClientSession] = None
        self._event_queue: List[Dict[str, Any]] = []
        self._queue_lock = asyncio.Lock()
        logger.info(f"Initialized N8N client with webhook: {self.webhook_url}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Flush any remaining events
        if self._event_queue:
            await self.flush_events()
        await self.close()

    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "H200-Mug-Positioning/1.0",
                }
            )

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _send_webhook(
        self, data: Dict[str, Any], timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Send data to N8N webhook with retry logic.

        Args:
            data: Data to send
            timeout: Request timeout

        Returns:
            Response data

        Raises:
            N8NWebhookError: On webhook errors
        """
        await self._ensure_session()

        timeout_obj = aiohttp.ClientTimeout(total=timeout or self.DEFAULT_TIMEOUT)

        try:
            async with self.session.post(
                self.webhook_url, json=data, timeout=timeout_obj
            ) as response:
                response_data = {}

                # Try to parse JSON response
                try:
                    response_data = await response.json()
                except:
                    response_data = {"status": response.status}

                if response.status >= 400:
                    error_msg = response_data.get("error", "Webhook request failed")
                    raise N8NWebhookError(
                        f"Webhook error {response.status}: {error_msg}"
                    )

                return response_data

        except aiohttp.ClientError as e:
            logger.error(f"Webhook request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise N8NWebhookError(f"Webhook failed: {e}")

    def _enrich_event(
        self, event_type: N8NEventType, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich event with metadata.

        Args:
            event_type: Type of event
            data: Event data

        Returns:
            Enriched event data
        """
        return {
            "event_type": event_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "h200-mug-positioning",
            "version": "1.0",
            "data": data,
        }

    async def send_event(
        self, event_type: N8NEventType, data: Dict[str, Any], priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Send a single event to N8N.

        Args:
            event_type: Type of event
            data: Event data
            priority: Event priority (low, normal, high)

        Returns:
            Response from N8N
        """
        event = self._enrich_event(event_type, data)
        event["priority"] = priority

        logger.info(f"Sending {event_type.value} event to N8N")
        response = await self._send_webhook(event)
        logger.info(f"Event sent successfully: {event_type.value}")

        return response

    async def send_batch_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Send multiple events in a single request.

        Args:
            events: List of events to send

        Returns:
            Response from N8N
        """
        if len(events) > self.BATCH_SIZE_LIMIT:
            raise ValueError(f"Batch size exceeds limit of {self.BATCH_SIZE_LIMIT}")

        # Enrich all events
        enriched_events = []
        for event in events:
            event_type = event.get("type", N8NEventType.SYSTEM_ALERT)
            if isinstance(event_type, str):
                event_type = N8NEventType(event_type)

            enriched = self._enrich_event(event_type, event.get("data", {}))
            enriched["priority"] = event.get("priority", "normal")
            enriched_events.append(enriched)

        batch_data = {
            "batch": True,
            "events": enriched_events,
            "batch_id": f"batch_{datetime.utcnow().timestamp()}",
        }

        logger.info(f"Sending batch of {len(events)} events to N8N")
        response = await self._send_webhook(batch_data)
        logger.info(f"Batch sent successfully: {len(events)} events")

        return response

    async def queue_event(
        self, event_type: N8NEventType, data: Dict[str, Any], priority: str = "normal"
    ):
        """
        Queue an event for batch sending.

        Args:
            event_type: Type of event
            data: Event data
            priority: Event priority
        """
        async with self._queue_lock:
            self._event_queue.append(
                {"type": event_type, "data": data, "priority": priority}
            )

            # Auto-flush if queue is full
            if len(self._event_queue) >= self.BATCH_SIZE_LIMIT:
                await self.flush_events()

    async def flush_events(self) -> Optional[Dict[str, Any]]:
        """
        Flush all queued events.

        Returns:
            Response from N8N if events were sent
        """
        async with self._queue_lock:
            if not self._event_queue:
                return None

            events = self._event_queue.copy()
            self._event_queue.clear()

        return await self.send_batch_events(events)

    async def send_mug_detection(
        self,
        image_id: str,
        mug_count: int,
        positions: List[Dict[str, Any]],
        confidence_scores: List[float],
        processing_time: float,
    ) -> Dict[str, Any]:
        """
        Send mug detection event.

        Args:
            image_id: ID of analyzed image
            mug_count: Number of mugs detected
            positions: List of mug positions
            confidence_scores: Confidence scores for each detection
            processing_time: Time taken to process

        Returns:
            Response from N8N
        """
        return await self.send_event(
            N8NEventType.MUG_DETECTED,
            {
                "image_id": image_id,
                "mug_count": mug_count,
                "positions": positions,
                "confidence_scores": confidence_scores,
                "processing_time_ms": processing_time * 1000,
                "average_confidence": (
                    sum(confidence_scores) / len(confidence_scores)
                    if confidence_scores
                    else 0
                ),
            },
        )

    async def send_positioning_analysis(
        self,
        image_id: str,
        positioning_result: Dict[str, Any],
        rules_applied: List[str],
        recommendations: List[str],
    ) -> Dict[str, Any]:
        """
        Send positioning analysis event.

        Args:
            image_id: ID of analyzed image
            positioning_result: Positioning analysis result
            rules_applied: List of rules that were applied
            recommendations: Positioning recommendations

        Returns:
            Response from N8N
        """
        return await self.send_event(
            N8NEventType.POSITIONING_ANALYZED,
            {
                "image_id": image_id,
                "result": positioning_result,
                "rules_applied": rules_applied,
                "recommendations": recommendations,
                "quality_score": positioning_result.get("confidence", 0),
            },
        )

    async def send_error(
        self,
        error_type: str,
        error_message: str,
        context: Dict[str, Any],
        severity: str = "error",
    ) -> Dict[str, Any]:
        """
        Send error event.

        Args:
            error_type: Type of error
            error_message: Error message
            context: Error context
            severity: Error severity (warning, error, critical)

        Returns:
            Response from N8N
        """
        return await self.send_event(
            N8NEventType.ERROR_OCCURRED,
            {
                "error_type": error_type,
                "message": error_message,
                "context": context,
                "severity": severity,
            },
            priority="high" if severity == "critical" else "normal",
        )

    async def send_system_alert(
        self, alert_type: str, message: str, metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send system alert.

        Args:
            alert_type: Type of alert
            message: Alert message
            metrics: Optional system metrics

        Returns:
            Response from N8N
        """
        return await self.send_event(
            N8NEventType.SYSTEM_ALERT,
            {"alert_type": alert_type, "message": message, "metrics": metrics or {}},
        )


# Usage example
async def main():
    """Example usage of N8NClient."""
    # Initialize client
    async with N8NClient() as client:
        # Example 1: Send mug detection event
        detection_result = await client.send_mug_detection(
            image_id="img_12345",
            mug_count=3,
            positions=[
                {"x": 100, "y": 150, "width": 80, "height": 120},
                {"x": 250, "y": 200, "width": 85, "height": 125},
                {"x": 400, "y": 180, "width": 82, "height": 122},
            ],
            confidence_scores=[0.95, 0.92, 0.88],
            processing_time=0.245,
        )
        logger.info(f"Detection event sent: {detection_result}")

        # Example 2: Send positioning analysis
        analysis_result = await client.send_positioning_analysis(
            image_id="img_12345",
            positioning_result={
                "status": "optimal",
                "confidence": 0.91,
                "alignment_score": 0.94,
            },
            rules_applied=["minimum_spacing", "edge_detection", "symmetry_check"],
            recommendations=["Move mug 2 slightly left for better alignment"],
        )
        logger.info(f"Analysis event sent: {analysis_result}")

        # Example 3: Queue multiple events and flush
        await client.queue_event(
            N8NEventType.SYSTEM_ALERT, {"type": "performance", "cpu_usage": 78.5}
        )
        await client.queue_event(
            N8NEventType.MODEL_UPDATED, {"model": "yolo", "version": "1.2.0"}
        )

        # Flush queued events
        flush_result = await client.flush_events()
        logger.info(f"Flushed events: {flush_result}")

        # Example 4: Send error event
        error_result = await client.send_error(
            error_type="ProcessingError",
            error_message="Failed to load YOLO model",
            context={"model_path": "/models/yolo.pth", "error_code": "MODEL_NOT_FOUND"},
            severity="critical",
        )
        logger.info(f"Error event sent: {error_result}")


if __name__ == "__main__":
    asyncio.run(main())
