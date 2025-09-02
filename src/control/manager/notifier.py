"""WebSocket notification system for real-time updates."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from enum import Enum

from src.control.api.routers.websocket import manager as ws_manager

logger = logging.getLogger(__name__)


class NotificationTopic(str, Enum):
    """WebSocket notification topics."""
    METRICS = "metrics"
    LOGS = "logs"
    ALERTS = "alerts"
    ACTIVITY = "activity"


class NotificationPriority(str, Enum):
    """Notification priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WebSocketNotifier:
    """Manages WebSocket notifications for control plane events."""
    
    def __init__(self, batch_interval: float = 0.1):
        """Initialize WebSocket notifier."""
        self.batch_interval = batch_interval
        self.is_running = False
        
        # Notification queues by topic
        self.queues: Dict[NotificationTopic, asyncio.Queue] = {
            topic: asyncio.Queue() for topic in NotificationTopic
        }
        
        # Batch tasks
        self._batch_tasks: Dict[NotificationTopic, Optional[asyncio.Task]] = {
            topic: None for topic in NotificationTopic
        }
        
        # Rate limiting
        self.rate_limits: Dict[str, int] = {
            NotificationTopic.METRICS: 10,  # Max 10/sec
            NotificationTopic.LOGS: 20,      # Max 20/sec
            NotificationTopic.ALERTS: 50,    # Max 50/sec (important)
            NotificationTopic.ACTIVITY: 30,  # Max 30/sec
        }
        
        # Notification statistics
        self.stats = {
            "sent": 0,
            "dropped": 0,
            "by_topic": {topic.value: 0 for topic in NotificationTopic},
        }
    
    async def start(self):
        """Start notification system."""
        if self.is_running:
            return
        
        logger.info("Starting WebSocket notifier")
        self.is_running = True
        
        # Start batch tasks for each topic
        for topic in NotificationTopic:
            self._batch_tasks[topic] = asyncio.create_task(
                self._batch_processor(topic)
            )
    
    async def stop(self):
        """Stop notification system."""
        if not self.is_running:
            return
        
        logger.info("Stopping WebSocket notifier")
        self.is_running = False
        
        # Cancel all batch tasks
        for task in self._batch_tasks.values():
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def notify_metrics(self, metrics: Dict[str, Any]):
        """Send metrics update."""
        await self._enqueue_notification(
            NotificationTopic.METRICS,
            {
                "type": "metrics",
                "timestamp": datetime.utcnow().isoformat(),
                "data": metrics,
            },
            priority=NotificationPriority.LOW,
        )
    
    async def notify_log(
        self,
        level: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Send log entry."""
        await self._enqueue_notification(
            NotificationTopic.LOGS,
            {
                "type": "log",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "level": level,
                    "message": message,
                    "context": context or {},
                },
            },
            priority=NotificationPriority.MEDIUM,
        )
    
    async def notify_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Send alert notification."""
        priority = (
            NotificationPriority.CRITICAL
            if severity == "critical"
            else NotificationPriority.HIGH
        )
        
        await self._enqueue_notification(
            NotificationTopic.ALERTS,
            {
                "type": "alert",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "alert_type": alert_type,
                    "severity": severity,
                    "message": message,
                    "metadata": metadata or {},
                },
            },
            priority=priority,
        )
    
    async def notify_activity(
        self,
        activity_type: str,
        description: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Send activity notification."""
        await self._enqueue_notification(
            NotificationTopic.ACTIVITY,
            {
                "type": "activity",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "activity_type": activity_type,
                    "description": description,
                    "user_id": user_id,
                    "metadata": metadata or {},
                },
            },
            priority=NotificationPriority.MEDIUM,
        )
    
    async def notify_server_event(
        self,
        server_id: str,
        event_type: str,
        details: Dict[str, Any],
    ):
        """Send server-specific event notification."""
        # Determine topic based on event type
        if event_type in ["start", "stop", "restart", "scale"]:
            topic = NotificationTopic.ACTIVITY
        elif event_type in ["error", "warning", "failure"]:
            topic = NotificationTopic.ALERTS
        else:
            topic = NotificationTopic.METRICS
        
        await self._enqueue_notification(
            topic,
            {
                "type": "server_event",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "server_id": server_id,
                    "event_type": event_type,
                    **details,
                },
            },
            priority=NotificationPriority.HIGH,
        )
    
    async def _enqueue_notification(
        self,
        topic: NotificationTopic,
        notification: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.MEDIUM,
    ):
        """Add notification to queue."""
        if not self.is_running:
            return
        
        # Add priority to notification
        notification["priority"] = priority.value
        
        # Check queue size for rate limiting
        queue = self.queues[topic]
        if queue.qsize() > self.rate_limits.get(topic.value, 100):
            self.stats["dropped"] += 1
            logger.warning(f"Dropping {topic} notification due to queue overflow")
            return
        
        await queue.put(notification)
    
    async def _batch_processor(self, topic: NotificationTopic):
        """Process notifications in batches."""
        queue = self.queues[topic]
        batch = []
        
        while self.is_running:
            try:
                # Collect notifications for batch
                deadline = asyncio.get_event_loop().time() + self.batch_interval
                
                while asyncio.get_event_loop().time() < deadline:
                    try:
                        timeout = deadline - asyncio.get_event_loop().time()
                        if timeout > 0:
                            notification = await asyncio.wait_for(
                                queue.get(),
                                timeout=timeout,
                            )
                            batch.append(notification)
                            
                            # Send immediately if critical
                            if notification.get("priority") == NotificationPriority.CRITICAL.value:
                                break
                    except asyncio.TimeoutError:
                        break
                
                # Send batch if not empty
                if batch:
                    await self._send_batch(topic, batch)
                    self.stats["sent"] += len(batch)
                    self.stats["by_topic"][topic.value] += len(batch)
                    batch = []
                
                # Small delay to prevent tight loop
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Batch processor error for {topic}: {e}")
                await asyncio.sleep(1)
    
    async def _send_batch(self, topic: NotificationTopic, batch: List[Dict[str, Any]]):
        """Send a batch of notifications."""
        # Group by priority
        priority_groups = {}
        for notification in batch:
            priority = notification.pop("priority", NotificationPriority.MEDIUM.value)
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(notification)
        
        # Send high priority first
        for priority in [
            NotificationPriority.CRITICAL.value,
            NotificationPriority.HIGH.value,
            NotificationPriority.MEDIUM.value,
            NotificationPriority.LOW.value,
        ]:
            if priority in priority_groups:
                notifications = priority_groups[priority]
                
                # Send individually for critical, batch for others
                if priority == NotificationPriority.CRITICAL.value:
                    for notification in notifications:
                        await ws_manager.broadcast(notification, topic.value)
                else:
                    # Batch notification
                    batch_notification = {
                        "type": "batch",
                        "topic": topic.value,
                        "count": len(notifications),
                        "notifications": notifications,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    await ws_manager.broadcast(batch_notification, topic.value)
    
    async def broadcast_custom(
        self,
        message: Dict[str, Any],
        topics: Optional[List[NotificationTopic]] = None,
    ):
        """Broadcast custom message to specific topics."""
        if not topics:
            topics = list(NotificationTopic)
        
        for topic in topics:
            await ws_manager.broadcast(message, topic.value)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification statistics."""
        return {
            "total_sent": self.stats["sent"],
            "total_dropped": self.stats["dropped"],
            "by_topic": dict(self.stats["by_topic"]),
            "queue_sizes": {
                topic.value: self.queues[topic].qsize()
                for topic in NotificationTopic
            },
            "connected_clients": len(ws_manager.active_connections),
            "subscriptions": {
                topic: len(subs)
                for topic, subs in ws_manager.subscriptions.items()
            },
        }
    
    async def send_system_status(self, status: Dict[str, Any]):
        """Send system-wide status update."""
        await self.broadcast_custom(
            {
                "type": "system_status",
                "timestamp": datetime.utcnow().isoformat(),
                "data": status,
            },
            topics=[NotificationTopic.METRICS, NotificationTopic.ACTIVITY],
        )
    
    async def send_performance_report(self, report: Dict[str, Any]):
        """Send performance report."""
        await self.notify_metrics({
            "performance": report,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    async def test_connection(self, client_id: str) -> bool:
        """Test if a client is connected."""
        return client_id in ws_manager.active_connections