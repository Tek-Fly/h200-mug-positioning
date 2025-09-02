"""External integrations for H200 Intelligent Mug Positioning System."""

from .templated import TemplatedClient
from .n8n import N8NClient, N8NEventType
from .notifications import NotificationClient, NotificationType, NotificationChannel
from .manager import IntegrationManager
from .enhanced_notifier import EnhancedNotifier, AlertSeverity

__all__ = [
    "TemplatedClient",
    "N8NClient", 
    "N8NEventType",
    "NotificationClient",
    "NotificationType",
    "NotificationChannel", 
    "IntegrationManager",
    "EnhancedNotifier",
    "AlertSeverity"
]
