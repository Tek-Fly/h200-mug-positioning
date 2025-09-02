"""External integrations for H200 Intelligent Mug Positioning System."""

# Local imports
from .enhanced_notifier import AlertSeverity, EnhancedNotifier
from .manager import IntegrationManager
from .n8n import N8NClient, N8NEventType
from .notifications import NotificationChannel, NotificationClient, NotificationType
from .templated import TemplatedClient

__all__ = [
    "TemplatedClient",
    "N8NClient",
    "N8NEventType",
    "NotificationClient",
    "NotificationType",
    "NotificationChannel",
    "IntegrationManager",
    "EnhancedNotifier",
    "AlertSeverity",
]
