"""Auto-shutdown scheduler for idle GPU instances."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import defaultdict

from src.control.api.models.servers import ServerInfo, ServerState, ServerType
from src.control.manager.server_manager import ServerManager

logger = logging.getLogger(__name__)


class IdleTracker:
    """Tracks server idle time."""
    
    def __init__(self):
        """Initialize idle tracker."""
        self.last_activity: Dict[str, datetime] = {}
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.idle_warnings: Dict[str, datetime] = {}
    
    def record_activity(self, server_id: str):
        """Record activity for a server."""
        self.last_activity[server_id] = datetime.utcnow()
        self.request_counts[server_id] += 1
        
        # Clear idle warning if any
        if server_id in self.idle_warnings:
            del self.idle_warnings[server_id]
    
    def get_idle_time(self, server_id: str) -> Optional[timedelta]:
        """Get idle time for a server."""
        if server_id not in self.last_activity:
            return None
        
        return datetime.utcnow() - self.last_activity[server_id]
    
    def is_idle(self, server_id: str, threshold_seconds: int) -> bool:
        """Check if server is idle."""
        idle_time = self.get_idle_time(server_id)
        if idle_time is None:
            return False
        
        return idle_time.total_seconds() > threshold_seconds
    
    def set_warning(self, server_id: str):
        """Set idle warning for a server."""
        self.idle_warnings[server_id] = datetime.utcnow()
    
    def has_warning(self, server_id: str) -> bool:
        """Check if server has idle warning."""
        return server_id in self.idle_warnings
    
    def clear_server(self, server_id: str):
        """Clear all tracking for a server."""
        self.last_activity.pop(server_id, None)
        self.request_counts.pop(server_id, None)
        self.idle_warnings.pop(server_id, None)


class AutoShutdownScheduler:
    """Manages automatic shutdown of idle servers."""
    
    def __init__(
        self,
        server_manager: ServerManager,
        idle_timeout_seconds: int = 600,  # 10 minutes default
        warning_before_seconds: int = 120,  # 2 minutes warning
        check_interval_seconds: int = 30,
    ):
        """Initialize auto-shutdown scheduler."""
        self.server_manager = server_manager
        self.idle_timeout_seconds = idle_timeout_seconds
        self.warning_before_seconds = warning_before_seconds
        self.check_interval_seconds = check_interval_seconds
        
        self.idle_tracker = IdleTracker()
        self.is_running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._protected_servers: Set[str] = set()
        self._shutdown_callbacks = []
        
        # Cost savings tracking
        self.total_savings = 0.0
        self.shutdowns_performed = 0
    
    async def start(self):
        """Start the auto-shutdown scheduler."""
        if self.is_running:
            return
        
        logger.info(
            f"Starting auto-shutdown scheduler "
            f"(idle_timeout={self.idle_timeout_seconds}s, "
            f"check_interval={self.check_interval_seconds}s)"
        )
        
        self.is_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def stop(self):
        """Stop the auto-shutdown scheduler."""
        if not self.is_running:
            return
        
        logger.info("Stopping auto-shutdown scheduler")
        self.is_running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
    
    def protect_server(self, server_id: str):
        """Protect a server from auto-shutdown."""
        self._protected_servers.add(server_id)
        logger.info(f"Server {server_id} protected from auto-shutdown")
    
    def unprotect_server(self, server_id: str):
        """Remove auto-shutdown protection."""
        self._protected_servers.discard(server_id)
        logger.info(f"Server {server_id} auto-shutdown protection removed")
    
    def record_activity(self, server_id: str):
        """Record activity for a server."""
        self.idle_tracker.record_activity(server_id)
    
    def add_shutdown_callback(self, callback):
        """Add callback for shutdown events."""
        self._shutdown_callbacks.append(callback)
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                await self._check_idle_servers()
                await asyncio.sleep(self.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(self.check_interval_seconds)
    
    async def _check_idle_servers(self):
        """Check all servers for idle timeout."""
        servers = await self.server_manager.list_servers()
        
        for server in servers:
            # Skip if not running
            if server.state != ServerState.RUNNING:
                continue
            
            # Skip protected servers
            if server.id in self._protected_servers:
                continue
            
            # Skip serverless with min_instances > 0
            if (
                server.type == ServerType.SERVERLESS
                and server.config.min_instances > 0
            ):
                continue
            
            # Check idle time
            idle_time = self.idle_tracker.get_idle_time(server.id)
            if idle_time is None:
                # No activity recorded yet, start tracking
                self.idle_tracker.record_activity(server.id)
                continue
            
            idle_seconds = idle_time.total_seconds()
            
            # Check for shutdown
            if idle_seconds > self.idle_timeout_seconds:
                await self._shutdown_idle_server(server)
            
            # Check for warning
            elif idle_seconds > (self.idle_timeout_seconds - self.warning_before_seconds):
                if not self.idle_tracker.has_warning(server.id):
                    await self._send_idle_warning(server, idle_seconds)
    
    async def _shutdown_idle_server(self, server: ServerInfo):
        """Shutdown an idle server."""
        logger.info(
            f"Auto-shutting down idle server {server.id} "
            f"(idle for {self.idle_tracker.get_idle_time(server.id)})"
        )
        
        try:
            # Calculate cost savings
            if server.started_at:
                runtime = datetime.utcnow() - server.started_at
                hours_saved = (self.idle_timeout_seconds / 3600)
                cost_saved = hours_saved * server.cost_per_hour
                self.total_savings += cost_saved
            
            # Shutdown server
            await self.server_manager.stop_server(server.id)
            
            # Update tracking
            self.shutdowns_performed += 1
            self.idle_tracker.clear_server(server.id)
            
            # Notify callbacks
            for callback in self._shutdown_callbacks:
                try:
                    await callback({
                        "event": "auto_shutdown",
                        "server_id": server.id,
                        "server_type": server.type.value,
                        "idle_time_seconds": self.idle_timeout_seconds,
                        "cost_saved": cost_saved if "cost_saved" in locals() else 0,
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                except Exception as e:
                    logger.error(f"Shutdown callback error: {e}")
            
            logger.info(
                f"Server {server.id} auto-shutdown complete "
                f"(total savings: ${self.total_savings:.2f})"
            )
            
        except Exception as e:
            logger.error(f"Failed to auto-shutdown server {server.id}: {e}")
    
    async def _send_idle_warning(self, server: ServerInfo, idle_seconds: float):
        """Send idle warning for a server."""
        remaining_seconds = self.idle_timeout_seconds - idle_seconds
        
        logger.warning(
            f"Server {server.id} idle warning: "
            f"will shutdown in {remaining_seconds:.0f} seconds"
        )
        
        self.idle_tracker.set_warning(server.id)
        
        # Notify callbacks
        for callback in self._shutdown_callbacks:
            try:
                await callback({
                    "event": "idle_warning",
                    "server_id": server.id,
                    "server_type": server.type.value,
                    "idle_seconds": idle_seconds,
                    "remaining_seconds": remaining_seconds,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            except Exception as e:
                logger.error(f"Warning callback error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get auto-shutdown statistics."""
        return {
            "is_running": self.is_running,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "warning_before_seconds": self.warning_before_seconds,
            "protected_servers": list(self._protected_servers),
            "total_shutdowns": self.shutdowns_performed,
            "total_savings": self.total_savings,
            "servers_tracking": len(self.idle_tracker.last_activity),
            "servers_with_warnings": len(self.idle_tracker.idle_warnings),
        }
    
    def reset_statistics(self):
        """Reset statistics."""
        self.total_savings = 0.0
        self.shutdowns_performed = 0
        logger.info("Auto-shutdown statistics reset")