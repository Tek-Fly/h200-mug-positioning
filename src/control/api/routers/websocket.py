"""WebSocket endpoints for real-time updates."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPBearer

from src.control.api.config import get_settings
from src.control.api.middleware.auth import JWTHandler

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Connection manager
class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {
            "metrics": set(),
            "logs": set(),
            "alerts": set(),
            "activity": set(),
        }
        self.jwt_handler = JWTHandler(
            secret_key=settings.secret_key.get_secret_value(),
            algorithm=settings.jwt_algorithm,
        )
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept and store connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove connection and subscriptions."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
            # Remove from all subscriptions
            for topic_subs in self.subscriptions.values():
                topic_subs.discard(client_id)
            
            logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict, topic: str):
        """Broadcast message to all subscribed clients."""
        if topic not in self.subscriptions:
            return
        
        # Send to all subscribed clients
        disconnected_clients = []
        
        for client_id in self.subscriptions[topic]:
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {e}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def subscribe(self, client_id: str, topic: str) -> bool:
        """Subscribe client to a topic."""
        if topic not in self.subscriptions:
            return False
        
        self.subscriptions[topic].add(client_id)
        logger.info(f"Client {client_id} subscribed to {topic}")
        return True
    
    def unsubscribe(self, client_id: str, topic: str) -> bool:
        """Unsubscribe client from a topic."""
        if topic not in self.subscriptions:
            return False
        
        self.subscriptions[topic].discard(client_id)
        logger.info(f"Client {client_id} unsubscribed from {topic}")
        return True
    
    async def authenticate_token(self, token: str) -> Optional[dict]:
        """Authenticate WebSocket token."""
        try:
            payload = self.jwt_handler.verify_token(token)
            return payload
        except Exception as e:
            logger.error(f"WebSocket authentication failed: {e}")
            return None


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/control-plane")
async def control_plane_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time control plane updates.
    
    Clients can subscribe to:
    - metrics: Real-time performance metrics
    - logs: Live log streaming
    - alerts: System alerts and notifications
    - activity: User activity updates
    
    Protocol:
    1. Connect with authentication token in query params
    2. Send subscription messages: {"action": "subscribe", "topic": "metrics"}
    3. Receive real-time updates for subscribed topics
    """
    client_id = None
    
    try:
        # Extract token from query params
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Missing token")
            return
        
        # Authenticate
        auth_payload = await manager.authenticate_token(token)
        if not auth_payload:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
            return
        
        # Generate client ID
        client_id = f"user_{auth_payload['user_id']}_{datetime.utcnow().timestamp()}"
        
        # Accept connection
        await manager.connect(websocket, client_id)
        
        # Send welcome message
        await manager.send_personal_message({
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat(),
        }, client_id)
        
        # Handle messages
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Process action
            action = data.get("action")
            
            if action == "subscribe":
                topic = data.get("topic")
                if manager.subscribe(client_id, topic):
                    await manager.send_personal_message({
                        "type": "subscription",
                        "status": "subscribed",
                        "topic": topic,
                        "timestamp": datetime.utcnow().isoformat(),
                    }, client_id)
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Invalid topic: {topic}",
                        "timestamp": datetime.utcnow().isoformat(),
                    }, client_id)
            
            elif action == "unsubscribe":
                topic = data.get("topic")
                if manager.unsubscribe(client_id, topic):
                    await manager.send_personal_message({
                        "type": "subscription",
                        "status": "unsubscribed",
                        "topic": topic,
                        "timestamp": datetime.utcnow().isoformat(),
                    }, client_id)
            
            elif action == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat(),
                }, client_id)
            
            else:
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Unknown action: {action}",
                    "timestamp": datetime.utcnow().isoformat(),
                }, client_id)
    
    except WebSocketDisconnect:
        if client_id:
            manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        if client_id:
            manager.disconnect(client_id)


# Background tasks for broadcasting updates
async def broadcast_metrics():
    """Periodically broadcast metrics."""
    while True:
        try:
            # Simulate metrics data
            metrics = {
                "type": "metrics",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "gpu_utilization": 75.5,
                    "requests_per_second": 25.5,
                    "average_latency_ms": 145,
                    "cache_hit_rate": 0.87,
                },
            }
            
            await manager.broadcast(metrics, "metrics")
            
        except Exception as e:
            logger.error(f"Error broadcasting metrics: {e}")
        
        await asyncio.sleep(5)  # Broadcast every 5 seconds


async def broadcast_logs():
    """Broadcast log entries."""
    while True:
        try:
            # Simulate log entry
            log_entry = {
                "type": "log",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "level": "INFO",
                    "message": "Sample log entry",
                    "context": {
                        "service": "api",
                    },
                },
            }
            
            await manager.broadcast(log_entry, "logs")
            
        except Exception as e:
            logger.error(f"Error broadcasting logs: {e}")
        
        await asyncio.sleep(10)  # Broadcast every 10 seconds


# Start background tasks when module loads
# In production, these would be started properly with the app lifecycle
# asyncio.create_task(broadcast_metrics())
# asyncio.create_task(broadcast_logs())