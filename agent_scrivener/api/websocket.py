"""
WebSocket endpoints for real-time progress updates.
"""

import json
import asyncio
from typing import Dict, Set
from datetime import datetime, timezone
from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.routing import APIRouter

from .auth import verify_token, TokenData
from .models import SessionStatus, ResearchStatus
from .orchestrator_adapter import APIOrchestrator
from ..utils.logging import get_logger

logger = get_logger(__name__)

# WebSocket router
ws_router = APIRouter()

# Connection manager for WebSocket connections
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        # Map of session_id -> set of websockets
        self.session_connections: Dict[str, Set[WebSocket]] = {}
        # Map of websocket -> user_id for authentication
        self.connection_users: Dict[WebSocket, str] = {}
        # Map of websocket -> session_id for cleanup
        self.connection_sessions: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str, user_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        if session_id not in self.session_connections:
            self.session_connections[session_id] = set()
        
        self.session_connections[session_id].add(websocket)
        self.connection_users[websocket] = user_id
        self.connection_sessions[websocket] = session_id
        
        logger.info(f"WebSocket connected for session {session_id}, user {user_id}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.connection_sessions:
            session_id = self.connection_sessions[websocket]
            
            # Remove from session connections
            if session_id in self.session_connections:
                self.session_connections[session_id].discard(websocket)
                if not self.session_connections[session_id]:
                    del self.session_connections[session_id]
            
            # Clean up mappings
            del self.connection_sessions[websocket]
            
        if websocket in self.connection_users:
            del self.connection_users[websocket]
        
        logger.info("WebSocket disconnected")
    
    async def send_session_update(self, session_id: str, update_data: dict):
        """Send update to all connections for a session."""
        if session_id not in self.session_connections:
            return
        
        # Create list to avoid modification during iteration
        connections = list(self.session_connections[session_id])
        
        for websocket in connections:
            try:
                await websocket.send_text(json.dumps(update_data))
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {str(e)}")
                # Remove failed connection
                self.disconnect(websocket)
    
    async def send_personal_message(self, websocket: WebSocket, message: dict):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send personal WebSocket message: {str(e)}")
            self.disconnect(websocket)


# Global connection manager
manager = ConnectionManager()
orchestrator = APIOrchestrator()


async def authenticate_websocket(websocket: WebSocket, token: str) -> TokenData:
    """Authenticate WebSocket connection using JWT token."""
    try:
        return verify_token(token)
    except HTTPException:
        await websocket.close(code=4001, reason="Authentication failed")
        raise


@ws_router.websocket("/research/{session_id}/progress")
async def websocket_progress_endpoint(websocket: WebSocket, session_id: str, token: str):
    """WebSocket endpoint for real-time progress updates."""
    try:
        # Authenticate the connection
        user_data = await authenticate_websocket(websocket, token)
        
        # Verify user has access to this session
        session = await orchestrator.get_session_status(session_id, user_data.user_id)
        if not session:
            await websocket.close(code=4004, reason="Session not found")
            return
        
        # Accept the connection
        await manager.connect(websocket, session_id, user_data.user_id)
        
        # Send initial status
        initial_status = {
            "type": "status_update",
            "session_id": session_id,
            "status": session.status,
            "progress_percentage": session.progress_percentage,
            "current_task": session.current_task,
            "completed_tasks": session.completed_tasks,
            "estimated_time_remaining_minutes": session.estimated_time_remaining_minutes,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await manager.send_personal_message(websocket, initial_status)
        
        # Keep connection alive and handle incoming messages
        try:
            while True:
                # Wait for messages from client (ping/pong, etc.)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await manager.send_personal_message(websocket, {
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                elif message.get("type") == "request_status":
                    # Send current status
                    current_session = await orchestrator.get_session_status(session_id, user_data.user_id)
                    if current_session:
                        status_update = {
                            "type": "status_update",
                            "session_id": session_id,
                            "status": current_session.status,
                            "progress_percentage": current_session.progress_percentage,
                            "current_task": current_session.current_task,
                            "completed_tasks": current_session.completed_tasks,
                            "estimated_time_remaining_minutes": current_session.estimated_time_remaining_minutes,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        await manager.send_personal_message(websocket, status_update)
                
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
        finally:
            manager.disconnect(websocket)
            
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
        try:
            await websocket.close(code=4000, reason="Connection error")
        except:
            pass


class ProgressTracker:
    """Tracks and broadcasts progress updates for research sessions."""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
    
    async def update_progress(
        self,
        session_id: str,
        status: ResearchStatus,
        progress_percentage: float,
        current_task: str = None,
        completed_tasks: list = None,
        estimated_time_remaining_minutes: int = None,
        error_message: str = None
    ):
        """Send progress update to all connected clients for a session."""
        update_data = {
            "type": "progress_update",
            "session_id": session_id,
            "status": status,
            "progress_percentage": progress_percentage,
            "current_task": current_task,
            "completed_tasks": completed_tasks or [],
            "estimated_time_remaining_minutes": estimated_time_remaining_minutes,
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self.manager.send_session_update(session_id, update_data)
        logger.info(f"Progress update sent for session {session_id}: {progress_percentage}%")
    
    async def task_started(self, session_id: str, task_name: str):
        """Notify that a new task has started."""
        update_data = {
            "type": "task_started",
            "session_id": session_id,
            "task_name": task_name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self.manager.send_session_update(session_id, update_data)
    
    async def task_completed(self, session_id: str, task_name: str, result_summary: str = None):
        """Notify that a task has completed."""
        update_data = {
            "type": "task_completed",
            "session_id": session_id,
            "task_name": task_name,
            "result_summary": result_summary,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self.manager.send_session_update(session_id, update_data)
    
    async def session_completed(self, session_id: str, final_result: dict):
        """Notify that the entire research session has completed."""
        update_data = {
            "type": "session_completed",
            "session_id": session_id,
            "final_result": final_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self.manager.send_session_update(session_id, update_data)
    
    async def session_failed(self, session_id: str, error_message: str):
        """Notify that the research session has failed."""
        update_data = {
            "type": "session_failed",
            "session_id": session_id,
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self.manager.send_session_update(session_id, update_data)


# Global progress tracker
progress_tracker = ProgressTracker(manager)

# Integrate progress tracker with orchestrator
orchestrator.set_progress_tracker(progress_tracker)