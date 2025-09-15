"""
Integration tests for WebSocket functionality.
"""

import pytest
import json
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from agent_scrivener.api.main import app
from agent_scrivener.api.auth import create_access_token
from agent_scrivener.api.websocket import ConnectionManager, ProgressTracker
from agent_scrivener.api.models import ResearchStatus
from agent_scrivener.models.core import ResearchSession


class TestWebSocketConnection:
    """Test WebSocket connection and messaging."""
    
    def setup_method(self):
        """Set up test client and authentication."""
        self.client = TestClient(app)
        self.test_token = create_access_token("ws_test_user", "testuser", ["read", "write"])
    
    @patch('agent_scrivener.api.websocket.orchestrator')
    def test_websocket_connection_success(self, mock_orchestrator):
        """Test successful WebSocket connection."""
        # Mock session data
        mock_session = ResearchSession(
            session_id="test-session-123",
            user_id="ws_test_user",
            query="Test query",
            status=ResearchStatus.IN_PROGRESS,
            progress_percentage=25.0,
            current_task="Searching web",
            completed_tasks=["Planning"],
            estimated_time_remaining_minutes=20,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        mock_orchestrator.get_session_status.return_value = mock_session
        
        with self.client.websocket_connect(
            f"/api/v1/ws/research/test-session-123/progress?token={self.test_token}"
        ) as websocket:
            # Should receive initial status
            data = websocket.receive_text()
            message = json.loads(data)
            
            assert message["type"] == "status_update"
            assert message["session_id"] == "test-session-123"
            assert message["status"] == "in_progress"
            assert message["progress_percentage"] == 25.0
            assert message["current_task"] == "Searching web"
            assert len(message["completed_tasks"]) == 1
    
    def test_websocket_connection_invalid_token(self):
        """Test WebSocket connection with invalid token."""
        with pytest.raises(Exception):  # Connection should be rejected
            with self.client.websocket_connect(
                "/api/v1/ws/research/test-session-123/progress?token=invalid_token"
            ):
                pass
    
    @patch('agent_scrivener.api.websocket.orchestrator')
    def test_websocket_connection_session_not_found(self, mock_orchestrator):
        """Test WebSocket connection for non-existent session."""
        mock_orchestrator.get_session_status.return_value = None
        
        with pytest.raises(Exception):  # Connection should be closed
            with self.client.websocket_connect(
                f"/api/v1/ws/research/nonexistent-session/progress?token={self.test_token}"
            ):
                pass
    
    @patch('agent_scrivener.api.websocket.orchestrator')
    def test_websocket_ping_pong(self, mock_orchestrator):
        """Test WebSocket ping/pong functionality."""
        mock_session = ResearchSession(
            session_id="test-session-123",
            user_id="ws_test_user",
            query="Test query",
            status=ResearchStatus.IN_PROGRESS,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        mock_orchestrator.get_session_status.return_value = mock_session
        
        with self.client.websocket_connect(
            f"/api/v1/ws/research/test-session-123/progress?token={self.test_token}"
        ) as websocket:
            # Receive initial status
            websocket.receive_text()
            
            # Send ping
            websocket.send_text(json.dumps({"type": "ping"}))
            
            # Should receive pong
            data = websocket.receive_text()
            message = json.loads(data)
            
            assert message["type"] == "pong"
            assert "timestamp" in message
    
    @patch('agent_scrivener.api.websocket.orchestrator')
    def test_websocket_request_status(self, mock_orchestrator):
        """Test requesting status update via WebSocket."""
        mock_session = ResearchSession(
            session_id="test-session-123",
            user_id="ws_test_user",
            query="Test query",
            status=ResearchStatus.IN_PROGRESS,
            progress_percentage=50.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        mock_orchestrator.get_session_status.return_value = mock_session
        
        with self.client.websocket_connect(
            f"/api/v1/ws/research/test-session-123/progress?token={self.test_token}"
        ) as websocket:
            # Receive initial status
            websocket.receive_text()
            
            # Request status update
            websocket.send_text(json.dumps({"type": "request_status"}))
            
            # Should receive status update
            data = websocket.receive_text()
            message = json.loads(data)
            
            assert message["type"] == "status_update"
            assert message["session_id"] == "test-session-123"
            assert message["progress_percentage"] == 50.0


class TestConnectionManager:
    """Test WebSocket connection manager."""
    
    def setup_method(self):
        """Set up connection manager."""
        self.manager = ConnectionManager()
    
    @pytest.mark.asyncio
    async def test_connection_management(self):
        """Test connection and disconnection."""
        # Mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        
        # Connect
        await self.manager.connect(mock_websocket, "session-123", "user-456")
        
        assert "session-123" in self.manager.session_connections
        assert mock_websocket in self.manager.session_connections["session-123"]
        assert self.manager.connection_users[mock_websocket] == "user-456"
        assert self.manager.connection_sessions[mock_websocket] == "session-123"
        
        # Disconnect
        self.manager.disconnect(mock_websocket)
        
        assert "session-123" not in self.manager.session_connections
        assert mock_websocket not in self.manager.connection_users
        assert mock_websocket not in self.manager.connection_sessions
    
    @pytest.mark.asyncio
    async def test_multiple_connections_same_session(self):
        """Test multiple connections for the same session."""
        mock_ws1 = AsyncMock()
        mock_ws1.accept = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws2.accept = AsyncMock()
        
        # Connect both to same session
        await self.manager.connect(mock_ws1, "session-123", "user-1")
        await self.manager.connect(mock_ws2, "session-123", "user-2")
        
        assert len(self.manager.session_connections["session-123"]) == 2
        assert mock_ws1 in self.manager.session_connections["session-123"]
        assert mock_ws2 in self.manager.session_connections["session-123"]
    
    @pytest.mark.asyncio
    async def test_send_session_update(self):
        """Test sending updates to all session connections."""
        mock_ws1 = AsyncMock()
        mock_ws1.accept = AsyncMock()
        mock_ws1.send_text = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws2.accept = AsyncMock()
        mock_ws2.send_text = AsyncMock()
        
        # Connect both to same session
        await self.manager.connect(mock_ws1, "session-123", "user-1")
        await self.manager.connect(mock_ws2, "session-123", "user-2")
        
        # Send update
        update_data = {"type": "progress_update", "progress": 50}
        await self.manager.send_session_update("session-123", update_data)
        
        # Both should receive the update
        mock_ws1.send_text.assert_called_once()
        mock_ws2.send_text.assert_called_once()
        
        # Check the message content
        sent_message = json.loads(mock_ws1.send_text.call_args[0][0])
        assert sent_message["type"] == "progress_update"
        assert sent_message["progress"] == 50
    
    @pytest.mark.asyncio
    async def test_send_personal_message(self):
        """Test sending message to specific connection."""
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        await self.manager.connect(mock_websocket, "session-123", "user-1")
        
        # Send personal message
        message = {"type": "personal", "data": "test"}
        await self.manager.send_personal_message(mock_websocket, message)
        
        mock_websocket.send_text.assert_called_once()
        sent_message = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_message["type"] == "personal"
        assert sent_message["data"] == "test"
    
    @pytest.mark.asyncio
    async def test_failed_message_cleanup(self):
        """Test cleanup when message sending fails."""
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock(side_effect=Exception("Connection failed"))
        
        await self.manager.connect(mock_websocket, "session-123", "user-1")
        
        # Send message that will fail
        await self.manager.send_personal_message(mock_websocket, {"type": "test"})
        
        # Connection should be cleaned up
        assert mock_websocket not in self.manager.connection_users
        assert mock_websocket not in self.manager.connection_sessions


class TestProgressTracker:
    """Test progress tracking functionality."""
    
    def setup_method(self):
        """Set up progress tracker with mock connection manager."""
        self.mock_manager = AsyncMock()
        self.tracker = ProgressTracker(self.mock_manager)
    
    @pytest.mark.asyncio
    async def test_update_progress(self):
        """Test progress update broadcasting."""
        await self.tracker.update_progress(
            session_id="session-123",
            status=ResearchStatus.IN_PROGRESS,
            progress_percentage=75.0,
            current_task="Analyzing data",
            completed_tasks=["Search", "Extract"],
            estimated_time_remaining_minutes=10
        )
        
        self.mock_manager.send_session_update.assert_called_once()
        call_args = self.mock_manager.send_session_update.call_args
        
        assert call_args[0][0] == "session-123"  # session_id
        update_data = call_args[0][1]  # update_data
        
        assert update_data["type"] == "progress_update"
        assert update_data["status"] == ResearchStatus.IN_PROGRESS
        assert update_data["progress_percentage"] == 75.0
        assert update_data["current_task"] == "Analyzing data"
        assert update_data["completed_tasks"] == ["Search", "Extract"]
        assert update_data["estimated_time_remaining_minutes"] == 10
        assert "timestamp" in update_data
    
    @pytest.mark.asyncio
    async def test_task_started(self):
        """Test task started notification."""
        await self.tracker.task_started("session-123", "Data Analysis")
        
        self.mock_manager.send_session_update.assert_called_once()
        call_args = self.mock_manager.send_session_update.call_args
        update_data = call_args[0][1]
        
        assert update_data["type"] == "task_started"
        assert update_data["task_name"] == "Data Analysis"
        assert "timestamp" in update_data
    
    @pytest.mark.asyncio
    async def test_task_completed(self):
        """Test task completed notification."""
        await self.tracker.task_completed(
            "session-123", 
            "Data Analysis", 
            "Found 5 key insights"
        )
        
        self.mock_manager.send_session_update.assert_called_once()
        call_args = self.mock_manager.send_session_update.call_args
        update_data = call_args[0][1]
        
        assert update_data["type"] == "task_completed"
        assert update_data["task_name"] == "Data Analysis"
        assert update_data["result_summary"] == "Found 5 key insights"
        assert "timestamp" in update_data
    
    @pytest.mark.asyncio
    async def test_session_completed(self):
        """Test session completed notification."""
        final_result = {
            "document_length": 2500,
            "sources_used": 12,
            "completion_time": 45.5
        }
        
        await self.tracker.session_completed("session-123", final_result)
        
        self.mock_manager.send_session_update.assert_called_once()
        call_args = self.mock_manager.send_session_update.call_args
        update_data = call_args[0][1]
        
        assert update_data["type"] == "session_completed"
        assert update_data["final_result"] == final_result
        assert "timestamp" in update_data
    
    @pytest.mark.asyncio
    async def test_session_failed(self):
        """Test session failed notification."""
        await self.tracker.session_failed("session-123", "Network timeout occurred")
        
        self.mock_manager.send_session_update.assert_called_once()
        call_args = self.mock_manager.send_session_update.call_args
        update_data = call_args[0][1]
        
        assert update_data["type"] == "session_failed"
        assert update_data["error_message"] == "Network timeout occurred"
        assert "timestamp" in update_data