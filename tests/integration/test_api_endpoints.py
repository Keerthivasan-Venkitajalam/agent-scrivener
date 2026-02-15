"""
Integration tests for API endpoints.
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from agent_scrivener.api.main import app
from agent_scrivener.api.auth import create_access_token
from agent_scrivener.api.models import ResearchStatus
from agent_scrivener.models.core import ResearchSession


class TestAPIEndpoints:
    """Test API endpoints with mocked dependencies."""
    
    def setup_method(self):
        """Set up test client and authentication."""
        self.client = TestClient(app)
        self.test_token = create_access_token("test_user", "testuser", ["read", "write"])
        self.auth_headers = {"Authorization": f"Bearer {self.test_token}"}
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Agent Scrivener API"
        assert data["version"] == "1.0.0"
        assert "/docs" in data["docs"]
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "uptime_seconds" in data
        assert "timestamp" in data
    
    @patch('agent_scrivener.api.routes.orchestrator')
    def test_start_research_success(self, mock_orchestrator):
        """Test successful research session creation."""
        # Mock orchestrator response
        mock_session = ResearchSession(
            session_id="test-session-123",
            user_id="test_user",
            query="Test research query",
            status=ResearchStatus.PENDING,
            estimated_duration_minutes=30,
            created_at=datetime.now(timezone.utc)
        )
        mock_orchestrator.start_research.return_value = mock_session
        
        request_data = {
            "query": "What are the latest developments in quantum computing?",
            "max_sources": 15,
            "include_academic": True,
            "include_web": True,
            "priority": "high"
        }
        
        response = self.client.post(
            "/api/v1/research",
            json=request_data,
            headers=self.auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert data["status"] == "pending"
        assert data["estimated_duration_minutes"] == 30
        assert data["query"] == "Test research query"
    
    def test_start_research_invalid_query(self):
        """Test research creation with invalid query."""
        request_data = {
            "query": "short",  # Too short
            "max_sources": 10
        }
        
        response = self.client.post(
            "/api/v1/research",
            json=request_data,
            headers=self.auth_headers
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_start_research_unauthorized(self):
        """Test research creation without authentication."""
        request_data = {
            "query": "What are the latest developments in quantum computing?",
            "max_sources": 10
        }
        
        response = self.client.post("/api/v1/research", json=request_data)
        
        assert response.status_code == 403  # Unauthorized
    
    @patch('agent_scrivener.api.routes.orchestrator')
    def test_get_session_status_success(self, mock_orchestrator):
        """Test successful session status retrieval."""
        mock_session = ResearchSession(
            session_id="test-session-123",
            user_id="test_user",
            query="Test query",
            status=ResearchStatus.IN_PROGRESS,
            progress_percentage=45.5,
            current_task="Analyzing content",
            completed_tasks=["Search web", "Extract content"],
            estimated_time_remaining_minutes=15,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        mock_orchestrator.get_session_status.return_value = mock_session
        
        response = self.client.get(
            "/api/v1/research/test-session-123/status",
            headers=self.auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert data["status"] == "in_progress"
        assert data["progress_percentage"] == 45.5
        assert data["current_task"] == "Analyzing content"
        assert len(data["completed_tasks"]) == 2
    
    @patch('agent_scrivener.api.routes.orchestrator')
    def test_get_session_status_not_found(self, mock_orchestrator):
        """Test session status retrieval for non-existent session."""
        mock_orchestrator.get_session_status.return_value = None
        
        response = self.client.get(
            "/api/v1/research/nonexistent-session/status",
            headers=self.auth_headers
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["message"].lower()
    
    @patch('agent_scrivener.api.routes.orchestrator')
    def test_get_research_result_success(self, mock_orchestrator):
        """Test successful research result retrieval."""
        mock_result = ResearchSession(
            session_id="test-session-123",
            user_id="test_user",
            query="Test query",
            status=ResearchStatus.COMPLETED,
            document_content="# Research Report\n\nContent here...",
            sources_count=8,
            word_count=1500,
            completion_time_minutes=25.5,
            created_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc)
        )
        mock_orchestrator.get_research_result.return_value = mock_result
        
        response = self.client.get(
            "/api/v1/research/test-session-123/result",
            headers=self.auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert data["status"] == "completed"
        assert data["document_content"].startswith("# Research Report")
        assert data["sources_count"] == 8
        assert data["word_count"] == 1500
    
    @patch('agent_scrivener.api.routes.orchestrator')
    def test_get_research_result_not_completed(self, mock_orchestrator):
        """Test research result retrieval for incomplete session."""
        mock_result = ResearchSession(
            session_id="test-session-123",
            user_id="test_user",
            query="Test query",
            status=ResearchStatus.IN_PROGRESS,
            created_at=datetime.now(timezone.utc)
        )
        mock_orchestrator.get_research_result.return_value = mock_result
        
        response = self.client.get(
            "/api/v1/research/test-session-123/result",
            headers=self.auth_headers
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "not completed" in data["message"].lower()
    
    @patch('agent_scrivener.api.routes.orchestrator')
    def test_cancel_research_success(self, mock_orchestrator):
        """Test successful research session cancellation."""
        mock_orchestrator.cancel_session.return_value = True
        
        request_data = {
            "reason": "User requested cancellation"
        }
        
        response = self.client.post(
            "/api/v1/research/test-session-123/cancel",
            json=request_data,
            headers=self.auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "cancelled successfully" in data["message"].lower()
    
    @patch('agent_scrivener.api.routes.orchestrator')
    def test_cancel_research_not_found(self, mock_orchestrator):
        """Test cancellation of non-existent session."""
        mock_orchestrator.cancel_session.return_value = False
        
        request_data = {
            "reason": "Test cancellation"
        }
        
        response = self.client.post(
            "/api/v1/research/nonexistent-session/cancel",
            json=request_data,
            headers=self.auth_headers
        )
        
        assert response.status_code == 404
    
    @patch('agent_scrivener.api.routes.orchestrator')
    def test_list_research_sessions(self, mock_orchestrator):
        """Test listing research sessions."""
        mock_sessions = [
            ResearchSession(
                session_id="session-1",
                user_id="test_user",
                query="Query 1",
                status=ResearchStatus.COMPLETED,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            ),
            ResearchSession(
                session_id="session-2",
                user_id="test_user",
                query="Query 2",
                status=ResearchStatus.IN_PROGRESS,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
        ]
        mock_orchestrator.list_user_sessions.return_value = (mock_sessions, 2)
        
        response = self.client.get(
            "/api/v1/research?page=1&page_size=10",
            headers=self.auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        assert len(data["sessions"]) == 2
        assert data["page"] == 1
        assert data["page_size"] == 10
    
    @patch('agent_scrivener.api.routes.orchestrator')
    def test_delete_research_session(self, mock_orchestrator):
        """Test deleting a research session."""
        mock_orchestrator.delete_session.return_value = True
        
        response = self.client.delete(
            "/api/v1/research/test-session-123",
            headers=self.auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "deleted successfully" in data["message"].lower()
    
    def test_404_error_handler(self):
        """Test 404 error handling."""
        response = self.client.get("/api/v1/nonexistent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "NOT_FOUND"
        assert "not found" in data["message"].lower()


class TestRateLimiting:
    """Test API rate limiting."""
    
    def setup_method(self):
        """Set up test client and clear rate limits."""
        self.client = TestClient(app)
        self.test_token = create_access_token("rate_test_user", "testuser", ["read", "write"])
        self.auth_headers = {"Authorization": f"Bearer {self.test_token}"}
        
        # Clear rate limit storage
        from agent_scrivener.api.auth import _rate_limit_storage
        _rate_limit_storage.clear()
    
    def test_rate_limit_enforcement(self):
        """Test that rate limiting is enforced."""
        from agent_scrivener.api.auth import auth_config
        
        # Make requests up to the limit
        for i in range(auth_config.rate_limit_requests_per_minute):
            response = self.client.get("/api/v1/health", headers=self.auth_headers)
            assert response.status_code == 200
        
        # Next request should be rate limited
        response = self.client.get("/api/v1/health", headers=self.auth_headers)
        assert response.status_code == 429
        
        data = response.json()
        assert "rate limit" in data["message"].lower()