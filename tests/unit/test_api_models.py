"""
Unit tests for API models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from agent_scrivener.api.models import (
    ResearchRequest, ResearchResponse, SessionStatus, ResearchResult,
    ErrorResponse, HealthCheck, SessionList, CancelRequest, ResearchStatus
)


class TestResearchRequest:
    """Test ResearchRequest model validation."""
    
    def test_valid_request(self):
        """Test valid research request."""
        request = ResearchRequest(
            query="What are the latest developments in quantum computing?",
            max_sources=15,
            include_academic=True,
            include_web=True,
            priority="high"
        )
        
        assert request.query == "What are the latest developments in quantum computing?"
        assert request.max_sources == 15
        assert request.include_academic is True
        assert request.include_web is True
        assert request.priority == "high"
    
    def test_minimal_request(self):
        """Test request with minimal required fields."""
        request = ResearchRequest(query="Test query for research")
        
        assert request.query == "Test query for research"
        assert request.max_sources == 10  # default
        assert request.include_academic is True  # default
        assert request.include_web is True  # default
        assert request.priority == "normal"  # default
    
    def test_query_too_short(self):
        """Test validation error for query too short."""
        with pytest.raises(ValidationError) as exc_info:
            ResearchRequest(query="short")
        
        assert "at least 10 characters" in str(exc_info.value)
    
    def test_query_too_long(self):
        """Test validation error for query too long."""
        long_query = "x" * 2001
        with pytest.raises(ValidationError) as exc_info:
            ResearchRequest(query=long_query)
        
        assert "at most 2000 characters" in str(exc_info.value)
    
    def test_invalid_max_sources(self):
        """Test validation error for invalid max_sources."""
        with pytest.raises(ValidationError):
            ResearchRequest(query="Valid query", max_sources=0)
        
        with pytest.raises(ValidationError):
            ResearchRequest(query="Valid query", max_sources=51)
    
    def test_invalid_priority(self):
        """Test validation error for invalid priority."""
        with pytest.raises(ValidationError):
            ResearchRequest(query="Valid query", priority="invalid")


class TestResearchResponse:
    """Test ResearchResponse model."""
    
    def test_valid_response(self):
        """Test valid research response."""
        now = datetime.utcnow()
        response = ResearchResponse(
            session_id="test-session-123",
            status=ResearchStatus.PENDING,
            estimated_duration_minutes=30,
            created_at=now,
            query="Test query"
        )
        
        assert response.session_id == "test-session-123"
        assert response.status == ResearchStatus.PENDING
        assert response.estimated_duration_minutes == 30
        assert response.created_at == now
        assert response.query == "Test query"


class TestSessionStatus:
    """Test SessionStatus model."""
    
    def test_valid_status(self):
        """Test valid session status."""
        now = datetime.utcnow()
        status = SessionStatus(
            session_id="test-session-123",
            status=ResearchStatus.IN_PROGRESS,
            progress_percentage=45.5,
            current_task="Analyzing content",
            completed_tasks=["Search web", "Extract content"],
            estimated_time_remaining_minutes=15,
            created_at=now,
            updated_at=now
        )
        
        assert status.session_id == "test-session-123"
        assert status.status == ResearchStatus.IN_PROGRESS
        assert status.progress_percentage == 45.5
        assert status.current_task == "Analyzing content"
        assert len(status.completed_tasks) == 2
        assert status.estimated_time_remaining_minutes == 15
    
    def test_invalid_progress_percentage(self):
        """Test validation error for invalid progress percentage."""
        now = datetime.utcnow()
        
        with pytest.raises(ValidationError):
            SessionStatus(
                session_id="test",
                status=ResearchStatus.IN_PROGRESS,
                progress_percentage=-1,
                created_at=now,
                updated_at=now
            )
        
        with pytest.raises(ValidationError):
            SessionStatus(
                session_id="test",
                status=ResearchStatus.IN_PROGRESS,
                progress_percentage=101,
                created_at=now,
                updated_at=now
            )


class TestResearchResult:
    """Test ResearchResult model."""
    
    def test_valid_result(self):
        """Test valid research result."""
        now = datetime.utcnow()
        result = ResearchResult(
            session_id="test-session-123",
            status=ResearchStatus.COMPLETED,
            document_content="# Research Report\n\nContent here...",
            sources_count=8,
            word_count=1500,
            completion_time_minutes=25.5,
            created_at=now,
            completed_at=now
        )
        
        assert result.session_id == "test-session-123"
        assert result.status == ResearchStatus.COMPLETED
        assert result.document_content.startswith("# Research Report")
        assert result.sources_count == 8
        assert result.word_count == 1500
        assert result.completion_time_minutes == 25.5


class TestErrorResponse:
    """Test ErrorResponse model."""
    
    def test_error_response(self):
        """Test error response creation."""
        error = ErrorResponse(
            error="VALIDATION_ERROR",
            message="Invalid input provided",
            details={"field": "query", "issue": "too short"}
        )
        
        assert error.error == "VALIDATION_ERROR"
        assert error.message == "Invalid input provided"
        assert error.details["field"] == "query"
        assert isinstance(error.timestamp, datetime)


class TestHealthCheck:
    """Test HealthCheck model."""
    
    def test_health_check(self):
        """Test health check response."""
        health = HealthCheck(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.5
        )
        
        assert health.status == "healthy"
        assert health.version == "1.0.0"
        assert health.uptime_seconds == 3600.5
        assert isinstance(health.timestamp, datetime)


class TestCancelRequest:
    """Test CancelRequest model."""
    
    def test_cancel_request(self):
        """Test cancel request validation."""
        request = CancelRequest(reason="User requested cancellation")
        assert request.reason == "User requested cancellation"
    
    def test_cancel_request_no_reason(self):
        """Test cancel request without reason."""
        request = CancelRequest()
        assert request.reason is None
    
    def test_cancel_request_reason_too_long(self):
        """Test validation error for reason too long."""
        long_reason = "x" * 501
        with pytest.raises(ValidationError):
            CancelRequest(reason=long_reason)