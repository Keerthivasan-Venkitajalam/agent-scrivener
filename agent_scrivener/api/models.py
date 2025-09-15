"""
API request and response models for Agent Scrivener.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ResearchStatus(str, Enum):
    """Status of a research session."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchRequest(BaseModel):
    """Request model for starting a research session."""
    query: str = Field(..., min_length=10, max_length=2000, description="Research query")
    max_sources: Optional[int] = Field(default=10, ge=1, le=50, description="Maximum number of sources to gather")
    include_academic: Optional[bool] = Field(default=True, description="Include academic database sources")
    include_web: Optional[bool] = Field(default=True, description="Include web sources")
    priority: Optional[str] = Field(default="normal", pattern="^(low|normal|high)$", description="Task priority")


class ResearchResponse(BaseModel):
    """Response model for research session creation."""
    session_id: str = Field(..., description="Unique session identifier")
    status: ResearchStatus = Field(..., description="Current session status")
    estimated_duration_minutes: int = Field(..., description="Estimated completion time in minutes")
    created_at: datetime = Field(..., description="Session creation timestamp")
    query: str = Field(..., description="Original research query")


class SessionStatus(BaseModel):
    """Model for session status information."""
    session_id: str = Field(..., description="Session identifier")
    status: ResearchStatus = Field(..., description="Current status")
    progress_percentage: float = Field(..., ge=0, le=100, description="Completion percentage")
    current_task: Optional[str] = Field(None, description="Currently executing task")
    completed_tasks: List[str] = Field(default_factory=list, description="List of completed tasks")
    estimated_time_remaining_minutes: Optional[int] = Field(None, description="Estimated time remaining")
    created_at: datetime = Field(..., description="Session creation time")
    updated_at: datetime = Field(..., description="Last update time")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class ResearchResult(BaseModel):
    """Model for completed research results."""
    session_id: str = Field(..., description="Session identifier")
    status: ResearchStatus = Field(..., description="Final status")
    document_content: str = Field(..., description="Generated research document in Markdown format")
    sources_count: int = Field(..., description="Number of sources used")
    word_count: int = Field(..., description="Word count of generated document")
    completion_time_minutes: float = Field(..., description="Actual completion time")
    created_at: datetime = Field(..., description="Session creation time")
    completed_at: datetime = Field(..., description="Session completion time")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class SessionList(BaseModel):
    """Model for listing research sessions."""
    sessions: List[SessionStatus] = Field(..., description="List of research sessions")
    total_count: int = Field(..., description="Total number of sessions")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")


class CancelRequest(BaseModel):
    """Request model for cancelling a research session."""
    reason: Optional[str] = Field(None, max_length=500, description="Reason for cancellation")