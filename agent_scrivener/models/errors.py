"""
Error handling models and exceptions for Agent Scrivener.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Categories of errors."""
    NETWORK = "network"
    VALIDATION = "validation"
    PROCESSING = "processing"
    EXTERNAL_API = "external_api"
    AGENT_COMMUNICATION = "agent_communication"
    SYSTEM = "system"


class ErrorDetails(BaseModel):
    """Detailed error information."""
    error_id: str = Field(..., min_length=1)
    category: ErrorCategory
    severity: ErrorSeverity
    message: str = Field(..., min_length=1)
    details: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    retry_count: int = Field(default=0, ge=0)
    recoverable: bool = True


class ErrorResponse(BaseModel):
    """Standardized error response."""
    success: bool = False
    error: ErrorDetails
    partial_results: Optional[Dict[str, Any]] = None
    suggested_actions: List[str] = Field(default_factory=list)


# Custom exceptions
class AgentScrivenerError(Exception):
    """Base exception for Agent Scrivener."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, **kwargs):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = kwargs


class NetworkError(AgentScrivenerError):
    """Network-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, **kwargs)


class ValidationError(AgentScrivenerError):
    """Data validation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.VALIDATION, ErrorSeverity.HIGH, **kwargs)


class ProcessingError(AgentScrivenerError):
    """Content processing errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.PROCESSING, ErrorSeverity.MEDIUM, **kwargs)


class ExternalAPIError(AgentScrivenerError):
    """External API errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.EXTERNAL_API, ErrorSeverity.MEDIUM, **kwargs)


class AgentCommunicationError(AgentScrivenerError):
    """Inter-agent communication errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.AGENT_COMMUNICATION, ErrorSeverity.HIGH, **kwargs)