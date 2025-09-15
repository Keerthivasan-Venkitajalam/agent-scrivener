"""
API routes for Agent Scrivener research platform.
"""

import time
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from .models import (
    ResearchRequest, ResearchResponse, SessionStatus, ResearchResult,
    ErrorResponse, HealthCheck, SessionList, CancelRequest, ResearchStatus
)
from .auth import get_current_user, require_scope, rate_limit_dependency, TokenData
from .orchestrator_adapter import APIOrchestrator
from ..models.core import ResearchSession
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter()

# Global orchestrator instance (in production, use dependency injection)
orchestrator = APIOrchestrator()

# Service start time for uptime calculation
_service_start_time = time.time()


@router.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy",
        version="1.0.0",
        uptime_seconds=time.time() - _service_start_time
    )


@router.post("/research", response_model=ResearchResponse, tags=["Research"])
async def start_research(
    request: ResearchRequest,
    current_user: TokenData = Depends(rate_limit_dependency)
):
    """Start a new research session."""
    try:
        logger.info(f"Starting research session for user {current_user.user_id}, query: {request.query[:100]}...")
        
        # Create research session
        session = await orchestrator.start_research(
            query=request.query,
            user_id=current_user.user_id,
            max_sources=request.max_sources,
            include_academic=request.include_academic,
            include_web=request.include_web,
            priority=request.priority
        )
        
        # Start progress simulation (in background)
        import asyncio
        asyncio.create_task(orchestrator.simulate_progress_updates(session.session_id))
        
        return ResearchResponse(
            session_id=session.session_id,
            status=ResearchStatus.PENDING,
            estimated_duration_minutes=session.estimated_duration_minutes,
            created_at=session.created_at,
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Failed to start research session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start research session")


@router.get("/research/{session_id}/status", response_model=SessionStatus, tags=["Research"])
async def get_session_status(
    session_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    """Get the status of a research session."""
    try:
        session = await orchestrator.get_session_status(session_id, current_user.user_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionStatus(
            session_id=session.session_id,
            status=session.status,
            progress_percentage=session.progress_percentage,
            current_task=session.current_task,
            completed_tasks=session.completed_tasks,
            estimated_time_remaining_minutes=session.estimated_time_remaining_minutes,
            created_at=session.created_at,
            updated_at=session.updated_at,
            error_message=session.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session status")


@router.get("/research/{session_id}/result", response_model=ResearchResult, tags=["Research"])
async def get_research_result(
    session_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    """Get the result of a completed research session."""
    try:
        result = await orchestrator.get_research_result(session_id, current_user.user_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Session not found or not completed")
        
        if result.status != ResearchStatus.COMPLETED:
            raise HTTPException(
                status_code=400, 
                detail=f"Session is not completed. Current status: {result.status}"
            )
        
        return ResearchResult(
            session_id=result.session_id,
            status=result.status,
            document_content=result.document_content,
            sources_count=result.sources_count,
            word_count=result.word_count,
            completion_time_minutes=result.completion_time_minutes,
            created_at=result.created_at,
            completed_at=result.completed_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get research result: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve research result")


@router.post("/research/{session_id}/cancel", tags=["Research"])
async def cancel_research(
    session_id: str,
    request: CancelRequest,
    current_user: TokenData = Depends(require_scope("write"))
):
    """Cancel a running research session."""
    try:
        success = await orchestrator.cancel_session(
            session_id, 
            current_user.user_id, 
            reason=request.reason
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found or cannot be cancelled")
        
        return {"message": "Session cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cancel session")


@router.get("/research", response_model=SessionList, tags=["Research"])
async def list_research_sessions(
    current_user: TokenData = Depends(get_current_user),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    status: Optional[ResearchStatus] = Query(default=None, description="Filter by status")
):
    """List research sessions for the current user."""
    try:
        sessions, total_count = await orchestrator.list_user_sessions(
            user_id=current_user.user_id,
            page=page,
            page_size=page_size,
            status_filter=status
        )
        
        session_statuses = []
        for session in sessions:
            session_statuses.append(SessionStatus(
                session_id=session.session_id,
                status=session.status,
                progress_percentage=session.progress_percentage,
                current_task=session.current_task,
                completed_tasks=session.completed_tasks,
                estimated_time_remaining_minutes=session.estimated_time_remaining_minutes,
                created_at=session.created_at,
                updated_at=session.updated_at,
                error_message=session.error_message
            ))
        
        return SessionList(
            sessions=session_statuses,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")


@router.delete("/research/{session_id}", tags=["Research"])
async def delete_research_session(
    session_id: str,
    current_user: TokenData = Depends(require_scope("write"))
):
    """Delete a research session and its data."""
    try:
        success = await orchestrator.delete_session(session_id, current_user.user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": "Session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


# Error handlers are defined in main.py