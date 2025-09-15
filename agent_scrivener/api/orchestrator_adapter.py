"""
API adapter for the AgentOrchestrator to provide the interface expected by API routes.
"""

import uuid
from typing import List, Optional, Tuple
from datetime import datetime

from ..orchestration.orchestrator import AgentOrchestrator
from ..agents.planner_agent import PlannerAgent
from ..models.core import ResearchSession, ResearchPlan
from ..api.models import ResearchStatus
from ..utils.logging import get_logger

logger = get_logger(__name__)


class APIOrchestrator:
    """
    Adapter class that provides the API interface for the AgentOrchestrator.
    
    This class bridges the gap between the API expectations and the actual
    orchestrator implementation.
    """
    
    def __init__(self):
        self.orchestrator = AgentOrchestrator()
        self.planner_agent = PlannerAgent()
        # In-memory storage for demo (in production, use persistent storage)
        self._sessions: dict[str, ResearchSession] = {}
        self._user_sessions: dict[str, list[str]] = {}  # user_id -> session_ids
        # Progress tracker will be injected by WebSocket module
        self._progress_tracker = None
    
    async def start_research(
        self,
        query: str,
        user_id: str,
        max_sources: int = 10,
        include_academic: bool = True,
        include_web: bool = True,
        priority: str = "normal"
    ) -> ResearchSession:
        """
        Start a new research session.
        
        Args:
            query: Research query
            user_id: User identifier
            max_sources: Maximum number of sources to gather
            include_academic: Whether to include academic sources
            include_web: Whether to include web sources
            priority: Task priority
            
        Returns:
            ResearchSession: Created research session
        """
        try:
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Create research plan using planner agent
            plan = await self.planner_agent.create_research_plan(
                query=query,
                session_id=session_id,
                max_sources=max_sources,
                include_academic=include_academic,
                include_web=include_web
            )
            
            # Create session
            session = ResearchSession(
                session_id=session_id,
                user_id=user_id,
                query=query,
                status=ResearchStatus.PENDING,
                estimated_duration_minutes=plan.estimated_duration_minutes,
                created_at=datetime.utcnow(),
                plan=plan
            )
            
            # Store session
            self._sessions[session_id] = session
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = []
            self._user_sessions[user_id].append(session_id)
            
            # Start orchestrator execution (async)
            await self.orchestrator.start_research_session(plan)
            
            # Notify progress tracker if available
            if self._progress_tracker:
                await self._progress_tracker.update_progress(
                    session_id=session_id,
                    status=ResearchStatus.PENDING,
                    progress_percentage=0.0,
                    current_task="Initializing research session"
                )
            
            logger.info(f"Started research session {session_id} for user {user_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to start research session: {str(e)}")
            raise
    
    async def get_session_status(self, session_id: str, user_id: str) -> Optional[ResearchSession]:
        """
        Get the status of a research session.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            ResearchSession or None if not found
        """
        try:
            session = self._sessions.get(session_id)
            if not session or session.user_id != user_id:
                return None
            
            # Update session with current orchestrator state
            orchestrator_session = self.orchestrator._active_sessions.get(session_id)
            if orchestrator_session:
                session.status = self._map_task_status_to_research_status(orchestrator_session.status)
                session.progress_percentage = self._calculate_progress_percentage(orchestrator_session)
                session.current_task = self._get_current_task(orchestrator_session)
                session.completed_tasks = self._get_completed_tasks(orchestrator_session)
                session.updated_at = datetime.utcnow()
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to get session status: {str(e)}")
            return None
    
    async def get_research_result(self, session_id: str, user_id: str) -> Optional[ResearchSession]:
        """
        Get the result of a completed research session.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            ResearchSession with results or None if not found/completed
        """
        try:
            session = await self.get_session_status(session_id, user_id)
            if not session:
                return None
            
            # Only return results for completed sessions
            if session.status != ResearchStatus.COMPLETED:
                return session  # Return session with current status
            
            # Get aggregated results from orchestrator
            orchestrator_session = self.orchestrator._active_sessions.get(session_id)
            if orchestrator_session and orchestrator_session.final_document:
                session.document_content = orchestrator_session.final_document
                session.sources_count = len(orchestrator_session.get_all_sources())
                session.word_count = len(orchestrator_session.final_document.split())
                session.completion_time_minutes = (
                    (session.updated_at - session.created_at).total_seconds() / 60
                )
                session.completed_at = session.updated_at
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to get research result: {str(e)}")
            return None
    
    async def cancel_session(self, session_id: str, user_id: str, reason: str = None) -> bool:
        """
        Cancel a running research session.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            reason: Cancellation reason
            
        Returns:
            bool: True if cancelled successfully
        """
        try:
            session = self._sessions.get(session_id)
            if not session or session.user_id != user_id:
                return False
            
            # Update session status
            session.status = ResearchStatus.CANCELLED
            session.updated_at = datetime.utcnow()
            session.error_message = reason or "Cancelled by user"
            
            # Cancel orchestrator session if active
            orchestrator_session = self.orchestrator._active_sessions.get(session_id)
            if orchestrator_session:
                # Cancel active tasks (simplified implementation)
                for execution in orchestrator_session.agent_executions:
                    if execution.status.value == "in_progress":
                        self.orchestrator.task_dispatcher.cancel_task(execution.execution_id)
            
            logger.info(f"Cancelled session {session_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel session: {str(e)}")
            return False
    
    async def list_user_sessions(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
        status_filter: Optional[ResearchStatus] = None
    ) -> Tuple[List[ResearchSession], int]:
        """
        List research sessions for a user.
        
        Args:
            user_id: User identifier
            page: Page number (1-based)
            page_size: Number of items per page
            status_filter: Optional status filter
            
        Returns:
            Tuple of (sessions, total_count)
        """
        try:
            user_session_ids = self._user_sessions.get(user_id, [])
            user_sessions = [self._sessions[sid] for sid in user_session_ids if sid in self._sessions]
            
            # Apply status filter
            if status_filter:
                user_sessions = [s for s in user_sessions if s.status == status_filter]
            
            # Sort by creation date (newest first)
            user_sessions.sort(key=lambda s: s.created_at, reverse=True)
            
            # Apply pagination
            total_count = len(user_sessions)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_sessions = user_sessions[start_idx:end_idx]
            
            # Update session statuses
            for session in paginated_sessions:
                await self.get_session_status(session.session_id, user_id)
            
            return paginated_sessions, total_count
            
        except Exception as e:
            logger.error(f"Failed to list user sessions: {str(e)}")
            return [], 0
    
    async def delete_session(self, session_id: str, user_id: str) -> bool:
        """
        Delete a research session and its data.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            session = self._sessions.get(session_id)
            if not session or session.user_id != user_id:
                return False
            
            # Cancel if still running
            if session.status in [ResearchStatus.PENDING, ResearchStatus.IN_PROGRESS]:
                await self.cancel_session(session_id, user_id, "Session deleted")
            
            # Remove from storage
            del self._sessions[session_id]
            if user_id in self._user_sessions:
                self._user_sessions[user_id] = [
                    sid for sid in self._user_sessions[user_id] if sid != session_id
                ]
            
            logger.info(f"Deleted session {session_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session: {str(e)}")
            return False
    
    def _map_task_status_to_research_status(self, task_status) -> ResearchStatus:
        """Map TaskStatus to ResearchStatus."""
        mapping = {
            "pending": ResearchStatus.PENDING,
            "in_progress": ResearchStatus.IN_PROGRESS,
            "completed": ResearchStatus.COMPLETED,
            "failed": ResearchStatus.FAILED,
            "cancelled": ResearchStatus.CANCELLED
        }
        return mapping.get(task_status.value if hasattr(task_status, 'value') else str(task_status), ResearchStatus.PENDING)
    
    def _calculate_progress_percentage(self, session) -> float:
        """Calculate progress percentage for a session."""
        if not session.plan or not session.plan.tasks:
            return 0.0
        
        total_tasks = len(session.plan.tasks)
        completed_tasks = len([t for t in session.plan.tasks if t.status.value == "completed"])
        
        return (completed_tasks / total_tasks) * 100.0
    
    def _get_current_task(self, session) -> Optional[str]:
        """Get the currently executing task."""
        for task in session.plan.tasks:
            if task.status.value == "in_progress":
                return task.task_type
        return None
    
    def _get_completed_tasks(self, session) -> List[str]:
        """Get list of completed task types."""
        return [task.task_type for task in session.plan.tasks if task.status.value == "completed"]
    
    def set_progress_tracker(self, progress_tracker):
        """Set the progress tracker for real-time updates."""
        self._progress_tracker = progress_tracker
    
    async def simulate_progress_updates(self, session_id: str):
        """
        Simulate progress updates for demonstration purposes.
        In production, this would be integrated with the actual orchestrator.
        """
        if not self._progress_tracker:
            return
        
        import asyncio
        
        # Simulate research workflow progress
        progress_steps = [
            (10, "Planning research approach"),
            (25, "Searching web sources"),
            (40, "Extracting content from articles"),
            (55, "Querying academic databases"),
            (70, "Analyzing collected data"),
            (85, "Generating insights"),
            (95, "Drafting final document"),
            (100, "Finalizing citations")
        ]
        
        for progress, task in progress_steps:
            await asyncio.sleep(2)  # Simulate work
            
            session = self._sessions.get(session_id)
            if not session:
                break
            
            # Update session
            session.progress_percentage = progress
            session.current_task = task
            session.updated_at = datetime.utcnow()
            
            if progress < 100:
                session.status = ResearchStatus.IN_PROGRESS
                await self._progress_tracker.update_progress(
                    session_id=session_id,
                    status=ResearchStatus.IN_PROGRESS,
                    progress_percentage=progress,
                    current_task=task,
                    estimated_time_remaining_minutes=max(1, int((100 - progress) / 10))
                )
                
                if progress in [25, 55, 85]:  # Task completion milestones
                    await self._progress_tracker.task_completed(
                        session_id=session_id,
                        task_name=task,
                        result_summary=f"Completed {task.lower()}"
                    )
            else:
                # Session completed
                session.status = ResearchStatus.COMPLETED
                session.completed_at = datetime.utcnow()
                session.document_content = self._generate_sample_document(session.query)
                session.sources_count = 8
                session.word_count = len(session.document_content.split())
                session.completion_time_minutes = (
                    (session.updated_at - session.created_at).total_seconds() / 60
                )
                
                await self._progress_tracker.session_completed(
                    session_id=session_id,
                    final_result={
                        "document_length": session.word_count,
                        "sources_used": session.sources_count,
                        "completion_time": session.completion_time_minutes
                    }
                )
    
    def _generate_sample_document(self, query: str) -> str:
        """Generate a sample research document for demonstration."""
        return f"""# Research Report: {query}

## Executive Summary

This research report provides a comprehensive analysis of {query.lower()}. The findings are based on analysis of multiple sources including academic papers, industry reports, and expert opinions.

## Introduction

The topic of {query.lower()} has gained significant attention in recent years. This report examines the current state, key developments, and future implications.

## Key Findings

1. **Primary Insight**: The research reveals several important trends and patterns.
2. **Secondary Analysis**: Supporting evidence indicates strong correlations.
3. **Future Implications**: The findings suggest important considerations for stakeholders.

## Methodology

This research employed a systematic approach to data collection and analysis:
- Web-based source identification and extraction
- Academic database queries across multiple disciplines  
- Content analysis using natural language processing
- Synthesis of findings into coherent insights

## Conclusions

Based on the comprehensive analysis, several key conclusions emerge regarding {query.lower()}. These findings provide valuable insights for decision-makers and researchers in the field.

## References

1. Source 1: Academic Paper on Related Topic
2. Source 2: Industry Report with Relevant Data
3. Source 3: Expert Analysis and Commentary
4. Source 4: Recent Research Findings
5. Source 5: Statistical Data and Trends
6. Source 6: Case Study Analysis
7. Source 7: Comparative Research
8. Source 8: Future Projections and Scenarios

---
*This report was generated by Agent Scrivener - Autonomous Research Platform*
"""