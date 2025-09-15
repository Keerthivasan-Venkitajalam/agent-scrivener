"""
Session management system for Agent Scrivener.

Handles session creation, lifecycle management, and short-term memory
for active research sessions.
"""

import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from contextlib import asynccontextmanager

from ..models.core import (
    ResearchSession, ResearchPlan, SessionState, TaskStatus,
    WorkflowStep, AgentExecution, SessionMetrics
)
from ..utils.logging import get_logger
from ..utils.error_handler import ErrorHandler
from .session_persistence import SessionPersistence

logger = get_logger(__name__)


class SessionManager:
    """
    Manages research session lifecycle and short-term memory.
    
    Provides session creation, state management, persistence, and recovery
    capabilities for active research sessions.
    """
    
    def __init__(self, persistence: Optional[SessionPersistence] = None):
        """Initialize session manager with optional persistence layer."""
        self.active_sessions: Dict[str, ResearchSession] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self.persistence = persistence or SessionPersistence()
        self.error_handler = ErrorHandler()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._session_timeout_minutes = 480  # 8 hours default timeout
        
    async def start(self):
        """Start the session manager and background cleanup task."""
        logger.info("Starting session manager")
        await self.persistence.initialize()
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        
        # Recover any persisted sessions
        await self._recover_sessions()
        
    async def stop(self):
        """Stop the session manager and cleanup resources."""
        logger.info("Stopping session manager")
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Persist all active sessions
        await self._persist_all_sessions()
        
        # Clear active sessions
        self.active_sessions.clear()
        self.session_locks.clear()
        
    async def create_session(self, query: str, plan: ResearchPlan) -> ResearchSession:
        """
        Create a new research session.
        
        Args:
            query: The research query
            plan: The research execution plan
            
        Returns:
            Created research session
            
        Raises:
            ValueError: If query or plan is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not plan or not plan.session_id:
            raise ValueError("Plan must have a valid session_id")
        
        session_id = plan.session_id
        
        # Check if session already exists
        if session_id in self.active_sessions:
            raise ValueError(f"Session {session_id} already exists")
        
        # Create session
        session = ResearchSession(
            session_id=session_id,
            original_query=query,
            plan=plan,
            status=TaskStatus.PENDING,
            session_state=SessionState.INITIALIZING
        )
        
        # Add session lock
        self.session_locks[session_id] = asyncio.Lock()
        
        # Store in active sessions
        self.active_sessions[session_id] = session
        
        # Persist session
        await self.persistence.save_session(session)
        
        logger.info(f"Created session {session_id} for query: {query[:100]}...")
        return session
    
    async def get_session(self, session_id: str) -> Optional[ResearchSession]:
        """
        Get a session by ID.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Research session if found, None otherwise
        """
        # Check active sessions first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to load from persistence
        session = await self.persistence.load_session(session_id)
        if session:
            # Add to active sessions
            self.active_sessions[session_id] = session
            self.session_locks[session_id] = asyncio.Lock()
            logger.info(f"Loaded session {session_id} from persistence")
        
        return session
    
    async def update_session(self, session: ResearchSession) -> None:
        """
        Update a session in memory and persistence.
        
        Args:
            session: The updated session
        """
        session_id = session.session_id
        
        async with self._get_session_lock(session_id):
            # Update timestamp
            session.update_timestamp()
            
            # Update active sessions
            self.active_sessions[session_id] = session
            
            # Persist changes
            await self.persistence.save_session(session)
            
        logger.debug(f"Updated session {session_id}")
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from memory and persistence.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if session was deleted, False if not found
        """
        async with self._get_session_lock(session_id):
            # Remove from active sessions
            session_removed = self.active_sessions.pop(session_id, None) is not None
            
            # Remove lock
            self.session_locks.pop(session_id, None)
            
            # Remove from persistence
            persistence_removed = await self.persistence.delete_session(session_id)
            
        if session_removed or persistence_removed:
            logger.info(f"Deleted session {session_id}")
            return True
        
        return False
    
    async def list_sessions(self, 
                          limit: Optional[int] = None,
                          status_filter: Optional[TaskStatus] = None,
                          state_filter: Optional[SessionState] = None) -> List[ResearchSession]:
        """
        List sessions with optional filtering.
        
        Args:
            limit: Maximum number of sessions to return
            status_filter: Filter by task status
            state_filter: Filter by session state
            
        Returns:
            List of matching sessions
        """
        # Get all sessions (active + persisted)
        all_sessions = list(self.active_sessions.values())
        
        # Load additional sessions from persistence
        persisted_sessions = await self.persistence.list_sessions()
        for session in persisted_sessions:
            if session.session_id not in self.active_sessions:
                all_sessions.append(session)
        
        # Apply filters
        filtered_sessions = all_sessions
        
        if status_filter:
            filtered_sessions = [s for s in filtered_sessions if s.status == status_filter]
        
        if state_filter:
            filtered_sessions = [s for s in filtered_sessions if s.session_state == state_filter]
        
        # Sort by creation time (newest first)
        filtered_sessions.sort(key=lambda s: s.created_at, reverse=True)
        
        # Apply limit
        if limit:
            filtered_sessions = filtered_sessions[:limit]
        
        return filtered_sessions
    
    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of session state and progress.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Session summary dict if found, None otherwise
        """
        session = await self.get_session(session_id)
        if not session:
            return None
        
        return session.get_session_summary()
    
    async def transition_session_state(self, session_id: str, new_state: SessionState) -> bool:
        """
        Transition a session to a new state.
        
        Args:
            session_id: The session identifier
            new_state: The new session state
            
        Returns:
            True if transition was successful, False otherwise
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        async with self._get_session_lock(session_id):
            old_state = session.session_state
            session.transition_state(new_state)
            await self.update_session(session)
            
        logger.info(f"Session {session_id} transitioned from {old_state} to {new_state}")
        return True
    
    async def add_agent_execution(self, session_id: str, execution: AgentExecution) -> bool:
        """
        Add an agent execution to a session.
        
        Args:
            session_id: The session identifier
            execution: The agent execution to add
            
        Returns:
            True if execution was added, False otherwise
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        async with self._get_session_lock(session_id):
            session.add_agent_execution(execution)
            await self.update_session(session)
        
        logger.debug(f"Added execution {execution.execution_id} to session {session_id}")
        return True
    
    async def update_session_metrics(self, session_id: str) -> bool:
        """
        Update session metrics based on current session state.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if metrics were updated, False otherwise
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        async with self._get_session_lock(session_id):
            # Update metrics from session data
            session.metrics.update_from_session(session)
            
            # Calculate execution metrics
            successful_executions = sum(
                1 for exec in session.agent_executions 
                if exec.status == TaskStatus.COMPLETED
            )
            failed_executions = sum(
                1 for exec in session.agent_executions 
                if exec.status == TaskStatus.FAILED
            )
            
            session.metrics.successful_agent_executions = successful_executions
            session.metrics.failed_agent_executions = failed_executions
            
            # Calculate total execution time
            total_time = sum(
                exec.execution_time_seconds or 0 
                for exec in session.agent_executions
            )
            session.metrics.total_execution_time_seconds = total_time
            
            await self.update_session(session)
        
        logger.debug(f"Updated metrics for session {session_id}")
        return True
    
    @asynccontextmanager
    async def session_context(self, session_id: str):
        """
        Context manager for safe session access with automatic locking.
        
        Args:
            session_id: The session identifier
            
        Yields:
            The session object if found
            
        Raises:
            ValueError: If session is not found
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        async with self._get_session_lock(session_id):
            try:
                yield session
            finally:
                # Always update session after context
                await self.update_session(session)
    
    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for a session."""
        if session_id not in self.session_locks:
            self.session_locks[session_id] = asyncio.Lock()
        return self.session_locks[session_id]
    
    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.active_sessions.items():
                    # Check if session has been inactive for too long
                    time_since_update = current_time - session.updated_at
                    if time_since_update > timedelta(minutes=self._session_timeout_minutes):
                        expired_sessions.append(session_id)
                
                # Remove expired sessions
                for session_id in expired_sessions:
                    logger.info(f"Cleaning up expired session {session_id}")
                    await self.delete_session(session_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _recover_sessions(self):
        """Recover sessions from persistence on startup."""
        try:
            persisted_sessions = await self.persistence.list_sessions()
            
            for session in persisted_sessions:
                # Only recover recent sessions (within timeout period)
                time_since_update = datetime.now() - session.updated_at
                if time_since_update <= timedelta(minutes=self._session_timeout_minutes):
                    self.active_sessions[session.session_id] = session
                    self.session_locks[session.session_id] = asyncio.Lock()
                    logger.info(f"Recovered session {session.session_id}")
                else:
                    # Clean up old sessions
                    await self.persistence.delete_session(session.session_id)
                    logger.info(f"Cleaned up old session {session.session_id}")
                    
        except Exception as e:
            logger.error(f"Error recovering sessions: {e}")
    
    async def _persist_all_sessions(self):
        """Persist all active sessions."""
        for session in self.active_sessions.values():
            try:
                await self.persistence.save_session(session)
            except Exception as e:
                logger.error(f"Error persisting session {session.session_id}: {e}")
    
    async def get_active_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self.active_sessions)
    
    async def get_session_statistics(self) -> Dict[str, Any]:
        """Get overall session statistics."""
        total_sessions = len(self.active_sessions)
        
        # Count sessions by state
        state_counts = {}
        status_counts = {}
        
        for session in self.active_sessions.values():
            state = session.session_state.value
            status = session.status.value
            
            state_counts[state] = state_counts.get(state, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_active_sessions": total_sessions,
            "sessions_by_state": state_counts,
            "sessions_by_status": status_counts,
            "session_timeout_minutes": self._session_timeout_minutes
        }