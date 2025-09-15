"""
Unit tests for SessionManager.
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from agent_scrivener.memory.session_manager import SessionManager
from agent_scrivener.memory.session_persistence import SessionPersistence
from agent_scrivener.models.core import (
    ResearchSession, ResearchPlan, SessionState, TaskStatus,
    AgentExecution, WorkflowStep
)


@pytest.fixture
def mock_persistence():
    """Mock session persistence."""
    persistence = AsyncMock(spec=SessionPersistence)
    persistence.initialize = AsyncMock()
    persistence.save_session = AsyncMock(return_value=True)
    persistence.load_session = AsyncMock(return_value=None)
    persistence.delete_session = AsyncMock(return_value=True)
    persistence.list_sessions = AsyncMock(return_value=[])
    return persistence


@pytest.fixture
def session_manager(mock_persistence):
    """Create session manager with mocked persistence."""
    return SessionManager(persistence=mock_persistence)


@pytest.fixture
def sample_plan():
    """Create a sample research plan."""
    return ResearchPlan(
        query="Test query",
        session_id=str(uuid.uuid4()),
        estimated_duration_minutes=60
    )


@pytest.fixture
def sample_session(sample_plan):
    """Create a sample research session."""
    return ResearchSession(
        session_id=sample_plan.session_id,
        original_query=sample_plan.query,
        plan=sample_plan
    )


class TestSessionManager:
    """Test cases for SessionManager."""
    
    async def test_start_and_stop(self, session_manager, mock_persistence):
        """Test session manager startup and shutdown."""
        await session_manager.start()
        
        # Verify persistence was initialized
        mock_persistence.initialize.assert_called_once()
        
        # Verify cleanup task is running
        assert session_manager._cleanup_task is not None
        assert not session_manager._cleanup_task.done()
        
        await session_manager.stop()
        
        # Verify cleanup task was cancelled
        assert session_manager._cleanup_task.cancelled()
        
        # Verify sessions were cleared
        assert len(session_manager.active_sessions) == 0
        assert len(session_manager.session_locks) == 0
    
    async def test_create_session_success(self, session_manager, sample_plan, mock_persistence):
        """Test successful session creation."""
        await session_manager.start()
        
        query = "Test research query"
        session = await session_manager.create_session(query, sample_plan)
        
        # Verify session properties
        assert session.session_id == sample_plan.session_id
        assert session.original_query == query
        assert session.plan == sample_plan
        assert session.status == TaskStatus.PENDING
        assert session.session_state == SessionState.INITIALIZING
        
        # Verify session is stored
        assert sample_plan.session_id in session_manager.active_sessions
        assert sample_plan.session_id in session_manager.session_locks
        
        # Verify persistence was called
        mock_persistence.save_session.assert_called_once_with(session)
        
        await session_manager.stop()
    
    async def test_create_session_invalid_query(self, session_manager, sample_plan):
        """Test session creation with invalid query."""
        await session_manager.start()
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await session_manager.create_session("", sample_plan)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await session_manager.create_session("   ", sample_plan)
        
        await session_manager.stop()
    
    async def test_create_session_invalid_plan(self, session_manager):
        """Test session creation with invalid plan."""
        await session_manager.start()
        
        with pytest.raises(ValueError, match="Plan must have a valid session_id"):
            await session_manager.create_session("query", None)
        
        invalid_plan = ResearchPlan(
            query="test",
            session_id="",
            estimated_duration_minutes=60
        )
        
        with pytest.raises(ValueError, match="Plan must have a valid session_id"):
            await session_manager.create_session("query", invalid_plan)
        
        await session_manager.stop()
    
    async def test_create_duplicate_session(self, session_manager, sample_plan):
        """Test creating duplicate session."""
        await session_manager.start()
        
        # Create first session
        await session_manager.create_session("query1", sample_plan)
        
        # Try to create duplicate
        with pytest.raises(ValueError, match=f"Session {sample_plan.session_id} already exists"):
            await session_manager.create_session("query2", sample_plan)
        
        await session_manager.stop()
    
    async def test_get_session_from_memory(self, session_manager, sample_plan):
        """Test getting session from active memory."""
        await session_manager.start()
        
        # Create session
        created_session = await session_manager.create_session("query", sample_plan)
        
        # Get session
        retrieved_session = await session_manager.get_session(sample_plan.session_id)
        
        assert retrieved_session is not None
        assert retrieved_session.session_id == created_session.session_id
        assert retrieved_session is created_session  # Same object reference
        
        await session_manager.stop()
    
    async def test_get_session_from_persistence(self, session_manager, sample_session, mock_persistence):
        """Test getting session from persistence."""
        await session_manager.start()
        
        # Mock persistence to return session
        mock_persistence.load_session.return_value = sample_session
        
        # Get session (not in active memory)
        retrieved_session = await session_manager.get_session(sample_session.session_id)
        
        assert retrieved_session is not None
        assert retrieved_session.session_id == sample_session.session_id
        
        # Verify session was loaded from persistence
        mock_persistence.load_session.assert_called_once_with(sample_session.session_id)
        
        # Verify session was added to active memory
        assert sample_session.session_id in session_manager.active_sessions
        assert sample_session.session_id in session_manager.session_locks
        
        await session_manager.stop()
    
    async def test_get_nonexistent_session(self, session_manager, mock_persistence):
        """Test getting non-existent session."""
        await session_manager.start()
        
        # Mock persistence to return None
        mock_persistence.load_session.return_value = None
        
        session = await session_manager.get_session("nonexistent")
        
        assert session is None
        mock_persistence.load_session.assert_called_once_with("nonexistent")
        
        await session_manager.stop()
    
    async def test_update_session(self, session_manager, sample_plan, mock_persistence):
        """Test updating a session."""
        await session_manager.start()
        
        # Create session
        session = await session_manager.create_session("query", sample_plan)
        original_updated_at = session.updated_at
        
        # Wait a bit to ensure timestamp changes
        await asyncio.sleep(0.01)
        
        # Update session
        session.session_state = SessionState.RESEARCHING
        await session_manager.update_session(session)
        
        # Verify timestamp was updated
        assert session.updated_at > original_updated_at
        
        # Verify persistence was called
        assert mock_persistence.save_session.call_count == 2  # Create + Update
        
        await session_manager.stop()
    
    async def test_delete_session(self, session_manager, sample_plan, mock_persistence):
        """Test deleting a session."""
        await session_manager.start()
        
        # Create session
        await session_manager.create_session("query", sample_plan)
        session_id = sample_plan.session_id
        
        # Verify session exists
        assert session_id in session_manager.active_sessions
        assert session_id in session_manager.session_locks
        
        # Delete session
        result = await session_manager.delete_session(session_id)
        
        assert result is True
        
        # Verify session was removed
        assert session_id not in session_manager.active_sessions
        assert session_id not in session_manager.session_locks
        
        # Verify persistence was called
        mock_persistence.delete_session.assert_called_once_with(session_id)
        
        await session_manager.stop()
    
    async def test_delete_nonexistent_session(self, session_manager, mock_persistence):
        """Test deleting non-existent session."""
        await session_manager.start()
        
        # Mock persistence to return False
        mock_persistence.delete_session.return_value = False
        
        result = await session_manager.delete_session("nonexistent")
        
        assert result is False
        mock_persistence.delete_session.assert_called_once_with("nonexistent")
        
        await session_manager.stop()
    
    async def test_list_sessions(self, session_manager, mock_persistence):
        """Test listing sessions."""
        await session_manager.start()
        
        # Create some active sessions
        plan1 = ResearchPlan(query="query1", session_id="session1", estimated_duration_minutes=60)
        plan2 = ResearchPlan(query="query2", session_id="session2", estimated_duration_minutes=60)
        
        session1 = await session_manager.create_session("query1", plan1)
        session2 = await session_manager.create_session("query2", plan2)
        
        # Mock persistence to return additional sessions
        persisted_session = ResearchSession(
            session_id="session3",
            original_query="query3",
            plan=ResearchPlan(query="query3", session_id="session3", estimated_duration_minutes=60)
        )
        mock_persistence.list_sessions.return_value = [persisted_session]
        
        # List sessions
        sessions = await session_manager.list_sessions()
        
        assert len(sessions) == 3
        session_ids = {s.session_id for s in sessions}
        assert session_ids == {"session1", "session2", "session3"}
        
        await session_manager.stop()
    
    async def test_list_sessions_with_filters(self, session_manager, mock_persistence):
        """Test listing sessions with filters."""
        await session_manager.start()
        
        # Create sessions with different states
        plan1 = ResearchPlan(query="query1", session_id="session1", estimated_duration_minutes=60)
        plan2 = ResearchPlan(query="query2", session_id="session2", estimated_duration_minutes=60)
        
        session1 = await session_manager.create_session("query1", plan1)
        session2 = await session_manager.create_session("query2", plan2)
        
        # Update session states
        session1.session_state = SessionState.RESEARCHING
        session2.session_state = SessionState.COMPLETED
        session2.status = TaskStatus.COMPLETED
        
        await session_manager.update_session(session1)
        await session_manager.update_session(session2)
        
        # Test state filter
        researching_sessions = await session_manager.list_sessions(
            state_filter=SessionState.RESEARCHING
        )
        assert len(researching_sessions) == 1
        assert researching_sessions[0].session_id == "session1"
        
        # Test status filter
        completed_sessions = await session_manager.list_sessions(
            status_filter=TaskStatus.COMPLETED
        )
        assert len(completed_sessions) == 1
        assert completed_sessions[0].session_id == "session2"
        
        # Test limit
        limited_sessions = await session_manager.list_sessions(limit=1)
        assert len(limited_sessions) == 1
        
        await session_manager.stop()
    
    async def test_transition_session_state(self, session_manager, sample_plan):
        """Test transitioning session state."""
        await session_manager.start()
        
        # Create session
        session = await session_manager.create_session("query", sample_plan)
        session_id = session.session_id
        
        # Transition state
        result = await session_manager.transition_session_state(
            session_id, SessionState.RESEARCHING
        )
        
        assert result is True
        
        # Verify state was changed
        updated_session = await session_manager.get_session(session_id)
        assert updated_session.session_state == SessionState.RESEARCHING
        
        await session_manager.stop()
    
    async def test_transition_nonexistent_session_state(self, session_manager):
        """Test transitioning state of non-existent session."""
        await session_manager.start()
        
        result = await session_manager.transition_session_state(
            "nonexistent", SessionState.RESEARCHING
        )
        
        assert result is False
        
        await session_manager.stop()
    
    async def test_add_agent_execution(self, session_manager, sample_plan):
        """Test adding agent execution to session."""
        await session_manager.start()
        
        # Create session
        await session_manager.create_session("query", sample_plan)
        session_id = sample_plan.session_id
        
        # Create agent execution
        execution = AgentExecution(
            execution_id="exec1",
            agent_name="TestAgent",
            task_id="task1"
        )
        
        # Add execution
        result = await session_manager.add_agent_execution(session_id, execution)
        
        assert result is True
        
        # Verify execution was added
        session = await session_manager.get_session(session_id)
        assert len(session.agent_executions) == 1
        assert session.agent_executions[0].execution_id == "exec1"
        
        await session_manager.stop()
    
    async def test_update_session_metrics(self, session_manager, sample_plan):
        """Test updating session metrics."""
        await session_manager.start()
        
        # Create session
        session = await session_manager.create_session("query", sample_plan)
        session_id = session.session_id
        
        # Add some executions
        exec1 = AgentExecution(
            execution_id="exec1",
            agent_name="Agent1",
            task_id="task1",
            status=TaskStatus.COMPLETED,
            execution_time_seconds=10.0
        )
        exec2 = AgentExecution(
            execution_id="exec2",
            agent_name="Agent2",
            task_id="task2",
            status=TaskStatus.FAILED,
            execution_time_seconds=5.0
        )
        
        await session_manager.add_agent_execution(session_id, exec1)
        await session_manager.add_agent_execution(session_id, exec2)
        
        # Update metrics
        result = await session_manager.update_session_metrics(session_id)
        
        assert result is True
        
        # Verify metrics were updated
        updated_session = await session_manager.get_session(session_id)
        assert updated_session.metrics.successful_agent_executions == 1
        assert updated_session.metrics.failed_agent_executions == 1
        assert updated_session.metrics.total_execution_time_seconds == 15.0
        
        await session_manager.stop()
    
    async def test_session_context_manager(self, session_manager, sample_plan):
        """Test session context manager."""
        await session_manager.start()
        
        # Create session
        await session_manager.create_session("query", sample_plan)
        session_id = sample_plan.session_id
        
        # Use context manager
        async with session_manager.session_context(session_id) as session:
            assert session is not None
            assert session.session_id == session_id
            
            # Modify session
            session.session_state = SessionState.ANALYZING
        
        # Verify session was updated
        updated_session = await session_manager.get_session(session_id)
        assert updated_session.session_state == SessionState.ANALYZING
        
        await session_manager.stop()
    
    async def test_session_context_manager_nonexistent(self, session_manager):
        """Test session context manager with non-existent session."""
        await session_manager.start()
        
        with pytest.raises(ValueError, match="Session nonexistent not found"):
            async with session_manager.session_context("nonexistent"):
                pass
        
        await session_manager.stop()
    
    async def test_get_session_statistics(self, session_manager):
        """Test getting session statistics."""
        await session_manager.start()
        
        # Create sessions with different states
        plan1 = ResearchPlan(query="query1", session_id="session1", estimated_duration_minutes=60)
        plan2 = ResearchPlan(query="query2", session_id="session2", estimated_duration_minutes=60)
        
        session1 = await session_manager.create_session("query1", plan1)
        session2 = await session_manager.create_session("query2", plan2)
        
        # Update states
        session1.session_state = SessionState.RESEARCHING
        session2.session_state = SessionState.COMPLETED
        session2.status = TaskStatus.COMPLETED
        
        await session_manager.update_session(session1)
        await session_manager.update_session(session2)
        
        # Get statistics
        stats = await session_manager.get_session_statistics()
        
        assert stats["total_active_sessions"] == 2
        assert stats["sessions_by_state"]["researching"] == 1
        assert stats["sessions_by_state"]["completed"] == 1
        assert stats["sessions_by_status"]["pending"] == 1
        assert stats["sessions_by_status"]["completed"] == 1
        
        await session_manager.stop()
    
    @patch('agent_scrivener.memory.session_manager.asyncio.sleep')
    async def test_cleanup_expired_sessions(self, mock_sleep, session_manager, sample_plan):
        """Test cleanup of expired sessions."""
        # Mock sleep to prevent actual waiting
        mock_sleep.side_effect = [None, asyncio.CancelledError()]
        
        await session_manager.start()
        
        # Create session and make it appear expired
        session = await session_manager.create_session("query", sample_plan)
        session.updated_at = datetime.now() - timedelta(hours=10)  # Make it expired
        session_manager.active_sessions[session.session_id] = session
        
        # Wait for cleanup task to run once
        try:
            await session_manager._cleanup_task
        except asyncio.CancelledError:
            pass
        
        # Verify expired session was removed
        assert session.session_id not in session_manager.active_sessions
        
        await session_manager.stop()
    
    async def test_get_active_session_count(self, session_manager):
        """Test getting active session count."""
        await session_manager.start()
        
        assert await session_manager.get_active_session_count() == 0
        
        # Create sessions
        plan1 = ResearchPlan(query="query1", session_id="session1", estimated_duration_minutes=60)
        plan2 = ResearchPlan(query="query2", session_id="session2", estimated_duration_minutes=60)
        
        await session_manager.create_session("query1", plan1)
        await session_manager.create_session("query2", plan2)
        
        assert await session_manager.get_active_session_count() == 2
        
        await session_manager.stop()
    
    async def test_get_session_summary(self, session_manager, sample_plan):
        """Test getting session summary."""
        await session_manager.start()
        
        # Create session
        await session_manager.create_session("query", sample_plan)
        session_id = sample_plan.session_id
        
        # Get summary
        summary = await session_manager.get_session_summary(session_id)
        
        assert summary is not None
        assert summary["session_id"] == session_id
        assert summary["query"] == "query"
        assert summary["status"] == "pending"
        assert summary["session_state"] == "initializing"
        assert "progress_percentage" in summary
        assert "metrics" in summary
        
        await session_manager.stop()
    
    async def test_get_session_summary_nonexistent(self, session_manager):
        """Test getting summary of non-existent session."""
        await session_manager.start()
        
        summary = await session_manager.get_session_summary("nonexistent")
        
        assert summary is None
        
        await session_manager.stop()