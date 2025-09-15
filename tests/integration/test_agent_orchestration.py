"""
Integration tests for agent orchestration system.

Tests multi-agent coordination, task dispatching, result aggregation,
and progress tracking across the entire orchestration system.
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from agent_scrivener.orchestration.orchestrator import (
    AgentOrchestrator, OrchestrationConfig
)
from agent_scrivener.orchestration.registry import (
    enhanced_registry, AgentInstance, AgentStatus
)
from agent_scrivener.agents.base import BaseAgent, AgentResult, agent_registry
from agent_scrivener.models.core import (
    ResearchPlan, ResearchTask, TaskStatus, ResearchSession,
    SessionState, ExtractedArticle, Source, SourceType, Insight
)


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def __init__(self, name: str, execution_delay: float = 0.1, should_fail: bool = False):
        super().__init__(name)
        self.execution_delay = execution_delay
        self.should_fail = should_fail
        self.execution_count = 0
        self.last_kwargs = {}
    
    async def execute(self, **kwargs) -> AgentResult:
        """Mock execution with configurable delay and failure."""
        self.execution_count += 1
        self.last_kwargs = kwargs
        
        # Simulate processing time
        await asyncio.sleep(self.execution_delay)
        
        if self.should_fail:
            return AgentResult(
                success=False,
                error=f"Mock failure from {self.name}",
                timestamp=datetime.now(),
                agent_name=self.name,
                execution_time_ms=int(self.execution_delay * 1000)
            )
        
        # Return mock data based on agent type
        mock_data = self._generate_mock_data(kwargs)
        
        return AgentResult(
            success=True,
            data=mock_data,
            timestamp=datetime.now(),
            agent_name=self.name,
            execution_time_ms=int(self.execution_delay * 1000)
        )
    
    def _generate_mock_data(self, kwargs: Dict[str, Any]) -> Any:
        """Generate mock data based on agent type."""
        if "research" in self.name:
            return [
                ExtractedArticle(
                    source=Source(
                        url="https://example.com/article1",
                        title="Mock Article 1",
                        source_type=SourceType.WEB
                    ),
                    content="Mock article content 1",
                    confidence_score=0.8
                ),
                ExtractedArticle(
                    source=Source(
                        url="https://example.com/article2", 
                        title="Mock Article 2",
                        source_type=SourceType.WEB
                    ),
                    content="Mock article content 2",
                    confidence_score=0.9
                )
            ]
        elif "analysis" in self.name:
            return [
                Insight(
                    topic="Mock Topic 1",
                    summary="Mock insight summary 1",
                    confidence_score=0.85,
                    supporting_evidence=["Evidence 1", "Evidence 2"]
                ),
                Insight(
                    topic="Mock Topic 2", 
                    summary="Mock insight summary 2",
                    confidence_score=0.75,
                    supporting_evidence=["Evidence 3", "Evidence 4"]
                )
            ]
        elif "drafting" in self.name:
            return "# Mock Research Document\n\nThis is a mock research document generated for testing."
        elif "citation" in self.name:
            return [
                {"citation_id": "cite1", "text": "Mock citation 1"},
                {"citation_id": "cite2", "text": "Mock citation 2"}
            ]
        else:
            return {"mock_result": f"Result from {self.name}"}


@pytest.fixture
def orchestrator():
    """Create orchestrator instance for testing."""
    config = OrchestrationConfig(
        max_concurrent_tasks=3,
        task_timeout_seconds=10,
        progress_update_interval_seconds=1,
        enable_parallel_execution=True
    )
    
    return AgentOrchestrator(config)


@pytest.fixture
def mock_agents():
    """Create mock agents for testing."""
    agents = {
        "research": MockAgent("research", execution_delay=0.2),
        "api": MockAgent("api", execution_delay=0.15),
        "analysis": MockAgent("analysis", execution_delay=0.3),
        "drafting": MockAgent("drafting", execution_delay=0.25),
        "citation": MockAgent("citation", execution_delay=0.1)
    }
    
    return agents


@pytest.fixture
def sample_research_plan():
    """Create a sample research plan for testing."""
    session_id = str(uuid.uuid4())
    
    tasks = [
        ResearchTask(
            task_id="web_research",
            task_type="web_search",
            description="Search web sources",
            parameters={"query": "test query", "max_sources": 5},
            assigned_agent="research"
        ),
        ResearchTask(
            task_id="academic_research",
            task_type="academic_search", 
            description="Search academic databases",
            parameters={"query": "test query", "databases": ["arxiv"]},
            assigned_agent="api"
        ),
        ResearchTask(
            task_id="data_analysis",
            task_type="content_analysis",
            description="Analyze collected data",
            parameters={"analysis_types": ["topic_modeling"]},
            dependencies=["web_research", "academic_research"],
            assigned_agent="analysis"
        ),
        ResearchTask(
            task_id="content_drafting",
            task_type="document_generation",
            description="Generate document",
            parameters={"document_type": "research_report"},
            dependencies=["data_analysis"],
            assigned_agent="drafting"
        ),
        ResearchTask(
            task_id="citation_management",
            task_type="citation_formatting",
            description="Format citations",
            parameters={"citation_style": "APA"},
            dependencies=["web_research", "academic_research", "content_drafting"],
            assigned_agent="citation"
        )
    ]
    
    return ResearchPlan(
        query="Test research query for integration testing",
        session_id=session_id,
        tasks=tasks,
        estimated_duration_minutes=30
    )


class TestAgentOrchestration:
    """Test cases for agent orchestration system."""
    
    @pytest.mark.asyncio
    async def test_start_research_session(self, orchestrator, mock_agents, sample_research_plan):
        """Test starting a research session."""
        # Register mock agents in the base registry
        from agent_scrivener.agents.base import agent_registry
        for name, agent in mock_agents.items():
            agent_registry.register_agent(agent)
        
        try:
            session = await orchestrator.start_research_session(sample_research_plan)
            
            assert session.session_id == sample_research_plan.session_id
            assert session.original_query == sample_research_plan.query
            assert session.status == TaskStatus.IN_PROGRESS
            assert session.session_state == SessionState.RESEARCHING
            assert len(session.plan.tasks) == 5
            
            # Verify session is tracked
            tracked_session = orchestrator.get_session(session.session_id)
            assert tracked_session is not None
            assert tracked_session.session_id == session.session_id
            
        finally:
            await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_task_execution_order(self, orchestrator, mock_agents, sample_research_plan):
        """Test that tasks execute in correct dependency order."""
        session = await orchestrator.start_research_session(sample_research_plan)
        
        # Wait for session to complete
        max_wait_time = 10  # seconds
        start_time = datetime.now()
        
        while session.status == TaskStatus.IN_PROGRESS:
            if (datetime.now() - start_time).total_seconds() > max_wait_time:
                break
            await asyncio.sleep(0.1)
            session = orchestrator.get_session(session.session_id)
        
        # Verify execution order
        completed_tasks = [task for task in session.plan.tasks if task.status == TaskStatus.COMPLETED]
        
        # Research tasks should complete first (no dependencies)
        web_task = next(task for task in completed_tasks if task.task_id == "web_research")
        api_task = next(task for task in completed_tasks if task.task_id == "academic_research")
        
        # Analysis task should complete after research tasks
        analysis_task = next(task for task in completed_tasks if task.task_id == "data_analysis")
        assert analysis_task.started_at >= web_task.completed_at
        assert analysis_task.started_at >= api_task.completed_at
        
        # Drafting should complete after analysis
        drafting_task = next(task for task in completed_tasks if task.task_id == "content_drafting")
        assert drafting_task.started_at >= analysis_task.completed_at
        
        # Citation should complete after all dependencies
        citation_task = next(task for task in completed_tasks if task.task_id == "citation_management")
        assert citation_task.started_at >= web_task.completed_at
        assert citation_task.started_at >= api_task.completed_at
        assert citation_task.started_at >= drafting_task.completed_at
    
    @pytest.mark.asyncio
    async def test_parallel_task_execution(self, orchestrator, mock_agents, sample_research_plan):
        """Test that independent tasks execute in parallel."""
        session = await orchestrator.start_research_session(sample_research_plan)
        
        # Wait for initial tasks to start
        await asyncio.sleep(0.5)
        
        # Get research tasks (should execute in parallel)
        web_task = session.plan.get_task_by_id("web_research")
        api_task = session.plan.get_task_by_id("academic_research")
        
        # Both should be in progress or completed
        assert web_task.status in [TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED]
        assert api_task.status in [TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED]
        
        # If both started, they should have started around the same time (parallel execution)
        if web_task.started_at and api_task.started_at:
            time_diff = abs((web_task.started_at - api_task.started_at).total_seconds())
            assert time_diff < 1.0  # Should start within 1 second of each other
    
    @pytest.mark.asyncio
    async def test_progress_tracking(self, orchestrator, mock_agents, sample_research_plan):
        """Test progress tracking functionality."""
        progress_updates = []
        
        def progress_callback(progress_data):
            progress_updates.append(progress_data.copy())
        
        session = await orchestrator.start_research_session(sample_research_plan)
        orchestrator.register_progress_callback(session.session_id, progress_callback)
        
        # Wait for some progress
        await asyncio.sleep(1.0)
        
        # Check that progress updates were received
        assert len(progress_updates) > 0
        
        latest_progress = progress_updates[-1]
        assert "session_id" in latest_progress
        assert "progress_percentage" in latest_progress
        assert "completed_tasks" in latest_progress
        assert "total_tasks" in latest_progress
        assert latest_progress["session_id"] == session.session_id
        assert latest_progress["total_tasks"] == 5
        assert 0 <= latest_progress["progress_percentage"] <= 100
    
    @pytest.mark.asyncio
    async def test_result_aggregation(self, orchestrator, mock_agents, sample_research_plan):
        """Test result aggregation from multiple agents."""
        session = await orchestrator.start_research_session(sample_research_plan)
        
        # Wait for session to complete
        max_wait_time = 15  # seconds
        start_time = datetime.now()
        
        while session.status == TaskStatus.IN_PROGRESS:
            if (datetime.now() - start_time).total_seconds() > max_wait_time:
                break
            await asyncio.sleep(0.1)
            session = orchestrator.get_session(session.session_id)
        
        # Get aggregated results
        results = await orchestrator.get_session_results(session.session_id)
        
        assert results is not None
        assert results["session_id"] == session.session_id
        assert results["query"] == sample_research_plan.query
        assert "sources" in results
        assert "analysis" in results
        assert "document" in results
        assert "execution" in results
        
        # Verify execution metrics
        execution_data = results["execution"]
        assert execution_data["total_tasks"] == 5
        assert execution_data["completed_tasks"] > 0
        assert execution_data["total_executions"] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_and_isolation(self, orchestrator, mock_agents, sample_research_plan):
        """Test error handling and task isolation."""
        # Make one agent fail
        mock_agents["analysis"].should_fail = True
        
        session = await orchestrator.start_research_session(sample_research_plan)
        
        # Wait for session to complete
        max_wait_time = 15  # seconds
        start_time = datetime.now()
        
        while session.status == TaskStatus.IN_PROGRESS:
            if (datetime.now() - start_time).total_seconds() > max_wait_time:
                break
            await asyncio.sleep(0.1)
            session = orchestrator.get_session(session.session_id)
        
        # Verify that other tasks still completed despite analysis failure
        web_task = session.plan.get_task_by_id("web_research")
        api_task = session.plan.get_task_by_id("academic_research")
        analysis_task = session.plan.get_task_by_id("data_analysis")
        
        assert web_task.status == TaskStatus.COMPLETED
        assert api_task.status == TaskStatus.COMPLETED
        assert analysis_task.status == TaskStatus.FAILED
        assert analysis_task.error_message is not None
        
        # Dependent tasks should not execute if their dependencies failed
        drafting_task = session.plan.get_task_by_id("content_drafting")
        assert drafting_task.status == TaskStatus.PENDING  # Should not have started
    
    @pytest.mark.asyncio
    async def test_session_cancellation(self, orchestrator, mock_agents, sample_research_plan):
        """Test session cancellation functionality."""
        # Use longer delays to ensure tasks are running when we cancel
        for agent in mock_agents.values():
            agent.execution_delay = 2.0
        
        session = await orchestrator.start_research_session(sample_research_plan)
        
        # Wait for tasks to start
        await asyncio.sleep(0.5)
        
        # Cancel the session
        cancelled = await orchestrator.cancel_session(session.session_id)
        assert cancelled is True
        
        # Verify session status
        session = orchestrator.get_session(session.session_id)
        assert session.status == TaskStatus.CANCELLED
        assert session.session_state == SessionState.CANCELLED
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, orchestrator, mock_agents):
        """Test handling multiple concurrent research sessions."""
        # Create multiple research plans
        plans = []
        for i in range(3):
            session_id = str(uuid.uuid4())
            plan = ResearchPlan(
                query=f"Test query {i+1}",
                session_id=session_id,
                tasks=[
                    ResearchTask(
                        task_id=f"task_{i}_1",
                        task_type="web_search",
                        description=f"Task for session {i+1}",
                        parameters={"query": f"query {i+1}"},
                        assigned_agent="research"
                    )
                ],
                estimated_duration_minutes=5
            )
            plans.append(plan)
        
        # Start all sessions
        sessions = []
        for plan in plans:
            session = await orchestrator.start_research_session(plan)
            sessions.append(session)
        
        # Verify all sessions are tracked
        active_sessions = orchestrator.list_active_sessions()
        assert len(active_sessions) == 3
        
        for session in sessions:
            assert session.session_id in active_sessions
        
        # Wait for sessions to complete
        await asyncio.sleep(2.0)
        
        # Verify all sessions completed successfully
        for session in sessions:
            updated_session = orchestrator.get_session(session.session_id)
            assert updated_session.status in [TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS]


class TestEnhancedAgentRegistry:
    """Test cases for enhanced agent registry."""
    
    @pytest.mark.asyncio
    async def test_agent_registration_and_lifecycle(self):
        """Test agent registration and lifecycle management."""
        registry = enhanced_registry
        await registry.start()
        
        try:
            # Register a mock agent
            mock_agent = MockAgent("test_agent")
            instance_id = await registry.register_agent_instance(
                mock_agent,
                max_concurrent_executions=2,
                capabilities=["test_capability"],
                tags={"test", "mock"}
            )
            
            assert instance_id is not None
            
            # Verify agent is registered
            instance = registry.get_agent_instance("test_agent")
            assert instance is not None
            assert instance.agent.name == "test_agent"
            assert instance.max_concurrent_executions == 2
            assert "test_capability" in instance.capabilities
            assert "test" in instance.tags
            
            # Test agent execution
            result = await registry.execute_agent("test_agent", test_param="test_value")
            assert result.success is True
            assert result.agent_name == "test_agent"
            
            # Verify metrics were updated
            metrics = registry.get_agent_metrics("test_agent")
            assert len(metrics) == 1
            assert metrics[0]["total_executions"] == 1
            assert metrics[0]["success_rate"] == 100.0
            
        finally:
            await registry.stop()
    
    @pytest.mark.asyncio
    async def test_load_balancing(self):
        """Test load balancing across multiple agent instances."""
        registry = enhanced_registry
        await registry.start()
        
        try:
            # Register multiple instances of the same agent type
            for i in range(3):
                mock_agent = MockAgent(f"load_test", execution_delay=0.1)
                await registry.register_agent_instance(
                    mock_agent,
                    max_concurrent_executions=1
                )
            
            # Execute multiple tasks concurrently
            tasks = []
            for i in range(6):
                task = asyncio.create_task(
                    registry.execute_agent("load_test", task_id=i)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)
            
            # Verify all tasks completed successfully
            for result in results:
                assert result.success is True
            
            # Verify load was distributed across instances
            metrics = registry.get_agent_metrics("load_test")
            assert len(metrics) == 3
            
            # Each instance should have executed at least one task
            total_executions = sum(m["total_executions"] for m in metrics)
            assert total_executions == 6
            
        finally:
            await registry.stop()
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self):
        """Test health monitoring functionality."""
        registry = enhanced_registry
        await registry.start()
        
        try:
            # Register an agent
            mock_agent = MockAgent("health_test")
            await registry.register_agent_instance(mock_agent)
            
            # Wait for health check
            await asyncio.sleep(1.5)
            
            # Verify health status
            instance = registry.get_agent_instance("health_test")
            assert instance is not None
            assert instance.is_healthy() is True
            assert instance.last_health_check is not None
            
            # Get registry status
            status = registry.get_registry_status()
            assert status["started"] is True
            assert status["total_instances"] == 1
            assert status["healthy_instances"] == 1
            assert status["available_instances"] == 1
            
        finally:
            await registry.stop()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])