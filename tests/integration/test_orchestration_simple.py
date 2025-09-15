"""
Simplified integration tests for agent orchestration system.
"""

import pytest
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any

from agent_scrivener.orchestration.orchestrator import (
    AgentOrchestrator, OrchestrationConfig
)
from agent_scrivener.orchestration.registry import enhanced_registry
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
                {
                    "source": {
                        "url": "https://example.com/article1",
                        "title": "Mock Article 1",
                        "source_type": "web"
                    },
                    "content": "Mock article content 1",
                    "confidence_score": 0.8
                }
            ]
        elif "analysis" in self.name:
            return [
                {
                    "topic": "Mock Topic 1",
                    "summary": "Mock insight summary 1",
                    "confidence_score": 0.85,
                    "supporting_evidence": ["Evidence 1", "Evidence 2"]
                }
            ]
        elif "drafting" in self.name:
            return "# Mock Research Document\n\nThis is a mock research document."
        elif "citation" in self.name:
            return [
                {"citation_id": "cite1", "text": "Mock citation 1"}
            ]
        else:
            return {"mock_result": f"Result from {self.name}"}


@pytest.mark.asyncio
async def test_basic_orchestration():
    """Test basic orchestration functionality."""
    # Create orchestrator
    config = OrchestrationConfig(
        max_concurrent_tasks=2,
        task_timeout_seconds=5,
        enable_parallel_execution=True
    )
    orchestrator = AgentOrchestrator(config)
    
    # Create and register mock agents
    research_agent = MockAgent("research", execution_delay=0.1)
    analysis_agent = MockAgent("analysis", execution_delay=0.1)
    
    agent_registry.register_agent(research_agent)
    agent_registry.register_agent(analysis_agent)
    
    try:
        # Create a simple research plan
        session_id = str(uuid.uuid4())
        tasks = [
            ResearchTask(
                task_id="web_research",
                task_type="web_search",
                description="Search web sources",
                parameters={"query": "test query"},
                assigned_agent="research"
            ),
            ResearchTask(
                task_id="data_analysis",
                task_type="content_analysis",
                description="Analyze collected data",
                parameters={"analysis_types": ["topic_modeling"]},
                dependencies=["web_research"],
                assigned_agent="analysis"
            )
        ]
        
        plan = ResearchPlan(
            query="Test research query",
            session_id=session_id,
            tasks=tasks,
            estimated_duration_minutes=5
        )
        
        # Start research session
        session = await orchestrator.start_research_session(plan)
        
        # Verify session was created
        assert session.session_id == session_id
        assert session.status == TaskStatus.IN_PROGRESS
        assert session.session_state == SessionState.RESEARCHING
        
        # Wait for tasks to complete
        max_wait = 10  # seconds
        start_time = datetime.now()
        
        while session.status == TaskStatus.IN_PROGRESS:
            if (datetime.now() - start_time).total_seconds() > max_wait:
                break
            await asyncio.sleep(0.1)
            session = orchestrator.get_session(session_id)
        
        # Verify tasks completed
        web_task = session.plan.get_task_by_id("web_research")
        analysis_task = session.plan.get_task_by_id("data_analysis")
        
        assert web_task.status == TaskStatus.COMPLETED
        assert analysis_task.status == TaskStatus.COMPLETED
        
        # Verify execution order (analysis should start after research)
        assert analysis_task.started_at >= web_task.completed_at
        
        # Verify agents were called
        assert research_agent.execution_count == 1
        assert analysis_agent.execution_count == 1
        
    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_progress_tracking():
    """Test progress tracking functionality."""
    config = OrchestrationConfig(max_concurrent_tasks=1, task_timeout_seconds=5)
    orchestrator = AgentOrchestrator(config)
    
    # Create mock agent
    mock_agent = MockAgent("test", execution_delay=0.2)
    agent_registry.register_agent(mock_agent)
    
    progress_updates = []
    
    def progress_callback(progress_data):
        progress_updates.append(progress_data.copy())
    
    try:
        # Create simple plan
        session_id = str(uuid.uuid4())
        plan = ResearchPlan(
            query="Test query",
            session_id=session_id,
            tasks=[
                ResearchTask(
                    task_id="task1",
                    task_type="test",
                    description="Test task",
                    parameters={},
                    assigned_agent="test"
                )
            ],
            estimated_duration_minutes=1
        )
        
        # Start session and register progress callback
        session = await orchestrator.start_research_session(plan)
        orchestrator.register_progress_callback(session_id, progress_callback)
        
        # Wait for completion
        await asyncio.sleep(1.0)
        
        # Verify progress updates were received
        assert len(progress_updates) > 0
        
        latest_progress = progress_updates[-1]
        assert "session_id" in latest_progress
        assert "progress_percentage" in latest_progress
        assert latest_progress["session_id"] == session_id
        
    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling and isolation."""
    config = OrchestrationConfig(max_concurrent_tasks=2, task_timeout_seconds=5)
    orchestrator = AgentOrchestrator(config)
    
    # Create agents - one that fails, one that succeeds
    good_agent = MockAgent("good", execution_delay=0.1)
    bad_agent = MockAgent("bad", execution_delay=0.1, should_fail=True)
    
    agent_registry.register_agent(good_agent)
    agent_registry.register_agent(bad_agent)
    
    try:
        # Create plan with both agents
        session_id = str(uuid.uuid4())
        plan = ResearchPlan(
            query="Test error handling",
            session_id=session_id,
            tasks=[
                ResearchTask(
                    task_id="good_task",
                    task_type="test",
                    description="Good task",
                    parameters={},
                    assigned_agent="good"
                ),
                ResearchTask(
                    task_id="bad_task",
                    task_type="test",
                    description="Bad task",
                    parameters={},
                    assigned_agent="bad"
                )
            ],
            estimated_duration_minutes=1
        )
        
        # Start session
        session = await orchestrator.start_research_session(plan)
        
        # Wait for completion
        await asyncio.sleep(2.0)
        
        # Verify results
        session = orchestrator.get_session(session_id)
        good_task = session.plan.get_task_by_id("good_task")
        bad_task = session.plan.get_task_by_id("bad_task")
        
        assert good_task.status == TaskStatus.COMPLETED
        assert bad_task.status == TaskStatus.FAILED
        assert bad_task.error_message is not None
        
    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_enhanced_registry():
    """Test enhanced agent registry functionality."""
    await enhanced_registry.start()
    
    try:
        # Register an agent
        mock_agent = MockAgent("registry_test")
        instance_id = await enhanced_registry.register_agent_instance(
            mock_agent,
            max_concurrent_executions=1,
            capabilities=["test_capability"]
        )
        
        assert instance_id is not None
        
        # Test agent execution
        result = await enhanced_registry.execute_agent("registry_test", test_param="value")
        assert result.success is True
        assert result.agent_name == "registry_test"
        
        # Verify metrics
        metrics = enhanced_registry.get_agent_metrics("registry_test")
        assert len(metrics) == 1
        assert metrics[0]["total_executions"] == 1
        assert metrics[0]["success_rate"] == 100.0
        
        # Test registry status
        status = enhanced_registry.get_registry_status()
        assert status["started"] is True
        assert status["total_instances"] == 1
        
    finally:
        await enhanced_registry.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])