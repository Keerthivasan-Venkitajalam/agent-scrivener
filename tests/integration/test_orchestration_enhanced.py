"""
Enhanced integration tests for the complete orchestration system.
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
    SessionState
)


class MockAgent(BaseAgent):
    """Enhanced mock agent for testing."""
    
    def __init__(self, name: str, execution_delay: float = 0.1, should_fail: bool = False, failure_rate: float = 0.0):
        super().__init__(name)
        self.execution_delay = execution_delay
        self.should_fail = should_fail
        self.failure_rate = failure_rate
        self.execution_count = 0
        self.last_kwargs = {}
    
    async def execute(self, **kwargs) -> AgentResult:
        """Mock execution with configurable behavior."""
        self.execution_count += 1
        self.last_kwargs = kwargs
        
        # Simulate processing time
        await asyncio.sleep(self.execution_delay)
        
        # Determine if this execution should fail
        should_fail = self.should_fail
        if not should_fail and self.failure_rate > 0:
            import random
            should_fail = random.random() < self.failure_rate
        
        if should_fail:
            return AgentResult(
                success=False,
                error=f"Mock failure from {self.name}",
                timestamp=datetime.now(),
                agent_name=self.name,
                execution_time_ms=int(self.execution_delay * 1000)
            )
        
        # Return mock data
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
        return {"result": f"Mock result from {self.name}", "execution_count": self.execution_count}


@pytest.mark.asyncio
async def test_enhanced_orchestration_features():
    """Test enhanced orchestration features including prioritization and metrics."""
    config = OrchestrationConfig(
        max_concurrent_tasks=3,
        task_timeout_seconds=10,
        enable_parallel_execution=True
    )
    orchestrator = AgentOrchestrator(config)
    
    # Create agents with different performance characteristics
    fast_agent = MockAgent("fast", execution_delay=0.1)
    slow_agent = MockAgent("slow", execution_delay=0.3)
    unreliable_agent = MockAgent("unreliable", execution_delay=0.2, failure_rate=0.3)
    
    agent_registry.register_agent(fast_agent)
    agent_registry.register_agent(slow_agent)
    agent_registry.register_agent(unreliable_agent)
    
    try:
        # Create a complex research plan
        session_id = str(uuid.uuid4())
        tasks = [
            ResearchTask(
                task_id="fast_task_1",
                task_type="web_search",
                description="Fast task 1",
                parameters={"query": "test"},
                assigned_agent="fast"
            ),
            ResearchTask(
                task_id="fast_task_2",
                task_type="web_search",
                description="Fast task 2",
                parameters={"query": "test"},
                assigned_agent="fast"
            ),
            ResearchTask(
                task_id="slow_task",
                task_type="content_analysis",
                description="Slow task",
                parameters={"data": "test"},
                dependencies=["fast_task_1"],
                assigned_agent="slow"
            ),
            ResearchTask(
                task_id="unreliable_task",
                task_type="document_generation",
                description="Unreliable task",
                parameters={"content": "test"},
                dependencies=["fast_task_2"],
                assigned_agent="unreliable"
            ),
            ResearchTask(
                task_id="final_task",
                task_type="citation_formatting",
                description="Final task",
                parameters={"citations": []},
                dependencies=["slow_task", "unreliable_task"],
                assigned_agent="fast"
            )
        ]
        
        plan = ResearchPlan(
            query="Enhanced orchestration test",
            session_id=session_id,
            tasks=tasks,
            estimated_duration_minutes=10
        )
        
        # Start session
        session = await orchestrator.start_research_session(plan)
        
        # Monitor progress
        progress_updates = []
        def progress_callback(progress_data):
            progress_updates.append(progress_data.copy())
        
        orchestrator.register_progress_callback(session_id, progress_callback)
        
        # Wait for completion
        max_wait = 15  # seconds
        start_time = datetime.now()
        
        while session.status == TaskStatus.IN_PROGRESS:
            if (datetime.now() - start_time).total_seconds() > max_wait:
                break
            await asyncio.sleep(0.2)
            session = orchestrator.get_session(session_id)
        
        # Verify orchestrator status
        status = orchestrator.get_orchestrator_status()
        assert "active_sessions" in status
        assert "load_balancing" in status
        assert "session_states" in status
        
        # Verify session diagnostics
        diagnostics = orchestrator.get_session_diagnostics(session_id)
        assert diagnostics is not None
        assert "dependency_graph" in diagnostics
        assert "performance" in diagnostics
        
        # Verify enhanced results aggregation
        results = await orchestrator.get_session_results(session_id)
        assert results is not None
        assert "quality" in results
        assert "performance" in results
        assert "agent_contributions" in results
        
        # Check that fast agent was prioritized
        agent_contributions = results["agent_contributions"]
        assert "fast" in agent_contributions
        assert "slow" in agent_contributions
        
        # Verify progress tracking worked
        assert len(progress_updates) > 0
        
        # Verify task prioritization worked (fast tasks should complete first)
        fast_task_1 = session.plan.get_task_by_id("fast_task_1")
        fast_task_2 = session.plan.get_task_by_id("fast_task_2")
        slow_task = session.plan.get_task_by_id("slow_task")
        
        assert fast_task_1.status == TaskStatus.COMPLETED
        assert fast_task_2.status == TaskStatus.COMPLETED
        
        # Slow task should complete after fast tasks
        if slow_task.status == TaskStatus.COMPLETED:
            assert slow_task.started_at >= fast_task_1.completed_at
        
    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_session_pause_resume():
    """Test session pause and resume functionality."""
    config = OrchestrationConfig(
        max_concurrent_tasks=1,
        task_timeout_seconds=10
    )
    orchestrator = AgentOrchestrator(config)
    
    # Create a slow agent for testing pause/resume
    slow_agent = MockAgent("slow", execution_delay=2.0)
    agent_registry.register_agent(slow_agent)
    
    try:
        # Create a simple plan
        session_id = str(uuid.uuid4())
        plan = ResearchPlan(
            query="Pause resume test",
            session_id=session_id,
            tasks=[
                ResearchTask(
                    task_id="slow_task",
                    task_type="test",
                    description="Slow task",
                    parameters={},
                    assigned_agent="slow"
                )
            ],
            estimated_duration_minutes=5
        )
        
        # Start session
        session = await orchestrator.start_research_session(plan)
        
        # Wait a bit for task to start
        await asyncio.sleep(0.5)
        
        # Pause session
        paused = await orchestrator.pause_session(session_id)
        assert paused is True
        
        # Verify session is paused
        session = orchestrator.get_session(session_id)
        assert session.status == TaskStatus.CANCELLED
        
        # Resume session
        resumed = await orchestrator.resume_session(session_id)
        assert resumed is True
        
        # Verify session is resumed
        session = orchestrator.get_session(session_id)
        assert session.status == TaskStatus.IN_PROGRESS
        
    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_concurrent_session_handling():
    """Test handling of multiple concurrent sessions."""
    config = OrchestrationConfig(
        max_concurrent_tasks=4,
        task_timeout_seconds=5
    )
    orchestrator = AgentOrchestrator(config)
    
    # Create test agents
    test_agent = MockAgent("test", execution_delay=0.2)
    agent_registry.register_agent(test_agent)
    
    try:
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session_id = str(uuid.uuid4())
            plan = ResearchPlan(
                query=f"Concurrent test {i+1}",
                session_id=session_id,
                tasks=[
                    ResearchTask(
                        task_id=f"task_{i}_{j}",
                        task_type="test",
                        description=f"Task {j} for session {i+1}",
                        parameters={"session": i+1, "task": j+1},
                        assigned_agent="test"
                    )
                    for j in range(2)
                ],
                estimated_duration_minutes=2
            )
            
            session = await orchestrator.start_research_session(plan)
            sessions.append(session)
        
        # Verify all sessions are active
        active_sessions = orchestrator.list_active_sessions()
        assert len(active_sessions) == 3
        
        # Wait for completion
        await asyncio.sleep(3.0)
        
        # Verify orchestrator status shows all sessions
        status = orchestrator.get_orchestrator_status()
        assert status["active_sessions"] == 3
        
        # Verify load balancing info
        load_info = orchestrator.task_dispatcher.get_load_balancing_info()
        assert "active_executions" in load_info
        assert "agent_metrics" in load_info
        
        # Check that all sessions made progress
        for session in sessions:
            updated_session = orchestrator.get_session(session.session_id)
            completed_tasks = len([t for t in updated_session.plan.tasks if t.status == TaskStatus.COMPLETED])
            assert completed_tasks > 0
        
    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_enhanced_registry_integration():
    """Test integration with enhanced agent registry."""
    await enhanced_registry.start()
    
    try:
        # Register multiple instances of the same agent type
        for i in range(2):
            mock_agent = MockAgent(f"load_balanced", execution_delay=0.1)
            await enhanced_registry.register_agent_instance(
                mock_agent,
                max_concurrent_executions=1,
                capabilities=["test"]
            )
        
        # Test load balancing
        results = []
        for i in range(4):
            result = await enhanced_registry.execute_agent("load_balanced", task_id=i)
            results.append(result)
        
        # Verify all executions succeeded
        for result in results:
            assert result.success is True
        
        # Verify metrics were collected
        metrics = enhanced_registry.get_agent_metrics("load_balanced")
        assert len(metrics) == 2  # Two instances
        
        total_executions = sum(m["total_executions"] for m in metrics)
        assert total_executions == 4
        
        # Verify registry status
        status = enhanced_registry.get_registry_status()
        assert status["started"] is True
        assert status["total_instances"] == 2
        
    finally:
        await enhanced_registry.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])