"""
Pytest configuration and fixtures for Agent Scrivener tests.
"""

import pytest
import asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from agent_scrivener.models.core import Source, SourceType, ResearchSession, ResearchPlan
from agent_scrivener.agents.base import BaseAgent, AgentRegistry


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_source() -> Source:
    """Create a sample Source object for testing."""
    return Source(
        url="https://example.com/article",
        title="Sample Article",
        author="Test Author",
        source_type=SourceType.WEB,
        metadata={"domain": "example.com"}
    )


@pytest.fixture
def sample_research_plan() -> ResearchPlan:
    """Create a sample ResearchPlan for testing."""
    return ResearchPlan(
        query="Test research query",
        session_id="test_session_001",
        estimated_duration_minutes=30
    )


@pytest.fixture
def sample_research_session(sample_research_plan: ResearchPlan) -> ResearchSession:
    """Create a sample ResearchSession for testing."""
    return ResearchSession(
        session_id="test_session_001",
        original_query="Test research query",
        plan=sample_research_plan
    )


@pytest.fixture
def mock_agentcore_browser():
    """Mock AgentCore browser tool."""
    mock = AsyncMock()
    mock.navigate.return_value = {"success": True, "content": "Sample content"}
    mock.extract_content.return_value = {"text": "Extracted text", "title": "Page Title"}
    return mock


@pytest.fixture
def mock_agentcore_gateway():
    """Mock AgentCore gateway tool."""
    mock = AsyncMock()
    mock.query_api.return_value = {"results": [], "status": "success"}
    return mock


@pytest.fixture
def mock_agentcore_code_interpreter():
    """Mock AgentCore code interpreter tool."""
    mock = AsyncMock()
    mock.execute_code.return_value = {"output": "Analysis complete", "success": True}
    return mock


@pytest.fixture
def mock_agent_registry():
    """Create a mock agent registry for testing."""
    registry = AgentRegistry()
    
    # Create mock agents
    mock_research_agent = AsyncMock(spec=BaseAgent)
    mock_research_agent.name = "research_agent"
    mock_research_agent.execute.return_value = MagicMock(success=True, data={"articles": []})
    
    mock_api_agent = AsyncMock(spec=BaseAgent)
    mock_api_agent.name = "api_agent"
    mock_api_agent.execute.return_value = MagicMock(success=True, data={"papers": []})
    
    # Register mock agents
    registry.register_agent(mock_research_agent)
    registry.register_agent(mock_api_agent)
    
    return registry


class MockAgent(BaseAgent):
    """Mock agent implementation for testing."""
    
    def __init__(self, name: str, should_fail: bool = False):
        super().__init__(name)
        self.should_fail = should_fail
        self.execution_count = 0
    
    async def execute(self, **kwargs):
        """Mock execute method."""
        self.execution_count += 1
        
        if self.should_fail:
            raise Exception(f"Mock failure in {self.name}")
        
        return await self._execute_with_timing(self._mock_operation, **kwargs)
    
    async def _mock_operation(self, **kwargs):
        """Mock operation that simulates work."""
        await asyncio.sleep(0.01)  # Simulate some work
        return {"result": f"Mock result from {self.name}", "kwargs": kwargs}


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return MockAgent("test_agent")


@pytest.fixture
def failing_mock_agent():
    """Create a mock agent that fails for testing error handling."""
    return MockAgent("failing_agent", should_fail=True)