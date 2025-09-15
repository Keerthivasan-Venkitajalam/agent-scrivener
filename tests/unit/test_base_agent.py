"""
Unit tests for base agent functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock

from agent_scrivener.agents.base import BaseAgent, AgentRegistry, AgentResult


class TestBaseAgent:
    """Test cases for BaseAgent class."""
    
    @pytest.mark.asyncio
    async def test_execute_with_timing_success(self, mock_agent):
        """Test successful execution with timing."""
        result = await mock_agent.execute(test_param="value")
        
        assert result.success is True
        assert result.agent_name == "test_agent"
        assert result.execution_time_ms is not None
        assert result.execution_time_ms > 0
        assert result.data["result"] == "Mock result from test_agent"
        assert result.data["kwargs"]["test_param"] == "value"
    
    @pytest.mark.asyncio
    async def test_execute_with_timing_failure(self, failing_mock_agent):
        """Test execution failure handling."""
        result = await failing_mock_agent.execute()
        
        assert result.success is False
        assert result.agent_name == "failing_agent"
        assert result.execution_time_ms is not None
        assert result.error == "Mock failure in failing_agent"
        assert result.data is None
    
    def test_validate_input_success(self, mock_agent):
        """Test successful input validation."""
        data = {"field1": "value1", "field2": "value2"}
        required_fields = ["field1", "field2"]
        
        result = mock_agent.validate_input(data, required_fields)
        assert result is True
    
    def test_validate_input_missing_fields(self, mock_agent):
        """Test input validation with missing fields."""
        data = {"field1": "value1"}
        required_fields = ["field1", "field2", "field3"]
        
        with pytest.raises(ValueError) as exc_info:
            mock_agent.validate_input(data, required_fields)
        
        assert "Missing required fields: field2, field3" in str(exc_info.value)
    
    def test_validate_input_empty_fields(self, mock_agent):
        """Test input validation with empty required fields."""
        data = {"field1": "value1", "field2": ""}
        required_fields = ["field1", "field2"]
        
        # Empty string should not raise error in base validation
        # (specific validation should be done in subclasses)
        result = mock_agent.validate_input(data, required_fields)
        assert result is True


class TestAgentRegistry:
    """Test cases for AgentRegistry class."""
    
    def test_register_agent(self):
        """Test registering an agent."""
        registry = AgentRegistry()
        mock_agent = AsyncMock(spec=BaseAgent)
        mock_agent.name = "test_agent"
        
        registry.register_agent(mock_agent)
        
        assert "test_agent" in registry.list_agents()
        assert registry.get_agent("test_agent") == mock_agent
    
    def test_get_nonexistent_agent(self):
        """Test getting a non-existent agent."""
        registry = AgentRegistry()
        
        result = registry.get_agent("nonexistent")
        assert result is None
    
    def test_list_agents(self):
        """Test listing registered agents."""
        registry = AgentRegistry()
        
        # Initially empty
        assert registry.list_agents() == []
        
        # Add agents
        agent1 = AsyncMock(spec=BaseAgent)
        agent1.name = "agent1"
        agent2 = AsyncMock(spec=BaseAgent)
        agent2.name = "agent2"
        
        registry.register_agent(agent1)
        registry.register_agent(agent2)
        
        agents = registry.list_agents()
        assert len(agents) == 2
        assert "agent1" in agents
        assert "agent2" in agents
    
    @pytest.mark.asyncio
    async def test_execute_agent_success(self):
        """Test successful agent execution through registry."""
        registry = AgentRegistry()
        
        mock_agent = AsyncMock(spec=BaseAgent)
        mock_agent.name = "test_agent"
        mock_result = AgentResult(
            success=True,
            data={"result": "success"},
            timestamp=pytest.approx(asyncio.get_event_loop().time(), abs=1),
            agent_name="test_agent"
        )
        mock_agent.execute.return_value = mock_result
        
        registry.register_agent(mock_agent)
        
        result = await registry.execute_agent("test_agent", param="value")
        
        assert result.success is True
        assert result.data["result"] == "success"
        mock_agent.execute.assert_called_once_with(param="value")
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_agent(self):
        """Test executing a non-existent agent."""
        registry = AgentRegistry()
        
        with pytest.raises(ValueError) as exc_info:
            await registry.execute_agent("nonexistent")
        
        assert "Agent not found: nonexistent" in str(exc_info.value)


class TestAgentResult:
    """Test cases for AgentResult model."""
    
    def test_successful_result_creation(self):
        """Test creating a successful AgentResult."""
        from datetime import datetime
        
        timestamp = datetime.now()
        result = AgentResult(
            success=True,
            data={"key": "value"},
            timestamp=timestamp,
            agent_name="test_agent",
            execution_time_ms=100
        )
        
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.timestamp == timestamp
        assert result.agent_name == "test_agent"
        assert result.execution_time_ms == 100
        assert result.error is None
    
    def test_failed_result_creation(self):
        """Test creating a failed AgentResult."""
        from datetime import datetime
        
        timestamp = datetime.now()
        result = AgentResult(
            success=False,
            error="Something went wrong",
            timestamp=timestamp,
            agent_name="test_agent",
            execution_time_ms=50
        )
        
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.timestamp == timestamp
        assert result.agent_name == "test_agent"
        assert result.execution_time_ms == 50
        assert result.data is None