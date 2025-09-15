"""
Base agent interface and common utilities for Agent Scrivener.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentResult(BaseModel):
    """Standard result format for all agents."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime
    agent_name: str
    execution_time_ms: Optional[int] = None


class BaseAgent(ABC):
    """
    Abstract base class for all Agent Scrivener agents.
    
    Provides common functionality including error handling, logging,
    and standardized result formatting.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"agent_scrivener.agents.{name}")
    
    @abstractmethod
    async def execute(self, **kwargs) -> AgentResult:
        """
        Execute the agent's primary function.
        
        Args:
            **kwargs: Agent-specific parameters
            
        Returns:
            AgentResult: Standardized result object
        """
        pass
    
    async def _execute_with_timing(self, operation, **kwargs) -> AgentResult:
        """
        Execute an operation with timing and error handling.
        
        Args:
            operation: Async function to execute
            **kwargs: Parameters for the operation
            
        Returns:
            AgentResult: Result with timing information
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting {self.name} execution")
            result = await operation(**kwargs)
            
            end_time = datetime.now()
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            self.logger.info(f"{self.name} completed successfully in {execution_time}ms")
            
            return AgentResult(
                success=True,
                data=result,
                timestamp=end_time,
                agent_name=self.name,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            self.logger.error(f"{self.name} failed after {execution_time}ms: {str(e)}")
            
            return AgentResult(
                success=False,
                error=str(e),
                timestamp=end_time,
                agent_name=self.name,
                execution_time_ms=execution_time
            )
    
    def validate_input(self, data: Dict[str, Any], required_fields: List[str]) -> bool:
        """
        Validate that required fields are present in input data.
        
        Args:
            data: Input data dictionary
            required_fields: List of required field names
            
        Returns:
            bool: True if all required fields are present
            
        Raises:
            ValueError: If required fields are missing
        """
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        return True


class AgentRegistry:
    """
    Registry for managing agent instances and lifecycle.
    """
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger("agent_scrivener.registry")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent instance.
        
        Args:
            agent: Agent instance to register
        """
        self._agents[agent.name] = agent
        self.logger.info(f"Registered agent: {agent.name}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """
        Retrieve an agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            BaseAgent: Agent instance or None if not found
        """
        return self._agents.get(name)
    
    def list_agents(self) -> List[str]:
        """
        Get list of registered agent names.
        
        Returns:
            List[str]: List of agent names
        """
        return list(self._agents.keys())
    
    async def execute_agent(self, name: str, **kwargs) -> AgentResult:
        """
        Execute an agent by name.
        
        Args:
            name: Agent name
            **kwargs: Parameters for agent execution
            
        Returns:
            AgentResult: Execution result
            
        Raises:
            ValueError: If agent is not found
        """
        agent = self.get_agent(name)
        if not agent:
            raise ValueError(f"Agent not found: {name}")
        
        return await agent.execute(**kwargs)


# Global agent registry instance
agent_registry = AgentRegistry()