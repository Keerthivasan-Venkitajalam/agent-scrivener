"""
Enhanced Agent Registry with Lifecycle Management for Agent Scrivener.

This module provides advanced agent registration, lifecycle management,
health monitoring, and load balancing capabilities.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..agents.base import BaseAgent, AgentResult
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AgentStatus(str, Enum):
    """Agent lifecycle status."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time_ms: float = 0.0
    average_execution_time_ms: float = 0.0
    last_execution_time: Optional[datetime] = None
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    
    def update_execution(self, success: bool, execution_time_ms: float, error: Optional[str] = None):
        """Update metrics after an execution."""
        self.total_executions += 1
        self.total_execution_time_ms += execution_time_ms
        self.average_execution_time_ms = self.total_execution_time_ms / self.total_executions
        self.last_execution_time = datetime.now()
        
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
            if error:
                self.last_error = error
                self.last_error_time = datetime.now()
    
    def get_success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100.0
    
    def get_error_rate(self) -> float:
        """Calculate error rate as a percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.failed_executions / self.total_executions) * 100.0


@dataclass
class AgentInstance:
    """Represents a registered agent instance with metadata."""
    agent: BaseAgent
    status: AgentStatus = AgentStatus.INITIALIZING
    registered_at: datetime = field(default_factory=datetime.now)
    last_health_check: Optional[datetime] = None
    health_check_interval_seconds: int = 60
    max_concurrent_executions: int = 1
    current_executions: int = 0
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    tags: Set[str] = field(default_factory=set)
    capabilities: List[str] = field(default_factory=list)
    
    def is_available(self) -> bool:
        """Check if agent is available for new tasks."""
        return (
            self.status == AgentStatus.READY and 
            self.current_executions < self.max_concurrent_executions
        )
    
    def is_healthy(self) -> bool:
        """Check if agent is healthy based on recent health checks."""
        if not self.last_health_check:
            return False
        
        time_since_check = datetime.now() - self.last_health_check
        return time_since_check.total_seconds() < (self.health_check_interval_seconds * 2)
    
    def start_execution(self):
        """Mark the start of a new execution."""
        self.current_executions += 1
        if self.current_executions >= self.max_concurrent_executions:
            self.status = AgentStatus.BUSY
    
    def end_execution(self, success: bool, execution_time_ms: float, error: Optional[str] = None):
        """Mark the end of an execution and update metrics."""
        self.current_executions = max(0, self.current_executions - 1)
        self.metrics.update_execution(success, execution_time_ms, error)
        
        if self.current_executions < self.max_concurrent_executions:
            self.status = AgentStatus.READY


class HealthChecker:
    """Monitors agent health and availability."""
    
    def __init__(self, check_interval_seconds: int = 30):
        self.check_interval_seconds = check_interval_seconds
        self.logger = get_logger(f"{__name__}.HealthChecker")
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def start(self, registry: 'EnhancedAgentRegistry'):
        """Start the health checking background task."""
        if self._running:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop(registry))
        self.logger.info("Health checker started")
    
    async def stop(self):
        """Stop the health checking background task."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Health checker stopped")
    
    async def _health_check_loop(self, registry: 'EnhancedAgentRegistry'):
        """Main health checking loop."""
        while self._running:
            try:
                await self._perform_health_checks(registry)
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.check_interval_seconds)
    
    async def _perform_health_checks(self, registry: 'EnhancedAgentRegistry'):
        """Perform health checks on all registered agents."""
        for agent_name, instances in registry._agents.items():
            for instance in instances:
                try:
                    await self._check_agent_health(agent_name, instance)
                except Exception as e:
                    self.logger.error(f"Health check failed for agent {agent_name}: {e}")
                    instance.status = AgentStatus.ERROR
    
    async def _check_agent_health(self, agent_name: str, instance: AgentInstance):
        """Check health of a single agent."""
        try:
            # Simple health check - could be enhanced with agent-specific checks
            start_time = time.time()
            
            # For now, just check if the agent is responsive
            # In a real implementation, this might call a health check method on the agent
            if hasattr(instance.agent, 'health_check'):
                await instance.agent.health_check()
            
            # Update health check timestamp
            instance.last_health_check = datetime.now()
            
            # If agent was in error state and health check passed, mark as ready
            if instance.status == AgentStatus.ERROR and instance.current_executions == 0:
                instance.status = AgentStatus.READY
            
            check_time_ms = (time.time() - start_time) * 1000
            self.logger.debug(f"Health check passed for {agent_name} in {check_time_ms:.2f}ms")
            
        except Exception as e:
            instance.status = AgentStatus.ERROR
            instance.metrics.last_error = str(e)
            instance.metrics.last_error_time = datetime.now()
            self.logger.warning(f"Health check failed for {agent_name}: {e}")


class LoadBalancer:
    """Balances load across multiple agent instances."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.LoadBalancer")
    
    def select_agent(self, agent_name: str, instances: List[AgentInstance]) -> Optional[AgentInstance]:
        """
        Select the best agent instance for a task.
        
        Args:
            agent_name: Name of the agent type
            instances: List of available instances
            
        Returns:
            Best agent instance or None if none available
        """
        # Filter to available and healthy instances
        available_instances = [
            instance for instance in instances 
            if instance.is_available() and instance.is_healthy()
        ]
        
        if not available_instances:
            self.logger.warning(f"No available instances for agent {agent_name}")
            return None
        
        # Select instance with lowest current load
        selected = min(available_instances, key=lambda x: x.current_executions)
        
        self.logger.debug(f"Selected instance for {agent_name} with {selected.current_executions} current executions")
        return selected


class EnhancedAgentRegistry:
    """
    Enhanced agent registry with lifecycle management, health monitoring,
    and load balancing capabilities.
    """
    
    def __init__(self, health_check_interval_seconds: int = 30):
        self._agents: Dict[str, List[AgentInstance]] = {}
        self._agent_types: Dict[str, type] = {}
        self.health_checker = HealthChecker(health_check_interval_seconds)
        self.load_balancer = LoadBalancer()
        self.logger = get_logger(__name__)
        self._started = False
    
    async def start(self):
        """Start the registry and background services."""
        if self._started:
            return
        
        await self.health_checker.start(self)
        self._started = True
        self.logger.info("Enhanced agent registry started")
    
    async def stop(self):
        """Stop the registry and background services."""
        if not self._started:
            return
        
        await self.health_checker.stop()
        
        # Shutdown all agents
        for agent_name in list(self._agents.keys()):
            await self.shutdown_agent_type(agent_name)
        
        self._started = False
        self.logger.info("Enhanced agent registry stopped")
    
    def register_agent_type(self, agent_class: type, max_instances: int = 1):
        """
        Register an agent type for dynamic instantiation.
        
        Args:
            agent_class: Agent class to register
            max_instances: Maximum number of instances to create
        """
        agent_name = agent_class.__name__.lower().replace('agent', '')
        self._agent_types[agent_name] = agent_class
        
        if agent_name not in self._agents:
            self._agents[agent_name] = []
        
        self.logger.info(f"Registered agent type {agent_name} with max {max_instances} instances")
    
    async def register_agent_instance(
        self, 
        agent: BaseAgent, 
        max_concurrent_executions: int = 1,
        capabilities: Optional[List[str]] = None,
        tags: Optional[Set[str]] = None
    ) -> str:
        """
        Register a specific agent instance.
        
        Args:
            agent: Agent instance to register
            max_concurrent_executions: Maximum concurrent executions
            capabilities: List of agent capabilities
            tags: Set of tags for categorization
            
        Returns:
            Instance ID
        """
        agent_name = agent.name
        
        instance = AgentInstance(
            agent=agent,
            max_concurrent_executions=max_concurrent_executions,
            capabilities=capabilities or [],
            tags=tags or set()
        )
        
        if agent_name not in self._agents:
            self._agents[agent_name] = []
        
        self._agents[agent_name].append(instance)
        
        # Initialize agent
        await self._initialize_agent_instance(instance)
        
        instance_id = f"{agent_name}_{len(self._agents[agent_name])}"
        self.logger.info(f"Registered agent instance {instance_id}")
        
        return instance_id
    
    async def _initialize_agent_instance(self, instance: AgentInstance):
        """Initialize an agent instance."""
        try:
            instance.status = AgentStatus.INITIALIZING
            
            # Call initialization if agent supports it
            if hasattr(instance.agent, 'initialize'):
                await instance.agent.initialize()
            
            instance.status = AgentStatus.READY
            instance.last_health_check = datetime.now()
            
            self.logger.info(f"Initialized agent {instance.agent.name}")
            
        except Exception as e:
            instance.status = AgentStatus.ERROR
            instance.metrics.last_error = str(e)
            instance.metrics.last_error_time = datetime.now()
            self.logger.error(f"Failed to initialize agent {instance.agent.name}: {e}")
    
    def get_agent_instance(self, agent_name: str) -> Optional[AgentInstance]:
        """
        Get the best available agent instance for a task.
        
        Args:
            agent_name: Name of the agent type
            
        Returns:
            Best available agent instance
        """
        instances = self._agents.get(agent_name, [])
        if not instances:
            self.logger.warning(f"No instances registered for agent {agent_name}")
            return None
        
        return self.load_balancer.select_agent(agent_name, instances)
    
    def list_agent_types(self) -> List[str]:
        """Get list of registered agent types."""
        return list(self._agents.keys())
    
    def list_agent_instances(self, agent_name: str) -> List[AgentInstance]:
        """Get all instances for an agent type."""
        return self._agents.get(agent_name, [])
    
    async def execute_agent(self, agent_name: str, **kwargs) -> AgentResult:
        """
        Execute an agent with load balancing and metrics tracking.
        
        Args:
            agent_name: Name of the agent type
            **kwargs: Parameters for agent execution
            
        Returns:
            AgentResult: Execution result
        """
        instance = self.get_agent_instance(agent_name)
        if not instance:
            return AgentResult(
                success=False,
                error=f"No available instances for agent {agent_name}",
                timestamp=datetime.now(),
                agent_name=agent_name
            )
        
        # Track execution
        instance.start_execution()
        start_time = time.time()
        
        try:
            result = await instance.agent.execute(**kwargs)
            execution_time_ms = (time.time() - start_time) * 1000
            
            instance.end_execution(
                success=result.success,
                execution_time_ms=execution_time_ms,
                error=result.error if not result.success else None
            )
            
            return result
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            instance.end_execution(
                success=False,
                execution_time_ms=execution_time_ms,
                error=str(e)
            )
            
            return AgentResult(
                success=False,
                error=str(e),
                timestamp=datetime.now(),
                agent_name=agent_name,
                execution_time_ms=int(execution_time_ms)
            )
    
    def get_agent_metrics(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get metrics for all instances of an agent type."""
        instances = self._agents.get(agent_name, [])
        
        metrics = []
        for i, instance in enumerate(instances):
            metrics.append({
                "instance_id": f"{agent_name}_{i+1}",
                "status": instance.status.value,
                "is_available": instance.is_available(),
                "is_healthy": instance.is_healthy(),
                "current_executions": instance.current_executions,
                "max_concurrent_executions": instance.max_concurrent_executions,
                "total_executions": instance.metrics.total_executions,
                "success_rate": instance.metrics.get_success_rate(),
                "error_rate": instance.metrics.get_error_rate(),
                "average_execution_time_ms": instance.metrics.average_execution_time_ms,
                "last_execution_time": instance.metrics.last_execution_time.isoformat() if instance.metrics.last_execution_time else None,
                "last_error": instance.metrics.last_error,
                "last_error_time": instance.metrics.last_error_time.isoformat() if instance.metrics.last_error_time else None,
                "registered_at": instance.registered_at.isoformat(),
                "capabilities": instance.capabilities,
                "tags": list(instance.tags)
            })
        
        return metrics
    
    async def shutdown_agent_type(self, agent_name: str):
        """Shutdown all instances of an agent type."""
        instances = self._agents.get(agent_name, [])
        
        for instance in instances:
            instance.status = AgentStatus.SHUTTING_DOWN
            
            # Call shutdown if agent supports it
            if hasattr(instance.agent, 'shutdown'):
                try:
                    await instance.agent.shutdown()
                except Exception as e:
                    self.logger.error(f"Error shutting down agent {instance.agent.name}: {e}")
            
            instance.status = AgentStatus.SHUTDOWN
        
        # Remove from registry
        if agent_name in self._agents:
            del self._agents[agent_name]
        
        self.logger.info(f"Shutdown agent type {agent_name}")
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get overall registry status and statistics."""
        total_instances = sum(len(instances) for instances in self._agents.values())
        healthy_instances = sum(
            len([i for i in instances if i.is_healthy()]) 
            for instances in self._agents.values()
        )
        available_instances = sum(
            len([i for i in instances if i.is_available()]) 
            for instances in self._agents.values()
        )
        
        return {
            "started": self._started,
            "total_agent_types": len(self._agents),
            "total_instances": total_instances,
            "healthy_instances": healthy_instances,
            "available_instances": available_instances,
            "agent_types": list(self._agents.keys()),
            "health_check_interval_seconds": self.health_checker.check_interval_seconds
        }


# Global enhanced registry instance
enhanced_registry = EnhancedAgentRegistry()