"""
Agent Orchestration System for Agent Scrivener.

This package provides comprehensive orchestration capabilities including:
- Multi-agent coordination and task dispatching
- Enhanced agent registry with lifecycle management
- Progress tracking and status reporting
- Result aggregation and session management
"""

from .orchestrator import (
    AgentOrchestrator,
    OrchestrationConfig,
    ProgressTracker,
    TaskDispatcher,
    ResultAggregator,
    orchestrator
)

from .registry import (
    EnhancedAgentRegistry,
    AgentInstance,
    AgentStatus,
    AgentMetrics,
    HealthChecker,
    LoadBalancer,
    enhanced_registry
)

__all__ = [
    # Orchestrator components
    'AgentOrchestrator',
    'OrchestrationConfig', 
    'ProgressTracker',
    'TaskDispatcher',
    'ResultAggregator',
    'orchestrator',
    
    # Registry components
    'EnhancedAgentRegistry',
    'AgentInstance',
    'AgentStatus',
    'AgentMetrics',
    'HealthChecker',
    'LoadBalancer',
    'enhanced_registry'
]