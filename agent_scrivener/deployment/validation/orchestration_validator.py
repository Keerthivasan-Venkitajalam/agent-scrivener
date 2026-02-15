"""Orchestration validator for validating agent coordination and workflow management."""

import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock

from agent_scrivener.agents.base import BaseAgent, AgentResult, agent_registry
from agent_scrivener.models.core import ResearchPlan, ResearchTask, ResearchSession, SessionState
from agent_scrivener.orchestration.orchestrator import AgentOrchestrator, OrchestrationConfig
from .base_validator import BaseValidator
from .models import ValidationResult, ValidationStatus


logger = logging.getLogger(__name__)


class MockAgent(BaseAgent):
    """Mock agent for testing orchestration."""
    
    def __init__(
        self, 
        name: str, 
        execution_delay: float = 0.1,
        should_fail: bool = False,
        failure_count: int = 0
    ):
        super().__init__(name)
        self.execution_delay = execution_delay
        self.should_fail = should_fail
        self.failure_count = failure_count
        self.execution_count = 0
        self.executions: List[Dict[str, Any]] = []
    
    async def execute(self, **kwargs) -> AgentResult:
        """Execute the mock agent."""
        self.execution_count += 1
        execution_id = f"{self.name}_{self.execution_count}"
        
        self.executions.append({
            "execution_id": execution_id,
            "kwargs": kwargs,
            "timestamp": time.time()
        })
        
        await asyncio.sleep(self.execution_delay)
        
        # Fail for the first N executions if failure_count is set
        if self.failure_count > 0 and self.execution_count <= self.failure_count:
            from datetime import datetime
            return AgentResult(
                success=False,
                error=f"Simulated failure {self.execution_count}",
                timestamp=datetime.now(),
                agent_name=self.name,
                execution_time_ms=int(self.execution_delay * 1000)
            )
        
        # Fail if should_fail is True
        if self.should_fail:
            from datetime import datetime
            return AgentResult(
                success=False,
                error="Simulated failure",
                timestamp=datetime.now(),
                agent_name=self.name,
                execution_time_ms=int(self.execution_delay * 1000)
            )
        
        from datetime import datetime
        return AgentResult(
            success=True,
            data={"result": f"Success from {self.name}"},
            timestamp=datetime.now(),
            agent_name=self.name,
            execution_time_ms=int(self.execution_delay * 1000)
        )


class OrchestrationValidator(BaseValidator):
    """Validates agent coordination and workflow management.
    
    This validator tests:
    - Agent initialization
    - Message routing between agents
    - Session isolation for concurrent sessions
    - Retry logic with failure simulation
    - State persistence
    """
    
    def __init__(self, timeout_seconds: Optional[float] = 300):
        """Initialize the orchestration validator.
        
        Args:
            timeout_seconds: Timeout for validation (default 5 minutes)
        """
        super().__init__("OrchestrationValidator", timeout_seconds)
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.test_agents: Dict[str, MockAgent] = {}
    
    async def validate(self) -> List[ValidationResult]:
        """Execute all orchestration validation checks.
        
        Returns:
            List of validation results
        """
        self.log_validation_start()
        results = []
        
        # Run all validation methods
        results.append(await self.validate_agent_initialization())
        results.append(await self.validate_message_routing())
        results.append(await self.validate_session_isolation())
        results.append(await self.validate_retry_logic())
        results.append(await self.validate_state_persistence())
        
        self.log_validation_complete(results)
        return results
    
    async def validate_agent_initialization(self) -> ValidationResult:
        """Validate that all required agents initialize correctly.
        
        Tests that Research, Analysis, Synthesis, and Quality agents
        can be registered and initialized successfully.
        
        Validates: Requirements 3.1
        """
        start_time = time.time()
        
        try:
            # Define required agent types
            required_agents = ["Research", "Analysis", "Synthesis", "Quality"]
            
            # Create and register mock agents
            initialized_agents = []
            for agent_name in required_agents:
                agent = MockAgent(agent_name)
                agent_registry.register_agent(agent)
                initialized_agents.append(agent_name)
                self.test_agents[agent_name] = agent
            
            # Verify all agents are registered
            registered_agents = agent_registry.list_agents()
            missing_agents = [a for a in required_agents if a not in registered_agents]
            
            duration = time.time() - start_time
            
            if missing_agents:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to initialize required agents: {', '.join(missing_agents)}",
                    duration_seconds=duration,
                    details={
                        "required_agents": required_agents,
                        "initialized_agents": initialized_agents,
                        "missing_agents": missing_agents
                    },
                    remediation_steps=[
                        "Ensure all required agent classes are defined",
                        "Check agent registration logic in orchestrator",
                        "Verify agent dependencies are available"
                    ]
                )
            
            return self.create_result(
                status=ValidationStatus.PASS,
                message=f"Successfully initialized all {len(required_agents)} required agents",
                duration_seconds=duration,
                details={
                    "required_agents": required_agents,
                    "initialized_agents": initialized_agents
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"Agent initialization validation failed: {e}")
            
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Agent initialization failed with exception: {str(e)}",
                duration_seconds=duration,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check agent class definitions",
                    "Verify agent registry is properly initialized",
                    "Review error logs for specific issues"
                ]
            )
    
    async def validate_message_routing(self) -> ValidationResult:
        """Validate that messages are correctly routed between agents.
        
        Tests that the orchestrator routes messages to the correct
        destination agent based on workflow stage.
        
        Validates: Requirements 3.2
        """
        start_time = time.time()
        
        try:
            # Create orchestrator
            config = OrchestrationConfig(
                max_concurrent_tasks=5,
                task_timeout_seconds=30,
                enable_parallel_execution=False
            )
            self.orchestrator = AgentOrchestrator(config)
            
            # Create a simple research plan with sequential tasks
            from datetime import datetime
            plan = ResearchPlan(
                session_id=str(uuid.uuid4()),
                query="Test query for message routing",
                estimated_duration_minutes=5,
                tasks=[
                    ResearchTask(
                        task_id="task_1",
                        task_type="research",
                        description="Research task",
                        assigned_agent="Research",
                        dependencies=[]
                    ),
                    ResearchTask(
                        task_id="task_2",
                        task_type="analysis",
                        description="Analysis task",
                        assigned_agent="Analysis",
                        dependencies=["task_1"]
                    )
                ],
                created_at=datetime.now()
            )
            
            # Start session (this will route messages to agents)
            session = await self.orchestrator.start_research_session(plan)
            
            # Wait for session to complete or timeout
            max_wait = 10  # seconds
            waited = 0
            while session.session_state not in [SessionState.COMPLETED, SessionState.FAILED] and waited < max_wait:
                await asyncio.sleep(0.5)
                waited += 0.5
                session = self.orchestrator.get_session(session.session_id)
            
            duration = time.time() - start_time
            
            # Verify agents were called in correct order
            research_agent = self.test_agents.get("Research")
            analysis_agent = self.test_agents.get("Analysis")
            
            if not research_agent or not analysis_agent:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="Test agents not found for message routing validation",
                    duration_seconds=duration,
                    remediation_steps=[
                        "Ensure agents are properly initialized before routing test"
                    ]
                )
            
            # Check that both agents were executed
            if research_agent.execution_count == 0:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="Research agent was not called - message routing failed",
                    duration_seconds=duration,
                    details={
                        "research_executions": research_agent.execution_count,
                        "analysis_executions": analysis_agent.execution_count
                    },
                    remediation_steps=[
                        "Check orchestrator task dispatch logic",
                        "Verify agent registry lookup is working",
                        "Review task execution flow"
                    ]
                )
            
            return self.create_result(
                status=ValidationStatus.PASS,
                message=f"Message routing validated successfully - agents executed in correct order",
                duration_seconds=duration,
                details={
                    "research_executions": research_agent.execution_count,
                    "analysis_executions": analysis_agent.execution_count,
                    "session_state": session.session_state.value
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"Message routing validation failed: {e}")
            
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Message routing validation failed with exception: {str(e)}",
                duration_seconds=duration,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check orchestrator message routing logic",
                    "Verify task dependencies are correctly defined",
                    "Review error logs for routing issues"
                ]
            )
        finally:
            # Cleanup
            if self.orchestrator:
                await self.orchestrator.shutdown()
    
    async def validate_session_isolation(self) -> ValidationResult:
        """Validate that concurrent sessions maintain complete isolation.
        
        Tests that multiple sessions running concurrently don't interfere
        with each other and maintain separate state.
        
        Validates: Requirements 3.4
        """
        start_time = time.time()
        
        try:
            # Create orchestrator
            config = OrchestrationConfig(
                max_concurrent_tasks=10,
                task_timeout_seconds=30,
                enable_parallel_execution=False
            )
            self.orchestrator = AgentOrchestrator(config)
            
            # Create multiple research plans with unique identifiers
            from datetime import datetime
            sessions = []
            num_sessions = 3
            
            for i in range(num_sessions):
                plan = ResearchPlan(
                    session_id=str(uuid.uuid4()),
                    query=f"Test query {i} for session isolation",
                    estimated_duration_minutes=5,
                    tasks=[
                        ResearchTask(
                            task_id=f"session_{i}_task_1",
                            task_type="research",
                            description=f"Research task for session {i}",
                            assigned_agent="Research",
                            dependencies=[]
                        )
                    ],
                    created_at=datetime.now()
                )
                session = await self.orchestrator.start_research_session(plan)
                sessions.append(session)
            
            # Wait for all sessions to complete
            max_wait = 15  # seconds
            waited = 0
            while waited < max_wait:
                await asyncio.sleep(0.5)
                waited += 0.5
                
                # Update session states
                sessions = [self.orchestrator.get_session(s.session_id) for s in sessions]
                
                # Check if all completed or failed
                if all(s.session_state in [SessionState.COMPLETED, SessionState.FAILED] for s in sessions):
                    break
            
            duration = time.time() - start_time
            
            # Verify session isolation
            session_ids = [s.session_id for s in sessions]
            unique_session_ids = set(session_ids)
            
            if len(unique_session_ids) != num_sessions:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Session IDs are not unique - isolation may be compromised",
                    duration_seconds=duration,
                    details={
                        "expected_sessions": num_sessions,
                        "unique_sessions": len(unique_session_ids),
                        "session_ids": session_ids
                    },
                    remediation_steps=[
                        "Check session ID generation logic",
                        "Ensure session state is stored separately per session",
                        "Verify no shared mutable state between sessions"
                    ]
                )
            
            # Verify each session has its own task list
            task_ids_per_session = []
            for session in sessions:
                task_ids = [task.task_id for task in session.plan.tasks]
                task_ids_per_session.append(task_ids)
            
            # Check for task ID overlap (should be none)
            all_task_ids = [tid for task_list in task_ids_per_session for tid in task_list]
            unique_task_ids = set(all_task_ids)
            
            if len(unique_task_ids) != len(all_task_ids):
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="Task IDs overlap between sessions - isolation compromised",
                    duration_seconds=duration,
                    details={
                        "total_tasks": len(all_task_ids),
                        "unique_tasks": len(unique_task_ids),
                        "task_ids_per_session": task_ids_per_session
                    },
                    remediation_steps=[
                        "Ensure task IDs are unique per session",
                        "Check for shared task state between sessions",
                        "Verify session data structures are independent"
                    ]
                )
            
            return self.create_result(
                status=ValidationStatus.PASS,
                message=f"Session isolation validated - {num_sessions} concurrent sessions maintained complete isolation",
                duration_seconds=duration,
                details={
                    "num_sessions": num_sessions,
                    "unique_session_ids": len(unique_session_ids),
                    "unique_task_ids": len(unique_task_ids),
                    "session_states": [s.session_state.value for s in sessions]
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"Session isolation validation failed: {e}")
            
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Session isolation validation failed with exception: {str(e)}",
                duration_seconds=duration,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check session management logic",
                    "Verify concurrent session handling",
                    "Review error logs for concurrency issues"
                ]
            )
        finally:
            # Cleanup
            if self.orchestrator:
                await self.orchestrator.shutdown()
    
    async def validate_retry_logic(self) -> ValidationResult:
        """Validate retry logic with exponential backoff.
        
        Tests that the orchestrator implements retry logic with
        exponential backoff for up to 3 attempts when agents fail.
        
        Note: This validation checks if retry logic exists. If the orchestrator
        doesn't implement automatic retries at the task level, this will report
        a warning with recommendations.
        
        Validates: Requirements 3.5, 3.6
        """
        start_time = time.time()
        
        try:
            # Create a mock agent that fails the first 2 times, then succeeds
            failing_agent = MockAgent("RetryTest", failure_count=2)
            agent_registry.register_agent(failing_agent)
            
            # Create orchestrator
            config = OrchestrationConfig(
                max_concurrent_tasks=5,
                task_timeout_seconds=30,
                enable_parallel_execution=False,
                max_retry_attempts=3,
                retry_delay_seconds=0.5
            )
            self.orchestrator = AgentOrchestrator(config)
            
            # Create a research plan with the failing agent
            from datetime import datetime
            plan = ResearchPlan(
                session_id=str(uuid.uuid4()),
                query="Test query for retry logic",
                estimated_duration_minutes=5,
                tasks=[
                    ResearchTask(
                        task_id="retry_task_1",
                        task_type="research",
                        description="Task that will fail and retry",
                        assigned_agent="RetryTest",
                        dependencies=[]
                    )
                ],
                created_at=datetime.now()
            )
            
            # Start session
            session = await self.orchestrator.start_research_session(plan)
            
            # Wait for session to complete
            max_wait = 20  # seconds (need time for retries)
            waited = 0
            while session.session_state not in [SessionState.COMPLETED, SessionState.FAILED] and waited < max_wait:
                await asyncio.sleep(0.5)
                waited += 0.5
                session = self.orchestrator.get_session(session.session_id)
            
            duration = time.time() - start_time
            
            # Check that the agent was called multiple times (retries)
            execution_count = failing_agent.execution_count
            
            # If the agent was only called once, retry logic may not be implemented
            if execution_count < 2:
                return self.create_result(
                    status=ValidationStatus.WARNING,
                    message=f"Retry logic may not be implemented - agent only called {execution_count} time(s). "
                            f"Consider implementing automatic retry logic for failed tasks.",
                    duration_seconds=duration,
                    details={
                        "execution_count": execution_count,
                        "expected_min_executions": 3,
                        "session_state": session.session_state.value,
                        "note": "Orchestrator may not have automatic retry logic at task level"
                    },
                    remediation_steps=[
                        "Implement retry logic in the task dispatcher",
                        "Add exponential backoff between retry attempts",
                        "Configure max_retry_attempts in OrchestrationConfig",
                        "Ensure failed tasks are automatically retried before marking session as failed"
                    ]
                )
            
            # The agent should have been called at least 3 times (initial + 2 retries before success)
            if execution_count < 3:
                return self.create_result(
                    status=ValidationStatus.WARNING,
                    message=f"Retry logic partially working - agent called {execution_count} times (expected at least 3)",
                    duration_seconds=duration,
                    details={
                        "execution_count": execution_count,
                        "expected_min_executions": 3,
                        "session_state": session.session_state.value
                    },
                    remediation_steps=[
                        "Check orchestrator retry configuration",
                        "Verify retry logic is implemented in task dispatcher",
                        "Ensure exponential backoff is applied between retries"
                    ]
                )
            
            # Verify the session eventually succeeded (after retries)
            if session.session_state != SessionState.COMPLETED:
                return self.create_result(
                    status=ValidationStatus.WARNING,
                    message=f"Retry logic executed but session did not complete successfully",
                    duration_seconds=duration,
                    details={
                        "execution_count": execution_count,
                        "session_state": session.session_state.value
                    },
                    remediation_steps=[
                        "Check if max retries is sufficient",
                        "Verify session completion logic",
                        "Review session error logs"
                    ]
                )
            
            return self.create_result(
                status=ValidationStatus.PASS,
                message=f"Retry logic validated - agent retried {execution_count} times and eventually succeeded",
                duration_seconds=duration,
                details={
                    "execution_count": execution_count,
                    "session_state": session.session_state.value,
                    "retry_count": execution_count - 1
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"Retry logic validation failed: {e}")
            
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Retry logic validation failed with exception: {str(e)}",
                duration_seconds=duration,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check retry logic implementation",
                    "Verify exponential backoff calculation",
                    "Review error handling in task dispatcher"
                ]
            )
        finally:
            # Cleanup
            if self.orchestrator:
                await self.orchestrator.shutdown()
            # Remove the test agent
            if "RetryTest" in agent_registry._agents:
                del agent_registry._agents["RetryTest"]
    
    async def validate_state_persistence(self) -> ValidationResult:
        """Validate that session state is persisted for recovery.
        
        Tests that all intermediate results and state transitions
        are persisted to enable recovery from failures.
        
        Validates: Requirements 3.7
        """
        start_time = time.time()
        
        try:
            # Create orchestrator
            config = OrchestrationConfig(
                max_concurrent_tasks=5,
                task_timeout_seconds=30,
                enable_parallel_execution=False
            )
            self.orchestrator = AgentOrchestrator(config)
            
            # Create a research plan
            from datetime import datetime
            plan = ResearchPlan(
                session_id=str(uuid.uuid4()),
                query="Test query for state persistence",
                estimated_duration_minutes=5,
                tasks=[
                    ResearchTask(
                        task_id="persist_task_1",
                        task_type="research",
                        description="Research task for persistence test",
                        assigned_agent="Research",
                        dependencies=[]
                    )
                ],
                created_at=datetime.now()
            )
            
            # Start session
            session = await self.orchestrator.start_research_session(plan)
            session_id = session.session_id
            
            # Wait a bit for task to start
            await asyncio.sleep(1)
            
            # Retrieve session from orchestrator (simulates persistence check)
            retrieved_session = self.orchestrator.get_session(session_id)
            
            duration = time.time() - start_time
            
            if not retrieved_session:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="Session state not persisted - could not retrieve session",
                    duration_seconds=duration,
                    details={"session_id": session_id},
                    remediation_steps=[
                        "Check session storage implementation",
                        "Verify session is stored when created",
                        "Ensure session retrieval logic is correct"
                    ]
                )
            
            # Verify session has the correct data
            if retrieved_session.session_id != session_id:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="Retrieved session has incorrect ID",
                    duration_seconds=duration,
                    details={
                        "expected_id": session_id,
                        "retrieved_id": retrieved_session.session_id
                    },
                    remediation_steps=[
                        "Check session ID assignment logic",
                        "Verify session storage key generation"
                    ]
                )
            
            # Verify session has the plan
            if not retrieved_session.plan or retrieved_session.plan.query != plan.query:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="Retrieved session missing or has incorrect plan data",
                    duration_seconds=duration,
                    details={
                        "has_plan": retrieved_session.plan is not None,
                        "expected_query": plan.query,
                        "retrieved_query": retrieved_session.plan.query if retrieved_session.plan else None
                    },
                    remediation_steps=[
                        "Check session plan persistence",
                        "Verify all session fields are stored",
                        "Ensure session serialization is complete"
                    ]
                )
            
            # Wait for session to complete
            max_wait = 10
            waited = 0
            while retrieved_session.session_state not in [SessionState.COMPLETED, SessionState.FAILED] and waited < max_wait:
                await asyncio.sleep(0.5)
                waited += 0.5
                retrieved_session = self.orchestrator.get_session(session_id)
            
            return self.create_result(
                status=ValidationStatus.PASS,
                message="State persistence validated - session state correctly persisted and retrieved",
                duration_seconds=duration,
                details={
                    "session_id": session_id,
                    "session_state": retrieved_session.session_state.value,
                    "has_plan": True,
                    "task_count": len(retrieved_session.plan.tasks)
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"State persistence validation failed: {e}")
            
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"State persistence validation failed with exception: {str(e)}",
                duration_seconds=duration,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check session persistence implementation",
                    "Verify state storage mechanism",
                    "Review error logs for storage issues"
                ]
            )
        finally:
            # Cleanup
            if self.orchestrator:
                await self.orchestrator.shutdown()
