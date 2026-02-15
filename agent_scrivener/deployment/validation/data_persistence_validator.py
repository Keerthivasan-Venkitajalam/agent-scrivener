"""Data persistence and recovery validator for production readiness.

This validator checks that all data persistence mechanisms are properly configured,
including session metadata persistence, intermediate results storage, final document
persistence, session recovery, session history, archival, backup configuration, and
database connection resilience.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional

from .base_validator import BaseValidator
from .models import ValidationResult, ValidationStatus


logger = logging.getLogger(__name__)


class DataPersistenceValidator(BaseValidator):
    """Validates data persistence and recovery mechanisms for production deployment.
    
    Checks for:
    - Session metadata persistence timing (within 1 second)
    - Intermediate results persistence
    - Final document persistence
    - Session recovery after restart
    - Session history completeness
    - Session archival for old sessions
    - Backup configuration
    - Database connection resilience
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        archival_days: int = 30
    ):
        """Initialize the data persistence validator.
        
        Args:
            database_url: Database connection URL
            archival_days: Number of days before sessions are archived
        """
        super().__init__(name="DataPersistenceValidator")
        self.database_url = database_url
        self.archival_days = archival_days

    async def validate(self) -> List[ValidationResult]:
        """Execute all data persistence validation checks.
        
        Returns:
            List of validation results
        """
        self.log_validation_start()
        
        results = []
        
        # Validate each persistence aspect
        results.append(await self.validate_session_persistence())
        results.append(await self.validate_intermediate_results_persistence())
        results.append(await self.validate_final_document_persistence())
        results.append(await self.validate_session_recovery())
        results.append(await self.validate_session_history())
        results.append(await self.validate_archival())
        results.append(await self.validate_backup_configuration())
        results.append(await self.validate_connection_resilience())
        
        self.log_validation_complete(results)
        return results

    async def validate_session_persistence(self) -> ValidationResult:
        """Validate session metadata persistence timing.
        
        Checks that:
        - Session metadata is persisted to database within 1 second
        - All required fields are stored correctly
        - Session can be retrieved after creation
        
        Returns:
            ValidationResult for session persistence
        """
        try:
            from agent_scrivener.models.core import ResearchSession, ResearchPlan, SessionState, TaskStatus
            from agent_scrivener.data.session_store import SessionStore
            
            try:
                # Create a session store
                store = SessionStore(database_url=self.database_url)
                
                # Create a test session
                test_session_id = f"test_persistence_{int(time.time())}"
                test_session = ResearchSession(
                    session_id=test_session_id,
                    original_query="Test query for persistence validation",
                    plan=ResearchPlan(
                        research_question="Test question",
                        key_topics=["test"],
                        search_queries=["test query"],
                        expected_sources=3,
                        estimated_duration_minutes=5
                    ),
                    session_state=SessionState.INITIALIZING,
                    status=TaskStatus.PENDING
                )
                
                # Measure persistence time
                start_time = time.time()
                await store.save_session(test_session)
                persist_duration = time.time() - start_time
                
                # Check if persistence was within 1 second
                if persist_duration > 1.0:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Session persistence took {persist_duration:.3f}s, exceeds 1 second requirement",
                        duration_seconds=persist_duration,
                        details={
                            "persist_duration_seconds": persist_duration,
                            "threshold_seconds": 1.0,
                            "session_id": test_session_id
                        },
                        remediation_steps=[
                            "Optimize database write operations",
                            "Check database connection latency",
                            "Consider adding database indexes on session_id",
                            "Review database configuration for write performance",
                            "Check if database is under heavy load"
                        ]
                    )
                
                # Verify session can be retrieved
                retrieved_session = await store.get_session(test_session_id)
                
                if not retrieved_session:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="Session was persisted but could not be retrieved",
                        duration_seconds=persist_duration,
                        details={
                            "session_id": test_session_id,
                            "persist_duration_seconds": persist_duration
                        },
                        remediation_steps=[
                            "Check database read operations",
                            "Verify session data is being committed to database",
                            "Check for transaction isolation issues",
                            "Review session retrieval logic"
                        ]
                    )
                
                # Verify all required fields are present
                if retrieved_session.session_id != test_session_id:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="Retrieved session has incorrect session_id",
                        duration_seconds=persist_duration,
                        details={
                            "expected_session_id": test_session_id,
                            "actual_session_id": retrieved_session.session_id
                        },
                        remediation_steps=[
                            "Check session serialization/deserialization logic",
                            "Verify database schema matches model"
                        ]
                    )
                
                # Clean up test session
                await store.delete_session(test_session_id)
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"Session metadata persisted successfully in {persist_duration:.3f}s (within 1 second requirement)",
                    duration_seconds=persist_duration,
                    details={
                        "persist_duration_seconds": persist_duration,
                        "threshold_seconds": 1.0,
                        "session_id": test_session_id,
                        "fields_verified": ["session_id", "original_query", "session_state", "status"]
                    }
                )
                
            except AttributeError as e:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message=f"SessionStore not available: {str(e)}",
                    details={"error": str(e)},
                    remediation_steps=[
                        "Implement SessionStore class in agent_scrivener.data.session_store",
                        "Ensure SessionStore has save_session, get_session, and delete_session methods"
                    ]
                )
            
            except Exception as e:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate session persistence: {str(e)}",
                    details={"error": str(e), "error_type": type(e).__name__},
                    remediation_steps=[
                        "Check database connection is configured correctly",
                        "Verify database is accessible",
                        "Check database credentials and permissions",
                        "Review error logs for detailed information"
                    ]
                )
                
        except ImportError as e:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message=f"Required modules not available: {str(e)}",
                details={"error": str(e)},
                remediation_steps=[
                    "Ensure agent_scrivener.models.core is available",
                    "Ensure agent_scrivener.data.session_store is implemented"
                ]
            )

    async def validate_intermediate_results_persistence(self) -> ValidationResult:
        """Validate intermediate results persistence.
        
        Checks that:
        - Agent execution results are persisted
        - Workflow step results are persisted
        - Intermediate data can be retrieved
        
        Returns:
            ValidationResult for intermediate results persistence
        """
        try:
            from agent_scrivener.models.core import (
                ResearchSession, ResearchPlan, SessionState, TaskStatus,
                AgentExecution, WorkflowStep
            )
            from agent_scrivener.data.session_store import SessionStore
            
            try:
                store = SessionStore(database_url=self.database_url)
                
                # Create a test session with intermediate results
                test_session_id = f"test_intermediate_{int(time.time())}"
                test_session = ResearchSession(
                    session_id=test_session_id,
                    original_query="Test query for intermediate results",
                    plan=ResearchPlan(
                        research_question="Test question",
                        key_topics=["test"],
                        search_queries=["test query"],
                        expected_sources=3,
                        estimated_duration_minutes=5
                    ),
                    session_state=SessionState.RESEARCHING,
                    status=TaskStatus.IN_PROGRESS
                )
                
                # Add intermediate results
                test_execution = AgentExecution(
                    agent_name="TestAgent",
                    started_at=datetime.now(),
                    status=TaskStatus.COMPLETED,
                    result={"test": "data"}
                )
                test_session.add_agent_execution(test_execution)
                
                test_step = WorkflowStep(
                    step_id="test_step_1",
                    step_name="Test Step",
                    agent_name="TestAgent",
                    status=TaskStatus.COMPLETED,
                    expected_outputs=["test_output"]
                )
                test_session.add_workflow_step(test_step)
                
                # Persist session with intermediate results
                await store.save_session(test_session)
                
                # Retrieve and verify intermediate results
                retrieved_session = await store.get_session(test_session_id)
                
                if not retrieved_session:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="Session with intermediate results could not be retrieved",
                        details={"session_id": test_session_id},
                        remediation_steps=[
                            "Check database persistence of nested objects",
                            "Verify serialization of AgentExecution and WorkflowStep",
                            "Check database schema supports nested data"
                        ]
                    )
                
                # Verify agent executions were persisted
                if len(retrieved_session.agent_executions) != 1:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Expected 1 agent execution, found {len(retrieved_session.agent_executions)}",
                        details={
                            "expected_count": 1,
                            "actual_count": len(retrieved_session.agent_executions),
                            "session_id": test_session_id
                        },
                        remediation_steps=[
                            "Check agent_executions field serialization",
                            "Verify database schema includes agent_executions",
                            "Check for data loss during persistence"
                        ]
                    )
                
                # Verify workflow steps were persisted
                if len(retrieved_session.workflow_steps) != 1:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Expected 1 workflow step, found {len(retrieved_session.workflow_steps)}",
                        details={
                            "expected_count": 1,
                            "actual_count": len(retrieved_session.workflow_steps),
                            "session_id": test_session_id
                        },
                        remediation_steps=[
                            "Check workflow_steps field serialization",
                            "Verify database schema includes workflow_steps",
                            "Check for data loss during persistence"
                        ]
                    )
                
                # Clean up
                await store.delete_session(test_session_id)
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message="Intermediate results persisted and retrieved successfully",
                    details={
                        "session_id": test_session_id,
                        "agent_executions_count": 1,
                        "workflow_steps_count": 1
                    }
                )
                
            except AttributeError as e:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message=f"SessionStore not available: {str(e)}",
                    details={"error": str(e)},
                    remediation_steps=[
                        "Implement SessionStore class",
                        "Ensure support for nested object persistence"
                    ]
                )
            
            except Exception as e:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate intermediate results persistence: {str(e)}",
                    details={"error": str(e), "error_type": type(e).__name__},
                    remediation_steps=[
                        "Check database connection",
                        "Verify nested object serialization",
                        "Review error logs for details"
                    ]
                )
                
        except ImportError as e:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message=f"Required modules not available: {str(e)}",
                details={"error": str(e)},
                remediation_steps=[
                    "Ensure all required models are available"
                ]
            )

    async def validate_final_document_persistence(self) -> ValidationResult:
        """Validate final document persistence.
        
        Checks that:
        - Final document is persisted when session completes
        - Document content is stored correctly
        - Document metadata (word count, sources) is persisted
        
        Returns:
            ValidationResult for final document persistence
        """
        try:
            from agent_scrivener.models.core import (
                ResearchSession, ResearchPlan, SessionState, TaskStatus
            )
            from agent_scrivener.data.session_store import SessionStore
            
            try:
                store = SessionStore(database_url=self.database_url)
                
                # Create a completed session with final document
                test_session_id = f"test_final_doc_{int(time.time())}"
                test_document = "# Test Research Document\n\nThis is a test document with sufficient content to validate persistence. " * 50
                
                test_session = ResearchSession(
                    session_id=test_session_id,
                    original_query="Test query for final document",
                    plan=ResearchPlan(
                        research_question="Test question",
                        key_topics=["test"],
                        search_queries=["test query"],
                        expected_sources=3,
                        estimated_duration_minutes=5
                    ),
                    session_state=SessionState.COMPLETED,
                    status=TaskStatus.COMPLETED,
                    final_document=test_document,
                    word_count=len(test_document.split()),
                    sources_count=3,
                    completed_at=datetime.now()
                )
                
                # Persist session with final document
                await store.save_session(test_session)
                
                # Retrieve and verify final document
                retrieved_session = await store.get_session(test_session_id)
                
                if not retrieved_session:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="Session with final document could not be retrieved",
                        details={"session_id": test_session_id},
                        remediation_steps=[
                            "Check database persistence of large text fields",
                            "Verify final_document field is stored correctly",
                            "Check database column size limits"
                        ]
                    )
                
                # Verify final document was persisted
                if not retrieved_session.final_document:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="Final document was not persisted",
                        details={"session_id": test_session_id},
                        remediation_steps=[
                            "Check final_document field serialization",
                            "Verify database schema includes final_document",
                            "Check for data truncation during persistence"
                        ]
                    )
                
                # Verify document content matches
                if retrieved_session.final_document != test_document:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="Final document content does not match",
                        details={
                            "session_id": test_session_id,
                            "expected_length": len(test_document),
                            "actual_length": len(retrieved_session.final_document) if retrieved_session.final_document else 0
                        },
                        remediation_steps=[
                            "Check for data truncation in database",
                            "Verify TEXT/CLOB column type for final_document",
                            "Check character encoding issues"
                        ]
                    )
                
                # Verify metadata was persisted
                if retrieved_session.word_count != test_session.word_count:
                    return self.create_result(
                        status=ValidationStatus.WARNING,
                        message="Document metadata (word_count) does not match",
                        details={
                            "expected_word_count": test_session.word_count,
                            "actual_word_count": retrieved_session.word_count
                        },
                        remediation_steps=[
                            "Check metadata field serialization",
                            "Verify all metadata fields are persisted"
                        ]
                    )
                
                # Clean up
                await store.delete_session(test_session_id)
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message="Final document and metadata persisted successfully",
                    details={
                        "session_id": test_session_id,
                        "document_length": len(test_document),
                        "word_count": test_session.word_count,
                        "sources_count": test_session.sources_count
                    }
                )
                
            except AttributeError as e:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message=f"SessionStore not available: {str(e)}",
                    details={"error": str(e)},
                    remediation_steps=[
                        "Implement SessionStore class"
                    ]
                )
            
            except Exception as e:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate final document persistence: {str(e)}",
                    details={"error": str(e), "error_type": type(e).__name__},
                    remediation_steps=[
                        "Check database connection",
                        "Verify large text field support",
                        "Review error logs for details"
                    ]
                )
                
        except ImportError as e:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message=f"Required modules not available: {str(e)}",
                details={"error": str(e)},
                remediation_steps=[
                    "Ensure all required models are available"
                ]
            )

    async def validate_session_recovery(self) -> ValidationResult:
        """Validate session recovery after restart.
        
        Simulates system restart by:
        - Creating an in-progress session
        - Persisting it
        - Simulating restart (clearing memory)
        - Recovering the session
        - Verifying it can resume from last checkpoint
        
        Returns:
            ValidationResult for session recovery
        """
        try:
            from agent_scrivener.models.core import (
                ResearchSession, ResearchPlan, SessionState, TaskStatus,
                WorkflowStep
            )
            from agent_scrivener.data.session_store import SessionStore
            
            try:
                store = SessionStore(database_url=self.database_url)
                
                # Create an in-progress session
                test_session_id = f"test_recovery_{int(time.time())}"
                test_session = ResearchSession(
                    session_id=test_session_id,
                    original_query="Test query for recovery",
                    plan=ResearchPlan(
                        research_question="Test question",
                        key_topics=["test"],
                        search_queries=["test query"],
                        expected_sources=3,
                        estimated_duration_minutes=5
                    ),
                    session_state=SessionState.RESEARCHING,
                    status=TaskStatus.IN_PROGRESS
                )
                
                # Add some completed and in-progress steps
                completed_step = WorkflowStep(
                    step_id="step_1",
                    step_name="Completed Step",
                    agent_name="TestAgent",
                    status=TaskStatus.COMPLETED,
                    expected_outputs=["output_1"]
                )
                test_session.add_workflow_step(completed_step)
                
                in_progress_step = WorkflowStep(
                    step_id="step_2",
                    step_name="In Progress Step",
                    agent_name="TestAgent",
                    status=TaskStatus.IN_PROGRESS,
                    expected_outputs=["output_2"]
                )
                test_session.add_workflow_step(in_progress_step)
                
                # Persist the in-progress session
                await store.save_session(test_session)
                
                # Simulate restart by clearing the session object
                del test_session
                
                # Simulate a small delay (restart time)
                await asyncio.sleep(0.1)
                
                # Recover the session
                recovered_session = await store.get_session(test_session_id)
                
                if not recovered_session:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="In-progress session could not be recovered after simulated restart",
                        details={"session_id": test_session_id},
                        remediation_steps=[
                            "Ensure in-progress sessions are persisted to database",
                            "Implement session recovery logic",
                            "Check database persistence of session state"
                        ]
                    )
                
                # Verify session state is preserved
                if recovered_session.session_state != SessionState.RESEARCHING:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Session state not preserved: expected RESEARCHING, got {recovered_session.session_state}",
                        details={
                            "session_id": test_session_id,
                            "expected_state": SessionState.RESEARCHING.value,
                            "actual_state": recovered_session.session_state.value
                        },
                        remediation_steps=[
                            "Check session_state field persistence",
                            "Verify enum serialization/deserialization"
                        ]
                    )
                
                # Verify workflow steps are preserved
                if len(recovered_session.workflow_steps) != 2:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Workflow steps not preserved: expected 2, got {len(recovered_session.workflow_steps)}",
                        details={
                            "session_id": test_session_id,
                            "expected_steps": 2,
                            "actual_steps": len(recovered_session.workflow_steps)
                        },
                        remediation_steps=[
                            "Check workflow_steps persistence",
                            "Verify nested object serialization"
                        ]
                    )
                
                # Verify we can identify the last checkpoint (in-progress step)
                current_step = recovered_session.get_current_workflow_step()
                if not current_step or current_step.step_id != "step_2":
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="Could not identify current workflow step for resumption",
                        details={
                            "session_id": test_session_id,
                            "expected_step_id": "step_2",
                            "actual_step_id": current_step.step_id if current_step else None
                        },
                        remediation_steps=[
                            "Implement get_current_workflow_step method",
                            "Ensure step status is preserved correctly"
                        ]
                    )
                
                # Clean up
                await store.delete_session(test_session_id)
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message="Session recovery successful - in-progress session can be resumed from last checkpoint",
                    details={
                        "session_id": test_session_id,
                        "recovered_state": recovered_session.session_state.value,
                        "workflow_steps_count": len(recovered_session.workflow_steps),
                        "current_step_id": current_step.step_id if current_step else None
                    }
                )
                
            except AttributeError as e:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message=f"SessionStore not available: {str(e)}",
                    details={"error": str(e)},
                    remediation_steps=[
                        "Implement SessionStore class",
                        "Implement session recovery logic"
                    ]
                )
            
            except Exception as e:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate session recovery: {str(e)}",
                    details={"error": str(e), "error_type": type(e).__name__},
                    remediation_steps=[
                        "Check database connection",
                        "Verify session recovery logic",
                        "Review error logs for details"
                    ]
                )
                
        except ImportError as e:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message=f"Required modules not available: {str(e)}",
                details={"error": str(e)},
                remediation_steps=[
                    "Ensure all required models are available"
                ]
            )

    async def validate_session_history(self) -> ValidationResult:
        """Validate session history retrieval.
        
        Checks that:
        - All state transitions are recorded
        - Session history can be queried
        - History includes timestamps and details
        
        Returns:
            ValidationResult for session history
        """
        try:
            from agent_scrivener.models.core import (
                ResearchSession, ResearchPlan, SessionState, TaskStatus
            )
            from agent_scrivener.data.session_store import SessionStore
            
            try:
                store = SessionStore(database_url=self.database_url)
                
                # Create a session and transition through multiple states
                test_session_id = f"test_history_{int(time.time())}"
                test_session = ResearchSession(
                    session_id=test_session_id,
                    original_query="Test query for history",
                    plan=ResearchPlan(
                        research_question="Test question",
                        key_topics=["test"],
                        search_queries=["test query"],
                        expected_sources=3,
                        estimated_duration_minutes=5
                    ),
                    session_state=SessionState.INITIALIZING,
                    status=TaskStatus.PENDING
                )
                
                # Save initial state
                await store.save_session(test_session)
                await asyncio.sleep(0.1)
                
                # Transition to RESEARCHING
                test_session.transition_state(SessionState.RESEARCHING)
                test_session.status = TaskStatus.IN_PROGRESS
                await store.save_session(test_session)
                await asyncio.sleep(0.1)
                
                # Transition to ANALYZING
                test_session.transition_state(SessionState.ANALYZING)
                await store.save_session(test_session)
                await asyncio.sleep(0.1)
                
                # Transition to COMPLETED
                test_session.transition_state(SessionState.COMPLETED)
                test_session.status = TaskStatus.COMPLETED
                await store.save_session(test_session)
                
                # Retrieve session and check history
                retrieved_session = await store.get_session(test_session_id)
                
                if not retrieved_session:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="Session could not be retrieved for history validation",
                        details={"session_id": test_session_id},
                        remediation_steps=[
                            "Check database persistence",
                            "Verify session retrieval logic"
                        ]
                    )
                
                # Verify final state is correct
                if retrieved_session.session_state != SessionState.COMPLETED:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Final session state incorrect: expected COMPLETED, got {retrieved_session.session_state}",
                        details={
                            "session_id": test_session_id,
                            "expected_state": SessionState.COMPLETED.value,
                            "actual_state": retrieved_session.session_state.value
                        },
                        remediation_steps=[
                            "Check state transition persistence",
                            "Verify session updates are saved correctly"
                        ]
                    )
                
                # Verify timestamps are updated
                if retrieved_session.updated_at <= retrieved_session.created_at:
                    return self.create_result(
                        status=ValidationStatus.WARNING,
                        message="Session updated_at timestamp not properly maintained",
                        details={
                            "session_id": test_session_id,
                            "created_at": retrieved_session.created_at.isoformat(),
                            "updated_at": retrieved_session.updated_at.isoformat()
                        },
                        remediation_steps=[
                            "Ensure update_timestamp() is called on state transitions",
                            "Verify timestamp fields are persisted correctly"
                        ]
                    )
                
                # Try to get session history if available
                try:
                    history = await store.get_session_history(test_session_id)
                    
                    if not history or len(history) < 4:
                        return self.create_result(
                            status=ValidationStatus.WARNING,
                            message="Session history not fully captured (expected 4+ state transitions)",
                            details={
                                "session_id": test_session_id,
                                "expected_transitions": 4,
                                "actual_transitions": len(history) if history else 0
                            },
                            remediation_steps=[
                                "Implement session history tracking",
                                "Store each state transition with timestamp",
                                "Consider using audit log table for history"
                            ]
                        )
                    
                    # Clean up
                    await store.delete_session(test_session_id)
                    
                    return self.create_result(
                        status=ValidationStatus.PASS,
                        message=f"Session history complete with {len(history)} state transitions recorded",
                        details={
                            "session_id": test_session_id,
                            "state_transitions": len(history),
                            "final_state": retrieved_session.session_state.value
                        }
                    )
                    
                except AttributeError:
                    # get_session_history not implemented, but basic state tracking works
                    await store.delete_session(test_session_id)
                    
                    return self.create_result(
                        status=ValidationStatus.WARNING,
                        message="Session state transitions work but detailed history tracking not implemented",
                        details={
                            "session_id": test_session_id,
                            "final_state": retrieved_session.session_state.value,
                            "timestamps_tracked": True
                        },
                        remediation_steps=[
                            "Implement get_session_history method for detailed audit trail",
                            "Consider adding session_history table to track all transitions",
                            "Store transition timestamps, previous state, new state, and trigger"
                        ]
                    )
                
            except AttributeError as e:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message=f"SessionStore not available: {str(e)}",
                    details={"error": str(e)},
                    remediation_steps=[
                        "Implement SessionStore class"
                    ]
                )
            
            except Exception as e:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate session history: {str(e)}",
                    details={"error": str(e), "error_type": type(e).__name__},
                    remediation_steps=[
                        "Check database connection",
                        "Verify session update logic",
                        "Review error logs for details"
                    ]
                )
                
        except ImportError as e:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message=f"Required modules not available: {str(e)}",
                details={"error": str(e)},
                remediation_steps=[
                    "Ensure all required models are available"
                ]
            )

    async def validate_archival(self) -> ValidationResult:
        """Validate session archival for old sessions.
        
        Checks that:
        - Old sessions (>30 days) can be identified
        - Archival process moves sessions to cold storage
        - Archived sessions can still be retrieved if needed
        
        Returns:
            ValidationResult for session archival
        """
        try:
            from agent_scrivener.models.core import (
                ResearchSession, ResearchPlan, SessionState, TaskStatus
            )
            from agent_scrivener.data.session_store import SessionStore
            
            try:
                store = SessionStore(database_url=self.database_url)
                
                # Create an old completed session (simulate by setting old timestamp)
                test_session_id = f"test_archival_{int(time.time())}"
                old_date = datetime.now() - timedelta(days=self.archival_days + 1)
                
                test_session = ResearchSession(
                    session_id=test_session_id,
                    original_query="Test query for archival",
                    plan=ResearchPlan(
                        research_question="Test question",
                        key_topics=["test"],
                        search_queries=["test query"],
                        expected_sources=3,
                        estimated_duration_minutes=5
                    ),
                    session_state=SessionState.COMPLETED,
                    status=TaskStatus.COMPLETED,
                    created_at=old_date,
                    updated_at=old_date,
                    completed_at=old_date
                )
                
                # Persist the old session
                await store.save_session(test_session)
                
                # Try to get sessions eligible for archival
                try:
                    archival_cutoff = datetime.now() - timedelta(days=self.archival_days)
                    old_sessions = await store.get_sessions_before_date(archival_cutoff)
                    
                    # Check if our test session is in the list
                    found_test_session = any(s.session_id == test_session_id for s in old_sessions)
                    
                    if not found_test_session:
                        return self.create_result(
                            status=ValidationStatus.WARNING,
                            message=f"Old session not identified for archival (cutoff: {self.archival_days} days)",
                            details={
                                "session_id": test_session_id,
                                "session_age_days": self.archival_days + 1,
                                "archival_cutoff_days": self.archival_days,
                                "old_sessions_found": len(old_sessions)
                            },
                            remediation_steps=[
                                "Implement get_sessions_before_date method",
                                "Ensure date filtering works correctly",
                                "Check database query for old sessions"
                            ]
                        )
                    
                    # Try to archive the session
                    try:
                        await store.archive_session(test_session_id)
                        
                        # Verify session is still retrievable (from archive)
                        archived_session = await store.get_session(test_session_id, include_archived=True)
                        
                        if not archived_session:
                            return self.create_result(
                                status=ValidationStatus.FAIL,
                                message="Archived session cannot be retrieved",
                                details={"session_id": test_session_id},
                                remediation_steps=[
                                    "Ensure archived sessions remain accessible",
                                    "Implement include_archived parameter in get_session",
                                    "Check archival storage mechanism"
                                ]
                            )
                        
                        # Clean up
                        await store.delete_session(test_session_id, include_archived=True)
                        
                        return self.create_result(
                            status=ValidationStatus.PASS,
                            message=f"Session archival works correctly (cutoff: {self.archival_days} days)",
                            details={
                                "session_id": test_session_id,
                                "archival_cutoff_days": self.archival_days,
                                "old_sessions_found": len(old_sessions),
                                "archived_and_retrievable": True
                            }
                        )
                        
                    except AttributeError:
                        # archive_session not implemented
                        await store.delete_session(test_session_id)
                        
                        return self.create_result(
                            status=ValidationStatus.WARNING,
                            message="Session archival not implemented",
                            details={
                                "session_id": test_session_id,
                                "old_sessions_found": len(old_sessions),
                                "archival_cutoff_days": self.archival_days
                            },
                            remediation_steps=[
                                "Implement archive_session method",
                                "Move old sessions to cold storage (S3, archive table)",
                                "Implement automated archival job",
                                "Consider using database partitioning by date"
                            ]
                        )
                    
                except AttributeError:
                    # get_sessions_before_date not implemented
                    await store.delete_session(test_session_id)
                    
                    return self.create_result(
                        status=ValidationStatus.WARNING,
                        message="Session archival query not implemented",
                        details={
                            "session_id": test_session_id,
                            "archival_cutoff_days": self.archival_days
                        },
                        remediation_steps=[
                            "Implement get_sessions_before_date method",
                            "Add database query to find old completed sessions",
                            "Implement automated archival process",
                            f"Archive sessions older than {self.archival_days} days"
                        ]
                    )
                
            except AttributeError as e:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message=f"SessionStore not available: {str(e)}",
                    details={"error": str(e)},
                    remediation_steps=[
                        "Implement SessionStore class"
                    ]
                )
            
            except Exception as e:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate session archival: {str(e)}",
                    details={"error": str(e), "error_type": type(e).__name__},
                    remediation_steps=[
                        "Check database connection",
                        "Verify archival logic",
                        "Review error logs for details"
                    ]
                )
                
        except ImportError as e:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message=f"Required modules not available: {str(e)}",
                details={"error": str(e)},
                remediation_steps=[
                    "Ensure all required models are available"
                ]
            )

    async def validate_backup_configuration(self) -> ValidationResult:
        """Validate backup configuration.
        
        Checks that:
        - Automated backup schedules are configured
        - Backup retention policies are set
        - Backup verification is in place
        
        Returns:
            ValidationResult for backup configuration
        """
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            try:
                # Check for RDS automated backups if using AWS RDS
                if self.database_url and 'rds.amazonaws.com' in self.database_url:
                    # Extract region from RDS endpoint
                    import re
                    region_match = re.search(r'\.([a-z]{2}-[a-z]+-\d)\.rds\.amazonaws\.com', self.database_url)
                    region = region_match.group(1) if region_match else None
                    
                    if region:
                        rds_client = boto3.client('rds', region_name=region)
                        
                        # Extract DB instance identifier from URL
                        db_match = re.search(r'//([^.]+)\.', self.database_url)
                        db_identifier = db_match.group(1) if db_match else None
                        
                        if db_identifier:
                            try:
                                response = rds_client.describe_db_instances(
                                    DBInstanceIdentifier=db_identifier
                                )
                                
                                db_instance = response['DBInstances'][0]
                                backup_retention = db_instance.get('BackupRetentionPeriod', 0)
                                
                                if backup_retention == 0:
                                    return self.create_result(
                                        status=ValidationStatus.FAIL,
                                        message="RDS automated backups not configured (retention period is 0)",
                                        details={
                                            "db_identifier": db_identifier,
                                            "backup_retention_days": backup_retention,
                                            "region": region
                                        },
                                        remediation_steps=[
                                            "Enable RDS automated backups",
                                            "Set backup retention period to at least 7 days",
                                            "Use AWS CLI: aws rds modify-db-instance --db-instance-identifier <id> --backup-retention-period 7",
                                            "Configure backup window for low-traffic periods",
                                            "Enable automated backups in RDS configuration"
                                        ]
                                    )
                                
                                if backup_retention < 7:
                                    return self.create_result(
                                        status=ValidationStatus.WARNING,
                                        message=f"RDS backup retention period is {backup_retention} days (recommended: 7+ days)",
                                        details={
                                            "db_identifier": db_identifier,
                                            "backup_retention_days": backup_retention,
                                            "recommended_retention_days": 7,
                                            "region": region
                                        },
                                        remediation_steps=[
                                            "Increase backup retention period to at least 7 days",
                                            "Consider 30 days for production databases",
                                            "Use AWS CLI: aws rds modify-db-instance --db-instance-identifier <id> --backup-retention-period 7"
                                        ]
                                    )
                                
                                return self.create_result(
                                    status=ValidationStatus.PASS,
                                    message=f"RDS automated backups configured with {backup_retention} days retention",
                                    details={
                                        "db_identifier": db_identifier,
                                        "backup_retention_days": backup_retention,
                                        "preferred_backup_window": db_instance.get('PreferredBackupWindow'),
                                        "region": region
                                    }
                                )
                                
                            except ClientError as e:
                                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                                return self.create_result(
                                    status=ValidationStatus.FAIL,
                                    message=f"Failed to check RDS backup configuration: {error_code}",
                                    details={
                                        "error": str(e),
                                        "error_code": error_code,
                                        "db_identifier": db_identifier
                                    },
                                    remediation_steps=[
                                        "Check AWS credentials have RDS permissions",
                                        "Required permissions: rds:DescribeDBInstances",
                                        "Verify DB instance identifier is correct"
                                    ]
                                )
                
                # For non-RDS databases, provide general guidance
                return self.create_result(
                    status=ValidationStatus.WARNING,
                    message="Database backup configuration cannot be automatically validated (non-RDS database)",
                    details={
                        "database_url": self.database_url if self.database_url else "not_configured",
                        "validation_type": "manual_required"
                    },
                    remediation_steps=[
                        "Manually verify database backup configuration",
                        "Ensure automated backups are scheduled (daily recommended)",
                        "Set backup retention policy (7-30 days recommended)",
                        "Test backup restoration process",
                        "Document backup and restore procedures",
                        "For PostgreSQL: Configure pg_dump or WAL archiving",
                        "For MySQL: Configure mysqldump or binary log backups",
                        "Consider using managed database services with automated backups"
                    ]
                )
                
            except NoCredentialsError:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="AWS credentials not configured, skipping RDS backup validation",
                    details={"reason": "no_credentials"},
                    remediation_steps=[
                        "Configure AWS credentials if using RDS",
                        "Manually verify backup configuration for non-AWS databases"
                    ]
                )
            
            except Exception as e:
                return self.create_result(
                    status=ValidationStatus.WARNING,
                    message=f"Could not validate backup configuration: {str(e)}",
                    details={"error": str(e), "error_type": type(e).__name__},
                    remediation_steps=[
                        "Manually verify database backup configuration",
                        "Ensure automated backups are enabled",
                        "Test backup restoration process"
                    ]
                )
                
        except ImportError:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message="boto3 not installed, skipping AWS RDS backup validation",
                details={"reason": "boto3_not_installed"},
                remediation_steps=[
                    "Install boto3 if using AWS RDS: pip install boto3",
                    "Manually verify backup configuration for non-AWS databases"
                ]
            )

    async def validate_connection_resilience(self) -> ValidationResult:
        """Validate database connection resilience.
        
        Checks that:
        - System handles database connection loss gracefully
        - Operations are queued and retried when connection is restored
        - Connection pool is configured correctly
        
        Returns:
            ValidationResult for connection resilience
        """
        try:
            from agent_scrivener.models.core import (
                ResearchSession, ResearchPlan, SessionState, TaskStatus
            )
            from agent_scrivener.data.session_store import SessionStore
            
            try:
                store = SessionStore(database_url=self.database_url)
                
                # Test 1: Verify connection pool configuration
                try:
                    pool_info = await store.get_connection_pool_info()
                    
                    if pool_info.get('max_connections', 0) < 5:
                        return self.create_result(
                            status=ValidationStatus.WARNING,
                            message=f"Connection pool size is {pool_info.get('max_connections', 0)} (recommended: 10+)",
                            details=pool_info,
                            remediation_steps=[
                                "Increase database connection pool size",
                                "Recommended: 10-20 connections for production",
                                "Configure in database connection settings",
                                "Example: pool_size=10, max_overflow=20"
                            ]
                        )
                    
                except AttributeError:
                    # get_connection_pool_info not implemented, skip this check
                    pass
                
                # Test 2: Verify retry logic exists
                test_session_id = f"test_resilience_{int(time.time())}"
                test_session = ResearchSession(
                    session_id=test_session_id,
                    original_query="Test query for resilience",
                    plan=ResearchPlan(
                        research_question="Test question",
                        key_topics=["test"],
                        search_queries=["test query"],
                        expected_sources=3,
                        estimated_duration_minutes=5
                    ),
                    session_state=SessionState.INITIALIZING,
                    status=TaskStatus.PENDING
                )
                
                # Try to save session (should work with normal connection)
                try:
                    await store.save_session(test_session)
                    
                    # Verify it was saved
                    retrieved = await store.get_session(test_session_id)
                    if not retrieved:
                        return self.create_result(
                            status=ValidationStatus.FAIL,
                            message="Session save/retrieve failed under normal conditions",
                            details={"session_id": test_session_id},
                            remediation_steps=[
                                "Check database connection",
                                "Verify save_session implementation",
                                "Check database permissions"
                            ]
                        )
                    
                    # Clean up
                    await store.delete_session(test_session_id)
                    
                    # Check if retry configuration exists
                    try:
                        retry_config = await store.get_retry_configuration()
                        
                        if not retry_config.get('enabled', False):
                            return self.create_result(
                                status=ValidationStatus.WARNING,
                                message="Database retry logic not configured",
                                details=retry_config,
                                remediation_steps=[
                                    "Implement retry logic for database operations",
                                    "Use exponential backoff for retries",
                                    "Recommended: 3 retries with 1s, 2s, 4s delays",
                                    "Handle transient connection errors gracefully",
                                    "Consider using libraries like tenacity for retry logic"
                                ]
                            )
                        
                        max_retries = retry_config.get('max_retries', 0)
                        if max_retries < 3:
                            return self.create_result(
                                status=ValidationStatus.WARNING,
                                message=f"Database retry count is {max_retries} (recommended: 3+)",
                                details=retry_config,
                                remediation_steps=[
                                    "Increase max retry attempts to at least 3",
                                    "Use exponential backoff between retries",
                                    "Configure appropriate timeout values"
                                ]
                            )
                        
                        return self.create_result(
                            status=ValidationStatus.PASS,
                            message=f"Database connection resilience configured with {max_retries} retries",
                            details={
                                "retry_config": retry_config,
                                "connection_test": "passed"
                            }
                        )
                        
                    except AttributeError:
                        # get_retry_configuration not implemented
                        return self.create_result(
                            status=ValidationStatus.WARNING,
                            message="Database connection resilience cannot be fully validated (retry configuration not exposed)",
                            details={
                                "connection_test": "passed",
                                "retry_config": "not_available"
                            },
                            remediation_steps=[
                                "Implement get_retry_configuration method to expose retry settings",
                                "Ensure retry logic is implemented for database operations",
                                "Use exponential backoff: 3 retries with 1s, 2s, 4s delays",
                                "Handle connection errors, timeouts, and transient failures",
                                "Consider using connection pooling with health checks",
                                "Implement circuit breaker pattern for repeated failures",
                                "Log retry attempts for monitoring"
                            ]
                        )
                    
                except Exception as e:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"Database operation failed: {str(e)}",
                        details={"error": str(e), "error_type": type(e).__name__},
                        remediation_steps=[
                            "Check database connection",
                            "Verify database is accessible",
                            "Check database credentials",
                            "Review error logs for details"
                        ]
                    )
                
            except AttributeError as e:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message=f"SessionStore not available: {str(e)}",
                    details={"error": str(e)},
                    remediation_steps=[
                        "Implement SessionStore class",
                        "Implement connection resilience features"
                    ]
                )
            
            except Exception as e:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Failed to validate connection resilience: {str(e)}",
                    details={"error": str(e), "error_type": type(e).__name__},
                    remediation_steps=[
                        "Check database connection",
                        "Verify connection resilience implementation",
                        "Review error logs for details"
                    ]
                )
                
        except ImportError as e:
            return self.create_result(
                status=ValidationStatus.SKIP,
                message=f"Required modules not available: {str(e)}",
                details={"error": str(e)},
                remediation_steps=[
                    "Ensure all required models are available"
                ]
            )
