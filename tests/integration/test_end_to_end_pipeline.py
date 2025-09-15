"""
End-to-end integration tests for the complete research pipeline.

Tests the entire research workflow from query input to final document generation,
validating all agent interactions and data flow.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

from tests.integration.test_framework import (
    IntegrationTestFramework, MockServiceConfig, TestDataGenerator
)
from agent_scrivener.models.core import TaskStatus, SessionState
from agent_scrivener.orchestration.orchestrator import OrchestrationConfig


class TestEndToEndPipeline:
    """Test complete research pipeline execution."""
    
    @pytest.mark.asyncio
    async def test_complete_research_workflow(self):
        """Test complete research workflow from query to final document."""
        config = MockServiceConfig(
            web_search_delay=0.1,
            api_query_delay=0.1,
            analysis_delay=0.15,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Execute end-to-end test
            query = "Machine learning applications in healthcare"
            results = await framework.run_end_to_end_test(query)
            
            session = results["session"]
            session_results = results["results"]
            metrics = results["metrics"]
            
            # Verify session completion
            assert session.status == TaskStatus.COMPLETED
            assert session.session_state == SessionState.COMPLETED
            assert session.original_query == query
            
            # Verify all tasks completed successfully
            completed_tasks = [task for task in session.plan.tasks if task.status == TaskStatus.COMPLETED]
            assert len(completed_tasks) == 7  # All 7 tasks should complete
            
            # Verify task execution order
            task_completion_order = sorted(completed_tasks, key=lambda t: t.completed_at)
            
            # Web research and API searches should complete first (parallel)
            early_tasks = [t.task_id for t in task_completion_order[:3]]
            assert "web_research" in early_tasks
            assert "arxiv_search" in early_tasks
            assert "pubmed_search" in early_tasks
            
            # Analysis tasks should complete after data collection
            analysis_tasks = [t for t in task_completion_order if "analysis" in t.task_id or "insight" in t.task_id]
            assert len(analysis_tasks) >= 2
            
            # Document generation should be last
            last_tasks = [t.task_id for t in task_completion_order[-2:]]
            assert "document_drafting" in last_tasks
            assert "citation_formatting" in last_tasks
            
            # Verify session results structure
            assert session_results is not None
            assert session_results["session_id"] == session.session_id
            assert session_results["query"] == query
            assert "sources" in session_results
            assert "analysis" in session_results
            assert "document" in session_results
            assert "execution" in session_results
            
            # Verify execution metrics
            execution_data = session_results["execution"]
            assert execution_data["total_tasks"] == 7
            assert execution_data["completed_tasks"] == 7
            assert execution_data["failed_tasks"] == 0
            assert execution_data["success_rate"] == 100.0
            
            # Verify performance metrics
            assert metrics.success_rate == 100.0
            assert metrics.error_count == 0
            assert metrics.execution_time_ms > 0
            assert len(metrics.task_completion_times) == 7
    
    @pytest.mark.asyncio
    async def test_pipeline_with_partial_failures(self):
        """Test pipeline resilience with partial task failures."""
        config = MockServiceConfig(
            web_search_delay=0.05,
            api_query_delay=0.05,
            analysis_delay=0.1,
            failure_rate=0.3  # 30% failure rate
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            query = "Quantum computing algorithms"
            results = await framework.run_end_to_end_test(query)
            
            session = results["session"]
            metrics = results["metrics"]
            
            # Session should complete even with some failures
            assert session.status in [TaskStatus.COMPLETED, TaskStatus.PARTIALLY_COMPLETED]
            
            # Some tasks should have completed successfully
            completed_tasks = [task for task in session.plan.tasks if task.status == TaskStatus.COMPLETED]
            failed_tasks = [task for task in session.plan.tasks if task.status == TaskStatus.FAILED]
            
            assert len(completed_tasks) > 0
            assert len(failed_tasks) > 0
            
            # Verify error handling
            assert metrics.error_count > 0
            assert metrics.success_rate < 100.0
            
            # Dependent tasks should not execute if dependencies failed
            for task in session.plan.tasks:
                if task.dependencies and task.status == TaskStatus.COMPLETED:
                    # Check that at least one dependency completed
                    dependency_statuses = [
                        session.plan.get_task_by_id(dep_id).status 
                        for dep_id in task.dependencies
                    ]
                    assert TaskStatus.COMPLETED in dependency_statuses
    
    @pytest.mark.asyncio
    async def test_multiple_research_queries(self):
        """Test pipeline with various types of research queries."""
        config = MockServiceConfig(
            web_search_delay=0.02,
            api_query_delay=0.03,
            analysis_delay=0.05,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        test_queries = TestDataGenerator.generate_research_queries()[:5]  # Test 5 queries
        
        async with framework.test_environment():
            results = []
            
            for query in test_queries:
                result = await framework.run_end_to_end_test(query)
                results.append(result)
                
                # Brief pause between queries
                await asyncio.sleep(0.1)
            
            # Verify all queries completed successfully
            for i, result in enumerate(results):
                session = result["session"]
                metrics = result["metrics"]
                
                assert session.status == TaskStatus.COMPLETED, f"Query {i+1} failed: {test_queries[i]}"
                assert session.original_query == test_queries[i]
                assert metrics.success_rate == 100.0
                assert metrics.error_count == 0
            
            # Verify performance consistency
            execution_times = [r["metrics"].execution_time_ms for r in results]
            avg_execution_time = sum(execution_times) / len(execution_times)
            
            # No execution should take more than 3x the average (performance consistency)
            for exec_time in execution_times:
                assert exec_time <= avg_execution_time * 3
    
    @pytest.mark.asyncio
    async def test_progress_tracking_accuracy(self):
        """Test accuracy of progress tracking throughout pipeline execution."""
        config = MockServiceConfig(
            web_search_delay=0.2,  # Longer delays to observe progress
            api_query_delay=0.2,
            analysis_delay=0.3,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            query = "Renewable energy storage solutions"
            results = await framework.run_end_to_end_test(query)
            
            progress_updates = results["progress_updates"]
            session = results["session"]
            
            # Should have received multiple progress updates
            assert len(progress_updates) >= 5
            
            # Progress should be monotonically increasing
            progress_values = [update["data"]["progress_percentage"] for update in progress_updates]
            for i in range(1, len(progress_values)):
                assert progress_values[i] >= progress_values[i-1]
            
            # Final progress should be 100%
            assert progress_values[-1] == 100.0
            
            # Verify progress update timing
            update_times = [update["timestamp"] for update in progress_updates]
            time_intervals = [
                (update_times[i] - update_times[i-1]).total_seconds()
                for i in range(1, len(update_times))
            ]
            
            # Updates should be reasonably frequent (within expected interval)
            avg_interval = sum(time_intervals) / len(time_intervals)
            assert avg_interval <= 2.0  # Should update at least every 2 seconds
            
            # Verify task-level progress tracking
            for update in progress_updates:
                data = update["data"]
                assert "completed_tasks" in data
                assert "total_tasks" in data
                assert "current_task" in data
                assert data["completed_tasks"] <= data["total_tasks"]
                assert data["total_tasks"] == 7
    
    @pytest.mark.asyncio
    async def test_data_flow_validation(self):
        """Test data flow between agents and validate data integrity."""
        config = MockServiceConfig(
            web_search_delay=0.05,
            api_query_delay=0.05,
            analysis_delay=0.1,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            query = "Artificial intelligence ethics"
            results = await framework.run_end_to_end_test(query)
            
            session = results["session"]
            session_results = results["results"]
            
            # Verify data collection phase
            sources_data = session_results["sources"]
            assert "web_sources" in sources_data
            assert "academic_papers" in sources_data
            assert len(sources_data["web_sources"]) > 0
            assert len(sources_data["academic_papers"]) > 0
            
            # Verify each web source has required fields
            for source in sources_data["web_sources"]:
                assert "url" in source
                assert "title" in source
                assert "content" in source
                assert "confidence_score" in source
                assert 0 <= source["confidence_score"] <= 1
            
            # Verify each academic paper has required fields
            for paper in sources_data["academic_papers"]:
                assert "title" in paper
                assert "authors" in paper
                assert "abstract" in paper
                assert "publication_year" in paper
                assert isinstance(paper["authors"], list)
                assert len(paper["authors"]) > 0
            
            # Verify analysis phase
            analysis_data = session_results["analysis"]
            assert "insights" in analysis_data
            assert "entities" in analysis_data
            assert "topics" in analysis_data
            
            # Verify insights structure
            insights = analysis_data["insights"]
            assert len(insights) > 0
            for insight in insights:
                assert "topic" in insight
                assert "summary" in insight
                assert "confidence_score" in insight
                assert "supporting_evidence" in insight
                assert 0 <= insight["confidence_score"] <= 1
                assert isinstance(insight["supporting_evidence"], list)
            
            # Verify document generation
            document_data = session_results["document"]
            assert "content" in document_data
            assert "sections" in document_data
            assert "citations" in document_data
            
            # Verify document structure
            content = document_data["content"]
            assert len(content) > 100  # Should be substantial content
            assert "# " in content  # Should have headers
            
            sections = document_data["sections"]
            expected_sections = ["introduction", "methodology", "findings", "conclusion"]
            for section in expected_sections:
                assert section in sections
                assert len(sections[section]) > 0
            
            # Verify citations
            citations = document_data["citations"]
            assert len(citations) > 0
            for citation in citations:
                assert "citation_id" in citation
                assert "text" in citation
    
    @pytest.mark.asyncio
    async def test_session_state_transitions(self):
        """Test proper session state transitions throughout pipeline execution."""
        config = MockServiceConfig(
            web_search_delay=0.1,
            api_query_delay=0.1,
            analysis_delay=0.15,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            query = "Space exploration technologies"
            
            # Track session state changes
            state_changes = []
            
            # Start session and monitor state
            session_id = await framework.orchestrator.start_research_session(
                framework._create_comprehensive_research_plan(query, "test_session")
            )
            
            # Monitor session state changes
            max_wait_time = 30
            start_time = asyncio.get_event_loop().time()
            
            while True:
                current_time = asyncio.get_event_loop().time()
                if current_time - start_time > max_wait_time:
                    break
                
                session = framework.orchestrator.get_session("test_session")
                if session:
                    current_state = {
                        "timestamp": datetime.now(),
                        "session_state": session.session_state,
                        "task_status": session.status,
                        "completed_tasks": len([t for t in session.plan.tasks if t.status == TaskStatus.COMPLETED])
                    }
                    
                    # Only record state changes
                    if not state_changes or state_changes[-1]["session_state"] != current_state["session_state"]:
                        state_changes.append(current_state)
                    
                    if session.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                        break
                
                await asyncio.sleep(0.2)
            
            # Verify state transition sequence
            assert len(state_changes) >= 3  # Should have multiple state transitions
            
            # First state should be INITIALIZING or RESEARCHING
            first_state = state_changes[0]["session_state"]
            assert first_state in [SessionState.INITIALIZING, SessionState.RESEARCHING]
            
            # Should transition through expected states
            state_sequence = [change["session_state"] for change in state_changes]
            
            # Should include RESEARCHING state
            assert SessionState.RESEARCHING in state_sequence
            
            # Final state should be COMPLETED
            final_state = state_changes[-1]["session_state"]
            assert final_state == SessionState.COMPLETED
            
            # Task completion should increase over time
            completion_counts = [change["completed_tasks"] for change in state_changes]
            for i in range(1, len(completion_counts)):
                assert completion_counts[i] >= completion_counts[i-1]
    
    @pytest.mark.asyncio
    async def test_resource_cleanup(self):
        """Test proper resource cleanup after pipeline execution."""
        config = MockServiceConfig(
            web_search_delay=0.02,
            api_query_delay=0.02,
            analysis_delay=0.05,
            failure_rate=0.0
        )
        
        framework = IntegrationTestFramework(config)
        
        async with framework.test_environment():
            # Run multiple sessions to test cleanup
            queries = ["Test query 1", "Test query 2", "Test query 3"]
            
            for i, query in enumerate(queries):
                results = await framework.run_end_to_end_test(query)
                session = results["session"]
                
                # Verify session completed
                assert session.status == TaskStatus.COMPLETED
                
                # Check active sessions count
                active_sessions = framework.orchestrator.list_active_sessions()
                
                # After completion, session should still be tracked but not active
                # (depending on cleanup policy)
                assert len(active_sessions) <= 1  # At most current session
        
        # After context manager exit, all resources should be cleaned up
        # This is verified by the context manager's cleanup code
        assert len(framework._patches) == 0  # All patches should be removed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])