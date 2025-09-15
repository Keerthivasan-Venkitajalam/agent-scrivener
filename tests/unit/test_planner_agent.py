"""
Unit tests for Planner Agent.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from agent_scrivener.agents.planner_agent import (
    PlannerAgent, TaskGraph, QueryAnalyzer
)
from agent_scrivener.models.core import (
    ResearchPlan, ResearchTask, TaskStatus, WorkflowStep
)


class TestTaskGraph:
    """Test TaskGraph functionality."""
    
    def test_add_task(self):
        """Test adding tasks to the graph."""
        graph = TaskGraph()
        task = ResearchTask(
            task_id="test_task",
            task_type="test",
            description="Test task"
        )
        
        graph.add_task(task)
        
        assert "test_task" in graph.nodes
        assert graph.nodes["test_task"] == task
        assert "test_task" in graph.edges
    
    def test_add_dependency(self):
        """Test adding dependencies between tasks."""
        graph = TaskGraph()
        
        task1 = ResearchTask(task_id="task1", task_type="test", description="Task 1")
        task2 = ResearchTask(task_id="task2", task_type="test", description="Task 2")
        
        graph.add_task(task1)
        graph.add_task(task2)
        graph.add_dependency("task2", "task1")
        
        assert "task2" in graph.edges["task1"]
        assert "task1" in task2.dependencies
    
    def test_add_dependency_missing_task(self):
        """Test adding dependency with missing task raises error."""
        graph = TaskGraph()
        task1 = ResearchTask(task_id="task1", task_type="test", description="Task 1")
        graph.add_task(task1)
        
        with pytest.raises(ValueError, match="Dependency task missing_task not found"):
            graph.add_dependency("task1", "missing_task")
    
    def test_get_ready_tasks(self):
        """Test getting tasks ready for execution."""
        graph = TaskGraph()
        
        task1 = ResearchTask(task_id="task1", task_type="test", description="Task 1")
        task2 = ResearchTask(task_id="task2", task_type="test", description="Task 2")
        task3 = ResearchTask(task_id="task3", task_type="test", description="Task 3")
        
        graph.add_task(task1)
        graph.add_task(task2)
        graph.add_task(task3)
        graph.add_dependency("task2", "task1")
        graph.add_dependency("task3", "task2")
        
        # Initially, only task1 should be ready
        ready_tasks = graph.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "task1"
        
        # Complete task1, now task2 should be ready
        task1.status = TaskStatus.COMPLETED
        ready_tasks = graph.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "task2"
        
        # Complete task2, now task3 should be ready
        task2.status = TaskStatus.COMPLETED
        ready_tasks = graph.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "task3"
    
    def test_validate_dag_valid(self):
        """Test DAG validation with valid graph."""
        graph = TaskGraph()
        
        task1 = ResearchTask(task_id="task1", task_type="test", description="Task 1")
        task2 = ResearchTask(task_id="task2", task_type="test", description="Task 2")
        task3 = ResearchTask(task_id="task3", task_type="test", description="Task 3")
        
        graph.add_task(task1)
        graph.add_task(task2)
        graph.add_task(task3)
        graph.add_dependency("task2", "task1")
        graph.add_dependency("task3", "task2")
        
        assert graph.validate_dag() is True
    
    def test_validate_dag_cycle(self):
        """Test DAG validation with cycle."""
        graph = TaskGraph()
        
        task1 = ResearchTask(task_id="task1", task_type="test", description="Task 1")
        task2 = ResearchTask(task_id="task2", task_type="test", description="Task 2")
        
        graph.add_task(task1)
        graph.add_task(task2)
        graph.add_dependency("task2", "task1")
        graph.add_dependency("task1", "task2")  # Creates cycle
        
        assert graph.validate_dag() is False
    
    def test_get_execution_order(self):
        """Test getting execution order for tasks."""
        graph = TaskGraph()
        
        task1 = ResearchTask(task_id="task1", task_type="test", description="Task 1")
        task2 = ResearchTask(task_id="task2", task_type="test", description="Task 2")
        task3 = ResearchTask(task_id="task3", task_type="test", description="Task 3")
        task4 = ResearchTask(task_id="task4", task_type="test", description="Task 4")
        
        graph.add_task(task1)
        graph.add_task(task2)
        graph.add_task(task3)
        graph.add_task(task4)
        
        # task1 and task2 can run in parallel
        # task3 depends on task1
        # task4 depends on both task2 and task3
        graph.add_dependency("task3", "task1")
        graph.add_dependency("task4", "task2")
        graph.add_dependency("task4", "task3")
        
        execution_order = graph.get_execution_order()
        
        # Should have 3 levels: [task1, task2], [task3], [task4]
        assert len(execution_order) == 3
        assert set(execution_order[0]) == {"task1", "task2"}
        assert execution_order[1] == ["task3"]
        assert execution_order[2] == ["task4"]


class TestQueryAnalyzer:
    """Test QueryAnalyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        return QueryAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_query_basic(self, analyzer):
        """Test basic query analysis."""
        query = "What are the latest developments in artificial intelligence?"
        
        analysis = await analyzer.analyze_query(query)
        
        assert "query_type" in analysis
        assert "complexity" in analysis
        assert "required_sources" in analysis
        assert "estimated_duration" in analysis
        assert "key_topics" in analysis
        assert "analysis_requirements" in analysis
        
        assert analysis["query_type"] == "explanatory"
        assert "web" in analysis["required_sources"]
    
    def test_classify_query_type(self, analyzer):
        """Test query type classification."""
        test_cases = [
            ("Compare machine learning and deep learning", "comparative"),
            ("What is the trend in AI research over the past decade?", "temporal"),
            ("How does neural network training work?", "explanatory"),
            ("Review of recent advances in computer vision", "survey"),
            ("Analyze the impact of AI on healthcare", "analytical"),
            ("Tell me about quantum computing", "general")
        ]
        
        for query, expected_type in test_cases:
            result = analyzer._classify_query_type(query)
            assert result == expected_type
    
    def test_assess_complexity(self, analyzer):
        """Test complexity assessment."""
        test_cases = [
            ("AI", "low"),
            ("What is machine learning and how does it work in practice and what are the benefits?", "medium"),
            ("Compare and analyze the various different approaches to neural network architectures and their impact on performance across multiple domains", "high")
        ]
        
        for query, expected_complexity in test_cases:
            result = analyzer._assess_complexity(query)
            assert result == expected_complexity
    
    def test_identify_required_sources(self, analyzer):
        """Test source identification."""
        test_cases = [
            ("What is AI?", ["web"]),
            ("Recent research papers on machine learning", ["web", "academic"]),
            ("Statistics on AI adoption in healthcare", ["web", "database"]),
            ("Academic study of neural networks with statistical data", ["web", "academic", "database"])
        ]
        
        for query, expected_sources in test_cases:
            result = analyzer._identify_required_sources(query)
            for source in expected_sources:
                assert source in result
    
    def test_extract_key_topics(self, analyzer):
        """Test key topic extraction."""
        query = "What are the latest developments in artificial intelligence and machine learning?"
        
        topics = analyzer._extract_key_topics(query)
        
        assert "latest" in topics
        assert "developments" in topics
        assert "artificial" in topics
        assert "intelligence" in topics
        assert "machine" in topics
        assert "learning" in topics
        
        # Should not include stop words
        assert "the" not in topics
        assert "are" not in topics
    
    def test_determine_analysis_needs(self, analyzer):
        """Test analysis needs determination."""
        test_cases = [
            ("Trends in AI over time", ["temporal_analysis"]),
            ("Public sentiment about artificial intelligence", ["sentiment_analysis"]),
            ("Topics in machine learning research", ["topic_modeling"]),
            ("Key people and organizations in AI", ["named_entity_recognition"]),
            ("Statistical analysis of AI performance", ["statistical_analysis"]),
            ("General AI overview", ["topic_modeling", "named_entity_recognition"])
        ]
        
        for query, expected_needs in test_cases:
            result = analyzer._determine_analysis_needs(query)
            for need in expected_needs:
                assert need in result


class TestPlannerAgent:
    """Test PlannerAgent functionality."""
    
    @pytest.fixture
    def planner(self):
        return PlannerAgent()
    
    @pytest.mark.asyncio
    async def test_execute_success(self, planner):
        """Test successful planner execution."""
        query = "What are the latest developments in artificial intelligence?"
        
        result = await planner.execute(query=query)
        
        assert result.success is True
        assert result.agent_name == "planner"
        assert isinstance(result.data, ResearchPlan)
        assert result.data.query == query
        assert len(result.data.tasks) > 0
    
    @pytest.mark.asyncio
    async def test_execute_missing_query(self, planner):
        """Test planner execution with missing query."""
        result = await planner.execute()
        
        assert result.success is False
        assert "missing 1 required positional argument: 'query'" in result.error
    
    @pytest.mark.asyncio
    async def test_create_research_plan_basic(self, planner):
        """Test basic research plan creation."""
        query = "What is machine learning?"
        
        plan = await planner._create_research_plan(query)
        
        assert isinstance(plan, ResearchPlan)
        assert plan.query == query
        assert len(plan.tasks) >= 4  # At least web, analysis, drafting, citation
        assert plan.estimated_duration_minutes > 0
        
        # Check that we have the expected task types
        task_types = [task.task_type for task in plan.tasks]
        assert "web_search" in task_types
        assert "content_analysis" in task_types
        assert "document_generation" in task_types
        assert "citation_formatting" in task_types
    
    @pytest.mark.asyncio
    async def test_create_research_plan_academic(self, planner):
        """Test research plan creation with academic sources."""
        query = "Recent research papers on neural network architectures"
        
        plan = await planner._create_research_plan(query)
        
        task_types = [task.task_type for task in plan.tasks]
        assert "academic_search" in task_types
        
        # Check dependencies
        analysis_task = next(task for task in plan.tasks if task.task_type == "content_analysis")
        assert "web_research" in analysis_task.dependencies
        assert "academic_research" in analysis_task.dependencies
    
    @pytest.mark.asyncio
    async def test_create_workflow_steps(self, planner):
        """Test workflow step creation from plan."""
        query = "What is AI?"
        plan = await planner._create_research_plan(query)
        
        workflow_steps = await planner.create_workflow_steps(plan)
        
        assert len(workflow_steps) == len(plan.tasks)
        
        for step in workflow_steps:
            assert isinstance(step, WorkflowStep)
            assert step.step_id
            assert step.step_name
            assert step.estimated_duration_minutes > 0
    
    @pytest.mark.asyncio
    async def test_update_plan_progress(self, planner):
        """Test updating plan progress."""
        query = "What is AI?"
        plan = await planner._create_research_plan(query)
        
        # Get first task
        first_task = plan.tasks[0]
        original_status = first_task.status
        
        # Update to in progress
        updated_plan = await planner.update_plan_progress(
            plan, first_task.task_id, TaskStatus.IN_PROGRESS
        )
        
        updated_task = updated_plan.get_task_by_id(first_task.task_id)
        assert updated_task.status == TaskStatus.IN_PROGRESS
        assert updated_task.started_at is not None
        assert updated_plan.status == TaskStatus.IN_PROGRESS
    
    @pytest.mark.asyncio
    async def test_update_plan_progress_completion(self, planner):
        """Test updating plan when all tasks complete."""
        query = "What is AI?"
        plan = await planner._create_research_plan(query)
        
        # Complete all tasks
        for task in plan.tasks:
            plan = await planner.update_plan_progress(
                plan, task.task_id, TaskStatus.COMPLETED, {"result": "test"}
            )
        
        assert plan.status == TaskStatus.COMPLETED
        
        for task in plan.tasks:
            assert task.status == TaskStatus.COMPLETED
            assert task.completed_at is not None
            assert task.result == {"result": "test"}
    
    @pytest.mark.asyncio
    async def test_update_plan_progress_invalid_task(self, planner):
        """Test updating progress for invalid task ID."""
        query = "What is AI?"
        plan = await planner._create_research_plan(query)
        
        with pytest.raises(ValueError, match="Task invalid_id not found"):
            await planner.update_plan_progress(
                plan, "invalid_id", TaskStatus.COMPLETED
            )
    
    def test_estimate_task_duration(self, planner):
        """Test task duration estimation."""
        test_cases = [
            ("web_search", 10),
            ("academic_search", 8),
            ("content_analysis", 15),
            ("document_generation", 12),
            ("citation_formatting", 5),
            ("unknown_type", 10)
        ]
        
        for task_type, expected_duration in test_cases:
            task = ResearchTask(
                task_id="test",
                task_type=task_type,
                description="Test task"
            )
            duration = planner._estimate_task_duration(task)
            assert duration == expected_duration


@pytest.mark.asyncio
async def test_integration_full_planning_workflow():
    """Test complete planning workflow integration."""
    planner = PlannerAgent()
    query = "Compare machine learning and deep learning approaches in computer vision"
    
    # Execute planner
    result = await planner.execute(query=query)
    
    assert result.success is True
    plan = result.data
    
    # Verify plan structure
    assert isinstance(plan, ResearchPlan)
    assert plan.query == query
    assert len(plan.tasks) >= 4
    
    # Verify task dependencies are valid
    task_graph = TaskGraph()
    for task in plan.tasks:
        task_graph.add_task(task)
    
    for task in plan.tasks:
        for dep_id in task.dependencies:
            task_graph.add_dependency(task.task_id, dep_id)
    
    assert task_graph.validate_dag() is True
    
    # Verify we can get ready tasks
    ready_tasks = plan.get_ready_tasks()
    assert len(ready_tasks) > 0
    
    # All ready tasks should have no dependencies or completed dependencies
    for task in ready_tasks:
        assert len(task.dependencies) == 0 or all(
            plan.get_task_by_id(dep_id).status == TaskStatus.COMPLETED 
            for dep_id in task.dependencies
        )
    
    # Create workflow steps
    workflow_steps = await planner.create_workflow_steps(plan)
    assert len(workflow_steps) == len(plan.tasks)
    
    # Verify workflow steps have proper structure
    for step in workflow_steps:
        assert step.step_id
        assert step.step_name
        assert step.estimated_duration_minutes > 0