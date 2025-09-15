"""
Unit tests for Agent Scrivener data models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from agent_scrivener.models.core import (
    Source, SourceType, ExtractedArticle, AcademicPaper, 
    Insight, ResearchPlan, ResearchSession, TaskStatus,
    DocumentSections, DocumentSection, ResearchTask, Citation,
    SessionState, AgentExecution, SessionMetrics, WorkflowStep
)


class TestSource:
    """Test cases for Source model."""
    
    def test_valid_source_creation(self):
        """Test creating a valid Source object."""
        source = Source(
            url="https://example.com",
            title="Test Article",
            author="Test Author",
            source_type=SourceType.WEB
        )
        
        assert str(source.url) == "https://example.com/"
        assert source.title == "Test Article"
        assert source.author == "Test Author"
        assert source.source_type == SourceType.WEB
        assert isinstance(source.metadata, dict)
    
    def test_invalid_url(self):
        """Test that invalid URLs raise validation errors."""
        with pytest.raises(ValidationError):
            Source(
                url="not-a-url",
                title="Test Article",
                source_type=SourceType.WEB
            )
    
    def test_empty_title(self):
        """Test that empty titles raise validation errors."""
        with pytest.raises(ValidationError):
            Source(
                url="https://example.com",
                title="",
                source_type=SourceType.WEB
            )


class TestExtractedArticle:
    """Test cases for ExtractedArticle model."""
    
    def test_valid_article_creation(self, sample_source):
        """Test creating a valid ExtractedArticle."""
        article = ExtractedArticle(
            source=sample_source,
            content="This is sample article content.",
            confidence_score=0.85
        )
        
        assert article.source == sample_source
        assert article.content == "This is sample article content."
        assert article.confidence_score == 0.85
        assert article.word_count == 5  # Auto-calculated
        assert isinstance(article.extraction_timestamp, datetime)
    
    def test_confidence_score_validation(self, sample_source):
        """Test confidence score validation."""
        # Valid confidence score
        article = ExtractedArticle(
            source=sample_source,
            content="Content",
            confidence_score=0.5
        )
        assert article.confidence_score == 0.5
        
        # Invalid confidence scores
        with pytest.raises(ValidationError):
            ExtractedArticle(
                source=sample_source,
                content="Content",
                confidence_score=1.5  # > 1.0
            )
        
        with pytest.raises(ValidationError):
            ExtractedArticle(
                source=sample_source,
                content="Content",
                confidence_score=-0.1  # < 0.0
            )


class TestAcademicPaper:
    """Test cases for AcademicPaper model."""
    
    def test_valid_paper_creation(self):
        """Test creating a valid AcademicPaper."""
        paper = AcademicPaper(
            title="Machine Learning Research",
            authors=["Smith, J.", "Doe, A."],
            abstract="This paper discusses machine learning applications.",
            publication_year=2023,
            doi="10.1234/example.2023.001",
            database_source="arXiv"
        )
        
        assert paper.title == "Machine Learning Research"
        assert len(paper.authors) == 2
        assert paper.publication_year == 2023
        assert paper.doi == "10.1234/example.2023.001"
    
    def test_doi_validation(self):
        """Test DOI format validation."""
        # Valid DOI
        paper = AcademicPaper(
            title="Test Paper",
            authors=["Author"],
            abstract="Abstract",
            publication_year=2023,
            doi="10.1234/test.2023.001",
            database_source="test"
        )
        assert paper.doi == "10.1234/test.2023.001"
        
        # Invalid DOI format
        with pytest.raises(ValidationError):
            AcademicPaper(
                title="Test Paper",
                authors=["Author"],
                abstract="Abstract",
                publication_year=2023,
                doi="invalid-doi",
                database_source="test"
            )
    
    def test_publication_year_validation(self):
        """Test publication year validation."""
        # Valid year
        paper = AcademicPaper(
            title="Test Paper",
            authors=["Author"],
            abstract="Abstract",
            publication_year=2023,
            database_source="test"
        )
        assert paper.publication_year == 2023
        
        # Invalid years
        with pytest.raises(ValidationError):
            AcademicPaper(
                title="Test Paper",
                authors=["Author"],
                abstract="Abstract",
                publication_year=1800,  # Too old
                database_source="test"
            )
        
        with pytest.raises(ValidationError):
            AcademicPaper(
                title="Test Paper",
                authors=["Author"],
                abstract="Abstract",
                publication_year=2050,  # Too far in future
                database_source="test"
            )


class TestResearchPlan:
    """Test cases for ResearchPlan model."""
    
    def test_valid_plan_creation(self):
        """Test creating a valid ResearchPlan."""
        plan = ResearchPlan(
            query="Test research query",
            session_id="session_001",
            estimated_duration_minutes=30
        )
        
        assert plan.query == "Test research query"
        assert plan.session_id == "session_001"
        assert plan.estimated_duration_minutes == 30
        assert plan.status == TaskStatus.PENDING
        assert isinstance(plan.created_at, datetime)
    
    def test_get_ready_tasks(self):
        """Test getting ready tasks from plan."""
        from agent_scrivener.models.core import ResearchTask
        
        plan = ResearchPlan(
            query="Test query",
            session_id="session_001",
            estimated_duration_minutes=30
        )
        
        # Add tasks with dependencies
        task1 = ResearchTask(
            task_id="task1",
            task_type="research",
            description="First task"
        )
        
        task2 = ResearchTask(
            task_id="task2",
            task_type="analysis",
            description="Second task",
            dependencies=["task1"]
        )
        
        task3 = ResearchTask(
            task_id="task3",
            task_type="synthesis",
            description="Third task",
            dependencies=["task1", "task2"]
        )
        
        plan.tasks = [task1, task2, task3]
        
        # Initially, only task1 should be ready
        ready_tasks = plan.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "task1"
        
        # Complete task1
        task1.status = TaskStatus.COMPLETED
        ready_tasks = plan.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "task2"
        
        # Complete task2
        task2.status = TaskStatus.COMPLETED
        ready_tasks = plan.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert ready_tasks[0].task_id == "task3"


class TestInsight:
    """Test cases for Insight model."""
    
    def test_valid_insight_creation(self, sample_source):
        """Test creating a valid Insight."""
        insight = Insight(
            topic="Machine Learning",
            summary="ML is transforming research automation",
            supporting_evidence=["Evidence 1", "Evidence 2"],
            confidence_score=0.9,
            related_sources=[sample_source],
            tags=["AI", "automation"]
        )
        
        assert insight.topic == "Machine Learning"
        assert insight.summary == "ML is transforming research automation"
        assert len(insight.supporting_evidence) == 2
        assert insight.confidence_score == 0.9
        assert len(insight.related_sources) == 1
        assert "AI" in insight.tags
        assert isinstance(insight.created_at, datetime)
    
    def test_insight_confidence_validation(self, sample_source):
        """Test insight confidence score validation."""
        # Valid confidence score
        insight = Insight(
            topic="Test Topic",
            summary="Test summary",
            confidence_score=0.75,
            related_sources=[sample_source]
        )
        assert insight.confidence_score == 0.75
        
        # Invalid confidence scores
        with pytest.raises(ValidationError):
            Insight(
                topic="Test Topic",
                summary="Test summary",
                confidence_score=1.5,  # > 1.0
                related_sources=[sample_source]
            )


class TestDocumentSections:
    """Test cases for DocumentSections model."""
    
    def test_valid_document_sections_creation(self):
        """Test creating valid DocumentSections."""
        
        intro = DocumentSection(
            title="Introduction",
            content="This is the introduction section.",
            section_type="introduction",
            order=1
        )
        
        methodology = DocumentSection(
            title="Methodology",
            content="This describes the methodology.",
            section_type="methodology",
            order=2
        )
        
        findings = DocumentSection(
            title="Findings",
            content="These are the key findings.",
            section_type="findings",
            order=3
        )
        
        conclusion = DocumentSection(
            title="Conclusion",
            content="This is the conclusion.",
            section_type="conclusion",
            order=4
        )
        
        doc_sections = DocumentSections(
            introduction=intro,
            methodology=methodology,
            findings=findings,
            conclusion=conclusion
        )
        
        assert doc_sections.introduction.title == "Introduction"
        assert doc_sections.methodology.title == "Methodology"
        assert doc_sections.findings.title == "Findings"
        assert doc_sections.conclusion.title == "Conclusion"
    
    def test_get_all_sections_ordering(self):
        """Test that sections are returned in correct order."""
        
        # Create sections with mixed order
        intro = DocumentSection(title="Intro", content="Content", section_type="intro", order=1)
        conclusion = DocumentSection(title="Conclusion", content="Content", section_type="conclusion", order=4)
        methodology = DocumentSection(title="Method", content="Content", section_type="method", order=2)
        findings = DocumentSection(title="Findings", content="Content", section_type="findings", order=3)
        
        # Add additional sections
        extra_section = DocumentSection(title="Extra", content="Content", section_type="extra", order=5)
        
        doc_sections = DocumentSections(
            introduction=intro,
            methodology=methodology,
            findings=findings,
            conclusion=conclusion,
            sections=[extra_section]
        )
        
        all_sections = doc_sections.get_all_sections()
        assert len(all_sections) == 5
        assert all_sections[0].title == "Intro"
        assert all_sections[1].title == "Method"
        assert all_sections[2].title == "Findings"
        assert all_sections[3].title == "Conclusion"
        assert all_sections[4].title == "Extra"


class TestResearchTask:
    """Test cases for ResearchTask model."""
    
    def test_valid_task_creation(self):
        """Test creating a valid ResearchTask."""
        
        task = ResearchTask(
            task_id="task_001",
            task_type="web_search",
            description="Search for information about AI",
            parameters={"query": "artificial intelligence", "max_results": 10},
            dependencies=["task_000"]
        )
        
        assert task.task_id == "task_001"
        assert task.task_type == "web_search"
        assert task.description == "Search for information about AI"
        assert task.parameters["query"] == "artificial intelligence"
        assert "task_000" in task.dependencies
        assert task.status == TaskStatus.PENDING
        assert isinstance(task.created_at, datetime)
    
    def test_task_status_transitions(self):
        """Test task status transitions."""
        
        task = ResearchTask(
            task_id="task_001",
            task_type="test",
            description="Test task"
        )
        
        # Initial state
        assert task.status == TaskStatus.PENDING
        assert task.started_at is None
        assert task.completed_at is None
        
        # Update to in progress
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None
        
        # Update to completed
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.result = {"output": "Task completed successfully"}
        
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task.result["output"] == "Task completed successfully"


class TestCitation:
    """Test cases for Citation model."""
    
    def test_valid_citation_creation(self, sample_source):
        """Test creating a valid Citation."""
        
        citation = Citation(
            citation_id="cite_001",
            source=sample_source,
            citation_text="Smith, J. (2023). Sample Article. Example.com.",
            page_numbers="pp. 15-20",
            quote="This is a direct quote from the source.",
            context="This quote supports the argument about..."
        )
        
        assert citation.citation_id == "cite_001"
        assert citation.source == sample_source
        assert citation.citation_text.startswith("Smith, J.")
        assert citation.page_numbers == "pp. 15-20"
        assert citation.quote.startswith("This is a direct quote")
        assert isinstance(citation.created_at, datetime)


class TestResearchSession:
    """Test cases for ResearchSession model."""
    
    def test_valid_session_creation(self, sample_research_plan):
        """Test creating a valid ResearchSession."""
        session = ResearchSession(
            session_id="session_001",
            original_query="Test query",
            plan=sample_research_plan
        )
        
        assert session.session_id == "session_001"
        assert session.original_query == "Test query"
        assert session.plan == sample_research_plan
        assert session.status == TaskStatus.PENDING
    
    def test_update_timestamp(self, sample_research_session):
        """Test updating session timestamp."""
        import time
        original_time = sample_research_session.updated_at
        time.sleep(0.001)  # Ensure time difference
        sample_research_session.update_timestamp()
        
        assert sample_research_session.updated_at > original_time
    
    def test_get_all_sources(self, sample_research_session, sample_source):
        """Test getting all sources from session."""
        # Add an extracted article
        from agent_scrivener.models.core import ExtractedArticle
        article = ExtractedArticle(
            source=sample_source,
            content="Test content",
            confidence_score=0.8
        )
        sample_research_session.extracted_articles.append(article)
        
        # Add an academic paper
        from agent_scrivener.models.core import AcademicPaper
        paper = AcademicPaper(
            title="Test Paper",
            authors=["Author"],
            abstract="Abstract",
            publication_year=2023,
            database_source="test"
        )
        sample_research_session.academic_papers.append(paper)
        
        sources = sample_research_session.get_all_sources()
        assert len(sources) >= 2  # At least the article source and paper source
    
    def test_session_state_management(self, sample_research_session, sample_source):
        """Test session state management capabilities."""
        # Add insights
        insight = Insight(
            topic="Test Topic",
            summary="Test insight summary",
            confidence_score=0.8,
            related_sources=[]
        )
        sample_research_session.insights.append(insight)
        
        # Add citations with a valid source
        citation = Citation(
            citation_id="cite_001",
            source=sample_source,
            citation_text="Test citation"
        )
        sample_research_session.citations.append(citation)
        
        # Update session
        sample_research_session.update_timestamp()
        
        assert len(sample_research_session.insights) == 1
        assert sample_research_session.insights[0].topic == "Test Topic"
        assert len(sample_research_session.citations) == 1
        assert sample_research_session.citations[0].citation_id == "cite_001"


class TestAgentExecution:
    """Test cases for AgentExecution model."""
    
    def test_valid_execution_creation(self):
        """Test creating a valid AgentExecution."""
        execution = AgentExecution(
            execution_id="exec_001",
            agent_name="research_agent",
            task_id="task_001",
            input_data={"query": "test query"}
        )
        
        assert execution.execution_id == "exec_001"
        assert execution.agent_name == "research_agent"
        assert execution.task_id == "task_001"
        assert execution.status == TaskStatus.PENDING
        assert execution.input_data["query"] == "test query"
        assert isinstance(execution.started_at, datetime)
    
    def test_mark_completed(self):
        """Test marking execution as completed."""
        execution = AgentExecution(
            execution_id="exec_001",
            agent_name="research_agent",
            task_id="task_001"
        )
        
        output_data = {"results": ["result1", "result2"]}
        execution.mark_completed(output_data)
        
        assert execution.status == TaskStatus.COMPLETED
        assert execution.output_data == output_data
        assert execution.completed_at is not None
        assert execution.execution_time_seconds is not None
        assert execution.execution_time_seconds >= 0
    
    def test_mark_failed(self):
        """Test marking execution as failed."""
        execution = AgentExecution(
            execution_id="exec_001",
            agent_name="research_agent",
            task_id="task_001"
        )
        
        error_message = "Network timeout error"
        execution.mark_failed(error_message)
        
        assert execution.status == TaskStatus.FAILED
        assert execution.error_message == error_message
        assert execution.completed_at is not None
        assert execution.execution_time_seconds is not None


class TestSessionMetrics:
    """Test cases for SessionMetrics model."""
    
    def test_calculate_success_rate(self):
        """Test calculating success rate."""
        metrics = SessionMetrics(
            successful_agent_executions=8,
            failed_agent_executions=2
        )
        
        success_rate = metrics.calculate_success_rate()
        assert success_rate == 0.8
        
        # Test with no executions
        empty_metrics = SessionMetrics()
        assert empty_metrics.calculate_success_rate() == 0.0
    
    def test_update_from_session(self, sample_research_session, sample_source):
        """Test updating metrics from session."""
        # Add some data to the session
        article = ExtractedArticle(
            source=sample_source,
            content="Test content",
            confidence_score=0.9
        )
        sample_research_session.extracted_articles.append(article)
        
        insight = Insight(
            topic="Test Topic",
            summary="Test summary",
            confidence_score=0.8,
            related_sources=[sample_source]
        )
        sample_research_session.insights.append(insight)
        
        sample_research_session.final_document = "This is a test document with multiple words."
        
        # Update metrics
        sample_research_session.metrics.update_from_session(sample_research_session)
        
        assert sample_research_session.metrics.total_sources_processed == 1
        assert sample_research_session.metrics.average_source_confidence == 0.9
        assert sample_research_session.metrics.total_insights_generated == 1
        assert sample_research_session.metrics.average_insight_confidence == 0.8
        assert sample_research_session.metrics.final_document_word_count == 8  # "This is a test document with multiple words."


class TestWorkflowStep:
    """Test cases for WorkflowStep model."""
    
    def test_valid_workflow_step_creation(self):
        """Test creating a valid WorkflowStep."""
        step = WorkflowStep(
            step_id="step_001",
            step_name="Web Research",
            description="Search and extract information from web sources",
            required_inputs=["research_query"],
            expected_outputs=["extracted_articles"],
            estimated_duration_minutes=15
        )
        
        assert step.step_id == "step_001"
        assert step.step_name == "Web Research"
        assert step.status == TaskStatus.PENDING
        assert "research_query" in step.required_inputs
        assert "extracted_articles" in step.expected_outputs
    
    def test_is_ready_to_execute(self):
        """Test checking if step is ready to execute."""
        step = WorkflowStep(
            step_id="step_002",
            step_name="Analysis",
            description="Analyze extracted content",
            required_inputs=["extracted_articles", "academic_papers"],
            expected_outputs=["insights"],
            estimated_duration_minutes=10
        )
        
        # Not ready - missing inputs
        assert not step.is_ready_to_execute(["extracted_articles"])
        
        # Ready - all inputs available
        assert step.is_ready_to_execute(["extracted_articles", "academic_papers", "extra_input"])
    
    def test_add_execution(self):
        """Test adding agent execution to workflow step."""
        step = WorkflowStep(
            step_id="step_001",
            step_name="Research",
            description="Research step",
            estimated_duration_minutes=10
        )
        
        execution = AgentExecution(
            execution_id="exec_001",
            agent_name="research_agent",
            task_id="task_001"
        )
        
        step.add_execution(execution)
        
        assert len(step.agent_executions) == 1
        assert step.agent_executions[0] == execution
        
        # Test getting execution by agent
        found_execution = step.get_execution_by_agent("research_agent")
        assert found_execution == execution
        
        # Test getting non-existent execution
        not_found = step.get_execution_by_agent("non_existent_agent")
        assert not_found is None


class TestEnhancedResearchSession:
    """Test cases for enhanced ResearchSession workflow features."""
    
    def test_workflow_step_management(self, sample_research_session):
        """Test workflow step management in research session."""
        step1 = WorkflowStep(
            step_id="step_001",
            step_name="Research",
            description="Research step",
            expected_outputs=["articles"],
            estimated_duration_minutes=10
        )
        
        step2 = WorkflowStep(
            step_id="step_002",
            step_name="Analysis",
            description="Analysis step",
            required_inputs=["articles"],
            expected_outputs=["insights"],
            estimated_duration_minutes=5
        )
        
        sample_research_session.add_workflow_step(step1)
        sample_research_session.add_workflow_step(step2)
        
        assert len(sample_research_session.workflow_steps) == 2
        
        # Initially, only step1 should be ready (no required inputs)
        ready_steps = sample_research_session.get_ready_workflow_steps()
        assert len(ready_steps) == 1
        assert ready_steps[0].step_id == "step_001"
        
        # Complete step1
        step1.status = TaskStatus.COMPLETED
        ready_steps = sample_research_session.get_ready_workflow_steps()
        assert len(ready_steps) == 1
        assert ready_steps[0].step_id == "step_002"
    
    def test_session_state_transitions(self, sample_research_session):
        """Test session state transitions."""
        assert sample_research_session.session_state == SessionState.INITIALIZING
        
        sample_research_session.transition_state(SessionState.PLANNING)
        assert sample_research_session.session_state == SessionState.PLANNING
        
        sample_research_session.transition_state(SessionState.RESEARCHING)
        assert sample_research_session.session_state == SessionState.RESEARCHING
    
    def test_progress_calculation(self, sample_research_session):
        """Test progress percentage calculation."""
        # No steps - 0% progress
        assert sample_research_session.calculate_progress_percentage() == 0.0
        
        # Add steps
        step1 = WorkflowStep(step_id="1", step_name="Step 1", description="Desc", estimated_duration_minutes=5)
        step2 = WorkflowStep(step_id="2", step_name="Step 2", description="Desc", estimated_duration_minutes=5)
        step3 = WorkflowStep(step_id="3", step_name="Step 3", description="Desc", estimated_duration_minutes=5)
        
        sample_research_session.workflow_steps = [step1, step2, step3]
        
        # No completed steps - 0%
        assert sample_research_session.calculate_progress_percentage() == 0.0
        
        # One completed step - 33.33%
        step1.status = TaskStatus.COMPLETED
        progress = sample_research_session.calculate_progress_percentage()
        assert abs(progress - 33.33333333333333) < 0.001
        
        # Two completed steps - 66.67%
        step2.status = TaskStatus.COMPLETED
        progress = sample_research_session.calculate_progress_percentage()
        assert abs(progress - 66.66666666666666) < 0.001
        
        # All completed - 100%
        step3.status = TaskStatus.COMPLETED
        assert sample_research_session.calculate_progress_percentage() == 100.0
    
    def test_session_summary(self, sample_research_session):
        """Test getting session summary."""
        summary = sample_research_session.get_session_summary()
        
        assert summary["session_id"] == sample_research_session.session_id
        assert summary["query"] == sample_research_session.original_query
        assert summary["status"] == sample_research_session.status.value
        assert summary["session_state"] == sample_research_session.session_state.value
        assert "progress_percentage" in summary
        assert "total_sources" in summary
        assert "metrics" in summary