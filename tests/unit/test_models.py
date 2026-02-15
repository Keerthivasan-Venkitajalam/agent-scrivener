"""
Unit tests for Agent Scrivener data models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
from hypothesis import given, strategies as st

from agent_scrivener.models.core import (
    Source, SourceType, ExtractedArticle, AcademicPaper, 
    Insight, ResearchPlan, ResearchSession, TaskStatus,
    DocumentSections, DocumentSection, ResearchTask, Citation,
    SessionState, AgentExecution, SessionMetrics, WorkflowStep
)
from agent_scrivener.models.analysis import TopicModel


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
    
    def test_datetime_serialization_with_none(self):
        """Test serialization with None datetime values."""
        source = Source(
            url="https://example.com/article",
            title="Test Article",
            source_type=SourceType.WEB,
            publication_date=None
        )
        
        # Serialize to dict with mode='json' to trigger JSON serialization
        serialized = source.model_dump(mode='json')
        
        # Check that None datetime is preserved
        assert serialized['publication_date'] is None
    
    def test_datetime_serialization_with_valid_datetime(self):
        """Test serialization with valid datetime objects."""
        test_date = datetime(2024, 3, 15, 10, 30, 45)
        source = Source(
            url="https://example.com/article",
            title="Test Article",
            author="Test Author",
            publication_date=test_date,
            source_type=SourceType.WEB
        )
        
        # Serialize to dict with mode='json' to trigger JSON serialization
        serialized = source.model_dump(mode='json')
        
        # Check that datetime is serialized to ISO format string
        assert serialized['publication_date'] == '2024-03-15T10:30:45'
        assert isinstance(serialized['publication_date'], str)
    
    def test_datetime_serialization_iso_format(self):
        """Test that serialized output matches ISO format."""
        # Test with various datetime values
        test_cases = [
            (datetime(2023, 1, 1, 0, 0, 0), '2023-01-01T00:00:00'),
            (datetime(2024, 12, 31, 23, 59, 59), '2024-12-31T23:59:59'),
            (datetime(2024, 6, 15, 12, 30, 45), '2024-06-15T12:30:45'),
        ]
        
        for test_date, expected_iso in test_cases:
            source = Source(
                url="https://example.com/article",
                title="Test Article",
                publication_date=test_date,
                source_type=SourceType.WEB
            )
            
            serialized = source.model_dump(mode='json')
            assert serialized['publication_date'] == expected_iso
    
    @given(
        url=st.from_regex(r'https://[a-z0-9]+\.com(/[a-z0-9]+)?', fullmatch=True),
        title=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
        author=st.one_of(st.none(), st.text(min_size=1, max_size=200)),
        publication_date=st.one_of(
            st.none(),
            st.datetimes(
                min_value=datetime(1900, 1, 1),
                max_value=datetime(2030, 12, 31)
            )
        ),
        source_type=st.sampled_from(list(SourceType))
    )
    def test_property_serialization_equivalence(
        self, url, title, author, publication_date, source_type
    ):
        """
        Property Test: Serialization Equivalence
        
        **Validates: Requirements 2.3, 5.2**
        
        For any Source instance with datetime fields, serializing to JSON
        SHALL produce consistent ISO format strings.
        
        This property verifies that:
        1. Datetime values are serialized to ISO format strings
        2. None datetime values remain None
        3. Serialization is consistent across all valid inputs
        """
        # Create Source instance with generated values
        source = Source(
            url=url,
            title=title,
            author=author,
            publication_date=publication_date,
            source_type=source_type
        )
        
        # Serialize to JSON format
        serialized = source.model_dump(mode='json')
        
        # Property 1: If publication_date is None, it should remain None
        if publication_date is None:
            assert serialized['publication_date'] is None
        else:
            # Property 2: If publication_date is a datetime, it should be serialized to ISO format string
            assert isinstance(serialized['publication_date'], str)
            
            # Property 3: The serialized string should match the ISO format
            expected_iso = publication_date.isoformat()
            assert serialized['publication_date'] == expected_iso
            
            # Property 4: The serialized string should be parseable back to datetime
            parsed_date = datetime.fromisoformat(serialized['publication_date'])
            assert parsed_date == publication_date
        
        # Property 5: All other fields should be preserved correctly
        assert serialized['title'] == title
        assert serialized['author'] == author
        assert serialized['source_type'] == source_type.value


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
    
    def test_authors_min_length_empty_list(self):
        """Test validation fails with empty list.
        
        **Validates: Requirements 3.2**
        """
        with pytest.raises(ValidationError) as exc_info:
            AcademicPaper(
                title="Test Paper",
                authors=[],  # Empty list should fail min_length=1
                abstract="Abstract content",
                publication_year=2023,
                database_source="test"
            )
        
        # Verify error message is clear and specific
        error = exc_info.value
        assert "authors" in str(error).lower()
        assert len(error.errors()) > 0
        # Check that the error is about list length
        error_dict = error.errors()[0]
        assert error_dict['loc'] == ('authors',)
        assert 'at least 1' in str(error_dict['msg']).lower() or 'min_length' in str(error_dict['type']).lower()
    
    def test_authors_min_length_single_author(self):
        """Test validation succeeds with list of length 1.
        
        **Validates: Requirements 3.2**
        """
        paper = AcademicPaper(
            title="Test Paper",
            authors=["Smith, J."],  # Single author should pass
            abstract="Abstract content",
            publication_year=2023,
            database_source="test"
        )
        
        assert len(paper.authors) == 1
        assert paper.authors[0] == "Smith, J."
    
    def test_authors_min_length_multiple_authors(self):
        """Test validation succeeds with longer lists.
        
        **Validates: Requirements 3.2**
        """
        # Test with 2 authors
        paper2 = AcademicPaper(
            title="Test Paper",
            authors=["Smith, J.", "Doe, A."],
            abstract="Abstract content",
            publication_year=2023,
            database_source="test"
        )
        assert len(paper2.authors) == 2
        
        # Test with 5 authors
        paper5 = AcademicPaper(
            title="Test Paper",
            authors=["Author 1", "Author 2", "Author 3", "Author 4", "Author 5"],
            abstract="Abstract content",
            publication_year=2023,
            database_source="test"
        )
        assert len(paper5.authors) == 5
        
        # Test with 10 authors (larger list)
        paper10 = AcademicPaper(
            title="Test Paper",
            authors=[f"Author {i}" for i in range(1, 11)],
            abstract="Abstract content",
            publication_year=2023,
            database_source="test"
        )
        assert len(paper10.authors) == 10
    
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

    @given(
        title=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
        authors=st.lists(
            st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
            min_size=0,  # Include empty lists to test validation failure
            max_size=10
        ),
        abstract=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip()),
        publication_year=st.integers(min_value=1900, max_value=2030),
        database_source=st.text(min_size=1, max_size=100).filter(lambda x: x.strip())
    )
    def test_property_validation_equivalence_authors(
        self, title, authors, abstract, publication_year, database_source
    ):
        """
        Property Test: Validation Equivalence for List Fields

        **Validates: Requirements 3.2, 5.3**

        For any input data to AcademicPaper model, validation SHALL succeed or fail
        based on the min_length constraint, with clear error messages.

        This property verifies that:
        1. Empty lists (length 0) fail validation with min_length error
        2. Lists with length >= 1 pass validation
        3. Error messages clearly indicate the constraint violation
        4. Validation behavior is consistent across all inputs
        """
        if len(authors) == 0:
            # Property 1: Empty lists should fail validation
            with pytest.raises(ValidationError) as exc_info:
                AcademicPaper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    publication_year=publication_year,
                    database_source=database_source
                )

            # Property 2: Error should be about the authors field
            error = exc_info.value
            errors = error.errors()
            assert len(errors) > 0

            # Property 3: Error location should point to 'authors' field
            assert any(err['loc'] == ('authors',) for err in errors)

            # Property 4: Error message should indicate minimum length constraint
            authors_error = next(err for err in errors if err['loc'] == ('authors',))
            error_msg = str(authors_error['msg']).lower()
            error_type = str(authors_error['type']).lower()

            # Check that error mentions the constraint (either in message or type)
            assert ('at least 1' in error_msg or
                    'min_length' in error_type or
                    'too_short' in error_type or
                    'list should have at least 1 item' in error_msg)
        else:
            # Property 5: Non-empty lists should pass validation
            paper = AcademicPaper(
                title=title,
                authors=authors,
                abstract=abstract,
                publication_year=publication_year,
                database_source=database_source
            )

            # Property 6: Validated data should match input
            assert paper.authors == authors
            assert len(paper.authors) >= 1
            assert paper.title == title
            assert paper.abstract == abstract
            assert paper.publication_year == publication_year
            assert paper.database_source == database_source



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




class TestTopicModel:
    """Test cases for TopicModel validation."""

    def test_valid_topic_model_creation(self):
        """Test creating a valid TopicModel."""
        from agent_scrivener.models.analysis import TopicModel

        topic = TopicModel(
            topic_id=0,
            keywords=["machine", "learning", "algorithm"],
            weight=0.8,
            description="Machine Learning Algorithms"
        )

        assert topic.topic_id == 0
        assert len(topic.keywords) == 3
        assert topic.weight == 0.8
        assert topic.description == "Machine Learning Algorithms"

    def test_keywords_min_length_empty_list(self):
        """Test validation fails with empty keywords list.

        **Validates: Requirements 3.2**
        """
        from agent_scrivener.models.analysis import TopicModel

        with pytest.raises(ValidationError) as exc_info:
            TopicModel(
                topic_id=0,
                keywords=[],  # Empty list should fail min_length=1
                weight=0.5
            )

        # Verify error message is clear and specific
        error = exc_info.value
        assert "keywords" in str(error).lower()
        assert len(error.errors()) > 0
        # Check that the error is about list length
        error_dict = error.errors()[0]
        assert error_dict['loc'] == ('keywords',)
        assert 'at least 1' in str(error_dict['msg']).lower() or 'min_length' in str(error_dict['type']).lower()

    def test_keywords_min_length_single_keyword(self):
        """Test validation succeeds with list of length 1.

        **Validates: Requirements 3.2**
        """
        from agent_scrivener.models.analysis import TopicModel

        topic = TopicModel(
            topic_id=0,
            keywords=["machine"],  # Single keyword should pass
            weight=0.5
        )

        assert len(topic.keywords) == 1
        assert topic.keywords[0] == "machine"

    @given(
        topic_id=st.integers(min_value=0, max_value=1000),
        keywords=st.lists(
            st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
            min_size=0,  # Include empty lists to test validation failure
            max_size=20
        ),
        weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        description=st.one_of(st.none(), st.text(min_size=1, max_size=200))
    )
    def test_property_validation_equivalence_keywords(
        self, topic_id, keywords, weight, description
    ):
        """
        Property Test: Validation Equivalence for TopicModel Keywords

        **Validates: Requirements 3.2, 5.3**

        For any input data to TopicModel, validation SHALL succeed or fail
        based on the min_length constraint on keywords field, with clear error messages.

        This property verifies that:
        1. Empty keyword lists (length 0) fail validation with min_length error
        2. Lists with length >= 1 pass validation
        3. Error messages clearly indicate the constraint violation
        4. Validation behavior is consistent across all inputs
        """
        from agent_scrivener.models.analysis import TopicModel

        if len(keywords) == 0:
            # Property 1: Empty lists should fail validation
            with pytest.raises(ValidationError) as exc_info:
                TopicModel(
                    topic_id=topic_id,
                    keywords=keywords,
                    weight=weight,
                    description=description
                )

            # Property 2: Error should be about the keywords field
            error = exc_info.value
            errors = error.errors()
            assert len(errors) > 0

            # Property 3: Error location should point to 'keywords' field
            assert any(err['loc'] == ('keywords',) for err in errors)

            # Property 4: Error message should indicate minimum length constraint
            keywords_error = next(err for err in errors if err['loc'] == ('keywords',))
            error_msg = str(keywords_error['msg']).lower()
            error_type = str(keywords_error['type']).lower()

            # Check that error mentions the constraint (either in message or type)
            assert ('at least 1' in error_msg or
                    'min_length' in error_type or
                    'too_short' in error_type or
                    'list should have at least 1 item' in error_msg)
        else:
            # Property 5: Non-empty lists should pass validation
            topic = TopicModel(
                topic_id=topic_id,
                keywords=keywords,
                weight=weight,
                description=description
            )

            # Property 6: Validated data should match input
            assert topic.keywords == keywords
            assert len(topic.keywords) >= 1
            assert topic.topic_id == topic_id
            assert topic.weight == weight
            assert topic.description == description




class TestAnalysisResultsConfigDict:
    """Test cases for AnalysisResults ConfigDict behavior.
    
    **Validates: Requirements 4.2**
    """

    def test_json_schema_extra_appears_in_schema(self):
        """Test that json_schema_extra appears in generated schema.
        
        **Validates: Requirements 4.2**
        """
        from agent_scrivener.models.analysis import AnalysisResults

        # Generate JSON schema for the model
        schema = AnalysisResults.model_json_schema()

        # Verify that the schema contains the example data from json_schema_extra
        assert "example" in schema
        assert isinstance(schema["example"], dict)
        
        # Verify the example contains expected fields
        example = schema["example"]
        assert "session_id" in example
        assert example["session_id"] == "research_001"
        assert "named_entities" in example
        assert "topics" in example
        assert "key_themes" in example
        assert "sentiment_scores" in example

    def test_example_data_included_in_schema(self):
        """Test that example data is included in schema with correct structure.
        
        **Validates: Requirements 4.2**
        """
        from agent_scrivener.models.analysis import AnalysisResults

        schema = AnalysisResults.model_json_schema()
        example = schema.get("example", {})

        # Verify named_entities example structure
        assert len(example.get("named_entities", [])) > 0
        entity_example = example["named_entities"][0]
        assert "text" in entity_example
        assert "label" in entity_example
        assert "confidence_score" in entity_example
        assert entity_example["text"] == "Machine Learning"
        assert entity_example["label"] == "TECHNOLOGY"

        # Verify topics example structure
        assert len(example.get("topics", [])) > 0
        topic_example = example["topics"][0]
        assert "topic_id" in topic_example
        assert "keywords" in topic_example
        assert "weight" in topic_example
        assert topic_example["topic_id"] == 0
        assert isinstance(topic_example["keywords"], list)
        assert len(topic_example["keywords"]) >= 1

        # Verify key_themes example
        assert isinstance(example.get("key_themes", []), list)
        assert "automation" in example["key_themes"]

        # Verify sentiment_scores example
        assert isinstance(example.get("sentiment_scores", {}), dict)
        assert "overall" in example["sentiment_scores"]

    def test_model_instantiation_works_identically(self):
        """Test model instantiation works identically with ConfigDict.
        
        **Validates: Requirements 4.2**
        
        This test verifies that the migration from class Config to ConfigDict
        does not affect model instantiation behavior.
        """
        from agent_scrivener.models.analysis import AnalysisResults
        from datetime import datetime

        # Test basic instantiation
        result = AnalysisResults(
            session_id="test_session_001"
        )

        assert result.session_id == "test_session_001"
        assert isinstance(result.named_entities, list)
        assert len(result.named_entities) == 0
        assert isinstance(result.topics, list)
        assert isinstance(result.key_themes, list)
        assert isinstance(result.sentiment_scores, dict)
        assert isinstance(result.analysis_timestamp, datetime)

    def test_model_instantiation_with_full_data(self):
        """Test model instantiation with complete data.
        
        **Validates: Requirements 4.2**
        """
        from agent_scrivener.models.analysis import (
            AnalysisResults, NamedEntity, TopicModel, StatisticalSummary
        )
        from agent_scrivener.models.core import Source, SourceType
        from datetime import datetime

        # Create sample data
        entity = NamedEntity(
            text="Python",
            label="TECHNOLOGY",
            confidence_score=0.95,
            start_pos=0,
            end_pos=6
        )

        topic = TopicModel(
            topic_id=1,
            keywords=["programming", "python"],
            weight=0.85
        )

        summary = StatisticalSummary(
            metric_name="average_score",
            value=0.75
        )

        source = Source(
            url="https://example.com",
            title="Test Source",
            source_type=SourceType.WEB
        )

        # Create AnalysisResults with full data
        result = AnalysisResults(
            session_id="full_test_001",
            named_entities=[entity],
            topics=[topic],
            statistical_summaries=[summary],
            key_themes=["AI", "ML"],
            sentiment_scores={"positive": 0.8},
            processed_sources=[source],
            processing_time_seconds=1.5
        )

        # Verify all fields are correctly set
        assert result.session_id == "full_test_001"
        assert len(result.named_entities) == 1
        assert result.named_entities[0].text == "Python"
        assert len(result.topics) == 1
        assert result.topics[0].topic_id == 1
        assert len(result.statistical_summaries) == 1
        assert result.statistical_summaries[0].metric_name == "average_score"
        assert len(result.key_themes) == 2
        assert "AI" in result.key_themes
        assert result.sentiment_scores["positive"] == 0.8
        assert len(result.processed_sources) == 1
        assert result.processing_time_seconds == 1.5

    def test_model_serialization_with_configdict(self):
        """Test that model serialization works correctly with ConfigDict.
        
        **Validates: Requirements 4.2**
        """
        from agent_scrivener.models.analysis import AnalysisResults

        result = AnalysisResults(
            session_id="serialize_test_001",
            key_themes=["theme1", "theme2"],
            sentiment_scores={"overall": 0.7}
        )

        # Serialize to dict
        serialized = result.model_dump()

        assert serialized["session_id"] == "serialize_test_001"
        assert serialized["key_themes"] == ["theme1", "theme2"]
        assert serialized["sentiment_scores"] == {"overall": 0.7}
        assert "analysis_timestamp" in serialized

        # Serialize to JSON
        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        assert "serialize_test_001" in json_str

    def test_configdict_preserves_validation(self):
        """Test that ConfigDict preserves field validation behavior.
        
        **Validates: Requirements 4.2**
        """
        from agent_scrivener.models.analysis import AnalysisResults

        # Test that validation still works correctly
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResults(
                session_id=""  # Empty string should fail min_length=1
            )

        error = exc_info.value
        assert "session_id" in str(error).lower()

    @given(
        session_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip())
    )
    def test_property_schema_generation_equivalence(self, session_id):
        """
        Property Test: Schema Generation Equivalence

        **Validates: Requirements 4.2**

        For any Pydantic model with json_schema_extra configuration, the generated
        JSON schema SHALL be equivalent and contain the expected structure and examples.

        This property verifies that:
        1. Schema generation succeeds for models with ConfigDict
        2. json_schema_extra content is included in the generated schema
        3. Example data structure matches the expected format
        4. Schema contains all required fields and properties
        5. Schema generation is consistent across all model instances
        """
        from agent_scrivener.models.analysis import AnalysisResults
        from agent_scrivener.models.core import AcademicPaper

        # Test AnalysisResults schema generation
        analysis_schema = AnalysisResults.model_json_schema()

        # Property 1: Schema should be a valid dictionary
        assert isinstance(analysis_schema, dict)
        assert len(analysis_schema) > 0

        # Property 2: Schema should contain the example from json_schema_extra
        assert "example" in analysis_schema
        assert isinstance(analysis_schema["example"], dict)

        # Property 3: Example should contain all expected fields
        example = analysis_schema["example"]
        assert "session_id" in example
        assert "named_entities" in example
        assert "topics" in example
        assert "key_themes" in example
        assert "sentiment_scores" in example

        # Property 4: Example structure should match expected format
        # Verify named_entities structure
        assert isinstance(example["named_entities"], list)
        if len(example["named_entities"]) > 0:
            entity = example["named_entities"][0]
            assert "text" in entity
            assert "label" in entity
            assert "confidence_score" in entity
            assert isinstance(entity["text"], str)
            assert isinstance(entity["label"], str)
            assert isinstance(entity["confidence_score"], (int, float))

        # Verify topics structure
        assert isinstance(example["topics"], list)
        if len(example["topics"]) > 0:
            topic = example["topics"][0]
            assert "topic_id" in topic
            assert "keywords" in topic
            assert "weight" in topic
            assert isinstance(topic["topic_id"], int)
            assert isinstance(topic["keywords"], list)
            assert len(topic["keywords"]) >= 1  # Validates min_length constraint
            assert isinstance(topic["weight"], (int, float))

        # Verify key_themes structure
        assert isinstance(example["key_themes"], list)

        # Verify sentiment_scores structure
        assert isinstance(example["sentiment_scores"], dict)

        # Property 5: Schema should contain properties definition
        assert "properties" in analysis_schema
        properties = analysis_schema["properties"]
        assert "session_id" in properties
        assert "named_entities" in properties
        assert "topics" in properties
        assert "key_themes" in properties
        assert "sentiment_scores" in properties

        # Property 6: Required fields should be marked as required
        assert "required" in analysis_schema
        assert "session_id" in analysis_schema["required"]

        # Test AcademicPaper schema generation (also has json_schema_extra)
        paper_schema = AcademicPaper.model_json_schema()

        # Property 7: AcademicPaper schema should also contain example
        assert isinstance(paper_schema, dict)
        assert "example" in paper_schema
        paper_example = paper_schema["example"]

        # Property 8: AcademicPaper example should contain expected fields
        assert "title" in paper_example
        assert "authors" in paper_example
        assert "abstract" in paper_example
        assert "publication_year" in paper_example
        assert "database_source" in paper_example

        # Property 9: Authors field should respect min_length constraint in example
        assert isinstance(paper_example["authors"], list)
        assert len(paper_example["authors"]) >= 1

        # Property 10: Schema properties should include all model fields
        paper_properties = paper_schema["properties"]
        assert "title" in paper_properties
        assert "authors" in paper_properties
        assert "abstract" in paper_properties
        assert "publication_year" in paper_properties
        assert "doi" in paper_properties
        assert "database_source" in paper_properties

        # Property 11: Create an instance and verify it can be validated
        # This ensures schema generation doesn't break model functionality
        result = AnalysisResults(session_id=session_id)
        assert result.session_id == session_id

        # Property 12: Serialization should work correctly
        serialized = result.model_dump()
        assert serialized["session_id"] == session_id

        # Property 13: Schema generation should be idempotent
        # Calling model_json_schema multiple times should produce equivalent results
        schema_second_call = AnalysisResults.model_json_schema()
        assert schema_second_call["example"] == analysis_schema["example"]
        assert set(schema_second_call["properties"].keys()) == set(analysis_schema["properties"].keys())
        assert schema_second_call["required"] == analysis_schema["required"]


class TestAPIFormatEquivalence:
    """Property tests for API format equivalence.
    
    **Feature: test-fixes-and-pydantic-migration, Property 4: API Format Equivalence**
    """
    
    @given(
        query=st.text(min_size=10, max_size=2000).filter(lambda x: x.strip()),
        max_sources=st.integers(min_value=1, max_value=50),
        include_academic=st.booleans(),
        include_web=st.booleans(),
        priority=st.sampled_from(["low", "normal", "high"])
    )
    def test_property_research_request_format_equivalence(
        self, query, max_sources, include_academic, include_web, priority
    ):
        """
        Property Test: API Format Equivalence for ResearchRequest
        
        **Validates: Requirements 5.4**
        
        For any API request using migrated models, the response format and data
        structure SHALL be identical to responses before migration.
        
        This property verifies that:
        1. Request models accept and validate data correctly
        2. Serialization produces consistent JSON format
        3. All fields are properly typed and validated
        4. Data validation works correctly across all valid inputs
        """
        from agent_scrivener.api.models import ResearchRequest
        
        # Create ResearchRequest with generated values
        request = ResearchRequest(
            query=query,
            max_sources=max_sources,
            include_academic=include_academic,
            include_web=include_web,
            priority=priority
        )
        
        # Property 1: Model should accept valid inputs without errors
        assert request.query == query
        assert request.max_sources == max_sources
        assert request.include_academic == include_academic
        assert request.include_web == include_web
        assert request.priority == priority
        
        # Property 2: Serialization should produce consistent JSON format
        serialized = request.model_dump(mode='json')
        assert isinstance(serialized, dict)
        
        # Property 3: All required fields should be present in serialized output
        assert 'query' in serialized
        assert 'max_sources' in serialized
        assert 'include_academic' in serialized
        assert 'include_web' in serialized
        assert 'priority' in serialized
        
        # Property 4: Serialized values should match input values
        assert serialized['query'] == query
        assert serialized['max_sources'] == max_sources
        assert serialized['include_academic'] == include_academic
        assert serialized['include_web'] == include_web
        assert serialized['priority'] == priority
        
        # Property 5: Deserialization should be idempotent
        # Deserializing serialized data should produce equivalent model
        deserialized = ResearchRequest(**serialized)
        assert deserialized.query == request.query
        assert deserialized.max_sources == request.max_sources
        assert deserialized.include_academic == request.include_academic
        assert deserialized.include_web == request.include_web
        assert deserialized.priority == request.priority
    
    @given(
        session_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        status=st.sampled_from(["pending", "in_progress", "completed", "failed", "cancelled"]),
        progress_percentage=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        current_task=st.one_of(st.none(), st.text(min_size=1, max_size=200)),
        completed_tasks=st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=10),
        estimated_time_remaining_minutes=st.one_of(st.none(), st.integers(min_value=0, max_value=1000)),
        created_at=st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 12, 31)),
        updated_at=st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 12, 31)),
        error_message=st.one_of(st.none(), st.text(min_size=1, max_size=500))
    )
    def test_property_session_status_format_equivalence(
        self, session_id, status, progress_percentage, current_task, 
        completed_tasks, estimated_time_remaining_minutes, created_at, 
        updated_at, error_message
    ):
        """
        Property Test: API Format Equivalence for SessionStatus
        
        **Validates: Requirements 5.4**
        
        For any SessionStatus response, the format and structure SHALL be
        consistent across all valid inputs, with proper datetime serialization.
        
        This property verifies that:
        1. Response models serialize correctly
        2. Datetime fields are serialized to ISO format strings
        3. Optional fields are handled correctly
        4. List fields maintain their structure
        """
        from agent_scrivener.api.models import SessionStatus, ResearchStatus
        
        # Create SessionStatus with generated values
        session_status = SessionStatus(
            session_id=session_id,
            status=ResearchStatus(status),
            progress_percentage=progress_percentage,
            current_task=current_task,
            completed_tasks=completed_tasks,
            estimated_time_remaining_minutes=estimated_time_remaining_minutes,
            created_at=created_at,
            updated_at=updated_at,
            error_message=error_message
        )
        
        # Property 1: Model should accept valid inputs
        assert session_status.session_id == session_id
        assert session_status.status.value == status
        assert session_status.progress_percentage == progress_percentage
        
        # Property 2: Serialization should produce consistent JSON format
        serialized = session_status.model_dump(mode='json')
        assert isinstance(serialized, dict)
        
        # Property 3: Datetime fields should be serialized to ISO format strings
        assert isinstance(serialized['created_at'], str)
        assert isinstance(serialized['updated_at'], str)
        assert 'T' in serialized['created_at']  # ISO format contains 'T'
        assert 'T' in serialized['updated_at']
        
        # Property 4: Optional fields should be handled correctly
        if current_task is None:
            assert serialized['current_task'] is None
        else:
            assert serialized['current_task'] == current_task
        
        if estimated_time_remaining_minutes is None:
            assert serialized['estimated_time_remaining_minutes'] is None
        else:
            assert serialized['estimated_time_remaining_minutes'] == estimated_time_remaining_minutes
        
        if error_message is None:
            assert serialized['error_message'] is None
        else:
            assert serialized['error_message'] == error_message
        
        # Property 5: List fields should maintain structure
        assert isinstance(serialized['completed_tasks'], list)
        assert len(serialized['completed_tasks']) == len(completed_tasks)
        
        # Property 6: Enum fields should be serialized to their string values
        assert serialized['status'] == status
        assert isinstance(serialized['status'], str)
    
    @given(
        session_id=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        status=st.sampled_from(["completed"]),  # Only completed status for results
        document_content=st.text(min_size=1, max_size=10000),
        sources_count=st.integers(min_value=0, max_value=100),
        word_count=st.integers(min_value=0, max_value=50000),
        completion_time_minutes=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
        created_at=st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 12, 31)),
        completed_at=st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 12, 31))
    )
    def test_property_research_result_format_equivalence(
        self, session_id, status, document_content, sources_count, 
        word_count, completion_time_minutes, created_at, completed_at
    ):
        """
        Property Test: API Format Equivalence for ResearchResult
        
        **Validates: Requirements 5.4**
        
        For any ResearchResult response, the format SHALL be consistent with
        proper datetime serialization and numeric field handling.
        
        This property verifies that:
        1. Result models serialize correctly
        2. Datetime fields are properly formatted
        3. Numeric fields maintain precision
        4. Large text content is handled correctly
        """
        from agent_scrivener.api.models import ResearchResult, ResearchStatus
        
        # Create ResearchResult with generated values
        result = ResearchResult(
            session_id=session_id,
            status=ResearchStatus(status),
            document_content=document_content,
            sources_count=sources_count,
            word_count=word_count,
            completion_time_minutes=completion_time_minutes,
            created_at=created_at,
            completed_at=completed_at
        )
        
        # Property 1: Model should accept valid inputs
        assert result.session_id == session_id
        assert result.status.value == status
        assert result.document_content == document_content
        assert result.sources_count == sources_count
        assert result.word_count == word_count
        
        # Property 2: Serialization should produce consistent JSON format
        serialized = result.model_dump(mode='json')
        assert isinstance(serialized, dict)
        
        # Property 3: All required fields should be present
        required_fields = [
            'session_id', 'status', 'document_content', 'sources_count',
            'word_count', 'completion_time_minutes', 'created_at', 'completed_at'
        ]
        for field in required_fields:
            assert field in serialized
        
        # Property 4: Datetime fields should be serialized to ISO format strings
        assert isinstance(serialized['created_at'], str)
        assert isinstance(serialized['completed_at'], str)
        
        # Property 5: Numeric fields should maintain their types
        assert isinstance(serialized['sources_count'], int)
        assert isinstance(serialized['word_count'], int)
        assert isinstance(serialized['completion_time_minutes'], (int, float))
        
        # Property 6: Text content should be preserved exactly
        assert serialized['document_content'] == document_content
        
        # Property 7: Deserialization should be idempotent
        deserialized = ResearchResult(**serialized)
        assert deserialized.session_id == result.session_id
        assert deserialized.status == result.status
        assert deserialized.document_content == result.document_content
        assert deserialized.sources_count == result.sources_count
        assert deserialized.word_count == result.word_count
