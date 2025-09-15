"""
Unit tests for CitationAgent.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import aiohttp

from agent_scrivener.agents.citation_agent import (
    CitationAgent, CitationTracker, APACitationFormatter, URLValidator
)
from agent_scrivener.models.core import Citation, Source, SourceType


class TestCitationTracker:
    """Test the CitationTracker class."""
    
    def test_init(self):
        """Test CitationTracker initialization."""
        tracker = CitationTracker()
        assert tracker.citations == {}
        assert tracker.source_citations == {}
        assert tracker.content_citations == {}
    
    def test_add_citation(self):
        """Test adding a citation to the tracker."""
        tracker = CitationTracker()
        
        source = Source(
            url="https://example.com/article",
            title="Test Article",
            author="Test Author",
            source_type=SourceType.WEB
        )
        
        citation = Citation(
            citation_id="test-123",
            source=source,
            citation_text="Test Author. (2024). Test Article. Retrieved from https://example.com/article",
            quote="This is a test quote"
        )
        
        tracker.add_citation(citation)
        
        assert "test-123" in tracker.citations
        assert tracker.citations["test-123"] == citation
        assert "https://example.com/article" in tracker.source_citations
        assert "test-123" in tracker.source_citations["https://example.com/article"]
    
    def test_get_citations_for_source(self):
        """Test retrieving citations for a specific source."""
        tracker = CitationTracker()
        
        source = Source(
            url="https://example.com/article",
            title="Test Article",
            source_type=SourceType.WEB
        )
        
        citation1 = Citation(
            citation_id="test-123",
            source=source,
            citation_text="Citation 1"
        )
        
        citation2 = Citation(
            citation_id="test-456",
            source=source,
            citation_text="Citation 2"
        )
        
        tracker.add_citation(citation1)
        tracker.add_citation(citation2)
        
        citations = tracker.get_citations_for_source("https://example.com/article")
        assert len(citations) == 2
        assert citation1 in citations
        assert citation2 in citations
    
    def test_get_citation_by_id(self):
        """Test retrieving a citation by ID."""
        tracker = CitationTracker()
        
        source = Source(
            url="https://example.com/article",
            title="Test Article",
            source_type=SourceType.WEB
        )
        
        citation = Citation(
            citation_id="test-123",
            source=source,
            citation_text="Test Citation"
        )
        
        tracker.add_citation(citation)
        
        retrieved = tracker.get_citation_by_id("test-123")
        assert retrieved == citation
        
        not_found = tracker.get_citation_by_id("nonexistent")
        assert not_found is None


class TestAPACitationFormatter:
    """Test the APA citation formatter."""
    
    def test_format_web_source(self):
        """Test formatting a web source in APA style."""
        source = Source(
            url="https://example.com/article",
            title="Understanding Machine Learning",
            author="Smith, John",
            publication_date=datetime(2024, 3, 15),
            source_type=SourceType.WEB
        )
        
        formatted = APACitationFormatter.format_web_source(source)
        expected = "Smith, John. (2024, March 15). Understanding Machine Learning. Retrieved from https://example.com/article"
        assert formatted == expected
    
    def test_format_web_source_no_author(self):
        """Test formatting a web source without author."""
        source = Source(
            url="https://example.com/article",
            title="Understanding Machine Learning",
            source_type=SourceType.WEB
        )
        
        formatted = APACitationFormatter.format_web_source(source)
        assert "Unknown Author" in formatted
        assert "n.d." in formatted
    
    def test_format_academic_source(self):
        """Test formatting an academic source in APA style."""
        source = Source(
            url="https://doi.org/10.1234/example",
            title="Machine Learning in Healthcare",
            author="Johnson, Mary",
            publication_date=datetime(2023, 1, 1),
            source_type=SourceType.ACADEMIC,
            metadata={
                "journal": "Journal of AI Research",
                "volume": "15",
                "issue": "3",
                "pages": "123-145",
                "doi": "10.1234/example"
            }
        )
        
        formatted = APACitationFormatter.format_academic_source(source)
        expected = "Johnson, Mary. (2023). Machine Learning in Healthcare. Journal of AI Research, 15(3), 123-145. https://doi.org/10.1234/example"
        assert formatted == expected
    
    def test_format_academic_source_minimal(self):
        """Test formatting an academic source with minimal metadata."""
        source = Source(
            url="https://example.com/paper",
            title="AI Research Paper",
            author="Brown, Alice",
            publication_date=datetime(2023, 1, 1),
            source_type=SourceType.ACADEMIC
        )
        
        formatted = APACitationFormatter.format_academic_source(source)
        assert "Brown, Alice. (2023). AI Research Paper. Unknown Journal" in formatted
        assert "Retrieved from https://example.com/paper" in formatted
    
    def test_format_database_source(self):
        """Test formatting a database source in APA style."""
        source = Source(
            url="https://pubmed.ncbi.nlm.nih.gov/12345",
            title="Clinical Study Results",
            author="Wilson, Robert",
            publication_date=datetime(2024, 2, 10),
            source_type=SourceType.DATABASE,
            metadata={"database": "PubMed"}
        )
        
        formatted = APACitationFormatter.format_database_source(source)
        expected = "Wilson, Robert. (2024, February 10). Clinical Study Results. PubMed. Retrieved from https://pubmed.ncbi.nlm.nih.gov/12345"
        assert formatted == expected
    
    def test_format_citation_dispatch(self):
        """Test that format_citation dispatches to correct formatter."""
        web_source = Source(
            url="https://example.com",
            title="Web Article",
            source_type=SourceType.WEB
        )
        
        academic_source = Source(
            url="https://doi.org/10.1234/test",
            title="Academic Paper",
            source_type=SourceType.ACADEMIC
        )
        
        database_source = Source(
            url="https://database.com/entry",
            title="Database Entry",
            source_type=SourceType.DATABASE
        )
        
        web_formatted = APACitationFormatter.format_citation(web_source)
        academic_formatted = APACitationFormatter.format_citation(academic_source)
        database_formatted = APACitationFormatter.format_citation(database_source)
        
        assert "Retrieved from https://example.com" in web_formatted
        assert "Academic Paper" in academic_formatted
        assert "Database Entry" in database_formatted


class TestURLValidator:
    """Test the URL validator."""
    
    def test_init(self):
        """Test URLValidator initialization."""
        validator = URLValidator()
        assert validator.timeout == 10
        assert validator.semaphore._value == 5
        
        validator_custom = URLValidator(timeout=5, max_concurrent=3)
        assert validator_custom.timeout == 5
        assert validator_custom.semaphore._value == 3


class TestCitationAgent:
    """Test the CitationAgent class."""
    
    @pytest.fixture
    def citation_agent(self):
        """Create a CitationAgent instance for testing."""
        return CitationAgent()
    
    @pytest.fixture
    def sample_source(self):
        """Create a sample source for testing."""
        return Source(
            url="https://example.com/article",
            title="Test Article",
            author="Test Author",
            publication_date=datetime(2024, 1, 15),
            source_type=SourceType.WEB
        )
    
    @pytest.fixture
    def sample_citation(self, sample_source):
        """Create a sample citation for testing."""
        return Citation(
            citation_id="test-123",
            source=sample_source,
            citation_text="Test Author. (2024, January 15). Test Article. Retrieved from https://example.com/article",
            quote="This is a test quote"
        )
    
    def test_init(self, citation_agent):
        """Test CitationAgent initialization."""
        assert citation_agent.name == "citation_agent"
        assert citation_agent.citation_tracker is not None
        assert citation_agent.formatter is not None
        assert citation_agent.url_validator is not None
        assert citation_agent.tracked_sources == set()
    
    @pytest.mark.asyncio
    async def test_track_sources(self, citation_agent, sample_source):
        """Test tracking sources."""
        content = "This article discusses machine learning applications."
        quote = "Machine learning is transforming healthcare."
        
        citation = await citation_agent.track_sources(content, sample_source, quote=quote)
        
        assert citation.source == sample_source
        assert citation.quote == quote
        assert citation.citation_id in citation_agent.citation_tracker.citations
        assert str(sample_source.url) in citation_agent.tracked_sources
    
    @pytest.mark.asyncio
    async def test_generate_bibliography_empty(self, citation_agent):
        """Test generating bibliography with no citations."""
        bibliography = await citation_agent.generate_bibliography([])
        
        assert "## References" in bibliography
        assert "No references found" in bibliography
    
    @pytest.mark.asyncio
    async def test_generate_bibliography_with_citations(self, citation_agent, sample_citation):
        """Test generating bibliography with citations."""
        citations = [sample_citation]
        
        bibliography = await citation_agent.generate_bibliography(citations)
        
        assert "## References" in bibliography
        assert sample_citation.citation_text in bibliography
    
    @pytest.mark.asyncio
    async def test_validate_citations(self, citation_agent, sample_citation):
        """Test citation validation."""
        with patch.object(citation_agent.url_validator, 'validate_urls') as mock_validate:
            mock_validate.return_value = [{
                'url': str(sample_citation.source.url),
                'accessible': True,
                'status_code': 200,
                'error': None
            }]
            
            result = await citation_agent.validate_citations([sample_citation])
            
            assert result['total_citations'] == 1
            assert result['accessible_count'] == 1
            assert result['inaccessible_count'] == 0
            assert len(result['accessible_citations']) == 1
    
    @pytest.mark.asyncio
    async def test_format_in_text_citations(self, citation_agent, sample_citation):
        """Test formatting in-text citations."""
        content = "The study on Test Article shows promising results."
        citations = [sample_citation]
        
        formatted = await citation_agent.format_in_text_citations(content, citations)
        
        # Should contain the original content plus citation
        assert "Test Article" in formatted
        # The author name should be extracted correctly - "Test Author" should become "(Author, 2024)"
        assert "(Author, 2024)" in formatted
    
    @pytest.mark.asyncio
    async def test_execute_track_sources_action(self, citation_agent, sample_source):
        """Test execute with track_sources action."""
        kwargs = {
            'action': 'track_sources',
            'sources': [sample_source.model_dump()],
            'content': 'Test content'
        }
        
        result = await citation_agent.execute(**kwargs)
        
        assert result.success is True
        assert result.data['action'] == 'track_sources'
        assert result.data['tracked_count'] == 1
        assert len(result.data['citations']) == 1
    
    @pytest.mark.asyncio
    async def test_execute_generate_bibliography_action(self, citation_agent, sample_citation):
        """Test execute with generate_bibliography action."""
        kwargs = {
            'action': 'generate_bibliography',
            'citations': [sample_citation.model_dump()]
        }
        
        result = await citation_agent.execute(**kwargs)
        
        assert result.success is True
        assert result.data['action'] == 'generate_bibliography'
        assert result.data['citation_count'] == 1
        assert "## References" in result.data['bibliography']
    
    @pytest.mark.asyncio
    async def test_execute_validate_citations_action(self, citation_agent, sample_citation):
        """Test execute with validate_citations action."""
        with patch.object(citation_agent.url_validator, 'validate_urls') as mock_validate:
            mock_validate.return_value = [{
                'url': str(sample_citation.source.url),
                'accessible': True,
                'status_code': 200,
                'error': None
            }]
            
            kwargs = {
                'action': 'validate_citations',
                'citations': [sample_citation.model_dump()]
            }
            
            result = await citation_agent.execute(**kwargs)
            
            assert result.success is True
            assert result.data['action'] == 'validate_citations'
            assert result.data['total_citations'] == 1
    
    @pytest.mark.asyncio
    async def test_execute_format_in_text_citations_action(self, citation_agent, sample_citation):
        """Test execute with format_in_text_citations action."""
        kwargs = {
            'action': 'format_in_text_citations',
            'content': 'Test content about Test Article',
            'citations': [sample_citation.model_dump()]
        }
        
        result = await citation_agent.execute(**kwargs)
        
        assert result.success is True
        assert result.data['action'] == 'format_in_text_citations'
        assert 'formatted_content' in result.data
        assert result.data['original_length'] > 0
    
    @pytest.mark.asyncio
    async def test_execute_invalid_action(self, citation_agent):
        """Test execute with invalid action."""
        kwargs = {'action': 'invalid_action'}
        
        result = await citation_agent.execute(**kwargs)
        
        assert result.success is False
        assert "Unknown action" in result.error
    
    @pytest.mark.asyncio
    async def test_get_citations_for_sources(self, citation_agent, sample_source):
        """Test getting citations for sources."""
        # First track a source
        await citation_agent.track_sources("test content", sample_source)
        
        # Then retrieve citations for that source
        citations = await citation_agent.get_citations_for_sources([sample_source])
        
        assert len(citations) == 1
        assert citations[0].source == sample_source
    
    def test_validate_input_missing_fields(self, citation_agent):
        """Test input validation with missing required fields."""
        data = {'field1': 'value1'}
        required_fields = ['field1', 'field2']
        
        with pytest.raises(ValueError, match="Missing required fields: field2"):
            citation_agent.validate_input(data, required_fields)
    
    def test_validate_input_success(self, citation_agent):
        """Test successful input validation."""
        data = {'field1': 'value1', 'field2': 'value2'}
        required_fields = ['field1', 'field2']
        
        result = citation_agent.validate_input(data, required_fields)
        assert result is True


class TestCitationAgentIntegration:
    """Integration tests for CitationAgent."""
    
    @pytest.fixture
    def citation_agent(self):
        """Create a CitationAgent instance for testing."""
        return CitationAgent()
    
    @pytest.mark.asyncio
    async def test_full_citation_workflow(self, citation_agent):
        """Test the complete citation workflow."""
        # Create test sources
        source1 = Source(
            url="https://example.com/article1",
            title="Machine Learning Basics",
            author="Smith, John",
            publication_date=datetime(2024, 1, 15),
            source_type=SourceType.WEB
        )
        
        source2 = Source(
            url="https://example.com/article2",
            title="Advanced AI Techniques",
            author="Johnson, Mary",
            publication_date=datetime(2023, 12, 10),
            source_type=SourceType.ACADEMIC,
            metadata={"journal": "AI Research Journal", "volume": "10", "issue": "2"}
        )
        
        # Track sources
        citation1 = await citation_agent.track_sources(
            "Content about ML basics", source1, quote="ML is fundamental"
        )
        citation2 = await citation_agent.track_sources(
            "Content about advanced AI", source2, quote="Advanced techniques are crucial"
        )
        
        citations = [citation1, citation2]
        
        # Generate bibliography
        bibliography = await citation_agent.generate_bibliography(citations)
        assert "## References" in bibliography
        assert "Smith, John" in bibliography
        assert "Johnson, Mary" in bibliography
        
        # Format in-text citations
        content = "The study on Machine Learning Basics and Advanced AI Techniques shows progress."
        formatted_content = await citation_agent.format_in_text_citations(content, citations)
        assert "(Smith, 2024)" in formatted_content
        assert "(Johnson, 2023)" in formatted_content
        
        # Validate citations (with mocked URL validation)
        with patch.object(citation_agent.url_validator, 'validate_urls') as mock_validate:
            mock_validate.return_value = [
                {'url': str(source1.url), 'accessible': True, 'status_code': 200, 'error': None},
                {'url': str(source2.url), 'accessible': True, 'status_code': 200, 'error': None}
            ]
            
            validation_result = await citation_agent.validate_citations(citations)
            assert validation_result['accessible_count'] == 2
            assert validation_result['inaccessible_count'] == 0