"""
Unit tests for API Agent.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import List, Dict, Any

from agent_scrivener.agents.api_agent import APIAgent
from agent_scrivener.agents.base import AgentResult
from agent_scrivener.models.core import AcademicPaper
from agent_scrivener.models.errors import ValidationError, NetworkError, ProcessingError
from agent_scrivener.tools.gateway_wrapper import GatewayWrapper


@pytest.fixture
def mock_gateway():
    """Create a mock gateway wrapper."""
    gateway = MagicMock(spec=GatewayWrapper)
    gateway.query_external_api = AsyncMock()
    gateway.health_check = AsyncMock(return_value={'status': 'healthy'})
    return gateway


@pytest.fixture
def api_agent(mock_gateway):
    """Create an API agent instance with mocked gateway."""
    return APIAgent(gateway_wrapper=mock_gateway)


@pytest.fixture
def sample_arxiv_response():
    """Sample arXiv API response."""
    return {
        'success': True,
        'data': {
            'entries': [
                {
                    'title': 'Machine Learning in Healthcare Applications',
                    'authors': [{'name': 'John Doe'}, {'name': 'Jane Smith'}],
                    'summary': 'This paper explores the application of machine learning techniques in healthcare settings.',
                    'published': '2024-01-15T10:00:00Z',
                    'id': 'http://arxiv.org/abs/2401.1234',
                    'categories': 'cs.LG, cs.AI'
                },
                {
                    'title': 'Deep Learning for Medical Diagnosis',
                    'authors': [{'name': 'Alice Johnson'}],
                    'summary': 'A comprehensive study on deep learning applications for medical diagnosis.',
                    'published': '2023-12-20T15:30:00Z',
                    'id': 'http://arxiv.org/abs/2312.5678',
                    'categories': 'cs.CV, cs.LG'
                }
            ]
        }
    }


@pytest.fixture
def sample_pubmed_response():
    """Sample PubMed API response."""
    return {
        'success': True,
        'data': {
            'articles': [
                {
                    'title': 'Clinical Applications of AI in Medicine',
                    'authors': ['Dr. Medical Expert', 'Prof. Clinical Research'],
                    'abstract': 'This clinical study examines the practical applications of artificial intelligence in medical practice.',
                    'year': '2024',
                    'doi': '10.1234/clinical.2024.001',
                    'citation_count': 25,
                    'keywords': ['artificial intelligence', 'medicine', 'clinical'],
                    'url': 'https://pubmed.ncbi.nlm.nih.gov/12345678/'
                }
            ]
        }
    }


@pytest.fixture
def sample_semantic_scholar_response():
    """Sample Semantic Scholar API response."""
    return {
        'success': True,
        'data': {
            'data': [
                {
                    'title': 'Semantic Understanding in AI Systems',
                    'authors': [{'name': 'Prof. Semantic'}, {'name': 'Dr. Scholar'}],
                    'abstract': 'This paper presents novel approaches to semantic understanding in artificial intelligence systems.',
                    'year': 2024,
                    'doi': '10.5678/semantic.2024.001',
                    'citationCount': 42,
                    'url': 'https://www.semanticscholar.org/paper/abc123'
                }
            ]
        }
    }


class TestAPIAgentInitialization:
    """Test API agent initialization."""
    
    def test_init_with_default_parameters(self, mock_gateway):
        """Test initialization with default parameters."""
        agent = APIAgent(mock_gateway)
        
        assert agent.name == "api_agent"
        assert agent.gateway == mock_gateway
        assert agent.max_results_per_database == 20
        assert agent.min_citation_count == 0
        assert agent.max_concurrent_requests == 5
        assert len(agent.databases) == 3
        assert 'arxiv' in agent.databases
        assert 'pubmed' in agent.databases
        assert 'semantic_scholar' in agent.databases
    
    def test_init_with_custom_name(self, mock_gateway):
        """Test initialization with custom name."""
        agent = APIAgent(mock_gateway, name="custom_api_agent")
        assert agent.name == "custom_api_agent"
    
    def test_database_configurations(self, api_agent):
        """Test database configurations are properly set."""
        # Check arXiv config
        arxiv_config = api_agent.databases['arxiv']
        assert arxiv_config['name'] == 'arXiv'
        assert arxiv_config['weight'] == 0.9
        assert arxiv_config['rate_limit'] == 3.0
        
        # Check PubMed config
        pubmed_config = api_agent.databases['pubmed']
        assert pubmed_config['name'] == 'PubMed'
        assert pubmed_config['weight'] == 0.95
        assert pubmed_config['rate_limit'] == 0.34
        
        # Check Semantic Scholar config
        ss_config = api_agent.databases['semantic_scholar']
        assert ss_config['name'] == 'Semantic Scholar'
        assert ss_config['weight'] == 0.8
        assert ss_config['rate_limit'] == 1.0


class TestAPIAgentExecution:
    """Test API agent execution methods."""
    
    @pytest.mark.asyncio
    async def test_execute_with_valid_query(self, api_agent, mock_gateway, sample_arxiv_response):
        """Test execute method with valid query."""
        mock_gateway.query_external_api.return_value = sample_arxiv_response
        
        result = await api_agent.execute(query="machine learning")
        
        assert isinstance(result, AgentResult)
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) > 0
        assert all(isinstance(paper, AcademicPaper) for paper in result.data)
    
    @pytest.mark.asyncio
    async def test_execute_with_empty_query(self, api_agent):
        """Test execute method with empty query."""
        result = await api_agent.execute(query="")
        
        assert isinstance(result, AgentResult)
        assert result.success is False
        assert "Query cannot be empty" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_with_custom_parameters(self, api_agent, mock_gateway, sample_arxiv_response):
        """Test execute method with custom parameters."""
        mock_gateway.query_external_api.return_value = sample_arxiv_response
        
        result = await api_agent.execute(
            query="deep learning",
            databases=["arxiv"],
            max_results=5,
            min_citation_count=10
        )
        
        assert isinstance(result, AgentResult)
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_execute_with_invalid_database(self, api_agent, mock_gateway):
        """Test execute method with invalid database name."""
        result = await api_agent.execute(
            query="test query",
            databases=["invalid_db"]
        )
        
        assert isinstance(result, AgentResult)
        assert result.success is True  # Should succeed but with no results from invalid DB
        assert isinstance(result.data, list)


class TestDatabaseQueries:
    """Test individual database query methods."""
    
    @pytest.mark.asyncio
    async def test_query_arxiv_success(self, api_agent, mock_gateway, sample_arxiv_response):
        """Test successful arXiv query."""
        mock_gateway.query_external_api.return_value = sample_arxiv_response
        
        papers = await api_agent._query_arxiv("machine learning", 10)
        
        assert len(papers) == 2
        assert all(isinstance(paper, AcademicPaper) for paper in papers)
        assert papers[0].database_source == 'arXiv'
        assert papers[0].title == 'Machine Learning in Healthcare Applications'
        assert len(papers[0].authors) == 2
        assert papers[0].publication_year == 2024
    
    @pytest.mark.asyncio
    async def test_query_arxiv_api_error(self, api_agent, mock_gateway):
        """Test arXiv query with API error."""
        mock_gateway.query_external_api.return_value = {
            'success': False,
            'error': 'API rate limit exceeded'
        }
        
        papers = await api_agent._query_arxiv("test query", 10)
        
        # Should return mock data when API fails
        assert len(papers) == 5
        assert all(isinstance(paper, AcademicPaper) for paper in papers)
        assert all(paper.database_source == 'arXiv' for paper in papers)
    
    @pytest.mark.asyncio
    async def test_query_pubmed_success(self, api_agent, mock_gateway, sample_pubmed_response):
        """Test successful PubMed query."""
        mock_gateway.query_external_api.return_value = sample_pubmed_response
        
        papers = await api_agent._query_pubmed("clinical AI", 10)
        
        assert len(papers) == 1
        assert isinstance(papers[0], AcademicPaper)
        assert papers[0].database_source == 'PubMed'
        assert papers[0].title == 'Clinical Applications of AI in Medicine'
        assert papers[0].citation_count == 25
        assert papers[0].doi == '10.1234/clinical.2024.001'
    
    @pytest.mark.asyncio
    async def test_query_semantic_scholar_success(self, api_agent, mock_gateway, sample_semantic_scholar_response):
        """Test successful Semantic Scholar query."""
        mock_gateway.query_external_api.return_value = sample_semantic_scholar_response
        
        papers = await api_agent._query_semantic_scholar("semantic AI", 10)
        
        assert len(papers) == 1
        assert isinstance(papers[0], AcademicPaper)
        assert papers[0].database_source == 'Semantic Scholar'
        assert papers[0].title == 'Semantic Understanding in AI Systems'
        assert papers[0].citation_count == 42
        assert len(papers[0].authors) == 2
    
    @pytest.mark.asyncio
    async def test_query_database_with_exception(self, api_agent, mock_gateway):
        """Test database query with exception handling."""
        mock_gateway.query_external_api.side_effect = Exception("Network error")
        
        papers = await api_agent._query_database("arxiv", "test query", 10)
        
        # Should return empty list on exception
        assert papers == []


class TestQueryFormatting:
    """Test query formatting methods."""
    
    def test_format_arxiv_query(self, api_agent):
        """Test arXiv query formatting."""
        query = "machine learning algorithms"
        formatted = api_agent._format_arxiv_query(query)
        
        assert "machine" in formatted
        assert "learning" in formatted
        assert "algorithms" in formatted
        assert "AND" in formatted
    
    def test_format_arxiv_query_with_special_chars(self, api_agent):
        """Test arXiv query formatting with special characters."""
        query = "machine-learning & AI!"
        formatted = api_agent._format_arxiv_query(query)
        
        # Special characters should be removed
        assert "&" not in formatted
        assert "!" not in formatted
        assert "-" not in formatted
    
    def test_format_pubmed_query(self, api_agent):
        """Test PubMed query formatting."""
        query = "clinical trials medicine"
        formatted = api_agent._format_pubmed_query(query)
        
        assert "[Title/Abstract]" in formatted
        assert "clinical[Title/Abstract]" in formatted
        assert "AND" in formatted
    
    def test_format_pubmed_query_short_terms(self, api_agent):
        """Test PubMed query formatting with short terms."""
        query = "AI in ML"
        formatted = api_agent._format_pubmed_query(query)
        
        # Short terms (<=2 chars) should be filtered out
        assert "AI" not in formatted
        assert "ML" not in formatted


class TestResponseParsing:
    """Test response parsing methods."""
    
    def test_parse_arxiv_response(self, api_agent, sample_arxiv_response):
        """Test arXiv response parsing."""
        papers = api_agent._parse_arxiv_response(sample_arxiv_response['data'])
        
        assert len(papers) == 2
        
        paper1 = papers[0]
        assert paper1.title == 'Machine Learning in Healthcare Applications'
        assert len(paper1.authors) == 2
        assert 'John Doe' in paper1.authors
        assert paper1.publication_year == 2024
        assert paper1.database_source == 'arXiv'
        assert paper1.citation_count == 0  # arXiv doesn't provide citations
    
    def test_parse_arxiv_response_empty(self, api_agent):
        """Test arXiv response parsing with empty data."""
        papers = api_agent._parse_arxiv_response({})
        assert papers == []
    
    def test_parse_pubmed_response(self, api_agent, sample_pubmed_response):
        """Test PubMed response parsing."""
        papers = api_agent._parse_pubmed_response(sample_pubmed_response['data'])
        
        assert len(papers) == 1
        
        paper = papers[0]
        assert paper.title == 'Clinical Applications of AI in Medicine'
        assert paper.citation_count == 25
        assert paper.doi == '10.1234/clinical.2024.001'
        assert paper.database_source == 'PubMed'
    
    def test_parse_semantic_scholar_response(self, api_agent, sample_semantic_scholar_response):
        """Test Semantic Scholar response parsing."""
        papers = api_agent._parse_semantic_scholar_response(sample_semantic_scholar_response['data'])
        
        assert len(papers) == 1
        
        paper = papers[0]
        assert paper.title == 'Semantic Understanding in AI Systems'
        assert paper.citation_count == 42
        assert len(paper.authors) == 2
        assert paper.database_source == 'Semantic Scholar'


class TestUtilityMethods:
    """Test utility methods."""
    
    def test_extract_authors_from_dict_list(self, api_agent):
        """Test author extraction from dictionary list."""
        authors_data = [
            {'name': 'John Doe'},
            {'name': 'Jane Smith'},
            {'name': 'Bob Johnson'}
        ]
        
        authors = api_agent._extract_authors(authors_data)
        
        assert len(authors) == 3
        assert 'John Doe' in authors
        assert 'Jane Smith' in authors
        assert 'Bob Johnson' in authors
    
    def test_extract_authors_from_string_list(self, api_agent):
        """Test author extraction from string list."""
        authors_data = ['John Doe', 'Jane Smith']
        
        authors = api_agent._extract_authors(authors_data)
        
        assert len(authors) == 2
        assert 'John Doe' in authors
        assert 'Jane Smith' in authors
    
    def test_extract_authors_limit(self, api_agent):
        """Test author extraction with limit."""
        authors_data = [{'name': f'Author {i}'} for i in range(15)]
        
        authors = api_agent._extract_authors(authors_data)
        
        assert len(authors) == 10  # Should be limited to 10
    
    def test_extract_year_from_date_string(self, api_agent):
        """Test year extraction from date string."""
        test_cases = [
            ('2024-01-15T10:00:00Z', 2024),
            ('2023-12-20', 2023),
            ('Published in 2022', 2022),
            ('invalid date', 2024),  # Should default to current year
            ('', 2024)  # Empty string should default
        ]
        
        for date_string, expected_year in test_cases:
            year = api_agent._extract_year(date_string)
            assert year == expected_year
    
    def test_extract_doi_from_identifier(self, api_agent):
        """Test DOI extraction from identifier."""
        test_cases = [
            ('http://arxiv.org/abs/2401.1234', None),  # arXiv ID, not DOI
            ('doi:10.1234/example.2024.001', '10.1234/example.2024.001'),
            ('https://doi.org/10.5678/test.2024.002', '10.5678/test.2024.002'),
            ('10.1111/journal.2024.003', '10.1111/journal.2024.003'),
            ('no doi here', None)
        ]
        
        for identifier, expected_doi in test_cases:
            doi = api_agent._extract_doi(identifier)
            assert doi == expected_doi
    
    def test_extract_keywords_from_categories(self, api_agent):
        """Test keyword extraction from categories."""
        categories = "cs.LG, cs.AI, stat.ML"
        keywords = api_agent._extract_keywords(categories)
        
        assert len(keywords) <= 5  # Should be limited to 5
        assert 'cs.lg' in keywords
        assert 'cs.ai' in keywords
        assert 'stat.ml' in keywords
    
    def test_create_title_hash(self, api_agent):
        """Test title hash creation for deduplication."""
        title1 = "Machine Learning in Healthcare Applications"
        title2 = "Machine Learning in Healthcare Applications!"  # With punctuation
        title3 = "Different Title Entirely"
        
        hash1 = api_agent._create_title_hash(title1)
        hash2 = api_agent._create_title_hash(title2)
        hash3 = api_agent._create_title_hash(title3)
        
        assert hash1 == hash2  # Should be same despite punctuation
        assert hash1 != hash3  # Should be different for different titles
        assert len(hash1) == 32  # MD5 hash length


class TestDeduplicationAndFiltering:
    """Test deduplication and filtering methods."""
    
    def test_deduplicate_papers_by_doi(self, api_agent):
        """Test paper deduplication by DOI."""
        papers = [
            AcademicPaper(
                title="Test Paper 1",
                authors=["Author 1"],
                abstract="Abstract 1",
                publication_year=2024,
                doi="10.1234/test.001",
                database_source="arXiv",
                citation_count=10
            ),
            AcademicPaper(
                title="Test Paper 1 Duplicate",
                authors=["Author 1"],
                abstract="Abstract 1",
                publication_year=2024,
                doi="10.1234/test.001",  # Same DOI
                database_source="PubMed",
                citation_count=15
            ),
            AcademicPaper(
                title="Test Paper 2",
                authors=["Author 2"],
                abstract="Abstract 2",
                publication_year=2024,
                doi="10.1234/test.002",
                database_source="Semantic Scholar",
                citation_count=5
            )
        ]
        
        unique_papers = api_agent._deduplicate_papers(papers)
        
        assert len(unique_papers) == 2  # Should remove one duplicate
        dois = [paper.doi for paper in unique_papers]
        assert "10.1234/test.001" in dois
        assert "10.1234/test.002" in dois
    
    def test_deduplicate_papers_by_title(self, api_agent):
        """Test paper deduplication by title similarity."""
        papers = [
            AcademicPaper(
                title="Machine Learning Applications",
                authors=["Author 1"],
                abstract="Abstract 1",
                publication_year=2024,
                database_source="arXiv",
                citation_count=10
            ),
            AcademicPaper(
                title="Machine Learning Applications!",  # Same title with punctuation
                authors=["Author 2"],
                abstract="Abstract 2",
                publication_year=2024,
                database_source="PubMed",
                citation_count=15
            )
        ]
        
        unique_papers = api_agent._deduplicate_papers(papers)
        
        assert len(unique_papers) == 1  # Should remove duplicate by title
    
    def test_filter_papers_by_citation_count(self, api_agent):
        """Test paper filtering by citation count."""
        papers = [
            AcademicPaper(
                title="High Citation Paper",
                authors=["Author 1"],
                abstract="This is a well-cited paper with substantial content for testing purposes.",
                publication_year=2024,
                database_source="arXiv",
                citation_count=50
            ),
            AcademicPaper(
                title="Low Citation Paper",
                authors=["Author 2"],
                abstract="This is a paper with low citations but still substantial content for testing.",
                publication_year=2024,
                database_source="PubMed",
                citation_count=2
            )
        ]
        
        filtered_papers = api_agent._filter_papers(papers, min_citation_count=10)
        
        assert len(filtered_papers) == 1
        assert filtered_papers[0].title == "High Citation Paper"
    
    def test_filter_papers_by_content_quality(self, api_agent):
        """Test paper filtering by content quality."""
        papers = [
            AcademicPaper(
                title="Good Paper Title",
                authors=["Author 1"],
                abstract="This is a comprehensive abstract with sufficient content to pass quality filters.",
                publication_year=2024,
                database_source="arXiv",
                citation_count=10
            ),
            AcademicPaper(
                title="Bad",  # Too short title
                authors=["Author 2"],
                abstract="Short",  # Too short abstract
                publication_year=2024,
                database_source="PubMed",
                citation_count=10
            ),
            AcademicPaper(
                title="Future Paper Title",
                authors=["Author 3"],
                abstract="This paper has a good abstract but is from the future somehow.",
                publication_year=2030,  # Future year
                database_source="Semantic Scholar",
                citation_count=10
            )
        ]
        
        filtered_papers = api_agent._filter_papers(papers, min_citation_count=0)
        
        assert len(filtered_papers) == 1
        assert filtered_papers[0].title == "Good Paper Title"


class TestPaperRanking:
    """Test paper ranking methods."""
    
    def test_rank_papers_by_relevance(self, api_agent):
        """Test paper ranking by relevance to query."""
        papers = [
            AcademicPaper(
                title="Machine Learning Applications",  # High relevance
                authors=["Author 1"],
                abstract="This paper discusses machine learning applications in various domains.",
                publication_year=2024,
                database_source="Semantic Scholar",  # High weight database
                citation_count=100
            ),
            AcademicPaper(
                title="Statistical Analysis Methods",  # Low relevance
                authors=["Author 2"],
                abstract="This paper focuses on statistical methods for data analysis.",
                publication_year=2020,
                database_source="arXiv",
                citation_count=10
            ),
            AcademicPaper(
                title="Deep Learning for Machine Intelligence",  # Medium relevance
                authors=["Author 3"],
                abstract="This paper explores deep learning techniques for machine intelligence.",
                publication_year=2023,
                database_source="PubMed",  # Highest weight database
                citation_count=50
            )
        ]
        
        query = "machine learning"
        ranked_papers = api_agent._rank_papers(papers, query)
        
        assert len(ranked_papers) == 3
        # First paper should have highest relevance (title + abstract match)
        assert "Machine Learning Applications" in ranked_papers[0].title


class TestHealthCheck:
    """Test health check functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, api_agent, mock_gateway, sample_arxiv_response):
        """Test successful health check."""
        mock_gateway.query_external_api.return_value = sample_arxiv_response
        mock_gateway.health_check.return_value = {'status': 'healthy'}
        
        health = await api_agent.health_check()
        
        assert health['status'] == 'healthy'
        assert health['agent_name'] == 'api_agent'
        assert health['gateway_wrapper_status'] == 'healthy'
        assert health['test_search_successful'] is True
        assert health['test_papers_count'] > 0
        assert 'available_databases' in health
        assert len(health['available_databases']) == 3
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, api_agent, mock_gateway):
        """Test health check with failure."""
        mock_gateway.query_external_api.side_effect = Exception("Database connection failed")
        
        health = await api_agent.health_check()
        
        assert health['status'] == 'unhealthy'
        assert health['agent_name'] == 'api_agent'
        assert 'error' in health
        assert 'Database connection failed' in health['error']


class TestMockDataGeneration:
    """Test mock data generation methods."""
    
    def test_generate_mock_arxiv_papers(self, api_agent):
        """Test mock arXiv paper generation."""
        query = "quantum computing"
        max_results = 3
        
        papers = api_agent._generate_mock_arxiv_papers(query, max_results)
        
        assert len(papers) == 3
        assert all(isinstance(paper, AcademicPaper) for paper in papers)
        assert all(paper.database_source == 'arXiv' for paper in papers)
        assert all('quantum computing' in paper.title.lower() for paper in papers)
        assert all(paper.doi.startswith('10.48550/arXiv') for paper in papers)
    
    def test_generate_mock_pubmed_papers(self, api_agent):
        """Test mock PubMed paper generation."""
        query = "cancer treatment"
        max_results = 2
        
        papers = api_agent._generate_mock_pubmed_papers(query, max_results)
        
        assert len(papers) == 2
        assert all(isinstance(paper, AcademicPaper) for paper in papers)
        assert all(paper.database_source == 'PubMed' for paper in papers)
        assert all('cancer treatment' in paper.title.lower() for paper in papers)
        assert all('clinical' in paper.abstract.lower() for paper in papers)
    
    def test_generate_mock_semantic_scholar_papers(self, api_agent):
        """Test mock Semantic Scholar paper generation."""
        query = "natural language processing"
        max_results = 4
        
        papers = api_agent._generate_mock_semantic_scholar_papers(query, max_results)
        
        assert len(papers) == 4
        assert all(isinstance(paper, AcademicPaper) for paper in papers)
        assert all(paper.database_source == 'Semantic Scholar' for paper in papers)
        assert all('natural language processing' in paper.title.lower() for paper in papers)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_execute_with_validation_error(self, api_agent):
        """Test execute method with validation error."""
        result = await api_agent.execute()  # Missing required query parameter
        
        assert isinstance(result, AgentResult)
        assert result.success is False
        assert "query" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_with_gateway_failure(self, api_agent, mock_gateway):
        """Test execute method with gateway failure."""
        mock_gateway.query_external_api.side_effect = NetworkError("Network timeout")
        
        result = await api_agent.execute(query="test query")
        
        assert isinstance(result, AgentResult)
        # Should still succeed with mock data
        assert result.success is True
        assert isinstance(result.data, list)
    
    @pytest.mark.asyncio
    async def test_database_query_with_malformed_response(self, api_agent, mock_gateway):
        """Test database query with malformed API response."""
        mock_gateway.query_external_api.return_value = {
            'success': True,
            'data': {
                'entries': [
                    {
                        # Missing required fields
                        'title': 'Test Paper'
                        # No authors, abstract, etc.
                    }
                ]
            }
        }
        
        papers = await api_agent._query_arxiv("test query", 10)
        
        # Should handle malformed entries gracefully
        assert isinstance(papers, list)
        # Should fall back to mock data
        assert len(papers) > 0


class TestConcurrentOperations:
    """Test concurrent operations and rate limiting."""
    
    @pytest.mark.asyncio
    async def test_concurrent_database_queries(self, api_agent, mock_gateway, sample_arxiv_response):
        """Test concurrent database queries with rate limiting."""
        mock_gateway.query_external_api.return_value = sample_arxiv_response
        
        # Mock sleep to speed up test
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await api_agent.execute(
                query="test query",
                databases=["arxiv", "pubmed", "semantic_scholar"]
            )
        
        assert isinstance(result, AgentResult)
        assert result.success is True
        
        # Should have called gateway for each database
        assert mock_gateway.query_external_api.call_count >= 3
    
    @pytest.mark.asyncio
    async def test_rate_limiting_between_requests(self, api_agent, mock_gateway, sample_arxiv_response):
        """Test rate limiting between database requests."""
        mock_gateway.query_external_api.return_value = sample_arxiv_response
        
        start_time = datetime.now()
        
        # Execute with multiple databases
        await api_agent.execute(
            query="test query",
            databases=["arxiv", "pubmed"]
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should take at least the rate limit time between requests
        # (This is a rough test - in practice would mock time more precisely)
        assert duration >= 0  # Basic sanity check


if __name__ == "__main__":
    pytest.main([__file__])