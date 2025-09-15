"""
Unit tests for Research Agent.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import List, Dict, Any

from agent_scrivener.agents.research_agent import ResearchAgent
from agent_scrivener.agents.base import AgentResult
from agent_scrivener.models.core import ExtractedArticle, Source, SourceType
from agent_scrivener.models.errors import ValidationError, NetworkError, ProcessingError
from agent_scrivener.tools.browser_wrapper import BrowserToolWrapper


@pytest.fixture
def mock_browser_wrapper():
    """Create mock browser wrapper."""
    wrapper = AsyncMock(spec=BrowserToolWrapper)
    wrapper.extract_multiple_urls = AsyncMock()
    wrapper.health_check = AsyncMock()
    return wrapper


@pytest.fixture
def research_agent(mock_browser_wrapper):
    """Create research agent with mocked dependencies."""
    return ResearchAgent(mock_browser_wrapper)


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {
            'url': 'https://example-research.edu/ai-study',
            'title': 'AI Research Study',
            'description': 'Comprehensive study on AI applications',
            'relevance_score': 0.9
        },
        {
            'url': 'https://tech-journal.com/machine-learning',
            'title': 'Machine Learning Advances',
            'description': 'Latest developments in ML algorithms',
            'relevance_score': 0.8
        },
        {
            'url': 'https://ai-news.org/neural-networks',
            'title': 'Neural Network Innovations',
            'description': 'Breakthrough in neural network architectures',
            'relevance_score': 0.7
        }
    ]


@pytest.fixture
def sample_extracted_articles():
    """Sample extracted articles for testing."""
    return [
        ExtractedArticle(
            source=Source(
                url='https://example-research.edu/ai-study',
                title='AI Research Study',
                source_type=SourceType.WEB,
                metadata={'credibility_score': 0.9}
            ),
            content='This is a comprehensive study on artificial intelligence applications in various domains.',
            key_findings=['AI applications', 'Domain analysis', 'Performance metrics'],
            confidence_score=0.9,
            extraction_timestamp=datetime(2024, 1, 15)
        ),
        ExtractedArticle(
            source=Source(
                url='https://tech-journal.com/machine-learning',
                title='Machine Learning Advances',
                source_type=SourceType.WEB,
                metadata={'credibility_score': 0.8}
            ),
            content='Recent advances in machine learning algorithms have shown significant improvements.',
            key_findings=['Algorithm improvements', 'Performance gains', 'New techniques'],
            confidence_score=0.8,
            extraction_timestamp=datetime(2024, 2, 10)
        )
    ]


class TestResearchAgent:
    """Test cases for Research Agent."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_browser_wrapper):
        """Test Research Agent initialization."""
        agent = ResearchAgent(mock_browser_wrapper)
        
        assert agent.browser == mock_browser_wrapper
        assert agent.name == "research_agent"
        assert hasattr(agent, 'max_sources_per_query')
        assert hasattr(agent, 'min_content_length')

    @pytest.mark.asyncio
    async def test_health_check_success(self, research_agent, mock_browser_wrapper):
        """Test successful health check."""
        mock_browser_wrapper.health_check.return_value = {
            'status': 'healthy',
            'browser_tool_available': True,
            'nova_act_available': False,
            'test_extraction_successful': True,
            'test_content_length': 500,
            'timestamp': datetime.now().isoformat()
        }
        
        # Mock the navigate_and_extract method to avoid the actual web search
        with patch.object(research_agent, 'search_web', return_value=[]):
            result = await research_agent.health_check()
            
            assert result['status'] == 'healthy'
            assert 'agent_name' in result

    @pytest.mark.asyncio
    async def test_health_check_failure(self, research_agent, mock_browser_wrapper):
        """Test health check failure."""
        mock_browser_wrapper.health_check.return_value = {
            'status': 'unhealthy',
            'error': 'Connection failed'
        }
        
        # Mock search_web to raise an exception
        with patch.object(research_agent, 'search_web', side_effect=NetworkError("Connection failed")):
            result = await research_agent.health_check()
            
            assert result['status'] == 'unhealthy'
            assert 'error' in result

    @pytest.mark.asyncio
    async def test_search_web_success(self, research_agent, mock_browser_wrapper, sample_search_results):
        """Test successful web search."""
        query = "artificial intelligence research"
        
        with patch.object(research_agent, '_simulate_search_results', return_value=sample_search_results):
            result = await research_agent.search_web(query)
            
            assert isinstance(result, list)
            assert len(result) == 3
            assert all('url' in item for item in result)
            assert all('title' in item for item in result)

    @pytest.mark.asyncio
    async def test_search_web_empty_query(self, research_agent):
        """Test web search with empty query."""
        # Empty query should still work but return empty results
        result = await research_agent.search_web("")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_search_web_with_max_results(self, research_agent, sample_search_results):
        """Test web search with max results limit."""
        query = "test query"
        max_results = 2
        
        with patch.object(research_agent, '_simulate_search_results', return_value=sample_search_results):
            result = await research_agent.search_web(query, max_results)
            
            assert isinstance(result, list)
            assert len(result) <= max_results

    @pytest.mark.asyncio
    async def test_extract_and_process_content_success(self, research_agent, mock_browser_wrapper):
        """Test successful content extraction and processing."""
        validated_sources = [
            {
                'url': 'https://example.com/article1',
                'title': 'Test Article 1',
                'quality_score': 0.8,
                'domain': 'example.com',
                'estimated_relevance': 0.9
            },
            {
                'url': 'https://example.com/article2',
                'title': 'Test Article 2',
                'quality_score': 0.7,
                'domain': 'example.com',
                'estimated_relevance': 0.8
            }
        ]
        
        # Mock browser extraction results
        extraction_results = [
            {
                'success': True,
                'url': 'https://example.com/article1',
                'title': 'Test Article 1',
                'content': 'This is test content for article 1. It contains sufficient text to pass validation.',
                'author': 'Test Author',
                'metadata': {'extraction_method': 'basic'}
            },
            {
                'success': True,
                'url': 'https://example.com/article2',
                'title': 'Test Article 2',
                'content': 'This is test content for article 2. It also contains sufficient text to pass validation.',
                'author': 'Another Author',
                'metadata': {'extraction_method': 'basic'}
            }
        ]
        
        mock_browser_wrapper.extract_multiple_urls.return_value = extraction_results
        
        result = await research_agent.extract_and_process_content(validated_sources, 0.3)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(article, ExtractedArticle) for article in result)
        mock_browser_wrapper.extract_multiple_urls.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_and_process_content_with_failures(self, research_agent, mock_browser_wrapper):
        """Test content extraction with some failures."""
        validated_sources = [
            {
                'url': 'https://example.com/article1',
                'title': 'Test Article 1',
                'quality_score': 0.8,
                'domain': 'example.com',
                'estimated_relevance': 0.9
            }
        ]
        
        # Mock browser extraction with failure
        extraction_results = [
            {
                'success': False,
                'url': 'https://example.com/article1',
                'error': 'Failed to extract content'
            }
        ]
        
        mock_browser_wrapper.extract_multiple_urls.return_value = extraction_results
        
        result = await research_agent.extract_and_process_content(validated_sources, 0.3)
        
        assert isinstance(result, list)
        assert len(result) == 0  # No successful extractions

    @pytest.mark.asyncio
    async def test_validate_sources_success(self, research_agent, sample_search_results):
        """Test successful source validation."""
        result = await research_agent.validate_sources(sample_search_results)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert all('quality_score' in item for item in result)
        assert all('url' in item for item in result)

    @pytest.mark.asyncio
    async def test_validate_sources_empty_list(self, research_agent):
        """Test source validation with empty list."""
        result = await research_agent.validate_sources([])
        
        assert result == []

    @pytest.mark.asyncio
    async def test_validate_sources_with_quality_scoring(self, research_agent):
        """Test source validation includes quality scoring."""
        search_results = [
            {
                'url': 'https://example.edu/research',
                'title': 'Academic Research Paper',
                'description': 'High quality academic content',
                'relevance_score': 0.9
            }
        ]
        
        result = await research_agent.validate_sources(search_results)
        
        assert len(result) == 1
        assert result[0]['quality_score'] > 0.5  # Should have good quality score for .edu domain

    @pytest.mark.asyncio
    async def test_execute_method_success(self, research_agent, mock_browser_wrapper):
        """Test the main execute method."""
        query = "artificial intelligence research"
        
        # Mock the browser extraction
        extraction_results = [
            {
                'success': True,
                'url': 'https://example.com/article1',
                'title': 'AI Research',
                'content': 'This is comprehensive content about artificial intelligence research with sufficient length.',
                'metadata': {'extraction_method': 'basic'}
            }
        ]
        mock_browser_wrapper.extract_multiple_urls.return_value = extraction_results
        
        result = await research_agent.execute(query=query)
        
        assert isinstance(result, AgentResult)
        assert result.success is True
        assert isinstance(result.data, list)
        assert result.agent_name == "research_agent"

    @pytest.mark.asyncio
    async def test_execute_method_validation_error(self, research_agent):
        """Test execute method with validation error."""
        # Missing required query parameter
        result = await research_agent.execute()
        
        assert isinstance(result, AgentResult)
        assert result.success is False
        assert "query" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_method_empty_query(self, research_agent):
        """Test execute method with empty query."""
        result = await research_agent.execute(query="")
        
        assert isinstance(result, AgentResult)
        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_full_research_workflow(self, research_agent, mock_browser_wrapper, sample_search_results):
        """Test complete research workflow."""
        query = "artificial intelligence research"
        
        # Mock search results
        with patch.object(research_agent, '_simulate_search_results', return_value=sample_search_results):
            # Mock extraction results
            extraction_results = [
                {
                    'success': True,
                    'url': 'https://example-research.edu/ai-study',
                    'title': 'AI Research Study',
                    'content': 'This is comprehensive content about artificial intelligence research with sufficient length to pass validation.',
                    'metadata': {'extraction_method': 'basic'}
                }
            ]
            mock_browser_wrapper.extract_multiple_urls.return_value = extraction_results
            
            result = await research_agent.execute(query=query)
            
            assert isinstance(result, AgentResult)
            assert result.success is True
            assert isinstance(result.data, list)
            assert len(result.data) > 0

    @pytest.mark.asyncio
    async def test_research_with_custom_parameters(self, research_agent, mock_browser_wrapper):
        """Test research with custom parameters."""
        query = "machine learning"
        max_sources = 5
        quality_threshold = 0.7
        
        # Mock extraction results
        extraction_results = [
            {
                'success': True,
                'url': 'https://example.com/ml-article',
                'title': 'Machine Learning Article',
                'content': 'This is comprehensive content about machine learning with sufficient length to pass validation.',
                'metadata': {'extraction_method': 'basic'}
            }
        ]
        mock_browser_wrapper.extract_multiple_urls.return_value = extraction_results
        
        result = await research_agent.execute(
            query=query, 
            max_sources=max_sources, 
            quality_threshold=quality_threshold
        )
        
        assert isinstance(result, AgentResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_research_with_network_error(self, research_agent, mock_browser_wrapper):
        """Test research handling network errors."""
        query = "test query"
        
        # Mock browser to raise network error
        mock_browser_wrapper.extract_multiple_urls.side_effect = NetworkError("Connection failed")
        
        result = await research_agent.execute(query=query)
        
        assert isinstance(result, AgentResult)
        assert result.success is False
        assert "Connection failed" in result.error

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, research_agent, mock_browser_wrapper):
        """Test handling of concurrent requests."""
        queries = ["AI research", "machine learning", "neural networks"]
        
        # Mock extraction results
        extraction_results = [
            {
                'success': True,
                'url': 'https://example.com/article',
                'title': 'Test Article',
                'content': 'This is test content with sufficient length to pass validation checks.',
                'metadata': {'extraction_method': 'basic'}
            }
        ]
        mock_browser_wrapper.extract_multiple_urls.return_value = extraction_results
        
        # Execute concurrent requests
        tasks = [research_agent.execute(query=query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(isinstance(result, AgentResult) for result in results)
        assert all(result.success for result in results)

    @pytest.mark.asyncio
    async def test_key_findings_extraction(self, research_agent):
        """Test key findings extraction from content."""
        content = """
        This study found that artificial intelligence applications are growing rapidly.
        The research revealed significant improvements in machine learning algorithms.
        Results indicate that neural networks perform better with larger datasets.
        Evidence suggests that deep learning models require substantial computational resources.
        """
        
        key_findings = await research_agent._extract_key_findings(content)
        
        assert isinstance(key_findings, list)
        assert len(key_findings) > 0
        assert any("found that" in finding for finding in key_findings)

    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self, research_agent):
        """Test confidence score calculation."""
        extraction_result = {
            'content': 'This is a comprehensive article with substantial content that should receive a good confidence score.',
            'title': 'Test Article',
            'author': 'Test Author',
            'metadata': {'extraction_method': 'nova_act'}
        }
        
        source_metadata = {
            'quality_score': 0.8,
            'estimated_relevance': 0.9
        }
        
        score = research_agent._calculate_article_confidence(extraction_result, source_metadata)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be reasonably high for good content

    def test_url_validation(self, research_agent):
        """Test URL validation utility."""
        valid_urls = [
            'https://example.com',
            'http://test.org/path',
            'https://subdomain.example.com/path?query=value'
        ]
        
        invalid_urls = [
            'not-a-url',
            'ftp://example.com',  # Unsupported protocol
            'https://',  # Incomplete
            ''  # Empty
        ]
        
        for url in valid_urls:
            assert research_agent._is_valid_research_url(url) is True
        
        for url in invalid_urls:
            assert research_agent._is_valid_research_url(url) is False

    @pytest.mark.asyncio
    async def test_source_quality_scoring(self, research_agent):
        """Test source quality scoring algorithm."""
        high_quality_source = {
            'url': 'https://example.edu/research/paper',
            'title': 'Academic Research Study on Machine Learning',
            'description': 'Comprehensive academic study with detailed methodology and results',
            'relevance_score': 0.9
        }
        
        low_quality_source = {
            'url': 'https://blog.example.com/post',
            'title': 'Quick thoughts',
            'description': 'Short blog post',
            'relevance_score': 0.3
        }
        
        high_score = await research_agent._calculate_source_quality_score(high_quality_source)
        low_score = await research_agent._calculate_source_quality_score(low_quality_source)
        
        assert high_score > low_score
        assert 0 <= high_score <= 1
        assert 0 <= low_score <= 1

    def test_domain_extraction(self, research_agent):
        """Test domain extraction from URLs."""
        test_cases = [
            ('https://example.com/path', 'example.com'),
            ('http://subdomain.test.org/page', 'subdomain.test.org'),
            ('https://research.university.edu/paper.html', 'research.university.edu'),
            ('invalid-url', ''),
            ('', '')
        ]
        
        for url, expected_domain in test_cases:
            assert research_agent._extract_domain(url) == expected_domain

    @pytest.mark.asyncio
    async def test_article_ranking_and_filtering(self, research_agent, sample_extracted_articles):
        """Test article ranking and filtering functionality."""
        # Add a low-confidence article
        low_confidence_article = ExtractedArticle(
            source=Source(
                url='https://low-quality.com/article',
                title='Low Quality Article',
                source_type=SourceType.WEB,
                metadata={'credibility_score': 0.2}
            ),
            content='Short content',
            key_findings=[],
            confidence_score=0.2,
            extraction_timestamp=datetime.now()
        )
        
        all_articles = sample_extracted_articles + [low_confidence_article]
        
        result = research_agent._rank_and_filter_articles(all_articles, max_articles=2)
        
        assert isinstance(result, list)
        assert len(result) <= 2
        # Should be sorted by confidence score (highest first)
        if len(result) > 1:
            assert result[0].confidence_score >= result[1].confidence_score