"""
Unit tests for GatewayWrapper.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from agent_scrivener.tools.gateway_wrapper import (
    GatewayWrapper, APIService, RateLimitConfig, APICredentials, 
    RateLimiter, RequestRecord
)
from agent_scrivener.models.core import AcademicPaper
from agent_scrivener.models.errors import ExternalAPIError, NetworkError


class TestRateLimiter:
    """Test cases for RateLimiter."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Rate limiter with test configuration."""
        config = RateLimitConfig(
            requests_per_minute=5,
            requests_per_hour=20,
            burst_limit=3
        )
        return RateLimiter(config)
    
    @pytest.mark.asyncio
    async def test_acquire_within_limits(self, rate_limiter):
        """Test acquiring permission within rate limits."""
        service = APIService.ARXIV
        endpoint = "test"
        
        # Should allow requests within limits
        for i in range(3):
            can_request = await rate_limiter.acquire(service, endpoint)
            assert can_request is True
            rate_limiter.record_request(service, endpoint, True)
    
    @pytest.mark.asyncio
    async def test_acquire_exceeds_minute_limit(self, rate_limiter):
        """Test rate limiting when minute limit is exceeded."""
        service = APIService.ARXIV
        endpoint = "test"
        
        # Fill up the minute limit
        for i in range(5):
            can_request = await rate_limiter.acquire(service, endpoint)
            assert can_request is True
            rate_limiter.record_request(service, endpoint, True)
        
        # Next request should be rate limited
        can_request = await rate_limiter.acquire(service, endpoint)
        assert can_request is False
    
    @pytest.mark.asyncio
    async def test_acquire_different_services(self, rate_limiter):
        """Test that different services have separate rate limits."""
        # Fill up limit for one service
        for i in range(5):
            can_request = await rate_limiter.acquire(APIService.ARXIV, "test")
            assert can_request is True
            rate_limiter.record_request(APIService.ARXIV, "test", True)
        
        # Different service should still be allowed
        can_request = await rate_limiter.acquire(APIService.PUBMED, "test")
        assert can_request is True
    
    def test_record_request_backoff_adjustment(self, rate_limiter):
        """Test backoff adjustment based on request success."""
        service = APIService.ARXIV
        endpoint = "test"
        
        initial_backoff = rate_limiter.current_backoff
        
        # Successful request should reduce backoff
        rate_limiter.record_request(service, endpoint, True)
        assert rate_limiter.current_backoff <= initial_backoff
        
        # Failed request should increase backoff
        rate_limiter.record_request(service, endpoint, False)
        assert rate_limiter.current_backoff > initial_backoff
    
    @pytest.mark.asyncio
    async def test_wait_if_needed(self, rate_limiter):
        """Test waiting when rate limited."""
        service = APIService.ARXIV
        endpoint = "test"
        
        # Fill up the rate limit
        for i in range(5):
            rate_limiter.record_request(service, endpoint, True)
        
        # Should wait when rate limited
        start_time = datetime.now()
        await rate_limiter.wait_if_needed(service, endpoint)
        end_time = datetime.now()
        
        # Should have waited some time
        assert (end_time - start_time).total_seconds() > 0


class TestGatewayWrapper:
    """Test cases for GatewayWrapper."""
    
    @pytest.fixture
    def mock_gateway_tool(self):
        """Mock AgentCore gateway tool."""
        mock_tool = AsyncMock()
        mock_tool.request.return_value = {
            'status': 'success',
            'data': {'mock': 'response'}
        }
        return mock_tool
    
    @pytest.fixture
    def gateway_wrapper(self, mock_gateway_tool):
        """Gateway wrapper instance with mocked dependencies."""
        return GatewayWrapper(mock_gateway_tool)
    
    @pytest.fixture
    def gateway_wrapper_no_tool(self):
        """Gateway wrapper instance without gateway tool for testing fallbacks."""
        return GatewayWrapper()
    
    def test_add_credentials(self, gateway_wrapper):
        """Test adding API credentials."""
        credentials = APICredentials(
            service=APIService.SEMANTIC_SCHOLAR,
            api_key="test-api-key",
            additional_headers={"Custom-Header": "value"}
        )
        
        gateway_wrapper.add_credentials(credentials)
        
        assert APIService.SEMANTIC_SCHOLAR in gateway_wrapper.credentials
        assert gateway_wrapper.credentials[APIService.SEMANTIC_SCHOLAR].api_key == "test-api-key"
    
    @pytest.mark.asyncio
    async def test_query_external_api_success(self, gateway_wrapper, mock_gateway_tool):
        """Test successful external API query."""
        service = APIService.ARXIV
        endpoint = "query"
        params = {"search_query": "test"}
        
        mock_gateway_tool.request.return_value = {
            'feed': {'entry': [{'title': 'Test Paper'}]}
        }
        
        result = await gateway_wrapper.query_external_api(service, endpoint, params)
        
        assert 'feed' in result
        assert result['feed']['entry'][0]['title'] == 'Test Paper'
        mock_gateway_tool.request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_external_api_with_retries(self, gateway_wrapper, mock_gateway_tool):
        """Test API query with retry logic."""
        service = APIService.ARXIV
        endpoint = "query"
        params = {"search_query": "test"}
        
        # Mock to fail twice, then succeed
        call_count = 0
        def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            return {'success': True}
        
        mock_gateway_tool.request.side_effect = mock_request
        
        # Mock sleep to speed up test
        with patch('asyncio.sleep', return_value=None):
            result = await gateway_wrapper.query_external_api(service, endpoint, params)
        
        assert result['success'] is True
        assert call_count == 3  # Failed twice, succeeded on third attempt
    
    @pytest.mark.asyncio
    async def test_query_external_api_all_retries_fail(self, gateway_wrapper, mock_gateway_tool):
        """Test API query when all retries fail."""
        service = APIService.ARXIV
        endpoint = "query"
        params = {"search_query": "test"}
        
        mock_gateway_tool.request.side_effect = Exception("Persistent failure")
        
        with patch('asyncio.sleep', return_value=None):
            with pytest.raises(ExternalAPIError) as exc_info:
                await gateway_wrapper.query_external_api(service, endpoint, params)
        
        assert "Failed to query" in str(exc_info.value)
        assert mock_gateway_tool.request.call_count == 4  # Initial + 3 retries
    
    @pytest.mark.asyncio
    async def test_query_external_api_caching(self, gateway_wrapper, mock_gateway_tool):
        """Test API response caching."""
        service = APIService.ARXIV
        endpoint = "query"
        params = {"search_query": "test"}
        
        mock_gateway_tool.request.return_value = {'cached': 'response'}
        
        # First request
        result1 = await gateway_wrapper.query_external_api(service, endpoint, params)
        
        # Second request should use cache
        result2 = await gateway_wrapper.query_external_api(service, endpoint, params)
        
        assert result1 == result2
        # Should only call the gateway once due to caching
        mock_gateway_tool.request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_arxiv(self, gateway_wrapper, mock_gateway_tool):
        """Test arXiv search functionality."""
        mock_response = {
            'feed': {
                'entry': [{
                    'id': 'http://arxiv.org/abs/2301.00001v1',
                    'title': 'Test Paper Title',
                    'summary': 'Test abstract content',
                    'author': [{'name': 'Test Author'}],
                    'published': '2023-01-01T00:00:00Z',
                    'doi': '10.48550/arXiv.2301.00001'
                }]
            }
        }
        
        mock_gateway_tool.request.return_value = mock_response
        
        papers = await gateway_wrapper.search_arxiv("machine learning", max_results=5)
        
        assert len(papers) == 1
        assert isinstance(papers[0], AcademicPaper)
        assert papers[0].title == 'Test Paper Title'
        assert papers[0].database_source == 'arXiv'
        assert papers[0].authors == ['Test Author']
    
    @pytest.mark.asyncio
    async def test_search_semantic_scholar(self, gateway_wrapper, mock_gateway_tool):
        """Test Semantic Scholar search functionality."""
        mock_response = {
            'data': [{
                'paperId': 'test-id',
                'title': 'Semantic Scholar Paper',
                'abstract': 'Test abstract',
                'authors': [{'name': 'Test Author'}],
                'year': 2023,
                'citationCount': 10,
                'url': 'https://example.com/paper',
                'doi': '10.1000/test.doi'
            }]
        }
        
        mock_gateway_tool.request.return_value = mock_response
        
        papers = await gateway_wrapper.search_semantic_scholar("deep learning", max_results=5)
        
        assert len(papers) == 1
        assert isinstance(papers[0], AcademicPaper)
        assert papers[0].title == 'Semantic Scholar Paper'
        assert papers[0].database_source == 'Semantic Scholar'
        assert papers[0].citation_count == 10
    
    @pytest.mark.asyncio
    async def test_search_pubmed(self, gateway_wrapper, mock_gateway_tool):
        """Test PubMed search functionality."""
        # Mock search response
        search_response = {
            'esearchresult': {
                'idlist': ['12345678', '87654321']
            }
        }
        
        # Mock fetch response (simplified)
        fetch_response = {'mock': 'pubmed_data'}
        
        # Configure mock to return different responses for different endpoints
        def mock_request(method, url, **kwargs):
            if 'esearch' in url:
                return search_response
            elif 'efetch' in url:
                return fetch_response
            return {}
        
        mock_gateway_tool.request.side_effect = mock_request
        
        papers = await gateway_wrapper.search_pubmed("cancer research", max_results=5)
        
        # Should return mock papers (implementation returns mock data for PubMed)
        assert len(papers) == 1
        assert isinstance(papers[0], AcademicPaper)
        assert papers[0].database_source == 'PubMed'
    
    @pytest.mark.asyncio
    async def test_get_paper_details_arxiv(self, gateway_wrapper, mock_gateway_tool):
        """Test getting paper details from arXiv."""
        paper_id = "2301.00001"
        
        mock_response = {
            'feed': {
                'entry': {
                    'id': f'http://arxiv.org/abs/{paper_id}v1',
                    'title': 'Detailed Paper Title',
                    'summary': 'Detailed abstract',
                    'author': [{'name': 'Detailed Author'}],
                    'published': '2023-01-01T00:00:00Z'
                }
            }
        }
        
        mock_gateway_tool.request.return_value = mock_response
        
        paper = await gateway_wrapper.get_paper_details(paper_id, APIService.ARXIV)
        
        assert paper is not None
        assert isinstance(paper, AcademicPaper)
        assert paper.title == 'Detailed Paper Title'
    
    @pytest.mark.asyncio
    async def test_get_paper_details_not_found(self, gateway_wrapper, mock_gateway_tool):
        """Test getting paper details when paper is not found."""
        paper_id = "nonexistent"
        
        mock_gateway_tool.request.return_value = {'feed': {'entry': []}}
        
        paper = await gateway_wrapper.get_paper_details(paper_id, APIService.ARXIV)
        
        assert paper is None
    
    @pytest.mark.asyncio
    async def test_batch_search(self, gateway_wrapper, mock_gateway_tool):
        """Test batch search across multiple services."""
        queries = ["machine learning", "deep learning"]
        services = [APIService.ARXIV, APIService.SEMANTIC_SCHOLAR]
        
        # Mock responses for different services
        def mock_request(method, url, **kwargs):
            if 'arxiv' in url:
                return {
                    'feed': {
                        'entry': [{
                            'title': 'arXiv Paper',
                            'summary': 'arXiv abstract',
                            'author': [{'name': 'arXiv Author'}],
                            'published': '2023-01-01T00:00:00Z'
                        }]
                    }
                }
            elif 'semanticscholar' in url:
                return {
                    'data': [{
                        'title': 'Semantic Scholar Paper',
                        'abstract': 'SS abstract',
                        'authors': [{'name': 'SS Author'}],
                        'year': 2023
                    }]
                }
            return {}
        
        mock_gateway_tool.request.side_effect = mock_request
        
        results = await gateway_wrapper.batch_search(queries, services, max_results_per_query=2)
        
        assert len(results) == 2
        assert "machine learning" in results
        assert "deep learning" in results
        
        # Each query should have results from both services
        for query_results in results.values():
            assert len(query_results) > 0
            assert all(isinstance(paper, AcademicPaper) for paper in query_results)
    
    @pytest.mark.asyncio
    async def test_batch_search_with_failures(self, gateway_wrapper, mock_gateway_tool):
        """Test batch search with some service failures."""
        queries = ["test query"]
        services = [APIService.ARXIV, APIService.SEMANTIC_SCHOLAR]
        
        # Mock one service to fail
        def mock_request(method, url, **kwargs):
            if 'arxiv' in url:
                raise Exception("arXiv service failed")
            elif 'semanticscholar' in url:
                return {
                    'data': [{
                        'title': 'Working Paper',
                        'abstract': 'Working abstract',
                        'authors': [{'name': 'Working Author'}],
                        'year': 2023
                    }]
                }
            return {}
        
        mock_gateway_tool.request.side_effect = mock_request
        
        results = await gateway_wrapper.batch_search(queries, services)
        
        # Should still return results from working service
        assert len(results) == 1
        assert len(results["test query"]) == 1
        assert results["test query"][0].title == "Working Paper"
    
    @pytest.mark.asyncio
    async def test_no_gateway_tool_fallback(self, gateway_wrapper_no_tool):
        """Test fallback behavior when no gateway tool is available."""
        papers = await gateway_wrapper_no_tool.search_arxiv("test query")
        
        # Should return mock data
        assert len(papers) == 1
        assert papers[0].title == "Mock arXiv Paper Title"
        assert papers[0].database_source == "arXiv"
    
    def test_generate_cache_key(self, gateway_wrapper):
        """Test cache key generation."""
        service = APIService.ARXIV
        endpoint = "query"
        params = {"search_query": "test", "max_results": 10}
        
        key1 = gateway_wrapper._generate_cache_key(service, endpoint, params)
        key2 = gateway_wrapper._generate_cache_key(service, endpoint, params)
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different parameters should generate different key
        different_params = {"search_query": "different", "max_results": 10}
        key3 = gateway_wrapper._generate_cache_key(service, endpoint, different_params)
        assert key1 != key3
    
    def test_cache_operations(self, gateway_wrapper):
        """Test cache storage and retrieval."""
        cache_key = "test_key"
        test_data = {"test": "data"}
        
        # Initially no cached data
        assert gateway_wrapper._get_cached_result(cache_key) is None
        
        # Cache data
        gateway_wrapper._cache_result(cache_key, test_data)
        
        # Should retrieve cached data
        cached = gateway_wrapper._get_cached_result(cache_key)
        assert cached == test_data
    
    def test_cache_expiration(self, gateway_wrapper):
        """Test cache expiration."""
        cache_key = "test_key"
        test_data = {"test": "data"}
        
        # Cache data
        gateway_wrapper._cache_result(cache_key, test_data)
        
        # Manually expire the cache entry
        gateway_wrapper.cache[cache_key]['timestamp'] = datetime.now() - timedelta(hours=2)
        
        # Should not retrieve expired data
        cached = gateway_wrapper._get_cached_result(cache_key)
        assert cached is None
        
        # Expired entry should be removed
        assert cache_key not in gateway_wrapper.cache
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, gateway_wrapper, mock_gateway_tool):
        """Test health check when all services are healthy."""
        # Mock successful responses
        def mock_request(method, url, **kwargs):
            if 'arxiv' in url:
                return {'feed': {'entry': [{'title': 'Test'}]}}
            elif 'semanticscholar' in url:
                return {'data': [{'title': 'Test'}]}
            return {}
        
        mock_gateway_tool.request.side_effect = mock_request
        
        health = await gateway_wrapper.health_check()
        
        assert health['status'] == 'healthy'
        assert health['gateway_available'] is True
        assert 'services' in health
        assert 'timestamp' in health
    
    @pytest.mark.asyncio
    async def test_health_check_degraded(self, gateway_wrapper, mock_gateway_tool):
        """Test health check when some services are unhealthy."""
        # Mock one service to fail
        def mock_request(method, url, **kwargs):
            if 'arxiv' in url:
                raise Exception("arXiv service down")
            elif 'semanticscholar' in url:
                return {'data': [{'title': 'Test'}]}
            return {}
        
        mock_gateway_tool.request.side_effect = mock_request
        
        health = await gateway_wrapper.health_check()
        
        assert health['status'] == 'degraded'
        assert health['services'][APIService.ARXIV.value]['status'] == 'unhealthy'
        assert 'arXiv service down' in health['services'][APIService.ARXIV.value]['error']
    
    def test_get_rate_limit_status(self, gateway_wrapper):
        """Test rate limit status reporting."""
        # Add some request history
        service = APIService.ARXIV
        for i in range(3):
            gateway_wrapper.rate_limiter.record_request(service, "test", True)
        
        status = gateway_wrapper.get_rate_limit_status()
        
        assert APIService.ARXIV.value in status
        arxiv_status = status[APIService.ARXIV.value]
        assert 'requests_last_minute' in arxiv_status
        assert 'requests_last_hour' in arxiv_status
        assert 'minute_limit' in arxiv_status
        assert 'hour_limit' in arxiv_status
        assert 'can_make_request' in arxiv_status
        assert arxiv_status['requests_last_minute'] == 3
    
    @pytest.mark.asyncio
    async def test_prepare_request_with_credentials(self, gateway_wrapper):
        """Test request preparation with API credentials."""
        # Add credentials
        credentials = APICredentials(
            service=APIService.SEMANTIC_SCHOLAR,
            api_key="test-api-key",
            additional_headers={"Custom-Header": "custom-value"}
        )
        gateway_wrapper.add_credentials(credentials)
        
        request_data = await gateway_wrapper._prepare_request(
            APIService.SEMANTIC_SCHOLAR,
            "paper/search",
            {"query": "test"},
            "GET",
            None
        )
        
        assert request_data['headers']['x-api-key'] == "test-api-key"
        assert request_data['headers']['Custom-Header'] == "custom-value"
        assert 'semanticscholar' in request_data['url']
    
    @pytest.mark.asyncio
    async def test_parse_arxiv_response(self, gateway_wrapper):
        """Test parsing of arXiv API response."""
        response = {
            'feed': {
                'entry': [
                    {
                        'id': 'http://arxiv.org/abs/2301.00001v1',
                        'title': 'Test Paper 1',
                        'summary': 'Abstract 1',
                        'author': [{'name': 'Author 1'}, {'name': 'Author 2'}],
                        'published': '2023-01-01T00:00:00Z',
                        'doi': '10.48550/arXiv.2301.00001'
                    },
                    {
                        'id': 'http://arxiv.org/abs/2301.00002v1',
                        'title': 'Test Paper 2',
                        'summary': 'Abstract 2',
                        'author': {'name': 'Single Author'},  # Test single author format
                        'published': '2023-02-01T00:00:00Z'
                    }
                ]
            }
        }
        
        papers = await gateway_wrapper._parse_arxiv_response(response)
        
        assert len(papers) == 2
        
        # Test first paper
        assert papers[0].title == 'Test Paper 1'
        assert papers[0].authors == ['Author 1', 'Author 2']
        assert papers[0].publication_year == 2023
        assert papers[0].database_source == 'arXiv'
        
        # Test second paper (single author)
        assert papers[1].title == 'Test Paper 2'
        assert papers[1].authors == ['Single Author']
    
    @pytest.mark.asyncio
    async def test_parse_semantic_scholar_response(self, gateway_wrapper):
        """Test parsing of Semantic Scholar API response."""
        response = {
            'data': [
                {
                    'paperId': 'test-id-1',
                    'title': 'SS Paper 1',
                    'abstract': 'SS Abstract 1',
                    'authors': [{'name': 'SS Author 1'}, {'name': 'SS Author 2'}],
                    'year': 2023,
                    'citationCount': 15,
                    'url': 'https://example.com/paper1',
                    'doi': '10.1000/test1'
                },
                {
                    'paperId': 'test-id-2',
                    'title': 'SS Paper 2',
                    'abstract': 'SS Abstract 2',
                    'authors': [{'name': 'SS Author 3'}],
                    'year': 2022,
                    'citationCount': 5,
                    'url': 'https://example.com/paper2'
                }
            ]
        }
        
        papers = await gateway_wrapper._parse_semantic_scholar_response(response)
        
        assert len(papers) == 2
        
        # Test first paper
        assert papers[0].title == 'SS Paper 1'
        assert papers[0].authors == ['SS Author 1', 'SS Author 2']
        assert papers[0].citation_count == 15
        assert papers[0].doi == '10.1000/test1'
        assert papers[0].database_source == 'Semantic Scholar'
        
        # Test second paper
        assert papers[1].title == 'SS Paper 2'
        assert papers[1].authors == ['SS Author 3']
        assert papers[1].citation_count == 5