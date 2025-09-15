"""
Unit tests for BrowserToolWrapper.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from agent_scrivener.tools.browser_wrapper import BrowserToolWrapper
from agent_scrivener.models.core import ExtractedArticle, Source, SourceType
from agent_scrivener.models.errors import NetworkError, ProcessingError


class TestBrowserToolWrapper:
    """Test cases for BrowserToolWrapper."""
    
    @pytest.fixture
    def mock_browser_tool(self):
        """Mock AgentCore browser tool."""
        mock_tool = AsyncMock()
        mock_tool.navigate.return_value = {
            'url': 'https://example.com/article',
            'status_code': 200,
            'html': '''
                <html>
                    <head>
                        <title>Test Article Title</title>
                        <meta name="author" content="John Doe">
                    </head>
                    <body>
                        <h1>Test Article Title</h1>
                        <p>This is a test article with meaningful content. The study found that testing is important for software quality.</p>
                        <p>Research suggests that comprehensive testing leads to better outcomes.</p>
                    </body>
                </html>
            ''',
            'title': 'Test Article Title'
        }
        return mock_tool
    
    @pytest.fixture
    def mock_nova_act(self):
        """Mock Nova Act SDK."""
        mock_sdk = AsyncMock()
        return mock_sdk
    
    @pytest.fixture
    def browser_wrapper(self, mock_browser_tool, mock_nova_act):
        """Browser wrapper instance with mocked dependencies."""
        return BrowserToolWrapper(mock_browser_tool, mock_nova_act)
    
    @pytest.fixture
    def browser_wrapper_no_tools(self):
        """Browser wrapper instance without tools for testing fallbacks."""
        return BrowserToolWrapper()
    
    @pytest.mark.asyncio
    async def test_navigate_and_extract_success(self, browser_wrapper, mock_browser_tool):
        """Test successful navigation and content extraction."""
        url = 'https://example.com/article'
        
        result = await browser_wrapper.navigate_and_extract(url)
        
        assert result['url'] == url
        assert result['title'] == 'Test Article Title'
        assert 'meaningful content' in result['content']
        assert result['author'] == 'John Doe'
        assert result['metadata']['extraction_method'] == 'nova_act'
        
        mock_browser_tool.navigate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_navigate_and_extract_invalid_url(self, browser_wrapper):
        """Test handling of invalid URLs."""
        invalid_url = 'not-a-valid-url'
        
        with pytest.raises(ProcessingError) as exc_info:
            await browser_wrapper.navigate_and_extract(invalid_url)
        
        assert 'Invalid URL format' in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_navigate_and_extract_network_error(self, browser_wrapper, mock_browser_tool):
        """Test handling of network errors with retry logic."""
        url = 'https://example.com/article'
        mock_browser_tool.navigate.side_effect = Exception("Connection failed")
        
        with pytest.raises(NetworkError) as exc_info:
            await browser_wrapper.navigate_and_extract(url, max_retries=2)
        
        assert 'Failed to extract content' in str(exc_info.value)
        # Should retry 3 times (initial + 2 retries)
        assert mock_browser_tool.navigate.call_count == 3
    
    @pytest.mark.asyncio
    async def test_navigate_and_extract_timeout(self, browser_wrapper, mock_browser_tool):
        """Test handling of navigation timeouts."""
        url = 'https://example.com/article'
        mock_browser_tool.navigate.side_effect = asyncio.TimeoutError()
        
        with pytest.raises(NetworkError) as exc_info:
            await browser_wrapper.navigate_and_extract(url, max_retries=1)
        
        assert 'Navigation timeout' in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_navigate_and_extract_insufficient_content(self, browser_wrapper, mock_browser_tool):
        """Test handling of pages with insufficient content."""
        url = 'https://example.com/empty'
        mock_browser_tool.navigate.return_value = {
            'url': url,
            'status_code': 200,
            'html': '<html><body><p>Short</p></body></html>',
            'title': 'Empty Page'
        }
        
        with pytest.raises(ProcessingError) as exc_info:
            await browser_wrapper.navigate_and_extract(url)
        
        assert 'Insufficient content extracted' in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_navigate_and_extract_fallback_to_basic(self, browser_wrapper, mock_browser_tool):
        """Test fallback to basic extraction when Nova Act fails."""
        url = 'https://example.com/article'
        
        # Mock Nova Act to fail
        with patch.object(browser_wrapper, '_extract_with_nova_act', side_effect=Exception("Nova Act failed")):
            result = await browser_wrapper.navigate_and_extract(url)
        
        assert result['url'] == url
        assert result['metadata']['extraction_method'] == 'basic'
        assert result['metadata']['extraction_confidence'] == 0.6
    
    @pytest.mark.asyncio
    async def test_navigate_and_extract_no_browser_tool(self, browser_wrapper_no_tools):
        """Test extraction without browser tool (mock mode)."""
        url = 'https://example.com/article'
        
        result = await browser_wrapper_no_tools.navigate_and_extract(url)
        
        assert result['url'] == url
        assert 'Mock content' in result['content']
        assert result['title'].startswith('Mock Title')
    
    @pytest.mark.asyncio
    async def test_extract_multiple_urls_success(self, browser_wrapper, mock_browser_tool):
        """Test concurrent extraction from multiple URLs."""
        urls = [
            'https://example.com/article1',
            'https://example.com/article2',
            'https://example.com/article3'
        ]
        
        # Mock different responses for each URL
        def mock_navigate(url, **kwargs):
            return {
                'url': url,
                'status_code': 200,
                'html': f'<html><body><h1>Title for {url}</h1><p>Content for {url} with sufficient length to pass validation checks.</p></body></html>',
                'title': f'Title for {url}'
            }
        
        mock_browser_tool.navigate.side_effect = mock_navigate
        
        results = await browser_wrapper.extract_multiple_urls(urls, max_concurrent=2)
        
        successful_results = [r for r in results if r.get('success')]
        assert len(successful_results) == 3
        
        for i, result in enumerate(successful_results):
            assert result['url'] == urls[i]
            assert result['success'] is True
    
    @pytest.mark.asyncio
    async def test_extract_multiple_urls_partial_failure(self, browser_wrapper, mock_browser_tool):
        """Test handling of partial failures in multiple URL extraction."""
        urls = [
            'https://example.com/good',
            'https://example.com/bad',
            'https://example.com/good2'
        ]
        
        def mock_navigate(url, **kwargs):
            if 'bad' in url:
                raise Exception("Network error")
            return {
                'url': url,
                'status_code': 200,
                'html': f'<html><body><h1>Title</h1><p>Good content for {url} with sufficient length.</p></body></html>',
                'title': 'Title'
            }
        
        mock_browser_tool.navigate.side_effect = mock_navigate
        
        results = await browser_wrapper.extract_multiple_urls(urls)
        
        successful_results = [r for r in results if r.get('success')]
        failed_results = [r for r in results if not r.get('success')]
        
        assert len(successful_results) == 2
        assert len(failed_results) == 1
        assert 'Network error' in failed_results[0]['error']
    
    @pytest.mark.asyncio
    async def test_create_extracted_article(self, browser_wrapper):
        """Test creation of ExtractedArticle from extraction result."""
        extraction_result = {
            'url': 'https://example.com/article',
            'title': 'Test Article',
            'content': 'This is test content. The study found that testing is crucial. Research suggests comprehensive validation.',
            'author': 'Jane Smith',
            'publication_date': None,
            'metadata': {
                'extraction_method': 'nova_act',
                'extraction_confidence': 0.85
            }
        }
        
        article = await browser_wrapper.create_extracted_article(extraction_result)
        
        assert isinstance(article, ExtractedArticle)
        assert str(article.source.url) == extraction_result['url']
        assert article.source.title == extraction_result['title']
        assert article.source.author == extraction_result['author']
        assert article.source.source_type == SourceType.WEB
        assert article.content == extraction_result['content']
        assert len(article.key_findings) > 0
        assert 0.0 <= article.confidence_score <= 1.0
        assert isinstance(article.extraction_timestamp, datetime)
    
    def test_clean_html_content(self, browser_wrapper):
        """Test HTML content cleaning."""
        html = '''
            <html>
                <head><title>Test</title></head>
                <body>
                    <script>alert('test');</script>
                    <style>body { color: red; }</style>
                    <p>This is <strong>clean</strong> content.</p>
                    <div>More content here.</div>
                </body>
            </html>
        '''
        
        cleaned = browser_wrapper._clean_html_content(html)
        
        assert 'alert' not in cleaned
        assert 'color: red' not in cleaned
        assert 'This is clean content' in cleaned
        assert 'More content here' in cleaned
    
    def test_extract_title(self, browser_wrapper):
        """Test title extraction from HTML."""
        html_with_title = '<html><head><title>Page Title</title></head><body></body></html>'
        html_with_h1 = '<html><body><h1>Header Title</h1></body></html>'
        html_without_title = '<html><body><p>No title</p></body></html>'
        
        assert browser_wrapper._extract_title(html_with_title) == 'Page Title'
        assert browser_wrapper._extract_title(html_with_h1) == 'Header Title'
        assert browser_wrapper._extract_title(html_without_title) == 'Untitled'
    
    def test_extract_author(self, browser_wrapper):
        """Test author extraction from HTML."""
        html_with_author = '<html><head><meta name="author" content="John Doe"></head></html>'
        html_with_article_author = '<html><head><meta property="article:author" content="Jane Smith"></head></html>'
        html_without_author = '<html><body><p>No author</p></body></html>'
        
        assert browser_wrapper._extract_author(html_with_author) == 'John Doe'
        assert browser_wrapper._extract_author(html_with_article_author) == 'Jane Smith'
        assert browser_wrapper._extract_author(html_without_author) is None
    
    @pytest.mark.asyncio
    async def test_extract_key_findings(self, browser_wrapper):
        """Test key findings extraction from content."""
        content = '''
            This is an introduction. The study found that machine learning improves accuracy.
            Some background information here. Research suggests that deep learning is effective.
            More text. The results indicate significant improvements in performance.
            Conclusion text here.
        '''
        
        findings = await browser_wrapper._extract_key_findings(content)
        
        assert len(findings) > 0
        assert any('study found that' in finding.lower() for finding in findings)
        assert any('research suggests' in finding.lower() for finding in findings)
        assert any('results indicate' in finding.lower() for finding in findings)
    
    def test_calculate_confidence_score(self, browser_wrapper):
        """Test confidence score calculation."""
        high_quality_result = {
            'content': 'A' * 1500,  # Long content
            'title': 'Good Title',
            'author': 'Author Name',
            'metadata': {'extraction_method': 'nova_act'}
        }
        
        low_quality_result = {
            'content': 'Short content',
            'title': 'Untitled',
            'author': None,
            'metadata': {'extraction_method': 'basic'}
        }
        
        high_score = browser_wrapper._calculate_confidence_score(high_quality_result)
        low_score = browser_wrapper._calculate_confidence_score(low_quality_result)
        
        assert high_score > low_score
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0
    
    def test_is_valid_url(self, browser_wrapper):
        """Test URL validation."""
        valid_urls = [
            'https://example.com',
            'http://test.org/path',
            'https://subdomain.example.com/path?query=value'
        ]
        
        invalid_urls = [
            'not-a-url',
            'ftp://example.com',  # Scheme not http/https
            'example.com',  # Missing scheme
            ''
        ]
        
        for url in valid_urls:
            assert browser_wrapper._is_valid_url(url), f"Should be valid: {url}"
        
        for url in invalid_urls:
            assert not browser_wrapper._is_valid_url(url), f"Should be invalid: {url}"
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, browser_wrapper, mock_browser_tool):
        """Test health check when system is healthy."""
        # Mock successful extraction
        with patch.object(browser_wrapper, 'navigate_and_extract') as mock_extract:
            mock_extract.return_value = {'content': 'Test content for health check'}
            
            health = await browser_wrapper.health_check()
            
            assert health['status'] == 'healthy'
            assert health['browser_tool_available'] is True
            assert health['nova_act_available'] is True
            assert health['test_extraction_successful'] is True
            assert 'timestamp' in health
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, browser_wrapper):
        """Test health check when system is unhealthy."""
        # Mock failed extraction
        with patch.object(browser_wrapper, 'navigate_and_extract', side_effect=Exception("Health check failed")):
            health = await browser_wrapper.health_check()
            
            assert health['status'] == 'unhealthy'
            assert health['test_extraction_successful'] is False
            assert 'Health check failed' in health['error']
            assert 'timestamp' in health
    
    @pytest.mark.asyncio
    async def test_retry_logic_with_exponential_backoff(self, browser_wrapper, mock_browser_tool):
        """Test retry logic with exponential backoff."""
        url = 'https://example.com/retry-test'
        
        # Mock to fail twice, then succeed
        call_count = 0
        def mock_navigate_with_failures(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            return {
                'url': url,
                'status_code': 200,
                'html': '<html><body><h1>Success</h1><p>Content after retries with sufficient length for validation.</p></body></html>',
                'title': 'Success'
            }
        
        mock_browser_tool.navigate.side_effect = mock_navigate_with_failures
        
        # Mock sleep to speed up test
        with patch('asyncio.sleep', return_value=None):
            result = await browser_wrapper.navigate_and_extract(url, max_retries=3)
        
        assert result['url'] == url
        assert result['title'] == 'Success'
        assert call_count == 3  # Failed twice, succeeded on third attempt
    
    @pytest.mark.asyncio
    async def test_concurrent_extraction_with_semaphore(self, browser_wrapper, mock_browser_tool):
        """Test that concurrent extraction respects semaphore limits."""
        urls = [f'https://example.com/article{i}' for i in range(10)]
        
        # Track concurrent calls
        active_calls = 0
        max_concurrent_calls = 0
        
        async def mock_navigate_with_tracking(url, **kwargs):
            nonlocal active_calls, max_concurrent_calls
            active_calls += 1
            max_concurrent_calls = max(max_concurrent_calls, active_calls)
            
            # Simulate some processing time
            await asyncio.sleep(0.01)
            
            active_calls -= 1
            return {
                'url': url,
                'status_code': 200,
                'html': f'<html><body><h1>Title</h1><p>Content for {url} with sufficient length.</p></body></html>',
                'title': 'Title'
            }
        
        mock_browser_tool.navigate.side_effect = mock_navigate_with_tracking
        
        results = await browser_wrapper.extract_multiple_urls(urls, max_concurrent=3)
        
        # Should not exceed the semaphore limit
        assert max_concurrent_calls <= 3
        assert len([r for r in results if r.get('success')]) == 10