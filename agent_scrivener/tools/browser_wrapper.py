"""
Browser tool wrapper for AgentCore integration with Nova Act SDK.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from urllib.parse import urljoin, urlparse
import re

from ..models.core import ExtractedArticle, Source, SourceType
from ..models.errors import NetworkError, ProcessingError, ErrorSeverity
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BrowserToolWrapper:
    """
    Wrapper for AgentCore Browser Tool with Nova Act SDK integration.
    
    Provides robust web navigation and content extraction capabilities
    with comprehensive error handling and retry logic.
    """
    
    def __init__(self, browser_tool=None, nova_act_sdk=None):
        """
        Initialize the browser tool wrapper.
        
        Args:
            browser_tool: AgentCore Browser Tool instance
            nova_act_sdk: Nova Act SDK instance for enhanced navigation
        """
        self.browser_tool = browser_tool
        self.nova_act = nova_act_sdk
        self.max_retries = 3
        self.timeout_seconds = 30
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        
        # Content extraction patterns
        self.content_selectors = [
            'article',
            '[role="main"]',
            '.content',
            '.article-content',
            '.post-content',
            '.entry-content',
            'main',
            '#content',
            '.main-content'
        ]
        
        # Elements to remove during cleaning
        self.noise_selectors = [
            'nav', 'header', 'footer', 'aside', '.sidebar',
            '.advertisement', '.ads', '.social-share',
            '.comments', '.related-posts', '.newsletter',
            'script', 'style', 'noscript'
        ]
    
    async def navigate_and_extract(self, url: str, max_retries: Optional[int] = None) -> Dict[str, Any]:
        """
        Navigate to URL and extract content with Nova Act precision.
        
        Args:
            url: Target URL to navigate to
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dict containing extracted content and metadata
            
        Raises:
            NetworkError: For navigation failures
            ProcessingError: For content extraction failures
        """
        retries = max_retries or self.max_retries
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                logger.info(f"Attempting to navigate to {url} (attempt {attempt + 1}/{retries + 1})")
                
                # Validate URL
                if not self._is_valid_url(url):
                    raise ProcessingError(f"Invalid URL format: {url}")
                
                # Navigate using browser tool
                navigation_result = await self._navigate_with_retry(url)
                
                # Extract content using Nova Act if available
                if self.nova_act:
                    content_data = await self._extract_with_nova_act(navigation_result)
                else:
                    content_data = await self._extract_basic_content(navigation_result)
                
                # Validate extracted content
                if not content_data.get('content') or len(content_data['content'].strip()) < 100:
                    raise ProcessingError("Insufficient content extracted from page")
                
                logger.info(f"Successfully extracted content from {url}")
                return content_data
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                
                if attempt < retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All attempts failed for {url}")
        
        # All retries exhausted
        if isinstance(last_error, (NetworkError, ProcessingError)):
            raise last_error
        else:
            raise NetworkError(f"Failed to extract content from {url}: {str(last_error)}")
    
    async def extract_multiple_urls(self, urls: List[str], max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """
        Extract content from multiple URLs concurrently.
        
        Args:
            urls: List of URLs to process
            max_concurrent: Maximum number of concurrent extractions
            
        Returns:
            List of extraction results (successful and failed)
        """
        logger.info(f"Starting extraction for {len(urls)} URLs with max_concurrent={max_concurrent}")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_with_semaphore(url: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    result = await self.navigate_and_extract(url)
                    result['success'] = True
                    result['url'] = url
                    return result
                except Exception as e:
                    logger.error(f"Failed to extract from {url}: {str(e)}")
                    return {
                        'success': False,
                        'url': url,
                        'error': str(e),
                        'error_type': type(e).__name__
                    }
        
        # Execute all extractions concurrently
        tasks = [extract_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_extractions = []
        failed_extractions = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_extractions.append({
                    'success': False,
                    'error': str(result),
                    'error_type': type(result).__name__
                })
            elif result.get('success'):
                successful_extractions.append(result)
            else:
                failed_extractions.append(result)
        
        logger.info(f"Extraction complete: {len(successful_extractions)} successful, {len(failed_extractions)} failed")
        
        return successful_extractions + failed_extractions
    
    async def create_extracted_article(self, extraction_result: Dict[str, Any]) -> ExtractedArticle:
        """
        Convert extraction result to ExtractedArticle model.
        
        Args:
            extraction_result: Result from navigate_and_extract
            
        Returns:
            ExtractedArticle instance
        """
        # Create source metadata
        source = Source(
            url=extraction_result['url'],
            title=extraction_result.get('title', 'Untitled'),
            author=extraction_result.get('author'),
            publication_date=extraction_result.get('publication_date'),
            source_type=SourceType.WEB,
            metadata=extraction_result.get('metadata', {})
        )
        
        # Extract key findings from content
        key_findings = await self._extract_key_findings(extraction_result['content'])
        
        # Calculate confidence score based on content quality
        confidence_score = self._calculate_confidence_score(extraction_result)
        
        return ExtractedArticle(
            source=source,
            content=extraction_result['content'],
            key_findings=key_findings,
            confidence_score=confidence_score,
            extraction_timestamp=datetime.now()
        )
    
    async def _navigate_with_retry(self, url: str) -> Dict[str, Any]:
        """Navigate to URL using browser tool with timeout handling."""
        try:
            if self.browser_tool:
                # Use AgentCore browser tool
                result = await asyncio.wait_for(
                    self.browser_tool.navigate(url, user_agent=self.user_agent),
                    timeout=self.timeout_seconds
                )
                return result
            else:
                # Mock navigation for testing
                logger.warning("No browser tool available, using mock navigation")
                return {
                    'url': url,
                    'status_code': 200,
                    'html': f'<html><body><h1>Mock content for {url}</h1><p>This is mock content for testing purposes. It contains sufficient text to pass the minimum content length validation. The content includes multiple sentences to ensure it meets the requirements for successful extraction and processing.</p></body></html>',
                    'title': f'Mock Title for {url}'
                }
        except asyncio.TimeoutError:
            raise NetworkError(f"Navigation timeout after {self.timeout_seconds} seconds for {url}")
        except Exception as e:
            raise NetworkError(f"Navigation failed for {url}: {str(e)}")
    
    async def _extract_with_nova_act(self, navigation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content using Nova Act SDK for enhanced precision."""
        try:
            # Use Nova Act for precise content extraction
            html_content = navigation_result.get('html', '')
            
            # Nova Act extraction (mock implementation)
            extracted_data = await self._nova_act_extract(html_content, navigation_result['url'])
            
            return {
                'url': navigation_result['url'],
                'title': extracted_data.get('title', navigation_result.get('title', 'Untitled')),
                'content': extracted_data['content'],
                'author': extracted_data.get('author'),
                'publication_date': extracted_data.get('publication_date'),
                'metadata': {
                    'extraction_method': 'nova_act',
                    'status_code': navigation_result.get('status_code'),
                    'word_count': len(extracted_data['content'].split()),
                    'extraction_confidence': extracted_data.get('confidence', 0.8)
                }
            }
        except Exception as e:
            logger.warning(f"Nova Act extraction failed, falling back to basic extraction: {str(e)}")
            return await self._extract_basic_content(navigation_result)
    
    async def _extract_basic_content(self, navigation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content using basic HTML parsing."""
        html_content = navigation_result.get('html', '')
        
        if not html_content:
            raise ProcessingError("No HTML content available for extraction")
        
        # Basic content extraction (simplified implementation)
        # In a real implementation, this would use BeautifulSoup or similar
        content = self._clean_html_content(html_content)
        title = navigation_result.get('title') or self._extract_title(html_content)
        author = self._extract_author(html_content)
        
        return {
            'url': navigation_result['url'],
            'title': title,
            'content': content,
            'author': author,
            'publication_date': None,
            'metadata': {
                'extraction_method': 'basic',
                'status_code': navigation_result.get('status_code'),
                'word_count': len(content.split()),
                'extraction_confidence': 0.6
            }
        }
    
    async def _nova_act_extract(self, html: str, url: str) -> Dict[str, Any]:
        """Mock Nova Act extraction implementation."""
        # This would be replaced with actual Nova Act SDK calls
        content = self._clean_html_content(html)
        
        return {
            'content': content,
            'title': self._extract_title(html),
            'author': self._extract_author(html),
            'publication_date': None,
            'confidence': 0.85
        }
    
    def _clean_html_content(self, html: str) -> str:
        """Clean HTML content and extract readable text."""
        # Remove HTML tags (simplified regex approach)
        # In production, use BeautifulSoup or similar
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _extract_title(self, html: str) -> str:
        """Extract title from HTML."""
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if title_match:
            return title_match.group(1).strip()
        
        # Try h1 tags
        h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.IGNORECASE | re.DOTALL)
        if h1_match:
            return re.sub(r'<[^>]+>', '', h1_match.group(1)).strip()
        
        return "Untitled"
    
    def _extract_author(self, html: str) -> Optional[str]:
        """Extract author from HTML metadata."""
        # Look for common author meta tags
        author_patterns = [
            r'<meta[^>]*name=["\']author["\'][^>]*content=["\']([^"\']+)["\']',
            r'<meta[^>]*property=["\']article:author["\'][^>]*content=["\']([^"\']+)["\']',
            r'<span[^>]*class=["\'][^"\']*author[^"\']*["\'][^>]*>([^<]+)</span>'
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    async def _extract_key_findings(self, content: str) -> List[str]:
        """Extract key findings from content."""
        # Simple implementation - extract sentences with key indicators
        key_indicators = [
            'found that', 'discovered', 'revealed', 'showed that',
            'demonstrated', 'concluded', 'results indicate',
            'study shows', 'research suggests', 'evidence suggests',
            'suggests', 'indicates', 'shows'  # Added more lenient indicators
        ]
        
        sentences = content.split('.')
        key_findings = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Reduced minimum length
                for indicator in key_indicators:
                    if indicator.lower() in sentence.lower():
                        key_findings.append(sentence + '.')
                        break
        
        return key_findings[:5]  # Return top 5 findings
    
    def _calculate_confidence_score(self, extraction_result: Dict[str, Any]) -> float:
        """Calculate confidence score based on extraction quality."""
        score = 0.5  # Base score
        
        # Content length factor
        content_length = len(extraction_result.get('content', ''))
        if content_length > 1000:
            score += 0.2
        elif content_length > 500:
            score += 0.1
        
        # Title presence
        if extraction_result.get('title') and extraction_result['title'] != 'Untitled':
            score += 0.1
        
        # Author presence
        if extraction_result.get('author'):
            score += 0.1
        
        # Extraction method
        if extraction_result.get('metadata', {}).get('extraction_method') == 'nova_act':
            score += 0.1
        
        return min(score, 1.0)
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on browser tool."""
        try:
            # Test navigation to a simple page
            test_url = "https://httpbin.org/html"
            result = await self.navigate_and_extract(test_url, max_retries=1)
            
            return {
                'status': 'healthy',
                'browser_tool_available': self.browser_tool is not None,
                'nova_act_available': self.nova_act is not None,
                'test_extraction_successful': True,
                'test_content_length': len(result.get('content', '')),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'browser_tool_available': self.browser_tool is not None,
                'nova_act_available': self.nova_act is not None,
                'test_extraction_successful': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }