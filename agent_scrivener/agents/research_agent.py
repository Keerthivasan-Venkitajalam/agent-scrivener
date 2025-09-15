"""
Research Agent for web search and content extraction.

Implements web search capabilities, content extraction and cleaning,
source validation and quality scoring for Agent Scrivener.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from urllib.parse import urljoin, urlparse

from .base import BaseAgent, AgentResult
from ..models.core import ExtractedArticle, Source, SourceType
from ..models.errors import NetworkError, ProcessingError, ValidationError
from ..tools.browser_wrapper import BrowserToolWrapper
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ResearchAgent(BaseAgent):
    """
    Research Agent for autonomous web search and content extraction.
    
    Handles web searches, content extraction, source validation,
    and quality scoring to gather comprehensive research data.
    """
    
    def __init__(self, browser_wrapper: BrowserToolWrapper, name: str = "research_agent"):
        """
        Initialize the Research Agent.
        
        Args:
            browser_wrapper: BrowserToolWrapper instance for web navigation
            name: Agent name for identification
        """
        super().__init__(name)
        self.browser = browser_wrapper
        self.max_sources_per_query = 10
        self.min_content_length = 100
        self.quality_threshold = 0.3
        
        # Search engines and patterns
        self.search_engines = [
            "https://www.google.com/search?q={query}",
            "https://www.bing.com/search?q={query}",
            "https://duckduckgo.com/?q={query}"
        ]
        
        # URL patterns to prioritize or avoid
        self.high_quality_domains = [
            'edu', 'gov', 'org', 'nature.com', 'science.org',
            'ieee.org', 'acm.org', 'springer.com', 'wiley.com'
        ]
        
        self.low_quality_domains = [
            'wikipedia.org',  # While useful, often not primary source
            'reddit.com', 'quora.com', 'yahoo.com',
            'pinterest.com', 'instagram.com', 'facebook.com'
        ]
    
    async def execute(self, **kwargs) -> AgentResult:
        """
        Execute research agent with search and extraction.
        
        Args:
            query: Research query string
            max_sources: Maximum number of sources to extract (optional)
            quality_threshold: Minimum quality score for sources (optional)
            
        Returns:
            AgentResult with extracted articles
        """
        return await self._execute_with_timing(self._perform_research, **kwargs)
    
    async def _perform_research(self, query: str, max_sources: Optional[int] = None, 
                              quality_threshold: Optional[float] = None) -> List[ExtractedArticle]:
        """
        Perform complete research workflow.
        
        Args:
            query: Research query string
            max_sources: Maximum number of sources to extract
            quality_threshold: Minimum quality score for sources
            
        Returns:
            List of ExtractedArticle objects
        """
        # Validate input
        self.validate_input(
            {"query": query}, 
            ["query"]
        )
        
        if not query.strip():
            raise ValidationError("Query cannot be empty")
        
        max_sources = max_sources or self.max_sources_per_query
        quality_threshold = quality_threshold or self.quality_threshold
        
        logger.info(f"Starting research for query: '{query}' (max_sources={max_sources})")
        
        # Step 1: Perform web search to find relevant URLs
        search_results = await self.search_web(query, max_sources * 2)  # Get more URLs than needed
        
        if not search_results:
            logger.warning(f"No search results found for query: '{query}'")
            return []
        
        logger.info(f"Found {len(search_results)} potential sources")
        
        # Step 2: Validate and score URLs
        validated_urls = await self.validate_sources(search_results)
        
        # Step 3: Extract content from top URLs
        extracted_articles = await self.extract_and_process_content(
            validated_urls[:max_sources], 
            quality_threshold
        )
        
        # Step 4: Final quality filtering and ranking
        final_articles = self._rank_and_filter_articles(extracted_articles, max_sources)
        
        logger.info(f"Research completed: {len(final_articles)} high-quality articles extracted")
        
        return final_articles
    
    async def search_web(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Perform web search to find relevant URLs.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search result dictionaries with URL and metadata
        """
        logger.info(f"Performing web search for: '{query}'")
        
        # For this implementation, we'll simulate search results
        # In production, this would integrate with actual search APIs
        search_results = await self._simulate_search_results(query, max_results)
        
        # Filter and clean results
        cleaned_results = []
        domain_counts = {}
        
        for result in search_results:
            url = result.get('url', '')
            domain = self._extract_domain(url)
            
            # Skip if we've seen too many from this domain
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                if domain_counts[domain] > 3:
                    continue
            
            # Basic URL validation
            if self._is_valid_research_url(url):
                cleaned_results.append(result)
        
        logger.info(f"Web search returned {len(cleaned_results)} valid URLs")
        return cleaned_results[:max_results]
    
    async def validate_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and score potential sources.
        
        Args:
            search_results: List of search result dictionaries
            
        Returns:
            List of validated sources with quality scores
        """
        logger.info(f"Validating {len(search_results)} potential sources")
        
        validated_sources = []
        
        for result in search_results:
            try:
                url = result['url']
                quality_score = await self._calculate_source_quality_score(result)
                
                validated_source = {
                    'url': url,
                    'title': result.get('title', 'Untitled'),
                    'description': result.get('description', ''),
                    'quality_score': quality_score,
                    'domain': self._extract_domain(url),
                    'estimated_relevance': result.get('relevance_score', 0.5)
                }
                
                validated_sources.append(validated_source)
                
            except Exception as e:
                logger.warning(f"Failed to validate source {result.get('url', 'unknown')}: {str(e)}")
                continue
        
        # Sort by quality score
        validated_sources.sort(key=lambda x: x['quality_score'], reverse=True)
        
        logger.info(f"Source validation complete: {len(validated_sources)} sources validated")
        return validated_sources
    
    async def extract_and_process_content(self, validated_sources: List[Dict[str, Any]], 
                                        quality_threshold: float) -> List[ExtractedArticle]:
        """
        Extract content from validated sources and create ExtractedArticle objects.
        
        Args:
            validated_sources: List of validated source dictionaries
            quality_threshold: Minimum quality threshold for inclusion
            
        Returns:
            List of ExtractedArticle objects
        """
        logger.info(f"Extracting content from {len(validated_sources)} sources")
        
        # Extract URLs for batch processing
        urls = [source['url'] for source in validated_sources]
        
        # Use browser wrapper for concurrent extraction
        extraction_results = await self.browser.extract_multiple_urls(urls, max_concurrent=5)
        
        extracted_articles = []
        
        for i, result in enumerate(extraction_results):
            try:
                if not result.get('success'):
                    logger.warning(f"Failed to extract content from {result.get('url', 'unknown')}: {result.get('error', 'Unknown error')}")
                    continue
                
                # Get corresponding source metadata
                source_metadata = validated_sources[i] if i < len(validated_sources) else {}
                
                # Create ExtractedArticle
                article = await self._create_extracted_article(result, source_metadata)
                
                # Apply quality threshold
                if article.confidence_score >= quality_threshold:
                    extracted_articles.append(article)
                    logger.debug(f"Article extracted: {article.source.title} (confidence: {article.confidence_score:.2f})")
                else:
                    logger.debug(f"Article rejected due to low confidence: {article.source.title} (confidence: {article.confidence_score:.2f})")
                
            except Exception as e:
                logger.error(f"Failed to process extraction result {i}: {str(e)}")
                continue
        
        logger.info(f"Content extraction complete: {len(extracted_articles)} articles processed")
        return extracted_articles
    
    async def _simulate_search_results(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Simulate search results for testing purposes.
        In production, this would call actual search APIs.
        """
        # Generate realistic mock search results based on query
        base_results = [
            {
                'url': f'https://example-research.edu/articles/{query.replace(" ", "-").lower()}-study',
                'title': f'Comprehensive Study on {query.title()}',
                'description': f'A detailed academic study examining various aspects of {query}.',
                'relevance_score': 0.9
            },
            {
                'url': f'https://scientific-journal.org/papers/{query.replace(" ", "_").lower()}_analysis',
                'title': f'Analysis of {query.title()}: Recent Developments',
                'description': f'Recent developments and analysis in the field of {query}.',
                'relevance_score': 0.85
            },
            {
                'url': f'https://research-institute.gov/reports/{query.replace(" ", "-").lower()}-report',
                'title': f'Government Report on {query.title()}',
                'description': f'Official government report detailing findings related to {query}.',
                'relevance_score': 0.8
            },
            {
                'url': f'https://tech-blog.com/articles/{query.replace(" ", "-").lower()}-overview',
                'title': f'{query.title()}: A Technical Overview',
                'description': f'Technical overview and practical applications of {query}.',
                'relevance_score': 0.7
            },
            {
                'url': f'https://industry-news.com/news/{query.replace(" ", "-").lower()}-trends',
                'title': f'Current Trends in {query.title()}',
                'description': f'Latest trends and industry insights about {query}.',
                'relevance_score': 0.65
            }
        ]
        
        # Extend results if more are needed
        extended_results = []
        for i in range(max_results):
            base_index = i % len(base_results)
            result = base_results[base_index].copy()
            
            if i >= len(base_results):
                # Modify URL and title for uniqueness
                result['url'] = result['url'] + f'-part-{i // len(base_results) + 1}'
                result['title'] = result['title'] + f' - Part {i // len(base_results) + 1}'
                result['relevance_score'] *= 0.9  # Slightly lower relevance for extended results
            
            extended_results.append(result)
        
        return extended_results[:max_results]
    
    async def _create_extracted_article(self, extraction_result: Dict[str, Any], 
                                      source_metadata: Dict[str, Any]) -> ExtractedArticle:
        """
        Create ExtractedArticle from extraction result and source metadata.
        
        Args:
            extraction_result: Result from browser extraction
            source_metadata: Additional source metadata from validation
            
        Returns:
            ExtractedArticle instance
        """
        # Create source object
        source = Source(
            url=extraction_result['url'],
            title=extraction_result.get('title', source_metadata.get('title', 'Untitled')),
            author=extraction_result.get('author'),
            publication_date=extraction_result.get('publication_date'),
            source_type=SourceType.WEB,
            metadata={
                'domain': source_metadata.get('domain', ''),
                'quality_score': source_metadata.get('quality_score', 0.5),
                'relevance_score': source_metadata.get('estimated_relevance', 0.5),
                'extraction_method': extraction_result.get('metadata', {}).get('extraction_method', 'unknown')
            }
        )
        
        # Extract key findings
        key_findings = await self._extract_key_findings(extraction_result['content'])
        
        # Calculate confidence score
        confidence_score = self._calculate_article_confidence(extraction_result, source_metadata)
        
        return ExtractedArticle(
            source=source,
            content=extraction_result['content'],
            key_findings=key_findings,
            confidence_score=confidence_score,
            extraction_timestamp=datetime.now()
        )
    
    async def _extract_key_findings(self, content: str) -> List[str]:
        """
        Extract key findings from article content.
        
        Args:
            content: Article content text
            
        Returns:
            List of key finding strings
        """
        # Key finding indicators
        finding_patterns = [
            r'[^.]*(?:found that|discovered|revealed|showed that|demonstrated)[^.]*\.',
            r'[^.]*(?:concluded|results indicate|study shows|research suggests)[^.]*\.',
            r'[^.]*(?:evidence suggests|data shows|analysis reveals)[^.]*\.',
            r'[^.]*(?:findings suggest|research indicates|study demonstrates)[^.]*\.'
        ]
        
        key_findings = []
        
        for pattern in finding_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                finding = match.strip()
                if len(finding) > 20 and finding not in key_findings:
                    key_findings.append(finding)
        
        # Also look for numbered findings or bullet points
        numbered_findings = re.findall(r'(?:^|\n)\s*(?:\d+\.|\*|\-)\s*([^.\n]+(?:\.[^.\n]*)*)', content, re.MULTILINE)
        for finding in numbered_findings:
            finding = finding.strip()
            if len(finding) > 20 and finding not in key_findings:
                key_findings.append(finding)
        
        return key_findings[:10]  # Return top 10 findings
    
    def _calculate_article_confidence(self, extraction_result: Dict[str, Any], 
                                    source_metadata: Dict[str, Any]) -> float:
        """
        Calculate confidence score for extracted article.
        
        Args:
            extraction_result: Browser extraction result
            source_metadata: Source validation metadata
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.3  # Base score
        
        # Content quality factors
        content_length = len(extraction_result.get('content', ''))
        if content_length > 2000:
            score += 0.2
        elif content_length > 1000:
            score += 0.15
        elif content_length > 500:
            score += 0.1
        
        # Source quality
        source_quality = source_metadata.get('quality_score', 0.5)
        score += source_quality * 0.3
        
        # Title and author presence
        if extraction_result.get('title') and extraction_result['title'] != 'Untitled':
            score += 0.1
        
        if extraction_result.get('author'):
            score += 0.1
        
        # Extraction method quality
        extraction_method = extraction_result.get('metadata', {}).get('extraction_method', '')
        if extraction_method == 'nova_act':
            score += 0.1
        elif extraction_method == 'basic':
            score += 0.05
        
        return min(score, 1.0)
    
    async def _calculate_source_quality_score(self, search_result: Dict[str, Any]) -> float:
        """
        Calculate quality score for a potential source.
        
        Args:
            search_result: Search result dictionary
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.5  # Base score
        url = search_result.get('url', '')
        domain = self._extract_domain(url)
        
        # Domain reputation
        if any(hq_domain in domain for hq_domain in self.high_quality_domains):
            score += 0.3
        elif any(lq_domain in domain for lq_domain in self.low_quality_domains):
            score -= 0.2
        
        # URL structure quality
        if '/research/' in url or '/study/' in url or '/paper/' in url:
            score += 0.1
        
        if '/blog/' in url or '/news/' in url:
            score += 0.05
        
        # Title quality
        title = search_result.get('title', '')
        if any(keyword in title.lower() for keyword in ['study', 'research', 'analysis', 'report']):
            score += 0.1
        
        # Description quality
        description = search_result.get('description', '')
        if len(description) > 100:
            score += 0.05
        
        # Relevance score from search
        relevance = search_result.get('relevance_score', 0.5)
        score += relevance * 0.2
        
        return max(0.0, min(score, 1.0))
    
    def _rank_and_filter_articles(self, articles: List[ExtractedArticle], 
                                max_articles: int) -> List[ExtractedArticle]:
        """
        Rank and filter articles by quality and relevance.
        
        Args:
            articles: List of extracted articles
            max_articles: Maximum number of articles to return
            
        Returns:
            Filtered and ranked list of articles
        """
        # Sort by confidence score
        ranked_articles = sorted(articles, key=lambda x: x.confidence_score, reverse=True)
        
        # Remove duplicates based on content similarity
        unique_articles = []
        seen_titles = set()
        
        for article in ranked_articles:
            title_key = article.source.title.lower().strip()
            if title_key not in seen_titles:
                unique_articles.append(article)
                seen_titles.add(title_key)
        
        return unique_articles[:max_articles]
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return ''
    
    def _is_valid_research_url(self, url: str) -> bool:
        """
        Check if URL is valid for research purposes.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid for research
        """
        if not url or not isinstance(url, str):
            return False
        
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                return False
            
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Skip certain file types
            skip_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx']
            if any(url.lower().endswith(ext) for ext in skip_extensions):
                return False
            
            # Skip social media and low-quality domains
            domain = parsed.netloc.lower()
            skip_domains = ['twitter.com', 'facebook.com', 'instagram.com', 'tiktok.com']
            if any(skip_domain in domain for skip_domain in skip_domains):
                return False
            
            return True
            
        except Exception:
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on research agent.
        
        Returns:
            Health check results
        """
        try:
            # Test basic functionality
            test_query = "artificial intelligence"
            search_results = await self.search_web(test_query, max_results=3)
            
            browser_health = await self.browser.health_check()
            
            return {
                'status': 'healthy',
                'agent_name': self.name,
                'browser_wrapper_status': browser_health.get('status', 'unknown'),
                'test_search_successful': len(search_results) > 0,
                'test_search_results_count': len(search_results),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'agent_name': self.name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }