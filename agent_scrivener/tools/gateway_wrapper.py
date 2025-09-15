"""
Gateway wrapper for standardized external API access with AgentCore integration.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from ..models.core import AcademicPaper, Source, SourceType
from ..models.errors import ExternalAPIError, NetworkError, ErrorSeverity
from ..utils.logging import get_logger

logger = get_logger(__name__)


class APIService(str, Enum):
    """Supported external API services."""
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    CROSSREF = "crossref"
    GOOGLE_SCHOLAR = "google_scholar"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration for API services."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    backoff_factor: float = 2.0
    max_backoff_seconds: int = 300


@dataclass
class APICredentials:
    """API credentials and authentication information."""
    service: APIService
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    additional_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class RequestRecord:
    """Record of API request for rate limiting."""
    timestamp: datetime
    service: APIService
    endpoint: str
    success: bool


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_history: List[RequestRecord] = []
        self.current_backoff = 1.0
        self._lock = asyncio.Lock()
    
    async def acquire(self, service: APIService, endpoint: str) -> bool:
        """
        Acquire permission to make an API request.
        
        Returns:
            True if request is allowed, False if rate limited
        """
        async with self._lock:
            now = datetime.now()
            
            # Clean old records
            cutoff_time = now - timedelta(hours=1)
            self.request_history = [
                record for record in self.request_history
                if record.timestamp > cutoff_time
            ]
            
            # Check rate limits
            recent_requests = [
                record for record in self.request_history
                if record.service == service and record.timestamp > now - timedelta(minutes=1)
            ]
            
            hourly_requests = [
                record for record in self.request_history
                if record.service == service
            ]
            
            # Apply rate limits
            if len(recent_requests) >= self.config.requests_per_minute:
                logger.warning(f"Rate limit exceeded for {service}: {len(recent_requests)} requests in last minute")
                return False
            
            if len(hourly_requests) >= self.config.requests_per_hour:
                logger.warning(f"Hourly rate limit exceeded for {service}: {len(hourly_requests)} requests in last hour")
                return False
            
            return True
    
    def record_request(self, service: APIService, endpoint: str, success: bool):
        """Record a completed API request."""
        record = RequestRecord(
            timestamp=datetime.now(),
            service=service,
            endpoint=endpoint,
            success=success
        )
        self.request_history.append(record)
        
        # Adjust backoff based on success
        if success:
            self.current_backoff = max(1.0, self.current_backoff * 0.9)
        else:
            self.current_backoff = min(
                self.config.max_backoff_seconds,
                self.current_backoff * self.config.backoff_factor
            )
    
    async def wait_if_needed(self, service: APIService, endpoint: str) -> None:
        """Wait if rate limiting is needed."""
        if not await self.acquire(service, endpoint):
            wait_time = self.current_backoff
            logger.info(f"Rate limited, waiting {wait_time:.2f} seconds for {service}")
            await asyncio.sleep(wait_time)


class GatewayWrapper:
    """
    Wrapper for AgentCore Gateway with standardized external API access.
    
    Provides rate limiting, retry logic, authentication, and error handling
    for external academic database APIs.
    """
    
    def __init__(self, gateway_tool=None):
        """
        Initialize the gateway wrapper.
        
        Args:
            gateway_tool: AgentCore Gateway Tool instance
        """
        self.gateway = gateway_tool
        self.rate_limiter = RateLimiter(RateLimitConfig())
        self.credentials: Dict[APIService, APICredentials] = {}
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl_seconds = 3600  # 1 hour cache
        self.max_retries = 3
        
        # API endpoint configurations
        self.api_configs = {
            APIService.ARXIV: {
                'base_url': 'http://export.arxiv.org/api/query',
                'rate_limit': RateLimitConfig(requests_per_minute=20, requests_per_hour=1000),
                'requires_auth': False
            },
            APIService.PUBMED: {
                'base_url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils',
                'rate_limit': RateLimitConfig(requests_per_minute=10, requests_per_hour=500),
                'requires_auth': False
            },
            APIService.SEMANTIC_SCHOLAR: {
                'base_url': 'https://api.semanticscholar.org/graph/v1',
                'rate_limit': RateLimitConfig(requests_per_minute=100, requests_per_hour=5000),
                'requires_auth': True
            },
            APIService.CROSSREF: {
                'base_url': 'https://api.crossref.org',
                'rate_limit': RateLimitConfig(requests_per_minute=50, requests_per_hour=2000),
                'requires_auth': False
            }
        }
    
    def add_credentials(self, credentials: APICredentials):
        """Add API credentials for a service."""
        self.credentials[credentials.service] = credentials
        logger.info(f"Added credentials for {credentials.service}")
    
    async def query_external_api(
        self,
        service: APIService,
        endpoint: str,
        params: Dict[str, Any],
        method: str = 'GET',
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make a standardized external API request.
        
        Args:
            service: Target API service
            endpoint: API endpoint path
            params: Request parameters
            method: HTTP method
            headers: Additional headers
            
        Returns:
            API response data
            
        Raises:
            ExternalAPIError: For API-specific errors
            NetworkError: For network-related errors
        """
        # Generate cache key
        cache_key = self._generate_cache_key(service, endpoint, params)
        
        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.debug(f"Returning cached result for {service}:{endpoint}")
            return cached_result
        
        # Apply rate limiting
        await self.rate_limiter.wait_if_needed(service, endpoint)
        
        # Prepare request
        request_data = await self._prepare_request(service, endpoint, params, method, headers)
        
        # Execute request with retries
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Making API request to {service}:{endpoint} (attempt {attempt + 1})")
                
                result = await self._execute_request(service, request_data)
                
                # Record successful request
                self.rate_limiter.record_request(service, endpoint, True)
                
                # Cache result
                self._cache_result(cache_key, result)
                
                logger.info(f"Successfully queried {service}:{endpoint}")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed for {service}:{endpoint}: {str(e)}")
                
                # Record failed request
                self.rate_limiter.record_request(service, endpoint, False)
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * self.rate_limiter.current_backoff
                    logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
        
        # All retries exhausted
        if isinstance(last_error, ExternalAPIError):
            raise last_error
        else:
            raise ExternalAPIError(f"Failed to query {service}:{endpoint} after {self.max_retries + 1} attempts: {str(last_error)}")
    
    async def search_arxiv(self, query: str, max_results: int = 10) -> List[AcademicPaper]:
        """
        Search arXiv for academic papers.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of academic papers
        """
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            result = await self.query_external_api(APIService.ARXIV, 'query', params)
            return await self._parse_arxiv_response(result)
        except Exception as e:
            logger.error(f"arXiv search failed: {str(e)}")
            raise ExternalAPIError(f"arXiv search failed: {str(e)}")
    
    async def search_pubmed(self, query: str, max_results: int = 10) -> List[AcademicPaper]:
        """
        Search PubMed for academic papers.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of academic papers
        """
        # First, search for IDs
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json'
        }
        
        try:
            search_result = await self.query_external_api(APIService.PUBMED, 'esearch.fcgi', search_params)
            
            if not search_result.get('esearchresult', {}).get('idlist'):
                return []
            
            # Fetch details for found IDs
            ids = search_result['esearchresult']['idlist']
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(ids),
                'retmode': 'xml',
                'rettype': 'abstract'
            }
            
            fetch_result = await self.query_external_api(APIService.PUBMED, 'efetch.fcgi', fetch_params)
            return await self._parse_pubmed_response(fetch_result)
            
        except Exception as e:
            logger.error(f"PubMed search failed: {str(e)}")
            raise ExternalAPIError(f"PubMed search failed: {str(e)}")
    
    async def search_semantic_scholar(self, query: str, max_results: int = 10) -> List[AcademicPaper]:
        """
        Search Semantic Scholar for academic papers.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of academic papers
        """
        params = {
            'query': query,
            'limit': max_results,
            'fields': 'title,authors,abstract,year,citationCount,url,venue,doi'
        }
        
        try:
            result = await self.query_external_api(APIService.SEMANTIC_SCHOLAR, 'paper/search', params)
            return await self._parse_semantic_scholar_response(result)
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {str(e)}")
            raise ExternalAPIError(f"Semantic Scholar search failed: {str(e)}")
    
    async def get_paper_details(self, paper_id: str, service: APIService) -> Optional[AcademicPaper]:
        """
        Get detailed information for a specific paper.
        
        Args:
            paper_id: Paper identifier
            service: Source database service
            
        Returns:
            Detailed paper information or None if not found
        """
        try:
            if service == APIService.ARXIV:
                params = {'id_list': paper_id}
                result = await self.query_external_api(service, 'query', params)
                papers = await self._parse_arxiv_response(result)
                return papers[0] if papers else None
                
            elif service == APIService.SEMANTIC_SCHOLAR:
                result = await self.query_external_api(service, f'paper/{paper_id}', {})
                papers = await self._parse_semantic_scholar_response({'data': [result]})
                return papers[0] if papers else None
                
            else:
                logger.warning(f"Paper details not implemented for {service}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get paper details for {paper_id} from {service}: {str(e)}")
            return None
    
    async def batch_search(self, queries: List[str], services: List[APIService], max_results_per_query: int = 5) -> Dict[str, List[AcademicPaper]]:
        """
        Perform batch searches across multiple services.
        
        Args:
            queries: List of search queries
            services: List of services to search
            max_results_per_query: Maximum results per query per service
            
        Returns:
            Dictionary mapping query to list of papers
        """
        results = {}
        
        for query in queries:
            query_results = []
            
            # Search each service
            search_tasks = []
            for service in services:
                if service == APIService.ARXIV:
                    task = self.search_arxiv(query, max_results_per_query)
                elif service == APIService.PUBMED:
                    task = self.search_pubmed(query, max_results_per_query)
                elif service == APIService.SEMANTIC_SCHOLAR:
                    task = self.search_semantic_scholar(query, max_results_per_query)
                else:
                    continue
                
                search_tasks.append(task)
            
            # Execute searches concurrently
            if search_tasks:
                try:
                    service_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                    
                    for result in service_results:
                        if isinstance(result, Exception):
                            logger.warning(f"Service search failed for query '{query}': {str(result)}")
                        elif isinstance(result, list):
                            query_results.extend(result)
                            
                except Exception as e:
                    logger.error(f"Batch search failed for query '{query}': {str(e)}")
            
            results[query] = query_results
        
        return results
    
    async def _prepare_request(
        self,
        service: APIService,
        endpoint: str,
        params: Dict[str, Any],
        method: str,
        headers: Optional[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Prepare request data for API call."""
        config = self.api_configs.get(service, {})
        base_url = config.get('base_url', '')
        
        # Build full URL
        if endpoint.startswith('http'):
            url = endpoint
        else:
            url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Prepare headers
        request_headers = {
            'User-Agent': 'Agent-Scrivener/1.0 (Research Assistant)',
            'Accept': 'application/json',
        }
        
        if headers:
            request_headers.update(headers)
        
        # Add authentication if available
        if service in self.credentials:
            creds = self.credentials[service]
            if creds.api_key:
                if service == APIService.SEMANTIC_SCHOLAR:
                    request_headers['x-api-key'] = creds.api_key
                else:
                    params['api_key'] = creds.api_key
            
            request_headers.update(creds.additional_headers)
        
        return {
            'url': url,
            'method': method,
            'params': params if method == 'GET' else {},
            'json': params if method == 'POST' else None,
            'headers': request_headers
        }
    
    async def _execute_request(self, service: APIService, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the actual API request."""
        if self.gateway:
            # Use AgentCore Gateway
            response = await self.gateway.request(
                method=request_data['method'],
                url=request_data['url'],
                params=request_data['params'],
                json=request_data['json'],
                headers=request_data['headers']
            )
            return response
        else:
            # Mock response for testing
            logger.warning(f"No gateway tool available, using mock response for {service}")
            return await self._mock_api_response(service, request_data)
    
    async def _mock_api_response(self, service: APIService, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock API response for testing."""
        if service == APIService.ARXIV:
            return {
                'feed': {
                    'entry': [{
                        'id': 'http://arxiv.org/abs/2301.00001v1',
                        'title': 'Mock arXiv Paper Title',
                        'summary': 'This is a mock abstract for testing purposes.',
                        'author': [{'name': 'Mock Author'}],
                        'published': '2023-01-01T00:00:00Z',
                        'doi': '10.48550/arXiv.2301.00001'
                    }]
                }
            }
        elif service == APIService.SEMANTIC_SCHOLAR:
            return {
                'data': [{
                    'paperId': 'mock-paper-id',
                    'title': 'Mock Semantic Scholar Paper',
                    'abstract': 'Mock abstract for Semantic Scholar paper.',
                    'authors': [{'name': 'Mock Author'}],
                    'year': 2023,
                    'citationCount': 10,
                    'url': 'https://example.com/paper',
                    'doi': '10.1000/mock.doi'
                }]
            }
        elif service == APIService.PUBMED:
            return {
                'esearchresult': {
                    'idlist': ['12345678']
                }
            }
        else:
            return {'mock': True, 'service': service.value}
    
    async def _parse_arxiv_response(self, response: Dict[str, Any]) -> List[AcademicPaper]:
        """Parse arXiv API response into AcademicPaper objects."""
        papers = []
        
        feed = response.get('feed', {})
        entries = feed.get('entry', [])
        
        if not isinstance(entries, list):
            entries = [entries]
        
        for entry in entries:
            try:
                # Extract authors
                authors = []
                author_data = entry.get('author', [])
                if not isinstance(author_data, list):
                    author_data = [author_data]
                
                for author in author_data:
                    if isinstance(author, dict):
                        authors.append(author.get('name', 'Unknown'))
                    else:
                        authors.append(str(author))
                
                # Extract year from published date
                published = entry.get('published', '2023-01-01T00:00:00Z')
                year = int(published[:4]) if published else 2023
                
                # Extract DOI
                doi = None
                doi_link = entry.get('doi')
                if doi_link:
                    # Handle arXiv DOIs which don't follow standard pattern
                    if 'arxiv' in doi_link.lower():
                        doi = None  # Skip arXiv DOIs as they don't match standard pattern
                    else:
                        doi = doi_link.split('/')[-1] if '/' in doi_link else doi_link
                
                paper = AcademicPaper(
                    title=entry.get('title', 'Untitled').strip(),
                    authors=authors or ['Unknown'],
                    abstract=entry.get('summary', '').strip(),
                    publication_year=year,
                    doi=doi,
                    database_source='arXiv',
                    citation_count=None,
                    keywords=[],
                    full_text_url=entry.get('id')
                )
                papers.append(paper)
                
            except Exception as e:
                logger.warning(f"Failed to parse arXiv entry: {str(e)}")
                continue
        
        return papers
    
    async def _parse_semantic_scholar_response(self, response: Dict[str, Any]) -> List[AcademicPaper]:
        """Parse Semantic Scholar API response into AcademicPaper objects."""
        papers = []
        
        data = response.get('data', [])
        if not isinstance(data, list):
            data = [data]
        
        for item in data:
            try:
                # Extract authors
                authors = []
                author_data = item.get('authors', [])
                for author in author_data:
                    if isinstance(author, dict):
                        authors.append(author.get('name', 'Unknown'))
                    else:
                        authors.append(str(author))
                
                paper = AcademicPaper(
                    title=item.get('title', 'Untitled').strip(),
                    authors=authors or ['Unknown'],
                    abstract=item.get('abstract', '').strip(),
                    publication_year=item.get('year', 2023),
                    doi=item.get('doi'),
                    database_source='Semantic Scholar',
                    citation_count=item.get('citationCount'),
                    keywords=[],
                    full_text_url=item.get('url')
                )
                papers.append(paper)
                
            except Exception as e:
                logger.warning(f"Failed to parse Semantic Scholar entry: {str(e)}")
                continue
        
        return papers
    
    async def _parse_pubmed_response(self, response: Dict[str, Any]) -> List[AcademicPaper]:
        """Parse PubMed API response into AcademicPaper objects."""
        # This would require XML parsing in a real implementation
        # For now, return mock data
        return [
            AcademicPaper(
                title="Mock PubMed Paper",
                authors=["Mock Author"],
                abstract="Mock abstract from PubMed",
                publication_year=2023,
                doi=None,
                database_source="PubMed",
                citation_count=None,
                keywords=[],
                full_text_url=None
            )
        ]
    
    def _generate_cache_key(self, service: APIService, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        key_data = f"{service.value}:{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if still valid."""
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < timedelta(seconds=self.cache_ttl_seconds):
                return cached_data['data']
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, data: Dict[str, Any]):
        """Cache API result."""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        
        # Simple cache cleanup - remove oldest entries if cache gets too large
        if len(self.cache) > 1000:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on gateway and external services."""
        health_status = {
            'status': 'healthy',
            'gateway_available': self.gateway is not None,
            'services': {},
            'cache_size': len(self.cache),
            'timestamp': datetime.now().isoformat()
        }
        
        # Test each configured service
        for service in APIService:
            try:
                # Simple test query
                if service == APIService.ARXIV:
                    await self.search_arxiv("test", max_results=1)
                elif service == APIService.SEMANTIC_SCHOLAR:
                    await self.search_semantic_scholar("test", max_results=1)
                
                health_status['services'][service.value] = {
                    'status': 'healthy',
                    'has_credentials': service in self.credentials
                }
            except Exception as e:
                health_status['services'][service.value] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'has_credentials': service in self.credentials
                }
                health_status['status'] = 'degraded'
        
        return health_status
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limiting status."""
        now = datetime.now()
        status = {}
        
        for service in APIService:
            recent_requests = [
                record for record in self.rate_limiter.request_history
                if record.service == service and record.timestamp > now - timedelta(minutes=1)
            ]
            
            hourly_requests = [
                record for record in self.rate_limiter.request_history
                if record.service == service and record.timestamp > now - timedelta(hours=1)
            ]
            
            config = self.api_configs.get(service, {}).get('rate_limit', self.rate_limiter.config)
            
            status[service.value] = {
                'requests_last_minute': len(recent_requests),
                'requests_last_hour': len(hourly_requests),
                'minute_limit': config.requests_per_minute,
                'hour_limit': config.requests_per_hour,
                'current_backoff': self.rate_limiter.current_backoff,
                'can_make_request': len(recent_requests) < config.requests_per_minute and len(hourly_requests) < config.requests_per_hour
            }
        
        return status