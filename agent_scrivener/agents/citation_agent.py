"""
Citation Agent for tracking sources and managing bibliographic references.
"""

import re
import uuid
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import asyncio
import aiohttp
from urllib.parse import urlparse

from ..agents.base import BaseAgent, AgentResult
from ..models.core import Citation, Source, SourceType
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CitationTracker:
    """Tracks citation provenance and manages citation relationships."""
    
    def __init__(self):
        self.citations: Dict[str, Citation] = {}
        self.source_citations: Dict[str, List[str]] = {}  # source_url -> citation_ids
        self.content_citations: Dict[str, List[str]] = {}  # content_hash -> citation_ids
    
    def add_citation(self, citation: Citation) -> None:
        """Add a citation to the tracker."""
        self.citations[citation.citation_id] = citation
        
        # Track by source URL
        source_url = str(citation.source.url)
        if source_url not in self.source_citations:
            self.source_citations[source_url] = []
        self.source_citations[source_url].append(citation.citation_id)
        
        # Track by content hash (simple hash of quote/context)
        content_key = self._generate_content_key(citation.quote, citation.context)
        if content_key not in self.content_citations:
            self.content_citations[content_key] = []
        self.content_citations[content_key].append(citation.citation_id)
    
    def get_citations_for_source(self, source_url: str) -> List[Citation]:
        """Get all citations for a specific source."""
        citation_ids = self.source_citations.get(source_url, [])
        return [self.citations[cid] for cid in citation_ids if cid in self.citations]
    
    def get_citation_by_id(self, citation_id: str) -> Optional[Citation]:
        """Get a citation by its ID."""
        return self.citations.get(citation_id)
    
    def _generate_content_key(self, quote: Optional[str], context: Optional[str]) -> str:
        """Generate a key for content-based citation tracking."""
        content = f"{quote or ''}{context or ''}"
        return str(hash(content))


class APACitationFormatter:
    """Formats citations in APA style."""
    
    @staticmethod
    def format_web_source(source: Source) -> str:
        """Format a web source in APA style."""
        author = source.author or "Unknown Author"
        title = source.title
        url = str(source.url)
        
        # Format publication date
        if source.publication_date:
            date_str = source.publication_date.strftime("%Y, %B %d")
        else:
            date_str = "n.d."
        
        # Basic APA web citation format
        return f"{author}. ({date_str}). {title}. Retrieved from {url}"
    
    @staticmethod
    def format_academic_source(source: Source) -> str:
        """Format an academic source in APA style."""
        author = source.author or "Unknown Author"
        title = source.title
        
        # Extract journal/publication info from metadata
        metadata = source.metadata or {}
        journal = metadata.get('journal', 'Unknown Journal')
        volume = metadata.get('volume', '')
        issue = metadata.get('issue', '')
        pages = metadata.get('pages', '')
        doi = metadata.get('doi', '')
        
        # Format publication date
        if source.publication_date:
            year = source.publication_date.year
        else:
            year = "n.d."
        
        # Build citation string
        citation = f"{author}. ({year}). {title}. {journal}"
        
        if volume:
            citation += f", {volume}"
            if issue:
                citation += f"({issue})"
        
        if pages:
            citation += f", {pages}"
        
        if doi:
            citation += f". https://doi.org/{doi}"
        elif source.url:
            citation += f". Retrieved from {source.url}"
        
        return citation
    
    @staticmethod
    def format_database_source(source: Source) -> str:
        """Format a database source in APA style."""
        author = source.author or "Unknown Author"
        title = source.title
        
        metadata = source.metadata or {}
        database = metadata.get('database', 'Unknown Database')
        
        # Format publication date
        if source.publication_date:
            date_str = source.publication_date.strftime("%Y, %B %d")
        else:
            date_str = "n.d."
        
        return f"{author}. ({date_str}). {title}. {database}. Retrieved from {source.url}"
    
    @classmethod
    def format_citation(cls, source: Source) -> str:
        """Format a citation based on source type."""
        if source.source_type == SourceType.ACADEMIC:
            return cls.format_academic_source(source)
        elif source.source_type == SourceType.DATABASE:
            return cls.format_database_source(source)
        else:  # WEB or API
            return cls.format_web_source(source)


class URLValidator:
    """Validates URLs and checks accessibility."""
    
    def __init__(self, timeout: int = 10, max_concurrent: int = 5):
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def validate_url(self, url: str) -> Dict[str, Any]:
        """
        Validate a single URL.
        
        Returns:
            Dict with validation results including status, accessible, and error info
        """
        async with self.semaphore:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.head(url, allow_redirects=True) as response:
                        return {
                            'url': url,
                            'accessible': True,
                            'status_code': response.status,
                            'final_url': str(response.url),
                            'content_type': response.headers.get('content-type', ''),
                            'error': None
                        }
            except asyncio.TimeoutError:
                return {
                    'url': url,
                    'accessible': False,
                    'status_code': None,
                    'final_url': None,
                    'content_type': None,
                    'error': 'Timeout'
                }
            except Exception as e:
                return {
                    'url': url,
                    'accessible': False,
                    'status_code': None,
                    'final_url': None,
                    'content_type': None,
                    'error': str(e)
                }
    
    async def validate_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Validate multiple URLs concurrently."""
        tasks = [self.validate_url(url) for url in urls]
        return await asyncio.gather(*tasks)


class CitationAgent(BaseAgent):
    """
    Agent responsible for tracking sources, managing citations, and ensuring
    academic rigor through proper attribution and bibliography generation.
    """
    
    def __init__(self, memory=None):
        super().__init__("citation_agent")
        self.memory = memory  # AgentCore Memory (placeholder for now)
        self.citation_tracker = CitationTracker()
        self.formatter = APACitationFormatter()
        self.url_validator = URLValidator()
        self.tracked_sources: Set[str] = set()
        
        # MCP client for enhanced capabilities
        self.mcp_client = None
    
    async def execute(self, **kwargs) -> AgentResult:
        """Execute citation management tasks."""
        return await self._execute_with_timing(self._execute_internal, **kwargs)
    
    async def _execute_internal(self, **kwargs) -> Dict[str, Any]:
        """Internal execution logic for citation management."""
        action = kwargs.get('action', 'track_sources')
        
        if action == 'track_sources':
            return await self._track_sources_action(**kwargs)
        elif action == 'generate_bibliography':
            return await self._generate_bibliography_action(**kwargs)
        elif action == 'validate_citations':
            return await self._validate_citations_action(**kwargs)
        elif action == 'format_in_text_citations':
            return await self._format_in_text_citations_action(**kwargs)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def track_sources(self, content: str, source: Source, quote: Optional[str] = None, 
                          context: Optional[str] = None) -> Citation:
        """
        Track the provenance of information from a source.
        
        Args:
            content: The content that references the source
            source: The source being referenced
            quote: Specific quote from the source (optional)
            context: Context around the quote (optional)
            
        Returns:
            Citation: Created citation object
        """
        citation_id = str(uuid.uuid4())
        
        # Generate citation text in APA format
        citation_text = self.formatter.format_citation(source)
        
        citation = Citation(
            citation_id=citation_id,
            source=source,
            citation_text=citation_text,
            quote=quote,
            context=context
        )
        
        self.citation_tracker.add_citation(citation)
        self.tracked_sources.add(str(source.url))
        
        logger.info(f"Tracked citation for source: {source.title}")
        return citation
    
    async def generate_bibliography(self, citations: List[Citation]) -> str:
        """
        Generate a formatted bibliography from citations.
        
        Args:
            citations: List of citations to include
            
        Returns:
            str: Formatted bibliography in APA style
        """
        if not citations:
            return "## References\n\nNo references found."
        
        # Sort citations alphabetically by author/title
        sorted_citations = sorted(citations, key=lambda c: c.citation_text.lower())
        
        bibliography = "## References\n\n"
        for citation in sorted_citations:
            bibliography += f"{citation.citation_text}\n\n"
        
        return bibliography.strip()
    
    async def validate_citations(self, citations: List[Citation]) -> Dict[str, Any]:
        """
        Validate that all citation URLs are accessible and metadata is accurate.
        
        Args:
            citations: List of citations to validate
            
        Returns:
            Dict: Validation results with accessible/inaccessible URLs
        """
        urls = [str(citation.source.url) for citation in citations]
        validation_results = await self.url_validator.validate_urls(urls)
        
        accessible_citations = []
        inaccessible_citations = []
        
        for citation, result in zip(citations, validation_results):
            if result['accessible']:
                accessible_citations.append({
                    'citation_id': citation.citation_id,
                    'title': citation.source.title,
                    'url': result['url'],
                    'status_code': result['status_code']
                })
            else:
                inaccessible_citations.append({
                    'citation_id': citation.citation_id,
                    'title': citation.source.title,
                    'url': result['url'],
                    'error': result['error']
                })
        
        return {
            'total_citations': len(citations),
            'accessible_count': len(accessible_citations),
            'inaccessible_count': len(inaccessible_citations),
            'accessible_citations': accessible_citations,
            'inaccessible_citations': inaccessible_citations,
            'validation_timestamp': datetime.now().isoformat()
        }
    
    async def format_in_text_citations(self, content: str, citations: List[Citation]) -> str:
        """
        Format in-text citations within content.
        
        Args:
            content: The content to add citations to
            citations: List of citations to reference
            
        Returns:
            str: Content with properly formatted in-text citations
        """
        # Create a mapping of source titles to citation info
        citation_map = {}
        for citation in citations:
            source_title = citation.source.title.lower()
            author = citation.source.author or "Unknown Author"
            year = citation.source.publication_date.year if citation.source.publication_date else "n.d."
            
            # Extract first author's last name for in-text citation
            if author != "Unknown Author":
                # Handle "Last, First" format
                if ',' in author:
                    last_name = author.split(',')[0].strip()
                else:
                    # Handle "First Last" format - take the last word as surname
                    name_parts = author.strip().split()
                    last_name = name_parts[-1] if name_parts else author
            else:
                last_name = "Unknown Author"
            
            citation_map[source_title] = f"({last_name}, {year})"
        
        # Simple pattern matching for adding citations
        # This is a basic implementation - in practice, you'd want more sophisticated NLP
        formatted_content = content
        
        for title, in_text_citation in citation_map.items():
            # Look for mentions of the source title and add citation
            title_pattern = re.compile(re.escape(title), re.IGNORECASE)
            formatted_content = title_pattern.sub(
                lambda m: f"{m.group()} {in_text_citation}",
                formatted_content
            )
        
        return formatted_content
    
    async def get_citations_for_sources(self, sources: List[Source]) -> List[Citation]:
        """Get all citations for a list of sources."""
        citations = []
        for source in sources:
            source_citations = self.citation_tracker.get_citations_for_source(str(source.url))
            citations.extend(source_citations)
        return citations
    
    async def _track_sources_action(self, **kwargs) -> Dict[str, Any]:
        """Handle track_sources action."""
        self.validate_input(kwargs, ['sources'])
        
        sources = kwargs['sources']
        content = kwargs.get('content', '')
        
        tracked_citations = []
        for source_data in sources:
            if isinstance(source_data, dict):
                source = Source(**source_data)
            else:
                source = source_data
            
            citation = await self.track_sources(content, source)
            tracked_citations.append(citation.model_dump())
        
        return {
            'action': 'track_sources',
            'tracked_count': len(tracked_citations),
            'citations': tracked_citations
        }
    
    async def _generate_bibliography_action(self, **kwargs) -> Dict[str, Any]:
        """Handle generate_bibliography action."""
        citations_data = kwargs.get('citations', [])
        
        citations = []
        for citation_data in citations_data:
            if isinstance(citation_data, dict):
                citation = Citation(**citation_data)
            else:
                citation = citation_data
            citations.append(citation)
        
        bibliography = await self.generate_bibliography(citations)
        
        return {
            'action': 'generate_bibliography',
            'bibliography': bibliography,
            'citation_count': len(citations)
        }
    
    async def _validate_citations_action(self, **kwargs) -> Dict[str, Any]:
        """Handle validate_citations action."""
        citations_data = kwargs.get('citations', [])
        
        citations = []
        for citation_data in citations_data:
            if isinstance(citation_data, dict):
                citation = Citation(**citation_data)
            else:
                citation = citation_data
            citations.append(citation)
        
        validation_results = await self.validate_citations(citations)
        
        return {
            'action': 'validate_citations',
            **validation_results
        }
    
    async def _format_in_text_citations_action(self, **kwargs) -> Dict[str, Any]:
        """Handle format_in_text_citations action."""
        self.validate_input(kwargs, ['content', 'citations'])
        
        content = kwargs['content']
        citations_data = kwargs['citations']
        
        citations = []
        for citation_data in citations_data:
            if isinstance(citation_data, dict):
                citation = Citation(**citation_data)
            else:
                citation = citation_data
            citations.append(citation)
        
        formatted_content = await self.format_in_text_citations(content, citations)
        
        return {
            'action': 'format_in_text_citations',
            'formatted_content': formatted_content,
            'original_length': len(content),
            'formatted_length': len(formatted_content)
        }
    
    async def format_citations_with_mcp(self, sources: List[Source]) -> List[str]:
        """
        Professional citation formatting using MCP server.
        
        Args:
            sources: List of Source objects to format
            
        Returns:
            List of formatted citation strings
        """
        if not self.mcp_client:
            # Fall back to standard formatting if MCP not available
            return [self.formatter.format_citation(source) for source in sources]
        
        formatted_citations = []
        
        for source in sources:
            try:
                # Use MCP server for professional citation formatting
                citation_result = await self.mcp_client.call_tool(
                    "citation-formatter",
                    "format_citation",
                    {
                        "source_type": source.source_type.value if hasattr(source.source_type, 'value') else str(source.source_type),
                        "title": source.title,
                        "authors": getattr(source, 'authors', []) or [],
                        "publication_date": source.publication_date.isoformat() if source.publication_date else None,
                        "url": str(source.url),
                        "doi": getattr(source, 'doi', None),
                        "style": "APA"
                    }
                )
                
                formatted_citations.append(citation_result["formatted_citation"])
                
            except Exception as e:
                logger.warning(f"MCP citation formatting failed for {source.url}, falling back to standard formatting: {str(e)}")
                # Fall back to standard formatting
                formatted_citations.append(self.formatter.format_citation(source))
        
        logger.info(f"Formatted {len(formatted_citations)} citations using MCP server")
        return formatted_citations
    
    async def validate_doi_with_mcp(self, doi: str) -> Dict[str, Any]:
        """
        Validate DOI using MCP server.
        
        Args:
            doi: DOI string to validate
            
        Returns:
            Validation results with metadata
        """
        if not self.mcp_client:
            # Fall back to basic validation if MCP not available
            return {"valid": bool(re.match(r'^10\.\d{4,}/[^\s]+$', doi)), "metadata": {}}
        
        try:
            # Use MCP server for DOI validation
            validation_result = await self.mcp_client.call_tool(
                "citation-formatter",
                "validate_doi",
                {"doi": doi}
            )
            
            return validation_result
            
        except Exception as e:
            logger.warning(f"MCP DOI validation failed for {doi}, falling back to basic validation: {str(e)}")
            # Fall back to basic validation
            return {"valid": bool(re.match(r'^10\.\d{4,}/[^\s]+$', doi)), "metadata": {}}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on citation agent.
        
        Returns:
            Health check results
        """
        try:
            # Test basic functionality
            test_source = Source(
                url="https://example.com/test",
                title="Test Article",
                source_type=SourceType.WEB
            )
            
            test_citation = await self.track_sources("Test content", test_source)
            
            # Test MCP integration if available
            mcp_status = "not_configured"
            if self.mcp_client:
                try:
                    await self.mcp_client.call_tool("citation-formatter", "health_check", {})
                    mcp_status = "healthy"
                except Exception:
                    mcp_status = "unhealthy"
            
            return {
                'status': 'healthy',
                'agent_name': self.name,
                'mcp_status': mcp_status,
                'test_citation_successful': test_citation is not None,
                'tracked_sources_count': len(self.tracked_sources),
                'citations_count': len(self.citation_tracker.citations),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'agent_name': self.name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }