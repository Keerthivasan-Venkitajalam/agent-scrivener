#!/usr/bin/env python3
"""
Web Research MCP Server

Provides tools for web search and content extraction for research purposes.
"""

from fastmcp import FastMCP
import httpx
from bs4 import BeautifulSoup
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("web-research")


@mcp.tool()
async def search_web(
    query: str,
    num_results: int = 10,
    search_engine: str = "duckduckgo"
) -> dict:
    """
    Search the web for information using DuckDuckGo.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 10)
        search_engine: Search engine to use (default: duckduckgo)
    
    Returns:
        Dictionary with search results including titles, URLs, and snippets
    """
    try:
        # Use DuckDuckGo HTML search (no API key required)
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                }
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Search failed with status {response.status_code}",
                    "results": []
                }
            
            # Parse HTML results
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for result in soup.find_all('div', class_='result')[:num_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if title_elem:
                    results.append({
                        "title": title_elem.get_text(strip=True),
                        "url": title_elem.get('href', ''),
                        "snippet": snippet_elem.get_text(strip=True) if snippet_elem else ""
                    })
            
            logger.info(f"Found {len(results)} results for query: {query}")
            
            return {
                "success": True,
                "query": query,
                "num_results": len(results),
                "results": results
            }
            
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


@mcp.tool()
async def extract_content(
    url: str,
    extract_links: bool = False,
    max_length: Optional[int] = None
) -> dict:
    """
    Extract main content from a web page.
    
    Args:
        url: The URL to extract content from
        extract_links: Whether to extract links from the page (default: False)
        max_length: Maximum content length in characters (default: None)
    
    Returns:
        Dictionary with extracted content, title, and metadata
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                }
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Failed to fetch URL with status {response.status_code}",
                    "url": url
                }
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else "No title"
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract main content
            # Try to find main content area
            main_content = (
                soup.find('main') or 
                soup.find('article') or 
                soup.find('div', class_=['content', 'main-content', 'article-content']) or
                soup.find('body')
            )
            
            if main_content:
                content = main_content.get_text(separator='\n', strip=True)
            else:
                content = soup.get_text(separator='\n', strip=True)
            
            # Truncate if needed
            if max_length and len(content) > max_length:
                content = content[:max_length] + "..."
            
            result = {
                "success": True,
                "url": url,
                "title": title_text,
                "content": content,
                "content_length": len(content)
            }
            
            # Extract links if requested
            if extract_links:
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.startswith('http'):
                        links.append({
                            "text": link.get_text(strip=True),
                            "url": href
                        })
                result["links"] = links[:50]  # Limit to 50 links
            
            logger.info(f"Extracted {len(content)} characters from {url}")
            
            return result
            
    except Exception as e:
        logger.error(f"Content extraction error: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "url": url
        }


@mcp.tool()
async def get_page_metadata(url: str) -> dict:
    """
    Extract metadata from a web page (title, description, author, etc.).
    
    Args:
        url: The URL to extract metadata from
    
    Returns:
        Dictionary with page metadata
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                }
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Failed to fetch URL with status {response.status_code}",
                    "url": url
                }
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            metadata = {
                "success": True,
                "url": url,
                "title": None,
                "description": None,
                "author": None,
                "keywords": None,
                "og_title": None,
                "og_description": None,
                "og_image": None
            }
            
            # Extract title
            title = soup.find('title')
            if title:
                metadata["title"] = title.get_text(strip=True)
            
            # Extract meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name', '').lower()
                property_name = meta.get('property', '').lower()
                content = meta.get('content', '')
                
                if name == 'description':
                    metadata["description"] = content
                elif name == 'author':
                    metadata["author"] = content
                elif name == 'keywords':
                    metadata["keywords"] = content
                elif property_name == 'og:title':
                    metadata["og_title"] = content
                elif property_name == 'og:description':
                    metadata["og_description"] = content
                elif property_name == 'og:image':
                    metadata["og_image"] = content
            
            logger.info(f"Extracted metadata from {url}")
            
            return metadata
            
    except Exception as e:
        logger.error(f"Metadata extraction error: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "url": url
        }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
