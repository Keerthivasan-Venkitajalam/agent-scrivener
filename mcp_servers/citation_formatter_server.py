#!/usr/bin/env python3
"""
Citation Formatter MCP Server

Provides tools for formatting citations in various styles (APA, MLA, Chicago, etc.)
and validating DOIs.
"""

from fastmcp import FastMCP
import httpx
from typing import Optional, Literal
import re
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("citation-formatter")


@mcp.tool()
async def format_citation(
    title: str,
    authors: list[str],
    year: int,
    style: Literal["apa", "mla", "chicago", "harvard"] = "apa",
    publication: Optional[str] = None,
    volume: Optional[str] = None,
    issue: Optional[str] = None,
    pages: Optional[str] = None,
    doi: Optional[str] = None,
    url: Optional[str] = None,
    publisher: Optional[str] = None,
    publication_type: Literal["journal", "book", "website", "conference"] = "journal"
) -> dict:
    """
    Format a citation in the specified style.
    
    Args:
        title: Title of the work
        authors: List of author names
        year: Publication year
        style: Citation style (apa, mla, chicago, harvard)
        publication: Journal/publication name
        volume: Volume number
        issue: Issue number
        pages: Page range (e.g., "123-145")
        doi: Digital Object Identifier
        url: URL of the work
        publisher: Publisher name (for books)
        publication_type: Type of publication
    
    Returns:
        Dictionary with formatted citation
    """
    try:
        citation = ""
        
        if style == "apa":
            citation = _format_apa(
                title, authors, year, publication, volume, issue, 
                pages, doi, url, publisher, publication_type
            )
        elif style == "mla":
            citation = _format_mla(
                title, authors, year, publication, volume, issue,
                pages, doi, url, publisher, publication_type
            )
        elif style == "chicago":
            citation = _format_chicago(
                title, authors, year, publication, volume, issue,
                pages, doi, url, publisher, publication_type
            )
        elif style == "harvard":
            citation = _format_harvard(
                title, authors, year, publication, volume, issue,
                pages, doi, url, publisher, publication_type
            )
        else:
            return {
                "success": False,
                "error": f"Unsupported citation style: {style}"
            }
        
        logger.info(f"Formatted citation in {style.upper()} style")
        
        return {
            "success": True,
            "style": style,
            "citation": citation,
            "in_text": _format_in_text(authors, year, style)
        }
        
    except Exception as e:
        logger.error(f"Citation formatting error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def _format_apa(title, authors, year, publication, volume, issue, pages, doi, url, publisher, pub_type):
    """Format citation in APA style."""
    # Format authors
    if len(authors) == 1:
        author_str = authors[0]
    elif len(authors) == 2:
        author_str = f"{authors[0]} & {authors[1]}"
    elif len(authors) <= 20:
        author_str = ", ".join(authors[:-1]) + f", & {authors[-1]}"
    else:
        author_str = ", ".join(authors[:19]) + ", ... " + authors[-1]
    
    citation = f"{author_str} ({year}). {title}. "
    
    if pub_type == "journal" and publication:
        citation += f"*{publication}*"
        if volume:
            citation += f", *{volume}*"
        if issue:
            citation += f"({issue})"
        if pages:
            citation += f", {pages}"
        citation += "."
    elif pub_type == "book" and publisher:
        citation += f"{publisher}."
    elif pub_type == "website":
        if publication:
            citation += f"*{publication}*."
    
    if doi:
        citation += f" https://doi.org/{doi}"
    elif url:
        citation += f" {url}"
    
    return citation


def _format_mla(title, authors, year, publication, volume, issue, pages, doi, url, publisher, pub_type):
    """Format citation in MLA style."""
    # Format authors (Last, First for first author, then First Last for others)
    if len(authors) == 1:
        author_str = authors[0]
    elif len(authors) == 2:
        author_str = f"{authors[0]}, and {authors[1]}"
    else:
        author_str = f"{authors[0]}, et al"
    
    citation = f'{author_str}. "{title}." '
    
    if pub_type == "journal" and publication:
        citation += f"*{publication}*"
        if volume:
            citation += f", vol. {volume}"
        if issue:
            citation += f", no. {issue}"
        citation += f", {year}"
        if pages:
            citation += f", pp. {pages}"
        citation += "."
    elif pub_type == "book" and publisher:
        citation += f"{publisher}, {year}."
    elif pub_type == "website":
        if publication:
            citation += f"*{publication}*, "
        citation += f"{year}."
    
    if doi:
        citation += f" doi:{doi}."
    elif url:
        citation += f" {url}."
    
    return citation


def _format_chicago(title, authors, year, publication, volume, issue, pages, doi, url, publisher, pub_type):
    """Format citation in Chicago style."""
    # Format authors
    if len(authors) == 1:
        author_str = authors[0]
    elif len(authors) == 2:
        author_str = f"{authors[0]} and {authors[1]}"
    elif len(authors) == 3:
        author_str = f"{authors[0]}, {authors[1]}, and {authors[2]}"
    else:
        author_str = f"{authors[0]} et al"
    
    citation = f'{author_str}. {year}. "{title}." '
    
    if pub_type == "journal" and publication:
        citation += f"*{publication}* "
        if volume:
            citation += f"{volume}"
        if issue:
            citation += f" ({issue})"
        if pages:
            citation += f": {pages}"
        citation += "."
    elif pub_type == "book" and publisher:
        citation += f"{publisher}."
    
    if doi:
        citation += f" https://doi.org/{doi}."
    elif url:
        citation += f" {url}."
    
    return citation


def _format_harvard(title, authors, year, publication, volume, issue, pages, doi, url, publisher, pub_type):
    """Format citation in Harvard style."""
    # Format authors
    if len(authors) == 1:
        author_str = authors[0]
    elif len(authors) == 2:
        author_str = f"{authors[0]} and {authors[1]}"
    else:
        author_str = f"{authors[0]} et al."
    
    citation = f"{author_str} ({year}) '{title}', "
    
    if pub_type == "journal" and publication:
        citation += f"*{publication}*"
        if volume:
            citation += f", vol. {volume}"
        if issue:
            citation += f", no. {issue}"
        if pages:
            citation += f", pp. {pages}"
        citation += "."
    elif pub_type == "book" and publisher:
        citation += f"{publisher}."
    
    if doi:
        citation += f" doi: {doi}."
    elif url:
        citation += f" Available at: {url}."
    
    return citation


def _format_in_text(authors, year, style):
    """Format in-text citation."""
    if style == "apa" or style == "harvard":
        if len(authors) == 1:
            return f"({authors[0].split()[-1]}, {year})"
        elif len(authors) == 2:
            return f"({authors[0].split()[-1]} & {authors[1].split()[-1]}, {year})"
        else:
            return f"({authors[0].split()[-1]} et al., {year})"
    elif style == "mla":
        if len(authors) == 1:
            return f"({authors[0].split()[-1]})"
        elif len(authors) == 2:
            return f"({authors[0].split()[-1]} and {authors[1].split()[-1]})"
        else:
            return f"({authors[0].split()[-1]} et al.)"
    elif style == "chicago":
        if len(authors) == 1:
            return f"({authors[0].split()[-1]} {year})"
        else:
            return f"({authors[0].split()[-1]} et al. {year})"
    
    return ""


@mcp.tool()
async def validate_doi(doi: str) -> dict:
    """
    Validate a DOI and retrieve metadata from CrossRef.
    
    Args:
        doi: Digital Object Identifier to validate
    
    Returns:
        Dictionary with validation result and metadata if valid
    """
    try:
        # Clean DOI
        doi = doi.strip()
        if doi.startswith('http'):
            doi = doi.split('doi.org/')[-1]
        
        # Validate DOI format
        doi_pattern = r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$'
        if not re.match(doi_pattern, doi, re.IGNORECASE):
            return {
                "success": False,
                "valid": False,
                "error": "Invalid DOI format",
                "doi": doi
            }
        
        # Query CrossRef API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"https://api.crossref.org/works/{doi}",
                headers={"User-Agent": "CitationFormatter/1.0"}
            )
            
            if response.status_code == 404:
                return {
                    "success": True,
                    "valid": False,
                    "error": "DOI not found in CrossRef database",
                    "doi": doi
                }
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "valid": False,
                    "error": f"CrossRef API error: {response.status_code}",
                    "doi": doi
                }
            
            data = response.json()
            message = data.get('message', {})
            
            # Extract metadata
            metadata = {
                "success": True,
                "valid": True,
                "doi": doi,
                "title": message.get('title', [''])[0],
                "authors": [
                    f"{author.get('given', '')} {author.get('family', '')}"
                    for author in message.get('author', [])
                ],
                "year": message.get('published-print', {}).get('date-parts', [[None]])[0][0] or
                        message.get('published-online', {}).get('date-parts', [[None]])[0][0],
                "publication": message.get('container-title', [''])[0],
                "volume": message.get('volume'),
                "issue": message.get('issue'),
                "pages": message.get('page'),
                "publisher": message.get('publisher'),
                "type": message.get('type')
            }
            
            logger.info(f"Validated DOI: {doi}")
            
            return metadata
            
    except Exception as e:
        logger.error(f"DOI validation error: {str(e)}")
        return {
            "success": False,
            "valid": False,
            "error": str(e),
            "doi": doi
        }


@mcp.tool()
async def generate_bibliography(
    citations: list[dict],
    style: Literal["apa", "mla", "chicago", "harvard"] = "apa",
    sort_by: Literal["author", "year", "title"] = "author"
) -> dict:
    """
    Generate a formatted bibliography from multiple citations.
    
    Args:
        citations: List of citation dictionaries (same format as format_citation)
        style: Citation style
        sort_by: How to sort the bibliography
    
    Returns:
        Dictionary with formatted bibliography
    """
    try:
        formatted_citations = []
        
        for citation_data in citations:
            result = await format_citation(
                title=citation_data.get('title', ''),
                authors=citation_data.get('authors', []),
                year=citation_data.get('year', datetime.now().year),
                style=style,
                publication=citation_data.get('publication'),
                volume=citation_data.get('volume'),
                issue=citation_data.get('issue'),
                pages=citation_data.get('pages'),
                doi=citation_data.get('doi'),
                url=citation_data.get('url'),
                publisher=citation_data.get('publisher'),
                publication_type=citation_data.get('publication_type', 'journal')
            )
            
            if result['success']:
                formatted_citations.append({
                    'citation': result['citation'],
                    'sort_key': citation_data.get('authors', [''])[0] if sort_by == 'author' else
                               citation_data.get('year', 0) if sort_by == 'year' else
                               citation_data.get('title', '')
                })
        
        # Sort citations
        formatted_citations.sort(key=lambda x: x['sort_key'])
        
        bibliography = "\n\n".join([c['citation'] for c in formatted_citations])
        
        logger.info(f"Generated bibliography with {len(formatted_citations)} citations")
        
        return {
            "success": True,
            "style": style,
            "count": len(formatted_citations),
            "bibliography": bibliography
        }
        
    except Exception as e:
        logger.error(f"Bibliography generation error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
