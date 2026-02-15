# Custom MCP Servers for Agent Scrivener

This directory contains custom MCP servers that provide research and citation tools for the Agent Scrivener project.

## Servers

### 1. Web Research Server (`web_research_server.py`)

Provides tools for web search and content extraction.

**Tools:**
- `search_web` - Search the web using DuckDuckGo
- `extract_content` - Extract main content from a web page
- `get_page_metadata` - Extract metadata (title, description, author, etc.)

**Features:**
- No API keys required (uses DuckDuckGo HTML search)
- Automatic content cleaning (removes scripts, styles, navigation)
- Link extraction support
- Configurable content length limits

### 2. Citation Formatter Server (`citation_formatter_server.py`)

Provides tools for formatting citations and validating DOIs.

**Tools:**
- `format_citation` - Format citations in APA, MLA, Chicago, or Harvard style
- `validate_doi` - Validate DOIs and retrieve metadata from CrossRef
- `generate_bibliography` - Generate formatted bibliographies from multiple citations

**Features:**
- Multiple citation styles (APA, MLA, Chicago, Harvard)
- In-text citation generation
- DOI validation with CrossRef API
- Automatic bibliography sorting
- Support for journals, books, websites, and conference papers

## Installation

1. Install dependencies:
```bash
pip install -r mcp_servers/requirements.txt
```

2. Make the servers executable:
```bash
chmod +x mcp_servers/web_research_server.py
chmod +x mcp_servers/citation_formatter_server.py
```

## Testing

Test each server individually:

```bash
# Test web research server
python mcp_servers/web_research_server.py

# Test citation formatter server
python mcp_servers/citation_formatter_server.py
```

## Configuration

The servers are configured in `.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "web-research": {
      "command": "python",
      "args": ["mcp_servers/web_research_server.py"],
      "disabled": false,
      "autoApprove": ["search_web", "extract_content", "get_page_metadata"]
    },
    "citation-formatter": {
      "command": "python",
      "args": ["mcp_servers/citation_formatter_server.py"],
      "disabled": false,
      "autoApprove": ["format_citation", "validate_doi", "generate_bibliography"]
    }
  }
}
```

## Usage Examples

### Web Research

```python
# Search the web
result = await search_web(
    query="machine learning research 2024",
    num_results=10
)

# Extract content from a URL
content = await extract_content(
    url="https://example.com/article",
    extract_links=True,
    max_length=5000
)

# Get page metadata
metadata = await get_page_metadata(
    url="https://example.com/article"
)
```

### Citation Formatting

```python
# Format a citation
citation = await format_citation(
    title="Deep Learning for Natural Language Processing",
    authors=["Smith, J.", "Doe, A."],
    year=2024,
    style="apa",
    publication="Journal of AI Research",
    volume="15",
    issue="3",
    pages="123-145",
    doi="10.1234/example.doi"
)

# Validate a DOI
validation = await validate_doi(
    doi="10.1234/example.doi"
)

# Generate a bibliography
bibliography = await generate_bibliography(
    citations=[
        {
            "title": "Paper 1",
            "authors": ["Author A"],
            "year": 2024,
            "publication": "Journal A"
        },
        {
            "title": "Paper 2",
            "authors": ["Author B"],
            "year": 2023,
            "publication": "Journal B"
        }
    ],
    style="apa",
    sort_by="author"
)
```

## Development

### Adding New Tools

To add a new tool to a server:

1. Define the tool function with the `@mcp.tool()` decorator
2. Add type hints and a docstring
3. Return a dictionary with `success` and relevant data
4. Handle errors gracefully

Example:
```python
@mcp.tool()
async def my_new_tool(param: str) -> dict:
    """
    Description of what the tool does.
    
    Args:
        param: Description of parameter
    
    Returns:
        Dictionary with results
    """
    try:
        # Implementation
        return {
            "success": True,
            "result": "data"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

### Testing Changes

After making changes:

1. Test the server directly:
```bash
python mcp_servers/web_research_server.py
```

2. Restart Kiro to reload the MCP servers

3. Check the MCP logs for connection status

## Troubleshooting

### Server Won't Start

- Check that all dependencies are installed: `pip install -r mcp_servers/requirements.txt`
- Verify Python version is 3.10 or higher: `python --version`
- Check for syntax errors: `python -m py_compile mcp_servers/web_research_server.py`

### Connection Errors

- Ensure the paths in `mcp.json` are correct (relative to workspace root)
- Check MCP logs in Kiro for detailed error messages
- Verify the server is executable: `chmod +x mcp_servers/*.py`

### Tool Not Working

- Check the server logs for error messages
- Verify network connectivity for web-based tools
- Test the tool with simple inputs first

## License

These servers are part of the Agent Scrivener project and follow the same license.
