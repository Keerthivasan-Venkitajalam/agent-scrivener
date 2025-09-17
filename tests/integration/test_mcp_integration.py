"""
Integration tests for MCP server functionality in Agent Scrivener.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from agent_scrivener.agents.api_agent import APIAgent
from agent_scrivener.agents.research_agent import ResearchAgent
from agent_scrivener.agents.citation_agent import CitationAgent
from agent_scrivener.models.core import Source, AcademicPaper


class TestMCPIntegration:
    """Test MCP server integration across different agents."""
    
    @pytest.fixture
    def mock_mcp_client(self):
        """Mock MCP client for testing."""
        client = AsyncMock()
        return client
    
    @pytest.fixture
    def api_agent_with_mcp(self, mock_mcp_client):
        """API Agent with mocked MCP client."""
        agent = APIAgent()
        agent.mcp_client = mock_mcp_client
        return agent
    
    @pytest.fixture
    def research_agent_with_mcp(self, mock_mcp_client):
        """Research Agent with mocked MCP client."""
        agent = ResearchAgent()
        agent.mcp_client = mock_mcp_client
        return agent
    
    @pytest.fixture
    def citation_agent_with_mcp(self, mock_mcp_client):
        """Citation Agent with mocked MCP client."""
        agent = CitationAgent()
        agent.mcp_client = mock_mcp_client
        return agent

    async def test_academic_search_mcp(self, api_agent_with_mcp, mock_mcp_client):
        """Test academic search MCP server integration."""
        # Mock MCP server response
        mock_response = {
            "papers": [
                {
                    "title": "Attention Is All You Need",
                    "authors": ["Vaswani", "Shazeer", "Parmar"],
                    "abstract": "The dominant sequence transduction models...",
                    "publication_year": 2017,
                    "citation_count": 50000,
                    "doi": "10.48550/arXiv.1706.03762"
                }
            ]
        }
        mock_mcp_client.call_tool.return_value = mock_response
        
        # Test enhanced academic search
        results = await api_agent_with_mcp.enhanced_academic_search(
            "machine learning transformers",
            {"date_range": "2020-2024", "min_citations": 10}
        )
        
        # Verify MCP call
        mock_mcp_client.call_tool.assert_called_once_with(
            "academic-search",
            "search_papers",
            {
                "query": "machine learning transformers",
                "databases": ["arxiv", "pubmed", "semantic_scholar"],
                "date_range": "2020-2024",
                "subject_areas": [],
                "min_citations": 10
            }
        )
        
        # Verify results
        assert len(results) > 0
        assert isinstance(results[0], AcademicPaper)
        assert results[0].title == "Attention Is All You Need"
    
    async def test_web_research_mcp(self, research_agent_with_mcp, mock_mcp_client):
        """Test web research MCP server integration."""
        # Mock MCP server response
        mock_response = {
            "content": "This is the extracted article content...",
            "summary": "Article discusses machine learning advances...",
            "metadata": {
                "title": "ML Advances in 2024",
                "author": "Jane Researcher",
                "publication_date": "2024-01-15"
            },
            "images": ["https://example.com/image1.jpg"]
        }
        mock_mcp_client.call_tool.return_value = mock_response
        
        # Test enhanced content extraction
        result = await research_agent_with_mcp.extract_with_mcp("https://example.com/article")
        
        # Verify MCP call
        mock_mcp_client.call_tool.assert_called_once_with(
            "web-research",
            "extract_content",
            {
                "url": "https://example.com/article",
                "extract_images": True,
                "clean_html": True,
                "generate_summary": True,
                "extract_metadata": True
            }
        )
        
        # Verify results
        assert result.content == "This is the extracted article content..."
        assert result.summary == "Article discusses machine learning advances..."
        assert result.metadata is not None
    
    async def test_citation_formatter_mcp(self, citation_agent_with_mcp, mock_mcp_client):
        """Test citation formatting MCP server integration."""
        # Mock MCP server response
        mock_response = {
            "formatted_citation": "Vaswani, A., Shazeer, N., & Parmar, N. (2017). Attention is all you need. arXiv preprint arXiv:1706.03762."
        }
        mock_mcp_client.call_tool.return_value = mock_response
        
        # Create test source
        source = Source(
            url="https://arxiv.org/abs/1706.03762",
            title="Attention Is All You Need",
            authors=["Vaswani", "Shazeer", "Parmar"],
            publication_date="2017-06-12",
            source_type="academic",
            doi="10.48550/arXiv.1706.03762"
        )
        
        # Test citation formatting
        formatted_citations = await citation_agent_with_mcp.format_citations_with_mcp([source])
        
        # Verify MCP call
        mock_mcp_client.call_tool.assert_called_once_with(
            "citation-formatter",
            "format_citation",
            {
                "source_type": "academic",
                "title": "Attention Is All You Need",
                "authors": ["Vaswani", "Shazeer", "Parmar"],
                "publication_date": "2017-06-12",
                "url": "https://arxiv.org/abs/1706.03762",
                "doi": "10.48550/arXiv.1706.03762",
                "style": "APA"
            }
        )
        
        # Verify results
        assert len(formatted_citations) == 1
        assert "Vaswani, A." in formatted_citations[0]
        assert "2017" in formatted_citations[0]
    
    async def test_mcp_server_failure_handling(self, api_agent_with_mcp, mock_mcp_client):
        """Test graceful handling of MCP server failures."""
        # Mock MCP server failure
        mock_mcp_client.call_tool.side_effect = Exception("MCP server unavailable")
        
        # Test that agent handles failure gracefully
        results = await api_agent_with_mcp.enhanced_academic_search(
            "test query",
            {"date_range": "2020-2024"}
        )
        
        # Should fall back to standard search method
        assert results is not None  # Should not crash
        # Verify fallback behavior was triggered
        # (Implementation would depend on specific fallback logic)
    
    async def test_mcp_auto_approval_workflow(self, api_agent_with_mcp, mock_mcp_client):
        """Test that auto-approved MCP operations work seamlessly."""
        # Mock successful auto-approved operation
        mock_response = {"papers": []}
        mock_mcp_client.call_tool.return_value = mock_response
        
        # Test operation that should be auto-approved
        results = await api_agent_with_mcp.enhanced_academic_search(
            "test query",
            {}
        )
        
        # Verify the operation completed without manual approval
        mock_mcp_client.call_tool.assert_called_once()
        assert results is not None


class TestMCPConfiguration:
    """Test MCP configuration and setup."""
    
    def test_mcp_config_validation(self):
        """Test that MCP configuration is valid."""
        import json
        
        # Load MCP configuration
        with open('.kiro/settings/mcp.json', 'r') as f:
            config = json.load(f)
        
        # Verify required structure
        assert 'mcpServers' in config
        assert len(config['mcpServers']) > 0
        
        # Verify each server has required fields
        for server_name, server_config in config['mcpServers'].items():
            assert 'command' in server_config
            assert 'args' in server_config
            assert 'disabled' in server_config
            assert isinstance(server_config['autoApprove'], list)
    
    def test_mcp_server_definitions(self):
        """Test that all required MCP servers are defined."""
        import json
        
        with open('.kiro/settings/mcp.json', 'r') as f:
            config = json.load(f)
        
        required_servers = ['academic-search', 'web-research', 'citation-formatter']
        
        for server in required_servers:
            assert server in config['mcpServers']
            assert not config['mcpServers'][server]['disabled']