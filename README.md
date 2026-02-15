# Agent Scrivener

Autonomous Research and Content Synthesis Platform

## Overview

Agent Scrivener is a cloud-native, serverless multi-agent system that transforms research queries into comprehensive, structured, and fully-cited research documents. The system orchestrates multiple specialized AI agents to automate the entire lifecycle of knowledge work, from information discovery and analysis to content synthesis and citation management.

Built with extensibility in mind, Agent Scrivener integrates seamlessly with modern agentic IDEs through the Model Context Protocol (MCP), enabling developers to leverage powerful research capabilities directly within their development environment.

## Key Features

- **Multi-Agent Orchestration**: Coordinated workflow of specialized agents for complex research tasks
- **Web Research**: Autonomous browsing and content extraction from diverse sources
- **Academic Database Integration**: Direct access to arXiv, PubMed, Semantic Scholar, Google Scholar, CORE, OpenAlex, and more
- **Content Analysis**: NLP-powered insight generation and topic modeling
- **Document Synthesis**: Professional research document generation with proper formatting
- **Citation Management**: Automatic source tracking and bibliography generation in multiple formats (APA, MLA, Chicago, Harvard)
- **MCP Server Integration**: Native support for Model Context Protocol enabling IDE integration
- **AWS AgentCore Integration**: Serverless deployment on AWS Bedrock for production workloads
- **Production Validation Framework**: Comprehensive validation suite for deployment readiness

## Architecture

### Core Agent System

The system consists of specialized agents working in concert:

- **Planner Agent**: Query analysis and task orchestration
- **Research Agent**: Web search and content extraction
- **API Agent**: Academic database queries
- **Analysis Agent**: Data processing and insight generation
- **Drafting Agent**: Content synthesis and formatting
- **Citation Agent**: Source tracking and bibliography management

### MCP Server Integration

Agent Scrivener provides three MCP servers for seamless integration with agentic IDEs:

- **Citation Formatter Server**: Format citations in multiple styles, validate DOIs, and generate bibliographies
- **Web Research Server**: Search the web, extract content, and retrieve page metadata
- **Academic Search Server**: Query 10+ academic databases including arXiv, PubMed, Google Scholar, and OpenAlex

These servers enable any MCP-compatible IDE to access Agent Scrivener's research capabilities through a standardized protocol.

## Installation

### Prerequisites

- Python 3.9 or higher
- AWS account with Bedrock access (for production deployment)
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Keerthivasan-Venkitajalam/agent-scrivener.git
cd agent-scrivener
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

### MCP Server Installation

For IDE integration, install the MCP server dependencies:

```bash
cd mcp_servers
pip install -r requirements.txt
```

## Configuration

Configuration can be provided through environment variables or a `config.json` file:

### Environment Variables

```bash
export AWS_REGION=us-east-1
export DEBUG=false
export LOG_LEVEL=INFO
export MAX_CONCURRENT_SESSIONS=10
```

### Configuration File

Create a `config.json` file in the project root:

```json
{
  "debug": false,
  "log_level": "INFO",
  "max_concurrent_sessions": 10,
  "agentcore": {
    "region": "us-east-1",
    "timeout_seconds": 300
  },
  "processing": {
    "max_sources_per_query": 20,
    "confidence_threshold": 0.7
  }
}
```

## Integration with Kiro and Agentic IDEs

### Kiro IDE Integration

Agent Scrivener provides native integration with Kiro through MCP servers. To integrate with Kiro:

1. Configure the MCP servers in your Kiro settings file (`.kiro/settings/mcp.json`):

```json
{
  "mcpServers": {
    "citation-formatter": {
      "command": "python",
      "args": ["/path/to/agent-scrivener/mcp_servers/citation_formatter_server.py"],
      "disabled": false
    },
    "web-research": {
      "command": "python",
      "args": ["/path/to/agent-scrivener/mcp_servers/web_research_server.py"],
      "disabled": false
    },
    "academic-search": {
      "command": "uvx",
      "args": ["mcp-server-academic-search"],
      "disabled": false
    }
  }
}
```

2. Restart Kiro or reconnect the MCP servers from the MCP Server view in the Kiro feature panel.

3. The research tools will now be available in your Kiro workspace for:
   - Formatting citations in multiple styles
   - Searching academic databases
   - Extracting web content
   - Validating DOIs and retrieving paper metadata

### Generic Agentic IDE Integration

Agent Scrivener follows the Model Context Protocol (MCP) specification, making it compatible with any MCP-enabled IDE:

1. Locate your IDE's MCP configuration file
2. Add the Agent Scrivener MCP servers using the configuration format above
3. Adjust the `command` and `args` fields to match your IDE's requirements
4. Restart your IDE or reload the MCP configuration

### Available MCP Tools

Once integrated, your IDE will have access to:

**Citation Formatter Tools:**
- `format_citation`: Format citations in APA, MLA, Chicago, or Harvard style
- `validate_doi`: Validate DOIs and retrieve metadata from CrossRef
- `generate_bibliography`: Generate formatted bibliographies from multiple citations

**Web Research Tools:**
- `search_web`: Search the web using DuckDuckGo
- `extract_content`: Extract main content from web pages
- `get_page_metadata`: Retrieve metadata from URLs

**Academic Search Tools:**
- `search_arxiv`, `search_pubmed`, `search_google_scholar`: Search academic databases
- `search_openalex`, `search_core`, `search_semantic`: Access comprehensive research indexes
- `get_paper_by_doi`: Retrieve papers by DOI
- `search_authors`, `get_author_papers`: Find authors and their publications
- `get_openalex_citations`, `get_openalex_references`: Explore citation networks
- `download_paper`, `read_paper`: Download and extract text from PDFs

## Usage

### Basic Usage

```python
from agent_scrivener import AgentScrivener

# Initialize the system
scrivener = AgentScrivener()

# Submit a research query
result = await scrivener.research(
    "What are the latest developments in machine learning for healthcare?"
)

# Access the generated document
print(result.final_document)
```

### API Usage

Start the FastAPI server:

```bash
uvicorn agent_scrivener.api.main:app --host 0.0.0.0 --port 8000
```

Submit a research request:

```bash
curl -X POST "http://localhost:8000/research" \
  -H "Content-Type: application/json" \
  -d '{"query": "Latest developments in quantum computing"}'
```

### MCP Server Usage

Run MCP servers individually for testing:

```bash
# Citation formatter server
python mcp_servers/citation_formatter_server.py

# Web research server
python mcp_servers/web_research_server.py
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agent_scrivener

# Run specific test categories
pytest -m unit
pytest -m integration
```

### Code Quality

```bash
# Format code
black agent_scrivener/
isort agent_scrivener/

# Lint code
flake8 agent_scrivener/
mypy agent_scrivener/
```

### Project Structure

```
agent_scrivener/
├── agents/                    # Agent implementations
├── models/                    # Pydantic v2 data models
├── tools/                     # AgentCore tool wrappers
├── utils/                     # Common utilities
├── deployment/
│   └── validation/           # Production readiness validation framework
├── api/                      # FastAPI application
│   ├── main.py              # API entry point
│   ├── models.py            # API request/response models
│   ├── websocket.py         # WebSocket support
│   └── routes.py            # API endpoints
├── mcp_servers/              # Model Context Protocol servers
│   ├── citation_formatter_server.py
│   ├── web_research_server.py
│   └── README.md
├── tests/
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   ├── validation/          # Validation framework tests
│   └── performance/         # Performance benchmarks
└── docs/                    # Documentation
```

## Deployment

### Production Validation

Before deploying to production, run the comprehensive validation suite:

```bash
python -m agent_scrivener.deployment.validation.cli validate-all
```

This validates:
- API endpoints and response formats
- Security configurations
- Data persistence mechanisms
- AWS infrastructure setup
- Monitoring and observability
- Documentation completeness
- Performance benchmarks
- End-to-end workflows

### AWS AgentCore Runtime

The system is designed for deployment on AWS Bedrock AgentCore Runtime. See the `deployment/` directory for configuration templates and deployment scripts.

### Docker

Build and run with Docker:

```bash
docker build -t agent-scrivener .
docker run -p 8000:8000 agent-scrivener
```

### MCP Server Deployment

For production MCP server deployment, consider:

1. Running servers as system services
2. Implementing proper logging and monitoring
3. Setting up health checks
4. Configuring appropriate timeouts
5. Securing inter-process communication

## Technology Stack

- **Python 3.9+**: Core language with async/await support
- **Pydantic v2**: Data validation and settings management
- **FastAPI**: High-performance API framework
- **AWS Bedrock**: Serverless AI agent runtime
- **Model Context Protocol**: IDE integration standard
- **pytest**: Comprehensive testing framework

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with clear commit messages
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Update documentation as needed
7. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests for new features
- Update type hints and docstrings
- Run the validation suite before submitting PRs
- Keep commits atomic and well-described

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with support from AWS Bedrock AgentCore
- MCP integration follows the Model Context Protocol specification
- Academic search capabilities powered by multiple open-access databases

## Support

For questions, issues, or feature requests:

- Open an issue on [GitHub](https://github.com/Keerthivasan-Venkitajalam/agent-scrivener/issues)
- Review the documentation in the `docs/` directory
- Check the MCP server README for integration help

## Roadmap

- Enhanced citation format support
- Additional academic database integrations
- Real-time collaboration features
- Advanced NLP analysis capabilities
- Expanded MCP tool offerings