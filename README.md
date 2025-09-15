# Agent Scrivener

Autonomous Research and Content Synthesis Platform

## Overview

Agent Scrivener is a cloud-native, serverless multi-agent system that transforms research queries into comprehensive, structured, and fully-cited research documents. The system orchestrates multiple specialized AI agents to automate the entire lifecycle of knowledge work, from information discovery and analysis to content synthesis and citation management.

## Features

- **Multi-Agent Orchestration**: Coordinated workflow of specialized agents
- **Web Research**: Autonomous browsing and content extraction
- **Academic Database Integration**: Access to arXiv, PubMed, and Semantic Scholar
- **Content Analysis**: NLP-powered insight generation and topic modeling
- **Document Synthesis**: Professional research document generation
- **Citation Management**: Automatic source tracking and bibliography generation
- **AWS AgentCore Integration**: Serverless deployment on AWS Bedrock

## Architecture

The system consists of specialized agents:

- **Planner Agent**: Query analysis and task orchestration
- **Research Agent**: Web search and content extraction
- **API Agent**: Academic database queries
- **Analysis Agent**: Data processing and insight generation
- **Drafting Agent**: Content synthesis and formatting
- **Citation Agent**: Source tracking and bibliography management

## Installation

### Prerequisites

- Python 3.9 or higher
- AWS account with Bedrock access
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/agent-scrivener/agent-scrivener.git
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
uvicorn agent_scrivener.api:app --host 0.0.0.0 --port 8000
```

Submit a research request:

```bash
curl -X POST "http://localhost:8000/research" \
  -H "Content-Type: application/json" \
  -d '{"query": "Latest developments in quantum computing"}'
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
├── agents/          # Agent implementations
├── models/          # Pydantic data models
├── tools/           # AgentCore tool wrappers
├── utils/           # Common utilities
├── deployment/      # AWS deployment configs
└── api/             # FastAPI application
```

## Deployment

### AWS AgentCore Runtime

The system is designed for deployment on AWS Bedrock AgentCore Runtime. See the `deployment/` directory for configuration templates and deployment scripts.

### Docker

Build and run with Docker:

```bash
docker build -t agent-scrivener .
docker run -p 8000:8000 agent-scrivener
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support, please open an issue on GitHub or contact the development team.