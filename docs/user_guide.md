# Agent Scrivener User Guide

## Overview

Agent Scrivener is an autonomous research and content synthesis platform that transforms research queries into comprehensive, structured, and fully-cited research documents. The system orchestrates multiple specialized AI agents to automate the entire knowledge work lifecycle.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- AWS account with Bedrock access
- AgentCore Runtime environment
- Internet connection for web research

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd agent-scrivener
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your AWS credentials and configuration
   ```

4. **Run setup script:**
   ```bash
   python scripts/setup_dev.py
   ```

### Quick Start

1. **Start the API server:**
   ```bash
   python -m agent_scrivener.api.main
   ```

2. **Submit a research query:**
   ```bash
   curl -X POST "http://localhost:8000/research" \
        -H "Content-Type: application/json" \
        -d '{"query": "What are the latest developments in quantum computing?"}'
   ```

3. **Monitor progress:**
   - Use the returned session ID to track progress
   - Connect to WebSocket endpoint for real-time updates

## Using the System

### Research Query Format

Agent Scrivener accepts natural language research queries. For best results:

- **Be specific**: Include the scope, timeframe, and focus areas
- **Provide context**: Mention the intended audience or use case
- **Set boundaries**: Specify what aspects to emphasize or exclude

#### Example Queries

**Good Query:**
```
Analyze the current applications and future potential of large language models 
in healthcare, including diagnostic assistance, medical record processing, and 
patient interaction. Examine both benefits and ethical concerns, with focus 
on recent developments in 2024.
```

**Basic Query:**
```
What are the latest developments in quantum computing hardware and algorithms 
in 2024? Focus on IBM, Google, and other major players.
```

### API Endpoints

#### Submit Research Request
```http
POST /research
Content-Type: application/json

{
  "query": "Your research question here",
  "options": {
    "max_sources": 20,
    "include_academic": true,
    "format": "markdown",
    "citation_style": "apa"
  }
}
```

**Response:**
```json
{
  "session_id": "uuid-string",
  "status": "processing",
  "estimated_completion": "2024-01-15T10:30:00Z"
}
```

#### Check Research Status
```http
GET /research/{session_id}/status
```

**Response:**
```json
{
  "session_id": "uuid-string",
  "status": "processing",
  "progress": 65,
  "current_step": "Analysis Agent processing content",
  "estimated_remaining": "3 minutes"
}
```

#### Retrieve Research Results
```http
GET /research/{session_id}/result
```

**Response:**
```json
{
  "session_id": "uuid-string",
  "status": "completed",
  "document": "# Research Title\n\n## Introduction...",
  "metadata": {
    "sources_analyzed": 15,
    "processing_time": "8.5 minutes",
    "confidence_score": 0.92
  }
}
```

### WebSocket Real-time Updates

Connect to the WebSocket endpoint for live progress updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/{session_id}');

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log(`Progress: ${update.progress}% - ${update.message}`);
};
```

### Configuration Options

#### Research Options

- **max_sources**: Maximum number of sources to analyze (default: 15)
- **include_academic**: Include academic database searches (default: true)
- **format**: Output format - "markdown" or "html" (default: "markdown")
- **citation_style**: Citation format - "apa", "mla", or "chicago" (default: "apa")
- **complexity**: Research depth - "simple", "medium", or "complex" (auto-detected)

#### Agent Configuration

Customize agent behavior through environment variables:

```bash
# Research Agent
RESEARCH_AGENT_MAX_RETRIES=3
RESEARCH_AGENT_TIMEOUT=30

# API Agent  
API_AGENT_RATE_LIMIT=10
API_AGENT_DATABASES=arxiv,pubmed,semantic_scholar

# Analysis Agent
ANALYSIS_AGENT_NER_MODEL=en_core_web_sm
ANALYSIS_AGENT_TOPIC_MODEL=lda

# Drafting Agent
DRAFTING_AGENT_MIN_SECTIONS=4
DRAFTING_AGENT_MAX_WORDS=5000
```

## Understanding the Output

### Document Structure

Generated research documents follow a consistent structure:

1. **Title and Table of Contents**
2. **Introduction**: Context and scope
3. **Main Sections**: Organized by topic/theme
4. **Analysis**: Key insights and findings
5. **Conclusion**: Summary and implications
6. **References**: Formatted citations

### Quality Indicators

Each document includes metadata indicating quality:

- **Confidence Score**: Overall reliability (0.0-1.0)
- **Source Count**: Number of sources analyzed
- **Academic Ratio**: Percentage of academic sources
- **Processing Time**: Total research duration

### Citation Format

All sources are properly cited with:
- In-text citations linked to references
- Complete bibliographic information
- URL verification status
- Access dates for web sources

## Advanced Usage

### Batch Processing

Process multiple queries simultaneously:

```python
import asyncio
from agent_scrivener.api.client import AgentScrivenerClient

async def batch_research():
    client = AgentScrivenerClient()
    
    queries = [
        "AI in healthcare applications",
        "Renewable energy storage solutions", 
        "Remote work productivity analysis"
    ]
    
    sessions = await asyncio.gather(*[
        client.submit_research(query) for query in queries
    ])
    
    results = await asyncio.gather(*[
        client.wait_for_completion(session.session_id) 
        for session in sessions
    ])
    
    return results
```

### Custom Agent Configuration

Override default agent behavior:

```python
from agent_scrivener.orchestration.orchestrator import ResearchOrchestrator
from agent_scrivener.agents.research_agent import ResearchAgent

# Custom research agent with specific parameters
custom_research_agent = ResearchAgent(
    max_sources=25,
    quality_threshold=0.8,
    timeout=60
)

orchestrator = ResearchOrchestrator()
orchestrator.register_agent("research", custom_research_agent)
```

### Memory and Session Management

Access previous research sessions:

```python
from agent_scrivener.memory.session_manager import SessionManager

session_manager = SessionManager()

# Retrieve previous session
session = await session_manager.get_session(session_id)

# Build on previous research
related_sessions = await session_manager.find_related_sessions(
    query="quantum computing",
    similarity_threshold=0.7
)
```

## Troubleshooting

### Common Issues

#### "No sources found" Error
- Check internet connectivity
- Verify API credentials for academic databases
- Try broader or more specific query terms

#### Slow Processing
- Reduce max_sources parameter
- Check system resource usage
- Verify external API response times

#### Citation Formatting Issues
- Ensure citation_style is supported
- Check source URL accessibility
- Verify academic database connectivity

### Error Codes

- **400**: Invalid query format
- **401**: Authentication required
- **429**: Rate limit exceeded
- **500**: Internal processing error
- **503**: External service unavailable

### Logging and Debugging

Enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
export ENABLE_AGENT_TRACING=true
python -m agent_scrivener.api.main
```

View logs:
```bash
tail -f logs/agent_scrivener.log
```

## Performance Optimization

### System Requirements

**Minimum:**
- 4 GB RAM
- 2 CPU cores
- 10 GB storage

**Recommended:**
- 8 GB RAM
- 4 CPU cores
- 50 GB storage
- SSD for better I/O performance

### Scaling Considerations

- **Concurrent Sessions**: System supports up to 10 concurrent research sessions
- **Memory Usage**: Each session uses approximately 500MB RAM
- **Processing Time**: Varies from 3-15 minutes depending on complexity
- **API Limits**: Respect external service rate limits

## Support and Resources

### Documentation
- [API Reference](api_reference.md)
- [Developer Guide](developer_guide.md)
- [Deployment Guide](deployment_guide.md)

### Community
- GitHub Issues: Report bugs and feature requests
- Discussions: Ask questions and share experiences

### Professional Support
- Enterprise deployment assistance
- Custom agent development
- Performance optimization consulting

## Changelog

### Version 1.0.0
- Initial release with core research capabilities
- Multi-agent orchestration system
- Web and academic database integration
- Real-time progress tracking
- Comprehensive citation management

### Upcoming Features
- Multi-language support
- Custom output templates
- Advanced visualization generation
- Integration with reference managers
- Collaborative research sessions