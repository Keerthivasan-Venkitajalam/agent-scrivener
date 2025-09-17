# Agent Scrivener API Reference

## Overview

The Agent Scrivener API provides RESTful endpoints for submitting research queries, monitoring progress, and retrieving results. The API is built with FastAPI and includes comprehensive request/response validation, authentication, and real-time WebSocket support.

## Base URL

```
Production: https://api.agent-scrivener.com
Development: http://localhost:8000
```

## Authentication

### API Key Authentication

Include your API key in the request headers:

```http
Authorization: Bearer your-api-key-here
```

### Rate Limiting

- **Free Tier**: 10 requests per hour
- **Pro Tier**: 100 requests per hour  
- **Enterprise**: Custom limits

Rate limit headers are included in all responses:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642694400
```

## Endpoints

### Research Operations

#### Submit Research Request

Submit a new research query for processing.

```http
POST /research
```

**Request Body:**
```json
{
  "query": "string (required, min: 10, max: 2000)",
  "options": {
    "max_sources": "integer (optional, default: 15, max: 50)",
    "include_academic": "boolean (optional, default: true)",
    "format": "string (optional, enum: ['markdown', 'html'], default: 'markdown')",
    "citation_style": "string (optional, enum: ['apa', 'mla', 'chicago'], default: 'apa')",
    "priority": "string (optional, enum: ['low', 'normal', 'high'], default: 'normal')"
  },
  "metadata": {
    "user_id": "string (optional)",
    "project_id": "string (optional)",
    "tags": "array of strings (optional)"
  }
}
```

**Response (201 Created):**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "estimated_completion": "2024-01-15T10:30:00Z",
  "query": "Your research question",
  "options": {
    "max_sources": 15,
    "include_academic": true,
    "format": "markdown",
    "citation_style": "apa"
  },
  "created_at": "2024-01-15T10:00:00Z"
}
```

**Error Responses:**
```json
// 400 Bad Request
{
  "error": "validation_error",
  "message": "Query must be between 10 and 2000 characters",
  "details": {
    "field": "query",
    "code": "length_validation"
  }
}

// 429 Too Many Requests
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded. Try again in 3600 seconds",
  "retry_after": 3600
}
```

#### Get Research Status

Retrieve the current status of a research session.

```http
GET /research/{session_id}/status
```

**Path Parameters:**
- `session_id` (string, required): UUID of the research session

**Response (200 OK):**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 65,
  "current_step": "Analysis Agent processing content",
  "steps_completed": [
    "Query analysis completed",
    "Web search completed", 
    "Academic database search completed",
    "Content extraction completed"
  ],
  "steps_remaining": [
    "Content analysis",
    "Document drafting",
    "Citation formatting"
  ],
  "estimated_remaining": "3 minutes",
  "started_at": "2024-01-15T10:00:00Z",
  "last_updated": "2024-01-15T10:05:30Z"
}
```

**Status Values:**
- `queued`: Request received and queued for processing
- `processing`: Research in progress
- `completed`: Research completed successfully
- `failed`: Research failed with errors
- `cancelled`: Research cancelled by user

#### Get Research Result

Retrieve the completed research document and metadata.

```http
GET /research/{session_id}/result
```

**Response (200 OK):**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "document": "# Research Title\n\n## Introduction\n\n...",
  "metadata": {
    "sources_analyzed": 15,
    "academic_sources": 8,
    "web_sources": 7,
    "processing_time": "8.5 minutes",
    "confidence_score": 0.92,
    "word_count": 2847,
    "citation_count": 15,
    "sections": [
      "Introduction",
      "Current State",
      "Analysis", 
      "Future Implications",
      "Conclusion"
    ]
  },
  "sources": [
    {
      "id": "source_1",
      "title": "Research Paper Title",
      "url": "https://example.com/paper",
      "type": "academic",
      "confidence": 0.95,
      "citation": "Author, A. (2024). Research Paper Title. Journal Name, 15(3), 45-62."
    }
  ],
  "completed_at": "2024-01-15T10:08:30Z"
}
```

#### Cancel Research

Cancel an ongoing research session.

```http
DELETE /research/{session_id}
```

**Response (200 OK):**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "message": "Research session cancelled successfully"
}
```

### Session Management

#### List Research Sessions

Retrieve a list of research sessions for the authenticated user.

```http
GET /research/sessions
```

**Query Parameters:**
- `limit` (integer, optional, default: 20, max: 100): Number of sessions to return
- `offset` (integer, optional, default: 0): Number of sessions to skip
- `status` (string, optional): Filter by status
- `sort` (string, optional, enum: ['created_at', 'updated_at'], default: 'created_at')
- `order` (string, optional, enum: ['asc', 'desc'], default: 'desc')

**Response (200 OK):**
```json
{
  "sessions": [
    {
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "query": "AI in healthcare applications",
      "status": "completed",
      "created_at": "2024-01-15T10:00:00Z",
      "completed_at": "2024-01-15T10:08:30Z",
      "metadata": {
        "word_count": 2847,
        "confidence_score": 0.92
      }
    }
  ],
  "total": 45,
  "limit": 20,
  "offset": 0
}
```

#### Get Session Details

Retrieve detailed information about a specific session.

```http
GET /research/{session_id}
```

**Response (200 OK):**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "AI in healthcare applications",
  "status": "completed",
  "options": {
    "max_sources": 15,
    "include_academic": true,
    "format": "markdown",
    "citation_style": "apa"
  },
  "progress_log": [
    {
      "timestamp": "2024-01-15T10:00:00Z",
      "step": "query_analysis",
      "message": "Query analysis started",
      "progress": 0
    },
    {
      "timestamp": "2024-01-15T10:00:30Z", 
      "step": "query_analysis",
      "message": "Query analysis completed",
      "progress": 10
    }
  ],
  "created_at": "2024-01-15T10:00:00Z",
  "completed_at": "2024-01-15T10:08:30Z"
}
```

### System Information

#### Health Check

Check system health and availability.

```http
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:00:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "redis": "healthy", 
    "external_apis": "healthy"
  },
  "metrics": {
    "active_sessions": 5,
    "queue_length": 2,
    "average_processing_time": "6.2 minutes"
  }
}
```

#### System Metrics

Get system performance metrics (requires admin access).

```http
GET /metrics
```

**Response (200 OK):**
```json
{
  "timestamp": "2024-01-15T10:00:00Z",
  "requests": {
    "total": 1250,
    "successful": 1180,
    "failed": 70,
    "success_rate": 0.944
  },
  "performance": {
    "average_processing_time": "6.2 minutes",
    "median_processing_time": "5.8 minutes",
    "95th_percentile": "12.1 minutes"
  },
  "resources": {
    "active_sessions": 5,
    "queue_length": 2,
    "memory_usage": "2.1 GB",
    "cpu_usage": "45%"
  }
}
```

## WebSocket API

### Real-time Progress Updates

Connect to receive real-time updates for a research session.

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/{session_id}');

ws.onopen = function(event) {
    console.log('Connected to research session');
};

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log('Progress update:', update);
};

ws.onclose = function(event) {
    console.log('Connection closed');
};
```

**Message Format:**
```json
{
  "type": "progress_update",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "progress": 65,
  "step": "analysis",
  "message": "Analysis Agent processing content",
  "timestamp": "2024-01-15T10:05:30Z",
  "data": {
    "sources_processed": 12,
    "insights_generated": 8
  }
}
```

**Message Types:**
- `progress_update`: Processing progress update
- `step_completed`: Processing step completed
- `error`: Error occurred during processing
- `completed`: Research completed
- `cancelled`: Research cancelled

## Data Models

### ResearchRequest

```json
{
  "query": "string (required)",
  "options": {
    "max_sources": "integer (optional)",
    "include_academic": "boolean (optional)",
    "format": "string (optional)",
    "citation_style": "string (optional)",
    "priority": "string (optional)"
  },
  "metadata": {
    "user_id": "string (optional)",
    "project_id": "string (optional)",
    "tags": "array (optional)"
  }
}
```

### ResearchResponse

```json
{
  "session_id": "string (uuid)",
  "status": "string (enum)",
  "document": "string (optional)",
  "metadata": {
    "sources_analyzed": "integer",
    "processing_time": "string",
    "confidence_score": "number (0-1)",
    "word_count": "integer",
    "citation_count": "integer"
  },
  "sources": "array (optional)",
  "created_at": "string (iso datetime)",
  "completed_at": "string (iso datetime, optional)"
}
```

### Source

```json
{
  "id": "string",
  "title": "string",
  "url": "string (url)",
  "type": "string (enum: academic, web, database)",
  "confidence": "number (0-1)",
  "citation": "string",
  "metadata": {
    "authors": "array (optional)",
    "publication_date": "string (optional)",
    "journal": "string (optional)",
    "doi": "string (optional)"
  }
}
```

## Error Handling

### Error Response Format

All API errors follow a consistent format:

```json
{
  "error": "error_code",
  "message": "Human-readable error message",
  "details": {
    "field": "field_name (optional)",
    "code": "validation_code (optional)",
    "context": "additional_context (optional)"
  },
  "timestamp": "2024-01-15T10:00:00Z",
  "request_id": "req_550e8400e29b41d4a716446655440000"
}
```

### Common Error Codes

- `validation_error`: Request validation failed
- `authentication_required`: Valid API key required
- `authorization_failed`: Insufficient permissions
- `rate_limit_exceeded`: Rate limit exceeded
- `session_not_found`: Research session not found
- `processing_error`: Error during research processing
- `external_service_error`: External API unavailable
- `internal_server_error`: Unexpected server error

## SDK and Client Libraries

### Python SDK

```python
from agent_scrivener import AgentScrivenerClient

client = AgentScrivenerClient(api_key="your-api-key")

# Submit research
session = await client.submit_research(
    query="AI in healthcare applications",
    options={"max_sources": 20}
)

# Monitor progress
async for update in client.stream_progress(session.session_id):
    print(f"Progress: {update.progress}%")

# Get result
result = await client.get_result(session.session_id)
print(result.document)
```

### JavaScript SDK

```javascript
import { AgentScrivenerClient } from '@agent-scrivener/client';

const client = new AgentScrivenerClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.agent-scrivener.com'
});

// Submit research
const session = await client.submitResearch({
  query: 'AI in healthcare applications',
  options: { maxSources: 20 }
});

// Monitor progress
client.onProgress(session.sessionId, (update) => {
  console.log(`Progress: ${update.progress}%`);
});

// Get result
const result = await client.getResult(session.sessionId);
console.log(result.document);
```

## Changelog

### v1.0.0
- Initial API release
- Core research endpoints
- WebSocket support
- Authentication and rate limiting

### v1.1.0 (Planned)
- Batch processing endpoints
- Advanced filtering options
- Export format options
- Webhook notifications