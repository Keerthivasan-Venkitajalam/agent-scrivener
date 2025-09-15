# Agent Scrivener API

This module provides a comprehensive REST API and WebSocket interface for the Agent Scrivener autonomous research platform.

## Features

### REST API Endpoints

#### Authentication & Security
- JWT-based authentication with configurable scopes
- Rate limiting (60 requests per minute per user)
- Request validation and error handling
- CORS and security middleware

#### Research Management
- `POST /api/v1/research` - Start new research session
- `GET /api/v1/research/{session_id}/status` - Get session status
- `GET /api/v1/research/{session_id}/result` - Get completed research results
- `POST /api/v1/research/{session_id}/cancel` - Cancel running session
- `GET /api/v1/research` - List user sessions (paginated)
- `DELETE /api/v1/research/{session_id}` - Delete session

#### System Health
- `GET /api/v1/health` - Health check endpoint
- `GET /` - API information

### WebSocket Real-time Updates

#### Progress Tracking
- Real-time progress updates during research execution
- Task completion notifications
- Session status changes
- Error reporting

#### Connection Management
- Authenticated WebSocket connections
- Multiple clients per session support
- Automatic cleanup on disconnect
- Ping/pong heartbeat support

## Architecture

### Core Components

1. **FastAPI Application** (`main.py`)
   - Main application with middleware
   - Error handling and logging
   - CORS and security configuration

2. **API Routes** (`routes.py`)
   - RESTful endpoint implementations
   - Request/response validation
   - Authentication and authorization

3. **WebSocket Handler** (`websocket.py`)
   - Real-time communication
   - Connection management
   - Progress broadcasting

4. **Authentication** (`auth.py`)
   - JWT token management
   - Rate limiting
   - Scope-based authorization

5. **Data Models** (`models.py`)
   - Pydantic models for validation
   - Request/response schemas
   - Error handling models

6. **Orchestrator Adapter** (`orchestrator_adapter.py`)
   - Bridge between API and core orchestrator
   - Session management
   - Progress simulation

### Data Flow

```
Client Request → Authentication → Rate Limiting → Route Handler → Orchestrator → Response
                                                      ↓
WebSocket Client ← Progress Updates ← Progress Tracker ← Orchestrator Events
```

## Usage Examples

### Starting a Research Session

```python
import httpx

headers = {"Authorization": "Bearer <jwt_token>"}
request_data = {
    "query": "What are the latest developments in quantum computing?",
    "max_sources": 10,
    "include_academic": True,
    "include_web": True,
    "priority": "high"
}

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/research",
        json=request_data,
        headers=headers
    )
    session_data = response.json()
    session_id = session_data["session_id"]
```

### WebSocket Progress Monitoring

```python
import websockets
import json

async def monitor_progress(session_id, token):
    uri = f"ws://localhost:8000/api/v1/ws/research/{session_id}/progress?token={token}"
    
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            data = json.loads(message)
            
            if data["type"] == "progress_update":
                print(f"Progress: {data['progress_percentage']}%")
            elif data["type"] == "session_completed":
                print("Research completed!")
                break
```

### Getting Results

```python
async with httpx.AsyncClient() as client:
    response = await client.get(
        f"http://localhost:8000/api/v1/research/{session_id}/result",
        headers=headers
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Document: {result['document_content']}")
        print(f"Sources: {result['sources_count']}")
        print(f"Word count: {result['word_count']}")
```

## Testing

The API includes comprehensive test coverage:

- **Unit Tests**: Model validation, authentication, rate limiting
- **Integration Tests**: WebSocket functionality, progress tracking
- **API Tests**: Endpoint behavior, error handling

Run tests with:
```bash
pytest tests/unit/test_api_*.py tests/integration/test_websocket.py -v
```

## Demo

A complete demo script is available at `examples/api_demo.py` that demonstrates:
- Health check
- Research session creation
- Real-time progress monitoring
- Result retrieval
- Session management

Run the demo:
```bash
python examples/api_demo.py
```

## Configuration

### Environment Variables

- `JWT_SECRET_KEY`: Secret key for JWT tokens (default: development key)
- `JWT_ALGORITHM`: JWT algorithm (default: HS256)
- `RATE_LIMIT_RPM`: Requests per minute limit (default: 60)
- `API_HOST`: API host (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)

### Production Considerations

1. **Security**
   - Use strong JWT secret keys
   - Configure allowed CORS origins
   - Set up proper TLS/SSL
   - Implement proper logging and monitoring

2. **Scalability**
   - Use Redis for rate limiting storage
   - Implement proper session persistence
   - Configure load balancing
   - Set up health checks

3. **Monitoring**
   - Add metrics collection
   - Implement distributed tracing
   - Set up alerting
   - Monitor WebSocket connections

## Dependencies

- FastAPI: Web framework
- Uvicorn: ASGI server
- WebSockets: Real-time communication
- PyJWT: JWT token handling
- Pydantic: Data validation
- HTTPx: HTTP client (for testing)

## API Documentation

When running the server, interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`