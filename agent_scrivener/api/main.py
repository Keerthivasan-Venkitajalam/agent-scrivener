"""
Main FastAPI application for Agent Scrivener.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from .routes import router
from .websocket import ws_router
from .models import ErrorResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Agent Scrivener API...")
    yield
    # Shutdown
    logger.info("Shutting down Agent Scrivener API...")


# Create FastAPI application
app = FastAPI(
    title="Agent Scrivener API",
    description="Autonomous research and content synthesis platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware (security)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # In production, specify allowed hosts
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response


# Include API routes
app.include_router(router, prefix="/api/v1")
app.include_router(ws_router, prefix="/api/v1/ws")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Agent Scrivener API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="NOT_FOUND",
            message="The requested resource was not found",
            details={"path": str(request.url.path)}
        ).dict()
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message="An internal server error occurred"
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)