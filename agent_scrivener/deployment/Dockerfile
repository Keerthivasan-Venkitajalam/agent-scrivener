# Multi-stage build for Agent Scrivener
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r agentuser && useradd -r -g agentuser agentuser

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /home/agentuser/.local

# Copy application code
COPY agent_scrivener/ ./agent_scrivener/
COPY setup.py .
COPY pyproject.toml .

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH=/home/agentuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R agentuser:agentuser /app

# Switch to non-root user
USER agentuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "agent_scrivener.api.main"]