# Agent Scrivener Deployment

This directory contains deployment configurations and scripts for Agent Scrivener on AWS AgentCore Runtime.

## Files Overview

### Core Configuration Files

- **`Dockerfile`** - Multi-stage Docker build configuration for containerizing the application
- **`docker-compose.yml`** - Local development and testing environment setup
- **`agentcore-config.yml`** - AgentCore Runtime configuration for agent orchestration
- **`.env.example`** - Template for environment variables (copy to `.env`)

### Python Modules

- **`environment.py`** - Environment configuration management and validation
- **`secrets.py`** - Secrets management for AWS Secrets Manager and Parameter Store
- **`health_check.py`** - Health checking and deployment validation utilities

### Scripts

- **`deploy.sh`** - Main deployment script for AgentCore Runtime
- **`validate.py`** - Deployment validation script

## Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp agent_scrivener/deployment/.env.example agent_scrivener/deployment/.env

# Edit with your configuration
nano agent_scrivener/deployment/.env
```

### 2. Build and Test

```bash
# Build Docker image
cd agent_scrivener/deployment
./deploy.sh build

# Run validation
python3 validate.py --verbose
```

### 3. Deploy

```bash
# Full deployment
./deploy.sh deploy

# Or deploy to specific environment
ENVIRONMENT=production ./deploy.sh deploy
```

## Configuration

### Environment Variables

Key environment variables that must be configured:

```bash
# AWS Configuration
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Security
JWT_SECRET_KEY=your_secure_jwt_secret
DB_PASSWORD=your_secure_db_password

# AgentCore Runtime
MAX_CONCURRENT_SESSIONS=10
AGENTCORE_MEMORY_MB=2048
```

### Secrets Management

The system supports multiple secret providers:

1. **AWS Secrets Manager** (recommended for production)
2. **AWS Parameter Store** (for configuration values)
3. **Environment Variables** (for development)
4. **Kubernetes Secrets** (for K8s deployments)

Configure secrets in AWS:

```bash
# Create JWT secret
aws secretsmanager create-secret \
    --name "agent-scrivener/jwt-secret" \
    --secret-string "your-secure-jwt-secret"

# Create database credentials
aws secretsmanager create-secret \
    --name "agent-scrivener/database-credentials" \
    --secret-string '{"password":"your-db-password"}'

# Create API keys
aws secretsmanager create-secret \
    --name "agent-scrivener/external-api-keys" \
    --secret-string '{"semantic_scholar":"key1","pubmed":"key2"}'
```

## AgentCore Runtime Configuration

The `agentcore-config.yml` file defines:

- **Agent specifications** - Memory, timeout, and environment settings for each agent type
- **Tool configurations** - Browser, Gateway, Code Interpreter, and Memory tool settings
- **Scaling policies** - Auto-scaling rules and resource limits
- **Security policies** - Network policies and secret management
- **Monitoring setup** - Metrics, logging, and health check configurations

### Agent Types

| Agent | Memory | Timeout | Purpose |
|-------|--------|---------|---------|
| Planner | 512Mi | 5min | Task orchestration and planning |
| Research | 1024Mi | 30min | Web research and content extraction |
| API | 512Mi | 10min | Academic database queries |
| Analysis | 1536Mi | 60min | Data analysis and NLP processing |
| Drafting | 1024Mi | 30min | Content synthesis and formatting |
| Citation | 512Mi | 10min | Citation management and validation |

## Deployment Methods

### Method 1: AgentCore CLI (Recommended)

If you have the AgentCore CLI installed:

```bash
# Deploy using AgentCore CLI
agentcore deploy \
    --config agentcore-config.yml \
    --image agent-scrivener:latest \
    --environment production
```

### Method 2: AWS Services Direct

For environments without AgentCore CLI:

```bash
# The deploy script will automatically use AWS services
AGENTCORE_RUNTIME=ecs ./deploy.sh deploy
```

## Health Checks and Monitoring

### Built-in Health Checks

The system includes comprehensive health checks:

- **API Health** - Endpoint responsiveness and status
- **Database Connectivity** - Connection and query validation
- **AWS Services** - Bedrock and other service access
- **Agent Availability** - Individual agent status and readiness
- **Memory Usage** - System resource monitoring
- **External Dependencies** - Third-party API accessibility

### Running Health Checks

```bash
# Basic health check
python3 -m agent_scrivener.deployment.health_check

# Full deployment validation
python3 -m agent_scrivener.deployment.health_check --validate

# Detailed validation with custom URL
python3 validate.py --base-url https://your-api-endpoint.com --verbose
```

### Monitoring Integration

The deployment automatically configures:

- **CloudWatch Metrics** - Performance and usage metrics
- **CloudWatch Logs** - Structured application logging
- **Health Check Endpoints** - `/health` and `/metrics` endpoints
- **Alerting** - Configurable alerts for critical issues

## Troubleshooting

### Common Issues

1. **Docker Build Failures**
   ```bash
   # Check Docker daemon
   docker info
   
   # Clean build cache
   docker system prune -f
   ```

2. **AWS Permission Issues**
   ```bash
   # Check AWS credentials
   aws sts get-caller-identity
   
   # Verify Bedrock access
   aws bedrock list-foundation-models --region us-east-1
   ```

3. **Environment Configuration Errors**
   ```bash
   # Validate configuration
   python3 -c "from agent_scrivener.deployment import validate_environment; validate_environment()"
   ```

4. **Secret Access Issues**
   ```bash
   # Test secret access
   python3 -c "from agent_scrivener.deployment import validate_secrets; validate_secrets()"
   ```

### Logs and Debugging

```bash
# View deployment logs
docker logs agent-scrivener-api

# Check health status
curl http://localhost:8000/health

# View detailed metrics
curl http://localhost:8000/metrics
```

## Security Considerations

### Network Security

- All external communication uses HTTPS/TLS
- VPC isolation for production deployments
- Security groups restrict access to necessary ports only
- API Gateway provides additional security layer

### Secrets Management

- No secrets stored in environment variables in production
- AWS Secrets Manager integration with automatic rotation
- Encrypted secrets in transit and at rest
- Least-privilege IAM roles for service access

### Container Security

- Non-root user execution in containers
- Minimal base image with security updates
- Regular vulnerability scanning
- Resource limits to prevent resource exhaustion

## Performance Tuning

### Memory Configuration

Adjust memory allocation based on workload:

```yaml
# In agentcore-config.yml
spec:
  agents:
    - name: analysis-agent
      memory: "2048Mi"  # Increase for large datasets
```

### Scaling Configuration

Configure auto-scaling parameters:

```yaml
spec:
  runtime:
    scaling:
      minInstances: 2      # Minimum for high availability
      maxInstances: 10     # Maximum based on cost constraints
      targetConcurrency: 5 # Requests per instance
```

### Database Optimization

- Use connection pooling for database access
- Configure appropriate timeout values
- Monitor query performance and optimize as needed

## Cost Optimization

### Resource Management

- Use spot instances for non-critical workloads
- Configure appropriate auto-scaling policies
- Monitor and optimize memory usage
- Use reserved instances for predictable workloads

### Monitoring Costs

- Set up billing alerts
- Monitor AgentCore Runtime usage
- Track API call costs for external services
- Optimize agent execution times

## Support and Maintenance

### Regular Maintenance Tasks

1. **Update Dependencies** - Regular security updates
2. **Monitor Performance** - Track metrics and optimize
3. **Backup Data** - Regular backup of persistent data
4. **Test Deployments** - Validate in staging before production
5. **Review Logs** - Monitor for errors and issues

### Getting Help

- Check the health check output for specific issues
- Review CloudWatch logs for detailed error information
- Use the validation script to identify configuration problems
- Monitor the `/metrics` endpoint for performance insights