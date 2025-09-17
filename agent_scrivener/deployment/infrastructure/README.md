# Agent Scrivener Infrastructure

This directory contains Infrastructure as Code (IaC) templates for deploying Agent Scrivener on AWS.

## Overview

The infrastructure supports two deployment methods:
1. **CloudFormation** - YAML templates for direct AWS deployment
2. **AWS CDK** - Python-based infrastructure code for advanced scenarios

## Architecture Components

### Core Infrastructure
- **API Gateway** - REST API with authentication and rate limiting
- **IAM Roles** - Least-privilege roles for AgentCore and individual agents
- **S3 Bucket** - Temporary data storage for analysis workflows
- **DynamoDB** - Memory store for session and research data
- **Secrets Manager** - Secure storage for API keys and credentials
- **Parameter Store** - Configuration management
- **CloudWatch** - Monitoring, logging, and alerting

### Production Additions
- **VPC** - Isolated network environment
- **Security Groups** - Network access controls
- **VPC Endpoints** - Private access to AWS services
- **NAT Gateways** - Secure outbound internet access

## Quick Start

### Prerequisites

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure

# For CDK deployment, install Node.js and CDK
npm install -g aws-cdk
```

### CloudFormation Deployment

```bash
# Set required environment variables
export ALERT_EMAIL="admin@yourcompany.com"
export AGENTCORE_ENDPOINT="https://your-agentcore-endpoint.com"
export ENVIRONMENT="development"

# Deploy infrastructure
./deploy-infrastructure.sh deploy
```

### CDK Deployment

```bash
# Set deployment method to CDK
export DEPLOYMENT_METHOD="cdk"
export ALERT_EMAIL="admin@yourcompany.com"
export AGENTCORE_ENDPOINT="https://your-agentcore-endpoint.com"
export ENVIRONMENT="development"

# Deploy infrastructure
./deploy-infrastructure.sh deploy
```

## Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ALERT_EMAIL` | Email for critical alerts | `admin@company.com` |
| `AGENTCORE_ENDPOINT` | AgentCore Runtime URL | `https://runtime.agentcore.com` |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEPLOYMENT_METHOD` | `cloudformation` or `cdk` | `cloudformation` |
| `ENVIRONMENT` | `development`, `staging`, `production` | `development` |
| `PROJECT_NAME` | Project name for resources | `agent-scrivener` |
| `AWS_REGION` | AWS deployment region | `us-east-1` |
| `DOMAIN_NAME` | Custom domain for API | _(none)_ |
| `CERTIFICATE_ARN` | ACM certificate ARN | _(none)_ |

## File Structure

```
infrastructure/
├── README.md                    # This file
├── deploy-infrastructure.sh     # Main deployment script
├── main-stack.yml              # Main CloudFormation template
├── api-gateway.yml             # API Gateway configuration
├── iam-roles.yml               # IAM roles and policies
├── cloudwatch-monitoring.yml   # Monitoring and alerting
├── vpc-networking.yml          # VPC and networking (production)
└── cdk/                        # AWS CDK implementation
    ├── app.py                  # CDK application entry point
    ├── cdk.json                # CDK configuration
    ├── requirements.txt        # Python dependencies
    └── stacks/                 # CDK stack definitions
        ├── main_stack.py       # Main infrastructure stack
        ├── monitoring_stack.py # Monitoring stack
        └── networking_stack.py # Networking stack
```

## CloudFormation Templates

### main-stack.yml
Master template that orchestrates all other stacks:
- Deploys IAM, API Gateway, and Monitoring stacks
- Creates secrets and parameters
- Sets up resource groups for cost tracking

### api-gateway.yml
API Gateway configuration:
- REST API with regional endpoints
- Request validation and rate limiting
- Usage plans and API keys
- CORS configuration
- Custom domain support

### iam-roles.yml
Security and access control:
- AgentCore execution role with Bedrock access
- Individual agent roles with least-privilege permissions
- S3 and DynamoDB access policies
- Secrets Manager and Parameter Store permissions

### cloudwatch-monitoring.yml
Monitoring and alerting:
- Log groups for all agents
- Custom metrics from log patterns
- CloudWatch alarms for errors and performance
- SNS topics for alert notifications
- CloudWatch dashboard for visualization

### vpc-networking.yml
Production networking (optional):
- VPC with public and private subnets
- NAT gateways for outbound internet access
- Security groups with restrictive rules
- VPC endpoints for AWS services

## AWS CDK Implementation

The CDK implementation provides the same infrastructure with additional benefits:
- Type safety and IDE support
- Programmatic resource configuration
- Advanced constructs and patterns
- Better testing capabilities

### CDK Stacks

- **MainStack** - Core infrastructure (IAM, API Gateway, storage)
- **MonitoringStack** - CloudWatch monitoring and alerting
- **NetworkingStack** - VPC and networking (production only)

## Deployment Environments

### Development
- Simplified networking (no VPC)
- Lower resource limits
- Shorter retention periods
- Cost-optimized settings

### Staging
- Production-like configuration
- Reduced scale for cost efficiency
- Full monitoring and alerting
- Integration testing environment

### Production
- Full VPC with private subnets
- High availability across AZs
- Enhanced security controls
- Comprehensive monitoring
- Backup and disaster recovery

## Security Features

### Network Security
- VPC isolation in production
- Security groups with minimal access
- VPC endpoints for AWS service access
- WAF integration (optional)

### Access Control
- IAM roles with least-privilege principles
- Resource-based policies
- API Gateway authentication
- Secrets rotation capabilities

### Data Protection
- Encryption at rest and in transit
- Secure secrets management
- Audit logging
- Data retention policies

## Monitoring and Alerting

### Metrics Tracked
- API request counts and latency
- Error rates and types
- Agent execution times
- Resource utilization
- Cost metrics

### Alerts Configured
- High error rates
- API latency spikes
- Agent failures
- Resource exhaustion
- Security events

### Dashboards
- Real-time system health
- Performance trends
- Cost analysis
- Error analysis

## Cost Optimization

### Resource Optimization
- Pay-per-request DynamoDB billing
- S3 lifecycle policies for temporary data
- CloudWatch log retention policies
- Right-sized compute resources

### Cost Monitoring
- Resource tagging for cost allocation
- Budget alerts and limits
- Usage tracking and reporting
- Reserved instance recommendations

## Troubleshooting

### Common Issues

1. **Stack Creation Fails**
   ```bash
   # Check CloudFormation events
   aws cloudformation describe-stack-events --stack-name agent-scrivener-development
   ```

2. **Permission Errors**
   ```bash
   # Verify AWS credentials
   aws sts get-caller-identity
   
   # Check required permissions
   aws iam simulate-principal-policy --policy-source-arn $(aws sts get-caller-identity --query Arn --output text) --action-names cloudformation:CreateStack
   ```

3. **CDK Bootstrap Issues**
   ```bash
   # Re-bootstrap CDK
   cdk bootstrap --force
   ```

4. **API Gateway Not Accessible**
   - Check security group rules
   - Verify API Gateway deployment
   - Test with curl or Postman

### Validation Commands

```bash
# Check stack status
aws cloudformation describe-stacks --stack-name agent-scrivener-development

# Test API Gateway
curl -f https://your-api-id.execute-api.us-east-1.amazonaws.com/development/health

# View CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /aws/agentcore/agent-scrivener
```

## Cleanup

### Delete CloudFormation Stack
```bash
aws cloudformation delete-stack --stack-name agent-scrivener-development
```

### Delete CDK Stacks
```bash
cd cdk/
cdk destroy --all
```

### Manual Cleanup
Some resources may require manual deletion:
- S3 buckets with objects
- CloudWatch log groups
- Secrets in Secrets Manager

## Support

For infrastructure issues:
1. Check CloudFormation events and stack status
2. Review CloudWatch logs for detailed error information
3. Validate IAM permissions and resource limits
4. Test individual components in isolation