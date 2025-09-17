#!/bin/bash

# Infrastructure Deployment Script for Agent Scrivener
# Deploys AWS infrastructure using CloudFormation or CDK

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_METHOD="${DEPLOYMENT_METHOD:-cloudformation}"  # cloudformation or cdk
ENVIRONMENT="${ENVIRONMENT:-development}"
PROJECT_NAME="${PROJECT_NAME:-agent-scrivener}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed or not in PATH"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid"
        exit 1
    fi
    
    # Check deployment method specific tools
    if [ "$DEPLOYMENT_METHOD" = "cdk" ]; then
        if ! command -v cdk &> /dev/null; then
            log_error "AWS CDK is not installed or not in PATH"
            log_info "Install with: npm install -g aws-cdk"
            exit 1
        fi
        
        if ! command -v python3 &> /dev/null; then
            log_error "Python 3 is not installed or not in PATH"
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Validate parameters
validate_parameters() {
    log_info "Validating deployment parameters..."
    
    # Required parameters
    if [ -z "$ALERT_EMAIL" ]; then
        log_error "ALERT_EMAIL environment variable is required"
        exit 1
    fi
    
    if [ -z "$AGENTCORE_ENDPOINT" ]; then
        log_error "AGENTCORE_ENDPOINT environment variable is required"
        exit 1
    fi
    
    # Validate environment
    if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
        log_error "ENVIRONMENT must be one of: development, staging, production"
        exit 1
    fi
    
    log_success "Parameter validation passed"
}

# Deploy using CloudFormation
deploy_cloudformation() {
    log_info "Deploying infrastructure using CloudFormation..."
    
    local stack_name="${PROJECT_NAME}-${ENVIRONMENT}"
    local template_file="$SCRIPT_DIR/main-stack.yml"
    
    # Check if stack exists
    if aws cloudformation describe-stacks --stack-name "$stack_name" --region "$AWS_REGION" &> /dev/null; then
        log_info "Stack exists, updating..."
        local action="update-stack"
    else
        log_info "Stack does not exist, creating..."
        local action="create-stack"
    fi
    
    # Deploy stack
    aws cloudformation "$action" \
        --stack-name "$stack_name" \
        --template-body "file://$template_file" \
        --parameters \
            ParameterKey=Environment,ParameterValue="$ENVIRONMENT" \
            ParameterKey=ProjectName,ParameterValue="$PROJECT_NAME" \
            ParameterKey=AlertEmail,ParameterValue="$ALERT_EMAIL" \
            ParameterKey=AgentCoreEndpoint,ParameterValue="$AGENTCORE_ENDPOINT" \
            ParameterKey=DomainName,ParameterValue="${DOMAIN_NAME:-}" \
            ParameterKey=CertificateArn,ParameterValue="${CERTIFICATE_ARN:-}" \
        --capabilities CAPABILITY_NAMED_IAM \
        --region "$AWS_REGION" \
        --tags \
            Key=Environment,Value="$ENVIRONMENT" \
            Key=Project,Value="$PROJECT_NAME" \
            Key=ManagedBy,Value=CloudFormation
    
    # Wait for deployment to complete
    log_info "Waiting for stack deployment to complete..."
    if [ "$action" = "create-stack" ]; then
        aws cloudformation wait stack-create-complete --stack-name "$stack_name" --region "$AWS_REGION"
    else
        aws cloudformation wait stack-update-complete --stack-name "$stack_name" --region "$AWS_REGION"
    fi
    
    log_success "CloudFormation deployment completed"
}

# Deploy using CDK
deploy_cdk() {
    log_info "Deploying infrastructure using AWS CDK..."
    
    cd "$SCRIPT_DIR/cdk"
    
    # Install Python dependencies
    if [ ! -d "venv" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install -r requirements.txt
    
    # Set environment variables for CDK
    export ENVIRONMENT="$ENVIRONMENT"
    export ALERT_EMAIL="$ALERT_EMAIL"
    export AGENTCORE_ENDPOINT="$AGENTCORE_ENDPOINT"
    export DOMAIN_NAME="${DOMAIN_NAME:-}"
    export CERTIFICATE_ARN="${CERTIFICATE_ARN:-}"
    
    # Bootstrap CDK (if needed)
    log_info "Bootstrapping CDK..."
    cdk bootstrap aws://"$(aws sts get-caller-identity --query Account --output text)"/"$AWS_REGION"
    
    # Deploy stacks
    log_info "Deploying CDK stacks..."
    cdk deploy --all --require-approval never
    
    deactivate
    cd - > /dev/null
    
    log_success "CDK deployment completed"
}

# Get stack outputs
get_stack_outputs() {
    log_info "Retrieving stack outputs..."
    
    local stack_name="${PROJECT_NAME}-${ENVIRONMENT}"
    
    if [ "$DEPLOYMENT_METHOD" = "cloudformation" ]; then
        aws cloudformation describe-stacks \
            --stack-name "$stack_name" \
            --region "$AWS_REGION" \
            --query 'Stacks[0].Outputs' \
            --output table
    else
        # For CDK, we need to check multiple stacks
        local main_stack="AgentScrivenerMain-${ENVIRONMENT}"
        aws cloudformation describe-stacks \
            --stack-name "$main_stack" \
            --region "$AWS_REGION" \
            --query 'Stacks[0].Outputs' \
            --output table
    fi
}

# Validate deployment
validate_deployment() {
    log_info "Validating deployment..."
    
    # Get API Gateway URL from stack outputs
    local stack_name="${PROJECT_NAME}-${ENVIRONMENT}"
    if [ "$DEPLOYMENT_METHOD" = "cdk" ]; then
        stack_name="AgentScrivenerMain-${ENVIRONMENT}"
    fi
    
    local api_url=$(aws cloudformation describe-stacks \
        --stack-name "$stack_name" \
        --region "$AWS_REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`APIGatewayURL`].OutputValue' \
        --output text)
    
    if [ -n "$api_url" ]; then
        log_info "Testing API Gateway endpoint: $api_url"
        
        # Test health endpoint
        if curl -f -s "${api_url}health" > /dev/null; then
            log_success "API Gateway health check passed"
        else
            log_warning "API Gateway health check failed (this may be expected if AgentCore is not deployed yet)"
        fi
    else
        log_warning "Could not retrieve API Gateway URL from stack outputs"
    fi
    
    log_success "Deployment validation completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add any cleanup logic here
}

# Main deployment function
main() {
    log_info "Starting Agent Scrivener infrastructure deployment..."
    log_info "Method: $DEPLOYMENT_METHOD"
    log_info "Environment: $ENVIRONMENT"
    log_info "Region: $AWS_REGION"
    
    # Trap errors for cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    validate_parameters
    
    if [ "$DEPLOYMENT_METHOD" = "cdk" ]; then
        deploy_cdk
    else
        deploy_cloudformation
    fi
    
    get_stack_outputs
    validate_deployment
    
    log_success "Agent Scrivener infrastructure deployment completed successfully!"
    log_info "Next steps:"
    log_info "1. Deploy the AgentCore Runtime using the deployment scripts"
    log_info "2. Update the AGENTCORE_ENDPOINT with the actual runtime URL"
    log_info "3. Configure external API keys in AWS Secrets Manager"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "outputs")
        get_stack_outputs
        ;;
    "validate")
        validate_deployment
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy    - Deploy infrastructure (default)"
        echo "  outputs   - Show stack outputs"
        echo "  validate  - Validate deployment"
        echo "  help      - Show this help"
        echo ""
        echo "Environment variables:"
        echo "  DEPLOYMENT_METHOD   - cloudformation or cdk (default: cloudformation)"
        echo "  ENVIRONMENT         - development|staging|production (default: development)"
        echo "  PROJECT_NAME        - Project name (default: agent-scrivener)"
        echo "  AWS_REGION          - AWS region (default: us-east-1)"
        echo "  ALERT_EMAIL         - Email for alerts (required)"
        echo "  AGENTCORE_ENDPOINT  - AgentCore Runtime endpoint (required)"
        echo "  DOMAIN_NAME         - Custom domain name (optional)"
        echo "  CERTIFICATE_ARN     - ACM certificate ARN (optional)"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac