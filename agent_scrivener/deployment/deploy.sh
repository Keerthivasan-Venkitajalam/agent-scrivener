#!/bin/bash

# Agent Scrivener Deployment Script
# This script handles deployment to AgentCore Runtime

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_ENV="${ENVIRONMENT:-development}"
AGENTCORE_CONFIG="$SCRIPT_DIR/agentcore-config.yml"
DOCKER_IMAGE_NAME="agent-scrivener"
DOCKER_TAG="${DOCKER_TAG:-latest}"

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
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed or not in PATH"
        exit 1
    fi
    
    # Check AgentCore CLI (if available)
    if command -v agentcore &> /dev/null; then
        log_info "AgentCore CLI found"
    else
        log_warning "AgentCore CLI not found - using alternative deployment method"
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Validate environment configuration
validate_environment() {
    log_info "Validating environment configuration..."
    
    cd "$PROJECT_ROOT"
    
    # Run environment validation
    if python3 -c "
from agent_scrivener.deployment.environment import validate_environment
import sys
if not validate_environment():
    sys.exit(1)
"; then
        log_success "Environment configuration is valid"
    else
        log_error "Environment configuration validation failed"
        exit 1
    fi
}

# Build Docker image
build_docker_image() {
    log_info "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build the image
    docker build \
        -f agent_scrivener/deployment/Dockerfile \
        -t "$DOCKER_IMAGE_NAME:$DOCKER_TAG" \
        --build-arg ENVIRONMENT="$DEPLOYMENT_ENV" \
        .
    
    if [ $? -eq 0 ]; then
        log_success "Docker image built successfully: $DOCKER_IMAGE_NAME:$DOCKER_TAG"
    else
        log_error "Docker image build failed"
        exit 1
    fi
}

# Run pre-deployment tests
run_tests() {
    log_info "Running pre-deployment tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run unit tests
    if python3 -m pytest tests/unit/ -v --tb=short; then
        log_success "Unit tests passed"
    else
        log_error "Unit tests failed"
        exit 1
    fi
    
    # Run integration tests (if available)
    if [ -d "tests/integration" ]; then
        if python3 -m pytest tests/integration/ -v --tb=short; then
            log_success "Integration tests passed"
        else
            log_warning "Integration tests failed - continuing with deployment"
        fi
    fi
}

# Deploy to AgentCore Runtime
deploy_to_agentcore() {
    log_info "Deploying to AgentCore Runtime..."
    
    # Method 1: Using AgentCore CLI (if available)
    if command -v agentcore &> /dev/null; then
        log_info "Using AgentCore CLI for deployment"
        
        # Deploy using AgentCore CLI
        agentcore deploy \
            --config "$AGENTCORE_CONFIG" \
            --image "$DOCKER_IMAGE_NAME:$DOCKER_TAG" \
            --environment "$DEPLOYMENT_ENV"
        
        if [ $? -eq 0 ]; then
            log_success "AgentCore deployment completed"
        else
            log_error "AgentCore deployment failed"
            exit 1
        fi
    else
        # Method 2: Using AWS services directly
        log_info "Using AWS services for deployment"
        deploy_to_aws
    fi
}

# Deploy to AWS (fallback method)
deploy_to_aws() {
    log_info "Deploying to AWS services..."
    
    # Push Docker image to ECR
    push_to_ecr
    
    # Deploy using ECS or Lambda (depending on configuration)
    if [ "$AGENTCORE_RUNTIME" = "lambda" ]; then
        deploy_to_lambda
    else
        deploy_to_ecs
    fi
}

# Push Docker image to ECR
push_to_ecr() {
    log_info "Pushing Docker image to ECR..."
    
    AWS_REGION="${AWS_REGION:-us-east-1}"
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REPOSITORY="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/agent-scrivener"
    
    # Login to ECR
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$ECR_REPOSITORY"
    
    # Tag and push image
    docker tag "$DOCKER_IMAGE_NAME:$DOCKER_TAG" "$ECR_REPOSITORY:$DOCKER_TAG"
    docker push "$ECR_REPOSITORY:$DOCKER_TAG"
    
    log_success "Docker image pushed to ECR: $ECR_REPOSITORY:$DOCKER_TAG"
}

# Deploy to ECS
deploy_to_ecs() {
    log_info "Deploying to ECS..."
    
    # This would typically use AWS CDK or CloudFormation
    # For now, we'll create a basic ECS service
    
    log_warning "ECS deployment not fully implemented - manual setup required"
    log_info "Please use the provided CloudFormation templates in the infrastructure/ directory"
}

# Deploy to Lambda
deploy_to_lambda() {
    log_info "Deploying to Lambda..."
    
    log_warning "Lambda deployment not fully implemented - manual setup required"
    log_info "Please use the provided SAM templates in the infrastructure/ directory"
}

# Run post-deployment validation
validate_deployment() {
    log_info "Running post-deployment validation..."
    
    cd "$PROJECT_ROOT"
    
    # Wait for deployment to be ready
    sleep 30
    
    # Run health checks
    if python3 -m agent_scrivener.deployment.health_check --validate; then
        log_success "Deployment validation passed"
    else
        log_error "Deployment validation failed"
        exit 1
    fi
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    # This would implement rollback logic
    log_info "Rollback functionality not implemented yet"
    log_info "Please manually revert to previous deployment"
}

# Main deployment function
main() {
    log_info "Starting Agent Scrivener deployment..."
    log_info "Environment: $DEPLOYMENT_ENV"
    log_info "Docker tag: $DOCKER_TAG"
    
    # Trap errors for rollback
    trap 'log_error "Deployment failed - consider rollback"; exit 1' ERR
    
    # Run deployment steps
    check_prerequisites
    validate_environment
    build_docker_image
    run_tests
    deploy_to_agentcore
    validate_deployment
    
    log_success "Agent Scrivener deployment completed successfully!"
    log_info "Access the API at: ${API_ENDPOINT:-http://localhost:8000}"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "build")
        check_prerequisites
        build_docker_image
        ;;
    "test")
        run_tests
        ;;
    "validate")
        validate_deployment
        ;;
    "rollback")
        rollback_deployment
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy    - Full deployment (default)"
        echo "  build     - Build Docker image only"
        echo "  test      - Run tests only"
        echo "  validate  - Validate deployment only"
        echo "  rollback  - Rollback deployment"
        echo "  help      - Show this help"
        echo ""
        echo "Environment variables:"
        echo "  ENVIRONMENT     - Deployment environment (development|staging|production)"
        echo "  DOCKER_TAG      - Docker image tag (default: latest)"
        echo "  AWS_REGION      - AWS region (default: us-east-1)"
        echo "  API_ENDPOINT    - API endpoint URL for validation"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac