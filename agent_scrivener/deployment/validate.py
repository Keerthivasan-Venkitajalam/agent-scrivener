#!/usr/bin/env python3
"""Deployment validation script for Agent Scrivener."""

import asyncio
import sys
import argparse
import logging
from typing import List, Dict, Any

from .environment import env_manager, validate_environment
from .secrets import secrets_manager, validate_secrets
from .health_check import DeploymentValidator, HealthChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_configuration() -> List[str]:
    """Validate deployment configuration."""
    errors = []
    
    logger.info("Validating environment configuration...")
    if not validate_environment():
        errors.append("Environment configuration validation failed")
    
    logger.info("Validating secrets access...")
    if not validate_secrets():
        errors.append("Secrets validation failed")
    
    return errors


def validate_docker_setup() -> List[str]:
    """Validate Docker setup and image."""
    errors = []
    
    try:
        import subprocess
        
        # Check if Docker is available
        result = subprocess.run(
            ["docker", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode != 0:
            errors.append("Docker is not available or not working properly")
        else:
            logger.info(f"Docker version: {result.stdout.strip()}")
        
        # Check if agent-scrivener image exists
        result = subprocess.run(
            ["docker", "images", "agent-scrivener", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            logger.info(f"Found Docker images: {result.stdout.strip()}")
        else:
            errors.append("Agent Scrivener Docker image not found - run build first")
            
    except subprocess.TimeoutExpired:
        errors.append("Docker commands timed out")
    except FileNotFoundError:
        errors.append("Docker command not found - ensure Docker is installed")
    except Exception as e:
        errors.append(f"Docker validation failed: {str(e)}")
    
    return errors


def validate_aws_access() -> List[str]:
    """Validate AWS access and permissions."""
    errors = []
    
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        aws_config = env_manager.get_aws_config()
        
        # Test AWS credentials
        try:
            sts_client = boto3.client("sts", region_name=aws_config.region)
            identity = sts_client.get_caller_identity()
            logger.info(f"AWS Identity: {identity.get('Arn', 'Unknown')}")
        except NoCredentialsError:
            errors.append("AWS credentials not configured")
        except ClientError as e:
            errors.append(f"AWS credentials validation failed: {e}")
        
        # Test Bedrock access
        try:
            bedrock_client = boto3.client("bedrock-runtime", region_name=aws_config.region)
            # This is just a client creation test - actual model access would require more setup
            logger.info("Bedrock client created successfully")
        except Exception as e:
            errors.append(f"Bedrock access validation failed: {e}")
        
        # Test other AWS services if configured
        if aws_config.vpc_id:
            try:
                ec2_client = boto3.client("ec2", region_name=aws_config.region)
                ec2_client.describe_vpcs(VpcIds=[aws_config.vpc_id])
                logger.info(f"VPC {aws_config.vpc_id} is accessible")
            except ClientError as e:
                errors.append(f"VPC {aws_config.vpc_id} validation failed: {e}")
        
    except ImportError:
        errors.append("boto3 not installed - required for AWS integration")
    except Exception as e:
        errors.append(f"AWS validation failed: {str(e)}")
    
    return errors


async def validate_runtime_health(base_url: str = None) -> List[str]:
    """Validate runtime health if service is running."""
    errors = []
    
    try:
        health_checker = HealthChecker(base_url)
        results = await health_checker.check_all()
        
        unhealthy_components = [
            result.component for result in results 
            if result.status.value == "unhealthy"
        ]
        
        if unhealthy_components:
            errors.extend([f"Unhealthy component: {comp}" for comp in unhealthy_components])
        
        # Generate health report
        validator = DeploymentValidator()
        report = validator.generate_health_report(results)
        logger.info(f"Health Report:\n{report}")
        
    except Exception as e:
        # Runtime health check is optional if service isn't running yet
        logger.warning(f"Runtime health check failed (service may not be running): {e}")
    
    return errors


async def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate Agent Scrivener deployment")
    parser.add_argument(
        "--skip-docker", 
        action="store_true", 
        help="Skip Docker validation"
    )
    parser.add_argument(
        "--skip-aws", 
        action="store_true", 
        help="Skip AWS validation"
    )
    parser.add_argument(
        "--skip-runtime", 
        action="store_true", 
        help="Skip runtime health validation"
    )
    parser.add_argument(
        "--base-url", 
        default="http://localhost:8000",
        help="Base URL for runtime health checks"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Agent Scrivener deployment validation...")
    
    all_errors = []
    
    # Configuration validation
    logger.info("=== Configuration Validation ===")
    config_errors = validate_configuration()
    all_errors.extend(config_errors)
    
    if config_errors:
        for error in config_errors:
            logger.error(error)
    else:
        logger.info("✅ Configuration validation passed")
    
    # Docker validation
    if not args.skip_docker:
        logger.info("=== Docker Validation ===")
        docker_errors = validate_docker_setup()
        all_errors.extend(docker_errors)
        
        if docker_errors:
            for error in docker_errors:
                logger.error(error)
        else:
            logger.info("✅ Docker validation passed")
    
    # AWS validation
    if not args.skip_aws:
        logger.info("=== AWS Validation ===")
        aws_errors = validate_aws_access()
        all_errors.extend(aws_errors)
        
        if aws_errors:
            for error in aws_errors:
                logger.error(error)
        else:
            logger.info("✅ AWS validation passed")
    
    # Runtime health validation
    if not args.skip_runtime:
        logger.info("=== Runtime Health Validation ===")
        runtime_errors = await validate_runtime_health(args.base_url)
        all_errors.extend(runtime_errors)
        
        if runtime_errors:
            for error in runtime_errors:
                logger.error(error)
        else:
            logger.info("✅ Runtime health validation passed")
    
    # Summary
    logger.info("=== Validation Summary ===")
    if all_errors:
        logger.error(f"❌ Validation failed with {len(all_errors)} errors:")
        for i, error in enumerate(all_errors, 1):
            logger.error(f"  {i}. {error}")
        sys.exit(1)
    else:
        logger.info("✅ All validations passed - deployment is ready!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())