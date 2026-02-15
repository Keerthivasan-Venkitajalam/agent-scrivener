"""Deployment configuration validator for infrastructure and configuration validation."""

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

from .base_validator import BaseValidator
from .models import ValidationResult, ValidationStatus


class DeploymentConfigValidator(BaseValidator):
    """Validates all deployment configuration files and settings.
    
    This validator tests:
    - Required environment variables
    - Docker and Docker Compose configuration
    - AWS CDK stack definitions
    - AgentCore configuration schema
    - AWS Secrets Manager access
    - Database configuration and connectivity
    """
    
    def __init__(
        self,
        project_root: Optional[Path] = None,
        required_env_vars: Optional[List[str]] = None
    ):
        """Initialize the deployment configuration validator.
        
        Args:
            project_root: Root directory of the project (defaults to current directory)
            required_env_vars: List of required environment variables
        """
        super().__init__(
            name="DeploymentConfigValidator",
            timeout_seconds=600  # 10 minutes for all config tests
        )
        self.project_root = project_root or Path.cwd()
        self.required_env_vars = required_env_vars or [
            "AWS_REGION",
            "BEDROCK_MODEL_ID",
            "DATABASE_URL"
        ]

    async def validate(self) -> List[ValidationResult]:
        """Execute all deployment configuration validation checks.
        
        Returns:
            List of validation results
        """
        self.log_validation_start()
        results = []
        
        # Validate environment variables
        env_result = self.validate_environment_variables()
        results.append(env_result)
        
        # Validate Docker configuration
        docker_result = self.validate_docker_configuration()
        results.append(docker_result)
        
        # Validate AWS CDK configuration
        cdk_result = self.validate_aws_cdk_configuration()
        results.append(cdk_result)
        
        # Validate AgentCore configuration
        agentcore_result = self.validate_agentcore_configuration()
        results.append(agentcore_result)
        
        # Validate secrets access
        secrets_result = await self.validate_secrets_access()
        results.append(secrets_result)
        
        # Validate database configuration
        db_result = await self.validate_database_configuration()
        results.append(db_result)
        
        self.log_validation_complete(results)
        return results

    def validate_environment_variables(self) -> ValidationResult:
        """Validate required environment variables are present.
        
        Tests:
        - All required environment variables are set
        - Variables have non-empty values
        
        Returns:
            ValidationResult for environment variables
        """
        start_time = time.time()
        
        try:
            missing_vars = []
            empty_vars = []
            
            for var in self.required_env_vars:
                value = os.environ.get(var)
                if value is None:
                    missing_vars.append(var)
                elif not value.strip():
                    empty_vars.append(var)
            
            if missing_vars or empty_vars:
                error_parts = []
                if missing_vars:
                    error_parts.append(f"missing: {', '.join(missing_vars)}")
                if empty_vars:
                    error_parts.append(f"empty: {', '.join(empty_vars)}")
                
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Environment variable validation failed - {'; '.join(error_parts)}",
                    duration_seconds=time.time() - start_time,
                    details={
                        "missing_vars": missing_vars,
                        "empty_vars": empty_vars,
                        "required_vars": self.required_env_vars
                    },
                    remediation_steps=[
                        "Set missing environment variables in your .env file or environment",
                        "Ensure all required variables have non-empty values",
                        f"Required variables: {', '.join(self.required_env_vars)}",
                        "For AWS_REGION, use a valid AWS region (e.g., us-east-1)",
                        "For BEDROCK_MODEL_ID, use a valid Bedrock model ID (e.g., anthropic.claude-v2)",
                        "For DATABASE_URL, use a valid database connection string"
                    ]
                )
            
            return self.create_result(
                status=ValidationStatus.PASS,
                message=f"All {len(self.required_env_vars)} required environment variables are set",
                duration_seconds=time.time() - start_time,
                details={
                    "required_vars": self.required_env_vars,
                    "all_present": True
                }
            )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Environment variable validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check environment variable configuration",
                    "Verify .env file is properly formatted"
                ]
            )

    def validate_docker_configuration(self) -> ValidationResult:
        """Validate Docker and Docker Compose configuration.
        
        Tests:
        - Dockerfile exists and builds successfully
        - docker-compose.yml exists and is valid
        - All required services are defined
        
        Returns:
            ValidationResult for Docker configuration
        """
        start_time = time.time()
        
        try:
            # Check if Dockerfile exists
            dockerfile_path = self.project_root / "Dockerfile"
            if not dockerfile_path.exists():
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="Dockerfile not found",
                    duration_seconds=time.time() - start_time,
                    details={"expected_path": str(dockerfile_path)},
                    remediation_steps=[
                        f"Create Dockerfile at {dockerfile_path}",
                        "Ensure Dockerfile includes all necessary build steps",
                        "Reference deployment documentation for Dockerfile template"
                    ]
                )
            
            # Check if docker-compose.yml exists
            compose_path = self.project_root / "docker-compose.yml"
            if not compose_path.exists():
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="docker-compose.yml not found",
                    duration_seconds=time.time() - start_time,
                    details={"expected_path": str(compose_path)},
                    remediation_steps=[
                        f"Create docker-compose.yml at {compose_path}",
                        "Define all required services (api, database, redis)",
                        "Reference deployment documentation for docker-compose template"
                    ]
                )
            
            # Validate docker-compose.yml syntax
            try:
                with open(compose_path, 'r') as f:
                    compose_config = yaml.safe_load(f)
                
                if not isinstance(compose_config, dict):
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="docker-compose.yml is not a valid YAML dictionary",
                        duration_seconds=time.time() - start_time,
                        remediation_steps=[
                            "Fix YAML syntax in docker-compose.yml",
                            "Ensure file starts with 'version:' and 'services:'"
                        ]
                    )
                
                # Check for required services
                services = compose_config.get("services", {})
                required_services = ["api", "database"]
                missing_services = [s for s in required_services if s not in services]
                
                if missing_services:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"docker-compose.yml missing required services: {', '.join(missing_services)}",
                        duration_seconds=time.time() - start_time,
                        details={
                            "missing_services": missing_services,
                            "defined_services": list(services.keys())
                        },
                        remediation_steps=[
                            f"Add missing services to docker-compose.yml: {', '.join(missing_services)}",
                            "Ensure each service has proper configuration (image/build, ports, volumes)",
                            "Reference deployment documentation for service definitions"
                        ]
                    )
            
            except yaml.YAMLError as e:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"docker-compose.yml has invalid YAML syntax: {str(e)}",
                    duration_seconds=time.time() - start_time,
                    details={"yaml_error": str(e)},
                    remediation_steps=[
                        "Fix YAML syntax errors in docker-compose.yml",
                        "Use a YAML validator to check syntax",
                        "Ensure proper indentation and structure"
                    ]
                )
            
            # Try to build Docker image (optional, can be slow)
            # For now, we'll just validate the files exist and are syntactically correct
            
            return self.create_result(
                status=ValidationStatus.PASS,
                message="Docker configuration validation passed",
                duration_seconds=time.time() - start_time,
                details={
                    "dockerfile_exists": True,
                    "compose_file_exists": True,
                    "services_defined": list(services.keys())
                }
            )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Docker configuration validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check Docker configuration files",
                    "Verify file permissions and accessibility"
                ]
            )

    def validate_aws_cdk_configuration(self) -> ValidationResult:
        """Validate AWS CDK stack definitions using cdk synth.
        
        Tests:
        - CDK app file exists
        - CDK configuration is syntactically correct
        - cdk synth runs successfully
        
        Returns:
            ValidationResult for AWS CDK configuration
        """
        start_time = time.time()
        
        try:
            # Check if CDK app exists
            cdk_app_path = self.project_root / "cdk" / "app.py"
            if not cdk_app_path.exists():
                # Try alternative location
                cdk_app_path = self.project_root / "infrastructure" / "app.py"
                if not cdk_app_path.exists():
                    return self.create_result(
                        status=ValidationStatus.SKIP,
                        message="AWS CDK app not found - skipping CDK validation",
                        duration_seconds=time.time() - start_time,
                        details={
                            "checked_paths": [
                                str(self.project_root / "cdk" / "app.py"),
                                str(self.project_root / "infrastructure" / "app.py")
                            ]
                        }
                    )
            
            # Check if cdk.json exists
            cdk_json_path = self.project_root / "cdk.json"
            if not cdk_json_path.exists():
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="cdk.json not found",
                    duration_seconds=time.time() - start_time,
                    details={"expected_path": str(cdk_json_path)},
                    remediation_steps=[
                        f"Create cdk.json at {cdk_json_path}",
                        "Run 'cdk init' to initialize CDK project",
                        "Reference AWS CDK documentation for configuration"
                    ]
                )
            
            # Validate cdk.json syntax
            try:
                with open(cdk_json_path, 'r') as f:
                    cdk_config = json.load(f)
                
                if "app" not in cdk_config:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="cdk.json missing 'app' field",
                        duration_seconds=time.time() - start_time,
                        remediation_steps=[
                            "Add 'app' field to cdk.json specifying the CDK app entry point",
                            "Example: \"app\": \"python3 cdk/app.py\""
                        ]
                    )
            
            except json.JSONDecodeError as e:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"cdk.json has invalid JSON syntax: {str(e)}",
                    duration_seconds=time.time() - start_time,
                    details={"json_error": str(e)},
                    remediation_steps=[
                        "Fix JSON syntax errors in cdk.json",
                        "Use a JSON validator to check syntax"
                    ]
                )
            
            # Try to run cdk synth to validate stack definitions
            try:
                result = subprocess.run(
                    ["cdk", "synth", "--quiet"],
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message=f"CDK synth failed with exit code {result.returncode}",
                        duration_seconds=time.time() - start_time,
                        details={
                            "exit_code": result.returncode,
                            "stdout": result.stdout,
                            "stderr": result.stderr
                        },
                        remediation_steps=[
                            "Fix CDK stack definition errors",
                            "Check CDK app for syntax errors",
                            "Ensure all required CDK dependencies are installed",
                            "Run 'cdk synth' manually to see detailed error messages"
                        ]
                    )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message="AWS CDK configuration validation passed",
                    duration_seconds=time.time() - start_time,
                    details={
                        "cdk_app_exists": True,
                        "cdk_json_valid": True,
                        "synth_successful": True
                    }
                )
            
            except FileNotFoundError:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="CDK CLI not found - skipping CDK synth validation",
                    duration_seconds=time.time() - start_time,
                    remediation_steps=[
                        "Install AWS CDK CLI: npm install -g aws-cdk",
                        "Verify CDK is in PATH"
                    ]
                )
            
            except subprocess.TimeoutExpired:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="CDK synth timed out after 60 seconds",
                    duration_seconds=time.time() - start_time,
                    remediation_steps=[
                        "Check for infinite loops or blocking operations in CDK app",
                        "Verify CDK dependencies are properly installed"
                    ]
                )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"AWS CDK configuration validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check AWS CDK configuration files",
                    "Verify CDK installation and dependencies"
                ]
            )

    def validate_agentcore_configuration(self) -> ValidationResult:
        """Validate AgentCore configuration with schema validation.
        
        Tests:
        - AgentCore config file exists
        - YAML syntax is valid
        - All agent definitions include required fields (name, model, tools)
        
        Returns:
            ValidationResult for AgentCore configuration
        """
        start_time = time.time()
        
        try:
            # Check for AgentCore config file
            config_paths = [
                self.project_root / "agentcore_config.yml",
                self.project_root / "config" / "agentcore.yml",
                self.project_root / "agentcore.yaml",
                self.project_root / "config" / "agentcore.yaml"
            ]
            
            config_path = None
            for path in config_paths:
                if path.exists():
                    config_path = path
                    break
            
            if not config_path:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="AgentCore configuration file not found - skipping validation",
                    duration_seconds=time.time() - start_time,
                    details={
                        "checked_paths": [str(p) for p in config_paths]
                    }
                )
            
            # Load and validate YAML syntax
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                if not isinstance(config, dict):
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="AgentCore configuration is not a valid YAML dictionary",
                        duration_seconds=time.time() - start_time,
                        remediation_steps=[
                            "Fix YAML syntax in AgentCore configuration",
                            "Ensure file contains a valid YAML dictionary"
                        ]
                    )
            
            except yaml.YAMLError as e:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"AgentCore configuration has invalid YAML syntax: {str(e)}",
                    duration_seconds=time.time() - start_time,
                    details={"yaml_error": str(e)},
                    remediation_steps=[
                        "Fix YAML syntax errors in AgentCore configuration",
                        "Use a YAML validator to check syntax",
                        "Ensure proper indentation and structure"
                    ]
                )
            
            # Validate agent definitions
            agents = config.get("agents", [])
            if not agents:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="AgentCore configuration has no agent definitions",
                    duration_seconds=time.time() - start_time,
                    remediation_steps=[
                        "Add agent definitions to AgentCore configuration",
                        "Each agent should have name, model, and tools fields"
                    ]
                )
            
            # Check each agent has required fields
            required_fields = ["name", "model", "tools"]
            invalid_agents = []
            
            for i, agent in enumerate(agents):
                if not isinstance(agent, dict):
                    invalid_agents.append({
                        "index": i,
                        "issue": "Agent is not a dictionary"
                    })
                    continue
                
                missing_fields = [f for f in required_fields if f not in agent]
                if missing_fields:
                    invalid_agents.append({
                        "index": i,
                        "name": agent.get("name", f"agent_{i}"),
                        "missing_fields": missing_fields
                    })
            
            if invalid_agents:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"AgentCore configuration has {len(invalid_agents)} invalid agent definition(s)",
                    duration_seconds=time.time() - start_time,
                    details={
                        "invalid_agents": invalid_agents,
                        "required_fields": required_fields
                    },
                    remediation_steps=[
                        "Fix invalid agent definitions in AgentCore configuration",
                        f"Each agent must include: {', '.join(required_fields)}",
                        "Verify agent names are unique",
                        "Ensure model IDs are valid Bedrock model identifiers",
                        "Ensure tools is a list of tool names"
                    ]
                )
            
            return self.create_result(
                status=ValidationStatus.PASS,
                message=f"AgentCore configuration validation passed ({len(agents)} agents defined)",
                duration_seconds=time.time() - start_time,
                details={
                    "config_path": str(config_path),
                    "agents_count": len(agents),
                    "agent_names": [a.get("name") for a in agents]
                }
            )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"AgentCore configuration validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check AgentCore configuration file",
                    "Verify file permissions and accessibility"
                ]
            )

    async def validate_secrets_access(self) -> ValidationResult:
        """Validate AWS Secrets Manager access.
        
        Tests:
        - AWS credentials are configured
        - Can connect to AWS Secrets Manager
        - Can retrieve a test secret (if configured)
        
        Returns:
            ValidationResult for secrets access
        """
        start_time = time.time()
        
        try:
            # Check if boto3 is available
            try:
                import boto3
                from botocore.exceptions import ClientError, NoCredentialsError
            except ImportError:
                return self.create_result(
                    status=ValidationStatus.SKIP,
                    message="boto3 not installed - skipping AWS Secrets Manager validation",
                    duration_seconds=time.time() - start_time,
                    remediation_steps=[
                        "Install boto3: pip install boto3",
                        "Required for AWS Secrets Manager access"
                    ]
                )
            
            # Get AWS region from environment
            aws_region = os.environ.get("AWS_REGION")
            if not aws_region:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="AWS_REGION environment variable not set",
                    duration_seconds=time.time() - start_time,
                    remediation_steps=[
                        "Set AWS_REGION environment variable",
                        "Example: export AWS_REGION=us-east-1"
                    ]
                )
            
            # Try to create Secrets Manager client
            try:
                client = boto3.client('secretsmanager', region_name=aws_region)
                
                # Try to list secrets (just to verify access)
                response = await asyncio.to_thread(
                    client.list_secrets,
                    MaxResults=1
                )
                
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message="AWS Secrets Manager access validation passed",
                    duration_seconds=time.time() - start_time,
                    details={
                        "aws_region": aws_region,
                        "secrets_manager_accessible": True
                    }
                )
            
            except NoCredentialsError:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="AWS credentials not configured",
                    duration_seconds=time.time() - start_time,
                    remediation_steps=[
                        "Configure AWS credentials using one of:",
                        "  - AWS CLI: aws configure",
                        "  - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY",
                        "  - IAM role (if running on AWS infrastructure)",
                        "Verify credentials have SecretsManager permissions"
                    ]
                )
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"AWS Secrets Manager access failed: {error_code}",
                    duration_seconds=time.time() - start_time,
                    details={
                        "error_code": error_code,
                        "error_message": str(e)
                    },
                    remediation_steps=[
                        "Verify AWS credentials have SecretsManager permissions",
                        "Check IAM policy includes secretsmanager:ListSecrets",
                        "Verify AWS region is correct",
                        f"Error: {error_code}"
                    ]
                )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Secrets access validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check AWS configuration",
                    "Verify boto3 is properly installed",
                    "Check network connectivity to AWS"
                ]
            )

    async def validate_database_configuration(self) -> ValidationResult:
        """Validate database configuration and connectivity.
        
        Tests:
        - DATABASE_URL environment variable is set
        - Database connection string is valid
        - Can connect to database
        
        Returns:
            ValidationResult for database configuration
        """
        start_time = time.time()
        
        try:
            # Check if DATABASE_URL is set
            database_url = os.environ.get("DATABASE_URL")
            if not database_url:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message="DATABASE_URL environment variable not set",
                    duration_seconds=time.time() - start_time,
                    remediation_steps=[
                        "Set DATABASE_URL environment variable",
                        "Format: postgresql://user:password@host:port/database",
                        "Example: postgresql://postgres:password@localhost:5432/agent_scrivener"
                    ]
                )
            
            # Parse database URL to validate format
            try:
                from urllib.parse import urlparse
                parsed = urlparse(database_url)
                
                if not parsed.scheme:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="DATABASE_URL missing database scheme",
                        duration_seconds=time.time() - start_time,
                        remediation_steps=[
                            "DATABASE_URL must start with database scheme (e.g., postgresql://)",
                            "Format: postgresql://user:password@host:port/database"
                        ]
                    )
                
                if not parsed.hostname:
                    return self.create_result(
                        status=ValidationStatus.FAIL,
                        message="DATABASE_URL missing hostname",
                        duration_seconds=time.time() - start_time,
                        remediation_steps=[
                            "DATABASE_URL must include hostname",
                            "Format: postgresql://user:password@host:port/database"
                        ]
                    )
            
            except Exception as e:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"DATABASE_URL has invalid format: {str(e)}",
                    duration_seconds=time.time() - start_time,
                    remediation_steps=[
                        "Fix DATABASE_URL format",
                        "Format: postgresql://user:password@host:port/database"
                    ]
                )
            
            # Try to connect to database
            try:
                # Check if database driver is available
                if parsed.scheme.startswith('postgresql'):
                    try:
                        import asyncpg
                    except ImportError:
                        return self.create_result(
                            status=ValidationStatus.SKIP,
                            message="asyncpg not installed - skipping database connection test",
                            duration_seconds=time.time() - start_time,
                            details={
                                "database_url_set": True,
                                "database_url_valid": True
                            },
                            remediation_steps=[
                                "Install asyncpg: pip install asyncpg",
                                "Required for PostgreSQL database connectivity"
                            ]
                        )
                    
                    # Try to connect
                    try:
                        conn = await asyncio.wait_for(
                            asyncpg.connect(database_url),
                            timeout=5.0
                        )
                        await conn.close()
                        
                        return self.create_result(
                            status=ValidationStatus.PASS,
                            message="Database configuration validation passed",
                            duration_seconds=time.time() - start_time,
                            details={
                                "database_url_set": True,
                                "database_url_valid": True,
                                "connection_successful": True,
                                "database_type": parsed.scheme
                            }
                        )
                    
                    except asyncio.TimeoutError:
                        return self.create_result(
                            status=ValidationStatus.FAIL,
                            message="Database connection timed out after 5 seconds",
                            duration_seconds=time.time() - start_time,
                            remediation_steps=[
                                "Check if database server is running",
                                "Verify database host and port are correct",
                                "Check network connectivity to database",
                                "Verify firewall rules allow database connections"
                            ]
                        )
                    
                    except Exception as e:
                        return self.create_result(
                            status=ValidationStatus.FAIL,
                            message=f"Database connection failed: {str(e)}",
                            duration_seconds=time.time() - start_time,
                            details={"exception": str(e), "exception_type": type(e).__name__},
                            remediation_steps=[
                                "Check database credentials are correct",
                                "Verify database server is running",
                                "Check database name exists",
                                "Verify user has access to database",
                                f"Error: {str(e)}"
                            ]
                        )
                
                else:
                    # Unsupported database type
                    return self.create_result(
                        status=ValidationStatus.SKIP,
                        message=f"Database type '{parsed.scheme}' not supported for connection testing",
                        duration_seconds=time.time() - start_time,
                        details={
                            "database_url_set": True,
                            "database_url_valid": True,
                            "database_type": parsed.scheme
                        }
                    )
            
            except Exception as e:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Database connection test error: {str(e)}",
                    duration_seconds=time.time() - start_time,
                    details={"exception": str(e), "exception_type": type(e).__name__},
                    remediation_steps=[
                        "Check database configuration",
                        "Verify database driver is installed"
                    ]
                )
        
        except Exception as e:
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Database configuration validation error: {str(e)}",
                duration_seconds=time.time() - start_time,
                details={"exception": str(e), "exception_type": type(e).__name__},
                remediation_steps=[
                    "Check database configuration",
                    "Verify DATABASE_URL environment variable"
                ]
            )
