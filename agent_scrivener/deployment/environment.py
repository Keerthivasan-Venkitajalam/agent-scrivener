"""Environment configuration management for Agent Scrivener deployment."""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class Environment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: int
    name: str
    username: str
    password: str
    ssl_enabled: bool = True


@dataclass
class AWSConfig:
    """AWS service configuration."""
    region: str
    bedrock_model_id: str
    iam_role_arn: Optional[str] = None
    vpc_id: Optional[str] = None
    subnet_ids: Optional[list] = None
    security_group_ids: Optional[list] = None


@dataclass
class AgentCoreConfig:
    """AgentCore Runtime configuration."""
    memory_mb: int = 2048
    timeout_seconds: int = 28800  # 8 hours
    max_concurrency: int = 10
    auto_scaling_enabled: bool = True
    min_instances: int = 1
    max_instances: int = 5


@dataclass
class SecurityConfig:
    """Security configuration."""
    api_key_header: str = "X-API-Key"
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    cors_origins: list = None
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60


class EnvironmentManager:
    """Manages environment configuration and secrets."""
    
    def __init__(self, environment: Environment = None):
        self.environment = environment or self._detect_environment()
        self._config_cache = {}
    
    def _detect_environment(self) -> Environment:
        """Detect current environment from environment variables."""
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        try:
            return Environment(env_name)
        except ValueError:
            return Environment.DEVELOPMENT
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration for current environment."""
        if "database" not in self._config_cache:
            self._config_cache["database"] = DatabaseConfig(
                host=self._get_required_env("DB_HOST"),
                port=int(self._get_env("DB_PORT", "5432")),
                name=self._get_required_env("DB_NAME"),
                username=self._get_required_env("DB_USERNAME"),
                password=self._get_required_env("DB_PASSWORD"),
                ssl_enabled=self._get_env("DB_SSL_ENABLED", "true").lower() == "true"
            )
        return self._config_cache["database"]
    
    def get_aws_config(self) -> AWSConfig:
        """Get AWS configuration for current environment."""
        if "aws" not in self._config_cache:
            self._config_cache["aws"] = AWSConfig(
                region=self._get_env("AWS_REGION", "us-east-1"),
                bedrock_model_id=self._get_env(
                    "BEDROCK_MODEL_ID", 
                    "anthropic.claude-3-sonnet-20240229-v1:0"
                ),
                iam_role_arn=self._get_env("AWS_IAM_ROLE_ARN"),
                vpc_id=self._get_env("AWS_VPC_ID"),
                subnet_ids=self._parse_list(self._get_env("AWS_SUBNET_IDS")),
                security_group_ids=self._parse_list(self._get_env("AWS_SECURITY_GROUP_IDS"))
            )
        return self._config_cache["aws"]
    
    def get_agentcore_config(self) -> AgentCoreConfig:
        """Get AgentCore Runtime configuration."""
        if "agentcore" not in self._config_cache:
            self._config_cache["agentcore"] = AgentCoreConfig(
                memory_mb=int(self._get_env("AGENTCORE_MEMORY_MB", "2048")),
                timeout_seconds=int(self._get_env("AGENTCORE_TIMEOUT_SECONDS", "28800")),
                max_concurrency=int(self._get_env("MAX_CONCURRENT_SESSIONS", "10")),
                auto_scaling_enabled=self._get_env("AUTO_SCALING_ENABLED", "true").lower() == "true",
                min_instances=int(self._get_env("MIN_INSTANCES", "1")),
                max_instances=int(self._get_env("MAX_INSTANCES", "5"))
            )
        return self._config_cache["agentcore"]
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        if "security" not in self._config_cache:
            cors_origins = self._parse_list(self._get_env("CORS_ORIGINS"))
            if not cors_origins and self.environment == Environment.DEVELOPMENT:
                cors_origins = ["http://localhost:3000", "http://localhost:8080"]
            
            self._config_cache["security"] = SecurityConfig(
                api_key_header=self._get_env("API_KEY_HEADER", "X-API-Key"),
                jwt_secret_key=self._get_required_env("JWT_SECRET_KEY"),
                jwt_algorithm=self._get_env("JWT_ALGORITHM", "HS256"),
                jwt_expiration_hours=int(self._get_env("JWT_EXPIRATION_HOURS", "24")),
                cors_origins=cors_origins,
                rate_limit_requests=int(self._get_env("RATE_LIMIT_REQUESTS", "100")),
                rate_limit_window_seconds=int(self._get_env("RATE_LIMIT_WINDOW_SECONDS", "60"))
            )
        return self._config_cache["security"]
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        return {
            "environment": self.environment.value,
            "database": self.get_database_config(),
            "aws": self.get_aws_config(),
            "agentcore": self.get_agentcore_config(),
            "security": self.get_security_config()
        }
    
    def validate_config(self) -> Dict[str, list]:
        """Validate all configuration and return any errors."""
        errors = {}
        
        try:
            self.get_database_config()
        except Exception as e:
            errors["database"] = [str(e)]
        
        try:
            self.get_aws_config()
        except Exception as e:
            errors["aws"] = [str(e)]
        
        try:
            self.get_security_config()
        except Exception as e:
            errors["security"] = [str(e)]
        
        return errors
    
    def _get_env(self, key: str, default: str = None) -> str:
        """Get environment variable with optional default."""
        return os.getenv(key, default)
    
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable, raise error if missing."""
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def _parse_list(self, value: str) -> Optional[list]:
        """Parse comma-separated string into list."""
        if not value:
            return None
        return [item.strip() for item in value.split(",") if item.strip()]


# Global environment manager instance
env_manager = EnvironmentManager()


def get_config() -> Dict[str, Any]:
    """Get current environment configuration."""
    return env_manager.get_all_config()


def validate_environment() -> bool:
    """Validate current environment configuration."""
    errors = env_manager.validate_config()
    if errors:
        print("Configuration validation errors:")
        for section, section_errors in errors.items():
            print(f"  {section}:")
            for error in section_errors:
                print(f"    - {error}")
        return False
    return True