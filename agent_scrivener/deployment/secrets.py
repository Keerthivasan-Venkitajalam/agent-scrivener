"""Secrets management for Agent Scrivener deployment."""

import os
import json
import boto3
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SecretProvider(Enum):
    """Secret provider types."""
    ENVIRONMENT = "environment"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    AWS_PARAMETER_STORE = "aws_parameter_store"
    KUBERNETES = "kubernetes"


@dataclass
class SecretConfig:
    """Secret configuration."""
    name: str
    provider: SecretProvider
    path: str
    required: bool = True
    cache_ttl: int = 300  # 5 minutes


class SecretsManager:
    """Manages secrets from various providers."""
    
    def __init__(self, aws_region: str = None):
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-east-1")
        self._cache = {}
        self._clients = {}
        
        # Define secret configurations
        self.secret_configs = {
            "jwt_secret": SecretConfig(
                name="jwt_secret",
                provider=SecretProvider.AWS_SECRETS_MANAGER,
                path="agent-scrivener/jwt-secret"
            ),
            "database_password": SecretConfig(
                name="database_password",
                provider=SecretProvider.AWS_SECRETS_MANAGER,
                path="agent-scrivener/database-credentials"
            ),
            "api_keys": SecretConfig(
                name="api_keys",
                provider=SecretProvider.AWS_SECRETS_MANAGER,
                path="agent-scrivener/external-api-keys"
            ),
            "bedrock_credentials": SecretConfig(
                name="bedrock_credentials",
                provider=SecretProvider.AWS_PARAMETER_STORE,
                path="/agent-scrivener/bedrock/credentials"
            )
        }
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get a secret value by name."""
        if secret_name not in self.secret_configs:
            logger.warning(f"Unknown secret: {secret_name}")
            return None
        
        config = self.secret_configs[secret_name]
        
        # Check cache first
        cache_key = f"{config.provider.value}:{config.path}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            value = self._fetch_secret(config)
            if value:
                self._cache[cache_key] = value
            return value
        except Exception as e:
            logger.error(f"Failed to fetch secret {secret_name}: {e}")
            if config.required:
                raise
            return None
    
    def get_secret_dict(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """Get a secret as a dictionary (for JSON secrets)."""
        value = self.get_secret(secret_name)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse secret {secret_name} as JSON: {e}")
        return None
    
    def _fetch_secret(self, config: SecretConfig) -> Optional[str]:
        """Fetch secret from the configured provider."""
        if config.provider == SecretProvider.ENVIRONMENT:
            return self._fetch_from_environment(config.path)
        elif config.provider == SecretProvider.AWS_SECRETS_MANAGER:
            return self._fetch_from_secrets_manager(config.path)
        elif config.provider == SecretProvider.AWS_PARAMETER_STORE:
            return self._fetch_from_parameter_store(config.path)
        elif config.provider == SecretProvider.KUBERNETES:
            return self._fetch_from_kubernetes(config.path)
        else:
            raise ValueError(f"Unsupported secret provider: {config.provider}")
    
    def _fetch_from_environment(self, path: str) -> Optional[str]:
        """Fetch secret from environment variables."""
        return os.getenv(path)
    
    def _fetch_from_secrets_manager(self, path: str) -> Optional[str]:
        """Fetch secret from AWS Secrets Manager."""
        client = self._get_secrets_manager_client()
        try:
            response = client.get_secret_value(SecretId=path)
            return response.get("SecretString")
        except client.exceptions.ResourceNotFoundException:
            logger.warning(f"Secret not found in Secrets Manager: {path}")
            return None
        except Exception as e:
            logger.error(f"Error fetching from Secrets Manager: {e}")
            raise
    
    def _fetch_from_parameter_store(self, path: str) -> Optional[str]:
        """Fetch secret from AWS Systems Manager Parameter Store."""
        client = self._get_ssm_client()
        try:
            response = client.get_parameter(Name=path, WithDecryption=True)
            return response["Parameter"]["Value"]
        except client.exceptions.ParameterNotFound:
            logger.warning(f"Parameter not found in Parameter Store: {path}")
            return None
        except Exception as e:
            logger.error(f"Error fetching from Parameter Store: {e}")
            raise
    
    def _fetch_from_kubernetes(self, path: str) -> Optional[str]:
        """Fetch secret from Kubernetes secret mount."""
        try:
            with open(f"/var/secrets/{path}", "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning(f"Kubernetes secret file not found: {path}")
            return None
        except Exception as e:
            logger.error(f"Error reading Kubernetes secret: {e}")
            raise
    
    def _get_secrets_manager_client(self):
        """Get AWS Secrets Manager client."""
        if "secrets_manager" not in self._clients:
            self._clients["secrets_manager"] = boto3.client(
                "secretsmanager",
                region_name=self.aws_region
            )
        return self._clients["secrets_manager"]
    
    def _get_ssm_client(self):
        """Get AWS Systems Manager client."""
        if "ssm" not in self._clients:
            self._clients["ssm"] = boto3.client(
                "ssm",
                region_name=self.aws_region
            )
        return self._clients["ssm"]
    
    def validate_secrets(self) -> Dict[str, bool]:
        """Validate that all required secrets are accessible."""
        results = {}
        for name, config in self.secret_configs.items():
            if config.required:
                try:
                    value = self.get_secret(name)
                    results[name] = value is not None
                except Exception:
                    results[name] = False
            else:
                results[name] = True
        return results
    
    def clear_cache(self):
        """Clear the secrets cache."""
        self._cache.clear()


# Global secrets manager instance
secrets_manager = SecretsManager()


def get_secret(name: str) -> Optional[str]:
    """Get a secret value."""
    return secrets_manager.get_secret(name)


def get_secret_dict(name: str) -> Optional[Dict[str, Any]]:
    """Get a secret as a dictionary."""
    return secrets_manager.get_secret_dict(name)


def validate_secrets() -> bool:
    """Validate all required secrets are accessible."""
    results = secrets_manager.validate_secrets()
    missing_secrets = [name for name, valid in results.items() if not valid]
    
    if missing_secrets:
        logger.error(f"Missing required secrets: {missing_secrets}")
        return False
    
    logger.info("All required secrets validated successfully")
    return True