"""
Configuration management for Agent Scrivener.
"""

import os
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from pathlib import Path


class AgentCoreConfig(BaseModel):
    """Configuration for AgentCore integration."""
    runtime_endpoint: str = Field(default="https://bedrock-agent-runtime.us-east-1.amazonaws.com")
    region: str = Field(default="us-east-1")
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    max_retries: int = Field(default=3, ge=0, le=10)


class DatabaseConfig(BaseModel):
    """Configuration for external database APIs."""
    arxiv_base_url: str = Field(default="http://export.arxiv.org/api/query")
    pubmed_base_url: str = Field(default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
    semantic_scholar_base_url: str = Field(default="https://api.semanticscholar.org/graph/v1")
    rate_limit_requests_per_minute: int = Field(default=60, ge=1, le=1000)


class ProcessingConfig(BaseModel):
    """Configuration for content processing."""
    max_content_length: int = Field(default=50000, ge=1000, le=200000)
    max_sources_per_query: int = Field(default=20, ge=1, le=100)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    analysis_timeout_seconds: int = Field(default=600, ge=60, le=3600)


class SystemConfig(BaseModel):
    """Main system configuration."""
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    json_logging: bool = Field(default=False)
    max_concurrent_sessions: int = Field(default=10, ge=1, le=100)
    session_timeout_minutes: int = Field(default=60, ge=5, le=480)
    
    # Component configurations
    agentcore: AgentCoreConfig = Field(default_factory=AgentCoreConfig)
    databases: DatabaseConfig = Field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)


def load_config_from_env() -> SystemConfig:
    """
    Load configuration from environment variables.
    
    Returns:
        SystemConfig: Configuration object with values from environment
    """
    config_data = {}
    
    # System settings
    if os.getenv("DEBUG"):
        config_data["debug"] = os.getenv("DEBUG").lower() == "true"
    
    if os.getenv("LOG_LEVEL"):
        config_data["log_level"] = os.getenv("LOG_LEVEL")
    
    if os.getenv("JSON_LOGGING"):
        config_data["json_logging"] = os.getenv("JSON_LOGGING").lower() == "true"
    
    if os.getenv("MAX_CONCURRENT_SESSIONS"):
        config_data["max_concurrent_sessions"] = int(os.getenv("MAX_CONCURRENT_SESSIONS"))
    
    # AgentCore settings
    agentcore_config = {}
    if os.getenv("AGENTCORE_RUNTIME_ENDPOINT"):
        agentcore_config["runtime_endpoint"] = os.getenv("AGENTCORE_RUNTIME_ENDPOINT")
    
    if os.getenv("AWS_REGION"):
        agentcore_config["region"] = os.getenv("AWS_REGION")
    
    if os.getenv("AGENTCORE_TIMEOUT"):
        agentcore_config["timeout_seconds"] = int(os.getenv("AGENTCORE_TIMEOUT"))
    
    if agentcore_config:
        config_data["agentcore"] = agentcore_config
    
    # Database settings
    database_config = {}
    if os.getenv("ARXIV_BASE_URL"):
        database_config["arxiv_base_url"] = os.getenv("ARXIV_BASE_URL")
    
    if os.getenv("PUBMED_BASE_URL"):
        database_config["pubmed_base_url"] = os.getenv("PUBMED_BASE_URL")
    
    if os.getenv("RATE_LIMIT_RPM"):
        database_config["rate_limit_requests_per_minute"] = int(os.getenv("RATE_LIMIT_RPM"))
    
    if database_config:
        config_data["databases"] = database_config
    
    # Processing settings
    processing_config = {}
    if os.getenv("MAX_CONTENT_LENGTH"):
        processing_config["max_content_length"] = int(os.getenv("MAX_CONTENT_LENGTH"))
    
    if os.getenv("MAX_SOURCES_PER_QUERY"):
        processing_config["max_sources_per_query"] = int(os.getenv("MAX_SOURCES_PER_QUERY"))
    
    if os.getenv("CONFIDENCE_THRESHOLD"):
        processing_config["confidence_threshold"] = float(os.getenv("CONFIDENCE_THRESHOLD"))
    
    if processing_config:
        config_data["processing"] = processing_config
    
    return SystemConfig(**config_data)


def load_config_from_file(config_path: Optional[Path] = None) -> SystemConfig:
    """
    Load configuration from a JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        SystemConfig: Configuration object
    """
    if config_path is None:
        config_path = Path("config.json")
    
    if not config_path.exists():
        return SystemConfig()
    
    import json
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return SystemConfig(**config_data)
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return SystemConfig()


# Global configuration instance
_config: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """
    Get the global configuration instance.
    
    Returns:
        SystemConfig: Global configuration
    """
    global _config
    if _config is None:
        # Try to load from file first, then environment
        _config = load_config_from_file()
        env_config = load_config_from_env()
        
        # Merge environment variables over file config
        if env_config.dict(exclude_unset=True):
            config_dict = _config.dict()
            config_dict.update(env_config.dict(exclude_unset=True))
            _config = SystemConfig(**config_dict)
    
    return _config


def set_config(config: SystemConfig) -> None:
    """
    Set the global configuration instance.
    
    Args:
        config: Configuration to set as global
    """
    global _config
    _config = config