"""
Pytest configuration and fixtures for validation tests.

This module provides fixtures for testing the production readiness validation framework,
including mock API responses, AWS resources, test sessions, and documents.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock
import yaml
from pathlib import Path

from agent_scrivener.deployment.validation.models import (
    ValidationResult,
    ValidationStatus,
    PerformanceMetrics,
)


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def validation_config() -> Dict[str, Any]:
    """Load validation configuration from YAML file."""
    config_path = Path(__file__).parent.parent.parent / "agent_scrivener" / "deployment" / "validation" / "validation_config.yml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Return default config if file doesn't exist
    return {
        "timeouts": {
            "end_to_end_workflow": 300,
            "api_health_check": 5,
            "api_endpoint": 10,
        },
        "performance": {
            "api_health_check_max_ms": 100,
            "api_status_query_max_ms": 200,
            "workflow_max_seconds": 300,
        },
        "document_quality": {
            "min_word_count": 500,
            "min_source_count": 3,
        },
    }


# ============================================================================
# Test Research Query Fixtures
# ============================================================================

@pytest.fixture
def test_research_queries() -> List[str]:
    """Provide a list of test research queries for validation."""
    return [
        "What are the latest developments in quantum computing?",
        "Explain the impact of climate change on ocean ecosystems",
        "What are the key principles of microservices architecture?",
        "How does machine learning improve natural language processing?",
        "What are the benefits and challenges of renewable energy adoption?",
    ]


@pytest.fixture
def simple_test_query() -> str:
    """Provide a simple test query for quick validation."""
    return "What are the latest developments in quantum computing?"


@pytest.fixture
def complex_test_query() -> str:
    """Provide a complex test query for comprehensive validation."""
    return "Analyze the intersection of artificial intelligence, quantum computing, and biotechnology in modern healthcare systems, including ethical considerations and regulatory frameworks."


# ============================================================================
# Mock API Response Fixtures
# ============================================================================

@pytest.fixture
def mock_health_response() -> Dict[str, Any]:
    """Mock response for health endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime_seconds": 3600,
        "timestamp": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def mock_research_request_response() -> Dict[str, Any]:
    """Mock response for research request creation."""
    return {
        "session_id": "test_session_001",
        "status": "queued",
        "query": "What are the latest developments in quantum computing?",
        "created_at": datetime.utcnow().isoformat(),
        "estimated_completion_minutes": 3,
    }


@pytest.fixture
def mock_session_status_response() -> Dict[str, Any]:
    """Mock response for session status query."""
    return {
        "session_id": "test_session_001",
        "status": "in_progress",
        "progress": {
            "current_stage": "analysis",
            "completed_stages": ["research"],
            "remaining_stages": ["synthesis", "quality"],
            "percent_complete": 50,
        },
        "started_at": (datetime.utcnow() - timedelta(minutes=2)).isoformat(),
        "estimated_completion": (datetime.utcnow() + timedelta(minutes=1)).isoformat(),
    }


@pytest.fixture
def mock_session_result_response() -> Dict[str, Any]:
    """Mock response for completed session result."""
    return {
        "session_id": "test_session_001",
        "status": "completed",
        "document": {
            "title": "Latest Developments in Quantum Computing",
            "content": """
# Introduction

Quantum computing represents a paradigm shift in computational capabilities, leveraging quantum mechanical phenomena to solve complex problems.

# Analysis

Recent developments include advances in quantum error correction, increased qubit coherence times, and the development of quantum algorithms for practical applications.

# Synthesis

The integration of quantum computing with classical systems is creating hybrid architectures that maximize the strengths of both approaches.

# Conclusion

Quantum computing continues to evolve rapidly, with significant implications for cryptography, drug discovery, and optimization problems.
            """,
            "word_count": 523,
            "sources": [
                {
                    "url": "https://example.com/quantum-computing-2024",
                    "title": "Quantum Computing Advances 2024",
                    "author": "Dr. Jane Smith",
                },
                {
                    "url": "https://example.com/quantum-algorithms",
                    "title": "Practical Quantum Algorithms",
                    "author": "Prof. John Doe",
                },
                {
                    "url": "https://example.com/quantum-error-correction",
                    "title": "Quantum Error Correction Breakthroughs",
                    "author": "Dr. Alice Johnson",
                },
            ],
        },
        "completed_at": datetime.utcnow().isoformat(),
        "duration_seconds": 165,
    }


@pytest.fixture
def mock_session_list_response() -> Dict[str, Any]:
    """Mock response for session list query."""
    return {
        "sessions": [
            {
                "session_id": "test_session_001",
                "status": "completed",
                "query": "What are the latest developments in quantum computing?",
                "created_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
            },
            {
                "session_id": "test_session_002",
                "status": "in_progress",
                "query": "Explain climate change impacts",
                "created_at": (datetime.utcnow() - timedelta(minutes=30)).isoformat(),
            },
            {
                "session_id": "test_session_003",
                "status": "failed",
                "query": "Invalid query test",
                "created_at": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            },
        ],
        "total": 3,
        "page": 1,
        "page_size": 10,
    }


@pytest.fixture
def mock_websocket_messages() -> List[Dict[str, Any]]:
    """Mock WebSocket progress messages."""
    return [
        {
            "type": "status_update",
            "session_id": "test_session_001",
            "stage": "research",
            "message": "Starting research phase",
            "timestamp": datetime.utcnow().isoformat(),
        },
        {
            "type": "progress",
            "session_id": "test_session_001",
            "stage": "research",
            "percent_complete": 25,
            "timestamp": datetime.utcnow().isoformat(),
        },
        {
            "type": "status_update",
            "session_id": "test_session_001",
            "stage": "analysis",
            "message": "Starting analysis phase",
            "timestamp": datetime.utcnow().isoformat(),
        },
        {
            "type": "progress",
            "session_id": "test_session_001",
            "stage": "analysis",
            "percent_complete": 50,
            "timestamp": datetime.utcnow().isoformat(),
        },
        {
            "type": "completion",
            "session_id": "test_session_001",
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
        },
    ]


# ============================================================================
# Mock AWS Resource Fixtures
# ============================================================================

@pytest.fixture
def mock_aws_credentials() -> Dict[str, str]:
    """Mock AWS credentials for testing."""
    return {
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "region_name": "us-east-1",
    }


@pytest.fixture
def mock_sts_response() -> Dict[str, Any]:
    """Mock STS GetCallerIdentity response."""
    return {
        "UserId": "AIDAI23HXS2EXAMPLE",
        "Account": "123456789012",
        "Arn": "arn:aws:iam::123456789012:user/test-user",
    }


@pytest.fixture
def mock_vpc_response() -> Dict[str, Any]:
    """Mock VPC describe response."""
    return {
        "Vpcs": [
            {
                "VpcId": "vpc-12345678",
                "State": "available",
                "CidrBlock": "10.0.0.0/16",
                "Tags": [{"Key": "Name", "Value": "agent-scrivener-vpc"}],
            }
        ]
    }


@pytest.fixture
def mock_subnet_response() -> Dict[str, Any]:
    """Mock subnet describe response."""
    return {
        "Subnets": [
            {
                "SubnetId": "subnet-12345678",
                "VpcId": "vpc-12345678",
                "CidrBlock": "10.0.1.0/24",
                "AvailabilityZone": "us-east-1a",
                "Tags": [{"Key": "Type", "Value": "public"}],
            },
            {
                "SubnetId": "subnet-87654321",
                "VpcId": "vpc-12345678",
                "CidrBlock": "10.0.2.0/24",
                "AvailabilityZone": "us-east-1b",
                "Tags": [{"Key": "Type", "Value": "private"}],
            },
        ]
    }


@pytest.fixture
def mock_iam_role_response() -> Dict[str, Any]:
    """Mock IAM role describe response."""
    return {
        "Role": {
            "RoleName": "AgentScrivenerLambdaExecutionRole",
            "Arn": "arn:aws:iam::123456789012:role/AgentScrivenerLambdaExecutionRole",
            "AssumeRolePolicyDocument": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            },
        }
    }


@pytest.fixture
def mock_s3_bucket_response() -> Dict[str, Any]:
    """Mock S3 bucket describe response."""
    return {
        "Buckets": [
            {
                "Name": "agent-scrivener-documents",
                "CreationDate": datetime.utcnow(),
            },
            {
                "Name": "agent-scrivener-logs",
                "CreationDate": datetime.utcnow(),
            },
        ]
    }


@pytest.fixture
def mock_cloudwatch_log_groups() -> Dict[str, Any]:
    """Mock CloudWatch log groups response."""
    return {
        "logGroups": [
            {
                "logGroupName": "/aws/agent-scrivener/api",
                "creationTime": int(datetime.utcnow().timestamp() * 1000),
                "storedBytes": 1024000,
            },
            {
                "logGroupName": "/aws/agent-scrivener/orchestrator",
                "creationTime": int(datetime.utcnow().timestamp() * 1000),
                "storedBytes": 512000,
            },
            {
                "logGroupName": "/aws/agent-scrivener/agents",
                "creationTime": int(datetime.utcnow().timestamp() * 1000),
                "storedBytes": 2048000,
            },
        ]
    }


@pytest.fixture
def mock_cloudwatch_alarms() -> Dict[str, Any]:
    """Mock CloudWatch alarms response."""
    return {
        "MetricAlarms": [
            {
                "AlarmName": "HighErrorRate",
                "MetricName": "error_rate",
                "Threshold": 5.0,
                "ComparisonOperator": "GreaterThanThreshold",
                "EvaluationPeriods": 2,
                "AlarmActions": ["arn:aws:sns:us-east-1:123456789012:agent-scrivener-critical-alerts"],
            },
            {
                "AlarmName": "HighLatency",
                "MetricName": "request_latency",
                "Threshold": 300.0,
                "ComparisonOperator": "GreaterThanThreshold",
                "EvaluationPeriods": 2,
                "AlarmActions": ["arn:aws:sns:us-east-1:123456789012:agent-scrivener-critical-alerts"],
            },
        ]
    }


@pytest.fixture
def mock_bedrock_models() -> List[Dict[str, Any]]:
    """Mock Bedrock model list response."""
    return [
        {
            "modelId": "anthropic.claude-3-sonnet-20240229",
            "modelName": "Claude 3 Sonnet",
            "providerName": "Anthropic",
        },
        {
            "modelId": "anthropic.claude-3-haiku-20240307",
            "modelName": "Claude 3 Haiku",
            "providerName": "Anthropic",
        },
    ]


# ============================================================================
# Test Session and Document Fixtures
# ============================================================================

@pytest.fixture
def test_session_metadata() -> Dict[str, Any]:
    """Test session metadata for persistence validation."""
    return {
        "session_id": "test_session_001",
        "user_id": "test_user_001",
        "query": "What are the latest developments in quantum computing?",
        "status": "in_progress",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "estimated_completion": (datetime.utcnow() + timedelta(minutes=3)).isoformat(),
    }


@pytest.fixture
def test_intermediate_results() -> Dict[str, Any]:
    """Test intermediate results for persistence validation."""
    return {
        "session_id": "test_session_001",
        "stage": "research",
        "results": {
            "sources_found": 5,
            "sources": [
                {
                    "url": "https://example.com/quantum-1",
                    "title": "Quantum Computing Basics",
                    "relevance_score": 0.95,
                },
                {
                    "url": "https://example.com/quantum-2",
                    "title": "Recent Quantum Advances",
                    "relevance_score": 0.92,
                },
            ],
            "search_queries_used": [
                "quantum computing 2024",
                "quantum algorithms recent",
            ],
        },
        "completed_at": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def test_final_document() -> Dict[str, Any]:
    """Test final document for validation."""
    return {
        "session_id": "test_session_001",
        "title": "Latest Developments in Quantum Computing",
        "content": """
# Introduction

Quantum computing represents a revolutionary approach to computation that leverages quantum mechanical phenomena such as superposition and entanglement. This technology has the potential to solve complex problems that are intractable for classical computers.

# Analysis

Recent developments in quantum computing include several key breakthroughs:

1. **Quantum Error Correction**: Researchers have made significant progress in developing error correction codes that can protect quantum information from decoherence and noise.

2. **Increased Qubit Coherence**: New materials and techniques have extended the coherence times of qubits, allowing for longer and more complex quantum computations.

3. **Quantum Algorithms**: Novel algorithms have been developed for optimization, machine learning, and cryptography applications.

4. **Hybrid Quantum-Classical Systems**: Integration of quantum processors with classical computing infrastructure is enabling practical applications.

# Synthesis

The convergence of these developments is creating a new paradigm in computing. Quantum computers are transitioning from research curiosities to practical tools for solving real-world problems. Industries such as pharmaceuticals, finance, and materials science are beginning to explore quantum computing applications.

The hybrid approach, combining quantum and classical computing, appears to be the most promising path forward in the near term. This allows leveraging the strengths of both paradigms while mitigating current limitations of quantum hardware.

# Conclusion

Quantum computing is rapidly evolving from theoretical concepts to practical implementations. While significant challenges remain, particularly in scaling and error correction, the progress made in recent years suggests that quantum computing will play an increasingly important role in solving complex computational problems. Organizations should begin exploring quantum computing applications relevant to their domains to prepare for this technological shift.
        """,
        "word_count": 523,
        "sources": [
            {
                "url": "https://example.com/quantum-computing-2024",
                "title": "Quantum Computing Advances 2024",
                "author": "Dr. Jane Smith",
                "citation": "[1]",
            },
            {
                "url": "https://example.com/quantum-algorithms",
                "title": "Practical Quantum Algorithms",
                "author": "Prof. John Doe",
                "citation": "[2]",
            },
            {
                "url": "https://example.com/quantum-error-correction",
                "title": "Quantum Error Correction Breakthroughs",
                "author": "Dr. Alice Johnson",
                "citation": "[3]",
            },
        ],
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "duration_seconds": 165,
            "model_used": "anthropic.claude-3-sonnet-20240229",
        },
    }


@pytest.fixture
def test_session_history() -> List[Dict[str, Any]]:
    """Test session history with state transitions."""
    base_time = datetime.utcnow() - timedelta(minutes=5)
    
    return [
        {
            "session_id": "test_session_001",
            "status": "queued",
            "timestamp": base_time.isoformat(),
            "details": {"message": "Session created and queued"},
        },
        {
            "session_id": "test_session_001",
            "status": "in_progress",
            "stage": "research",
            "timestamp": (base_time + timedelta(seconds=10)).isoformat(),
            "details": {"message": "Starting research phase"},
        },
        {
            "session_id": "test_session_001",
            "status": "in_progress",
            "stage": "analysis",
            "timestamp": (base_time + timedelta(minutes=1)).isoformat(),
            "details": {"message": "Research complete, starting analysis"},
        },
        {
            "session_id": "test_session_001",
            "status": "in_progress",
            "stage": "synthesis",
            "timestamp": (base_time + timedelta(minutes=2)).isoformat(),
            "details": {"message": "Analysis complete, starting synthesis"},
        },
        {
            "session_id": "test_session_001",
            "status": "in_progress",
            "stage": "quality",
            "timestamp": (base_time + timedelta(minutes=3)).isoformat(),
            "details": {"message": "Synthesis complete, starting quality check"},
        },
        {
            "session_id": "test_session_001",
            "status": "completed",
            "timestamp": (base_time + timedelta(minutes=4)).isoformat(),
            "details": {"message": "Document generation complete"},
        },
    ]


# ============================================================================
# Validation Result Fixtures
# ============================================================================

@pytest.fixture
def sample_validation_result() -> ValidationResult:
    """Sample validation result for testing."""
    return ValidationResult(
        validator_name="TestValidator",
        status=ValidationStatus.PASS,
        message="Validation passed successfully",
        details={"test_key": "test_value"},
        duration_seconds=1.5,
        timestamp=datetime.utcnow(),
        remediation_steps=None,
    )


@pytest.fixture
def sample_failed_validation_result() -> ValidationResult:
    """Sample failed validation result for testing."""
    return ValidationResult(
        validator_name="TestValidator",
        status=ValidationStatus.FAIL,
        message="Validation failed",
        details={"error": "Test error", "expected": "value1", "actual": "value2"},
        duration_seconds=0.5,
        timestamp=datetime.utcnow(),
        remediation_steps=[
            "Check configuration file",
            "Verify environment variables",
            "Restart the service",
        ],
    )


@pytest.fixture
def sample_performance_metrics() -> PerformanceMetrics:
    """Sample performance metrics for testing."""
    return PerformanceMetrics(
        metric_name="api_response_time",
        p50_ms=50.0,
        p90_ms=95.0,
        p95_ms=120.0,
        p99_ms=180.0,
        min_ms=25.0,
        max_ms=250.0,
        mean_ms=75.0,
        std_dev_ms=35.0,
        sample_count=100,
    )


# ============================================================================
# Mock Client Fixtures
# ============================================================================

@pytest.fixture
def mock_httpx_client():
    """Mock httpx AsyncClient for API testing."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


@pytest.fixture
def mock_boto3_client():
    """Mock boto3 client for AWS testing."""
    mock_client = MagicMock()
    return mock_client


@pytest.fixture
def mock_database_connection():
    """Mock database connection for persistence testing."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.fetch = AsyncMock()
    mock_conn.fetchrow = AsyncMock()
    return mock_conn


# ============================================================================
# Agent and Orchestration Fixtures
# ============================================================================

@pytest.fixture
def mock_agent_registry():
    """Mock agent registry with all required agents."""
    registry = MagicMock()
    registry.get_agent = MagicMock()
    registry.list_agents = MagicMock(return_value=[
        "research_agent",
        "analysis_agent",
        "synthesis_agent",
        "quality_agent",
    ])
    return registry


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator for testing."""
    orchestrator = AsyncMock()
    orchestrator.start_session = AsyncMock(return_value="test_session_001")
    orchestrator.get_session_status = AsyncMock(return_value={"status": "in_progress"})
    orchestrator.get_session_result = AsyncMock(return_value={"status": "completed"})
    return orchestrator
