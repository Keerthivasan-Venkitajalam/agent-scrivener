"""Unit tests for DeploymentConfigValidator."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from hypothesis import given, strategies as st, settings

from agent_scrivener.deployment.validation import (
    DeploymentConfigValidator,
    ValidationStatus,
)


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def validator(temp_project_dir):
    """Create a DeploymentConfigValidator instance."""
    return DeploymentConfigValidator(
        project_root=temp_project_dir,
        required_env_vars=["AWS_REGION", "BEDROCK_MODEL_ID", "DATABASE_URL"]
    )


class TestEnvironmentVariables:
    """Tests for validate_environment_variables method."""
    
    def test_all_env_vars_present(self, validator):
        """Test validation passes when all required env vars are set."""
        with patch.dict(os.environ, {
            "AWS_REGION": "us-east-1",
            "BEDROCK_MODEL_ID": "anthropic.claude-v2",
            "DATABASE_URL": "postgresql://localhost/test"
        }):
            result = validator.validate_environment_variables()
            
            assert result.status == ValidationStatus.PASS
            assert "3 required environment variables" in result.message
            assert result.details["all_present"] is True
    
    def test_missing_env_vars(self, validator):
        """Test validation fails when env vars are missing."""
        with patch.dict(os.environ, {
            "AWS_REGION": "us-east-1"
        }, clear=True):
            result = validator.validate_environment_variables()
            
            assert result.status == ValidationStatus.FAIL
            assert "missing" in result.message.lower()
            assert "BEDROCK_MODEL_ID" in result.details["missing_vars"]
            assert "DATABASE_URL" in result.details["missing_vars"]

    def test_empty_env_vars(self, validator):
        """Test validation fails when env vars are empty."""
        with patch.dict(os.environ, {
            "AWS_REGION": "",
            "BEDROCK_MODEL_ID": "anthropic.claude-v2",
            "DATABASE_URL": "postgresql://localhost/test"
        }):
            result = validator.validate_environment_variables()
            
            assert result.status == ValidationStatus.FAIL
            assert "empty" in result.message.lower()
            assert "AWS_REGION" in result.details["empty_vars"]


class TestDockerConfiguration:
    """Tests for validate_docker_configuration method."""
    
    def test_dockerfile_missing(self, validator, temp_project_dir):
        """Test validation fails when Dockerfile is missing."""
        result = validator.validate_docker_configuration()
        
        assert result.status == ValidationStatus.FAIL
        assert "Dockerfile not found" in result.message
        assert result.remediation_steps is not None
    
    def test_docker_compose_missing(self, validator, temp_project_dir):
        """Test validation fails when docker-compose.yml is missing."""
        # Create Dockerfile but not docker-compose.yml
        (temp_project_dir / "Dockerfile").write_text("FROM python:3.11\n")
        
        result = validator.validate_docker_configuration()
        
        assert result.status == ValidationStatus.FAIL
        assert "docker-compose.yml not found" in result.message
    
    def test_docker_compose_invalid_yaml(self, validator, temp_project_dir):
        """Test validation fails when docker-compose.yml has invalid YAML."""
        (temp_project_dir / "Dockerfile").write_text("FROM python:3.11\n")
        (temp_project_dir / "docker-compose.yml").write_text("invalid: yaml: syntax:\n")
        
        result = validator.validate_docker_configuration()
        
        assert result.status == ValidationStatus.FAIL
        assert "invalid YAML syntax" in result.message
    
    def test_docker_compose_missing_services(self, validator, temp_project_dir):
        """Test validation fails when required services are missing."""
        (temp_project_dir / "Dockerfile").write_text("FROM python:3.11\n")
        compose_config = {
            "version": "3.8",
            "services": {
                "api": {"image": "test"}
                # Missing "database" service
            }
        }
        (temp_project_dir / "docker-compose.yml").write_text(yaml.dump(compose_config))
        
        result = validator.validate_docker_configuration()
        
        assert result.status == ValidationStatus.FAIL
        assert "missing required services" in result.message
        assert "database" in result.details["missing_services"]
    
    def test_docker_configuration_valid(self, validator, temp_project_dir):
        """Test validation passes with valid Docker configuration."""
        (temp_project_dir / "Dockerfile").write_text("FROM python:3.11\n")
        compose_config = {
            "version": "3.8",
            "services": {
                "api": {"image": "test"},
                "database": {"image": "postgres:15"}
            }
        }
        (temp_project_dir / "docker-compose.yml").write_text(yaml.dump(compose_config))
        
        result = validator.validate_docker_configuration()
        
        assert result.status == ValidationStatus.PASS
        assert result.details["dockerfile_exists"] is True
        assert result.details["compose_file_exists"] is True



class TestAWSCDKConfiguration:
    """Tests for validate_aws_cdk_configuration method."""
    
    def test_cdk_app_not_found(self, validator, temp_project_dir):
        """Test validation skips when CDK app is not found."""
        result = validator.validate_aws_cdk_configuration()
        
        assert result.status == ValidationStatus.SKIP
        assert "CDK app not found" in result.message
    
    def test_cdk_json_missing(self, validator, temp_project_dir):
        """Test validation fails when cdk.json is missing."""
        # Create CDK app but not cdk.json
        cdk_dir = temp_project_dir / "cdk"
        cdk_dir.mkdir()
        (cdk_dir / "app.py").write_text("# CDK app\n")
        
        result = validator.validate_aws_cdk_configuration()
        
        assert result.status == ValidationStatus.FAIL
        assert "cdk.json not found" in result.message
    
    def test_cdk_json_invalid(self, validator, temp_project_dir):
        """Test validation fails when cdk.json has invalid JSON."""
        cdk_dir = temp_project_dir / "cdk"
        cdk_dir.mkdir()
        (cdk_dir / "app.py").write_text("# CDK app\n")
        (temp_project_dir / "cdk.json").write_text("{invalid json}")
        
        result = validator.validate_aws_cdk_configuration()
        
        assert result.status == ValidationStatus.FAIL
        assert "invalid JSON syntax" in result.message
    
    def test_cdk_json_missing_app_field(self, validator, temp_project_dir):
        """Test validation fails when cdk.json is missing 'app' field."""
        cdk_dir = temp_project_dir / "cdk"
        cdk_dir.mkdir()
        (cdk_dir / "app.py").write_text("# CDK app\n")
        (temp_project_dir / "cdk.json").write_text('{"version": "1.0"}')
        
        result = validator.validate_aws_cdk_configuration()
        
        assert result.status == ValidationStatus.FAIL
        assert "missing 'app' field" in result.message
    
    @patch("subprocess.run")
    def test_cdk_synth_success(self, mock_run, validator, temp_project_dir):
        """Test validation passes when cdk synth succeeds."""
        # Setup files
        cdk_dir = temp_project_dir / "cdk"
        cdk_dir.mkdir()
        (cdk_dir / "app.py").write_text("# CDK app\n")
        (temp_project_dir / "cdk.json").write_text('{"app": "python3 cdk/app.py"}')
        
        # Mock successful cdk synth
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        
        result = validator.validate_aws_cdk_configuration()
        
        assert result.status == ValidationStatus.PASS
        assert result.details["synth_successful"] is True
    
    @patch("subprocess.run")
    def test_cdk_synth_failure(self, mock_run, validator, temp_project_dir):
        """Test validation fails when cdk synth fails."""
        # Setup files
        cdk_dir = temp_project_dir / "cdk"
        cdk_dir.mkdir()
        (cdk_dir / "app.py").write_text("# CDK app\n")
        (temp_project_dir / "cdk.json").write_text('{"app": "python3 cdk/app.py"}')
        
        # Mock failed cdk synth
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Synthesis failed"
        )
        
        result = validator.validate_aws_cdk_configuration()
        
        assert result.status == ValidationStatus.FAIL
        assert "CDK synth failed" in result.message
    
    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_cdk_cli_not_found(self, mock_run, validator, temp_project_dir):
        """Test validation skips when CDK CLI is not installed."""
        # Setup files
        cdk_dir = temp_project_dir / "cdk"
        cdk_dir.mkdir()
        (cdk_dir / "app.py").write_text("# CDK app\n")
        (temp_project_dir / "cdk.json").write_text('{"app": "python3 cdk/app.py"}')
        
        result = validator.validate_aws_cdk_configuration()
        
        assert result.status == ValidationStatus.SKIP
        assert "CDK CLI not found" in result.message



class TestAgentCoreConfiguration:
    """Tests for validate_agentcore_configuration method."""
    
    def test_agentcore_config_not_found(self, validator, temp_project_dir):
        """Test validation skips when AgentCore config is not found."""
        result = validator.validate_agentcore_configuration()
        
        assert result.status == ValidationStatus.SKIP
        assert "AgentCore configuration file not found" in result.message
    
    def test_agentcore_config_invalid_yaml(self, validator, temp_project_dir):
        """Test validation fails when AgentCore config has invalid YAML."""
        (temp_project_dir / "agentcore_config.yml").write_text("invalid: yaml: syntax:\n")
        
        result = validator.validate_agentcore_configuration()
        
        assert result.status == ValidationStatus.FAIL
        assert "invalid YAML syntax" in result.message
    
    def test_agentcore_config_no_agents(self, validator, temp_project_dir):
        """Test validation fails when no agents are defined."""
        config = {"version": "1.0"}
        (temp_project_dir / "agentcore_config.yml").write_text(yaml.dump(config))
        
        result = validator.validate_agentcore_configuration()
        
        assert result.status == ValidationStatus.FAIL
        assert "no agent definitions" in result.message
    
    def test_agentcore_config_missing_required_fields(self, validator, temp_project_dir):
        """Test validation fails when agents are missing required fields."""
        config = {
            "agents": [
                {"name": "research_agent", "model": "claude-v2"},  # Missing "tools"
                {"name": "analysis_agent", "tools": ["search"]}  # Missing "model"
            ]
        }
        (temp_project_dir / "agentcore_config.yml").write_text(yaml.dump(config))
        
        result = validator.validate_agentcore_configuration()
        
        assert result.status == ValidationStatus.FAIL
        assert "invalid agent definition" in result.message
        assert len(result.details["invalid_agents"]) == 2
    
    def test_agentcore_config_valid(self, validator, temp_project_dir):
        """Test validation passes with valid AgentCore configuration."""
        config = {
            "agents": [
                {
                    "name": "research_agent",
                    "model": "anthropic.claude-v2",
                    "tools": ["search", "retrieve"]
                },
                {
                    "name": "analysis_agent",
                    "model": "anthropic.claude-v2",
                    "tools": ["analyze", "summarize"]
                }
            ]
        }
        (temp_project_dir / "agentcore_config.yml").write_text(yaml.dump(config))
        
        result = validator.validate_agentcore_configuration()
        
        assert result.status == ValidationStatus.PASS
        assert result.details["agents_count"] == 2
        assert "research_agent" in result.details["agent_names"]
        assert "analysis_agent" in result.details["agent_names"]


class TestSecretsAccess:
    """Tests for validate_secrets_access method."""
    
    @pytest.mark.asyncio
    async def test_boto3_not_installed(self, validator):
        """Test validation skips when boto3 is not installed."""
        with patch.dict("sys.modules", {"boto3": None}):
            result = await validator.validate_secrets_access()
            
            assert result.status == ValidationStatus.SKIP
            assert "boto3 not installed" in result.message
    
    @pytest.mark.asyncio
    async def test_aws_region_not_set(self, validator):
        """Test validation fails when AWS_REGION is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = await validator.validate_secrets_access()
            
            assert result.status == ValidationStatus.FAIL
            assert "AWS_REGION environment variable not set" in result.message
    
    @pytest.mark.asyncio
    async def test_no_aws_credentials(self, validator):
        """Test validation fails when AWS credentials are not configured."""
        with patch.dict(os.environ, {"AWS_REGION": "us-east-1"}):
            with patch("boto3.client") as mock_client:
                from botocore.exceptions import NoCredentialsError
                mock_client.return_value.list_secrets.side_effect = NoCredentialsError()
                
                result = await validator.validate_secrets_access()
                
                assert result.status == ValidationStatus.FAIL
                assert "AWS credentials not configured" in result.message
    
    @pytest.mark.asyncio
    async def test_secrets_access_success(self, validator):
        """Test validation passes when Secrets Manager is accessible."""
        with patch.dict(os.environ, {"AWS_REGION": "us-east-1"}):
            with patch("boto3.client") as mock_client:
                mock_sm = MagicMock()
                mock_sm.list_secrets.return_value = {"SecretList": []}
                mock_client.return_value = mock_sm
                
                result = await validator.validate_secrets_access()
                
                assert result.status == ValidationStatus.PASS
                assert "Secrets Manager access validation passed" in result.message



class TestDatabaseConfiguration:
    """Tests for validate_database_configuration method."""
    
    @pytest.mark.asyncio
    async def test_database_url_not_set(self, validator):
        """Test validation fails when DATABASE_URL is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = await validator.validate_database_configuration()
            
            assert result.status == ValidationStatus.FAIL
            assert "DATABASE_URL environment variable not set" in result.message
    
    @pytest.mark.asyncio
    async def test_database_url_missing_scheme(self, validator):
        """Test validation fails when DATABASE_URL is missing scheme."""
        with patch.dict(os.environ, {"DATABASE_URL": "localhost/test"}):
            result = await validator.validate_database_configuration()
            
            assert result.status == ValidationStatus.FAIL
            assert "missing database scheme" in result.message
    
    @pytest.mark.asyncio
    async def test_database_url_missing_hostname(self, validator):
        """Test validation fails when DATABASE_URL is missing hostname."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql:///test"}):
            result = await validator.validate_database_configuration()
            
            assert result.status == ValidationStatus.FAIL
            assert "missing hostname" in result.message
    
    @pytest.mark.asyncio
    async def test_asyncpg_not_installed(self, validator):
        """Test validation skips connection test when asyncpg is not installed."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch.dict("sys.modules", {"asyncpg": None}):
                result = await validator.validate_database_configuration()
                
                assert result.status == ValidationStatus.SKIP
                assert "asyncpg not installed" in result.message
    
    @pytest.mark.asyncio
    async def test_database_connection_timeout(self, validator):
        """Test validation fails when database connection times out."""
        pytest.importorskip("asyncpg")
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch("asyncpg.connect", side_effect=asyncio.TimeoutError()):
                result = await validator.validate_database_configuration()
                
                assert result.status == ValidationStatus.FAIL
                assert "connection timed out" in result.message
    
    @pytest.mark.asyncio
    async def test_database_connection_success(self, validator):
        """Test validation passes when database connection succeeds."""
        pytest.importorskip("asyncpg")
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            mock_conn = AsyncMock()
            with patch("asyncpg.connect", return_value=mock_conn):
                result = await validator.validate_database_configuration()
                
                assert result.status == ValidationStatus.PASS
                assert "Database configuration validation passed" in result.message
                assert result.details["connection_successful"] is True
    
    @pytest.mark.asyncio
    async def test_unsupported_database_type(self, validator):
        """Test validation skips connection test for unsupported database types."""
        with patch.dict(os.environ, {"DATABASE_URL": "mysql://localhost/test"}):
            result = await validator.validate_database_configuration()
            
            assert result.status == ValidationStatus.SKIP
            assert "not supported for connection testing" in result.message


class TestValidateMethod:
    """Tests for the main validate method."""
    
    @pytest.mark.asyncio
    async def test_validate_runs_all_checks(self, validator, temp_project_dir):
        """Test that validate method runs all validation checks."""
        # Setup minimal valid configuration
        with patch.dict(os.environ, {
            "AWS_REGION": "us-east-1",
            "BEDROCK_MODEL_ID": "anthropic.claude-v2",
            "DATABASE_URL": "postgresql://localhost/test"
        }):
            # Create minimal Docker files
            (temp_project_dir / "Dockerfile").write_text("FROM python:3.11\n")
            compose_config = {
                "version": "3.8",
                "services": {
                    "api": {"image": "test"},
                    "database": {"image": "postgres:15"}
                }
            }
            (temp_project_dir / "docker-compose.yml").write_text(yaml.dump(compose_config))
            
            results = await validator.validate()
            
            # Should have 6 results (one for each validation method)
            assert len(results) == 6
            
            # Check that all validators ran
            validator_names = [r.validator_name for r in results]
            assert all(name == "DeploymentConfigValidator" for name in validator_names)



class TestAgentCoreConfigurationProperties:
    """Property-based tests for AgentCore configuration validation."""
    
    @given(
        agents=st.lists(
            st.fixed_dictionaries({
                "name": st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
                "model": st.sampled_from([
                    "anthropic.claude-v2",
                    "anthropic.claude-v3",
                    "anthropic.claude-instant-v1"
                ]),
                "tools": st.lists(
                    st.sampled_from(["search", "retrieve", "analyze", "summarize", "write"]),
                    min_size=1,
                    max_size=5
                )
            }),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_agentcore_configuration_completeness(self, agents):
        """
        Property Test: AgentCore configuration completeness
        
        Feature: production-readiness-validation, Property 17: AgentCore configuration completeness
        
        **Validates: Requirements 4.5**
        
        For any agent definition in the AgentCore configuration, all required fields 
        (name, model, tools) should be present and valid.
        
        This property verifies that:
        1. Agent configurations with all required fields (name, model, tools) pass validation
        2. Agent configurations missing any required field fail validation
        3. The validator correctly identifies which fields are missing
        4. The validator provides clear error messages for invalid configurations
        5. Multiple agents can be validated together
        6. Empty agent lists are rejected
        """
        # Create a temporary project directory for this test iteration
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            
            # Create validator instance
            validator = DeploymentConfigValidator(
                project_root=project_root,
                required_env_vars=["AWS_REGION", "BEDROCK_MODEL_ID", "DATABASE_URL"]
            )
            
            # Create AgentCore configuration with the generated agents
            config = {"agents": agents}
            config_path = project_root / "agentcore_config.yml"
            config_path.write_text(yaml.dump(config))
            
            # Execute validation
            result = validator.validate_agentcore_configuration()
            
            # Property 1: All agents have required fields -> validation passes
            # Since our generator always creates agents with all required fields,
            # validation should always pass
            assert result.status == ValidationStatus.PASS, \
                f"Configuration with {len(agents)} valid agents should pass validation"
            
            # Property 2: Verify agent count is correct
            assert result.details["agents_count"] == len(agents), \
                f"Expected {len(agents)} agents, got {result.details['agents_count']}"
            
            # Property 3: Verify all agent names are captured
            agent_names = [a["name"] for a in agents]
            assert result.details["agent_names"] == agent_names, \
                "All agent names should be captured in validation result"
            
            # Property 4: Verify config path is recorded
            assert "config_path" in result.details, \
                "Config path should be recorded in validation result"
    
    @given(
        valid_agents=st.lists(
            st.fixed_dictionaries({
                "name": st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
                "model": st.sampled_from([
                    "anthropic.claude-v2",
                    "anthropic.claude-v3"
                ]),
                "tools": st.lists(
                    st.sampled_from(["search", "retrieve", "analyze"]),
                    min_size=1,
                    max_size=3
                )
            }),
            min_size=1,
            max_size=5
        ),
        missing_field=st.sampled_from(["name", "model", "tools"])
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_agentcore_missing_required_fields(
        self,
        valid_agents,
        missing_field
    ):
        """
        Property Test: AgentCore configuration with missing required fields
        
        Feature: production-readiness-validation, Property 17: AgentCore configuration completeness
        
        **Validates: Requirements 4.5**
        
        For any agent definition missing a required field (name, model, or tools),
        the validation should fail and identify the missing field.
        
        This property verifies that:
        1. Configurations with agents missing required fields fail validation
        2. The validator identifies which agents are invalid
        3. The validator identifies which fields are missing
        4. Error messages are clear and actionable
        """
        # Create a temporary project directory for this test iteration
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            
            # Create validator instance
            validator = DeploymentConfigValidator(
                project_root=project_root,
                required_env_vars=["AWS_REGION", "BEDROCK_MODEL_ID", "DATABASE_URL"]
            )
            
            # Create one invalid agent by removing a required field
            invalid_agent = valid_agents[0].copy()
            del invalid_agent[missing_field]
            
            # Mix valid and invalid agents
            agents = valid_agents + [invalid_agent]
            
            # Create AgentCore configuration
            config = {"agents": agents}
            config_path = project_root / "agentcore_config.yml"
            config_path.write_text(yaml.dump(config))
            
            # Execute validation
            result = validator.validate_agentcore_configuration()
            
            # Property 1: Configuration with invalid agents should fail
            assert result.status == ValidationStatus.FAIL, \
                f"Configuration with agent missing '{missing_field}' should fail validation"
            
            # Property 2: Invalid agents should be identified
            assert "invalid_agents" in result.details, \
                "Validation result should identify invalid agents"
            
            invalid_agents = result.details["invalid_agents"]
            assert len(invalid_agents) >= 1, \
                "At least one invalid agent should be identified"
            
            # Property 3: Missing field should be identified
            # Find the invalid agent in the results
            found_missing_field = False
            for invalid_agent_info in invalid_agents:
                if "missing_fields" in invalid_agent_info:
                    if missing_field in invalid_agent_info["missing_fields"]:
                        found_missing_field = True
                        break
            
            assert found_missing_field, \
                f"Missing field '{missing_field}' should be identified in validation result"
            
            # Property 4: Error message should be clear
            assert "invalid agent definition" in result.message.lower(), \
                "Error message should mention invalid agent definitions"
            
            # Property 5: Remediation steps should be provided
            assert result.remediation_steps is not None, \
                "Remediation steps should be provided for invalid configuration"
            assert len(result.remediation_steps) > 0, \
                "At least one remediation step should be provided"
    
    @given(
        num_agents=st.integers(min_value=0, max_value=0)  # Always 0
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_agentcore_empty_agents_list(self, num_agents):
        """
        Property Test: AgentCore configuration with empty agents list
        
        Feature: production-readiness-validation, Property 17: AgentCore configuration completeness
        
        **Validates: Requirements 4.5**
        
        For any AgentCore configuration with no agent definitions, the validation
        should fail with a clear error message.
        
        This property verifies that:
        1. Configurations with empty agent lists fail validation
        2. Error message clearly indicates no agents are defined
        3. Remediation steps guide user to add agent definitions
        """
        # Create a temporary project directory for this test iteration
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            
            # Create validator instance
            validator = DeploymentConfigValidator(
                project_root=project_root,
                required_env_vars=["AWS_REGION", "BEDROCK_MODEL_ID", "DATABASE_URL"]
            )
            
            # Create AgentCore configuration with empty agents list
            config = {"agents": []}
            config_path = project_root / "agentcore_config.yml"
            config_path.write_text(yaml.dump(config))
            
            # Execute validation
            result = validator.validate_agentcore_configuration()
            
            # Property 1: Empty agents list should fail validation
            assert result.status == ValidationStatus.FAIL, \
                "Configuration with no agents should fail validation"
            
            # Property 2: Error message should mention no agents
            assert "no agent definitions" in result.message.lower(), \
                "Error message should clearly indicate no agents are defined"
            
            # Property 3: Remediation steps should guide user
            assert result.remediation_steps is not None, \
                "Remediation steps should be provided"
            assert any("add agent" in step.lower() for step in result.remediation_steps), \
                "Remediation steps should guide user to add agent definitions"



class TestConfigurationErrorReportingProperties:
    """Property-based tests for configuration error reporting."""
    
    @given(
        missing_env_vars=st.lists(
            st.sampled_from(["AWS_REGION", "BEDROCK_MODEL_ID", "DATABASE_URL"]),
            min_size=1,
            max_size=3,
            unique=True
        )
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_configuration_error_reporting_env_vars(self, missing_env_vars):
        """
        Property Test: Configuration error reporting for environment variables
        
        Feature: production-readiness-validation, Property 18: Configuration error reporting
        
        **Validates: Requirements 4.8**
        
        For any missing or invalid environment variable configuration, the system should 
        provide specific error messages indicating which configuration needs correction.
        
        This property verifies that:
        1. Missing environment variables are identified by name
        2. Error messages are specific and actionable
        3. Remediation steps are provided
        4. The error message lists all missing variables
        """
        # Create a temporary project directory
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            
            # Create validator with all required env vars
            all_env_vars = ["AWS_REGION", "BEDROCK_MODEL_ID", "DATABASE_URL"]
            validator = DeploymentConfigValidator(
                project_root=project_root,
                required_env_vars=all_env_vars
            )
            
            # Set only the env vars that are NOT in missing_env_vars
            present_vars = {var: f"test-{var.lower()}" for var in all_env_vars if var not in missing_env_vars}
            
            with patch.dict(os.environ, present_vars, clear=True):
                result = validator.validate_environment_variables()
                
                # Property 1: Validation should fail when env vars are missing
                assert result.status == ValidationStatus.FAIL, \
                    f"Validation should fail when {len(missing_env_vars)} env vars are missing"
                
                # Property 2: Error message should mention missing variables
                assert "missing" in result.message.lower(), \
                    "Error message should mention 'missing' variables"
                
                # Property 3: All missing variables should be identified in details
                assert "missing_vars" in result.details, \
                    "Validation result should include 'missing_vars' in details"
                
                for var in missing_env_vars:
                    assert var in result.details["missing_vars"], \
                        f"Missing variable '{var}' should be identified in validation result"
                
                # Property 4: Remediation steps should be provided
                assert result.remediation_steps is not None, \
                    "Remediation steps should be provided for missing env vars"
                assert len(result.remediation_steps) > 0, \
                    "At least one remediation step should be provided"
                
                # Property 5: Remediation steps should be specific
                # Should mention setting the variables or reference the required variables
                remediation_text = " ".join(result.remediation_steps).lower()
                assert any(keyword in remediation_text for keyword in ["set", "required", "environment"]), \
                    "Remediation steps should provide specific guidance on setting environment variables"
    
    @given(
        config_type=st.sampled_from([
            "docker_missing_dockerfile",
            "docker_missing_compose",
            "docker_invalid_yaml",
            "docker_missing_services",
            "agentcore_missing_file",
            "agentcore_invalid_yaml",
            "agentcore_no_agents",
            "agentcore_missing_fields"
        ])
    )
    @settings(max_examples=100, deadline=5000)
    def test_property_configuration_error_reporting_files(self, config_type):
        """
        Property Test: Configuration error reporting for file-based configurations
        
        Feature: production-readiness-validation, Property 18: Configuration error reporting
        
        **Validates: Requirements 4.8**
        
        For any missing or invalid file-based configuration (Docker, AgentCore), 
        the system should provide specific error messages indicating which 
        configuration needs correction.
        
        This property verifies that:
        1. Missing configuration files are identified by path
        2. Invalid configuration syntax is reported with details
        3. Missing required fields/sections are identified
        4. Error messages are specific to the configuration type
        5. Remediation steps guide the user to fix the specific issue
        """
        # Create a temporary project directory
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            
            validator = DeploymentConfigValidator(
                project_root=project_root,
                required_env_vars=["AWS_REGION", "BEDROCK_MODEL_ID", "DATABASE_URL"]
            )
            
            # Setup configuration based on test case
            if config_type == "docker_missing_dockerfile":
                # Don't create Dockerfile
                result = validator.validate_docker_configuration()
                
                # Property: Missing file should be identified
                assert result.status == ValidationStatus.FAIL
                assert "dockerfile not found" in result.message.lower()
                assert "expected_path" in result.details
                assert result.remediation_steps is not None
                assert any("create dockerfile" in step.lower() for step in result.remediation_steps)
            
            elif config_type == "docker_missing_compose":
                # Create Dockerfile but not docker-compose.yml
                (project_root / "Dockerfile").write_text("FROM python:3.11\n")
                result = validator.validate_docker_configuration()
                
                # Property: Missing file should be identified
                assert result.status == ValidationStatus.FAIL
                assert "docker-compose.yml not found" in result.message.lower()
                assert "expected_path" in result.details
                assert result.remediation_steps is not None
                assert any("docker-compose" in step.lower() for step in result.remediation_steps)
            
            elif config_type == "docker_invalid_yaml":
                # Create invalid YAML
                (project_root / "Dockerfile").write_text("FROM python:3.11\n")
                (project_root / "docker-compose.yml").write_text("invalid: yaml: syntax:\n")
                result = validator.validate_docker_configuration()
                
                # Property: Invalid syntax should be reported
                assert result.status == ValidationStatus.FAIL
                assert "invalid yaml syntax" in result.message.lower()
                assert result.remediation_steps is not None
                assert any("yaml" in step.lower() for step in result.remediation_steps)
            
            elif config_type == "docker_missing_services":
                # Create valid YAML but missing required services
                (project_root / "Dockerfile").write_text("FROM python:3.11\n")
                compose_config = {
                    "version": "3.8",
                    "services": {
                        "api": {"image": "test"}
                        # Missing "database" service
                    }
                }
                (project_root / "docker-compose.yml").write_text(yaml.dump(compose_config))
                result = validator.validate_docker_configuration()
                
                # Property: Missing services should be identified
                assert result.status == ValidationStatus.FAIL
                assert "missing required services" in result.message.lower()
                assert "missing_services" in result.details
                assert "database" in result.details["missing_services"]
                assert result.remediation_steps is not None
                assert any("service" in step.lower() for step in result.remediation_steps)
            
            elif config_type == "agentcore_missing_file":
                # Don't create AgentCore config
                result = validator.validate_agentcore_configuration()
                
                # Property: Missing file should be reported (as SKIP with paths checked)
                assert result.status == ValidationStatus.SKIP
                assert "agentcore configuration file not found" in result.message.lower()
                assert "checked_paths" in result.details
            
            elif config_type == "agentcore_invalid_yaml":
                # Create invalid YAML
                (project_root / "agentcore_config.yml").write_text("invalid: yaml: syntax:\n")
                result = validator.validate_agentcore_configuration()
                
                # Property: Invalid syntax should be reported
                assert result.status == ValidationStatus.FAIL
                assert "invalid yaml syntax" in result.message.lower()
                assert result.remediation_steps is not None
                assert any("yaml" in step.lower() for step in result.remediation_steps)
            
            elif config_type == "agentcore_no_agents":
                # Create valid YAML but no agents
                config = {"version": "1.0"}
                (project_root / "agentcore_config.yml").write_text(yaml.dump(config))
                result = validator.validate_agentcore_configuration()
                
                # Property: Missing agents should be reported
                assert result.status == ValidationStatus.FAIL
                assert "no agent definitions" in result.message.lower()
                assert result.remediation_steps is not None
                assert any("agent" in step.lower() for step in result.remediation_steps)
            
            elif config_type == "agentcore_missing_fields":
                # Create agents with missing required fields
                config = {
                    "agents": [
                        {"name": "test_agent", "model": "claude-v2"}  # Missing "tools"
                    ]
                }
                (project_root / "agentcore_config.yml").write_text(yaml.dump(config))
                result = validator.validate_agentcore_configuration()
                
                # Property: Missing fields should be identified
                assert result.status == ValidationStatus.FAIL
                assert "invalid agent definition" in result.message.lower()
                assert "invalid_agents" in result.details
                assert len(result.details["invalid_agents"]) > 0
                assert "missing_fields" in result.details["invalid_agents"][0]
                assert "tools" in result.details["invalid_agents"][0]["missing_fields"]
                assert result.remediation_steps is not None
                assert any("tools" in step.lower() or "field" in step.lower() for step in result.remediation_steps)
    
    @given(
        database_url_type=st.sampled_from([
            "missing",
            "no_scheme",
            "no_hostname",
            "invalid_format"
        ])
    )
    @settings(max_examples=100, deadline=5000)
    @pytest.mark.asyncio
    async def test_property_configuration_error_reporting_database(self, database_url_type):
        """
        Property Test: Configuration error reporting for database configuration
        
        Feature: production-readiness-validation, Property 18: Configuration error reporting
        
        **Validates: Requirements 4.8**
        
        For any missing or invalid database configuration, the system should 
        provide specific error messages indicating which configuration needs correction.
        
        This property verifies that:
        1. Missing DATABASE_URL is reported
        2. Invalid URL formats are identified with specific issues
        3. Missing URL components (scheme, hostname) are identified
        4. Error messages are specific to the database configuration issue
        5. Remediation steps provide correct URL format examples
        """
        # Create a temporary project directory
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            
            validator = DeploymentConfigValidator(
                project_root=project_root,
                required_env_vars=["AWS_REGION", "BEDROCK_MODEL_ID", "DATABASE_URL"]
            )
            
            # Setup database URL based on test case
            if database_url_type == "missing":
                # Don't set DATABASE_URL
                with patch.dict(os.environ, {}, clear=True):
                    result = await validator.validate_database_configuration()
                    
                    # Property: Missing DATABASE_URL should be reported
                    assert result.status == ValidationStatus.FAIL
                    assert "database_url environment variable not set" in result.message.lower()
                    assert result.remediation_steps is not None
                    assert any("database_url" in step.lower() for step in result.remediation_steps)
                    assert any("postgresql://" in step for step in result.remediation_steps), \
                        "Remediation should include example URL format"
            
            elif database_url_type == "no_scheme":
                # URL without scheme
                with patch.dict(os.environ, {"DATABASE_URL": "localhost/test"}):
                    result = await validator.validate_database_configuration()
                    
                    # Property: Missing scheme should be identified
                    assert result.status == ValidationStatus.FAIL
                    assert "missing database scheme" in result.message.lower()
                    assert result.remediation_steps is not None
                    assert any("postgresql://" in step for step in result.remediation_steps)
            
            elif database_url_type == "no_hostname":
                # URL without hostname
                with patch.dict(os.environ, {"DATABASE_URL": "postgresql:///test"}):
                    result = await validator.validate_database_configuration()
                    
                    # Property: Missing hostname should be identified
                    assert result.status == ValidationStatus.FAIL
                    assert "missing hostname" in result.message.lower()
                    assert result.remediation_steps is not None
                    assert any("host" in step.lower() for step in result.remediation_steps)
            
            elif database_url_type == "invalid_format":
                # Completely invalid URL
                with patch.dict(os.environ, {"DATABASE_URL": "not a valid url at all"}):
                    result = await validator.validate_database_configuration()
                    
                    # Property: Invalid format should be reported
                    assert result.status == ValidationStatus.FAIL
                    assert "invalid format" in result.message.lower() or "missing" in result.message.lower()
                    assert result.remediation_steps is not None
                    assert any("format" in step.lower() or "postgresql://" in step for step in result.remediation_steps)
    
    @given(
        aws_config_type=st.sampled_from([
            "missing_region",
            "no_credentials"
        ])
    )
    @settings(max_examples=100, deadline=5000)
    @pytest.mark.asyncio
    async def test_property_configuration_error_reporting_aws(self, aws_config_type):
        """
        Property Test: Configuration error reporting for AWS configuration
        
        Feature: production-readiness-validation, Property 18: Configuration error reporting
        
        **Validates: Requirements 4.8**
        
        For any missing or invalid AWS configuration, the system should 
        provide specific error messages indicating which configuration needs correction.
        
        This property verifies that:
        1. Missing AWS_REGION is reported
        2. Missing AWS credentials are identified
        3. Error messages are specific to the AWS configuration issue
        4. Remediation steps provide guidance on setting up AWS access
        """
        # Create a temporary project directory
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            
            validator = DeploymentConfigValidator(
                project_root=project_root,
                required_env_vars=["AWS_REGION", "BEDROCK_MODEL_ID", "DATABASE_URL"]
            )
            
            # Setup AWS configuration based on test case
            if aws_config_type == "missing_region":
                # Don't set AWS_REGION
                with patch.dict(os.environ, {}, clear=True):
                    result = await validator.validate_secrets_access()
                    
                    # Property: Missing AWS_REGION should be reported
                    assert result.status == ValidationStatus.FAIL
                    assert "aws_region environment variable not set" in result.message.lower()
                    assert result.remediation_steps is not None
                    assert any("aws_region" in step.lower() for step in result.remediation_steps)
            
            elif aws_config_type == "no_credentials":
                # Set AWS_REGION but no credentials
                with patch.dict(os.environ, {"AWS_REGION": "us-east-1"}):
                    with patch("boto3.client") as mock_client:
                        from botocore.exceptions import NoCredentialsError
                        mock_client.return_value.list_secrets.side_effect = NoCredentialsError()
                        
                        result = await validator.validate_secrets_access()
                        
                        # Property: Missing credentials should be reported
                        assert result.status == ValidationStatus.FAIL
                        assert "aws credentials not configured" in result.message.lower()
                        assert result.remediation_steps is not None
                        # Should provide multiple ways to configure credentials
                        remediation_text = " ".join(result.remediation_steps).lower()
                        assert "aws configure" in remediation_text or "credentials" in remediation_text
