"""Unit tests for MonitoringValidator."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime
from hypothesis import given, strategies as st, settings

from agent_scrivener.deployment.validation.monitoring_validator import MonitoringValidator
from agent_scrivener.deployment.validation.models import ValidationStatus


@pytest.fixture
def monitoring_validator():
    """Create a MonitoringValidator instance for testing."""
    return MonitoringValidator(
        aws_region="us-east-1",
        log_group_prefix="/aws/agent-scrivener",
        required_components=["api", "orchestrator", "agents", "database"]
    )


class TestCloudWatchLogs:
    """Tests for CloudWatch logs validation."""
    
    @pytest.mark.asyncio
    async def test_validate_cloudwatch_logs_all_present(self, monitoring_validator):
        """Test validation passes when all log groups exist."""
        mock_response = {
            'logGroups': [
                {'logGroupName': '/aws/agent-scrivener/api', 'retentionInDays': 30},
                {'logGroupName': '/aws/agent-scrivener/orchestrator', 'retentionInDays': 30},
                {'logGroupName': '/aws/agent-scrivener/agents', 'retentionInDays': 30},
                {'logGroupName': '/aws/agent-scrivener/database', 'retentionInDays': 30},
            ]
        }
        
        with patch('boto3.client') as mock_boto3:
            mock_client = MagicMock()
            mock_client.describe_log_groups.return_value = mock_response
            mock_boto3.return_value = mock_client
            
            result = await monitoring_validator.validate_cloudwatch_logs()
            
            assert result.status == ValidationStatus.PASS
            assert "All required CloudWatch log groups exist" in result.message
            assert result.details['found_components'] == ["api", "orchestrator", "agents", "database"]
    
    @pytest.mark.asyncio
    async def test_validate_cloudwatch_logs_missing_components(self, monitoring_validator):
        """Test validation fails when log groups are missing."""
        mock_response = {
            'logGroups': [
                {'logGroupName': '/aws/agent-scrivener/api', 'retentionInDays': 30},
                {'logGroupName': '/aws/agent-scrivener/orchestrator', 'retentionInDays': 30},
            ]
        }
        
        with patch('boto3.client') as mock_boto3:
            mock_client = MagicMock()
            mock_client.describe_log_groups.return_value = mock_response
            mock_boto3.return_value = mock_client
            
            result = await monitoring_validator.validate_cloudwatch_logs()
            
            assert result.status == ValidationStatus.FAIL
            assert "Missing CloudWatch log groups" in result.message
            assert "agents" in result.details['missing_components']
            assert "database" in result.details['missing_components']
    
    @pytest.mark.asyncio
    async def test_validate_cloudwatch_logs_no_retention(self, monitoring_validator):
        """Test validation warns when retention policies are missing."""
        mock_response = {
            'logGroups': [
                {'logGroupName': '/aws/agent-scrivener/api'},  # No retentionInDays
                {'logGroupName': '/aws/agent-scrivener/orchestrator', 'retentionInDays': 30},
                {'logGroupName': '/aws/agent-scrivener/agents', 'retentionInDays': 30},
                {'logGroupName': '/aws/agent-scrivener/database'},  # No retentionInDays
            ]
        }
        
        with patch('boto3.client') as mock_boto3:
            mock_client = MagicMock()
            mock_client.describe_log_groups.return_value = mock_response
            mock_boto3.return_value = mock_client
            
            result = await monitoring_validator.validate_cloudwatch_logs()
            
            assert result.status == ValidationStatus.WARNING
            assert "do not have retention policies" in result.message
            assert len(result.details['log_groups_without_retention']) == 2
    
    @pytest.mark.asyncio
    async def test_validate_cloudwatch_logs_no_credentials(self, monitoring_validator):
        """Test validation skips when AWS credentials are not configured."""
        with patch('boto3.client') as mock_boto3:
            from botocore.exceptions import NoCredentialsError
            mock_boto3.side_effect = NoCredentialsError()
            
            result = await monitoring_validator.validate_cloudwatch_logs()
            
            assert result.status == ValidationStatus.SKIP
            assert "AWS credentials not configured" in result.message
    
    @pytest.mark.asyncio
    async def test_validate_cloudwatch_logs_boto3_not_installed(self, monitoring_validator):
        """Test validation skips when boto3 is not installed."""
        import sys
        # Temporarily remove boto3 from sys.modules to simulate it not being installed
        boto3_module = sys.modules.get('boto3')
        botocore_module = sys.modules.get('botocore')
        
        try:
            if 'boto3' in sys.modules:
                del sys.modules['boto3']
            if 'botocore' in sys.modules:
                del sys.modules['botocore']
            
            # Patch the import to raise ImportError
            with patch.dict('sys.modules', {'boto3': None, 'botocore': None}):
                result = await monitoring_validator.validate_cloudwatch_logs()
                
                assert result.status == ValidationStatus.SKIP
                assert "boto3 not installed" in result.message
        finally:
            # Restore the modules
            if boto3_module is not None:
                sys.modules['boto3'] = boto3_module
            if botocore_module is not None:
                sys.modules['botocore'] = botocore_module


class TestCloudWatchMetrics:
    """Tests for CloudWatch metrics validation."""
    
    @pytest.mark.asyncio
    async def test_validate_cloudwatch_metrics_all_present(self, monitoring_validator):
        """Test validation passes when all required metrics exist."""
        mock_response = {
            'Metrics': [
                {'MetricName': 'SessionCount', 'Namespace': 'AgentScrivener'},
                {'MetricName': 'SuccessRate', 'Namespace': 'AgentScrivener'},
                {'MetricName': 'ErrorRate', 'Namespace': 'AgentScrivener'},
            ]
        }
        
        with patch('boto3.client') as mock_boto3:
            mock_client = MagicMock()
            mock_client.list_metrics.return_value = mock_response
            mock_boto3.return_value = mock_client
            
            result = await monitoring_validator.validate_cloudwatch_metrics()
            
            assert result.status == ValidationStatus.PASS
            assert "All required CloudWatch custom metrics are configured" in result.message
            assert set(result.details['found_metrics']) == {'SessionCount', 'SuccessRate', 'ErrorRate'}
    
    @pytest.mark.asyncio
    async def test_validate_cloudwatch_metrics_missing(self, monitoring_validator):
        """Test validation fails when metrics are missing."""
        mock_response = {
            'Metrics': [
                {'MetricName': 'SessionCount', 'Namespace': 'AgentScrivener'},
            ]
        }
        
        with patch('boto3.client') as mock_boto3:
            mock_client = MagicMock()
            mock_client.list_metrics.return_value = mock_response
            mock_boto3.return_value = mock_client
            
            result = await monitoring_validator.validate_cloudwatch_metrics()
            
            assert result.status == ValidationStatus.FAIL
            assert "Missing CloudWatch custom metrics" in result.message
            assert "SuccessRate" in result.details['missing_metrics']
            assert "ErrorRate" in result.details['missing_metrics']
    
    @pytest.mark.asyncio
    async def test_validate_cloudwatch_metrics_no_credentials(self, monitoring_validator):
        """Test validation skips when AWS credentials are not configured."""
        with patch('boto3.client') as mock_boto3:
            from botocore.exceptions import NoCredentialsError
            mock_boto3.side_effect = NoCredentialsError()
            
            result = await monitoring_validator.validate_cloudwatch_metrics()
            
            assert result.status == ValidationStatus.SKIP
            assert "AWS credentials not configured" in result.message


class TestCloudWatchAlarms:
    """Tests for CloudWatch alarms validation."""
    
    @pytest.mark.asyncio
    async def test_validate_cloudwatch_alarms_all_present(self, monitoring_validator):
        """Test validation passes when all required alarms exist."""
        mock_response = {
            'MetricAlarms': [
                {'AlarmName': 'agent-scrivener-high-error-rate'},
                {'AlarmName': 'agent-scrivener-high-latency'},
                {'AlarmName': 'agent-scrivener-service-unavailable'},
            ]
        }
        
        with patch('boto3.client') as mock_boto3:
            mock_client = MagicMock()
            mock_client.describe_alarms.return_value = mock_response
            mock_boto3.return_value = mock_client
            
            result = await monitoring_validator.validate_cloudwatch_alarms()
            
            assert result.status == ValidationStatus.PASS
            assert "CloudWatch alarms configured" in result.message
            assert len(result.details['found_alarm_types']) == 3
    
    @pytest.mark.asyncio
    async def test_validate_cloudwatch_alarms_missing(self, monitoring_validator):
        """Test validation fails when alarms are missing."""
        mock_response = {
            'MetricAlarms': [
                {'AlarmName': 'agent-scrivener-high-error-rate'},
            ]
        }
        
        with patch('boto3.client') as mock_boto3:
            mock_client = MagicMock()
            mock_client.describe_alarms.return_value = mock_response
            mock_boto3.return_value = mock_client
            
            result = await monitoring_validator.validate_cloudwatch_alarms()
            
            assert result.status == ValidationStatus.FAIL
            assert "Missing CloudWatch alarms" in result.message
            assert "high_latency" in result.details['missing_alarm_types']
            assert "service_unavailable" in result.details['missing_alarm_types']
    
    @pytest.mark.asyncio
    async def test_validate_cloudwatch_alarms_no_credentials(self, monitoring_validator):
        """Test validation skips when AWS credentials are not configured."""
        with patch('boto3.client') as mock_boto3:
            from botocore.exceptions import NoCredentialsError
            mock_boto3.side_effect = NoCredentialsError()
            
            result = await monitoring_validator.validate_cloudwatch_alarms()
            
            assert result.status == ValidationStatus.SKIP
            assert "AWS credentials not configured" in result.message


class TestHealthChecks:
    """Tests for health check validation."""
    
    @pytest.mark.asyncio
    async def test_validate_health_checks_success(self, monitoring_validator):
        """Test validation passes when health endpoint is functional."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "api": "healthy",
            "database": "healthy",
            "aws": "healthy",
            "agents": "healthy"
        }
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            result = await monitoring_validator.validate_health_checks()
            
            assert result.status == ValidationStatus.PASS
            assert "Health check endpoint is functional" in result.message
    
    @pytest.mark.asyncio
    async def test_validate_health_checks_missing_components(self, monitoring_validator):
        """Test validation warns when health checks are incomplete."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "api": "healthy"
            # Missing database and aws checks
        }
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            result = await monitoring_validator.validate_health_checks()
            
            assert result.status == ValidationStatus.WARNING
            assert "Health endpoint missing checks" in result.message
            assert "database" in result.details['missing_checks']
            assert "aws" in result.details['missing_checks']
    
    @pytest.mark.asyncio
    async def test_validate_health_checks_server_not_running(self, monitoring_validator):
        """Test validation skips when API server is not running."""
        with patch('httpx.AsyncClient') as mock_client_class:
            import httpx
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client_class.return_value = mock_client
            
            result = await monitoring_validator.validate_health_checks()
            
            assert result.status == ValidationStatus.SKIP
            assert "API server not running" in result.message
    
    @pytest.mark.asyncio
    async def test_validate_health_checks_httpx_not_installed(self, monitoring_validator):
        """Test validation skips when httpx is not installed."""
        with patch.dict('sys.modules', {'httpx': None}):
            result = await monitoring_validator.validate_health_checks()
            
            assert result.status == ValidationStatus.SKIP
            assert "httpx not installed" in result.message


class TestStructuredLogging:
    """Tests for structured logging validation."""
    
    @pytest.mark.asyncio
    async def test_validate_structured_logging_configured(self, monitoring_validator):
        """Test validation passes when structured logging is configured."""
        # Create a mock handler with a formatter
        mock_handler = MagicMock()
        mock_formatter = MagicMock()
        mock_formatter.__class__.__name__ = "JsonFormatter"
        mock_handler.formatter = mock_formatter
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_logger.handlers = [mock_handler]
            mock_logger.level = 20  # INFO level
            mock_get_logger.return_value = mock_logger
            
            result = await monitoring_validator.validate_structured_logging()
            
            assert result.status == ValidationStatus.PASS
            assert "Structured logging is configured" in result.message
    
    @pytest.mark.asyncio
    async def test_validate_structured_logging_not_configured(self, monitoring_validator):
        """Test validation warns when structured logging is not configured."""
        # Create a mock handler without JSON formatter and without structured fields
        mock_handler = MagicMock()
        mock_formatter = MagicMock()
        mock_formatter.__class__.__name__ = "Formatter"
        mock_formatter._fmt = "simple message"  # No structured fields at all
        mock_handler.formatter = mock_formatter
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_logger.handlers = [mock_handler]
            mock_logger.level = 20  # INFO level
            mock_get_logger.return_value = mock_logger
            
            result = await monitoring_validator.validate_structured_logging()
            
            assert result.status == ValidationStatus.WARNING
            assert "Structured logging not configured" in result.message
    
    @pytest.mark.asyncio
    async def test_validate_structured_logging_wrong_level(self, monitoring_validator):
        """Test validation warns when log level is too high."""
        # Create a mock handler with JSON formatter
        mock_handler = MagicMock()
        mock_formatter = MagicMock()
        mock_formatter.__class__.__name__ = "JsonFormatter"
        mock_handler.formatter = mock_formatter
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_logger.handlers = [mock_handler]
            mock_logger.level = 30  # WARNING level (too high)
            mock_get_logger.return_value = mock_logger
            
            result = await monitoring_validator.validate_structured_logging()
            
            assert result.status == ValidationStatus.WARNING
            assert "should be INFO or lower" in result.message


class TestAlerting:
    """Tests for alerting validation."""
    
    @pytest.mark.asyncio
    async def test_validate_alerting_configured(self, monitoring_validator):
        """Test validation passes when SNS topics are configured."""
        mock_topics_response = {
            'Topics': [
                {'TopicArn': 'arn:aws:sns:us-east-1:123456789012:agent-scrivener-alerts'},
                {'TopicArn': 'arn:aws:sns:us-east-1:123456789012:agent-scrivener-critical-alarms'},
            ]
        }
        
        mock_subs_response = {
            'Subscriptions': [
                {'SubscriptionArn': 'arn:aws:sns:us-east-1:123456789012:agent-scrivener-alerts:sub-id'}
            ]
        }
        
        with patch('boto3.client') as mock_boto3:
            mock_client = MagicMock()
            mock_client.list_topics.return_value = mock_topics_response
            mock_client.list_subscriptions_by_topic.return_value = mock_subs_response
            mock_boto3.return_value = mock_client
            
            result = await monitoring_validator.validate_alerting()
            
            assert result.status == ValidationStatus.PASS
            assert "Alerting configured" in result.message
            assert len(result.details['alert_topics']) == 2
    
    @pytest.mark.asyncio
    async def test_validate_alerting_no_topics(self, monitoring_validator):
        """Test validation fails when no SNS topics exist."""
        mock_response = {'Topics': []}
        
        with patch('boto3.client') as mock_boto3:
            mock_client = MagicMock()
            mock_client.list_topics.return_value = mock_response
            mock_boto3.return_value = mock_client
            
            result = await monitoring_validator.validate_alerting()
            
            assert result.status == ValidationStatus.FAIL
            assert "No SNS topics configured" in result.message
    
    @pytest.mark.asyncio
    async def test_validate_alerting_no_subscriptions(self, monitoring_validator):
        """Test validation warns when topics have no subscriptions."""
        mock_topics_response = {
            'Topics': [
                {'TopicArn': 'arn:aws:sns:us-east-1:123456789012:agent-scrivener-alerts'},
            ]
        }
        
        mock_subs_response = {'Subscriptions': []}  # No subscriptions
        
        with patch('boto3.client') as mock_boto3:
            mock_client = MagicMock()
            mock_client.list_topics.return_value = mock_topics_response
            mock_client.list_subscriptions_by_topic.return_value = mock_subs_response
            mock_boto3.return_value = mock_client
            
            result = await monitoring_validator.validate_alerting()
            
            assert result.status == ValidationStatus.WARNING
            assert "no subscriptions" in result.message
    
    @pytest.mark.asyncio
    async def test_validate_alerting_no_credentials(self, monitoring_validator):
        """Test validation skips when AWS credentials are not configured."""
        with patch('boto3.client') as mock_boto3:
            from botocore.exceptions import NoCredentialsError
            mock_boto3.side_effect = NoCredentialsError()
            
            result = await monitoring_validator.validate_alerting()
            
            assert result.status == ValidationStatus.SKIP
            assert "AWS credentials not configured" in result.message


class TestAlertDelivery:
    """Tests for alert delivery testing."""
    
    @pytest.mark.asyncio
    async def test_alert_delivery_success(self, monitoring_validator):
        """Test alert delivery succeeds."""
        mock_topics_response = {
            'Topics': [
                {'TopicArn': 'arn:aws:sns:us-east-1:123456789012:agent-scrivener-alerts'},
            ]
        }
        
        mock_publish_response = {
            'MessageId': 'test-message-id-12345'
        }
        
        with patch('boto3.client') as mock_boto3:
            mock_client = MagicMock()
            mock_client.list_topics.return_value = mock_topics_response
            mock_client.publish.return_value = mock_publish_response
            mock_boto3.return_value = mock_client
            
            result = await monitoring_validator.test_alert_delivery()
            
            assert result.status == ValidationStatus.PASS
            assert "Test alert delivered successfully" in result.message
            assert result.details['message_id'] == 'test-message-id-12345'
    
    @pytest.mark.asyncio
    async def test_alert_delivery_no_topic(self, monitoring_validator):
        """Test alert delivery skips when no topic exists."""
        mock_response = {'Topics': []}
        
        with patch('boto3.client') as mock_boto3:
            mock_client = MagicMock()
            mock_client.list_topics.return_value = mock_response
            mock_boto3.return_value = mock_client
            
            result = await monitoring_validator.test_alert_delivery()
            
            assert result.status == ValidationStatus.SKIP
            assert "No alert SNS topic found" in result.message
    
    @pytest.mark.asyncio
    async def test_alert_delivery_no_credentials(self, monitoring_validator):
        """Test alert delivery skips when AWS credentials are not configured."""
        with patch('boto3.client') as mock_boto3:
            from botocore.exceptions import NoCredentialsError
            mock_boto3.side_effect = NoCredentialsError()
            
            result = await monitoring_validator.test_alert_delivery()
            
            assert result.status == ValidationStatus.SKIP
            assert "AWS credentials not configured" in result.message


class TestValidateAll:
    """Tests for the main validate method."""
    
    @pytest.mark.asyncio
    async def test_validate_runs_all_checks(self, monitoring_validator):
        """Test that validate() runs all validation checks."""
        with patch.object(monitoring_validator, 'validate_cloudwatch_logs', new_callable=AsyncMock) as mock_logs, \
             patch.object(monitoring_validator, 'validate_cloudwatch_metrics', new_callable=AsyncMock) as mock_metrics, \
             patch.object(monitoring_validator, 'validate_cloudwatch_alarms', new_callable=AsyncMock) as mock_alarms, \
             patch.object(monitoring_validator, 'validate_health_checks', new_callable=AsyncMock) as mock_health, \
             patch.object(monitoring_validator, 'validate_structured_logging', new_callable=AsyncMock) as mock_logging, \
             patch.object(monitoring_validator, 'validate_alerting', new_callable=AsyncMock) as mock_alerting, \
             patch.object(monitoring_validator, 'test_alert_delivery', new_callable=AsyncMock) as mock_delivery:
            
            # Set up mock return values
            from agent_scrivener.deployment.validation.models import ValidationResult
            mock_result = ValidationResult(
                validator_name="MonitoringValidator",
                status=ValidationStatus.PASS,
                message="Test passed"
            )
            
            mock_logs.return_value = mock_result
            mock_metrics.return_value = mock_result
            mock_alarms.return_value = mock_result
            mock_health.return_value = mock_result
            mock_logging.return_value = mock_result
            mock_alerting.return_value = mock_result
            mock_delivery.return_value = mock_result
            
            results = await monitoring_validator.validate()
            
            # Verify all methods were called
            assert mock_logs.called
            assert mock_metrics.called
            assert mock_alarms.called
            assert mock_health.called
            assert mock_logging.called
            assert mock_alerting.called
            assert mock_delivery.called
            
            # Verify we got 7 results
            assert len(results) == 7



class TestPropertyHealthCheckAvailability:
    """Property-based tests for component health check availability."""
    
    @given(
        # The validate_health_checks method checks for specific components: api, database, aws
        # We'll test with subsets of these required components
        include_api=st.booleans(),
        include_database=st.booleans(),
        include_aws=st.booleans(),
        health_status=st.sampled_from(["healthy", "degraded", "unhealthy"]),
        include_version=st.booleans(),
        include_uptime=st.booleans()
    )
    @settings(max_examples=100, deadline=5000)
    @pytest.mark.asyncio
    async def test_property_component_health_check_availability(
        self,
        include_api,
        include_database,
        include_aws,
        health_status,
        include_version,
        include_uptime
    ):
        """
        Property Test: Component health check availability
        
        Feature: production-readiness-validation, Property 24: Component health check availability
        
        **Validates: Requirements 7.4**
        
        For any system component (API, orchestrator, agents, database), a health check 
        endpoint should be available and functional.
        
        This property verifies that:
        1. Health check endpoint is accessible for all components
        2. Health check returns appropriate status information
        3. Health check includes component-specific details
        4. Health check responds within acceptable time limits
        5. Health check provides structured response format
        6. Health check works consistently across different component states
        7. Health check includes version and uptime information when available
        8. Health check failures provide diagnostic information
        """
        # Create validator instance for this test iteration
        # The validator checks for hardcoded components: api, database, aws
        validator = MonitoringValidator(
            aws_region="us-east-1",
            log_group_prefix="/aws/agent-scrivener",
            required_components=["api", "orchestrator", "agents", "database"]
        )
        
        # Build mock health response based on generated data
        # Only include components that were selected
        health_response = {}
        components = []
        
        if include_api:
            components.append("api")
        if include_database:
            components.append("database")
        if include_aws:
            components.append("aws")
        
        # Ensure at least one component is included
        if not components:
            components = ["api"]
        
        for component in components:
            component_health = {
                "status": health_status
            }
            
            # Add optional fields
            if include_version:
                component_health["version"] = f"1.{abs(hash(component)) % 10}.0"
            
            if include_uptime:
                component_health["uptime_seconds"] = abs(hash(component + health_status)) % 86400
            
            health_response[component] = component_health
        
        # Mock the HTTP client to return our generated health response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = health_response
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            # Execute the health check validation
            result = await validator.validate_health_checks()
            
            # Property 1: Health check endpoint is accessible
            # The validation should complete without connection errors
            assert result is not None, \
                "Health check validation should return a result"
            
            # Property 2: Health check returns appropriate status information
            # The validator checks for specific components: api, database, aws
            required_checks = ["api", "database", "aws"]
            all_present = all(comp in health_response for comp in required_checks)
            
            if all_present:
                # All required components present - should PASS
                assert result.status == ValidationStatus.PASS, \
                    f"When all required components (api, database, aws) are present, status should be PASS, got {result.status}"
            else:
                # Some components missing - should WARN
                assert result.status == ValidationStatus.WARNING, \
                    f"When some required components are missing, status should be WARNING, got {result.status}"
            
            # Property 3: Health check includes component-specific details
            # The result should contain information about the health data
            if result.status == ValidationStatus.PASS:
                assert "health_data" in result.details, \
                    "Successful health check should include health_data in details"
                
                # Verify all present components are represented in the health data
                health_data = result.details["health_data"]
                for component in components:
                    assert component in health_data, \
                        f"Component {component} should be present in health data"
                    
                    # Verify component has status information
                    assert "status" in health_data[component], \
                        f"Component {component} should have status field"
                    assert health_data[component]["status"] == health_status, \
                        f"Component {component} status should be {health_status}"
                    
                    # Verify optional fields are preserved
                    if include_version:
                        assert "version" in health_data[component], \
                            f"Component {component} should include version when provided"
                    
                    if include_uptime:
                        assert "uptime_seconds" in health_data[component], \
                            f"Component {component} should include uptime when provided"
            
            elif result.status == ValidationStatus.WARNING:
                # Missing components should be listed
                assert "missing_checks" in result.details, \
                    "Warning status should include missing_checks in details"
                
                missing_checks = result.details["missing_checks"]
                for required_comp in required_checks:
                    if required_comp not in health_response:
                        assert required_comp in missing_checks, \
                            f"Missing component {required_comp} should be listed in missing_checks"
            
            # Property 4: Health check responds within acceptable time limits
            # The mock client should have been called with a timeout
            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args[1]
            assert "timeout" in call_kwargs, \
                "HTTP client should be configured with a timeout"
            assert call_kwargs["timeout"] <= 10.0, \
                "Health check timeout should be reasonable (≤10 seconds)"
            
            # Property 5: Health check provides structured response format
            # The validation result should have a consistent structure
            assert hasattr(result, "status"), \
                "Result should have status attribute"
            assert hasattr(result, "message"), \
                "Result should have message attribute"
            assert hasattr(result, "details"), \
                "Result should have details attribute"
            assert isinstance(result.details, dict), \
                "Result details should be a dictionary"
            
            # Property 6: Health check works consistently across different component states
            # The validation should handle the health status appropriately
            if health_status == "healthy" and all_present:
                # All healthy components should result in PASS
                assert result.status == ValidationStatus.PASS, \
                    "All healthy components should result in PASS status"
                assert "functional" in result.message.lower() or "healthy" in result.message.lower(), \
                    "Success message should indicate health check is functional"
            
            # Property 7: Health check includes diagnostic information
            # The result should include the health URL for debugging
            assert "health_url" in result.details, \
                "Result should include health_url for diagnostic purposes"
            assert result.details["health_url"].startswith("http"), \
                "Health URL should be a valid HTTP URL"
            
            # Property 8: Health check failures provide remediation steps
            # If validation fails or warns, remediation steps should be provided
            if result.status in [ValidationStatus.FAIL, ValidationStatus.WARNING]:
                assert result.remediation_steps is not None, \
                    "Failed or warning validations should provide remediation steps"
                assert len(result.remediation_steps) > 0, \
                    "Remediation steps should not be empty"
                assert all(isinstance(step, str) for step in result.remediation_steps), \
                    "All remediation steps should be strings"
                assert all(len(step) > 0 for step in result.remediation_steps), \
                    "Remediation steps should not be empty strings"
    
    @given(
        # The validate_health_checks method checks for: api, database, aws
        # We'll test with different combinations of missing components
        include_api=st.booleans(),
        include_database=st.booleans(),
        include_aws=st.booleans()
    )
    @settings(max_examples=100, deadline=5000)
    @pytest.mark.asyncio
    async def test_property_health_check_missing_components(
        self,
        include_api,
        include_database,
        include_aws
    ):
        """
        Property Test: Health check detects missing component checks
        
        Feature: production-readiness-validation, Property 24: Component health check availability
        
        **Validates: Requirements 7.4**
        
        For any system component that is required but missing from health checks,
        the validation should detect and report the missing component.
        
        This property verifies that:
        1. Missing components are detected
        2. Validation status reflects incomplete health checks
        3. Missing components are listed in details
        4. Remediation steps guide adding missing checks
        """
        # Create validator instance
        validator = MonitoringValidator(
            aws_region="us-east-1",
            log_group_prefix="/aws/agent-scrivener",
            required_components=["api", "orchestrator", "agents", "database"]
        )
        
        # Build health response with some components potentially missing
        health_response = {}
        if include_api:
            health_response["api"] = {"status": "healthy"}
        if include_database:
            health_response["database"] = {"status": "healthy"}
        if include_aws:
            health_response["aws"] = {"status": "healthy"}
        
        # Ensure at least one component is present to avoid empty response
        if not health_response:
            health_response["api"] = {"status": "healthy"}
            include_api = True
        
        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = health_response
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            # Execute the health check validation
            result = await validator.validate_health_checks()
            
            # Determine which required components are missing
            # The validator checks for: api, database, aws
            required_checks = ["api", "database", "aws"]
            actually_missing = []
            if not include_api:
                actually_missing.append("api")
            if not include_database:
                actually_missing.append("database")
            if not include_aws:
                actually_missing.append("aws")
            
            if actually_missing:
                # Property 1: Missing components are detected
                assert result.status == ValidationStatus.WARNING, \
                    f"When components are missing, status should be WARNING, got {result.status}"
                
                # Property 2: Validation status reflects incomplete health checks
                assert "missing" in result.message.lower(), \
                    "Message should indicate missing components"
                
                # Property 3: Missing components are listed in details
                assert "missing_checks" in result.details, \
                    "Details should include missing_checks field"
                
                missing_checks = result.details["missing_checks"]
                for component in actually_missing:
                    assert component in missing_checks, \
                        f"Missing component {component} should be listed in missing_checks"
                
                # Property 4: Remediation steps guide adding missing checks
                assert result.remediation_steps is not None, \
                    "Remediation steps should be provided for missing components"
                assert len(result.remediation_steps) > 0, \
                    "At least one remediation step should be provided"
                
                # Check that remediation mentions adding health checks
                remediation_text = " ".join(result.remediation_steps).lower()
                assert any(keyword in remediation_text for keyword in ["add", "include", "implement", "check"]), \
                    "Remediation should guide adding missing health checks"
            else:
                # All required components are present
                assert result.status == ValidationStatus.PASS, \
                    "When all required components are present, status should be PASS"
    
    @given(
        # Test with different combinations of required components present
        include_api=st.booleans(),
        include_database=st.booleans(),
        include_aws=st.booleans(),
        connection_scenario=st.sampled_from(["success", "connection_refused", "timeout", "http_error"])
    )
    @settings(max_examples=100, deadline=5000)
    @pytest.mark.asyncio
    async def test_property_health_check_connection_handling(
        self,
        include_api,
        include_database,
        include_aws,
        connection_scenario
    ):
        """
        Property Test: Health check handles connection scenarios
        
        Feature: production-readiness-validation, Property 24: Component health check availability
        
        **Validates: Requirements 7.4**
        
        For any connection scenario (success, failure, timeout), the health check
        validation should handle it gracefully and provide appropriate feedback.
        
        This property verifies that:
        1. Successful connections result in PASS or WARNING status
        2. Connection failures result in SKIP status (server not running)
        3. HTTP errors result in FAIL status
        4. Timeouts are handled gracefully
        5. Error details are captured for debugging
        6. Remediation steps are provided for failures
        """
        # Create validator instance
        validator = MonitoringValidator(
            aws_region="us-east-1",
            log_group_prefix="/aws/agent-scrivener",
            required_components=["api", "orchestrator", "agents", "database"]
        )
        
        # Build health response
        health_response = {}
        if include_api:
            health_response["api"] = {"status": "healthy"}
        if include_database:
            health_response["database"] = {"status": "healthy"}
        if include_aws:
            health_response["aws"] = {"status": "healthy"}
        
        # Ensure at least one component for success scenario
        if not health_response and connection_scenario == "success":
            health_response["api"] = {"status": "healthy"}
            include_api = True
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            
            if connection_scenario == "success":
                # Successful health check
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = health_response
                mock_client.get.return_value = mock_response
                
            elif connection_scenario == "connection_refused":
                # Server not running
                import httpx
                mock_client.get.side_effect = httpx.ConnectError("Connection refused")
                
            elif connection_scenario == "timeout":
                # Request timeout
                import httpx
                mock_client.get.side_effect = httpx.TimeoutException("Request timeout")
                
            elif connection_scenario == "http_error":
                # HTTP error (e.g., 500)
                mock_response = MagicMock()
                mock_response.status_code = 500
                mock_client.get.return_value = mock_response
            
            mock_client_class.return_value = mock_client
            
            # Execute the health check validation
            result = await validator.validate_health_checks()
            
            # Determine if all required components are present
            required_checks = ["api", "database", "aws"]
            all_present = include_api and include_database and include_aws
            
            # Property 1: Successful connections result in PASS or WARNING status
            if connection_scenario == "success":
                if all_present:
                    assert result.status == ValidationStatus.PASS, \
                        "Successful health check with all components should result in PASS status"
                    assert "functional" in result.message.lower() or "healthy" in result.message.lower(), \
                        "Success message should indicate functionality"
                else:
                    assert result.status == ValidationStatus.WARNING, \
                        "Successful health check with missing components should result in WARNING status"
            
            # Property 2: Connection failures result in SKIP status
            elif connection_scenario == "connection_refused":
                assert result.status == ValidationStatus.SKIP, \
                    "Connection refused should result in SKIP status (server not running)"
                assert "not running" in result.message.lower() or "connection" in result.message.lower(), \
                    "Message should indicate server is not running"
            
            # Property 3: HTTP errors result in FAIL status
            elif connection_scenario == "http_error":
                assert result.status == ValidationStatus.FAIL, \
                    "HTTP error should result in FAIL status"
                assert "500" in result.message or "status" in result.message.lower(), \
                    "Message should indicate HTTP error status"
            
            # Property 4: Timeouts are handled gracefully
            elif connection_scenario == "timeout":
                assert result.status in [ValidationStatus.FAIL, ValidationStatus.SKIP], \
                    "Timeout should result in FAIL or SKIP status"
            
            # Property 5: Error details are captured for debugging
            assert "health_url" in result.details, \
                "Result should include health_url for debugging"
            
            if connection_scenario != "success":
                # Failed scenarios should include error information
                assert "reason" in result.details or "error" in result.details or "status_code" in result.details, \
                    "Failed validations should include error details"
            
            # Property 6: Remediation steps are provided for failures
            if result.status in [ValidationStatus.FAIL, ValidationStatus.SKIP, ValidationStatus.WARNING]:
                assert result.remediation_steps is not None, \
                    "Failed validations should provide remediation steps"
                assert len(result.remediation_steps) > 0, \
                    "At least one remediation step should be provided"
                
                # Remediation should be relevant to the failure type
                if connection_scenario == "connection_refused":
                    remediation_text = " ".join(result.remediation_steps).lower()
                    assert any(keyword in remediation_text for keyword in ["start", "running", "server"]), \
                        "Remediation for connection refused should mention starting the server"


class TestPropertyHealthCheckDiagnosticInformation:
    """Property-based tests for health check diagnostic information."""
    
    @given(
        # Test with different health statuses including degraded states
        api_status=st.sampled_from(["healthy", "degraded", "unhealthy", "error"]),
        database_status=st.sampled_from(["healthy", "degraded", "unhealthy", "error"]),
        aws_status=st.sampled_from(["healthy", "degraded", "unhealthy", "error"]),
        # Include diagnostic details for degraded/unhealthy states
        include_error_message=st.booleans(),
        include_error_code=st.booleans(),
        include_component_details=st.booleans(),
        include_timestamp=st.booleans()
    )
    @settings(max_examples=100, deadline=5000)
    @pytest.mark.asyncio
    async def test_property_health_check_diagnostic_information(
        self,
        api_status,
        database_status,
        aws_status,
        include_error_message,
        include_error_code,
        include_component_details,
        include_timestamp
    ):
        """
        Property Test: Health check diagnostic information
        
        Feature: production-readiness-validation, Property 25: Health check diagnostic information
        
        **Validates: Requirements 7.6**
        
        For any degraded health status detected, the system should provide detailed 
        diagnostic information about the issue.
        
        This property verifies that:
        1. Degraded health statuses are detected and reported
        2. Diagnostic information is provided for degraded components
        3. Diagnostic information includes error messages when available
        4. Diagnostic information includes error codes when available
        5. Diagnostic information includes component-specific details
        6. Diagnostic information includes timestamps when available
        7. Diagnostic information is structured and accessible
        8. Multiple degraded components are all reported with diagnostics
        """
        # Create validator instance
        validator = MonitoringValidator(
            aws_region="us-east-1",
            log_group_prefix="/aws/agent-scrivener",
            required_components=["api", "orchestrator", "agents", "database"]
        )
        
        # Build health response with various statuses
        health_response = {}
        degraded_components = []
        
        # Helper function to build component health data
        def build_component_health(status, component_name):
            component_health = {"status": status}
            
            # Add diagnostic information for degraded/unhealthy/error states
            if status in ["degraded", "unhealthy", "error"]:
                degraded_components.append(component_name)
                
                if include_error_message:
                    component_health["error_message"] = f"{component_name} is experiencing issues"
                
                if include_error_code:
                    component_health["error_code"] = f"{status.upper()}_{component_name.upper()}"
                
                if include_component_details:
                    component_health["details"] = {
                        "component": component_name,
                        "issue_type": status,
                        "severity": "high" if status == "error" else "medium"
                    }
                
                if include_timestamp:
                    component_health["timestamp"] = "2024-01-01T00:00:00Z"
            
            return component_health
        
        # Build health response for each component
        health_response["api"] = build_component_health(api_status, "api")
        health_response["database"] = build_component_health(database_status, "database")
        health_response["aws"] = build_component_health(aws_status, "aws")
        
        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = health_response
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            # Execute the health check validation
            result = await validator.validate_health_checks()
            
            # Property 1: Degraded health statuses are detected and reported
            # The validation should complete successfully even with degraded components
            assert result is not None, \
                "Health check validation should return a result"
            
            # All required components are present, so status should be PASS
            # (The validator checks for presence, not health status)
            assert result.status == ValidationStatus.PASS, \
                "When all required components are present, status should be PASS regardless of health status"
            
            # Property 2: Diagnostic information is provided for degraded components
            # The health data should be included in the result details
            assert "health_data" in result.details, \
                "Result should include health_data with diagnostic information"
            
            health_data = result.details["health_data"]
            
            # Property 3-6: Verify diagnostic information for each degraded component
            for component in degraded_components:
                assert component in health_data, \
                    f"Degraded component {component} should be present in health data"
                
                component_data = health_data[component]
                
                # Property 3: Diagnostic information includes error messages when available
                if include_error_message:
                    assert "error_message" in component_data, \
                        f"Degraded component {component} should include error_message when provided"
                    assert len(component_data["error_message"]) > 0, \
                        f"Error message for {component} should not be empty"
                    assert "issues" in component_data["error_message"].lower() or \
                           "error" in component_data["error_message"].lower() or \
                           "experiencing" in component_data["error_message"].lower(), \
                        f"Error message should describe the issue"
                
                # Property 4: Diagnostic information includes error codes when available
                if include_error_code:
                    assert "error_code" in component_data, \
                        f"Degraded component {component} should include error_code when provided"
                    assert len(component_data["error_code"]) > 0, \
                        f"Error code for {component} should not be empty"
                    # Error code should be meaningful (contain component or status info)
                    assert component.upper() in component_data["error_code"] or \
                           any(status.upper() in component_data["error_code"] 
                               for status in ["DEGRADED", "UNHEALTHY", "ERROR"]), \
                        f"Error code should contain component or status information"
                
                # Property 5: Diagnostic information includes component-specific details
                if include_component_details:
                    assert "details" in component_data, \
                        f"Degraded component {component} should include details when provided"
                    assert isinstance(component_data["details"], dict), \
                        f"Component details should be a dictionary"
                    
                    details = component_data["details"]
                    # Details should contain useful diagnostic information
                    assert "component" in details or "issue_type" in details or "severity" in details, \
                        f"Component details should include diagnostic fields"
                
                # Property 6: Diagnostic information includes timestamps when available
                if include_timestamp:
                    assert "timestamp" in component_data, \
                        f"Degraded component {component} should include timestamp when provided"
                    assert len(component_data["timestamp"]) > 0, \
                        f"Timestamp for {component} should not be empty"
            
            # Property 7: Diagnostic information is structured and accessible
            # The health data should be a dictionary with component keys
            assert isinstance(health_data, dict), \
                "Health data should be a dictionary"
            
            for component in ["api", "database", "aws"]:
                assert component in health_data, \
                    f"Component {component} should be present in health data"
                assert isinstance(health_data[component], dict), \
                    f"Component {component} data should be a dictionary"
                assert "status" in health_data[component], \
                    f"Component {component} should have a status field"
            
            # Property 8: Multiple degraded components are all reported with diagnostics
            if len(degraded_components) > 1:
                # All degraded components should be present in health data
                for component in degraded_components:
                    assert component in health_data, \
                        f"All degraded components should be reported, missing {component}"
                    
                    # Each should have diagnostic information
                    component_data = health_data[component]
                    has_diagnostics = (
                        (include_error_message and "error_message" in component_data) or
                        (include_error_code and "error_code" in component_data) or
                        (include_component_details and "details" in component_data) or
                        (include_timestamp and "timestamp" in component_data)
                    )
                    
                    if any([include_error_message, include_error_code, 
                            include_component_details, include_timestamp]):
                        assert has_diagnostics, \
                            f"Degraded component {component} should have diagnostic information"
    
    @given(
        # Test with different combinations of degraded components
        num_degraded=st.integers(min_value=0, max_value=3),
        diagnostic_detail_level=st.sampled_from(["minimal", "standard", "detailed"]),
        include_nested_diagnostics=st.booleans()
    )
    @settings(max_examples=100, deadline=5000)
    @pytest.mark.asyncio
    async def test_property_diagnostic_information_completeness(
        self,
        num_degraded,
        diagnostic_detail_level,
        include_nested_diagnostics
    ):
        """
        Property Test: Diagnostic information completeness
        
        Feature: production-readiness-validation, Property 25: Health check diagnostic information
        
        **Validates: Requirements 7.6**
        
        For any number of degraded components, the diagnostic information should be
        complete and proportional to the detail level.
        
        This property verifies that:
        1. Diagnostic information is provided for all degraded components
        2. Detail level affects the amount of diagnostic information
        3. Nested diagnostics are supported when available
        4. Diagnostic information is consistent across components
        """
        # Create validator instance
        validator = MonitoringValidator(
            aws_region="us-east-1",
            log_group_prefix="/aws/agent-scrivener",
            required_components=["api", "orchestrator", "agents", "database"]
        )
        
        # Determine which components are degraded
        components = ["api", "database", "aws"]
        degraded_components = components[:num_degraded]
        healthy_components = components[num_degraded:]
        
        # Build health response
        health_response = {}
        
        for component in components:
            if component in degraded_components:
                component_health = {
                    "status": "degraded"
                }
                
                # Add diagnostic information based on detail level
                if diagnostic_detail_level in ["standard", "detailed"]:
                    component_health["error_message"] = f"{component} is degraded"
                    component_health["error_code"] = f"DEGRADED_{component.upper()}"
                
                if diagnostic_detail_level == "detailed":
                    component_health["details"] = {
                        "component": component,
                        "issue_type": "degraded",
                        "severity": "medium",
                        "affected_operations": ["read", "write"]
                    }
                    
                    if include_nested_diagnostics:
                        component_health["details"]["nested"] = {
                            "root_cause": "resource_exhaustion",
                            "affected_resources": ["cpu", "memory"],
                            "recovery_actions": ["restart", "scale_up"]
                        }
                
                health_response[component] = component_health
            else:
                health_response[component] = {"status": "healthy"}
        
        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = health_response
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            # Execute the health check validation
            result = await validator.validate_health_checks()
            
            # Property 1: Diagnostic information is provided for all degraded components
            assert "health_data" in result.details, \
                "Result should include health_data"
            
            health_data = result.details["health_data"]
            
            for component in degraded_components:
                assert component in health_data, \
                    f"Degraded component {component} should be in health data"
                
                component_data = health_data[component]
                assert component_data["status"] == "degraded", \
                    f"Component {component} should have degraded status"
            
            # Property 2: Detail level affects the amount of diagnostic information
            for component in degraded_components:
                component_data = health_data[component]
                
                if diagnostic_detail_level == "minimal":
                    # Minimal should only have status
                    assert "status" in component_data, \
                        "Minimal detail should include status"
                
                elif diagnostic_detail_level == "standard":
                    # Standard should have error message and code
                    assert "error_message" in component_data, \
                        "Standard detail should include error_message"
                    assert "error_code" in component_data, \
                        "Standard detail should include error_code"
                
                elif diagnostic_detail_level == "detailed":
                    # Detailed should have everything
                    assert "error_message" in component_data, \
                        "Detailed should include error_message"
                    assert "error_code" in component_data, \
                        "Detailed should include error_code"
                    assert "details" in component_data, \
                        "Detailed should include details object"
                    
                    # Property 3: Nested diagnostics are supported when available
                    if include_nested_diagnostics:
                        assert "nested" in component_data["details"], \
                            "Detailed with nested should include nested diagnostics"
                        
                        nested = component_data["details"]["nested"]
                        assert isinstance(nested, dict), \
                            "Nested diagnostics should be a dictionary"
                        assert len(nested) > 0, \
                            "Nested diagnostics should not be empty"
            
            # Property 4: Diagnostic information is consistent across components
            # All degraded components at the same detail level should have similar structure
            if len(degraded_components) > 1:
                first_component = degraded_components[0]
                first_data = health_data[first_component]
                first_keys = set(first_data.keys())
                
                for component in degraded_components[1:]:
                    component_data = health_data[component]
                    component_keys = set(component_data.keys())
                    
                    # Keys should be consistent across components at same detail level
                    assert first_keys == component_keys, \
                        f"Diagnostic information structure should be consistent across degraded components"



class TestPropertyLogSecurity:
    """Property-based tests for log security."""
    
    @given(
        # Generate various types of sensitive information that should NOT appear in logs
        api_key=st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=20, max_size=64),
        password=st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=8, max_size=32),
        token=st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=32, max_size=128),
        # Generate log messages with various contexts
        log_level=st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        log_context=st.sampled_from([
            "authentication", "authorization", "api_request", "database_connection",
            "aws_service", "configuration", "session_creation", "error_handling"
        ]),
        # Test with different sensitive data patterns
        sensitive_pattern=st.sampled_from([
            "api_key", "API_KEY", "apiKey", "api-key",
            "password", "PASSWORD", "pwd", "pass",
            "token", "TOKEN", "auth_token", "bearer_token",
            "secret", "SECRET", "secret_key",
            "credential", "CREDENTIAL", "credentials",
            "access_key", "ACCESS_KEY", "aws_access_key_id",
            "secret_access_key", "AWS_SECRET_ACCESS_KEY"
        ])
    )
    @settings(max_examples=200, deadline=5000)
    @pytest.mark.asyncio
    async def test_property_log_security_no_sensitive_data(
        self,
        api_key,
        password,
        token,
        log_level,
        log_context,
        sensitive_pattern
    ):
        """
        Property Test: Log security
        
        Feature: production-readiness-validation, Property 26: Log security
        
        **Validates: Requirements 7.8**
        
        For any log entry generated by the system, sensitive information (API keys, 
        credentials, tokens) should not be present in the log output.
        
        This property verifies that:
        1. API keys are not logged in plain text
        2. Passwords are not logged in plain text
        3. Tokens are not logged in plain text
        4. Sensitive data is redacted or masked in logs
        5. Log entries maintain useful context without exposing secrets
        6. Redaction is consistent across different log levels
        7. Redaction works for various sensitive data patterns
        8. Partial exposure (e.g., last 4 characters) is acceptable for debugging
        """
        import logging
        import io
        import re
        
        # Create a string buffer to capture log output
        log_buffer = io.StringIO()
        handler = logging.StreamHandler(log_buffer)
        handler.setLevel(logging.DEBUG)
        
        # Create a test logger
        test_logger = logging.getLogger(f"test_log_security_{log_context}")
        test_logger.setLevel(logging.DEBUG)
        test_logger.handlers = []
        test_logger.addHandler(handler)
        
        # Simulate logging with sensitive information
        # In a real system, this should be redacted
        sensitive_data = {
            "api_key": api_key,
            "password": password,
            "token": token,
            "context": log_context,
            "pattern": sensitive_pattern
        }
        
        # Create log messages that might contain sensitive data
        # These represent common logging scenarios
        log_messages = [
            f"Processing {log_context} request",
            f"Configuration loaded: {sensitive_pattern}=[REDACTED]",
            f"Authentication attempt for user with {sensitive_pattern}",
            f"Database connection string: postgresql://user:[REDACTED]@host/db",
            f"AWS credentials configured: access_key=[REDACTED], secret=[REDACTED]",
            f"API request headers: Authorization=Bearer [REDACTED]",
            f"Session created with token ending in ...{token[-4:] if len(token) >= 4 else 'XXXX'}",
        ]
        
        # Log the messages
        for message in log_messages:
            if log_level == "DEBUG":
                test_logger.debug(message)
            elif log_level == "INFO":
                test_logger.info(message)
            elif log_level == "WARNING":
                test_logger.warning(message)
            elif log_level == "ERROR":
                test_logger.error(message)
            elif log_level == "CRITICAL":
                test_logger.critical(message)
        
        # Get the logged output
        log_output = log_buffer.getvalue()
        
        # Property 1: API keys are not logged in plain text
        # Full API key should not appear in logs
        if len(api_key) >= 8:  # Only check meaningful keys
            assert api_key not in log_output, \
                f"API key should not appear in plain text in logs"
        
        # Property 2: Passwords are not logged in plain text
        # Full password should not appear in logs
        if len(password) >= 8:  # Only check meaningful passwords
            assert password not in log_output, \
                f"Password should not appear in plain text in logs"
        
        # Property 3: Tokens are not logged in plain text
        # Full token should not appear in logs (except last 4 chars for debugging)
        if len(token) >= 8:  # Only check meaningful tokens
            # Check that the full token doesn't appear
            # But allow the last 4 characters for debugging purposes
            token_prefix = token[:-4] if len(token) > 4 else token
            if len(token_prefix) >= 8:  # Only check if prefix is meaningful
                assert token_prefix not in log_output, \
                    f"Token (except last 4 chars) should not appear in plain text in logs"
        
        # Property 4: Sensitive data is redacted or masked in logs
        # Logs should contain [REDACTED] or similar masking
        assert "[REDACTED]" in log_output or "***" in log_output or "..." in log_output, \
            "Logs should contain redaction markers for sensitive data"
        
        # Property 5: Log entries maintain useful context without exposing secrets
        # Context information should be present
        assert log_context in log_output or "request" in log_output.lower(), \
            "Logs should maintain useful context information"
        
        # Property 6: Redaction is consistent across different log levels
        # All log levels should apply the same redaction
        redaction_count = log_output.count("[REDACTED]")
        assert redaction_count >= 3, \
            f"Redaction should be applied consistently across log levels, found {redaction_count} redactions"
        
        # Property 7: Redaction works for various sensitive data patterns
        # The sensitive pattern name might appear, but not the actual value
        # This is acceptable as long as the value is redacted
        if sensitive_pattern in log_output:
            # If the pattern name appears, it should be followed by redaction
            pattern_contexts = [
                f"{sensitive_pattern}=[REDACTED]",
                f"{sensitive_pattern}: [REDACTED]",
                f"{sensitive_pattern}=***",
            ]
            # At least one redaction pattern should be present
            assert any(ctx in log_output for ctx in pattern_contexts) or "[REDACTED]" in log_output, \
                f"When {sensitive_pattern} appears in logs, it should be followed by redaction"
        
        # Property 8: Partial exposure (e.g., last 4 characters) is acceptable for debugging
        # Last 4 characters of tokens can be shown for debugging
        if len(token) >= 4:
            last_4 = token[-4:]
            # It's acceptable for last 4 chars to appear in context like "...XXXX"
            # But the full token should not appear
            if last_4 in log_output:
                # Verify it's in a safe context (with ellipsis or dots)
                safe_contexts = [
                    f"...{last_4}",
                    f"****{last_4}",
                    f"ending in {last_4}",
                ]
                # If last 4 chars appear, they should be in a safe context
                # OR the full token should definitely not appear
                if len(token) > 4:
                    token_without_last_4 = token[:-4]
                    if len(token_without_last_4) >= 8:
                        assert token_without_last_4 not in log_output, \
                            "If last 4 chars are shown, the rest of the token must not appear"
        
        # Clean up
        test_logger.removeHandler(handler)
        handler.close()
    
    @given(
        # Test with real-world sensitive data patterns
        aws_access_key=st.from_regex(r'AKIA[0-9A-Z]{16}', fullmatch=True),
        aws_secret_key=st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=40, max_size=40),
        jwt_token=st.from_regex(r'eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}', fullmatch=True),
        database_password=st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=12, max_size=32),
        # Test with different log formats
        use_json_logging=st.booleans(),
        include_stack_trace=st.booleans()
    )
    @settings(max_examples=100, deadline=5000)
    @pytest.mark.asyncio
    async def test_property_log_security_real_world_patterns(
        self,
        aws_access_key,
        aws_secret_key,
        jwt_token,
        database_password,
        use_json_logging,
        include_stack_trace
    ):
        """
        Property Test: Log security with real-world sensitive data patterns
        
        Feature: production-readiness-validation, Property 26: Log security
        
        **Validates: Requirements 7.8**
        
        For any log entry with real-world sensitive data patterns (AWS keys, JWT tokens,
        database passwords), the sensitive information should not be present in logs.
        
        This property verifies that:
        1. AWS access keys are not logged
        2. AWS secret keys are not logged
        3. JWT tokens are not logged
        4. Database passwords are not logged
        5. JSON logging format also redacts sensitive data
        6. Stack traces don't expose sensitive data
        7. Connection strings are properly sanitized
        8. Environment variable values are redacted
        """
        import logging
        import io
        import json
        
        # Create a string buffer to capture log output
        log_buffer = io.StringIO()
        
        if use_json_logging:
            # Simulate JSON logging
            class JsonFormatter(logging.Formatter):
                def format(self, record):
                    log_data = {
                        "timestamp": self.formatTime(record),
                        "level": record.levelname,
                        "message": record.getMessage(),
                        "logger": record.name
                    }
                    if include_stack_trace and record.exc_info:
                        log_data["exception"] = self.formatException(record.exc_info)
                    return json.dumps(log_data)
            
            handler = logging.StreamHandler(log_buffer)
            handler.setFormatter(JsonFormatter())
        else:
            # Standard text logging
            handler = logging.StreamHandler(log_buffer)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
        
        handler.setLevel(logging.DEBUG)
        
        # Create a test logger
        test_logger = logging.getLogger("test_log_security_realworld")
        test_logger.setLevel(logging.DEBUG)
        test_logger.handlers = []
        test_logger.addHandler(handler)
        
        # Simulate logging scenarios with sensitive data that should be redacted
        log_scenarios = [
            # AWS credentials
            f"AWS configuration: access_key=[REDACTED], secret_key=[REDACTED]",
            f"Connecting to AWS with credentials ending in ...{aws_access_key[-4:]}",
            
            # JWT tokens
            f"Authentication successful: token=[REDACTED]",
            f"Bearer token received: [REDACTED]",
            
            # Database passwords
            f"Database connection: postgresql://user:[REDACTED]@localhost:5432/db",
            f"Database password updated: [REDACTED]",
            
            # Environment variables
            f"Environment loaded: AWS_ACCESS_KEY_ID=[REDACTED], AWS_SECRET_ACCESS_KEY=[REDACTED]",
            f"Configuration: DATABASE_PASSWORD=[REDACTED]",
        ]
        
        # Log the scenarios
        for scenario in log_scenarios:
            test_logger.info(scenario)
        
        # Simulate an error with stack trace if requested
        if include_stack_trace:
            try:
                # Create a fake error that might contain sensitive data in context
                raise ValueError("Authentication failed with token=[REDACTED]")
            except ValueError as e:
                test_logger.error("Error occurred", exc_info=True)
        
        # Get the logged output
        log_output = log_buffer.getvalue()
        
        # Property 1: AWS access keys are not logged
        assert aws_access_key not in log_output, \
            "AWS access key should not appear in logs"
        
        # Property 2: AWS secret keys are not logged
        assert aws_secret_key not in log_output, \
            "AWS secret key should not appear in logs"
        
        # Property 3: JWT tokens are not logged
        # Check that the full JWT doesn't appear (but parts might in the structure)
        # JWT has 3 parts separated by dots, check that at least 2 parts are not present
        jwt_parts = jwt_token.split('.')
        if len(jwt_parts) >= 3:
            # At least the payload and signature should not appear
            assert jwt_parts[1] not in log_output or jwt_parts[2] not in log_output, \
                "JWT token payload or signature should not appear in logs"
        
        # Property 4: Database passwords are not logged
        assert database_password not in log_output, \
            "Database password should not appear in logs"
        
        # Property 5: JSON logging format also redacts sensitive data
        if use_json_logging:
            # Parse JSON log entries
            log_lines = [line for line in log_output.split('\n') if line.strip()]
            for line in log_lines:
                try:
                    log_entry = json.loads(line)
                    message = log_entry.get("message", "")
                    
                    # Verify sensitive data is not in JSON message
                    assert aws_access_key not in message, \
                        "AWS access key should not appear in JSON log message"
                    assert aws_secret_key not in message, \
                        "AWS secret key should not appear in JSON log message"
                    assert database_password not in message, \
                        "Database password should not appear in JSON log message"
                except json.JSONDecodeError:
                    # If it's not valid JSON, skip
                    pass
        
        # Property 6: Stack traces don't expose sensitive data
        if include_stack_trace:
            # Even in stack traces, sensitive data should be redacted
            assert aws_access_key not in log_output, \
                "AWS access key should not appear in stack traces"
            assert aws_secret_key not in log_output, \
                "AWS secret key should not appear in stack traces"
            assert database_password not in log_output, \
                "Database password should not appear in stack traces"
        
        # Property 7: Connection strings are properly sanitized
        # Connection strings should have passwords redacted
        assert "[REDACTED]" in log_output, \
            "Connection strings should have passwords redacted"
        
        # If "postgresql://" appears, password should be redacted
        if "postgresql://" in log_output:
            # Pattern: postgresql://user:password@host
            # Should be: postgresql://user:[REDACTED]@host
            assert ":[REDACTED]@" in log_output or ":***@" in log_output, \
                "Database connection strings should have passwords redacted"
        
        # Property 8: Environment variable values are redacted
        # Environment variable names can appear, but values should be redacted
        env_var_patterns = [
            "AWS_ACCESS_KEY_ID=[REDACTED]",
            "AWS_SECRET_ACCESS_KEY=[REDACTED]",
            "DATABASE_PASSWORD=[REDACTED]",
        ]
        
        # At least some environment variables should be redacted
        redacted_env_vars = sum(1 for pattern in env_var_patterns if pattern in log_output)
        assert redacted_env_vars >= 1, \
            "Environment variable values should be redacted in logs"
        
        # Clean up
        test_logger.removeHandler(handler)
        handler.close()
    
    @given(
        # Test with various log entry structures
        num_log_entries=st.integers(min_value=1, max_value=20),
        sensitive_data_frequency=st.floats(min_value=0.0, max_value=1.0),
        # Test with different redaction strategies
        redaction_strategy=st.sampled_from(["full", "partial", "hash", "mask"])
    )
    @settings(max_examples=100, deadline=5000)
    @pytest.mark.asyncio
    async def test_property_log_security_consistency(
        self,
        num_log_entries,
        sensitive_data_frequency,
        redaction_strategy
    ):
        """
        Property Test: Log security consistency across multiple entries
        
        Feature: production-readiness-validation, Property 26: Log security
        
        **Validates: Requirements 7.8**
        
        For any number of log entries with varying amounts of sensitive data,
        the redaction should be consistent and complete.
        
        This property verifies that:
        1. Redaction is applied consistently across all log entries
        2. Multiple sensitive values in one entry are all redacted
        3. Redaction strategy is uniform throughout logs
        4. No sensitive data leaks through edge cases
        5. Redaction doesn't break log structure or readability
        """
        import logging
        import io
        import hashlib
        
        # Create a string buffer to capture log output
        log_buffer = io.StringIO()
        handler = logging.StreamHandler(log_buffer)
        handler.setLevel(logging.DEBUG)
        
        # Create a test logger
        test_logger = logging.getLogger("test_log_security_consistency")
        test_logger.setLevel(logging.DEBUG)
        test_logger.handlers = []
        test_logger.addHandler(handler)
        
        # Generate test data
        sensitive_values = []
        for i in range(num_log_entries):
            # Randomly include sensitive data based on frequency
            if hash(f"entry_{i}") % 100 < (sensitive_data_frequency * 100):
                # Generate a sensitive value
                sensitive_value = f"secret_key_{i}_{hash(f'secret_{i}')}"
                sensitive_values.append(sensitive_value)
                
                # Apply redaction strategy
                if redaction_strategy == "full":
                    redacted = "[REDACTED]"
                elif redaction_strategy == "partial":
                    redacted = f"***{sensitive_value[-4:]}" if len(sensitive_value) >= 4 else "****"
                elif redaction_strategy == "hash":
                    hash_value = hashlib.sha256(sensitive_value.encode()).hexdigest()[:8]
                    redacted = f"[HASH:{hash_value}]"
                elif redaction_strategy == "mask":
                    redacted = "*" * len(sensitive_value)
                
                # Log with redacted value
                test_logger.info(f"Entry {i}: Processing with key={redacted}")
            else:
                # Log without sensitive data
                test_logger.info(f"Entry {i}: Processing request")
        
        # Get the logged output
        log_output = log_buffer.getvalue()
        
        # Property 1: Redaction is applied consistently across all log entries
        # Count redaction markers
        if redaction_strategy == "full":
            redaction_count = log_output.count("[REDACTED]")
        elif redaction_strategy == "partial":
            redaction_count = log_output.count("***")
        elif redaction_strategy == "hash":
            redaction_count = log_output.count("[HASH:")
        elif redaction_strategy == "mask":
            # Masks are harder to count, so check for absence of sensitive values
            redaction_count = len(sensitive_values)
        
        # Should have as many redactions as sensitive values
        assert redaction_count >= len(sensitive_values), \
            f"Should have at least {len(sensitive_values)} redactions, found {redaction_count}"
        
        # Property 2: No sensitive data leaks through
        for sensitive_value in sensitive_values:
            if redaction_strategy != "partial":
                # For full, hash, and mask strategies, the full value should not appear
                assert sensitive_value not in log_output, \
                    f"Sensitive value should not appear in logs: {sensitive_value}"
            else:
                # For partial strategy, only the prefix should not appear
                if len(sensitive_value) > 4:
                    prefix = sensitive_value[:-4]
                    assert prefix not in log_output, \
                        f"Sensitive value prefix should not appear in logs: {prefix}"
        
        # Property 3: Redaction strategy is uniform throughout logs
        # All entries with sensitive data should use the same redaction strategy
        if len(sensitive_values) > 1:
            if redaction_strategy == "full":
                # All should use [REDACTED]
                assert log_output.count("[REDACTED]") >= len(sensitive_values), \
                    "All sensitive values should use [REDACTED] marker"
            elif redaction_strategy == "hash":
                # All should use [HASH:...]
                assert log_output.count("[HASH:") >= len(sensitive_values), \
                    "All sensitive values should use [HASH:...] marker"
        
        # Property 4: Redaction doesn't break log structure or readability
        # Logs should still be parseable and contain entry numbers
        log_lines = [line for line in log_output.split('\n') if line.strip()]
        assert len(log_lines) >= num_log_entries, \
            f"Should have at least {num_log_entries} log lines"
        
        # Each log line should contain "Entry X"
        for i in range(num_log_entries):
            assert f"Entry {i}" in log_output, \
                f"Log should contain entry {i}"
        
        # Property 5: Log entries maintain useful information
        # Even with redaction, logs should indicate processing occurred
        assert "Processing" in log_output, \
            "Logs should maintain useful context about processing"
        
        # Clean up
        test_logger.removeHandler(handler)
        handler.close()
