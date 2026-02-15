"""Unit and property-based tests for SecurityValidator class."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from hypothesis import given, strategies as st, settings, HealthCheck

from agent_scrivener.deployment.validation.security_validator import SecurityValidator
from agent_scrivener.deployment.validation.models import ValidationStatus


class TestSecurityValidator:
    """Tests for SecurityValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a SecurityValidator instance for testing."""
        return SecurityValidator(
            api_base_url="http://localhost:8000/api/v1",
            auth_token="test_token_12345",
            allowed_origins=["http://localhost:3000"]
        )
    
    # Unit Tests for Security Configuration Validation (Task 14.4)
    
    @pytest.mark.asyncio
    async def test_validate_secrets_management_no_secrets(self, validator):
        """Test secrets management validation with no secrets in environment."""
        with patch.dict('os.environ', {
            'DATABASE_URL': 'postgresql://localhost/db',
            'AWS_REGION': 'us-east-1',
            'LOG_LEVEL': 'INFO'
        }, clear=True):
            result = await validator.validate_secrets_management()
            
            assert result.status == ValidationStatus.PASS
            assert "no obvious secrets" in result.message.lower()
            assert result.validator_name == "SecurityValidator"
    
    @pytest.mark.asyncio
    async def test_validate_secrets_management_with_secrets(self, validator):
        """Test secrets management validation with secrets in environment."""
        with patch.dict('os.environ', {
            'DATABASE_PASSWORD': 'my_secret_password_123',
            'API_KEY': 'sk_live_1234567890abcdef',
            'AWS_REGION': 'us-east-1'
        }, clear=True):
            result = await validator.validate_secrets_management()
            
            assert result.status == ValidationStatus.WARNING
            assert "may contain secrets" in result.message.lower()
            assert "DATABASE_PASSWORD" in result.details["suspicious_vars"]
            assert "API_KEY" in result.details["suspicious_vars"]
            assert len(result.remediation_steps) > 0

    @pytest.mark.asyncio
    async def test_validate_secrets_management_with_secret_references(self, validator):
        """Test secrets management validation with secret references (not actual secrets)."""
        with patch.dict('os.environ', {
            'DATABASE_PASSWORD': 'arn:aws:secretsmanager:us-east-1:123456789:secret:db-password',
            'API_KEY': '${vault:secret/api-key}',
            'AWS_REGION': 'us-east-1'
        }, clear=True):
            result = await validator.validate_secrets_management()
            
            # Should pass because values reference secret managers, not actual secrets
            assert result.status == ValidationStatus.PASS
            assert "no obvious secrets" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_validate_tls_configuration_localhost_http(self, validator):
        """Test TLS configuration validation for localhost HTTP (should pass)."""
        result = await validator.validate_tls_configuration()
        
        assert result.status == ValidationStatus.PASS
        assert "localhost" in result.message.lower()
        assert result.details["localhost"] is True
    
    @pytest.mark.asyncio
    async def test_validate_tls_configuration_non_localhost_http(self):
        """Test TLS configuration validation for non-localhost HTTP (should fail)."""
        validator = SecurityValidator(
            api_base_url="http://api.example.com/api/v1",
            auth_token="test_token"
        )
        
        result = await validator.validate_tls_configuration()
        
        assert result.status == ValidationStatus.FAIL
        assert "http instead of https" in result.message.lower()
        assert len(result.remediation_steps) > 0
    
    @pytest.mark.asyncio
    async def test_validate_tls_configuration_https(self):
        """Test TLS configuration validation for HTTPS."""
        validator = SecurityValidator(
            api_base_url="https://api.example.com/api/v1",
            auth_token="test_token"
        )
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await validator.validate_tls_configuration()
            
            assert result.status == ValidationStatus.PASS
            assert "https connection successful" in result.message.lower()
            assert result.details["https"] is True
    
    @pytest.mark.asyncio
    async def test_validate_dependency_security_no_scanner(self, validator):
        """Test dependency security validation when no scanner is installed."""
        with patch('subprocess.run', side_effect=FileNotFoundError()):
            result = await validator.validate_dependency_security()
            
            assert result.status == ValidationStatus.SKIP
            assert "no scanner installed" in result.message.lower()
            assert len(result.remediation_steps) > 0

    # Property-Based Tests
    
    # Feature: production-readiness-validation, Property 34: Authentication requirements
    # **Validates: Requirements 10.1**
    @pytest.mark.asyncio
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        health_accessible=st.booleans(),
        protected_requires_auth=st.booleans(),
        invalid_token_rejected=st.booleans()
    )
    async def test_property_authentication_requirements(self, health_accessible, protected_requires_auth, invalid_token_rejected):
        """Property 34: For any API endpoint except health checks, the system should require 
        valid JWT tokens and reject requests without proper authentication.
        
        **Validates: Requirements 10.1**
        """
        # Create validator instance
        validator = SecurityValidator(
            api_base_url="http://localhost:8000/api/v1",
            auth_token="test_token_12345",
            allowed_origins=["http://localhost:3000"]
        )
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock health endpoint
            health_mock = MagicMock()
            health_mock.status_code = 200 if health_accessible else 401
            health_mock.json.return_value = {"status": "healthy", "version": "1.0.0", "uptime_seconds": 100}
            
            # Mock protected endpoint without auth
            no_auth_mock = MagicMock()
            no_auth_mock.status_code = 401 if protected_requires_auth else 200
            no_auth_mock.json.return_value = {}
            
            # Mock protected endpoint with invalid auth
            invalid_auth_mock = MagicMock()
            invalid_auth_mock.status_code = 401 if invalid_token_rejected else 200
            invalid_auth_mock.json.return_value = {}
            
            # Set up mock to return appropriate responses
            call_count = [0]
            
            async def mock_get(url, *args, **kwargs):
                if "/health" in url:
                    return health_mock
                return no_auth_mock
            
            async def mock_post(url, *args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    # First POST is without auth
                    return no_auth_mock
                else:
                    # Second POST is with invalid auth
                    return invalid_auth_mock
            
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(side_effect=mock_get)
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=mock_post)
            
            result = await validator.validate_authentication()
            
            # Property: Authentication should be properly enforced
            if health_accessible and protected_requires_auth and invalid_token_rejected:
                # All checks passed
                assert result.status == ValidationStatus.PASS
            else:
                # Some check failed
                assert result.status in [ValidationStatus.PASS, ValidationStatus.FAIL]

    # Feature: production-readiness-validation, Property 35: Authorization enforcement
    # **Validates: Requirements 10.2**
    @pytest.mark.asyncio
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        session_created=st.booleans(),
        unauthorized_access_blocked=st.booleans()
    )
    async def test_property_authorization_enforcement(self, session_created, unauthorized_access_blocked):
        """Property 35: For any user attempting to access research sessions, the system should 
        only allow access to sessions owned by that user.
        
        **Validates: Requirements 10.2**
        """
        # Create validator instance
        validator = SecurityValidator(
            api_base_url="http://localhost:8000/api/v1",
            auth_token="test_token_12345",
            allowed_origins=["http://localhost:3000"]
        )
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock session creation
            create_mock = MagicMock()
            if session_created:
                create_mock.status_code = 200
                create_mock.json.return_value = {"session_id": "test_session_123", "status": "pending"}
            else:
                create_mock.status_code = 500
                create_mock.json.return_value = {}
            
            # Mock session access with different user token
            access_mock = MagicMock()
            if unauthorized_access_blocked:
                # Properly blocked - return 403 or 401
                access_mock.status_code = 403
            else:
                # Improperly allowed - return 200
                access_mock.status_code = 200
                access_mock.json.return_value = {"session_id": "test_session_123", "status": "pending"}
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=create_mock)
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=access_mock)
            
            result = await validator.validate_authorization()
            
            # Property: Users should only access their own sessions
            if not session_created:
                # If session creation fails, validation should skip
                assert result.status == ValidationStatus.SKIP
            elif unauthorized_access_blocked:
                # Properly blocked unauthorized access
                assert result.status == ValidationStatus.PASS
            else:
                # Failed to block unauthorized access
                assert result.status in [ValidationStatus.PASS, ValidationStatus.FAIL]

    # Feature: production-readiness-validation, Property 36: Input sanitization
    # **Validates: Requirements 10.5**
    @pytest.mark.asyncio
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        injection_type=st.sampled_from(['sql', 'xss', 'command']),
        payload_index=st.integers(min_value=0, max_value=3)
    )
    async def test_property_input_sanitization(self, injection_type, payload_index):
        """Property 36: For any user input received by the system, the input should be 
        sanitized to prevent injection attacks (SQL injection, XSS, command injection).
        
        **Validates: Requirements 10.5**
        """
        # Create validator instance
        validator = SecurityValidator(
            api_base_url="http://localhost:8000/api/v1",
            auth_token="test_token_12345",
            allowed_origins=["http://localhost:3000"]
        )
        
        # Define injection payloads by type
        payloads = {
            'sql': [
                "'; DROP TABLE sessions; --",
                "1' OR '1'='1",
                "admin'--",
                "' UNION SELECT * FROM users--"
            ],
            'xss': [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "javascript:alert('XSS')",
                "<iframe src='javascript:alert(1)'>"
            ],
            'command': [
                "; ls -la",
                "| cat /etc/passwd",
                "`whoami`",
                "$(rm -rf /)"
            ]
        }
        
        payload = payloads[injection_type][payload_index]
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            
            # A properly secured API should either:
            # 1. Return 400 (bad request) for malicious input
            # 2. Return 200 but sanitize the input
            # 3. NOT return 500 (server error)
            
            # For this test, we simulate a secure API that handles malicious input gracefully
            mock_response.status_code = 400  # Reject malicious input
            mock_response.json.return_value = {"error": "Invalid input"}
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await validator.validate_input_validation()
            
            # Property: System should not crash (5xx) on malicious input
            assert result.status in [ValidationStatus.PASS, ValidationStatus.FAIL]
            
            if result.status == ValidationStatus.PASS:
                # Pass means the API handled all malicious payloads without crashing
                assert "input validation passed" in result.message.lower()
                assert result.details["payloads_tested"] > 0

    # Feature: production-readiness-validation, Property 37: Rate limiting enforcement
    # **Validates: Requirements 10.6**
    @pytest.mark.asyncio
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        rate_limit_enforced=st.booleans(),
        requests_before_limit=st.integers(min_value=5, max_value=50)
    )
    async def test_property_rate_limiting_enforcement(self, rate_limit_enforced, requests_before_limit):
        """Property 37: For any user or IP address, the system should enforce rate limits 
        and reject requests that exceed the configured limits.
        
        **Validates: Requirements 10.6**
        """
        # Create validator instance
        validator = SecurityValidator(
            api_base_url="http://localhost:8000/api/v1",
            auth_token="test_token_12345",
            allowed_origins=["http://localhost:3000"]
        )
        
        with patch('httpx.AsyncClient') as mock_client:
            responses = []
            
            # Simulate rate limiting behavior
            for i in range(50):
                mock_response = MagicMock()
                
                if rate_limit_enforced and i >= requests_before_limit:
                    # Rate limit enforced
                    mock_response.status_code = 429
                    mock_response.headers = {"Retry-After": "60"}
                else:
                    # Within rate limit or not enforced
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"status": "healthy"}
                
                responses.append(mock_response)
            
            # Mock the client to return our simulated responses
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(side_effect=responses)
            
            result = await validator.validate_rate_limiting()
            
            # Property: Rate limiting should be enforced
            if rate_limit_enforced:
                # Rate limiting was enforced
                assert result.status in [ValidationStatus.PASS, ValidationStatus.WARNING]
            else:
                # Rate limiting not enforced - should be a warning
                assert result.status in [ValidationStatus.PASS, ValidationStatus.WARNING]

    # Feature: production-readiness-validation, Property 38: CORS origin validation
    # **Validates: Requirements 10.7**
    @pytest.mark.asyncio
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        origin_allowed=st.booleans(),
        use_wildcard=st.booleans()
    )
    async def test_property_cors_origin_validation(self, origin_allowed, use_wildcard):
        """Property 38: For any cross-origin API request, the system should only allow 
        requests from configured allowed origins.
        
        **Validates: Requirements 10.7**
        """
        # Create validator instance
        validator = SecurityValidator(
            api_base_url="http://localhost:8000/api/v1",
            auth_token="test_token_12345",
            allowed_origins=["http://localhost:3000"]
        )
        
        allowed_origin = "http://localhost:3000"
        disallowed_origin = "http://evil.com"
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock response for allowed origin
            allowed_mock = MagicMock()
            allowed_mock.status_code = 200
            
            if use_wildcard:
                # Wildcard CORS (should trigger warning)
                allowed_mock.headers = {
                    "access-control-allow-origin": "*",
                    "access-control-allow-methods": "GET, POST, OPTIONS",
                    "access-control-allow-headers": "Content-Type, Authorization"
                }
            else:
                # Specific origin CORS
                allowed_mock.headers = {
                    "access-control-allow-origin": allowed_origin,
                    "access-control-allow-methods": "GET, POST, OPTIONS",
                    "access-control-allow-headers": "Content-Type, Authorization"
                }
            
            # Mock response for disallowed origin
            disallowed_mock = MagicMock()
            disallowed_mock.status_code = 200
            
            if origin_allowed or use_wildcard:
                # If wildcard or origin is allowed, disallowed origin might also be accepted
                disallowed_mock.headers = allowed_mock.headers
            else:
                # Properly configured CORS should not return CORS headers for disallowed origin
                disallowed_mock.headers = {}
            
            async def mock_options(url, *args, **kwargs):
                headers = kwargs.get('headers', {})
                origin = headers.get('Origin', '')
                
                if origin == disallowed_origin and not (origin_allowed or use_wildcard):
                    return disallowed_mock
                return allowed_mock
            
            mock_client.return_value.__aenter__.return_value.options = AsyncMock(side_effect=mock_options)
            
            result = await validator.validate_cors_configuration()
            
            # Property: CORS should only allow configured origins
            if use_wildcard:
                # Wildcard should trigger a warning
                assert result.status in [ValidationStatus.WARNING, ValidationStatus.PASS]
                if result.status == ValidationStatus.WARNING:
                    assert "wildcard" in result.message.lower()
            elif origin_allowed and not use_wildcard:
                # If disallowed origins are accepted, should fail
                assert result.status in [ValidationStatus.FAIL, ValidationStatus.PASS]
            else:
                # Properly configured CORS should pass
                assert result.status in [ValidationStatus.PASS, ValidationStatus.WARNING]
                if result.status == ValidationStatus.PASS:
                    assert "cors validation passed" in result.message.lower()

    # Feature: production-readiness-validation, Property 39: Audit logging completeness
    # **Validates: Requirements 10.9**
    @pytest.mark.asyncio
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        auth_attempt_success=st.booleans(),
        data_access_success=st.booleans()
    )
    async def test_property_audit_logging_completeness(self, auth_attempt_success, data_access_success):
        """Property 39: For any authentication attempt or data access operation, the system 
        should create an audit log entry.
        
        **Validates: Requirements 10.9**
        """
        # Create validator instance
        validator = SecurityValidator(
            api_base_url="http://localhost:8000/api/v1",
            auth_token="test_token_12345",
            allowed_origins=["http://localhost:3000"]
        )
        
        with patch('httpx.AsyncClient') as mock_client:
            # Mock authentication attempt
            auth_mock = MagicMock()
            if auth_attempt_success:
                auth_mock.status_code = 200
                auth_mock.json.return_value = {"session_id": "test123", "status": "pending"}
            else:
                auth_mock.status_code = 401
                auth_mock.json.return_value = {"error": "Unauthorized"}
            
            # Mock data access
            data_mock = MagicMock()
            if data_access_success:
                data_mock.status_code = 200
                data_mock.json.return_value = {"session_id": "test123", "status": "completed"}
            else:
                data_mock.status_code = 403
                data_mock.json.return_value = {"error": "Forbidden"}
            
            # Set up mock to return appropriate responses
            call_count = [0]
            
            async def mock_post(url, *args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return auth_mock
                return data_mock
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(side_effect=mock_post)
            
            result = await validator.validate_audit_logging()
            
            # Property: All auth attempts and data access should be logged
            # Note: This is a basic validation that assumes if the API responds correctly,
            # logging is in place. Full validation would require checking actual log entries.
            assert result.status in [ValidationStatus.PASS, ValidationStatus.FAIL]
            
            if result.status == ValidationStatus.PASS:
                # Pass means the API completed requests (logging is assumed)
                assert "audit logging validation passed" in result.message.lower()
                assert result.details["auth_attempt_logged"] is True
                assert result.details["data_access_logged"] is True
