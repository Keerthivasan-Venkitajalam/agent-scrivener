"""
Unit tests for the centralized error handling system.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from agent_scrivener.utils.error_handler import (
    ErrorHandler, RetryConfig, CircuitBreakerConfig, CircuitBreaker,
    CircuitBreakerState, with_error_handling, error_handler
)
from agent_scrivener.models.errors import (
    ErrorCategory, ErrorSeverity, AgentScrivenerError,
    NetworkError, ExternalAPIError, ValidationError
)


class TestRetryConfig:
    """Test retry configuration."""
    
    def test_default_config(self):
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_factor == 2.0
        assert config.jitter is True
    
    def test_custom_config(self):
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            backoff_factor=1.5,
            jitter=False
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.backoff_factor == 1.5
        assert config.jitter is False


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""
    
    def test_default_config(self):
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60
        assert config.expected_exception == Exception
    
    def test_custom_config(self):
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=NetworkError
        )
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30
        assert config.expected_exception == NetworkError


class TestCircuitBreaker:
    """Test circuit breaker implementation."""
    
    @pytest.fixture
    def circuit_breaker(self):
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        return CircuitBreaker(config)
    
    @pytest.fixture
    def failing_function(self):
        def func():
            raise NetworkError("Test network error")
        return func
    
    @pytest.fixture
    def success_function(self):
        def func():
            return "success"
        return func
    
    def test_initial_state(self, circuit_breaker):
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.last_failure_time is None
    
    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker, success_function):
        result = await circuit_breaker.call(success_function)
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_failure_tracking(self, circuit_breaker, failing_function):
        # First failure
        with pytest.raises(NetworkError):
            await circuit_breaker.call(failing_function)
        
        assert circuit_breaker.failure_count == 1
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.last_failure_time is not None
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, circuit_breaker, failing_function):
        # Trigger failures to reach threshold
        for _ in range(2):
            with pytest.raises(NetworkError):
                await circuit_breaker.call(failing_function)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.failure_count == 2
    
    @pytest.mark.asyncio
    async def test_open_circuit_blocks_calls(self, circuit_breaker, failing_function, success_function):
        # Open the circuit
        for _ in range(2):
            with pytest.raises(NetworkError):
                await circuit_breaker.call(failing_function)
        
        # Now even successful functions should be blocked
        with pytest.raises(ExternalAPIError) as exc_info:
            await circuit_breaker.call(success_function)
        
        assert "Circuit breaker is OPEN" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_half_open_state_recovery(self, circuit_breaker, failing_function, success_function):
        # Open the circuit
        for _ in range(2):
            with pytest.raises(NetworkError):
                await circuit_breaker.call(failing_function)
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Next call should transition to half-open and succeed
        result = await circuit_breaker.call(success_function)
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_async_function_support(self, circuit_breaker):
        async def async_func():
            return "async_success"
        
        result = await circuit_breaker.call(async_func)
        assert result == "async_success"


class TestErrorHandler:
    """Test centralized error handler."""
    
    @pytest.fixture
    def handler(self):
        return ErrorHandler()
    
    @pytest.fixture
    def failing_operation(self):
        call_count = 0
        
        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary failure")
            return "success"
        
        operation.call_count = lambda: call_count
        return operation
    
    @pytest.fixture
    def always_failing_operation(self):
        def operation():
            raise ValidationError("Permanent failure")
        return operation
    
    def test_circuit_breaker_creation(self, handler):
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = handler.get_circuit_breaker("test_service", config)
        
        assert isinstance(breaker, CircuitBreaker)
        assert breaker.config.failure_threshold == 3
        
        # Should return same instance for same service
        breaker2 = handler.get_circuit_breaker("test_service")
        assert breaker is breaker2
    
    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self, handler, failing_operation):
        config = RetryConfig(max_retries=3, base_delay=0.01)
        
        result = await handler.with_retry(failing_operation, config, (NetworkError,))
        
        assert result == "success"
        assert failing_operation.call_count() == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, handler, always_failing_operation):
        config = RetryConfig(max_retries=2, base_delay=0.01)
        
        with pytest.raises(ValidationError):
            await handler.with_retry(always_failing_operation, config, (ValidationError,))
    
    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self, handler):
        call_times = []
        
        def operation():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise NetworkError("Temporary failure")
            return "success"
        
        config = RetryConfig(max_retries=3, base_delay=0.1, backoff_factor=2.0, jitter=False)
        
        start_time = time.time()
        result = await handler.with_retry(operation, config, (NetworkError,))
        
        assert result == "success"
        assert len(call_times) == 3
        
        # Check that delays increase exponentially
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        
        assert delay1 >= 0.1  # First retry delay
        assert delay2 >= 0.2  # Second retry delay (doubled)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, handler):
        call_count = 0
        
        def operation():
            nonlocal call_count
            call_count += 1
            raise NetworkError("Service unavailable")
        
        config = CircuitBreakerConfig(failure_threshold=2, expected_exception=NetworkError)
        
        # First two calls should fail normally
        for _ in range(2):
            with pytest.raises(NetworkError):
                await handler.with_circuit_breaker(operation, "test_service", config)
        
        # Third call should be blocked by circuit breaker
        with pytest.raises(ExternalAPIError) as exc_info:
            await handler.with_circuit_breaker(operation, "test_service", config)
        
        assert "Circuit breaker is OPEN" in str(exc_info.value)
        assert call_count == 2  # Operation wasn't called the third time
    
    @pytest.mark.asyncio
    async def test_combined_retry_and_circuit_breaker(self, handler):
        call_count = 0
        
        def operation():
            nonlocal call_count
            call_count += 1
            raise NetworkError("Service unavailable")
        
        retry_config = RetryConfig(max_retries=1, base_delay=0.01)
        circuit_config = CircuitBreakerConfig(failure_threshold=3, expected_exception=NetworkError)
        
        with pytest.raises(NetworkError):
            await handler.with_retry_and_circuit_breaker(
                operation, "test_service", retry_config, circuit_config, (NetworkError,)
            )
        
        # Should have retried once (2 total calls)
        assert call_count == 2
    
    def test_agent_error_handling(self, handler):
        error = NetworkError("Connection failed", session_id="test_session")
        
        response = handler.handle_agent_error("research_agent", error, "test_session")
        
        assert not response.success
        assert response.error.category == ErrorCategory.NETWORK
        assert response.error.severity == ErrorSeverity.MEDIUM
        assert response.error.agent_name == "research_agent"
        assert response.error.session_id == "test_session"
        assert len(response.suggested_actions) > 0
    
    def test_error_classification(self, handler):
        # Test network error classification
        network_error = ConnectionError("Connection timeout")
        details = handler._create_error_details(network_error, {})
        assert details.category == ErrorCategory.NETWORK
        
        # Test validation error classification
        validation_error = ValueError("Invalid input")
        details = handler._create_error_details(validation_error, {})
        assert details.category == ErrorCategory.VALIDATION
        
        # Test custom error classification
        custom_error = NetworkError("Custom network error")
        details = handler._create_error_details(custom_error, {})
        assert details.category == ErrorCategory.NETWORK
        assert details.severity == ErrorSeverity.MEDIUM
    
    def test_recovery_strategies(self, handler):
        # Network error strategies
        network_error = NetworkError("Connection failed")
        details = handler._create_error_details(network_error, {})
        strategies = handler._get_recovery_strategies(details)
        
        assert "Check network connectivity" in strategies
        assert "Retry with exponential backoff" in strategies
        
        # Validation error strategies
        validation_error = ValidationError("Invalid data")
        details = handler._create_error_details(validation_error, {})
        strategies = handler._get_recovery_strategies(details)
        
        assert "Validate input data format" in strategies
        assert "Check required fields" in strategies
    
    def test_error_stats_tracking(self, handler):
        # Generate some errors
        handler.handle_agent_error("agent1", NetworkError("Error 1"))
        handler.handle_agent_error("agent1", NetworkError("Error 2"))
        handler.handle_agent_error("agent2", ValidationError("Error 3"))
        
        stats = handler.get_error_stats()
        
        assert stats["agent1"]["network"] == 2
        assert stats["agent2"]["validation"] == 1
        
        # Test reset
        handler.reset_error_stats()
        stats = handler.get_error_stats()
        assert len(stats) == 0


class TestErrorHandlingDecorator:
    """Test error handling decorator."""
    
    @pytest.mark.asyncio
    async def test_async_function_decorator(self):
        call_count = 0
        
        @with_error_handling(
            retry_config=RetryConfig(max_retries=2, base_delay=0.01),
            error_types=(NetworkError,)
        )
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary failure")
            return "success"
        
        result = await test_function()
        assert result == "success"
        assert call_count == 3
    
    def test_sync_function_decorator(self):
        @with_error_handling()
        def test_function():
            raise NetworkError("Test error")
        
        with pytest.raises(NetworkError):
            test_function()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator(self):
        call_count = 0
        
        @with_error_handling(
            retry_config=RetryConfig(max_retries=0),  # No retries to avoid confusion
            circuit_breaker_service="test_service",
            circuit_config=CircuitBreakerConfig(failure_threshold=1),
            error_types=(NetworkError,)
        )
        async def test_function():
            nonlocal call_count
            call_count += 1
            raise NetworkError("Service error")
        
        # First call should fail normally
        with pytest.raises(NetworkError):
            await test_function()
        
        # Second call should be blocked by circuit breaker
        with pytest.raises(ExternalAPIError):
            await test_function()
        
        assert call_count == 1


class TestGlobalErrorHandler:
    """Test global error handler instance."""
    
    def test_global_instance_exists(self):
        assert error_handler is not None
        assert isinstance(error_handler, ErrorHandler)
    
    def test_global_instance_functionality(self):
        # Test that global instance works
        response = error_handler.handle_agent_error(
            "test_agent", 
            NetworkError("Test error")
        )
        
        assert not response.success
        assert response.error.agent_name == "test_agent"


@pytest.mark.asyncio
async def test_real_world_scenario():
    """Test a realistic error handling scenario."""
    handler = ErrorHandler()
    
    # Simulate a flaky external API
    call_count = 0
    
    async def flaky_api_call():
        nonlocal call_count
        call_count += 1
        
        if call_count <= 2:
            raise NetworkError("Temporary network issue")
        elif call_count <= 4:
            raise ExternalAPIError("API rate limit exceeded")
        else:
            return {"data": "success", "call_count": call_count}
    
    # Configure retry and circuit breaker
    retry_config = RetryConfig(max_retries=6, base_delay=0.01)
    circuit_config = CircuitBreakerConfig(
        failure_threshold=10,  # High threshold for this test
        expected_exception=(NetworkError, ExternalAPIError)
    )
    
    # Should eventually succeed after retries
    result = await handler.with_retry_and_circuit_breaker(
        flaky_api_call,
        "external_api",
        retry_config,
        circuit_config,
        (NetworkError, ExternalAPIError)
    )
    
    assert result["data"] == "success"
    assert result["call_count"] == 5  # Should have retried 4 times