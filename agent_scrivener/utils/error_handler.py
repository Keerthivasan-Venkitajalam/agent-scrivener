"""
Centralized error handling system with retry logic and circuit breaker pattern.
"""

import asyncio
import logging
import time
import uuid
from typing import Callable, Any, Dict, Optional, Type, Tuple, List
from functools import wraps
from datetime import datetime, timedelta
from enum import Enum

from ..models.errors import (
    ErrorDetails, ErrorResponse, ErrorCategory, ErrorSeverity,
    AgentScrivenerError, NetworkError, ExternalAPIError
)


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter


class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception


class CircuitBreaker:
    """Circuit breaker implementation for external services."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise ExternalAPIError(
                        f"Circuit breaker is OPEN. Service unavailable.",
                        circuit_breaker_state=self.state.value,
                        failure_count=self.failure_count
                    )
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._on_success()
            return result
        except self.config.expected_exception as e:
            await self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).seconds >= self.config.recovery_timeout
    
    async def _on_success(self):
        """Handle successful operation."""
        async with self._lock:
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED
    
    async def _on_failure(self):
        """Handle failed operation."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN


class ErrorHandler:
    """Centralized error handling system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_stats: Dict[str, Dict[str, int]] = {}
    
    def get_circuit_breaker(self, service_name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if service_name not in self.circuit_breakers:
            config = config or CircuitBreakerConfig()
            self.circuit_breakers[service_name] = CircuitBreaker(config)
        return self.circuit_breakers[service_name]
    
    async def with_retry(
        self,
        operation: Callable,
        retry_config: Optional[RetryConfig] = None,
        error_types: Tuple[Type[Exception], ...] = (Exception,),
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute operation with retry logic."""
        config = retry_config or RetryConfig()
        context = context or {}
        
        last_exception = None
        
        for attempt in range(config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation()
                else:
                    return operation()
            except error_types as e:
                last_exception = e
                
                if attempt == config.max_retries:
                    # Final attempt failed
                    error_details = self._create_error_details(
                        e, context, retry_count=attempt
                    )
                    self._log_error(error_details)
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(
                    config.base_delay * (config.backoff_factor ** attempt),
                    config.max_delay
                )
                
                if config.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)  # Add jitter
                
                self.logger.warning(
                    f"Operation failed (attempt {attempt + 1}/{config.max_retries + 1}). "
                    f"Retrying in {delay:.2f}s. Error: {str(e)}"
                )
                
                await asyncio.sleep(delay)
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
    
    async def with_circuit_breaker(
        self,
        operation: Callable,
        service_name: str,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute operation with circuit breaker protection."""
        circuit_breaker = self.get_circuit_breaker(service_name, circuit_config)
        context = context or {}
        
        try:
            return await circuit_breaker.call(operation)
        except Exception as e:
            error_details = self._create_error_details(e, context, service_name=service_name)
            self._log_error(error_details)
            raise
    
    async def with_retry_and_circuit_breaker(
        self,
        operation: Callable,
        service_name: str,
        retry_config: Optional[RetryConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        error_types: Tuple[Type[Exception], ...] = (Exception,),
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute operation with both retry logic and circuit breaker protection."""
        circuit_breaker = self.get_circuit_breaker(service_name, circuit_config)
        
        async def protected_operation():
            return await circuit_breaker.call(operation)
        
        return await self.with_retry(
            protected_operation,
            retry_config,
            error_types,
            context
        )
    
    def handle_agent_error(
        self,
        agent_name: str,
        error: Exception,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorResponse:
        """Handle errors from agents with standardized response."""
        context = context or {}
        context.update({
            "agent_name": agent_name,
            "session_id": session_id
        })
        
        error_details = self._create_error_details(error, context, agent_name=agent_name)
        self._log_error(error_details)
        self._update_error_stats(agent_name, error_details.category.value)
        
        # Determine recovery strategies
        suggested_actions = self._get_recovery_strategies(error_details)
        
        return ErrorResponse(
            error=error_details,
            suggested_actions=suggested_actions
        )
    
    def _create_error_details(
        self,
        error: Exception,
        context: Dict[str, Any],
        agent_name: Optional[str] = None,
        service_name: Optional[str] = None,
        retry_count: int = 0
    ) -> ErrorDetails:
        """Create standardized error details."""
        error_id = str(uuid.uuid4())
        
        # Determine category and severity
        if isinstance(error, AgentScrivenerError):
            category = error.category
            severity = error.severity
            message = error.message
            details = str(error)
        else:
            category = self._classify_error(error)
            severity = self._determine_severity(error, category)
            message = str(error)
            details = f"{type(error).__name__}: {str(error)}"
        
        # Add service context
        if service_name:
            context["service_name"] = service_name
        
        return ErrorDetails(
            error_id=error_id,
            category=category,
            severity=severity,
            message=message,
            details=details,
            context=context,
            agent_name=agent_name,
            session_id=context.get("session_id"),
            retry_count=retry_count,
            recoverable=self._is_recoverable(error, category)
        )
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error into appropriate category."""
        error_type = type(error).__name__.lower()
        
        if any(keyword in error_type for keyword in ["network", "connection", "timeout", "http"]):
            return ErrorCategory.NETWORK
        elif any(keyword in error_type for keyword in ["validation", "value", "type"]):
            return ErrorCategory.VALIDATION
        elif any(keyword in error_type for keyword in ["api", "request", "response"]):
            return ErrorCategory.EXTERNAL_API
        else:
            return ErrorCategory.SYSTEM
    
    def _determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on type and category."""
        if category == ErrorCategory.VALIDATION:
            return ErrorSeverity.HIGH
        elif category in [ErrorCategory.NETWORK, ErrorCategory.EXTERNAL_API]:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _is_recoverable(self, error: Exception, category: ErrorCategory) -> bool:
        """Determine if error is recoverable."""
        if category == ErrorCategory.VALIDATION:
            return False
        elif category in [ErrorCategory.NETWORK, ErrorCategory.EXTERNAL_API]:
            return True
        else:
            return True
    
    def _get_recovery_strategies(self, error_details: ErrorDetails) -> List[str]:
        """Get suggested recovery strategies for error."""
        strategies = []
        
        if error_details.category == ErrorCategory.NETWORK:
            strategies.extend([
                "Check network connectivity",
                "Verify service endpoints",
                "Retry with exponential backoff"
            ])
        elif error_details.category == ErrorCategory.EXTERNAL_API:
            strategies.extend([
                "Check API credentials and permissions",
                "Verify API rate limits",
                "Use alternative data sources if available"
            ])
        elif error_details.category == ErrorCategory.VALIDATION:
            strategies.extend([
                "Validate input data format",
                "Check required fields",
                "Review data constraints"
            ])
        elif error_details.category == ErrorCategory.AGENT_COMMUNICATION:
            strategies.extend([
                "Restart affected agents",
                "Check agent registry status",
                "Verify message queue connectivity"
            ])
        
        return strategies
    
    def _log_error(self, error_details: ErrorDetails):
        """Log error with appropriate level."""
        log_message = (
            f"Error {error_details.error_id}: {error_details.message} "
            f"[{error_details.category.value}/{error_details.severity.value}]"
        )
        
        if error_details.agent_name:
            log_message += f" Agent: {error_details.agent_name}"
        
        if error_details.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, extra={"error_details": error_details.model_dump()})
        elif error_details.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, extra={"error_details": error_details.model_dump()})
        elif error_details.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message, extra={"error_details": error_details.model_dump()})
        else:
            self.logger.info(log_message, extra={"error_details": error_details.model_dump()})
    
    def _update_error_stats(self, agent_name: str, category: str):
        """Update error statistics."""
        if agent_name not in self.error_stats:
            self.error_stats[agent_name] = {}
        
        if category not in self.error_stats[agent_name]:
            self.error_stats[agent_name][category] = 0
        
        self.error_stats[agent_name][category] += 1
    
    def get_error_stats(self) -> Dict[str, Dict[str, int]]:
        """Get current error statistics."""
        return self.error_stats.copy()
    
    def reset_error_stats(self):
        """Reset error statistics."""
        self.error_stats.clear()


# Global error handler instance
error_handler = ErrorHandler()


def with_error_handling(
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_service: Optional[str] = None,
    circuit_config: Optional[CircuitBreakerConfig] = None,
    error_types: Tuple[Type[Exception], ...] = (Exception,)
):
    """Decorator for adding error handling to functions."""
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                context = {
                    "function_name": func.__name__,
                    "module": func.__module__
                }
                
                async def operation():
                    return await func(*args, **kwargs)
                
                if circuit_breaker_service:
                    return await error_handler.with_retry_and_circuit_breaker(
                        operation,
                        circuit_breaker_service,
                        retry_config,
                        circuit_config,
                        error_types,
                        context
                    )
                else:
                    return await error_handler.with_retry(
                        operation,
                        retry_config,
                        error_types,
                        context
                    )
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For synchronous functions, we can't use async error handling
                # So we just handle the error and re-raise
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = {
                        "function_name": func.__name__,
                        "module": func.__module__
                    }
                    error_response = error_handler.handle_agent_error(
                        func.__module__, e, context=context
                    )
                    raise
            
            return sync_wrapper
    
    return decorator