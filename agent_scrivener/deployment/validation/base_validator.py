"""Base validator class with common functionality for all validators."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from .models import ValidationResult, ValidationStatus, RemediationStep


logger = logging.getLogger(__name__)


class BaseValidator(ABC):
    """Base class for all validators.
    
    Provides common functionality like timing, error handling, and result creation.
    All specific validators should inherit from this class.
    """
    
    def __init__(self, name: str, timeout_seconds: Optional[float] = None):
        """Initialize the base validator.
        
        Args:
            name: Name of the validator
            timeout_seconds: Optional timeout for validation execution
        """
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def validate(self) -> List[ValidationResult]:
        """Execute validation checks.
        
        This method must be implemented by all subclasses.
        
        Returns:
            List of validation results
        """
        pass
    
    async def run_with_timeout(self) -> List[ValidationResult]:
        """Run validation with timeout handling.
        
        Returns:
            List of validation results, or a timeout result if exceeded
        """
        start_time = time.time()
        
        try:
            if self.timeout_seconds:
                results = await asyncio.wait_for(
                    self.validate(),
                    timeout=self.timeout_seconds
                )
            else:
                results = await self.validate()
            
            return results
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            self.logger.error(f"Validation timed out after {duration:.2f} seconds")
            
            return [
                self.create_result(
                    status=ValidationStatus.TIMEOUT,
                    message=f"Validation timed out after {self.timeout_seconds} seconds",
                    duration_seconds=duration,
                    remediation_steps=[
                        "Check if the system is responding",
                        "Increase timeout if validation is expected to take longer",
                        "Check logs for any blocking operations"
                    ]
                )
            ]
        
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"Validation failed with exception: {e}")
            
            return [
                self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Validation failed with exception: {str(e)}",
                    duration_seconds=duration,
                    details={"exception": str(e), "exception_type": type(e).__name__},
                    remediation_steps=[
                        "Check the error message for specific issues",
                        "Review validator logs for detailed stack trace",
                        "Ensure all dependencies are properly configured"
                    ]
                )
            ]
    
    def create_result(
        self,
        status: ValidationStatus,
        message: str,
        duration_seconds: float = 0.0,
        details: Optional[dict] = None,
        remediation_steps: Optional[List[str]] = None
    ) -> ValidationResult:
        """Create a validation result.
        
        Args:
            status: Validation status
            message: Human-readable message
            duration_seconds: Time taken for validation
            details: Additional structured details
            remediation_steps: Steps to fix failures
            
        Returns:
            ValidationResult instance
        """
        return ValidationResult(
            validator_name=self.name,
            status=status,
            message=message,
            details=details or {},
            duration_seconds=duration_seconds,
            remediation_steps=remediation_steps
        )
    
    def create_remediation_step(
        self,
        issue: str,
        steps: List[str],
        priority: str = "MEDIUM",
        documentation_link: Optional[str] = None
    ) -> RemediationStep:
        """Create a remediation step.
        
        Args:
            issue: Description of the issue
            steps: List of steps to fix the issue
            priority: Priority level (CRITICAL, HIGH, MEDIUM, LOW)
            documentation_link: Optional link to documentation
            
        Returns:
            RemediationStep instance
        """
        return RemediationStep(
            validator_name=self.name,
            issue=issue,
            steps=steps,
            priority=priority,
            documentation_link=documentation_link
        )
    
    async def measure_operation(self, operation, *args, **kwargs):
        """Measure the execution time of an operation.
        
        Args:
            operation: Async function to measure
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Tuple of (result, duration_seconds)
        """
        start_time = time.time()
        result = await operation(*args, **kwargs)
        duration = time.time() - start_time
        return result, duration
    
    def log_validation_start(self):
        """Log the start of validation."""
        self.logger.info(f"Starting validation: {self.name}")
    
    def log_validation_complete(self, results: List[ValidationResult]):
        """Log the completion of validation.
        
        Args:
            results: List of validation results
        """
        passed = sum(1 for r in results if r.is_success())
        failed = sum(1 for r in results if r.is_failure())
        warnings = sum(1 for r in results if r.is_warning())
        
        self.logger.info(
            f"Validation complete: {self.name} - "
            f"Passed: {passed}, Failed: {failed}, Warnings: {warnings}"
        )
