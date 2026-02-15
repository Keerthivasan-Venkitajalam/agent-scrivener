"""Core data models for the production readiness validation framework."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ValidationStatus(Enum):
    """Status of a validation result."""
    
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"
    TIMEOUT = "timeout"


@dataclass
class ValidationResult:
    """Result of a single validation check.
    
    Attributes:
        validator_name: Name of the validator that produced this result
        status: Status of the validation (PASS, FAIL, WARNING, SKIP, TIMEOUT)
        message: Human-readable message describing the result
        details: Additional details about the validation (structured data)
        duration_seconds: Time taken to execute the validation
        timestamp: When the validation was executed
        remediation_steps: Optional list of steps to fix failures
    """
    
    validator_name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    remediation_steps: Optional[List[str]] = None
    
    def is_success(self) -> bool:
        """Check if validation was successful."""
        return self.status in (ValidationStatus.PASS, ValidationStatus.SKIP)
    
    def is_failure(self) -> bool:
        """Check if validation failed."""
        return self.status == ValidationStatus.FAIL
    
    def is_warning(self) -> bool:
        """Check if validation produced a warning."""
        return self.status == ValidationStatus.WARNING


@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific operation.
    
    Attributes:
        metric_name: Name of the metric being measured
        p50_ms: 50th percentile (median) latency in milliseconds
        p90_ms: 90th percentile latency in milliseconds
        p95_ms: 95th percentile latency in milliseconds
        p99_ms: 99th percentile latency in milliseconds
        min_ms: Minimum latency in milliseconds
        max_ms: Maximum latency in milliseconds
        mean_ms: Mean latency in milliseconds
        std_dev_ms: Standard deviation of latency in milliseconds
        sample_count: Number of samples measured
    """
    
    metric_name: str
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    std_dev_ms: float
    sample_count: int
    
    def meets_threshold(self, threshold_ms: float, percentile: str = "p90") -> bool:
        """Check if a specific percentile meets a threshold.
        
        Args:
            threshold_ms: Maximum acceptable latency in milliseconds
            percentile: Which percentile to check (p50, p90, p95, p99)
            
        Returns:
            True if the percentile is below the threshold
        """
        percentile_value = getattr(self, f"{percentile}_ms", None)
        if percentile_value is None:
            raise ValueError(f"Invalid percentile: {percentile}")
        return percentile_value <= threshold_ms


@dataclass
class RemediationStep:
    """A step to remediate a validation failure.
    
    Attributes:
        validator_name: Name of the validator that failed
        issue: Description of the issue
        steps: List of steps to fix the issue
        priority: Priority level (CRITICAL, HIGH, MEDIUM, LOW)
        documentation_link: Optional link to relevant documentation
    """
    
    validator_name: str
    issue: str
    steps: List[str]
    priority: str = "MEDIUM"
    documentation_link: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report.
    
    Attributes:
        overall_status: Overall validation status
        readiness_score: Score from 0-100 indicating production readiness
        total_validations: Total number of validations run
        passed_validations: Number of validations that passed
        failed_validations: Number of validations that failed
        warning_validations: Number of validations with warnings
        skipped_validations: Number of validations that were skipped
        validation_results: List of all validation results
        performance_metrics: List of performance metrics collected
        remediation_guide: List of remediation steps for failures
        generated_at: When the report was generated
    """
    
    overall_status: ValidationStatus
    readiness_score: float
    total_validations: int
    passed_validations: int
    failed_validations: int
    warning_validations: int
    skipped_validations: int
    validation_results: List[ValidationResult]
    performance_metrics: List[PerformanceMetrics] = field(default_factory=list)
    remediation_guide: List[RemediationStep] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    
    def is_production_ready(self) -> bool:
        """Check if the system is ready for production deployment.
        
        Returns:
            True if there are no critical failures
        """
        return self.failed_validations == 0 and self.overall_status != ValidationStatus.FAIL
    
    def get_critical_failures(self) -> List[ValidationResult]:
        """Get all critical validation failures.
        
        Returns:
            List of validation results with FAIL status
        """
        return [r for r in self.validation_results if r.is_failure()]
    
    def get_warnings(self) -> List[ValidationResult]:
        """Get all validation warnings.
        
        Returns:
            List of validation results with WARNING status
        """
        return [r for r in self.validation_results if r.is_warning()]
