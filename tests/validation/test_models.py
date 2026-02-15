"""Unit tests for validation framework data models."""

import pytest
from datetime import datetime

from agent_scrivener.deployment.validation.models import (
    ValidationStatus,
    ValidationResult,
    PerformanceMetrics,
    ValidationReport,
    RemediationStep,
)


class TestValidationResult:
    """Tests for ValidationResult model."""
    
    def test_create_validation_result(self):
        """Test creating a validation result."""
        result = ValidationResult(
            validator_name="test_validator",
            status=ValidationStatus.PASS,
            message="Test passed",
            details={"key": "value"},
            duration_seconds=1.5
        )
        
        assert result.validator_name == "test_validator"
        assert result.status == ValidationStatus.PASS
        assert result.message == "Test passed"
        assert result.details == {"key": "value"}
        assert result.duration_seconds == 1.5
        assert isinstance(result.timestamp, datetime)
    
    def test_is_success(self):
        """Test is_success method."""
        pass_result = ValidationResult(
            validator_name="test",
            status=ValidationStatus.PASS,
            message="Passed"
        )
        skip_result = ValidationResult(
            validator_name="test",
            status=ValidationStatus.SKIP,
            message="Skipped"
        )
        fail_result = ValidationResult(
            validator_name="test",
            status=ValidationStatus.FAIL,
            message="Failed"
        )
        
        assert pass_result.is_success() is True
        assert skip_result.is_success() is True
        assert fail_result.is_success() is False
    
    def test_is_failure(self):
        """Test is_failure method."""
        fail_result = ValidationResult(
            validator_name="test",
            status=ValidationStatus.FAIL,
            message="Failed"
        )
        pass_result = ValidationResult(
            validator_name="test",
            status=ValidationStatus.PASS,
            message="Passed"
        )
        
        assert fail_result.is_failure() is True
        assert pass_result.is_failure() is False
    
    def test_is_warning(self):
        """Test is_warning method."""
        warning_result = ValidationResult(
            validator_name="test",
            status=ValidationStatus.WARNING,
            message="Warning"
        )
        pass_result = ValidationResult(
            validator_name="test",
            status=ValidationStatus.PASS,
            message="Passed"
        )
        
        assert warning_result.is_warning() is True
        assert pass_result.is_warning() is False


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics model."""
    
    def test_create_performance_metrics(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            metric_name="api_response_time",
            p50_ms=100.0,
            p90_ms=200.0,
            p95_ms=250.0,
            p99_ms=300.0,
            min_ms=50.0,
            max_ms=400.0,
            mean_ms=150.0,
            std_dev_ms=75.0,
            sample_count=100
        )
        
        assert metrics.metric_name == "api_response_time"
        assert metrics.p50_ms == 100.0
        assert metrics.p90_ms == 200.0
        assert metrics.sample_count == 100
    
    def test_meets_threshold_p90(self):
        """Test meets_threshold for p90."""
        metrics = PerformanceMetrics(
            metric_name="test",
            p50_ms=100.0,
            p90_ms=200.0,
            p95_ms=250.0,
            p99_ms=300.0,
            min_ms=50.0,
            max_ms=400.0,
            mean_ms=150.0,
            std_dev_ms=75.0,
            sample_count=100
        )
        
        assert metrics.meets_threshold(250.0, "p90") is True
        assert metrics.meets_threshold(150.0, "p90") is False
    
    def test_meets_threshold_p99(self):
        """Test meets_threshold for p99."""
        metrics = PerformanceMetrics(
            metric_name="test",
            p50_ms=100.0,
            p90_ms=200.0,
            p95_ms=250.0,
            p99_ms=300.0,
            min_ms=50.0,
            max_ms=400.0,
            mean_ms=150.0,
            std_dev_ms=75.0,
            sample_count=100
        )
        
        assert metrics.meets_threshold(350.0, "p99") is True
        assert metrics.meets_threshold(250.0, "p99") is False
    
    def test_meets_threshold_invalid_percentile(self):
        """Test meets_threshold with invalid percentile."""
        metrics = PerformanceMetrics(
            metric_name="test",
            p50_ms=100.0,
            p90_ms=200.0,
            p95_ms=250.0,
            p99_ms=300.0,
            min_ms=50.0,
            max_ms=400.0,
            mean_ms=150.0,
            std_dev_ms=75.0,
            sample_count=100
        )
        
        with pytest.raises(ValueError, match="Invalid percentile"):
            metrics.meets_threshold(100.0, "p75")


class TestRemediationStep:
    """Tests for RemediationStep model."""
    
    def test_create_remediation_step(self):
        """Test creating a remediation step."""
        step = RemediationStep(
            validator_name="test_validator",
            issue="Configuration missing",
            steps=["Step 1", "Step 2"],
            priority="HIGH",
            documentation_link="https://docs.example.com"
        )
        
        assert step.validator_name == "test_validator"
        assert step.issue == "Configuration missing"
        assert step.steps == ["Step 1", "Step 2"]
        assert step.priority == "HIGH"
        assert step.documentation_link == "https://docs.example.com"
    
    def test_default_priority(self):
        """Test default priority is MEDIUM."""
        step = RemediationStep(
            validator_name="test",
            issue="Issue",
            steps=["Fix it"]
        )
        
        assert step.priority == "MEDIUM"


class TestValidationReport:
    """Tests for ValidationReport model."""
    
    def test_create_validation_report(self):
        """Test creating a validation report."""
        results = [
            ValidationResult(
                validator_name="test1",
                status=ValidationStatus.PASS,
                message="Passed"
            ),
            ValidationResult(
                validator_name="test2",
                status=ValidationStatus.FAIL,
                message="Failed"
            )
        ]
        
        report = ValidationReport(
            overall_status=ValidationStatus.FAIL,
            readiness_score=50.0,
            total_validations=2,
            passed_validations=1,
            failed_validations=1,
            warning_validations=0,
            skipped_validations=0,
            validation_results=results
        )
        
        assert report.overall_status == ValidationStatus.FAIL
        assert report.readiness_score == 50.0
        assert report.total_validations == 2
        assert report.passed_validations == 1
        assert report.failed_validations == 1
        assert len(report.validation_results) == 2
    
    def test_is_production_ready_success(self):
        """Test is_production_ready returns True when no failures."""
        report = ValidationReport(
            overall_status=ValidationStatus.PASS,
            readiness_score=100.0,
            total_validations=2,
            passed_validations=2,
            failed_validations=0,
            warning_validations=0,
            skipped_validations=0,
            validation_results=[]
        )
        
        assert report.is_production_ready() is True
    
    def test_is_production_ready_failure(self):
        """Test is_production_ready returns False when there are failures."""
        report = ValidationReport(
            overall_status=ValidationStatus.FAIL,
            readiness_score=50.0,
            total_validations=2,
            passed_validations=1,
            failed_validations=1,
            warning_validations=0,
            skipped_validations=0,
            validation_results=[]
        )
        
        assert report.is_production_ready() is False
    
    def test_get_critical_failures(self):
        """Test get_critical_failures returns only failed results."""
        results = [
            ValidationResult(
                validator_name="test1",
                status=ValidationStatus.PASS,
                message="Passed"
            ),
            ValidationResult(
                validator_name="test2",
                status=ValidationStatus.FAIL,
                message="Failed"
            ),
            ValidationResult(
                validator_name="test3",
                status=ValidationStatus.WARNING,
                message="Warning"
            )
        ]
        
        report = ValidationReport(
            overall_status=ValidationStatus.FAIL,
            readiness_score=50.0,
            total_validations=3,
            passed_validations=1,
            failed_validations=1,
            warning_validations=1,
            skipped_validations=0,
            validation_results=results
        )
        
        failures = report.get_critical_failures()
        assert len(failures) == 1
        assert failures[0].validator_name == "test2"
    
    def test_get_warnings(self):
        """Test get_warnings returns only warning results."""
        results = [
            ValidationResult(
                validator_name="test1",
                status=ValidationStatus.PASS,
                message="Passed"
            ),
            ValidationResult(
                validator_name="test2",
                status=ValidationStatus.FAIL,
                message="Failed"
            ),
            ValidationResult(
                validator_name="test3",
                status=ValidationStatus.WARNING,
                message="Warning"
            )
        ]
        
        report = ValidationReport(
            overall_status=ValidationStatus.FAIL,
            readiness_score=50.0,
            total_validations=3,
            passed_validations=1,
            failed_validations=1,
            warning_validations=1,
            skipped_validations=0,
            validation_results=results
        )
        
        warnings = report.get_warnings()
        assert len(warnings) == 1
        assert warnings[0].validator_name == "test3"
