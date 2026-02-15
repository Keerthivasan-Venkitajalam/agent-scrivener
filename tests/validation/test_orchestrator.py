"""Integration tests for the validation orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from agent_scrivener.deployment.validation.orchestrator import ValidationOrchestrator
from agent_scrivener.deployment.validation.models import (
    ValidationResult,
    ValidationStatus,
    ValidationReport
)


@pytest.fixture
def orchestrator():
    """Create a validation orchestrator for testing."""
    return ValidationOrchestrator(
        api_base_url="http://localhost:8000",
        database_url="postgresql://test:test@localhost/test",
        aws_region="us-east-1",
        timeout_seconds=60.0
    )


@pytest.fixture
def mock_validation_results():
    """Create mock validation results."""
    return [
        ValidationResult(
            validator_name="test-validator-1",
            status=ValidationStatus.PASS,
            message="Validation passed",
            duration_seconds=1.0,
            timestamp=datetime.now()
        ),
        ValidationResult(
            validator_name="test-validator-2",
            status=ValidationStatus.PASS,
            message="Validation passed",
            duration_seconds=2.0,
            timestamp=datetime.now()
        )
    ]


@pytest.fixture
def mock_validation_results_with_failures():
    """Create mock validation results with failures."""
    return [
        ValidationResult(
            validator_name="test-validator-1",
            status=ValidationStatus.PASS,
            message="Validation passed",
            duration_seconds=1.0,
            timestamp=datetime.now()
        ),
        ValidationResult(
            validator_name="test-validator-2",
            status=ValidationStatus.FAIL,
            message="Validation failed",
            duration_seconds=2.0,
            timestamp=datetime.now(),
            remediation_steps=["Fix the issue", "Try again"]
        ),
        ValidationResult(
            validator_name="test-validator-3",
            status=ValidationStatus.WARNING,
            message="Validation warning",
            duration_seconds=1.5,
            timestamp=datetime.now()
        )
    ]


class TestValidationOrchestrator:
    """Test suite for ValidationOrchestrator."""
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initializes with all validators."""
        assert orchestrator.api_base_url == "http://localhost:8000"
        assert orchestrator.database_url == "postgresql://test:test@localhost/test"
        assert orchestrator.aws_region == "us-east-1"
        assert orchestrator.timeout_seconds == 60.0
        
        # Check all validators are initialized
        expected_validators = [
            "end-to-end",
            "api-endpoints",
            "orchestration",
            "deployment-config",
            "performance",
            "documentation",
            "monitoring",
            "aws-infrastructure",
            "data-persistence",
            "security"
        ]
        
        for validator_name in expected_validators:
            assert validator_name in orchestrator.validators
    
    def test_get_available_validators(self, orchestrator):
        """Test getting list of available validators."""
        validators = orchestrator.get_available_validators()
        
        assert isinstance(validators, list)
        assert len(validators) == 10
        assert "api-endpoints" in validators
        assert "security" in validators
        assert "performance" in validators
    
    def test_get_validator_info(self, orchestrator):
        """Test getting validator information."""
        info = orchestrator.get_validator_info()
        
        assert isinstance(info, dict)
        assert len(info) == 10
        assert "api-endpoints" in info
        assert "Validates all REST and WebSocket API endpoints" in info["api-endpoints"]
        assert "security" in info
        assert "Validates security configurations" in info["security"]
    
    @pytest.mark.asyncio
    async def test_run_all_validations_success(self, orchestrator, mock_validation_results):
        """Test running all validators successfully."""
        # Mock all validators to return success results
        for validator in orchestrator.validators.values():
            validator.run_with_timeout = AsyncMock(return_value=mock_validation_results)
        
        report = await orchestrator.run_all_validations()
        
        assert isinstance(report, ValidationReport)
        assert report.total_validations > 0
        assert report.passed_validations > 0
        assert report.failed_validations == 0
        assert report.overall_status == ValidationStatus.PASS
        assert report.is_production_ready()
    
    @pytest.mark.asyncio
    async def test_run_all_validations_with_failures(
        self, 
        orchestrator, 
        mock_validation_results_with_failures
    ):
        """Test running all validators with some failures."""
        # Mock validators to return mixed results
        for validator in orchestrator.validators.values():
            validator.run_with_timeout = AsyncMock(
                return_value=mock_validation_results_with_failures
            )
        
        report = await orchestrator.run_all_validations()
        
        assert isinstance(report, ValidationReport)
        assert report.total_validations > 0
        assert report.failed_validations > 0
        assert report.warning_validations > 0
        assert not report.is_production_ready()
        
        # Check critical failures are identified
        critical_failures = report.get_critical_failures()
        assert len(critical_failures) > 0
        assert all(f.status == ValidationStatus.FAIL for f in critical_failures)
    
    @pytest.mark.asyncio
    async def test_run_all_validations_with_skip(self, orchestrator, mock_validation_results):
        """Test running validations with skip flags."""
        # Mock validators
        for validator in orchestrator.validators.values():
            validator.run_with_timeout = AsyncMock(return_value=mock_validation_results)
        
        skip_validators = {"aws-infrastructure", "performance"}
        report = await orchestrator.run_all_validations(skip_validators=skip_validators)
        
        assert isinstance(report, ValidationReport)
        assert report.skipped_validations >= 2
        
        # Check that skipped validators have SKIP status
        skipped_results = [
            r for r in report.validation_results 
            if r.status == ValidationStatus.SKIP
        ]
        assert len(skipped_results) >= 2
        
        skipped_names = {r.validator_name for r in skipped_results}
        assert "aws-infrastructure" in skipped_names
        assert "performance" in skipped_names
    
    @pytest.mark.asyncio
    async def test_run_specific_validators(self, orchestrator, mock_validation_results):
        """Test running specific validators."""
        # Mock validators
        for validator in orchestrator.validators.values():
            validator.run_with_timeout = AsyncMock(return_value=mock_validation_results)
        
        validator_names = ["api-endpoints", "security"]
        report = await orchestrator.run_specific_validators(validator_names)
        
        assert isinstance(report, ValidationReport)
        # Should only have results from the 2 specified validators
        # Each validator returns 2 results in our mock
        assert report.total_validations == 4
    
    @pytest.mark.asyncio
    async def test_run_specific_validators_invalid_name(self, orchestrator):
        """Test running specific validators with invalid name."""
        with pytest.raises(ValueError) as exc_info:
            await orchestrator.run_specific_validators(["invalid-validator"])
        
        assert "Unknown validators" in str(exc_info.value)
        assert "invalid-validator" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_run_quick_validation(self, orchestrator, mock_validation_results):
        """Test running quick validation."""
        # Mock validators
        for validator in orchestrator.validators.values():
            validator.run_with_timeout = AsyncMock(return_value=mock_validation_results)
        
        report = await orchestrator.run_quick_validation()
        
        assert isinstance(report, ValidationReport)
        # Quick validation runs 4 validators: deployment-config, security, api-endpoints, documentation
        # Each returns 2 results in our mock
        assert report.total_validations == 8
    
    @pytest.mark.asyncio
    async def test_validator_exception_handling(self, orchestrator):
        """Test that orchestrator handles validator exceptions gracefully."""
        # Mock one validator to raise an exception
        orchestrator.validators["api-endpoints"].run_with_timeout = AsyncMock(
            side_effect=Exception("Test exception")
        )
        
        # Mock other validators to return success
        mock_result = [
            ValidationResult(
                validator_name="test",
                status=ValidationStatus.PASS,
                message="Success",
                duration_seconds=1.0
            )
        ]
        
        for name, validator in orchestrator.validators.items():
            if name != "api-endpoints":
                validator.run_with_timeout = AsyncMock(return_value=mock_result)
        
        report = await orchestrator.run_all_validations()
        
        # Should still complete and include failure result for the exception
        assert isinstance(report, ValidationReport)
        assert report.failed_validations >= 1
        
        # Find the failure result
        failure_results = [
            r for r in report.validation_results 
            if r.validator_name == "api-endpoints" and r.status == ValidationStatus.FAIL
        ]
        assert len(failure_results) == 1
        assert "Test exception" in failure_results[0].message
    
    @pytest.mark.asyncio
    async def test_result_aggregation(self, orchestrator):
        """Test that results from all validators are properly aggregated."""
        # Create unique results for each validator
        validator_results = {}
        for i, validator_name in enumerate(orchestrator.validators.keys()):
            validator_results[validator_name] = [
                ValidationResult(
                    validator_name=validator_name,
                    status=ValidationStatus.PASS,
                    message=f"Result from {validator_name}",
                    duration_seconds=float(i),
                    timestamp=datetime.now()
                )
            ]
        
        # Mock each validator to return its unique result
        for validator_name, validator in orchestrator.validators.items():
            validator.run_with_timeout = AsyncMock(
                return_value=validator_results[validator_name]
            )
        
        report = await orchestrator.run_all_validations()
        
        # Check all results are included
        assert report.total_validations == len(orchestrator.validators)
        
        # Check each validator's result is present
        result_names = {r.validator_name for r in report.validation_results}
        expected_names = set(orchestrator.validators.keys())
        assert result_names == expected_names
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, orchestrator):
        """Test that orchestrator handles validator timeouts."""
        # Mock one validator to timeout
        timeout_result = ValidationResult(
            validator_name="api-endpoints",
            status=ValidationStatus.TIMEOUT,
            message="Validation timed out after 60.0 seconds",
            duration_seconds=60.0
        )
        
        orchestrator.validators["api-endpoints"].run_with_timeout = AsyncMock(
            return_value=[timeout_result]
        )
        
        # Mock other validators to return success
        mock_result = [
            ValidationResult(
                validator_name="test",
                status=ValidationStatus.PASS,
                message="Success",
                duration_seconds=1.0
            )
        ]
        
        for name, validator in orchestrator.validators.items():
            if name != "api-endpoints":
                validator.run_with_timeout = AsyncMock(return_value=mock_result)
        
        report = await orchestrator.run_all_validations()
        
        # Check timeout is recorded
        timeout_results = [
            r for r in report.validation_results 
            if r.status == ValidationStatus.TIMEOUT
        ]
        assert len(timeout_results) >= 1
        assert timeout_results[0].validator_name == "api-endpoints"


class TestValidationOrchestratorIntegration:
    """Integration tests that test orchestrator with real validator instances."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_with_real_validators(self):
        """Test orchestrator with real validator instances (mocked dependencies)."""
        orchestrator = ValidationOrchestrator(
            api_base_url="http://localhost:8000",
            database_url="postgresql://test:test@localhost/test",
            aws_region="us-east-1",
            timeout_seconds=300.0  # 5 minutes
        )
        
        # This test verifies that the orchestrator can be instantiated
        # with real validators and that all validators are properly initialized
        assert len(orchestrator.validators) == 10
        
        # Verify validators are initialized (don't check timeout as it varies by validator)
        for validator_name, validator in orchestrator.validators.items():
            assert validator is not None
            assert hasattr(validator, 'validate')
    
    @pytest.mark.asyncio
    async def test_deployment_blocking_on_critical_failures(
        self, 
        orchestrator, 
        mock_validation_results_with_failures
    ):
        """Test that critical failures block deployment."""
        # Mock validators to return failures
        for validator in orchestrator.validators.values():
            validator.run_with_timeout = AsyncMock(
                return_value=mock_validation_results_with_failures
            )
        
        report = await orchestrator.run_all_validations()
        
        # System should not be production ready with failures
        assert not report.is_production_ready()
        assert report.failed_validations > 0
        
        # Critical failures should be identified
        critical_failures = report.get_critical_failures()
        assert len(critical_failures) > 0
        
        # Remediation steps should be present
        assert any(
            f.remediation_steps is not None 
            for f in critical_failures
        )
