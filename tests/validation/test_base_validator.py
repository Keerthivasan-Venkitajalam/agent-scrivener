"""Unit tests for BaseValidator class."""

import asyncio
import pytest

from agent_scrivener.deployment.validation.base_validator import BaseValidator
from agent_scrivener.deployment.validation.models import ValidationStatus, ValidationResult


class TestValidator(BaseValidator):
    """Test implementation of BaseValidator."""
    
    def __init__(self, name: str = "test_validator", timeout_seconds: float = None, should_fail: bool = False):
        super().__init__(name, timeout_seconds)
        self.should_fail = should_fail
        self.validate_called = False
    
    async def validate(self):
        """Test validation implementation."""
        self.validate_called = True
        
        if self.should_fail:
            raise ValueError("Test validation error")
        
        return [
            self.create_result(
                status=ValidationStatus.PASS,
                message="Test validation passed",
                duration_seconds=0.1
            )
        ]


class SlowValidator(BaseValidator):
    """Validator that takes a long time to complete."""
    
    def __init__(self, delay_seconds: float = 5.0):
        super().__init__("slow_validator", timeout_seconds=1.0)
        self.delay_seconds = delay_seconds
    
    async def validate(self):
        """Slow validation that will timeout."""
        await asyncio.sleep(self.delay_seconds)
        return [
            self.create_result(
                status=ValidationStatus.PASS,
                message="Slow validation completed"
            )
        ]


class TestBaseValidator:
    """Tests for BaseValidator class."""
    
    @pytest.mark.asyncio
    async def test_create_result(self):
        """Test creating a validation result."""
        validator = TestValidator()
        
        result = validator.create_result(
            status=ValidationStatus.PASS,
            message="Test message",
            duration_seconds=1.5,
            details={"key": "value"},
            remediation_steps=["Step 1"]
        )
        
        assert result.validator_name == "test_validator"
        assert result.status == ValidationStatus.PASS
        assert result.message == "Test message"
        assert result.duration_seconds == 1.5
        assert result.details == {"key": "value"}
        assert result.remediation_steps == ["Step 1"]
    
    @pytest.mark.asyncio
    async def test_validate_success(self):
        """Test successful validation."""
        validator = TestValidator()
        results = await validator.validate()
        
        assert len(results) == 1
        assert results[0].status == ValidationStatus.PASS
        assert results[0].message == "Test validation passed"
        assert validator.validate_called is True
    
    @pytest.mark.asyncio
    async def test_run_with_timeout_success(self):
        """Test run_with_timeout with successful validation."""
        validator = TestValidator()
        results = await validator.run_with_timeout()
        
        assert len(results) == 1
        assert results[0].status == ValidationStatus.PASS
    
    @pytest.mark.asyncio
    async def test_run_with_timeout_exception(self):
        """Test run_with_timeout handles exceptions."""
        validator = TestValidator(should_fail=True)
        results = await validator.run_with_timeout()
        
        assert len(results) == 1
        assert results[0].status == ValidationStatus.FAIL
        assert "Test validation error" in results[0].message
        assert results[0].details["exception_type"] == "ValueError"
    
    @pytest.mark.asyncio
    async def test_run_with_timeout_timeout(self):
        """Test run_with_timeout handles timeout."""
        validator = SlowValidator(delay_seconds=2.0)
        results = await validator.run_with_timeout()
        
        assert len(results) == 1
        assert results[0].status == ValidationStatus.TIMEOUT
        assert "timed out" in results[0].message.lower()
    
    @pytest.mark.asyncio
    async def test_measure_operation(self):
        """Test measuring operation duration."""
        validator = TestValidator()
        
        async def test_operation():
            await asyncio.sleep(0.1)
            return "result"
        
        result, duration = await validator.measure_operation(test_operation)
        
        assert result == "result"
        assert duration >= 0.1
        assert duration < 0.2  # Should complete quickly
    
    def test_create_remediation_step(self):
        """Test creating a remediation step."""
        validator = TestValidator()
        
        step = validator.create_remediation_step(
            issue="Test issue",
            steps=["Step 1", "Step 2"],
            priority="HIGH",
            documentation_link="https://docs.example.com"
        )
        
        assert step.validator_name == "test_validator"
        assert step.issue == "Test issue"
        assert step.steps == ["Step 1", "Step 2"]
        assert step.priority == "HIGH"
        assert step.documentation_link == "https://docs.example.com"
    
    @pytest.mark.asyncio
    async def test_log_validation_start(self, caplog):
        """Test logging validation start."""
        import logging
        caplog.set_level(logging.INFO)
        
        validator = TestValidator()
        validator.log_validation_start()
        
        assert "Starting validation: test_validator" in caplog.text
    
    @pytest.mark.asyncio
    async def test_log_validation_complete(self, caplog):
        """Test logging validation complete."""
        import logging
        caplog.set_level(logging.INFO)
        
        validator = TestValidator()
        results = [
            ValidationResult(
                validator_name="test",
                status=ValidationStatus.PASS,
                message="Passed"
            ),
            ValidationResult(
                validator_name="test",
                status=ValidationStatus.FAIL,
                message="Failed"
            ),
            ValidationResult(
                validator_name="test",
                status=ValidationStatus.WARNING,
                message="Warning"
            )
        ]
        
        validator.log_validation_complete(results)
        
        assert "Validation complete: test_validator" in caplog.text
        assert "Passed: 1" in caplog.text
        assert "Failed: 1" in caplog.text
        assert "Warnings: 1" in caplog.text
