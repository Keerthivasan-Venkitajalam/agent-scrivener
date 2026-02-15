# Production Readiness Validation Framework

This directory contains the production readiness validation framework for Agent Scrivener. The framework provides comprehensive validation of all system components before production deployment.

## Structure

```
validation/
├── __init__.py              # Package exports
├── models.py                # Core data models
├── base_validator.py        # Base validator class
├── orchestrator.py          # Validation orchestrator
├── cli.py                   # Command-line interface
├── __main__.py              # Module entry point
├── report_generator.py      # Report generation
├── end_to_end_validator.py  # End-to-end workflow validation
├── api_endpoint_validator.py # API endpoint validation
├── orchestration_validator.py # Agent orchestration validation
├── deployment_config_validator.py # Deployment config validation
├── performance_benchmarker.py # Performance benchmarking
├── documentation_validator.py # Documentation validation
├── monitoring_validator.py  # Monitoring infrastructure validation
├── aws_infrastructure_validator.py # AWS infrastructure validation
├── data_persistence_validator.py # Data persistence validation
├── security_validator.py    # Security validation
└── README.md               # This file
```

## Core Components

### Data Models (`models.py`)

- **ValidationStatus**: Enum for validation statuses (PASS, FAIL, WARNING, SKIP, TIMEOUT)
- **ValidationResult**: Result of a single validation check
- **PerformanceMetrics**: Performance metrics with percentile breakdowns
- **RemediationStep**: Steps to fix validation failures
- **ValidationReport**: Comprehensive validation report

### Base Validator (`base_validator.py`)

Abstract base class that all validators inherit from. Provides:
- Timeout handling
- Error handling
- Result creation helpers
- Operation timing
- Logging utilities

### Report Generator (`report_generator.py`)

Generates comprehensive validation reports in multiple formats:
- Markdown (human-readable)
- JSON (machine-readable)
- HTML (web-viewable)

## Usage Example

### Using the CLI (Recommended)

The easiest way to run validations is through the command-line interface:

```bash
# Run all validations
python -m agent_scrivener.deployment.validation.cli

# Run quick validation (critical validators only)
python -m agent_scrivener.deployment.validation.cli --quick

# Run specific validators
python -m agent_scrivener.deployment.validation.cli --only api-endpoints security

# Skip AWS and performance validations
python -m agent_scrivener.deployment.validation.cli --skip-aws --skip-performance

# Run with verbose logging and save report
python -m agent_scrivener.deployment.validation.cli --verbose --output-dir ./reports

# List all available validators
python -m agent_scrivener.deployment.validation.cli --list-validators
```

### Using the Orchestrator Programmatically

```python
from agent_scrivener.deployment.validation import ValidationOrchestrator

# Initialize orchestrator
orchestrator = ValidationOrchestrator(
    api_base_url="http://localhost:8000",
    database_url="postgresql://user:pass@localhost/db",
    aws_region="us-east-1",
    timeout_seconds=300.0
)

# Run all validations
report = await orchestrator.run_all_validations()

# Run specific validators
report = await orchestrator.run_specific_validators(["api-endpoints", "security"])

# Run quick validation
report = await orchestrator.run_quick_validation()

# Skip certain validators
report = await orchestrator.run_all_validations(
    skip_validators={"aws-infrastructure", "performance"}
)

# Check if system is production ready
if report.is_production_ready():
    print("✓ System is ready for production deployment")
else:
    print("✗ System is NOT ready for production")
    for failure in report.get_critical_failures():
        print(f"  - {failure.validator_name}: {failure.message}")
```

### Using Individual Validators

```python
from agent_scrivener.deployment.validation import (
    BaseValidator,
    ValidationStatus,
    ValidationReportGenerator
)

# Create a custom validator
class MyValidator(BaseValidator):
    def __init__(self):
        super().__init__("my_validator", timeout_seconds=30)
    
    async def validate(self):
        # Perform validation checks
        results = []
        
        # Example check
        if some_condition:
            results.append(
                self.create_result(
                    status=ValidationStatus.PASS,
                    message="Check passed",
                    duration_seconds=1.5
                )
            )
        else:
            results.append(
                self.create_result(
                    status=ValidationStatus.FAIL,
                    message="Check failed",
                    remediation_steps=[
                        "Step 1: Fix the issue",
                        "Step 2: Verify the fix"
                    ]
                )
            )
        
        return results

# Run validation
validator = MyValidator()
results = await validator.run_with_timeout()

# Generate report
generator = ValidationReportGenerator()
report = generator.generate_summary_report(results)

# Export report
markdown_report = generator.export_report(report, format="markdown")
print(markdown_report)
```

## Creating New Validators

To create a new validator:

1. Inherit from `BaseValidator`
2. Implement the `validate()` method
3. Use `create_result()` to create validation results
4. Use `create_remediation_step()` for failure remediation
5. Set appropriate timeout in constructor

Example:

```python
class DatabaseValidator(BaseValidator):
    def __init__(self):
        super().__init__("database_validator", timeout_seconds=60)
    
    async def validate(self):
        self.log_validation_start()
        results = []
        
        # Test database connection
        try:
            connection, duration = await self.measure_operation(
                self.test_connection
            )
            
            results.append(
                self.create_result(
                    status=ValidationStatus.PASS,
                    message="Database connection successful",
                    duration_seconds=duration,
                    details={"host": connection.host}
                )
            )
        except Exception as e:
            results.append(
                self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Database connection failed: {e}",
                    remediation_steps=[
                        "Check database credentials",
                        "Verify database is running",
                        "Check network connectivity"
                    ]
                )
            )
        
        self.log_validation_complete(results)
        return results
    
    async def test_connection(self):
        # Implementation here
        pass
```

## Testing

Tests are located in `tests/validation/`:
- `test_models.py` - Tests for data models
- `test_base_validator.py` - Tests for base validator
- `test_report_generator.py` - Tests for report generator

Run tests:
```bash
pytest tests/validation/ -v
```

## Next Steps

All validators have been implemented! The validation framework is complete and ready for use.

### Available Validators

1. **End-to-End Validator** - Validates complete research workflow from query to document
2. **API Endpoint Validator** - Validates all REST and WebSocket API endpoints
3. **Orchestration Validator** - Validates agent coordination and workflow management
4. **Deployment Configuration Validator** - Validates deployment configuration files and settings
5. **Performance Benchmarker** - Measures system performance and establishes baselines
6. **Documentation Validator** - Validates completeness of documentation
7. **Monitoring Validator** - Validates production monitoring infrastructure
8. **AWS Infrastructure Validator** - Validates AWS infrastructure readiness
9. **Data Persistence Validator** - Validates data persistence and recovery mechanisms
10. **Security Validator** - Validates security configurations and compliance

### Running the Complete Validation Suite

To validate your system is ready for production:

```bash
# Run complete validation suite
python -m agent_scrivener.deployment.validation.cli \
    --api-url http://localhost:8000 \
    --database-url postgresql://user:pass@localhost/db \
    --aws-region us-east-1 \
    --output-dir ./validation-reports \
    --format markdown \
    --verbose

# For CI/CD pipelines (quick validation)
python -m agent_scrivener.deployment.validation.cli --quick

# Exit code 0 = production ready, 1 = not ready
```

### Integration with CI/CD

Add to your deployment pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run Production Readiness Validation
  run: |
    python -m agent_scrivener.deployment.validation.cli \
      --quick \
      --output-dir ./reports
  
- name: Upload Validation Report
  uses: actions/upload-artifact@v2
  with:
    name: validation-report
    path: ./reports/
```
