# Production Readiness Validation - Development Guide

## Overview

This guide explains how to extend the Production Readiness Validation framework by adding new validators, writing property-based tests, and contributing to the validation system.

## Architecture

### Core Components

```
validation/
├── models.py                    # Data models
├── base_validator.py            # Base validator class
├── orchestrator.py              # Validation orchestrator
├── cli.py                       # Command-line interface
├── report_generator.py          # Report generation
└── [validator_name]_validator.py # Individual validators
```

### Data Flow

```
CLI → Orchestrator → Validators → Results → Report Generator → Output
```

1. **CLI** parses command-line arguments
2. **Orchestrator** initializes and runs validators
3. **Validators** perform checks and return results
4. **Report Generator** aggregates results into reports
5. **Output** displays or saves reports

## Data Models

### ValidationStatus

Enum representing validation outcomes:

```python
from enum import Enum

class ValidationStatus(Enum):
    PASS = "pass"      # Validation succeeded
    FAIL = "fail"      # Validation failed (critical)
    WARNING = "warning"  # Validation passed with concerns
    SKIP = "skip"      # Validation was skipped
    TIMEOUT = "timeout"  # Validation exceeded timeout
```

### ValidationResult

Result of a single validation check:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

@dataclass
class ValidationResult:
    validator_name: str              # Name of the validator
    status: ValidationStatus         # Validation status
    message: str                     # Human-readable message
    duration_seconds: float          # Time taken
    details: Dict[str, Any] = None   # Additional details
    remediation_steps: List[str] = None  # Steps to fix failures
    timestamp: datetime = None       # When validation ran
```

### PerformanceMetrics

Performance metrics with percentile breakdowns:

```python
@dataclass
class PerformanceMetrics:
    metric_name: str    # Name of the metric
    p50_ms: float       # 50th percentile (median)
    p90_ms: float       # 90th percentile
    p95_ms: float       # 95th percentile
    p99_ms: float       # 99th percentile
    min_ms: float       # Minimum value
    max_ms: float       # Maximum value
    mean_ms: float      # Mean value
    std_dev_ms: float   # Standard deviation
    sample_count: int   # Number of samples
```

### ValidationReport

Comprehensive validation report:

```python
@dataclass
class ValidationReport:
    overall_status: ValidationStatus      # Overall status
    readiness_score: float                # Score 0-100
    total_validations: int                # Total checks
    passed_validations: int               # Passed checks
    failed_validations: int               # Failed checks
    warning_validations: int              # Warning checks
    skipped_validations: int              # Skipped checks
    validation_results: List[ValidationResult]  # All results
    performance_metrics: List[PerformanceMetrics]  # Performance data
    remediation_guide: List[RemediationStep]  # Fix instructions
    generated_at: datetime                # Report timestamp
    
    def is_production_ready(self) -> bool:
        """Check if system is ready for production."""
        return (
            self.overall_status != ValidationStatus.FAIL and
            self.readiness_score >= 80.0
        )
    
    def get_critical_failures(self) -> List[ValidationResult]:
        """Get all critical failures."""
        return [
            r for r in self.validation_results
            if r.status == ValidationStatus.FAIL
        ]
```

## Creating a New Validator

### Step 1: Inherit from BaseValidator

All validators must inherit from `BaseValidator`:

```python
from agent_scrivener.deployment.validation.base_validator import BaseValidator
from agent_scrivener.deployment.validation.models import ValidationStatus

class MyValidator(BaseValidator):
    def __init__(self, timeout_seconds: float = 60.0):
        super().__init__(
            validator_name="my-validator",
            timeout_seconds=timeout_seconds
        )
    
    async def validate(self):
        """Perform validation checks.
        
        Returns:
            List[ValidationResult]: Validation results
        """
        self.log_validation_start()
        results = []
        
        # Add validation checks here
        
        self.log_validation_complete(results)
        return results
```

### Step 2: Implement Validation Logic

Add validation checks using helper methods:

```python
async def validate(self):
    self.log_validation_start()
    results = []
    
    # Example: Test a connection
    try:
        connection, duration = await self.measure_operation(
            self.test_connection
        )
        
        results.append(
            self.create_result(
                status=ValidationStatus.PASS,
                message="Connection successful",
                duration_seconds=duration,
                details={"host": connection.host, "port": connection.port}
            )
        )
    except Exception as e:
        results.append(
            self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Connection failed: {e}",
                remediation_steps=[
                    "Check network connectivity",
                    "Verify credentials",
                    "Ensure service is running"
                ]
            )
        )
    
    # Example: Check configuration
    config_valid = self.check_configuration()
    if config_valid:
        results.append(
            self.create_result(
                status=ValidationStatus.PASS,
                message="Configuration is valid"
            )
        )
    else:
        results.append(
            self.create_result(
                status=ValidationStatus.WARNING,
                message="Configuration has non-critical issues",
                remediation_steps=[
                    "Review configuration file",
                    "Update deprecated settings"
                ]
            )
        )
    
    self.log_validation_complete(results)
    return results

async def test_connection(self):
    """Test connection to service."""
    # Implementation here
    pass

def check_configuration(self) -> bool:
    """Check configuration validity."""
    # Implementation here
    pass
```

### Step 3: Use BaseValidator Helper Methods

#### create_result()

Create a validation result:

```python
result = self.create_result(
    status=ValidationStatus.PASS,
    message="Check passed",
    duration_seconds=1.5,
    details={"key": "value"},
    remediation_steps=["Step 1", "Step 2"]
)
```

#### measure_operation()

Measure operation duration:

```python
result, duration = await self.measure_operation(my_async_function)
# or
result, duration = await self.measure_operation(
    lambda: my_function_with_args(arg1, arg2)
)
```

#### log_validation_start() / log_validation_complete()

Log validation lifecycle:

```python
self.log_validation_start()
# ... perform validations ...
self.log_validation_complete(results)
```

#### create_remediation_step()

Create remediation steps:

```python
step = self.create_remediation_step(
    step_number=1,
    description="Check database connection",
    command="psql $DATABASE_URL",
    expected_outcome="Connection successful"
)
```

### Step 4: Register Validator in Orchestrator

Add your validator to `orchestrator.py`:

```python
def _initialize_validators(self):
    # ... existing validators ...
    
    # Add your validator
    self.validators["my-validator"] = MyValidator(
        timeout_seconds=self.timeout_seconds
    )
```

Update `get_validator_info()`:

```python
def get_validator_info(self) -> Dict[str, str]:
    return {
        # ... existing validators ...
        "my-validator": "Description of what my validator does"
    }
```

### Step 5: Write Tests

Create test file `tests/validation/test_my_validator.py`:

```python
import pytest
from agent_scrivener.deployment.validation.my_validator import MyValidator
from agent_scrivener.deployment.validation.models import ValidationStatus

@pytest.mark.asyncio
async def test_my_validator_success():
    """Test validator with valid configuration."""
    validator = MyValidator()
    results = await validator.validate()
    
    assert len(results) > 0
    assert all(r.status in [ValidationStatus.PASS, ValidationStatus.WARNING] for r in results)

@pytest.mark.asyncio
async def test_my_validator_failure():
    """Test validator with invalid configuration."""
    validator = MyValidator()
    # Set up failure condition
    
    results = await validator.validate()
    
    assert any(r.status == ValidationStatus.FAIL for r in results)
    assert any(r.remediation_steps for r in results)
```

## Validator Interface Requirements

All validators must:

1. **Inherit from BaseValidator**
2. **Implement async validate() method** that returns `List[ValidationResult]`
3. **Handle exceptions gracefully** and return FAIL results
4. **Provide remediation steps** for failures
5. **Use appropriate timeouts** (default: 60 seconds)
6. **Log validation lifecycle** (start/complete)
7. **Include detailed error information** in results

### Required Methods

```python
class MyValidator(BaseValidator):
    async def validate(self) -> List[ValidationResult]:
        """Perform validation checks.
        
        Must be implemented by all validators.
        
        Returns:
            List[ValidationResult]: Validation results
        """
        raise NotImplementedError
```

### Optional Methods

```python
class MyValidator(BaseValidator):
    async def setup(self):
        """Optional setup before validation."""
        pass
    
    async def teardown(self):
        """Optional cleanup after validation."""
        pass
```

## Property-Based Testing

### Overview

Property-based testing validates universal properties across all inputs using randomized testing. The framework uses **pytest** with **Hypothesis**.

### Property Test Structure

```python
import pytest
from hypothesis import given, strategies as st
from agent_scrivener.deployment.validation.my_validator import MyValidator

# Feature: production-readiness-validation, Property 1: Description of property
@pytest.mark.asyncio
@given(
    input_data=st.text(min_size=1, max_size=100)
)
async def test_property_input_validation(input_data):
    """Property: All inputs should be validated and sanitized.
    
    Validates: Requirements X.Y
    """
    validator = MyValidator()
    results = await validator.validate_input(input_data)
    
    # Property: No unhandled exceptions
    assert results is not None
    
    # Property: All results have valid status
    assert all(r.status in ValidationStatus for r in results)
```

### Property Test Patterns

#### 1. Invariant Properties

Properties that always hold true:

```python
# Feature: production-readiness-validation, Property 2: Session isolation
@pytest.mark.asyncio
@given(
    session_count=st.integers(min_value=1, max_value=10)
)
async def test_property_session_isolation(session_count):
    """Property: Concurrent sessions should be isolated.
    
    Validates: Requirements 3.4
    """
    validator = OrchestrationValidator()
    
    # Create multiple sessions
    sessions = await validator.create_concurrent_sessions(session_count)
    
    # Property: Each session has unique ID
    session_ids = [s.id for s in sessions]
    assert len(session_ids) == len(set(session_ids))
    
    # Property: No cross-session data leakage
    for session in sessions:
        assert session.data_belongs_to_session(session.id)
```

#### 2. Round-Trip Properties

Operations that should be reversible:

```python
# Feature: production-readiness-validation, Property 3: Data persistence
@pytest.mark.asyncio
@given(
    session_data=st.dictionaries(
        keys=st.text(min_size=1, max_size=50),
        values=st.text(min_size=1, max_size=100)
    )
)
async def test_property_data_persistence_roundtrip(session_data):
    """Property: Persisted data should be retrievable unchanged.
    
    Validates: Requirements 9.1, 9.2
    """
    validator = DataPersistenceValidator()
    
    # Persist data
    session_id = await validator.persist_session(session_data)
    
    # Retrieve data
    retrieved_data = await validator.retrieve_session(session_id)
    
    # Property: Retrieved data matches original
    assert retrieved_data == session_data
```

#### 3. Error Handling Properties

Properties about error handling:

```python
# Feature: production-readiness-validation, Property 4: Error capture
@pytest.mark.asyncio
@given(
    error_type=st.sampled_from([ValueError, TypeError, RuntimeError])
)
async def test_property_error_capture(error_type):
    """Property: All errors should be captured with details.
    
    Validates: Requirements 1.6
    """
    validator = EndToEndValidator()
    
    # Simulate error
    with pytest.raises(error_type):
        await validator.simulate_error(error_type)
    
    # Property: Error was captured
    error_info = validator.get_last_error()
    assert error_info is not None
    assert error_info.error_type == error_type.__name__
    assert error_info.stack_trace is not None
```

#### 4. Performance Properties

Properties about performance characteristics:

```python
# Feature: production-readiness-validation, Property 5: Response time
@pytest.mark.asyncio
@given(
    endpoint=st.sampled_from(["/health", "/status", "/research"])
)
async def test_property_response_time(endpoint):
    """Property: API endpoints should respond within time limits.
    
    Validates: Requirements 2.2, 5.2, 5.3
    """
    validator = APIEndpointValidator()
    
    # Measure response time
    start = time.time()
    response = await validator.call_endpoint(endpoint)
    duration = time.time() - start
    
    # Property: Response time within limits
    time_limits = {
        "/health": 0.1,  # 100ms
        "/status": 0.2,  # 200ms
        "/research": 2.0  # 2 seconds
    }
    assert duration <= time_limits[endpoint]
```

### Hypothesis Strategies

Common strategies for generating test data:

```python
from hypothesis import strategies as st

# Text data
st.text(min_size=1, max_size=100)
st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')))

# Numbers
st.integers(min_value=1, max_value=100)
st.floats(min_value=0.0, max_value=1.0)

# Collections
st.lists(st.integers(), min_size=1, max_size=10)
st.dictionaries(keys=st.text(), values=st.integers())

# Choices
st.sampled_from(['option1', 'option2', 'option3'])
st.one_of(st.integers(), st.text(), st.none())

# Composite strategies
@st.composite
def session_data(draw):
    return {
        'id': draw(st.uuids()),
        'query': draw(st.text(min_size=10, max_size=200)),
        'status': draw(st.sampled_from(['pending', 'running', 'completed']))
    }
```

### Property Test Configuration

Configure Hypothesis in `pytest.ini` or test file:

```python
from hypothesis import settings, HealthCheck

# Run 100 iterations per test
@settings(max_examples=100)
@given(...)
async def test_property(...):
    pass

# Increase timeout for slow tests
@settings(max_examples=50, deadline=5000)  # 5 second deadline
@given(...)
async def test_property(...):
    pass

# Suppress specific health checks
@settings(suppress_health_check=[HealthCheck.too_slow])
@given(...)
async def test_property(...):
    pass
```

## Report Generation

### Custom Report Formats

Extend `ValidationReportGenerator` to add custom formats:

```python
from agent_scrivener.deployment.validation.report_generator import ValidationReportGenerator

class CustomReportGenerator(ValidationReportGenerator):
    def export_report(self, report: ValidationReport, format: str = "markdown") -> str:
        if format == "custom":
            return self._generate_custom_report(report)
        return super().export_report(report, format)
    
    def _generate_custom_report(self, report: ValidationReport) -> str:
        """Generate custom format report."""
        lines = []
        lines.append(f"Custom Report: {report.generated_at}")
        lines.append(f"Status: {report.overall_status.value}")
        lines.append(f"Score: {report.readiness_score}")
        # Add custom formatting
        return "\n".join(lines)
```

### Adding Metrics to Reports

Include custom metrics in reports:

```python
# In your validator
async def validate(self):
    results = []
    
    # Collect performance metrics
    latencies = []
    for i in range(100):
        start = time.time()
        await self.perform_operation()
        latencies.append((time.time() - start) * 1000)
    
    # Create performance metrics
    metrics = PerformanceMetrics(
        metric_name="operation_latency",
        p50_ms=np.percentile(latencies, 50),
        p90_ms=np.percentile(latencies, 90),
        p95_ms=np.percentile(latencies, 95),
        p99_ms=np.percentile(latencies, 99),
        min_ms=min(latencies),
        max_ms=max(latencies),
        mean_ms=np.mean(latencies),
        std_dev_ms=np.std(latencies),
        sample_count=len(latencies)
    )
    
    # Include in result details
    results.append(
        self.create_result(
            status=ValidationStatus.PASS,
            message="Performance metrics collected",
            details={"metrics": metrics}
        )
    )
    
    return results
```

## Best Practices

### Validator Design

1. **Single Responsibility**: Each validator should focus on one aspect of the system
2. **Idempotent**: Running a validator multiple times should produce the same results
3. **Independent**: Validators should not depend on each other
4. **Fast**: Aim for validators to complete in under 60 seconds
5. **Informative**: Provide detailed error messages and remediation steps

### Error Handling

1. **Catch all exceptions**: Never let exceptions propagate unhandled
2. **Provide context**: Include relevant details in error messages
3. **Suggest fixes**: Always provide remediation steps for failures
4. **Log appropriately**: Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)

### Testing

1. **Test success and failure paths**: Cover both valid and invalid scenarios
2. **Use property tests**: Validate universal properties across all inputs
3. **Mock external dependencies**: Use mocks for AWS, databases, APIs
4. **Test timeouts**: Verify validators handle timeouts correctly
5. **Test concurrency**: Ensure validators work with concurrent execution

### Performance

1. **Use async/await**: All validators should be async
2. **Parallel execution**: Run independent checks concurrently
3. **Appropriate timeouts**: Set reasonable timeouts for operations
4. **Resource cleanup**: Always clean up resources (connections, files)
5. **Efficient algorithms**: Use efficient data structures and algorithms

### Documentation

1. **Docstrings**: Document all public methods
2. **Type hints**: Use type hints for all parameters and return values
3. **Examples**: Provide usage examples in docstrings
4. **Requirements**: Link validation checks to requirements
5. **Properties**: Document properties being validated

## Example: Complete Validator

Here's a complete example of a well-designed validator:

```python
"""Database validator for production readiness validation."""

import asyncio
import logging
from typing import List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

from .base_validator import BaseValidator
from .models import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


class DatabaseValidator(BaseValidator):
    """Validates database configuration and connectivity.
    
    Checks:
    - Database connection
    - Required tables exist
    - Database version
    - Connection pool configuration
    - Backup configuration
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        timeout_seconds: float = 60.0
    ):
        """Initialize database validator.
        
        Args:
            database_url: Database connection URL
            timeout_seconds: Timeout for validation (default: 60s)
        """
        super().__init__(
            validator_name="database",
            timeout_seconds=timeout_seconds
        )
        self.database_url = database_url
        self.connection = None
    
    async def validate(self) -> List[ValidationResult]:
        """Perform database validation checks.
        
        Returns:
            List[ValidationResult]: Validation results
        """
        self.log_validation_start()
        results = []
        
        # Check database URL configured
        if not self.database_url:
            results.append(
                self.create_result(
                    status=ValidationStatus.FAIL,
                    message="Database URL not configured",
                    remediation_steps=[
                        "Set DATABASE_URL environment variable",
                        "Format: postgresql://user:pass@host:port/dbname"
                    ]
                )
            )
            self.log_validation_complete(results)
            return results
        
        # Test database connection
        connection_result = await self._test_connection()
        results.append(connection_result)
        
        if connection_result.status == ValidationStatus.FAIL:
            self.log_validation_complete(results)
            return results
        
        # Check required tables
        tables_result = await self._check_required_tables()
        results.append(tables_result)
        
        # Check database version
        version_result = await self._check_database_version()
        results.append(version_result)
        
        # Check backup configuration
        backup_result = await self._check_backup_configuration()
        results.append(backup_result)
        
        # Cleanup
        await self._cleanup()
        
        self.log_validation_complete(results)
        return results
    
    async def _test_connection(self) -> ValidationResult:
        """Test database connection.
        
        Returns:
            ValidationResult: Connection test result
        """
        try:
            connection, duration = await self.measure_operation(
                self._connect_to_database
            )
            self.connection = connection
            
            return self.create_result(
                status=ValidationStatus.PASS,
                message="Database connection successful",
                duration_seconds=duration,
                details={
                    "host": connection.info.host,
                    "port": connection.info.port,
                    "database": connection.info.dbname
                }
            )
        except Exception as e:
            logger.exception("Database connection failed")
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Database connection failed: {e}",
                remediation_steps=[
                    "Verify database is running",
                    "Check database credentials",
                    "Verify network connectivity",
                    "Test connection: psql $DATABASE_URL"
                ]
            )
    
    async def _check_required_tables(self) -> ValidationResult:
        """Check that required tables exist.
        
        Returns:
            ValidationResult: Table check result
        """
        required_tables = ["sessions", "documents", "agents", "logs"]
        
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            existing_tables = {row['table_name'] for row in cursor.fetchall()}
            cursor.close()
            
            missing_tables = set(required_tables) - existing_tables
            
            if missing_tables:
                return self.create_result(
                    status=ValidationStatus.FAIL,
                    message=f"Missing required tables: {missing_tables}",
                    details={"missing_tables": list(missing_tables)},
                    remediation_steps=[
                        "Run database migrations",
                        "Execute: python -m agent_scrivener.db.migrate"
                    ]
                )
            
            return self.create_result(
                status=ValidationStatus.PASS,
                message="All required tables exist",
                details={"tables": list(existing_tables)}
            )
        except Exception as e:
            logger.exception("Failed to check tables")
            return self.create_result(
                status=ValidationStatus.FAIL,
                message=f"Failed to check tables: {e}"
            )
    
    async def _check_database_version(self) -> ValidationResult:
        """Check database version.
        
        Returns:
            ValidationResult: Version check result
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            cursor.close()
            
            # Check minimum version (PostgreSQL 12+)
            if "PostgreSQL 12" in version or "PostgreSQL 13" in version or \
               "PostgreSQL 14" in version or "PostgreSQL 15" in version:
                return self.create_result(
                    status=ValidationStatus.PASS,
                    message=f"Database version: {version}",
                    details={"version": version}
                )
            else:
                return self.create_result(
                    status=ValidationStatus.WARNING,
                    message=f"Database version may be outdated: {version}",
                    details={"version": version},
                    remediation_steps=[
                        "Consider upgrading to PostgreSQL 12 or later"
                    ]
                )
        except Exception as e:
            logger.exception("Failed to check database version")
            return self.create_result(
                status=ValidationStatus.WARNING,
                message=f"Could not determine database version: {e}"
            )
    
    async def _check_backup_configuration(self) -> ValidationResult:
        """Check backup configuration.
        
        Returns:
            ValidationResult: Backup check result
        """
        # This is a placeholder - actual implementation would check
        # backup schedules, retention policies, etc.
        return self.create_result(
            status=ValidationStatus.WARNING,
            message="Backup configuration check not implemented",
            remediation_steps=[
                "Verify automated backup schedules are configured",
                "Check backup retention policies",
                "Test backup restoration process"
            ]
        )
    
    async def _connect_to_database(self):
        """Connect to database.
        
        Returns:
            Database connection
        """
        return psycopg2.connect(self.database_url)
    
    async def _cleanup(self):
        """Clean up database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
```

## Contributing

When contributing new validators:

1. Follow the validator interface requirements
2. Write comprehensive tests (unit and property tests)
3. Document all public methods
4. Provide clear remediation steps
5. Update orchestrator to register validator
6. Update CLI help text
7. Add examples to documentation

## Next Steps

- Review existing validators for examples
- Read the [Validation User Guide](validation_guide.md)
- Explore the test suite in `tests/validation/`
- Contribute new validators or improvements

For questions or support, refer to the project documentation or open an issue.
