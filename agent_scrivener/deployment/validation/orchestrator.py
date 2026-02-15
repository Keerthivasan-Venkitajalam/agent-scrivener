"""Validation orchestrator for coordinating all validators."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set

from .models import ValidationResult, ValidationStatus, ValidationReport
from .base_validator import BaseValidator
from .report_generator import ValidationReportGenerator
from .end_to_end_validator import EndToEndValidator
from .api_endpoint_validator import APIEndpointValidator
from .orchestration_validator import OrchestrationValidator
from .deployment_config_validator import DeploymentConfigValidator
from .performance_benchmarker import PerformanceBenchmarker
from .documentation_validator import DocumentationValidator
from .monitoring_validator import MonitoringValidator
from .aws_infrastructure_validator import AWSInfrastructureValidator
from .data_persistence_validator import DataPersistenceValidator
from .security_validator import SecurityValidator


logger = logging.getLogger(__name__)


class ValidationOrchestrator:
    """Orchestrates execution of all validation checks.
    
    Coordinates running multiple validators, handles timeouts, aggregates results,
    and generates comprehensive validation reports.
    """
    
    def __init__(
        self,
        api_base_url: Optional[str] = None,
        database_url: Optional[str] = None,
        aws_region: Optional[str] = None,
        timeout_seconds: float = 300.0
    ):
        """Initialize the validation orchestrator.
        
        Args:
            api_base_url: Base URL for API endpoint validation
            database_url: Database connection URL for persistence validation
            aws_region: AWS region for infrastructure validation
            timeout_seconds: Default timeout for each validator (default: 5 minutes)
        """
        self.api_base_url = api_base_url
        self.database_url = database_url
        self.aws_region = aws_region
        self.timeout_seconds = timeout_seconds
        self.report_generator = ValidationReportGenerator()
        
        # Initialize all validators
        self.validators: Dict[str, BaseValidator] = {}
        self._initialize_validators()
    
    def _initialize_validators(self):
        """Initialize all available validators."""
        # Default auth token for testing (should be provided via environment or config)
        auth_token = "test_token_12345"
        
        # End-to-end validator
        self.validators["end-to-end"] = EndToEndValidator(
            api_base_url=self.api_base_url or "http://localhost:8000",
            auth_token=auth_token,
            timeout_minutes=int(self.timeout_seconds / 60)
        )
        
        # API endpoint validator
        self.validators["api-endpoints"] = APIEndpointValidator(
            api_base_url=self.api_base_url or "http://localhost:8000",
            ws_base_url=(self.api_base_url or "http://localhost:8000").replace("http://", "ws://").replace("https://", "wss://"),
            auth_token=auth_token
        )
        
        # Orchestration validator
        self.validators["orchestration"] = OrchestrationValidator(
            timeout_seconds=self.timeout_seconds
        )
        
        # Deployment configuration validator
        self.validators["deployment-config"] = DeploymentConfigValidator()
        
        # Performance benchmarker
        self.validators["performance"] = PerformanceBenchmarker(
            api_base_url=self.api_base_url or "http://localhost:8000",
            database_url=self.database_url,
            timeout_seconds=self.timeout_seconds * 2  # Performance tests may take longer
        )
        
        # Documentation validator
        self.validators["documentation"] = DocumentationValidator()
        
        # Monitoring validator
        self.validators["monitoring"] = MonitoringValidator(
            aws_region=self.aws_region
        )
        
        # AWS infrastructure validator
        self.validators["aws-infrastructure"] = AWSInfrastructureValidator(
            aws_region=self.aws_region
        )
        
        # Data persistence validator
        self.validators["data-persistence"] = DataPersistenceValidator(
            database_url=self.database_url
        )
        
        # Security validator
        self.validators["security"] = SecurityValidator(
            api_base_url=self.api_base_url or "http://localhost:8000",
            auth_token=auth_token
        )
    
    async def run_all_validations(
        self,
        skip_validators: Optional[Set[str]] = None
    ) -> ValidationReport:
        """Run all validators and generate a comprehensive report.
        
        Args:
            skip_validators: Set of validator names to skip (e.g., {"aws-infrastructure", "performance"})
            
        Returns:
            ValidationReport with aggregated results from all validators
        """
        logger.info("Starting complete validation suite")
        start_time = time.time()
        
        skip_validators = skip_validators or set()
        all_results: List[ValidationResult] = []
        
        # Run all validators that are not skipped
        for validator_name, validator in self.validators.items():
            if validator_name in skip_validators:
                logger.info(f"Skipping validator: {validator_name}")
                # Add a skip result
                all_results.append(
                    ValidationResult(
                        validator_name=validator_name,
                        status=ValidationStatus.SKIP,
                        message=f"Validator skipped by user request",
                        duration_seconds=0.0
                    )
                )
                continue
            
            logger.info(f"Running validator: {validator_name}")
            try:
                results = await validator.run_with_timeout()
                all_results.extend(results)
            except Exception as e:
                logger.exception(f"Validator {validator_name} failed with exception: {e}")
                all_results.append(
                    ValidationResult(
                        validator_name=validator_name,
                        status=ValidationStatus.FAIL,
                        message=f"Validator failed with exception: {str(e)}",
                        details={"exception": str(e), "exception_type": type(e).__name__},
                        duration_seconds=0.0
                    )
                )
        
        total_duration = time.time() - start_time
        logger.info(f"Validation suite completed in {total_duration:.2f} seconds")
        
        # Generate comprehensive report
        report = self.report_generator.generate_summary_report(all_results)
        return report
    
    async def run_specific_validators(
        self,
        validator_names: List[str]
    ) -> ValidationReport:
        """Run specific validators by name.
        
        Args:
            validator_names: List of validator names to run (e.g., ["api-endpoints", "security"])
            
        Returns:
            ValidationReport with results from specified validators
            
        Raises:
            ValueError: If any validator name is not recognized
        """
        logger.info(f"Running specific validators: {validator_names}")
        start_time = time.time()
        
        # Validate that all requested validators exist
        unknown_validators = set(validator_names) - set(self.validators.keys())
        if unknown_validators:
            raise ValueError(
                f"Unknown validators: {unknown_validators}. "
                f"Available validators: {list(self.validators.keys())}"
            )
        
        all_results: List[ValidationResult] = []
        
        # Run requested validators
        for validator_name in validator_names:
            validator = self.validators[validator_name]
            logger.info(f"Running validator: {validator_name}")
            
            try:
                results = await validator.run_with_timeout()
                all_results.extend(results)
            except Exception as e:
                logger.exception(f"Validator {validator_name} failed with exception: {e}")
                all_results.append(
                    ValidationResult(
                        validator_name=validator_name,
                        status=ValidationStatus.FAIL,
                        message=f"Validator failed with exception: {str(e)}",
                        details={"exception": str(e), "exception_type": type(e).__name__},
                        duration_seconds=0.0
                    )
                )
        
        total_duration = time.time() - start_time
        logger.info(f"Specific validators completed in {total_duration:.2f} seconds")
        
        # Generate report
        report = self.report_generator.generate_summary_report(all_results)
        return report
    
    async def run_quick_validation(self) -> ValidationReport:
        """Run a quick validation with only critical validators.
        
        Runs: deployment-config, security, api-endpoints
        Skips: performance, aws-infrastructure, end-to-end
        
        Returns:
            ValidationReport with results from critical validators
        """
        logger.info("Running quick validation (critical validators only)")
        
        critical_validators = [
            "deployment-config",
            "security",
            "api-endpoints",
            "documentation"
        ]
        
        return await self.run_specific_validators(critical_validators)
    
    def get_available_validators(self) -> List[str]:
        """Get list of all available validator names.
        
        Returns:
            List of validator names
        """
        return list(self.validators.keys())
    
    def get_validator_info(self) -> Dict[str, str]:
        """Get information about all available validators.
        
        Returns:
            Dictionary mapping validator names to descriptions
        """
        return {
            "end-to-end": "Validates complete research workflow from query to document",
            "api-endpoints": "Validates all REST and WebSocket API endpoints",
            "orchestration": "Validates agent coordination and workflow management",
            "deployment-config": "Validates deployment configuration files and settings",
            "performance": "Measures system performance and establishes baselines",
            "documentation": "Validates completeness of documentation",
            "monitoring": "Validates production monitoring infrastructure",
            "aws-infrastructure": "Validates AWS infrastructure readiness",
            "data-persistence": "Validates data persistence and recovery mechanisms",
            "security": "Validates security configurations and compliance"
        }
