"""Unit tests for ValidationReportGenerator."""

import json
import pytest

from agent_scrivener.deployment.validation.report_generator import ValidationReportGenerator
from agent_scrivener.deployment.validation.models import (
    ValidationStatus,
    ValidationResult,
    PerformanceMetrics,
)


class TestValidationReportGenerator:
    """Tests for ValidationReportGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ValidationReportGenerator()
    
    def test_generate_summary_report_empty(self):
        """Test generating summary report with no results."""
        report = self.generator.generate_summary_report([])
        
        assert report.overall_status == ValidationStatus.SKIP
        assert report.readiness_score == 0.0
        assert report.total_validations == 0
        assert report.passed_validations == 0
        assert report.failed_validations == 0
    
    def test_generate_summary_report_all_pass(self):
        """Test generating summary report with all passing validations."""
        results = [
            ValidationResult(
                validator_name="test1",
                status=ValidationStatus.PASS,
                message="Passed 1"
            ),
            ValidationResult(
                validator_name="test2",
                status=ValidationStatus.PASS,
                message="Passed 2"
            )
        ]
        
        report = self.generator.generate_summary_report(results)
        
        assert report.overall_status == ValidationStatus.PASS
        assert report.readiness_score == 100.0
        assert report.total_validations == 2
        assert report.passed_validations == 2
        assert report.failed_validations == 0
        assert report.is_production_ready() is True
    
    def test_generate_summary_report_with_failures(self):
        """Test generating summary report with failures."""
        results = [
            ValidationResult(
                validator_name="test1",
                status=ValidationStatus.PASS,
                message="Passed"
            ),
            ValidationResult(
                validator_name="test2",
                status=ValidationStatus.FAIL,
                message="Failed",
                remediation_steps=["Fix the issue"]
            )
        ]
        
        report = self.generator.generate_summary_report(results)
        
        assert report.overall_status == ValidationStatus.FAIL
        assert report.readiness_score == 50.0
        assert report.total_validations == 2
        assert report.passed_validations == 1
        assert report.failed_validations == 1
        assert report.is_production_ready() is False
    
    def test_generate_summary_report_with_warnings(self):
        """Test generating summary report with warnings."""
        results = [
            ValidationResult(
                validator_name="test1",
                status=ValidationStatus.PASS,
                message="Passed"
            ),
            ValidationResult(
                validator_name="test2",
                status=ValidationStatus.WARNING,
                message="Warning"
            )
        ]
        
        report = self.generator.generate_summary_report(results)
        
        assert report.overall_status == ValidationStatus.WARNING
        assert report.readiness_score == 75.0  # (100 + 50) / 2
        assert report.total_validations == 2
        assert report.passed_validations == 1
        assert report.warning_validations == 1
    
    def test_generate_summary_report_with_skipped(self):
        """Test generating summary report with skipped validations."""
        results = [
            ValidationResult(
                validator_name="test1",
                status=ValidationStatus.PASS,
                message="Passed"
            ),
            ValidationResult(
                validator_name="test2",
                status=ValidationStatus.SKIP,
                message="Skipped"
            )
        ]
        
        report = self.generator.generate_summary_report(results)
        
        assert report.overall_status == ValidationStatus.PASS
        assert report.readiness_score == 100.0  # Skipped doesn't affect score
        assert report.total_validations == 2
        assert report.passed_validations == 1
        assert report.skipped_validations == 1
    
    def test_generate_summary_report_with_performance_metrics(self):
        """Test generating summary report with performance metrics."""
        results = [
            ValidationResult(
                validator_name="test1",
                status=ValidationStatus.PASS,
                message="Passed"
            )
        ]
        
        metrics = [
            PerformanceMetrics(
                metric_name="api_response",
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
        ]
        
        report = self.generator.generate_summary_report(results, metrics)
        
        assert len(report.performance_metrics) == 1
        assert report.performance_metrics[0].metric_name == "api_response"
    
    def test_generate_remediation_guide(self):
        """Test generating remediation guide."""
        results = [
            ValidationResult(
                validator_name="test1",
                status=ValidationStatus.FAIL,
                message="Failed validation",
                remediation_steps=["Step 1", "Step 2"]
            ),
            ValidationResult(
                validator_name="security_test",
                status=ValidationStatus.FAIL,
                message="Security issue",
                remediation_steps=["Fix security"]
            ),
            ValidationResult(
                validator_name="test2",
                status=ValidationStatus.PASS,
                message="Passed"
            )
        ]
        
        guide = self.generator.generate_remediation_guide(results)
        
        assert len(guide) == 2
        # Security issues should be first (CRITICAL priority)
        assert guide[0].validator_name == "security_test"
        assert guide[0].priority == "CRITICAL"
        assert guide[1].validator_name == "test1"
        assert guide[1].priority == "HIGH"
    
    def test_export_report_markdown(self):
        """Test exporting report as markdown."""
        results = [
            ValidationResult(
                validator_name="test1",
                status=ValidationStatus.PASS,
                message="Passed",
                duration_seconds=1.5
            )
        ]
        
        report = self.generator.generate_summary_report(results)
        markdown = self.generator.export_report(report, format="markdown")
        
        assert "# Production Readiness Validation Report" in markdown
        assert "Overall Status:" in markdown
        assert "Readiness Score:" in markdown
        assert "test1" in markdown
        assert "✅" in markdown
    
    def test_export_report_json(self):
        """Test exporting report as JSON."""
        results = [
            ValidationResult(
                validator_name="test1",
                status=ValidationStatus.PASS,
                message="Passed",
                duration_seconds=1.5
            )
        ]
        
        report = self.generator.generate_summary_report(results)
        json_output = self.generator.export_report(report, format="json")
        
        data = json.loads(json_output)
        assert data["overall_status"] == "pass"
        assert data["readiness_score"] == 100.0
        assert data["production_ready"] is True
        assert len(data["validation_results"]) == 1
        assert data["validation_results"][0]["validator"] == "test1"
    
    def test_export_report_html(self):
        """Test exporting report as HTML."""
        results = [
            ValidationResult(
                validator_name="test1",
                status=ValidationStatus.PASS,
                message="Passed",
                duration_seconds=1.5
            )
        ]
        
        report = self.generator.generate_summary_report(results)
        html = self.generator.export_report(report, format="html")
        
        assert "<!DOCTYPE html>" in html
        assert "<title>Production Readiness Validation Report</title>" in html
        assert "test1" in html
        assert "Passed" in html
    
    def test_export_report_with_performance_metrics(self):
        """Test exporting report with performance metrics."""
        results = [
            ValidationResult(
                validator_name="test1",
                status=ValidationStatus.PASS,
                message="Passed"
            )
        ]
        
        metrics = [
            PerformanceMetrics(
                metric_name="api_response",
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
        ]
        
        report = self.generator.generate_summary_report(results, metrics)
        markdown = self.generator.export_report(report, format="markdown")
        
        assert "## Performance Metrics" in markdown
        assert "api_response" in markdown
        assert "p50: 100.00ms" in markdown
        assert "p90: 200.00ms" in markdown
    
    def test_export_report_with_remediation_guide(self):
        """Test exporting report with remediation guide."""
        results = [
            ValidationResult(
                validator_name="test1",
                status=ValidationStatus.FAIL,
                message="Failed validation",
                remediation_steps=["Step 1", "Step 2"]
            )
        ]
        
        report = self.generator.generate_summary_report(results)
        markdown = self.generator.export_report(report, format="markdown")
        
        assert "## Remediation Guide" in markdown
        assert "test1" in markdown
        assert "Step 1" in markdown
        assert "Step 2" in markdown
