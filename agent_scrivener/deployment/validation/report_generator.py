"""Validation report generator for creating comprehensive validation reports."""

import json
from datetime import datetime
from typing import List

from .models import (
    ValidationResult,
    ValidationStatus,
    ValidationReport,
    PerformanceMetrics,
    RemediationStep,
)


class ValidationReportGenerator:
    """Generates comprehensive validation reports in multiple formats."""
    
    def generate_summary_report(
        self, 
        results: List[ValidationResult],
        performance_metrics: List[PerformanceMetrics] = None
    ) -> ValidationReport:
        """Generate a summary validation report.
        
        Args:
            results: List of validation results
            performance_metrics: Optional list of performance metrics
            
        Returns:
            ValidationReport with aggregated results
        """
        if not results:
            return ValidationReport(
                overall_status=ValidationStatus.SKIP,
                readiness_score=0.0,
                total_validations=0,
                passed_validations=0,
                failed_validations=0,
                warning_validations=0,
                skipped_validations=0,
                validation_results=[],
                performance_metrics=performance_metrics or [],
                remediation_guide=[]
            )
        
        # Count validation statuses
        passed = sum(1 for r in results if r.status == ValidationStatus.PASS)
        failed = sum(1 for r in results if r.status == ValidationStatus.FAIL)
        warnings = sum(1 for r in results if r.status == ValidationStatus.WARNING)
        skipped = sum(1 for r in results if r.status == ValidationStatus.SKIP)
        total = len(results)
        
        # Calculate readiness score (0-100)
        # Pass = 100%, Warning = 50%, Fail/Timeout = 0%, Skip = neutral (not counted)
        scored_validations = total - skipped
        if scored_validations > 0:
            score_points = (passed * 100) + (warnings * 50)
            readiness_score = score_points / scored_validations
        else:
            readiness_score = 0.0
        
        # Determine overall status
        if failed > 0:
            overall_status = ValidationStatus.FAIL
        elif warnings > 0:
            overall_status = ValidationStatus.WARNING
        elif passed > 0:
            overall_status = ValidationStatus.PASS
        else:
            overall_status = ValidationStatus.SKIP
        
        # Generate remediation guide
        remediation_guide = self.generate_remediation_guide(results)
        
        return ValidationReport(
            overall_status=overall_status,
            readiness_score=readiness_score,
            total_validations=total,
            passed_validations=passed,
            failed_validations=failed,
            warning_validations=warnings,
            skipped_validations=skipped,
            validation_results=results,
            performance_metrics=performance_metrics or [],
            remediation_guide=remediation_guide
        )
    
    def generate_detailed_report(
        self, 
        results: List[ValidationResult],
        performance_metrics: List[PerformanceMetrics] = None
    ) -> ValidationReport:
        """Generate a detailed validation report.
        
        This is currently the same as summary report but can be extended
        with additional details in the future.
        
        Args:
            results: List of validation results
            performance_metrics: Optional list of performance metrics
            
        Returns:
            ValidationReport with detailed information
        """
        return self.generate_summary_report(results, performance_metrics)
    
    def generate_remediation_guide(
        self, 
        failures: List[ValidationResult]
    ) -> List[RemediationStep]:
        """Generate remediation guide for failures.
        
        Args:
            failures: List of validation results (typically failures)
            
        Returns:
            List of remediation steps
        """
        remediation_steps = []
        
        for result in failures:
            if result.is_failure() and result.remediation_steps:
                # Determine priority based on validator name
                priority = "CRITICAL" if "security" in result.validator_name.lower() else "HIGH"
                
                step = RemediationStep(
                    validator_name=result.validator_name,
                    issue=result.message,
                    steps=result.remediation_steps,
                    priority=priority
                )
                remediation_steps.append(step)
        
        # Sort by priority
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        remediation_steps.sort(key=lambda s: priority_order.get(s.priority, 4))
        
        return remediation_steps
    
    def export_report(
        self, 
        report: ValidationReport, 
        format: str = "markdown"
    ) -> str:
        """Export report in specified format.
        
        Args:
            report: ValidationReport to export
            format: Output format (markdown, json, html)
            
        Returns:
            Formatted report as string
        """
        if format == "json":
            return self._export_json(report)
        elif format == "html":
            return self._export_html(report)
        else:  # default to markdown
            return self._export_markdown(report)
    
    def _export_markdown(self, report: ValidationReport) -> str:
        """Export report as Markdown.
        
        Args:
            report: ValidationReport to export
            
        Returns:
            Markdown formatted report
        """
        lines = []
        
        # Header
        lines.append("# Production Readiness Validation Report")
        lines.append(f"\n**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\n**Overall Status:** {report.overall_status.value.upper()}")
        lines.append(f"\n**Readiness Score:** {report.readiness_score:.1f}/100")
        
        # Summary
        lines.append("\n## Summary")
        lines.append(f"\n- Total Validations: {report.total_validations}")
        lines.append(f"- Passed: {report.passed_validations} ✅")
        lines.append(f"- Failed: {report.failed_validations} ❌")
        lines.append(f"- Warnings: {report.warning_validations} ⚠️")
        lines.append(f"- Skipped: {report.skipped_validations} ⏭️")
        
        # Production readiness
        lines.append("\n## Production Readiness")
        if report.is_production_ready():
            lines.append("\n✅ **System is ready for production deployment**")
        else:
            lines.append("\n❌ **System is NOT ready for production deployment**")
            lines.append("\nCritical issues must be resolved before deployment.")
        
        # Validation results
        lines.append("\n## Validation Results")
        
        # Group by status
        for status in [ValidationStatus.FAIL, ValidationStatus.WARNING, ValidationStatus.PASS]:
            status_results = [r for r in report.validation_results if r.status == status]
            if status_results:
                status_icon = {"fail": "❌", "warning": "⚠️", "pass": "✅"}[status.value]
                lines.append(f"\n### {status.value.upper()} {status_icon}")
                for result in status_results:
                    lines.append(f"\n**{result.validator_name}**")
                    lines.append(f"- Message: {result.message}")
                    lines.append(f"- Duration: {result.duration_seconds:.2f}s")
                    if result.details:
                        lines.append(f"- Details: {json.dumps(result.details, indent=2)}")
        
        # Performance metrics
        if report.performance_metrics:
            lines.append("\n## Performance Metrics")
            for metric in report.performance_metrics:
                lines.append(f"\n### {metric.metric_name}")
                lines.append(f"- p50: {metric.p50_ms:.2f}ms")
                lines.append(f"- p90: {metric.p90_ms:.2f}ms")
                lines.append(f"- p95: {metric.p95_ms:.2f}ms")
                lines.append(f"- p99: {metric.p99_ms:.2f}ms")
                lines.append(f"- Min: {metric.min_ms:.2f}ms")
                lines.append(f"- Max: {metric.max_ms:.2f}ms")
                lines.append(f"- Mean: {metric.mean_ms:.2f}ms")
                lines.append(f"- Samples: {metric.sample_count}")
        
        # Remediation guide
        if report.remediation_guide:
            lines.append("\n## Remediation Guide")
            for step in report.remediation_guide:
                lines.append(f"\n### {step.priority}: {step.validator_name}")
                lines.append(f"\n**Issue:** {step.issue}")
                lines.append("\n**Steps to resolve:**")
                for i, s in enumerate(step.steps, 1):
                    lines.append(f"{i}. {s}")
                if step.documentation_link:
                    lines.append(f"\n**Documentation:** {step.documentation_link}")
        
        return "\n".join(lines)
    
    def _export_json(self, report: ValidationReport) -> str:
        """Export report as JSON.
        
        Args:
            report: ValidationReport to export
            
        Returns:
            JSON formatted report
        """
        data = {
            "generated_at": report.generated_at.isoformat(),
            "overall_status": report.overall_status.value,
            "readiness_score": report.readiness_score,
            "production_ready": report.is_production_ready(),
            "summary": {
                "total": report.total_validations,
                "passed": report.passed_validations,
                "failed": report.failed_validations,
                "warnings": report.warning_validations,
                "skipped": report.skipped_validations
            },
            "validation_results": [
                {
                    "validator": r.validator_name,
                    "status": r.status.value,
                    "message": r.message,
                    "duration_seconds": r.duration_seconds,
                    "timestamp": r.timestamp.isoformat(),
                    "details": r.details,
                    "remediation_steps": r.remediation_steps
                }
                for r in report.validation_results
            ],
            "performance_metrics": [
                {
                    "metric_name": m.metric_name,
                    "p50_ms": m.p50_ms,
                    "p90_ms": m.p90_ms,
                    "p95_ms": m.p95_ms,
                    "p99_ms": m.p99_ms,
                    "min_ms": m.min_ms,
                    "max_ms": m.max_ms,
                    "mean_ms": m.mean_ms,
                    "std_dev_ms": m.std_dev_ms,
                    "sample_count": m.sample_count
                }
                for m in report.performance_metrics
            ],
            "remediation_guide": [
                {
                    "validator": s.validator_name,
                    "priority": s.priority,
                    "issue": s.issue,
                    "steps": s.steps,
                    "documentation_link": s.documentation_link
                }
                for s in report.remediation_guide
            ]
        }
        
        return json.dumps(data, indent=2)
    
    def _export_html(self, report: ValidationReport) -> str:
        """Export report as HTML.
        
        Args:
            report: ValidationReport to export
            
        Returns:
            HTML formatted report
        """
        # Simple HTML template
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Production Readiness Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .status-pass {{ color: green; }}
        .status-fail {{ color: red; }}
        .status-warning {{ color: orange; }}
        .metric {{ margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>Production Readiness Validation Report</h1>
    <p><strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Overall Status:</strong> <span class="status-{report.overall_status.value}">{report.overall_status.value.upper()}</span></p>
    <p><strong>Readiness Score:</strong> {report.readiness_score:.1f}/100</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <ul>
            <li>Total Validations: {report.total_validations}</li>
            <li>Passed: {report.passed_validations} ✅</li>
            <li>Failed: {report.failed_validations} ❌</li>
            <li>Warnings: {report.warning_validations} ⚠️</li>
            <li>Skipped: {report.skipped_validations} ⏭️</li>
        </ul>
    </div>
    
    <h2>Production Readiness</h2>
    <p class="status-{'pass' if report.is_production_ready() else 'fail'}">
        {'✅ System is ready for production deployment' if report.is_production_ready() else '❌ System is NOT ready for production deployment'}
    </p>
    
    <h2>Validation Results</h2>
    <table>
        <tr>
            <th>Validator</th>
            <th>Status</th>
            <th>Message</th>
            <th>Duration</th>
        </tr>
"""
        
        for result in report.validation_results:
            html += f"""
        <tr>
            <td>{result.validator_name}</td>
            <td class="status-{result.status.value}">{result.status.value.upper()}</td>
            <td>{result.message}</td>
            <td>{result.duration_seconds:.2f}s</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        return html
