"""
Performance test runner for Agent Scrivener.

Executes comprehensive performance test suite and generates reports.
"""

import asyncio
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pytest

try:
    from tests.performance.report_generator import PerformanceReportGenerator
    from tests.performance.config import PERF_CONFIG
    REPORTING_AVAILABLE = True
except ImportError:
    REPORTING_AVAILABLE = False
    print("Advanced reporting not available - using basic reporting")


class PerformanceTestRunner:
    """Runner for executing performance tests with reporting."""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def run_test_suite(self, test_categories: List[str] = None) -> Dict[str, Any]:
        """Run performance test suite."""
        
        if test_categories is None:
            test_categories = ["concurrent", "memory", "scalability", "limits"]
        
        print("Starting Agent Scrivener Performance Test Suite")
        print("=" * 60)
        
        overall_start = time.perf_counter()
        
        for category in test_categories:
            print(f"\nRunning {category} tests...")
            category_start = time.perf_counter()
            
            try:
                result = self._run_category_tests(category)
                category_duration = time.perf_counter() - category_start
                
                self.results[category] = {
                    "status": "completed",
                    "duration_seconds": category_duration,
                    "result": result
                }
                
                print(f"✓ {category} tests completed in {category_duration:.2f}s")
                
            except Exception as e:
                category_duration = time.perf_counter() - category_start
                
                self.results[category] = {
                    "status": "failed",
                    "duration_seconds": category_duration,
                    "error": str(e)
                }
                
                print(f"✗ {category} tests failed: {e}")
        
        overall_duration = time.perf_counter() - overall_start
        
        # Generate summary report
        summary = self._generate_summary_report(overall_duration)
        self._save_results(summary)
        
        # Generate comprehensive report if available
        if REPORTING_AVAILABLE:
            try:
                report_generator = PerformanceReportGenerator(str(self.output_dir))
                comprehensive_report = report_generator.generate_comprehensive_report(
                    summary, 
                    include_charts=True
                )
                print(f"Comprehensive report generated: {comprehensive_report.test_run_id}")
            except Exception as e:
                print(f"Warning: Could not generate comprehensive report: {e}")
        
        return summary
    
    def _run_category_tests(self, category: str) -> Dict[str, Any]:
        """Run tests for a specific category."""
        
        test_files = {
            "concurrent": "tests/performance/test_concurrent_requests.py",
            "memory": "tests/performance/test_memory_usage.py", 
            "scalability": "tests/performance/test_scalability.py",
            "limits": "tests/performance/test_system_limits.py"
        }
        
        if category not in test_files:
            raise ValueError(f"Unknown test category: {category}")
        
        test_file = test_files[category]
        
        # Run pytest with JSON report
        result_file = self.output_dir / f"{category}_results.json"
        
        exit_code = pytest.main([
            test_file,
            "-v",
            "--tb=short",
            f"--json-report={result_file}",
            "--json-report-omit=collectors"
        ])
        
        # Load and return results
        if result_file.exists():
            with open(result_file, 'r') as f:
                return json.load(f)
        else:
            return {"exit_code": exit_code, "tests": []}
    
    def _generate_summary_report(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        
        summary = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "total_duration_seconds": total_duration,
                "categories_tested": list(self.results.keys())
            },
            "overall_status": "passed",
            "category_results": {},
            "performance_metrics": {},
            "recommendations": []
        }
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        for category, result in self.results.items():
            if result["status"] == "completed" and "result" in result:
                test_result = result["result"]
                
                # Extract test statistics
                if "summary" in test_result:
                    test_summary = test_result["summary"]
                    category_total = test_summary.get("total", 0)
                    category_passed = test_summary.get("passed", 0)
                    category_failed = test_summary.get("failed", 0)
                    
                    total_tests += category_total
                    total_passed += category_passed
                    total_failed += category_failed
                    
                    summary["category_results"][category] = {
                        "status": "passed" if category_failed == 0 else "failed",
                        "total_tests": category_total,
                        "passed": category_passed,
                        "failed": category_failed,
                        "duration_seconds": result["duration_seconds"]
                    }
                else:
                    summary["category_results"][category] = {
                        "status": "unknown",
                        "duration_seconds": result["duration_seconds"]
                    }
            else:
                summary["category_results"][category] = {
                    "status": "failed",
                    "error": result.get("error", "Unknown error"),
                    "duration_seconds": result["duration_seconds"]
                }
        
        # Set overall status
        if total_failed > 0:
            summary["overall_status"] = "failed"
        
        # Add performance metrics summary
        summary["performance_metrics"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "pass_rate": (total_passed / total_tests) * 100 if total_tests > 0 else 0,
            "total_duration": total_duration
        }
        
        # Generate recommendations based on results
        summary["recommendations"] = self._generate_recommendations(summary)
        
        return summary
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on test results."""
        recommendations = []
        
        # Check overall performance
        if summary["performance_metrics"]["pass_rate"] < 80:
            recommendations.append(
                "Overall test pass rate is below 80%. Consider investigating failing tests "
                "and optimizing system performance."
            )
        
        # Check individual categories
        for category, result in summary["category_results"].items():
            if result["status"] == "failed":
                recommendations.append(
                    f"{category.title()} tests failed. Review {category} performance "
                    f"and address any bottlenecks or resource constraints."
                )
            elif result.get("total_tests", 0) > 0:
                pass_rate = (result.get("passed", 0) / result["total_tests"]) * 100
                if pass_rate < 90:
                    recommendations.append(
                        f"{category.title()} tests have {pass_rate:.1f}% pass rate. "
                        f"Consider optimizing {category} performance."
                    )
        
        # Check test duration
        if summary["test_run"]["total_duration_seconds"] > 300:  # 5 minutes
            recommendations.append(
                "Performance test suite took longer than 5 minutes to complete. "
                "Consider optimizing test execution or reducing test scope for faster feedback."
            )
        
        # Add general recommendations
        if not recommendations:
            recommendations.append(
                "All performance tests passed successfully. System performance appears optimal."
            )
        
        return recommendations
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = self.output_dir / f"performance_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save human-readable report
        text_file = self.output_dir / f"performance_report_{timestamp}.txt"
        with open(text_file, 'w') as f:
            self._write_text_report(f, summary)
        
        print(f"\nReports saved:")
        print(f"  JSON: {json_file}")
        print(f"  Text: {text_file}")
    
    def _write_text_report(self, file, summary: Dict[str, Any]):
        """Write human-readable text report."""
        file.write("Agent Scrivener Performance Test Report\n")
        file.write("=" * 50 + "\n\n")
        
        # Test run summary
        test_run = summary["test_run"]
        file.write(f"Test Run: {test_run['timestamp']}\n")
        file.write(f"Duration: {test_run['total_duration_seconds']:.2f} seconds\n")
        file.write(f"Categories: {', '.join(test_run['categories_tested'])}\n\n")
        
        # Overall status
        file.write(f"Overall Status: {summary['overall_status'].upper()}\n\n")
        
        # Performance metrics
        metrics = summary["performance_metrics"]
        file.write("Performance Metrics:\n")
        file.write(f"  Total Tests: {metrics['total_tests']}\n")
        file.write(f"  Passed: {metrics['total_passed']}\n")
        file.write(f"  Failed: {metrics['total_failed']}\n")
        file.write(f"  Pass Rate: {metrics['pass_rate']:.1f}%\n\n")
        
        # Category results
        file.write("Category Results:\n")
        for category, result in summary["category_results"].items():
            file.write(f"  {category.title()}:\n")
            file.write(f"    Status: {result['status']}\n")
            file.write(f"    Duration: {result['duration_seconds']:.2f}s\n")
            
            if "total_tests" in result:
                file.write(f"    Tests: {result['passed']}/{result['total_tests']} passed\n")
            
            if "error" in result:
                file.write(f"    Error: {result['error']}\n")
            
            file.write("\n")
        
        # Recommendations
        file.write("Recommendations:\n")
        for i, rec in enumerate(summary["recommendations"], 1):
            file.write(f"  {i}. {rec}\n")


def main():
    """Main entry point for performance test runner."""
    parser = argparse.ArgumentParser(description="Run Agent Scrivener performance tests")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["concurrent", "memory", "scalability", "limits"],
        default=["concurrent", "memory", "scalability", "limits"],
        help="Test categories to run"
    )
    parser.add_argument(
        "--output-dir",
        default="test_results",
        help="Output directory for test results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick performance tests (subset of full suite)"
    )
    
    args = parser.parse_args()
    
    # Adjust categories for quick run
    if args.quick:
        args.categories = ["concurrent", "memory"]  # Run only essential tests
    
    # Run performance tests
    runner = PerformanceTestRunner(args.output_dir)
    summary = runner.run_test_suite(args.categories)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST SUMMARY")
    print("=" * 60)
    print(f"Overall Status: {summary['overall_status'].upper()}")
    print(f"Total Tests: {summary['performance_metrics']['total_tests']}")
    print(f"Pass Rate: {summary['performance_metrics']['pass_rate']:.1f}%")
    print(f"Duration: {summary['test_run']['total_duration_seconds']:.2f}s")
    
    # Exit with appropriate code
    exit_code = 0 if summary["overall_status"] == "passed" else 1
    return exit_code


if __name__ == "__main__":
    exit(main())