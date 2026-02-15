"""Command-line interface for running production readiness validations."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Set

from .orchestrator import ValidationOrchestrator
from .models import ValidationStatus


def setup_logging(verbose: bool = False):
    """Configure logging for the CLI.
    
    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_arguments():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Production Readiness Validation for Agent Scrivener",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
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
        """
    )
    
    # Validation mode options
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation with only critical validators (deployment-config, security, api-endpoints, documentation)"
    )
    mode_group.add_argument(
        "--only",
        nargs="+",
        metavar="VALIDATOR",
        help="Run only specific validators (e.g., --only api-endpoints security)"
    )
    
    # Skip options
    parser.add_argument(
        "--skip-aws",
        action="store_true",
        help="Skip AWS infrastructure validation"
    )
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Skip performance benchmarking"
    )
    parser.add_argument(
        "--skip-end-to-end",
        action="store_true",
        help="Skip end-to-end workflow validation"
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="VALIDATOR",
        help="Skip specific validators (e.g., --skip monitoring orchestration)"
    )
    
    # Configuration options
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_BASE_URL", "http://localhost:8000"),
        help="Base URL for API endpoint validation (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="Database connection URL for persistence validation"
    )
    parser.add_argument(
        "--aws-region",
        default=os.getenv("AWS_REGION", "us-east-1"),
        help="AWS region for infrastructure validation (default: us-east-1)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for each validator (default: 300)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save validation reports (default: current directory)"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "html"],
        default="markdown",
        help="Report output format (default: markdown)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # List validators
    parser.add_argument(
        "--list-validators",
        action="store_true",
        help="List all available validators and exit"
    )
    
    return parser.parse_args()


def get_skip_validators(args) -> Set[str]:
    """Determine which validators to skip based on arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Set of validator names to skip
    """
    skip_validators = set()
    
    if args.skip_aws:
        skip_validators.add("aws-infrastructure")
    
    if args.skip_performance:
        skip_validators.add("performance")
    
    if args.skip_end_to_end:
        skip_validators.add("end-to-end")
    
    if args.skip:
        skip_validators.update(args.skip)
    
    return skip_validators


def save_report(report_content: str, output_dir: Optional[Path], format: str):
    """Save validation report to file.
    
    Args:
        report_content: Report content to save
        output_dir: Directory to save report (None for current directory)
        format: Report format (markdown, json, html)
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()
    
    # Generate filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"validation_report_{timestamp}.{format}"
    filepath = output_dir / filename
    
    filepath.write_text(report_content)
    print(f"\nReport saved to: {filepath}")


async def run_validation(args):
    """Run validation based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Initialize orchestrator
    orchestrator = ValidationOrchestrator(
        api_base_url=args.api_url,
        database_url=args.database_url,
        aws_region=args.aws_region,
        timeout_seconds=args.timeout
    )
    
    # List validators if requested
    if args.list_validators:
        print("\nAvailable Validators:")
        print("=" * 80)
        for name, description in orchestrator.get_validator_info().items():
            print(f"  {name:25} {description}")
        print()
        return 0
    
    # Run validations based on mode
    try:
        if args.quick:
            print("\nRunning quick validation (critical validators only)...")
            report = await orchestrator.run_quick_validation()
        elif args.only:
            print(f"\nRunning specific validators: {', '.join(args.only)}...")
            report = await orchestrator.run_specific_validators(args.only)
        else:
            skip_validators = get_skip_validators(args)
            if skip_validators:
                print(f"\nSkipping validators: {', '.join(skip_validators)}")
            print("\nRunning complete validation suite...")
            report = await orchestrator.run_all_validations(skip_validators=skip_validators)
    
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUnexpected error during validation: {e}", file=sys.stderr)
        logging.exception("Validation failed with exception")
        return 1
    
    # Generate and display report
    print("\n" + "=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)
    
    report_content = orchestrator.report_generator.export_report(report, format=args.format)
    print(report_content)
    
    # Save report if output directory specified
    if args.output_dir:
        save_report(report_content, args.output_dir, args.format)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Overall Status: {report.overall_status.value.upper()}")
    print(f"Readiness Score: {report.readiness_score:.1f}/100")
    print(f"Total Validations: {report.total_validations}")
    print(f"  Passed: {report.passed_validations}")
    print(f"  Failed: {report.failed_validations}")
    print(f"  Warnings: {report.warning_validations}")
    print(f"  Skipped: {report.skipped_validations}")
    
    # Print production readiness status
    print("\n" + "=" * 80)
    if report.is_production_ready():
        print("✓ SYSTEM IS READY FOR PRODUCTION DEPLOYMENT")
        print("=" * 80)
        return 0
    else:
        print("✗ SYSTEM IS NOT READY FOR PRODUCTION DEPLOYMENT")
        print("=" * 80)
        print("\nCritical failures must be resolved before deployment.")
        
        # Print critical failures
        critical_failures = report.get_critical_failures()
        if critical_failures:
            print(f"\nCritical Failures ({len(critical_failures)}):")
            for failure in critical_failures:
                print(f"  - {failure.validator_name}: {failure.message}")
        
        return 1


def main():
    """Main entry point for the CLI."""
    args = parse_arguments()
    setup_logging(verbose=args.verbose)
    
    # Run validation
    exit_code = asyncio.run(run_validation(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
