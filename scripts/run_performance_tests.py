#!/usr/bin/env python3
"""
Script to run Agent Scrivener performance tests.

This script provides a convenient way to run performance tests
with various configurations and options.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.run_performance_tests import main as run_tests


def main():
    """Main entry point for performance test script."""
    parser = argparse.ArgumentParser(
        description="Run Agent Scrivener performance tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_performance_tests.py                    # Run all tests
  python scripts/run_performance_tests.py --quick           # Run quick tests only
  python scripts/run_performance_tests.py --categories concurrent memory  # Run specific categories
  python scripts/run_performance_tests.py --output-dir results  # Custom output directory
        """
    )
    
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["concurrent", "memory", "scalability", "limits"],
        default=["concurrent", "memory", "scalability", "limits"],
        help="Test categories to run (default: all)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="test_results",
        help="Output directory for test results (default: test_results)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick performance tests (concurrent and memory only)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation in reports"
    )
    
    args = parser.parse_args()
    
    # Set environment variables for configuration
    if args.verbose:
        os.environ["PYTEST_VERBOSE"] = "1"
    
    if args.no_charts:
        os.environ["PERF_NO_CHARTS"] = "1"
    
    # Adjust categories for quick run
    if args.quick:
        args.categories = ["concurrent", "memory"]
        print("Running quick performance tests (concurrent and memory only)")
    
    print(f"Running performance tests for categories: {', '.join(args.categories)}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 60)
    
    # Run the tests
    try:
        # Temporarily modify sys.argv for the test runner
        original_argv = sys.argv[:]
        sys.argv = [
            "run_performance_tests.py",
            "--categories", *args.categories,
            "--output-dir", args.output_dir
        ]
        
        if args.quick:
            sys.argv.append("--quick")
        
        exit_code = run_tests()
        
        # Restore original argv
        sys.argv = original_argv
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\nPerformance tests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running performance tests: {e}")
        return 1


if __name__ == "__main__":
    exit(main())