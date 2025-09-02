"""Test runner script with coverage reporting."""

import sys
import subprocess
import argparse
from pathlib import Path
import os


def run_tests(test_type="all", coverage=True, parallel=False, verbose=False, html_report=False):
    """Run tests with specified options."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add coverage options
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
        
        if html_report:
            cmd.append("--cov-report=html")
    
    # Add parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add test selection based on type
    test_paths = []
    markers = []
    
    if test_type == "unit":
        test_paths.append("tests/unit/")
        markers.append("unit")
    elif test_type == "integration":
        test_paths.append("tests/integration/")
        markers.append("integration")
    elif test_type == "e2e":
        test_paths.append("tests/e2e/")
        markers.append("e2e")
    elif test_type == "performance":
        test_paths.append("tests/performance/")
        markers.append("performance")
    elif test_type == "gpu":
        markers.append("gpu")
    elif test_type == "external":
        markers.append("external")
    elif test_type == "fast":
        # Run all tests except slow ones
        markers.append("not slow")
        markers.append("not external")
        markers.append("not gpu")
    elif test_type == "all":
        test_paths.append("tests/")
    else:
        print(f"Unknown test type: {test_type}")
        return 1
    
    # Add test paths
    if test_paths:
        cmd.extend(test_paths)
    
    # Add markers
    if markers:
        for marker in markers:
            cmd.extend(["-m", marker])
    
    # Add HTML report generation
    if html_report:
        cmd.extend(["--html=reports/test-report.html", "--self-contained-html"])
    
    # Set environment variables for testing
    env = os.environ.copy()
    env.update({
        "TESTING": "true",
        "LOG_LEVEL": "WARNING",
        "SKIP_EXTERNAL_TESTS": "true" if test_type != "external" else "false",
        "PYTHONPATH": str(Path.cwd() / "src")
    })
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Test type: {test_type}")
    print(f"Coverage: {'enabled' if coverage else 'disabled'}")
    print(f"Parallel: {'enabled' if parallel else 'disabled'}")
    print("-" * 50)
    
    # Create reports directory if it doesn't exist
    Path("reports").mkdir(exist_ok=True)
    
    # Run the tests
    result = subprocess.run(cmd, env=env)
    
    # Generate coverage badge if coverage was enabled
    if coverage and result.returncode == 0:
        try:
            generate_coverage_badge()
        except Exception as e:
            print(f"Failed to generate coverage badge: {e}")
    
    return result.returncode


def generate_coverage_badge():
    """Generate coverage badge from coverage data."""
    try:
        import coverage
        import json
        
        # Load coverage data
        cov = coverage.Coverage()
        cov.load()
        
        # Get total coverage percentage
        total = cov.report(show_missing=False, skip_covered=False)
        
        # Create simple badge data
        badge_data = {
            "schemaVersion": 1,
            "label": "coverage",
            "message": f"{total:.1f}%",
            "color": "green" if total >= 90 else "yellow" if total >= 75 else "red"
        }
        
        # Save badge data
        with open("reports/coverage-badge.json", "w") as f:
            json.dump(badge_data, f, indent=2)
        
        print(f"Coverage badge generated: {total:.1f}%")
        
    except ImportError:
        print("Coverage module not available for badge generation")
    except Exception as e:
        print(f"Error generating coverage badge: {e}")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="H200 Test Runner")
    
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "e2e", "performance", "gpu", "external", "fast"],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage and test reports"
    )
    
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run in CI mode (fast tests only, with coverage and reports)"
    )
    
    args = parser.parse_args()
    
    # CI mode configuration
    if args.ci:
        test_type = "fast"
        coverage = True
        parallel = True
        verbose = False
        html_report = True
    else:
        test_type = args.test_type
        coverage = not args.no_coverage
        parallel = args.parallel
        verbose = args.verbose
        html_report = args.html
    
    # Run tests
    exit_code = run_tests(
        test_type=test_type,
        coverage=coverage,
        parallel=parallel,
        verbose=verbose,
        html_report=html_report
    )
    
    # Print summary
    print("-" * 50)
    if exit_code == 0:
        print("✅ All tests passed!")
        if coverage and html_report:
            print("📊 Reports generated in reports/ directory")
    else:
        print("❌ Some tests failed!")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()