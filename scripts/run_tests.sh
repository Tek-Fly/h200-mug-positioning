#!/bin/bash
set -e

# H200 Test Runner Script
# Usage: ./scripts/run_tests.sh [test_type] [options]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="fast"
COVERAGE=true
PARALLEL=false
VERBOSE=false
HTML_REPORTS=false
CLEAN_FIRST=false
DOCKER=false

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [TEST_TYPE] [OPTIONS]

TEST_TYPES:
    fast         Run fast tests (unit + integration, no external/GPU)
    unit         Run unit tests only
    integration  Run integration tests only
    e2e          Run end-to-end tests
    performance  Run performance tests
    gpu          Run GPU-dependent tests
    external     Run tests requiring external services
    all          Run all tests

OPTIONS:
    --no-coverage    Disable coverage reporting
    --parallel       Run tests in parallel
    --verbose        Verbose output
    --html           Generate HTML reports
    --clean          Clean cache and reports before running
    --docker         Run tests in Docker containers
    --help           Show this help message

EXAMPLES:
    $0 unit --parallel --verbose
    $0 all --html --coverage
    $0 fast --clean
    $0 --docker integration

Environment Variables:
    TESTING=true                     Enable testing mode
    SKIP_EXTERNAL_TESTS=true         Skip external service tests
    SKIP_GPU_TESTS=true             Skip GPU tests
    MONGODB_ATLAS_URI=...           MongoDB connection string
    REDIS_HOST=localhost            Redis host
    REDIS_PASSWORD=...              Redis password
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        fast|unit|integration|e2e|performance|gpu|external|all)
            TEST_TYPE="$1"
            shift
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --html)
            HTML_REPORTS=true
            shift
            ;;
        --clean)
            CLEAN_FIRST=true
            shift
            ;;
        --docker)
            DOCKER=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            print_color $RED "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Print configuration
print_color $BLUE "H200 Test Runner Configuration"
print_color $BLUE "=============================="
echo "Test Type: $TEST_TYPE"
echo "Coverage: $COVERAGE"
echo "Parallel: $PARALLEL"
echo "Verbose: $VERBOSE"
echo "HTML Reports: $HTML_REPORTS"
echo "Clean First: $CLEAN_FIRST"
echo "Docker: $DOCKER"
echo ""

# Clean if requested
if [ "$CLEAN_FIRST" = true ]; then
    print_color $YELLOW "Cleaning cache and reports..."
    make clean
fi

# Set environment variables
export TESTING=true
export PYTHONPATH="$(pwd)/src"

# Docker execution
if [ "$DOCKER" = true ]; then
    print_color $BLUE "Running tests in Docker..."
    
    # Build and run tests in Docker
    docker-compose -f docker-compose.test.yml build
    docker-compose -f docker-compose.test.yml up --abort-on-container-exit
    
    # Extract results
    print_color $YELLOW "Extracting test results..."
    docker-compose -f docker-compose.test.yml cp h200-test-runner:/app/reports ./reports || true
    
    # Cleanup
    docker-compose -f docker-compose.test.yml down -v
    
    print_color $GREEN "Docker tests completed!"
    exit 0
fi

# Check if virtual environment exists
if [ ! -d ".venv" ] && [ ! -n "$VIRTUAL_ENV" ]; then
    print_color $YELLOW "No virtual environment detected. Consider creating one:"
    print_color $YELLOW "  python -m venv .venv"
    print_color $YELLOW "  source .venv/bin/activate"
    print_color $YELLOW "  pip install -r requirements-dev.txt"
    echo ""
fi

# Check dependencies
print_color $BLUE "Checking dependencies..."
if ! python -c "import pytest" 2>/dev/null; then
    print_color $RED "pytest not found. Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Start test services if needed
if [ "$TEST_TYPE" = "integration" ] || [ "$TEST_TYPE" = "e2e" ] || [ "$TEST_TYPE" = "all" ]; then
    print_color $YELLOW "Starting test services (MongoDB, Redis)..."
    docker-compose -f docker-compose.test.yml up -d mongo-test redis-test
    
    # Wait for services to be ready
    print_color $YELLOW "Waiting for services to be ready..."
    sleep 10
    
    # Set connection strings for test services
    export MONGODB_ATLAS_URI="mongodb://localhost:27018/test_h200"
    export REDIS_HOST="localhost"
    export REDIS_PORT="6380"
    export REDIS_PASSWORD="test_password"
fi

# Build test runner command
CMD="python tests/test_runner.py $TEST_TYPE"

if [ "$COVERAGE" = false ]; then
    CMD="$CMD --no-coverage"
fi

if [ "$PARALLEL" = true ]; then
    CMD="$CMD --parallel"
fi

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

if [ "$HTML_REPORTS" = true ]; then
    CMD="$CMD --html"
fi

# Run tests
print_color $BLUE "Running tests..."
print_color $BLUE "Command: $CMD"
echo ""

if eval $CMD; then
    print_color $GREEN "✅ Tests passed!"
    
    # Display coverage summary if enabled
    if [ "$COVERAGE" = true ] && [ -f "coverage.xml" ]; then
        print_color $BLUE "Coverage Summary:"
        python -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('coverage.xml')
    root = tree.getroot()
    coverage = root.attrib.get('line-rate', 0)
    percentage = float(coverage) * 100
    print(f'Overall Coverage: {percentage:.1f}%')
except Exception as e:
    print('Could not parse coverage report')
"
    fi
    
    # Display report locations
    if [ "$HTML_REPORTS" = true ]; then
        echo ""
        print_color $BLUE "Reports generated:"
        [ -d "htmlcov" ] && echo "  📊 HTML Coverage: htmlcov/index.html"
        [ -d "reports" ] && echo "  📋 Test Reports: reports/"
        echo "  🐳 View reports: make test-reports-server (starts on port 8081)"
    fi
    
    TESTS_PASSED=true
else
    print_color $RED "❌ Tests failed!"
    TESTS_PASSED=false
fi

# Cleanup test services
if [ "$TEST_TYPE" = "integration" ] || [ "$TEST_TYPE" = "e2e" ] || [ "$TEST_TYPE" = "all" ]; then
    print_color $YELLOW "Stopping test services..."
    docker-compose -f docker-compose.test.yml down
fi

# Exit with appropriate code
if [ "$TESTS_PASSED" = true ]; then
    print_color $GREEN "🎉 All tests completed successfully!"
    exit 0
else
    print_color $RED "💥 Some tests failed. Check the output above."
    exit 1
fi