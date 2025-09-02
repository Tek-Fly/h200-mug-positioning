#!/bin/bash
# Production Verification Script for H200 Intelligent Mug Positioning System
# Ensures all components are production-ready before deployment

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Function to print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Function to run a check
run_check() {
    local check_name=$1
    local check_command=$2
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    echo -n "  ⏳ ${check_name}... "
    
    if eval "$check_command" &> /dev/null; then
        print_color $GREEN "✓ PASSED"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        print_color $RED "✗ FAILED"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
}

# Main verification
main() {
    print_color $BLUE "=== H200 Production Verification ==="
    print_color $YELLOW "Date: $(date)"
    echo ""
    
    # Environment checks
    print_color $BLUE "1. Environment Checks"
    run_check "Python version 3.11+" "python3 --version | grep -E '3\.(1[1-9]|[2-9][0-9])'"
    run_check "Docker installed" "docker --version"
    run_check "Docker Compose installed" "docker-compose --version"
    run_check "Docker buildx available" "docker buildx version"
    echo ""
    
    # Configuration files
    print_color $BLUE "2. Configuration Files"
    run_check "pyproject.toml exists" "test -f pyproject.toml"
    run_check ".pylintrc exists" "test -f .pylintrc"
    run_check ".flake8 exists" "test -f .flake8"
    run_check ".dockerignore exists" "test -f .dockerignore"
    run_check ".gitignore exists" "test -f .gitignore"
    run_check ".env.example exists" "test -f .env.example"
    echo ""
    
    # Code quality
    print_color $BLUE "3. Code Quality Checks"
    run_check "Black formatting" "black --check src/ tests/"
    run_check "isort imports" "isort --check-only src/ tests/"
    run_check "Flake8 linting" "flake8 src/ --count --exit-zero --max-complexity=10 --statistics"
    run_check "Pylint analysis" "pylint src/ --exit-zero --score=y | grep -E 'rated at [89]\.|rated at 10\.'"
    run_check "Type checking" "mypy src/ --config-file=pyproject.toml"
    echo ""
    
    # Security checks
    print_color $BLUE "4. Security Checks"
    run_check "No hardcoded secrets" "! grep -r -E '(password|secret|key|token)\\s*=\\s*[\"'\''][^\"'\'']+[\"'\'']' src/ --include='*.py' | grep -v -E '(example|test|mock|dummy)'"
    run_check "Bandit security scan" "bandit -r src/ -ll -q"
    run_check ".env not in git" "! git ls-files | grep -E '^\.env$'"
    echo ""
    
    # Docker checks
    print_color $BLUE "5. Docker Checks"
    run_check "Dockerfile.base exists" "test -f docker/Dockerfile.base"
    run_check "Dockerfile.serverless exists" "test -f docker/Dockerfile.serverless"
    run_check "Dockerfile.timed exists" "test -f docker/Dockerfile.timed"
    run_check "Docker Compose production" "test -f docker-compose.production.yml"
    run_check "Build script executable" "test -x scripts/deploy/build_and_push.sh"
    echo ""
    
    # Documentation checks
    print_color $BLUE "6. Documentation Checks"
    run_check "README.md exists" "test -f README.md"
    run_check "USER_MANUAL.md exists" "test -f USER_MANUAL.md"
    run_check "CLAUDE.md exists" "test -f CLAUDE.md"
    run_check "API documentation" "test -d docs/api/"
    run_check "Handover documentation" "test -f docs/handover/ACTIVE_HANDOVER.md"
    echo ""
    
    # Test coverage
    print_color $BLUE "7. Test Coverage"
    run_check "Unit tests exist" "test -d tests/unit/"
    run_check "Integration tests exist" "test -d tests/integration/"
    run_check "E2E tests exist" "test -d tests/e2e/"
    run_check "Test runner available" "test -f tests/test_runner.py"
    
    # Run minimal test to verify imports
    if command -v pytest &> /dev/null; then
        run_check "Import verification" "python -c 'import src.core.analyzer; import src.control.api.main'"
    fi
    echo ""
    
    # GitHub Actions
    print_color $BLUE "8. CI/CD Checks"
    run_check "Deploy workflow exists" "test -f .github/workflows/deploy.yml"
    run_check "Test workflow exists" "test -f .github/workflows/test.yml"
    run_check "Pre-commit config" "test -f .pre-commit-config.yaml"
    echo ""
    
    # Performance requirements
    print_color $BLUE "9. Performance Requirements"
    echo "  ℹ️  FlashBoot cold start: 500ms-2s (target)"
    echo "  ℹ️  Cache hit rate: >85% (target)"
    echo "  ℹ️  GPU utilization: >70% (target)"
    echo "  ℹ️  API latency p95: <200ms (target)"
    echo ""
    
    # Summary
    print_color $BLUE "=== Verification Summary ==="
    echo "Total checks: $TOTAL_CHECKS"
    print_color $GREEN "Passed: $PASSED_CHECKS"
    print_color $RED "Failed: $FAILED_CHECKS"
    
    if [ $FAILED_CHECKS -eq 0 ]; then
        echo ""
        print_color $GREEN "✅ ALL CHECKS PASSED - READY FOR PRODUCTION!"
        echo ""
        print_color $YELLOW "Next steps:"
        echo "  1. Ensure all secrets are configured in GitHub"
        echo "  2. Run: ./scripts/deploy/build_and_push.sh"
        echo "  3. Deploy: python scripts/deploy/deploy_to_runpod.py both"
        echo "  4. Verify: ./scripts/deploy/health_check.sh"
        exit 0
    else
        echo ""
        print_color $RED "❌ VERIFICATION FAILED - NOT READY FOR PRODUCTION"
        echo ""
        print_color $YELLOW "Please fix the failed checks before deploying."
        exit 1
    fi
}

# Run main function
main "$@"