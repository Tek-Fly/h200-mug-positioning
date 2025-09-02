# H200 Intelligent Mug Positioning System - Test Makefile

.PHONY: help test test-unit test-integration test-e2e test-performance test-gpu test-fast test-all
.PHONY: coverage coverage-html coverage-xml lint format check-format mypy
.PHONY: clean clean-cache clean-coverage setup-dev install-dev
.PHONY: ci-test ci-lint ci-full docker-test

# Default target
help:
	@echo "H200 Test Suite Commands"
	@echo "========================"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test-fast        - Run fast tests (unit + integration, no external/GPU)"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-e2e         - Run end-to-end tests"
	@echo "  test-performance - Run performance tests"
	@echo "  test-gpu         - Run GPU-dependent tests"
	@echo "  test-all         - Run all tests"
	@echo "  test             - Alias for test-fast"
	@echo ""
	@echo "Coverage Commands:"
	@echo "  coverage         - Generate coverage report (terminal)"
	@echo "  coverage-html    - Generate HTML coverage report"
	@echo "  coverage-xml     - Generate XML coverage report"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint             - Run all linters (pylint, mypy, flake8)"
	@echo "  format           - Auto-format code with black"
	@echo "  check-format     - Check if code formatting is correct"
	@echo "  mypy             - Run type checking"
	@echo ""
	@echo "CI/CD Commands:"
	@echo "  ci-test          - Run CI test suite"
	@echo "  ci-lint          - Run CI linting"
	@echo "  ci-full          - Run complete CI pipeline"
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup-dev        - Setup development environment"
	@echo "  install-dev      - Install development dependencies"
	@echo ""
	@echo "Cleanup Commands:"
	@echo "  clean            - Clean all generated files"
	@echo "  clean-cache      - Clean Python cache files"
	@echo "  clean-coverage   - Clean coverage reports"

# Testing Commands
test: test-fast

test-fast:
	@echo "Running fast tests (unit + integration, no external/GPU)..."
	python tests/test_runner.py fast --parallel

test-unit:
	@echo "Running unit tests..."
	python tests/test_runner.py unit --parallel

test-integration:
	@echo "Running integration tests..."
	python tests/test_runner.py integration

test-e2e:
	@echo "Running end-to-end tests..."
	python tests/test_runner.py e2e

test-performance:
	@echo "Running performance tests..."
	python tests/test_runner.py performance

test-gpu:
	@echo "Running GPU-dependent tests..."
	python tests/test_runner.py gpu

test-external:
	@echo "Running tests requiring external services..."
	SKIP_EXTERNAL_TESTS=false python tests/test_runner.py external

test-all:
	@echo "Running all tests..."
	python tests/test_runner.py all --html

# Coverage Commands
coverage:
	@echo "Generating coverage report..."
	python -m pytest tests/ --cov=src --cov-report=term-missing

coverage-html:
	@echo "Generating HTML coverage report..."
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "HTML coverage report generated in htmlcov/"

coverage-xml:
	@echo "Generating XML coverage report..."
	python -m pytest tests/ --cov=src --cov-report=xml
	@echo "XML coverage report generated as coverage.xml"

# Code Quality Commands
lint: pylint mypy flake8

pylint:
	@echo "Running pylint..."
	python -m pylint src/ --rcfile=.pylintrc --exit-zero

mypy:
	@echo "Running mypy type checking..."
	python -m mypy src/ --config-file=pyproject.toml --show-error-codes

flake8:
	@echo "Running flake8..."
	python -m flake8 src/ --config=.flake8 --show-source

format:
	@echo "Formatting code with black..."
	python -m black src/ tests/ --line-length=88

check-format:
	@echo "Checking code formatting..."
	python -m black src/ tests/ --line-length=88 --check --diff

# CI/CD Commands
ci-test:
	@echo "Running CI test suite..."
	python tests/test_runner.py --ci

ci-lint:
	@echo "Running CI linting..."
	@$(MAKE) check-format
	@$(MAKE) pylint
	@$(MAKE) mypy
	@$(MAKE) flake8

ci-full: ci-lint ci-test
	@echo "CI pipeline completed successfully!"

# Docker Testing
docker-test:
	@echo "Running tests in Docker..."
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
	docker-compose -f docker-compose.test.yml down

# Setup Commands
setup-dev: install-dev
	@echo "Setting up development environment..."
	@echo "Creating reports directory..."
	@mkdir -p reports
	@mkdir -p htmlcov
	@echo "Development environment setup complete!"

install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements-dev.txt

# Cleanup Commands
clean: clean-cache clean-coverage
	@echo "Cleaning generated files..."
	@find . -name "*.log" -delete
	@find . -name "junit.xml" -delete
	@rm -rf reports/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@echo "Cleanup complete!"

clean-cache:
	@echo "Cleaning Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@find . -name "*.pyd" -delete

clean-coverage:
	@echo "Cleaning coverage reports..."
	@rm -rf htmlcov/
	@rm -f coverage.xml
	@rm -f coverage.json
	@rm -f .coverage*

# Performance and Load Testing
test-load:
	@echo "Running load tests..."
	python -m pytest tests/performance/ -m "not gpu" -v

benchmark:
	@echo "Running benchmarks..."
	python -m pytest tests/performance/ --benchmark-only

# Documentation Testing
test-docs:
	@echo "Testing documentation examples..."
	python -m doctest src/core/analyzer.py -v
	python -m doctest src/core/rules/engine.py -v

# Security Testing
test-security:
	@echo "Running security tests..."
	python -m pytest tests/ -m "security" -v

# Database Testing (requires services)
test-db:
	@echo "Running database tests..."
	python -m pytest tests/ -m "mongodb or redis" -v

# Model Testing (requires GPU)
test-models:
	@echo "Running model tests..."
	python -m pytest tests/unit/test_models.py tests/performance/test_gpu_operations.py -v

# Integration with external services
test-external-integration:
	@echo "Running external integration tests..."
	SKIP_EXTERNAL_TESTS=false python -m pytest tests/e2e/ -m "external" -v

# Monitoring test metrics
test-metrics:
	@echo "Collecting test metrics..."
	python tests/test_runner.py all --html --verbose > reports/test-metrics.log 2>&1
	@echo "Test metrics collected in reports/test-metrics.log"

# Quick smoke test
smoke-test:
	@echo "Running smoke tests..."
	python -m pytest tests/unit/test_analyzer.py::TestH200ImageAnalyzer::test_initialization -v
	python -m pytest tests/integration/test_api_analysis.py -k "test_analyze_image_success" -v