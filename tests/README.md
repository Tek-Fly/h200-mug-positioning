# H200 Intelligent Mug Positioning System - Test Suite

This directory contains a comprehensive test suite for the H200 Intelligent Mug Positioning System, designed to ensure code quality, reliability, and performance.

## 📁 Test Structure

```
tests/
├── conftest.py              # Pytest configuration and global fixtures
├── base.py                  # Base test classes and utilities
├── test_runner.py           # Custom test runner with coverage
├── README.md               # This file
├── unit/                   # Unit tests for individual components
│   ├── test_analyzer.py    # H200ImageAnalyzer tests
│   ├── test_models.py      # Model component tests
│   ├── test_rules_engine.py # Rules engine tests
│   └── test_cache.py       # Cache system tests
├── integration/            # Integration tests for API endpoints
│   ├── test_api_analysis.py # Analysis API tests
│   ├── test_api_rules.py   # Rules API tests
│   ├── test_api_dashboard.py # Dashboard API tests
│   └── test_api_servers.py # Server management API tests
├── e2e/                    # End-to-end workflow tests
│   └── test_complete_workflow.py # Complete user workflows
├── performance/            # Performance and GPU operation tests
│   └── test_gpu_operations.py # GPU performance tests
└── fixtures/               # Test fixtures and mocks
    ├── database_fixtures.py # MongoDB/Redis mocks
    └── external_service_fixtures.py # External service mocks
```

## 🏃‍♂️ Quick Start

### Run Fast Tests (Recommended for Development)
```bash
# Using make
make test

# Using test runner directly
python tests/test_runner.py fast --parallel

# Using shell script
./scripts/run_tests.sh fast --parallel --verbose
```

### Run All Tests
```bash
# With coverage and HTML reports
make test-all

# Using test runner
python tests/test_runner.py all --html

# Using shell script
./scripts/run_tests.sh all --html --coverage
```

## 🧪 Test Types

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Scope**: Core modules, models, analyzers, rule engines
- **Mock Level**: Heavy mocking of external dependencies
- **Speed**: Very fast (< 5 seconds total)
- **Run with**: `make test-unit` or `python tests/test_runner.py unit`

**Example:**
```python
@pytest.mark.unit
class TestH200ImageAnalyzer(ModelTestBase):
    async def test_analyze_image_success(self, sample_mug_image):
        # Test analyzer with mocked dependencies
        pass
```

### Integration Tests (`tests/integration/`)
- **Purpose**: Test API endpoints and service integration
- **Scope**: FastAPI routes, database operations, external services
- **Mock Level**: Database and external services mocked
- **Speed**: Medium (10-30 seconds)
- **Run with**: `make test-integration`

**Example:**
```python
@pytest.mark.integration
class TestAnalysisAPI(APITestBase):
    def test_analyze_image_success(self, test_client, auth_headers):
        # Test complete API endpoint
        pass
```

### End-to-End Tests (`tests/e2e/`)
- **Purpose**: Test complete user workflows
- **Scope**: Full system integration, user journeys
- **Mock Level**: Minimal mocking, realistic scenarios
- **Speed**: Slow (30-60 seconds)
- **Run with**: `make test-e2e`

**Example:**
```python
@pytest.mark.e2e
class TestCompleteImageAnalysisWorkflow(E2ETestBase):
    async def test_single_image_analysis_workflow(self):
        # Test complete workflow from upload to result
        pass
```

### Performance Tests (`tests/performance/`)
- **Purpose**: Test GPU operations and performance requirements
- **Scope**: Model inference, batch processing, memory usage
- **Mock Level**: Minimal, realistic performance simulation
- **Speed**: Variable (depends on hardware)
- **Run with**: `make test-performance`
- **Requirements**: GPU recommended (can run on CPU with mocks)

**Example:**
```python
@pytest.mark.performance
@pytest.mark.gpu
class TestGPUPerformance(PerformanceTestBase):
    @requires_gpu
    async def test_yolo_inference_performance(self):
        # Test GPU inference performance
        pass
```

## 🎯 Test Markers

Use pytest markers to run specific test categories:

```bash
# Run only unit tests
pytest -m "unit"

# Run tests that don't require GPU
pytest -m "not gpu"

# Run tests that don't require external services
pytest -m "not external"

# Run fast tests only
pytest -m "not slow and not external and not gpu"

# Run GPU tests only (requires GPU hardware)
pytest -m "gpu"

# Run external service tests (requires services)
pytest -m "external"
```

### Available Markers:
- `unit`: Unit tests
- `integration`: Integration tests
- `e2e`: End-to-end tests
- `performance`: Performance tests
- `slow`: Slow-running tests
- `gpu`: Tests requiring GPU access
- `external`: Tests requiring external services
- `mongodb`: Tests requiring MongoDB
- `redis`: Tests requiring Redis
- `runpod`: Tests requiring RunPod API

## 🔧 Configuration

### Environment Variables
```bash
# Core testing configuration
export TESTING=true
export LOG_LEVEL=WARNING
export PYTHONPATH="$(pwd)/src"

# Database configuration for tests
export MONGODB_ATLAS_URI="mongodb://localhost:27017/test_h200"
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export REDIS_PASSWORD="test_password"

# Service configuration
export SKIP_EXTERNAL_TESTS=true  # Skip external service tests
export SKIP_GPU_TESTS=true       # Skip GPU tests
```

### Test Services (Docker Compose)
```bash
# Start test databases
docker-compose -f docker-compose.test.yml up -d mongo-test redis-test

# Run tests with real databases
SKIP_EXTERNAL_TESTS=false python tests/test_runner.py integration

# Clean up
docker-compose -f docker-compose.test.yml down
```

## 📊 Coverage Reports

### Generate Coverage Reports
```bash
# Terminal coverage report
make coverage

# HTML coverage report
make coverage-html
# View at: htmlcov/index.html

# XML coverage report (for CI)
make coverage-xml
# Output: coverage.xml
```

### Coverage Targets
- **Overall Coverage**: >85%
- **Core Modules**: >90%
- **API Endpoints**: >80%
- **Critical Paths**: 100%

### Coverage Configuration
Coverage settings are in `.coveragerc`:
- **Source**: `src/` directory
- **Omit**: Test files, examples, migrations
- **Branch Coverage**: Enabled
- **Exclusions**: Abstract methods, debug code, main blocks

## 🚀 CI/CD Integration

### GitHub Actions
Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests
- Manual workflow dispatch

**Workflow Jobs:**
1. **Lint**: Code quality checks (black, pylint, mypy, flake8)
2. **Test**: Run test matrix (unit, integration, e2e)
3. **Coverage**: Generate and upload coverage reports
4. **Performance**: Run performance tests on main branch
5. **Docker**: Test Docker containerization
6. **Security**: Security scanning (bandit, safety)

### Local CI Simulation
```bash
# Run CI test suite locally
python tests/test_runner.py --ci

# Or using make
make ci-full
```

## 🐳 Docker Testing

### Run Tests in Docker
```bash
# Build and run tests in containers
docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit

# Or using script
./scripts/run_tests.sh --docker integration
```

### Docker Test Environment
- **Base Image**: Python 3.11 slim
- **Test Services**: MongoDB 7.0, Redis 7.2
- **Isolation**: Complete test environment isolation
- **Artifacts**: Test reports extracted automatically

## 🔍 Writing Tests

### Test Base Classes
Use appropriate base classes for different test types:

```python
from tests.base import (
    BaseTest,           # Basic test utilities
    AsyncBaseTest,      # Async test support
    ModelTestBase,      # Model testing utilities
    APITestBase,        # API testing utilities
    IntegrationTestBase,# Integration test helpers
    E2ETestBase,        # E2E workflow helpers
    PerformanceTestBase # Performance measurement
)
```

### Fixtures Usage
Common fixtures available in all tests:

```python
def test_example(sample_image, auth_headers, mock_mongodb, performance_thresholds):
    # sample_image: Test image for analysis
    # auth_headers: Authentication headers
    # mock_mongodb: Mocked MongoDB client
    # performance_thresholds: Performance expectations
    pass
```

### Mocking Guidelines
1. **Unit Tests**: Mock all external dependencies
2. **Integration Tests**: Mock external services, use real components
3. **E2E Tests**: Minimal mocking, realistic scenarios
4. **Performance Tests**: Mock only what's necessary for measurement

### Best Practices
1. **Test Naming**: Use descriptive names (`test_analyze_image_with_high_confidence`)
2. **Test Structure**: Arrange, Act, Assert pattern
3. **Assertions**: Use specific assertions with meaningful messages
4. **Cleanup**: Use fixtures for setup/teardown
5. **Documentation**: Document complex test scenarios

## 📈 Performance Testing

### GPU Performance Tests
- **Cold Start**: Model loading + first inference
- **Warm Start**: Subsequent inferences
- **Batch Processing**: Multiple images simultaneously
- **Memory Usage**: GPU memory allocation patterns
- **Concurrent Processing**: Multiple requests handling

### Performance Thresholds
```python
performance_thresholds = {
    "cold_start_ms": 2000,      # FlashBoot: 500ms-2s
    "warm_start_ms": 100,       # <100ms
    "image_processing_ms": 500, # <500ms for 1080p
    "cache_hit_rate": 0.85,     # >85%
    "gpu_utilization": 0.70,    # >70%
    "api_latency_p95_ms": 200   # <200ms p95
}
```

## 🐛 Debugging Tests

### Debug Failed Tests
```bash
# Run with verbose output
pytest -vv tests/unit/test_analyzer.py::TestH200ImageAnalyzer::test_analyze_image_success

# Run with pdb debugger
pytest --pdb tests/unit/test_analyzer.py -k "test_analyze_image_success"

# Run with coverage and keep failed
pytest --lf --tb=short tests/
```

### Common Issues
1. **Import Errors**: Check PYTHONPATH environment variable
2. **Async Test Failures**: Ensure pytest-asyncio is installed
3. **Mock Issues**: Verify mock configurations match actual interfaces
4. **Database Tests**: Ensure test databases are running
5. **GPU Tests**: Check CUDA availability and drivers

## 📋 Test Checklist

Before submitting code, ensure:

- [ ] All new code has corresponding unit tests
- [ ] API changes have integration tests
- [ ] Performance-critical code has performance tests
- [ ] All tests pass locally
- [ ] Coverage meets minimum thresholds
- [ ] No linting errors
- [ ] Type checking passes
- [ ] Documentation is updated

## 🤝 Contributing Tests

### Adding New Tests
1. Choose appropriate test type and location
2. Use existing fixtures and base classes
3. Follow naming conventions
4. Add appropriate markers
5. Update this README if adding new test categories

### Test Review Guidelines
1. Tests should be independent and idempotent
2. Mock external dependencies appropriately
3. Use descriptive assertions with clear error messages
4. Ensure tests are maintainable and readable
5. Performance tests should have reasonable thresholds

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Plugin](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Docker Compose Testing](https://docs.docker.com/compose/gettingstarted/)
- [GitHub Actions Testing](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)

---

For questions about the test suite, please refer to the project documentation or contact the development team.