# Development Setup Guide

Complete guide for setting up a local development environment for the H200 Intelligent Mug Positioning System.

## Development Environment Overview

The H200 System development environment provides:
- **Hot Reload**: Automatic code reloading for rapid iteration
- **GPU Support**: Local GPU development with fallback to CPU
- **Testing Framework**: Comprehensive test suite with CI/CD integration
- **Debugging Tools**: Advanced debugging and profiling capabilities
- **Documentation**: Live documentation generation and validation

## Prerequisites

### System Requirements

**Minimum Requirements:**
- Python 3.11+
- Docker Desktop 4.0+
- Git 2.30+
- 8GB RAM
- 50GB available disk space

**Recommended Requirements:**
- Python 3.11+
- Docker Desktop 4.20+
- Git 2.40+
- 16GB RAM
- 100GB SSD storage
- NVIDIA GPU (GTX 1060 or better)

**Optional but Helpful:**
- Visual Studio Code or PyCharm
- NVIDIA Docker runtime for GPU development
- Make utility for build automation

### Software Installation

#### macOS
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 git docker make
brew install --cask docker

# Install NVIDIA drivers (if you have an eGPU)
# Download from NVIDIA website
```

#### Linux (Ubuntu/Debian)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Docker (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-docker2
sudo systemctl restart docker
```

#### Windows
```powershell
# Install via Chocolatey (run as Administrator)
Set-ExecutionPolicy Bypass -Scope Process -Force
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Install dependencies
choco install python311 git docker-desktop make

# Install NVIDIA drivers
# Download from NVIDIA website
```

## Project Setup

### 1. Repository Setup

```bash
# Clone the repository
git clone https://github.com/tekfly/h200-mug-positioning.git
cd h200-mug-positioning

# Set up Git hooks
cp scripts/git-hooks/* .git/hooks/
chmod +x .git/hooks/*

# Create development branch
git checkout -b feature/your-feature-name
```

### 2. Python Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Verify Python version
python --version  # Should be 3.11+

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Dependencies Installation

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install main dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"
```

### 4. Environment Configuration

```bash
# Copy environment template
cp .env.example .env.development

# Edit configuration file
nano .env.development
```

**Development Environment Variables:**

```env
# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
SECRET_KEY=dev-secret-key-change-in-production

# Database (Local Docker containers)
MONGODB_URI=mongodb://admin:devpassword@localhost:27017/h200?authSource=admin
REDIS_URL=redis://:devpassword@localhost:6379/0

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
ENABLE_GPU=true
GPU_MEMORY_FRACTION=0.8

# Development Features
ENABLE_HOT_RELOAD=true
ENABLE_DEBUG_ENDPOINTS=true
ENABLE_PROFILING=true
MOCK_EXTERNAL_SERVICES=true

# Model Configuration (Development)
MODEL_CACHE_SIZE=1024  # MB
DOWNLOAD_MODELS_ON_STARTUP=true
USE_QUANTIZED_MODELS=true  # Faster inference

# Development Services
JUPYTER_ENABLE=true
JUPYTER_PORT=8888
TENSORBOARD_ENABLE=true
TENSORBOARD_PORT=6006

# Testing
TEST_DATABASE_URI=mongodb://admin:devpassword@localhost:27017/h200_test?authSource=admin
PYTEST_WORKERS=auto
COVERAGE_THRESHOLD=80
```

### 5. Local Services Setup

```bash
# Start development services
docker-compose -f docker-compose.dev.yml up -d

# Verify services are running
docker-compose -f docker-compose.dev.yml ps

# Check service health
curl http://localhost:27017  # MongoDB
redis-cli -h localhost -p 6379 -a devpassword ping  # Redis
```

**Development Docker Compose:**

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  # MongoDB for development
  mongodb-dev:
    image: mongo:7
    container_name: h200-mongodb-dev
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: devpassword
      MONGO_INITDB_DATABASE: h200
    volumes:
      - mongodb_data:/data/db
      - ./scripts/init-db.js:/docker-entrypoint-initdb.d/init-db.js
    command: mongod --auth --bind_ip_all
    restart: unless-stopped

  # Redis for development
  redis-dev:
    image: redis:7-alpine
    container_name: h200-redis-dev
    ports:
      - "6379:6379"
    command: redis-server --requirepass devpassword --maxmemory 1gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # MinIO for local S3-compatible storage (R2 alternative)
  minio-dev:
    image: minio/minio:latest
    container_name: h200-minio-dev
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: devuser
      MINIO_ROOT_PASSWORD: devpassword123
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"
    restart: unless-stopped

  # Mailhog for email testing
  mailhog-dev:
    image: mailhog/mailhog:latest
    container_name: h200-mailhog-dev
    ports:
      - "1025:1025"  # SMTP
      - "8025:8025"  # Web interface
    restart: unless-stopped

volumes:
  mongodb_data:
  redis_data:
  minio_data:
```

## Development Workflow

### 1. Starting Development

```bash
# Activate environment
source venv/bin/activate

# Start development servers
make dev-start

# Or manually:
uvicorn src.control.api.main:app --reload --host 0.0.0.0 --port 8000 &
cd dashboard && npm run dev &
```

**Development Makefile:**

```makefile
# Makefile for development automation
.PHONY: dev-start dev-stop test lint format setup clean

# Start development environment
dev-start:
	@echo "Starting H200 development environment..."
	docker-compose -f docker-compose.dev.yml up -d
	sleep 5
	@echo "Starting API server..."
	uvicorn src.control.api.main:app --reload --host 0.0.0.0 --port 8000 &
	@echo "Starting dashboard..."
	cd dashboard && npm run dev &
	@echo "Development environment ready!"
	@echo "API: http://localhost:8000"
	@echo "Dashboard: http://localhost:3000" 
	@echo "Docs: http://localhost:8000/api/docs"

# Stop development environment
dev-stop:
	@echo "Stopping development environment..."
	pkill -f uvicorn || true
	pkill -f "npm run dev" || true
	docker-compose -f docker-compose.dev.yml down

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Lint code
lint:
	pylint src/ tests/
	mypy src/ --strict
	black --check src/ tests/
	isort --check-only src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/
	autoflake --recursive --in-place --remove-all-unused-imports src/ tests/

# Initial setup
setup: 
	python -m pip install --upgrade pip
	pip install -r requirements-dev.txt
	pre-commit install
	docker-compose -f docker-compose.dev.yml up -d
	@echo "Setup complete! Run 'make dev-start' to begin development."

# Clean environment
clean:
	docker-compose -f docker-compose.dev.yml down -v
	docker system prune -f
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
```

### 2. Code Development

#### Project Structure

```
src/
├── control/          # API and control plane
│   ├── api/         # FastAPI endpoints
│   └── manager/     # System management
├── core/            # Core analysis logic
│   ├── models/      # ML model management
│   ├── rules/       # Rules engine
│   └── mcp/         # MCP protocol
├── database/        # Database connections
├── deployment/      # Deployment automation
├── integrations/    # External integrations
└── utils/          # Shared utilities

tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
├── performance/   # Performance tests
└── fixtures/      # Test data and mocks

dashboard/
├── src/           # Vue.js frontend
├── public/        # Static assets
└── dist/         # Built assets

docs/              # Documentation
scripts/           # Automation scripts
configs/           # Configuration files
```

#### Development Guidelines

**Code Style:**
```python
# Use type hints everywhere
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class AnalysisRequest(BaseModel):
    """Request model for image analysis."""
    
    image_data: bytes
    confidence_threshold: float = 0.7
    include_feedback: bool = True
    metadata: Optional[Dict[str, Any]] = None

async def analyze_image(
    request: AnalysisRequest,
    user_id: str,
    db: Database
) -> AnalysisResponse:
    """
    Analyze image and return positioning feedback.
    
    Args:
        request: Analysis request with image data
        user_id: ID of requesting user
        db: Database connection
        
    Returns:
        Analysis results with positioning data
        
    Raises:
        ValidationError: If image data is invalid
        ProcessingError: If analysis fails
    """
    # Implementation here
    pass
```

**Error Handling:**
```python
from src.utils.exceptions import H200Exception, ValidationError, ProcessingError

class ImageAnalysisError(H200Exception):
    """Raised when image analysis fails."""
    pass

async def safe_analysis(image_data: bytes) -> AnalysisResponse:
    """Analysis with comprehensive error handling."""
    try:
        # Validate input
        if not image_data:
            raise ValidationError("Image data cannot be empty")
        
        # Process image
        result = await process_image(image_data)
        
        # Validate result
        if not result.detections:
            logger.warning("No detections found in image")
        
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise
    except ProcessingError as e:
        logger.error(f"Processing error: {e}")
        raise ImageAnalysisError(f"Failed to analyze image: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise ImageAnalysisError(f"Analysis failed: {e}") from e
```

**Async Patterns:**
```python
import asyncio
from typing import List
from contextlib import asynccontextmanager

@asynccontextmanager
async def model_context(model_name: str):
    """Context manager for model lifecycle."""
    model = await load_model(model_name)
    try:
        yield model
    finally:
        await unload_model(model_name)

async def batch_process_images(
    image_list: List[bytes],
    batch_size: int = 8
) -> List[AnalysisResponse]:
    """Process multiple images efficiently."""
    results = []
    
    # Process in batches
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        
        # Process batch concurrently
        batch_tasks = [
            analyze_single_image(img_data) 
            for img_data in batch
        ]
        
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
        
        # Brief pause between batches to avoid overwhelming GPU
        await asyncio.sleep(0.1)
    
    return results
```

### 3. Testing Framework

#### Test Configuration

```python
# tests/conftest.py
import pytest
import asyncio
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from src.control.api.main import app
from src.database.get_db import get_mongodb, get_redis
from tests.fixtures.database_fixtures import MockDatabase, MockRedis

# Override dependencies for testing
app.dependency_overrides[get_mongodb] = lambda: MockDatabase()
app.dependency_overrides[get_redis] = lambda: MockRedis()

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def client():
    """Test client for API endpoints."""
    return TestClient(app)

@pytest.fixture
async def mock_model_manager():
    """Mock model manager for testing."""
    manager = AsyncMock()
    manager.analyze_image.return_value = {
        'detections': [{'bbox': [100, 100, 200, 200], 'confidence': 0.9}],
        'processing_time': 0.5
    }
    return manager

@pytest.fixture
def sample_image_data():
    """Sample image data for testing."""
    # Return minimal valid JPEG bytes
    return b'\xff\xd8\xff\xe0\x00\x10JFIF...'  # Truncated for example
```

#### Unit Tests

```python
# tests/unit/test_analyzer.py
import pytest
from unittest.mock import AsyncMock, patch

from src.core.analyzer import H200ImageAnalyzer
from src.core.models.manager import ModelManager

class TestH200ImageAnalyzer:
    """Test suite for image analyzer."""
    
    @pytest.fixture
    async def analyzer(self, mock_model_manager):
        """Create analyzer with mocked dependencies."""
        return H200ImageAnalyzer(model_manager=mock_model_manager)
    
    async def test_analyze_image_success(self, analyzer, sample_image_data):
        """Test successful image analysis."""
        result = await analyzer.analyze_image(sample_image_data)
        
        assert result is not None
        assert 'detections' in result
        assert len(result['detections']) > 0
        assert result['detections'][0]['confidence'] > 0.7
    
    async def test_analyze_image_invalid_data(self, analyzer):
        """Test analysis with invalid image data."""
        with pytest.raises(ValidationError):
            await analyzer.analyze_image(b'invalid image data')
    
    async def test_analyze_image_empty_data(self, analyzer):
        """Test analysis with empty data."""
        with pytest.raises(ValidationError):
            await analyzer.analyze_image(b'')
    
    @patch('src.core.analyzer.preprocess_image')
    async def test_preprocessing_called(
        self, mock_preprocess, analyzer, sample_image_data
    ):
        """Test that preprocessing is called."""
        mock_preprocess.return_value = sample_image_data
        
        await analyzer.analyze_image(sample_image_data)
        
        mock_preprocess.assert_called_once_with(sample_image_data)
```

#### Integration Tests

```python
# tests/integration/test_api_analysis.py
import pytest
from httpx import AsyncClient

from src.control.api.main import app

class TestAnalysisAPI:
    """Integration tests for analysis API."""
    
    @pytest.fixture
    async def client(self):
        """Async test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    async def test_analyze_image_endpoint(
        self, client: AsyncClient, sample_image_data
    ):
        """Test image analysis endpoint."""
        files = {"image": ("test.jpg", sample_image_data, "image/jpeg")}
        data = {"confidence_threshold": 0.8, "include_feedback": True}
        
        response = await client.post(
            "/api/v1/analyze/with-feedback",
            files=files,
            data=data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        result = response.json()
        
        assert "request_id" in result
        assert "detections" in result
        assert "positioning" in result
        assert result["processing_time_ms"] > 0
    
    async def test_analyze_image_unauthorized(
        self, client: AsyncClient, sample_image_data
    ):
        """Test unauthorized access to analysis endpoint."""
        files = {"image": ("test.jpg", sample_image_data, "image/jpeg")}
        
        response = await client.post(
            "/api/v1/analyze/with-feedback",
            files=files
        )
        
        assert response.status_code == 401
```

#### Performance Tests

```python
# tests/performance/test_gpu_operations.py
import pytest
import time
import statistics
from typing import List

from src.core.analyzer import H200ImageAnalyzer

class TestGPUPerformance:
    """Performance tests for GPU operations."""
    
    @pytest.mark.gpu
    @pytest.mark.slow
    async def test_cold_start_performance(
        self, analyzer: H200ImageAnalyzer, sample_image_data
    ):
        """Test cold start performance meets requirements."""
        start_time = time.time()
        
        # First request (cold start)
        result = await analyzer.analyze_image(sample_image_data)
        
        cold_start_time = (time.time() - start_time) * 1000  # ms
        
        # Should meet cold start requirement (500ms-2s)
        assert 500 <= cold_start_time <= 2000, f"Cold start too slow: {cold_start_time}ms"
        assert result is not None
    
    @pytest.mark.gpu
    async def test_warm_start_performance(
        self, analyzer: H200ImageAnalyzer, sample_image_data
    ):
        """Test warm start performance meets requirements."""
        # Warm up with one request
        await analyzer.analyze_image(sample_image_data)
        
        # Measure warm requests
        times = []
        for _ in range(10):
            start_time = time.time()
            await analyzer.analyze_image(sample_image_data)
            times.append((time.time() - start_time) * 1000)
        
        avg_time = statistics.mean(times)
        p95_time = sorted(times)[int(0.95 * len(times))]
        
        # Should meet warm start requirement (<100ms)
        assert avg_time < 100, f"Warm start too slow: {avg_time}ms average"
        assert p95_time < 150, f"P95 too slow: {p95_time}ms"
    
    @pytest.mark.gpu
    async def test_concurrent_requests(
        self, analyzer: H200ImageAnalyzer, sample_image_data
    ):
        """Test concurrent request handling."""
        import asyncio
        
        # Create multiple concurrent requests
        tasks = [
            analyzer.analyze_image(sample_image_data)
            for _ in range(5)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = (time.time() - start_time) * 1000
        
        # All requests should complete successfully
        assert all(r is not None for r in results)
        
        # Total time should not be much more than single request
        # (due to batching and GPU parallelism)
        assert total_time < 1000, f"Concurrent requests too slow: {total_time}ms"
```

### 4. Debugging and Profiling

#### Debug Configuration

```python
# src/utils/debug.py
import logging
import cProfile
import pstats
from functools import wraps
from typing import Any, Callable

def debug_performance(func: Callable) -> Callable:
    """Decorator to profile function performance."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        if not DEBUG:
            return await func(*args, **kwargs)
        
        # Profile async function
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = await func(*args, **kwargs)
        finally:
            profiler.disable()
            
            # Save profiling stats
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(10)  # Top 10 functions
        
        return result
    
    return wrapper

class DebugContext:
    """Context manager for detailed debugging."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        logger.debug(f"Starting {self.operation_name}")
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type:
            logger.error(
                f"{self.operation_name} failed after {duration:.3f}s: {exc_val}"
            )
        else:
            logger.debug(f"{self.operation_name} completed in {duration:.3f}s")

# Usage example
@debug_performance
async def analyze_image_debug(image_data: bytes):
    with DebugContext("Image preprocessing"):
        preprocessed = await preprocess_image(image_data)
    
    with DebugContext("GPU inference"):
        result = await run_gpu_inference(preprocessed)
    
    return result
```

#### GPU Debugging

```python
# src/utils/gpu_debug.py
import torch
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def gpu_memory_debug(operation_name: str):
    """Monitor GPU memory usage during operation."""
    if not torch.cuda.is_available():
        yield
        return
    
    # Get initial memory state
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    max_memory_before = torch.cuda.max_memory_allocated()
    
    logger.debug(
        f"{operation_name} - Initial GPU memory: "
        f"{initial_memory / 1024**2:.1f} MB"
    )
    
    try:
        yield
    finally:
        # Get final memory state
        final_memory = torch.cuda.memory_allocated()
        max_memory_after = torch.cuda.max_memory_allocated()
        
        memory_diff = final_memory - initial_memory
        peak_memory = max_memory_after - max_memory_before
        
        logger.debug(
            f"{operation_name} - Memory change: {memory_diff / 1024**2:.1f} MB, "
            f"Peak usage: {peak_memory / 1024**2:.1f} MB"
        )
        
        if memory_diff > 100 * 1024**2:  # > 100MB
            logger.warning(
                f"{operation_name} used {memory_diff / 1024**2:.1f} MB GPU memory"
            )

# Usage
async def analyze_with_memory_tracking(image_data: bytes):
    with gpu_memory_debug("Image Analysis"):
        result = await analyze_image(image_data)
    return result
```

### 5. Development Tools Integration

#### VS Code Configuration

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".mypy_cache": true,
        "venv/": true
    },
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "FastAPI Development",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "src.control.api.main:app",
                "--reload",
                "--host", "0.0.0.0",
                "--port", "8000"
            ],
            "env": {
                "ENVIRONMENT": "development",
                "DEBUG": "true"
            },
            "console": "integratedTerminal"
        },
        {
            "name": "Run Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "env": {
                "ENVIRONMENT": "test"
            },
            "console": "integratedTerminal"
        }
    ]
}
```

#### Jupyter Integration

```python
# notebooks/development_playground.ipynb
"""
Development playground for H200 System
"""

# Cell 1: Setup
import sys
sys.path.append('../src')

import asyncio
from src.core.analyzer import H200ImageAnalyzer
from src.core.models.manager import ModelManager

# Cell 2: Initialize components
async def setup_analyzer():
    model_manager = ModelManager()
    await model_manager.initialize()
    
    analyzer = H200ImageAnalyzer(model_manager)
    return analyzer

analyzer = await setup_analyzer()

# Cell 3: Test analysis
sample_image_path = "../tests/fixtures/sample_mug.jpg"
with open(sample_image_path, 'rb') as f:
    image_data = f.read()

result = await analyzer.analyze_image(image_data)
print(f"Detections: {len(result.detections)}")
```

### 6. Documentation Development

#### Live Documentation

```bash
# Start documentation server
cd docs
python -m mkdocs serve --dev-addr=0.0.0.0:8080

# Auto-generate API docs
python scripts/generate_api_docs.py

# Validate documentation
python scripts/validate_docs.py
```

#### Documentation Guidelines

**API Documentation:**
```python
# Use comprehensive docstrings
async def analyze_image(
    image: UploadFile,
    confidence_threshold: float = 0.7,
    context: Optional[str] = None
) -> AnalysisResponse:
    """
    Analyze uploaded image for mug positioning.
    
    This endpoint performs comprehensive mug detection and positioning analysis,
    providing feedback and suggestions for improvement based on configured rules.
    
    Args:
        image: Uploaded image file (JPEG, PNG, WebP)
        confidence_threshold: Minimum detection confidence (0.0-1.0)
        context: Optional context for rule application
    
    Returns:
        AnalysisResponse containing:
            - Detected mugs with bounding boxes and confidence
            - Positioning analysis with offset measurements
            - Rule violations and improvement suggestions
            - Processing time and metadata
    
    Raises:
        ValidationError: Invalid image format or parameters
        ProcessingError: Analysis processing failed
        
    Examples:
        Basic analysis:
        ```python
        with open('mug_image.jpg', 'rb') as f:
            response = await analyze_image(f)
        ```
        
        With custom threshold:
        ```python
        response = await analyze_image(
            image_file, 
            confidence_threshold=0.9
        )
        ```
    """
```

This development setup guide provides a comprehensive foundation for local development with all the tools, configurations, and best practices needed for productive work on the H200 System.