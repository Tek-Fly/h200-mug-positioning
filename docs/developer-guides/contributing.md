# Contributing Guide

Welcome to the H200 Intelligent Mug Positioning System! This guide will help you contribute effectively to the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Process](#development-process)
3. [Code Standards](#code-standards)
4. [Testing Guidelines](#testing-guidelines)
5. [Documentation Requirements](#documentation-requirements)
6. [Pull Request Process](#pull-request-process)
7. [Release Process](#release-process)

## Getting Started

### Prerequisites

Before contributing, ensure you have:
- Read the [Development Setup Guide](./development.md)
- Set up your local development environment
- Reviewed the [Architecture Overview](./architecture.md)
- Understood the project's goals and scope

### Contributor Agreement

By contributing to this project, you agree that:
- Your contributions will be licensed under the MIT License
- You have the right to contribute the code
- You understand the project's open-source nature

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Design discussions and questions
- **Pull Requests**: Code contributions and reviews
- **Email**: security@tekfly.co.uk (for security issues)

## Development Process

### 1. Issue Creation and Assignment

#### Creating Issues

**Bug Reports:**
```markdown
**Bug Description**
Brief description of the issue

**Steps to Reproduce**
1. Step one
2. Step two
3. Expected vs. actual behavior

**Environment**
- OS: macOS 14.0
- Python: 3.11.5
- Docker: 24.0.6
- GPU: NVIDIA RTX 4090

**Logs**
```
Relevant log entries or error messages
```

**Additional Context**
Screenshots, configuration files, etc.
```

**Feature Requests:**
```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why this feature is needed and who would use it

**Proposed Solution**
How you envision this feature working

**Alternatives Considered**
Other approaches you've considered

**Implementation Notes**
Technical considerations or constraints
```

#### Issue Labels

Use appropriate labels:
- **Type**: `bug`, `feature`, `enhancement`, `documentation`
- **Priority**: `critical`, `high`, `medium`, `low`
- **Component**: `api`, `frontend`, `models`, `deployment`
- **Status**: `needs-triage`, `accepted`, `in-progress`, `blocked`

### 2. Branch Management

#### Branch Naming Convention

```bash
# Feature branches
feature/add-batch-analysis
feature/improve-gpu-memory-management

# Bug fixes
bugfix/fix-redis-connection-timeout
bugfix/resolve-model-loading-issue

# Documentation
docs/update-api-documentation
docs/add-deployment-guide

# Hotfixes (critical production issues)
hotfix/fix-memory-leak-in-analyzer
```

#### Branch Workflow

```bash
# 1. Create and switch to feature branch
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# 2. Make your changes
# ... development work ...

# 3. Regular commits with good messages
git add .
git commit -m "Add initial batch analysis endpoint

- Implement batch processing logic
- Add request/response models
- Include basic error handling
- Add unit tests for core functionality"

# 4. Keep branch updated
git fetch origin
git rebase origin/main

# 5. Push branch
git push -u origin feature/your-feature-name
```

## Code Standards

### Python Code Style

#### Formatting and Linting

**Automatic Formatting:**
```bash
# Format code
black src/ tests/
isort src/ tests/

# Remove unused imports
autoflake --recursive --in-place --remove-all-unused-imports src/ tests/
```

**Linting:**
```bash
# Run pylint
pylint src/ tests/ --rcfile=.pylintrc

# Type checking
mypy src/ --strict

# Security checks
bandit -r src/
```

#### Code Style Guidelines

**Function Design:**
```python
# Good: Clear, single responsibility
async def analyze_mug_positioning(
    detections: List[MugDetection],
    image_dimensions: Tuple[int, int],
    rules_context: Optional[str] = None
) -> PositioningResult:
    """
    Analyze mug positioning quality and provide feedback.
    
    Args:
        detections: List of detected mugs with bounding boxes
        image_dimensions: Image width and height in pixels
        rules_context: Optional context for rule evaluation
        
    Returns:
        Positioning analysis with confidence and suggestions
    """
    # Clear implementation
    pass

# Avoid: Unclear purpose, too many responsibilities
def do_stuff(data, options=None):
    # Does too many things
    pass
```

**Error Handling:**
```python
# Good: Specific exceptions with context
from src.utils.exceptions import ModelLoadingError, GPUMemoryError

async def load_model_safely(model_path: str) -> Model:
    """Load model with comprehensive error handling."""
    try:
        return await load_model(model_path)
    except FileNotFoundError as e:
        raise ModelLoadingError(f"Model file not found: {model_path}") from e
    except torch.cuda.OutOfMemoryError as e:
        raise GPUMemoryError(f"Insufficient GPU memory for model: {model_path}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading model {model_path}: {e}")
        raise ModelLoadingError(f"Failed to load model: {e}") from e

# Avoid: Generic exception handling
def load_model_bad(model_path):
    try:
        return load_model(model_path)
    except:
        return None  # Loses error information
```

**Async Best Practices:**
```python
# Good: Proper async patterns
async def process_image_batch(images: List[bytes]) -> List[AnalysisResult]:
    """Process multiple images concurrently."""
    # Use asyncio.gather for concurrent processing
    tasks = [analyze_single_image(img) for img in images]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions in results
    successful_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Failed to process image {i}: {result}")
        else:
            successful_results.append(result)
    
    return successful_results

# Avoid: Blocking operations in async functions
async def bad_batch_processing(images):
    results = []
    for img in images:  # Sequential processing
        result = await analyze_single_image(img)  # Blocking
        results.append(result)
    return results
```

### Frontend Code Standards

#### Vue.js Guidelines

**Component Structure:**
```vue
<!-- Good: Clear, well-organized component -->
<template>
  <div class="analysis-viewer">
    <div class="upload-section">
      <ImageUpload @upload="handleImageUpload" />
    </div>
    
    <div v-if="analysisResult" class="results-section">
      <AnalysisResults :result="analysisResult" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { AnalysisResult } from '@/types/api'

// Props
interface Props {
  readonly?: boolean
}
const props = withDefaults(defineProps<Props>(), {
  readonly: false
})

// State
const analysisResult = ref<AnalysisResult | null>(null)

// Computed
const isAnalyzing = computed(() => analysisResult.value?.status === 'processing')

// Methods
async function handleImageUpload(file: File) {
  try {
    analysisResult.value = await analyzeImage(file)
  } catch (error) {
    console.error('Analysis failed:', error)
    // Handle error appropriately
  }
}
</script>

<style scoped>
.analysis-viewer {
  @apply space-y-6;
}

.upload-section {
  @apply bg-white rounded-lg shadow p-6;
}

.results-section {
  @apply bg-gray-50 rounded-lg p-6;
}
</style>
```

**TypeScript Integration:**
```typescript
// types/api.ts - Strong typing for all API interactions
export interface AnalysisRequest {
  image: File
  includeF feedback?: boolean
  confidenceThreshold?: number
  rulesContext?: string
}

export interface AnalysisResponse {
  requestId: string
  timestamp: string
  processingTimeMs: number
  detections: MugDetection[]
  positioning: PositioningResult
  feedback?: string
  suggestions: string[]
  metadata: Record<string, unknown>
}

export interface MugDetection {
  id: string
  bbox: BoundingBox
  attributes: Record<string, unknown>
}

// api/client.ts - Centralized API client
class H200ApiClient {
  private baseUrl: string
  private token: string | null = null

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl
  }

  async analyzeImage(request: AnalysisRequest): Promise<AnalysisResponse> {
    const formData = new FormData()
    formData.append('image', request.image)
    
    if (request.includeFeedback !== undefined) {
      formData.append('include_feedback', String(request.includeFeedback))
    }
    
    const response = await fetch(`${this.baseUrl}/api/v1/analyze/with-feedback`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: formData
    })
    
    if (!response.ok) {
      throw new Error(`Analysis failed: ${response.statusText}`)
    }
    
    return await response.json()
  }

  private getAuthHeaders(): Record<string, string> {
    const headers: Record<string, string> = {}
    
    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`
    }
    
    return headers
  }
}
```

### Database and Storage

#### MongoDB Best Practices

```python
# Good: Proper indexing and error handling
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING, DESCENDING

class AnalysisRepository:
    def __init__(self, db: AsyncIOMotorClient):
        self.db = db
        self.collection = db.analysis_results
    
    async def initialize(self):
        """Create indexes for optimal performance."""
        indexes = [
            IndexModel([("user_id", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("request_id", ASCENDING)], unique=True),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("positioning.confidence", DESCENDING)]),
        ]
        
        await self.collection.create_indexes(indexes)
    
    async def store_analysis(self, result: AnalysisResult) -> str:
        """Store analysis result with proper error handling."""
        try:
            document = {
                "request_id": result.request_id,
                "user_id": result.user_id,
                "timestamp": result.timestamp,
                "detections": [d.model_dump() for d in result.detections],
                "positioning": result.positioning.model_dump(),
                "metadata": result.metadata,
                "ttl": datetime.utcnow() + timedelta(days=90)  # Auto-expire
            }
            
            insert_result = await self.collection.insert_one(document)
            return str(insert_result.inserted_id)
            
        except DuplicateKeyError:
            raise ValueError(f"Analysis {result.request_id} already exists")
        except Exception as e:
            logger.error(f"Failed to store analysis: {e}")
            raise DatabaseError(f"Storage failed: {e}") from e
```

## Testing Guidelines

### Test Categories

#### 1. Unit Tests
Test individual functions and classes in isolation:

```python
# tests/unit/test_positioning_engine.py
class TestPositioningEngine:
    @pytest.fixture
    def engine(self):
        return PositioningEngine()
    
    async def test_calculate_center_offset(self, engine):
        """Test center offset calculation."""
        detection = MugDetection(
            bbox=BoundingBox(
                top_left=Point2D(x=100, y=100),
                bottom_right=Point2D(x=200, y=200)
            )
        )
        image_size = (400, 400)
        
        result = await engine.calculate_center_offset(detection, image_size)
        
        assert result.x == -50  # 150 (mug center) - 200 (image center)
        assert result.y == -50
```

#### 2. Integration Tests
Test component interactions:

```python
# tests/integration/test_analysis_pipeline.py
class TestAnalysisPipeline:
    async def test_full_analysis_pipeline(
        self, client, sample_image, mock_database
    ):
        """Test complete analysis pipeline integration."""
        # Upload image
        response = await client.post(
            "/api/v1/analyze/with-feedback",
            files={"image": sample_image},
            data={"confidence_threshold": 0.8}
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify database storage
        stored_result = await mock_database.analysis_results.find_one(
            {"request_id": result["request_id"]}
        )
        assert stored_result is not None
        
        # Verify cache storage
        cached_result = await redis_client.get(f"analysis:{result['request_id']}")
        assert cached_result is not None
```

#### 3. End-to-End Tests
Test complete user workflows:

```python
# tests/e2e/test_complete_workflow.py
class TestCompleteWorkflow:
    async def test_user_analysis_workflow(self, browser):
        """Test complete user workflow from login to analysis."""
        page = await browser.new_page()
        
        # Login
        await page.goto("http://localhost:3000/login")
        await page.fill("#username", "test@example.com")
        await page.fill("#password", "testpassword")
        await page.click("#login-button")
        
        # Navigate to analysis
        await page.click("nav a[href='/analysis']")
        
        # Upload image
        await page.set_input_files("#image-upload", "tests/fixtures/sample_mug.jpg")
        
        # Configure analysis
        await page.fill("#confidence-threshold", "0.8")
        await page.check("#include-feedback")
        
        # Run analysis
        await page.click("#analyze-button")
        
        # Wait for results
        await page.wait_for_selector(".analysis-results", timeout=10000)
        
        # Verify results displayed
        detections = await page.query_selector_all(".detection-item")
        assert len(detections) > 0
        
        positioning = await page.query_selector(".positioning-result")
        assert positioning is not None
```

### Testing Best Practices

#### Test Data Management

```python
# tests/fixtures/data_factory.py
from factory import Factory, Faker, SubFactory
from factory.fuzzy import FuzzyChoice, FuzzyFloat

class MugDetectionFactory(Factory):
    class Meta:
        model = MugDetection
    
    id = Faker('uuid4')
    bbox = SubFactory('tests.fixtures.BoundingBoxFactory')
    confidence = FuzzyFloat(0.7, 1.0)
    attributes = {
        'color': FuzzyChoice(['white', 'black', 'blue', 'red']),
        'size': FuzzyChoice(['small', 'medium', 'large'])
    }

class AnalysisResultFactory(Factory):
    class Meta:
        model = AnalysisResult
    
    request_id = Faker('uuid4')
    timestamp = Faker('date_time')
    processing_time_ms = FuzzyFloat(100, 1000)
    detections = SubFactory(MugDetectionFactory.create_batch, size=2)

# Usage in tests
def test_analysis_with_multiple_mugs():
    analysis = AnalysisResultFactory(
        detections=MugDetectionFactory.create_batch(size=3)
    )
    # Test with generated data
```

#### Mock Strategies

```python
# tests/mocks/external_services.py
from unittest.mock import AsyncMock, MagicMock

class MockModelManager:
    """Mock model manager for testing."""
    
    def __init__(self):
        self.is_initialized = True
        self.models = {}
    
    async def analyze_image(self, image_data: bytes, **kwargs):
        """Mock image analysis."""
        return {
            'detections': [
                {
                    'bbox': [100, 100, 200, 200],
                    'confidence': 0.95,
                    'class': 'mug'
                }
            ],
            'processing_time_ms': 250,
            'model_version': 'mock-v1.0'
        }

class MockRunPodClient:
    """Mock RunPod client for testing deployment."""
    
    def __init__(self):
        self.deployed_pods = {}
    
    async def create_serverless_endpoint(self, config):
        """Mock serverless endpoint creation."""
        endpoint_id = f"mock-{len(self.deployed_pods)}"
        self.deployed_pods[endpoint_id] = {
            'id': endpoint_id,
            'status': 'running',
            'url': f'https://mock-{endpoint_id}.runpod.ai'
        }
        return self.deployed_pods[endpoint_id]
```

## Code Standards

### Documentation Standards

#### Docstring Requirements

**Function Documentation:**
```python
def calculate_positioning_score(
    detections: List[MugDetection],
    rules: List[Rule],
    image_dimensions: Tuple[int, int]
) -> PositioningScore:
    """
    Calculate overall positioning quality score.
    
    Evaluates mug positioning against configured rules and calculates
    a normalized score from 0.0 (poor) to 1.0 (perfect).
    
    Args:
        detections: List of detected mugs with bounding boxes and attributes
        rules: Active positioning rules to evaluate against
        image_dimensions: Image width and height in pixels for normalization
        
    Returns:
        PositioningScore containing:
            - overall_score: Normalized score 0.0-1.0
            - component_scores: Individual rule scores
            - violations: List of rule violations
            - suggestions: Improvement recommendations
            
    Raises:
        ValueError: If detections list is empty
        RuleEvaluationError: If rule evaluation fails
        
    Examples:
        Basic usage:
        >>> score = calculate_positioning_score(detections, rules, (1920, 1080))
        >>> print(f"Score: {score.overall_score:.2f}")
        
        With error handling:
        >>> try:
        ...     score = calculate_positioning_score([], rules, (1920, 1080))
        >>> except ValueError as e:
        ...     print(f"Error: {e}")
        
    Note:
        This function uses GPU acceleration when available and falls back
        to CPU processing for compatibility.
    """
```

**Class Documentation:**
```python
class H200ImageAnalyzer:
    """
    Main image analyzer for mug detection and positioning analysis.
    
    This class orchestrates the complete image analysis pipeline from
    preprocessing through model inference to result formatting. It's
    designed for high-performance GPU processing with intelligent
    caching and memory management.
    
    Attributes:
        model_manager: Manages ML model lifecycle and GPU memory
        cache_manager: Handles result caching for performance
        positioning_engine: Calculates positioning quality scores
        
    Examples:
        Basic usage:
        >>> analyzer = H200ImageAnalyzer(model_manager)
        >>> await analyzer.initialize()
        >>> result = await analyzer.analyze_image(image_data)
        
        With custom configuration:
        >>> config = AnalysisConfig(confidence_threshold=0.9)
        >>> analyzer = H200ImageAnalyzer(model_manager, config)
        >>> result = await analyzer.analyze_image(image_data, config)
        
    Performance:
        - Cold start: 500ms-2s (first request)
        - Warm start: <100ms (subsequent requests)
        - GPU memory: 2-8GB depending on model configuration
        
    Thread Safety:
        This class is not thread-safe. Use separate instances for
        concurrent processing or implement appropriate locking.
    """
```

### Security Guidelines

#### Input Validation

```python
from pydantic import BaseModel, validator, Field
from typing import Literal

class AnalysisRequest(BaseModel):
    """Validated analysis request model."""
    
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold"
    )
    
    rules_context: Optional[str] = Field(
        default=None,
        max_length=100,
        regex="^[a-zA-Z0-9_-]+$",
        description="Rules context identifier"
    )
    
    @validator('confidence_threshold')
    def validate_confidence(cls, v):
        """Validate confidence threshold range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v

# File upload validation
async def validate_image_upload(file: UploadFile):
    """Validate uploaded image file."""
    # Check file type
    if not file.content_type.startswith('image/'):
        raise ValueError(f"Invalid file type: {file.content_type}")
    
    # Check file size (10MB limit)
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Reset position
    
    if size > 10 * 1024 * 1024:  # 10MB
        raise ValueError(f"File too large: {size} bytes (max: 10MB)")
    
    # Validate image content
    try:
        from PIL import Image
        image = Image.open(file.file)
        image.verify()
    except Exception as e:
        raise ValueError(f"Invalid image content: {e}")
```

#### Secret Management

```python
# src/utils/secrets.py
from google.cloud import secretmanager
import os
from typing import Optional

class SecretManager:
    """Secure secret management with Google Secret Manager."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()
        self.cache = {}  # Short-term cache for performance
    
    async def get_secret(self, secret_name: str) -> str:
        """Retrieve secret with caching."""
        # Check cache first (5-minute TTL)
        cached_value = self.cache.get(secret_name)
        if cached_value and cached_value['expires'] > datetime.utcnow():
            return cached_value['value']
        
        # Retrieve from Secret Manager
        try:
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
            response = self.client.access_secret_version(request={"name": name})
            secret_value = response.payload.data.decode("UTF-8")
            
            # Cache with TTL
            self.cache[secret_name] = {
                'value': secret_value,
                'expires': datetime.utcnow() + timedelta(minutes=5)
            }
            
            return secret_value
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            raise

# Never hardcode secrets
# Bad:
DATABASE_PASSWORD = "hardcoded_password"

# Good:
secret_manager = SecretManager(os.getenv("GOOGLE_CLOUD_PROJECT"))
DATABASE_PASSWORD = await secret_manager.get_secret("database_password")
```

## Pull Request Process

### 1. Pre-Pull Request Checklist

Before creating a pull request:

```bash
# 1. Ensure code quality
make lint          # No linting errors
make test          # All tests pass
make format        # Code properly formatted

# 2. Update documentation
# - Update relevant documentation files
# - Add docstrings to new functions/classes
# - Update API documentation if needed

# 3. Test your changes
# - Run full test suite
# - Test manually in browser
# - Verify performance hasn't regressed

# 4. Update dependencies if needed
pip-compile requirements.in    # Update requirements.txt
pip-compile requirements-dev.in  # Update requirements-dev.txt

# 5. Rebase on latest main
git fetch origin
git rebase origin/main
```

### 2. Pull Request Creation

**PR Title Format:**
```
type(scope): brief description

Examples:
feat(api): add batch analysis endpoint
fix(gpu): resolve memory leak in model loading
docs(deployment): update RunPod configuration guide
refactor(rules): improve rule evaluation performance
```

**PR Description Template:**
```markdown
## Summary
Brief description of changes and motivation.

## Changes
- [ ] Added batch analysis endpoint
- [ ] Updated API documentation
- [ ] Added integration tests
- [ ] Updated deployment scripts

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance validated

## Documentation
- [ ] API documentation updated
- [ ] User guide updated (if needed)
- [ ] Changelog updated

## Deployment Notes
Any special considerations for deployment:
- New environment variables required
- Database migrations needed
- Breaking changes (with migration guide)

## Screenshots (if UI changes)
[Include before/after screenshots]

## Checklist
- [ ] Code follows project standards
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact assessed
```

### 3. Code Review Process

#### Review Guidelines

**For Authors:**
- Keep PRs focused and reasonably sized (<500 lines)
- Provide clear descriptions and context
- Respond promptly to review comments
- Address all feedback before requesting re-review

**For Reviewers:**
- Review within 24 hours of request
- Focus on correctness, readability, and performance
- Provide constructive feedback with suggestions
- Approve when satisfied with changes

**Review Checklist:**
```markdown
## Code Quality
- [ ] Code follows project style guidelines
- [ ] Functions are well-named and documented
- [ ] Error handling is appropriate
- [ ] No obvious security vulnerabilities

## Functionality
- [ ] Changes work as described
- [ ] Edge cases are handled
- [ ] Performance is acceptable
- [ ] No regressions introduced

## Testing
- [ ] Adequate test coverage
- [ ] Tests are meaningful and correct
- [ ] CI/CD passes

## Documentation
- [ ] Code is self-documenting
- [ ] API changes documented
- [ ] User-facing changes documented
```

### 4. Continuous Integration

#### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    
    - name: Lint code
      run: |
        pylint src/ tests/
        mypy src/ --strict
        black --check src/ tests/
        isort --check-only src/ tests/
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
    
    - name: Build and push development image
      uses: docker/build-push-action@v5
      with:
        context: .
        file: docker/Dockerfile.development
        push: ${{ github.ref == 'refs/heads/main' }}
        tags: tekfly/h200:dev-latest
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

## Release Process

### Versioning Strategy

**Semantic Versioning (SemVer):**
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

**Version Examples:**
- `1.0.0`: Initial release
- `1.1.0`: Added batch analysis feature
- `1.1.1`: Fixed GPU memory leak
- `2.0.0`: Breaking API changes

### Release Workflow

#### 1. Prepare Release

```bash
# 1. Create release branch
git checkout main
git pull origin main
git checkout -b release/v1.1.0

# 2. Update version numbers
echo "1.1.0" > VERSION
python scripts/update_version.py 1.1.0

# 3. Update changelog
nano CHANGELOG.md

# 4. Update documentation
python scripts/generate_docs.py
```

#### 2. Release Checklist

```markdown
## Pre-Release Checklist
- [ ] All tests pass
- [ ] Performance benchmarks meet requirements
- [ ] Security scan completed
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version numbers bumped
- [ ] Migration scripts tested (if applicable)

## Release Checklist  
- [ ] Release branch created
- [ ] Release notes drafted
- [ ] Docker images built and tested
- [ ] Staging deployment validated
- [ ] Production deployment plan reviewed

## Post-Release Checklist
- [ ] Production deployment successful
- [ ] Health checks passing
- [ ] Monitoring alerts configured
- [ ] Performance metrics within targets
- [ ] User documentation updated
- [ ] Team notified of release
```

#### 3. Release Automation

```python
# scripts/release.py
import asyncio
import semver
from typing import Dict, Any

class ReleaseManager:
    def __init__(self, version: str):
        self.version = version
        self.git = GitManager()
        self.docker = DockerManager()
        self.docs = DocumentationManager()
    
    async def create_release(self) -> Dict[str, Any]:
        """Execute automated release process."""
        
        # 1. Validate version
        if not semver.VersionInfo.isvalid(self.version):
            raise ValueError(f"Invalid version: {self.version}")
        
        # 2. Run pre-release checks
        await self.run_pre_release_checks()
        
        # 3. Build and test Docker images
        images = await self.build_release_images()
        
        # 4. Create Git tag and release
        await self.create_git_release()
        
        # 5. Deploy to staging
        staging_result = await self.deploy_to_staging()
        
        # 6. Run release validation
        await self.validate_staging_deployment()
        
        # 7. Create GitHub release
        release_url = await self.create_github_release()
        
        return {
            "version": self.version,
            "status": "success",
            "images": images,
            "staging_url": staging_result["url"],
            "release_url": release_url
        }
```

This contributing guide establishes clear standards and processes for effective collaboration on the H200 System while maintaining code quality and project consistency.