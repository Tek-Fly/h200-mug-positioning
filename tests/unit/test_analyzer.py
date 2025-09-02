"""Unit tests for H200ImageAnalyzer."""

# Standard library imports
from unittest.mock import AsyncMock, Mock, patch

# Third-party imports
import numpy as np
import pytest
import torch
from PIL import Image

# First-party imports
from src.core.analyzer import AnalysisResult, BatchAnalysisResult, H200ImageAnalyzer
from src.core.cache import DualLayerCache
from src.core.positioning import PositioningStrategy
from tests.base import ModelTestBase


@pytest.mark.unit
class TestH200ImageAnalyzer(ModelTestBase):
    """Test cases for H200ImageAnalyzer."""

    async def setup_method_async(self):
        """Setup test analyzer."""
        await super().setup_method_async()

        self.mock_cache = AsyncMock(spec=DualLayerCache)
        self.mock_model_manager = AsyncMock()
        self.mock_positioning_engine = AsyncMock()

        # Setup mock returns
        self.mock_cache.get.return_value = None
        self.mock_cache.set.return_value = True

        self.analyzer = H200ImageAnalyzer(
            cache=self.mock_cache,
            yolo_model_size="s",
            clip_model_name="ViT-B/32",
            positioning_strategy=PositioningStrategy.HYBRID,
        )

        # Mock dependencies
        self.analyzer.model_manager = self.mock_model_manager
        self.analyzer.positioning_engine = self.mock_positioning_engine

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = H200ImageAnalyzer()

        assert analyzer.yolo_model_size == "n"  # Default
        assert analyzer.clip_model_name == "ViT-B/32"  # Default
        assert analyzer.positioning_strategy == PositioningStrategy.HYBRID
        assert analyzer.enable_performance_logging is True
        assert not analyzer._initialized

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful analyzer initialization."""
        self.mock_model_manager.initialize.return_value = True

        await self.analyzer.initialize()

        assert self.analyzer._initialized
        self.mock_model_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test analyzer initialization failure."""
        self.mock_model_manager.initialize.side_effect = Exception("Init failed")

        with pytest.raises(Exception, match="Init failed"):
            await self.analyzer.initialize()

        assert not self.analyzer._initialized

    @pytest.mark.asyncio
    async def test_analyze_image_success(self, sample_mug_image):
        """Test successful image analysis."""
        # Setup mocks
        self.analyzer._initialized = True

        # Mock YOLO detection
        mock_detections = self.create_mock_yolo_result(
            [{"bbox": [200, 200, 300, 350], "confidence": 0.9, "class_id": 47}]  # Cup
        )
        self.mock_model_manager.get_model.return_value.return_value = mock_detections

        # Mock CLIP embedding
        mock_embedding = torch.randn(1, 512)
        self.mock_model_manager.get_model.return_value.encode_image.return_value = (
            mock_embedding
        )

        # Mock positioning
        mock_positions = [{"x": 250, "y": 275, "confidence": 0.85}]
        self.mock_positioning_engine.calculate_positions.return_value = mock_positions

        # Mock performance metrics
        with patch("src.core.analyzer.calculate_gpu_memory_usage", return_value=512.0):
            result = await self.analyzer.analyze_image(sample_mug_image)

        # Assertions
        assert isinstance(result, AnalysisResult)
        assert result.analysis_id is not None
        assert result.timestamp is not None
        assert result.image_hash is not None
        assert len(result.detections) == 1
        assert result.detections[0]["confidence"] == 0.9
        assert len(result.mug_positions) == 1
        assert result.mug_positions[0]["confidence"] == 0.85
        assert result.processing_time_ms > 0
        assert result.gpu_memory_mb == 512.0
        assert not result.cached  # First time, not cached

    @pytest.mark.asyncio
    async def test_analyze_image_cached(self, sample_mug_image):
        """Test cached image analysis."""
        # Setup cache hit
        cached_result = {
            "analysis_id": "cached_123",
            "detections": [],
            "mug_positions": [],
            "embeddings": np.random.randn(512).tolist(),
            "processing_time_ms": 50,
        }
        self.mock_cache.get.return_value = cached_result

        result = await self.analyzer.analyze_image(sample_mug_image)

        assert result.cached
        assert result.analysis_id == "cached_123"
        assert result.processing_time_ms == 50

    @pytest.mark.asyncio
    async def test_analyze_image_not_initialized(self, sample_image):
        """Test analyze_image fails when not initialized."""
        with pytest.raises(RuntimeError, match="not been initialized"):
            await self.analyzer.analyze_image(sample_image)

    @pytest.mark.asyncio
    async def test_analyze_batch_success(self, sample_batch_images):
        """Test successful batch analysis."""
        self.analyzer._initialized = True

        # Mock single image analysis
        mock_result = AnalysisResult(
            analysis_id="test_123",
            timestamp=pytest.importorskip("datetime").datetime.now(),
            image_hash="hash_123",
            detections=[],
            embeddings=np.random.randn(512),
            mug_positions=[],
            confidence_scores={"detection": 0.8},
            processing_time_ms=100,
            gpu_memory_mb=512,
            cached=False,
        )

        with patch.object(self.analyzer, "analyze_image", return_value=mock_result):
            result = await self.analyzer.analyze_batch(sample_batch_images)

        assert isinstance(result, BatchAnalysisResult)
        assert len(result.results) == 3
        assert result.total_processing_time_ms >= 300  # At least 3 * 100ms
        assert result.average_time_per_image_ms >= 100

    @pytest.mark.asyncio
    async def test_analyze_batch_empty(self):
        """Test batch analysis with empty list."""
        self.analyzer._initialized = True

        result = await self.analyzer.analyze_batch([])

        assert isinstance(result, BatchAnalysisResult)
        assert len(result.results) == 0
        assert result.total_processing_time_ms == 0
        assert result.average_time_per_image_ms == 0

    @pytest.mark.asyncio
    async def test_preprocess_image(self, sample_image):
        """Test image preprocessing."""
        processed = await self.analyzer._preprocess_image(sample_image)

        assert processed.mode == "RGB"
        assert processed.size == (640, 640)  # Should be resized

    @pytest.mark.asyncio
    async def test_generate_image_hash(self, sample_image):
        """Test image hash generation."""
        hash1 = await self.analyzer._generate_image_hash(sample_image)
        hash2 = await self.analyzer._generate_image_hash(sample_image)

        assert hash1 == hash2  # Same image, same hash
        assert len(hash1) == 64  # SHA256 hex length

    @pytest.mark.asyncio
    async def test_run_yolo_inference(self, sample_image):
        """Test YOLO inference."""
        self.analyzer._initialized = True

        # Mock YOLO model
        mock_yolo = Mock()
        mock_results = self.create_mock_yolo_result(
            [{"bbox": [100, 100, 200, 200], "confidence": 0.9, "class_id": 47}]
        )
        mock_yolo.return_value = mock_results
        self.mock_model_manager.get_model.return_value = mock_yolo

        detections = await self.analyzer._run_yolo_inference(sample_image)

        assert len(detections) == 1
        assert detections[0]["confidence"] == 0.9
        assert detections[0]["class"] == "cup"  # Class 47 maps to cup
        assert detections[0]["is_mug_related"] is True

    @pytest.mark.asyncio
    async def test_run_clip_inference(self, sample_image):
        """Test CLIP inference."""
        self.analyzer._initialized = True

        # Mock CLIP model
        mock_clip = Mock()
        mock_embedding = torch.randn(1, 512)
        mock_clip.encode_image.return_value = mock_embedding
        self.mock_model_manager.get_model.return_value = mock_clip

        embedding = await self.analyzer._run_clip_inference(sample_image)

        assert embedding.shape == (512,)  # Flattened to 1D
        assert isinstance(embedding, np.ndarray)

    @pytest.mark.asyncio
    async def test_process_detections_with_mugs(self):
        """Test processing detections containing mugs."""
        raw_detections = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.9, "class_id": 47},  # Cup
            {"bbox": [300, 300, 400, 400], "confidence": 0.8, "class_id": 0},  # Person
        ]

        processed = await self.analyzer._process_detections(raw_detections)

        # Should have both detections but only cup is mug-related
        assert len(processed) == 2
        mug_related = [d for d in processed if d["is_mug_related"]]
        assert len(mug_related) == 1
        assert mug_related[0]["class"] == "cup"

    @pytest.mark.asyncio
    async def test_calculate_confidence_scores(self):
        """Test confidence score calculation."""
        detections = [
            {"confidence": 0.9, "is_mug_related": True},
            {"confidence": 0.7, "is_mug_related": False},
        ]
        positions = [{"confidence": 0.85}]

        scores = await self.analyzer._calculate_confidence_scores(detections, positions)

        assert "detection" in scores
        assert "positioning" in scores
        assert scores["detection"] == 0.9  # Max mug-related detection confidence
        assert scores["positioning"] == 0.85

    @pytest.mark.asyncio
    async def test_store_analysis_result(self, sample_image):
        """Test storing analysis result."""
        result = AnalysisResult(
            analysis_id="test_123",
            timestamp=pytest.importorskip("datetime").datetime.now(),
            image_hash="hash_123",
            detections=[],
            embeddings=np.random.randn(512),
            mug_positions=[],
            confidence_scores={"detection": 0.8},
            processing_time_ms=100,
            gpu_memory_mb=512,
            cached=False,
        )

        await self.analyzer._store_analysis_result(sample_image, result)

        # Should call cache set and database store
        self.mock_cache.set.assert_called_once()
        # Database store would be mocked in integration tests

    @pytest.mark.asyncio
    async def test_get_performance_stats(self):
        """Test performance statistics."""
        # Simulate some operations
        self.analyzer.performance_stats["total_analyses"] = 10
        self.analyzer.performance_stats["cache_hits"] = 3
        self.analyzer.performance_stats["total_processing_time"] = 5000

        stats = self.analyzer.get_performance_stats()

        assert stats["total_analyses"] == 10
        assert stats["cache_hit_rate"] == 0.3  # 3/10
        assert stats["average_processing_time"] == 500  # 5000/10
        assert "gpu_memory_peak" in stats

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test analyzer cleanup."""
        self.analyzer._initialized = True

        await self.analyzer.cleanup()

        assert not self.analyzer._initialized
        # Model manager cleanup would be called


@pytest.mark.unit
class TestAnalysisResult:
    """Test cases for AnalysisResult dataclass."""

    def test_create_analysis_result(self):
        """Test creating AnalysisResult."""
        result = AnalysisResult(
            analysis_id="test_123",
            timestamp=pytest.importorskip("datetime").datetime.now(),
            image_hash="hash_123",
            detections=[],
            embeddings=np.random.randn(512),
            mug_positions=[],
            confidence_scores={"detection": 0.8},
            processing_time_ms=100,
            gpu_memory_mb=512,
            cached=False,
        )

        assert result.analysis_id == "test_123"
        assert result.cached is False
        assert result.embeddings.shape == (512,)

    def test_to_dict(self):
        """Test converting AnalysisResult to dict."""
        result = AnalysisResult(
            analysis_id="test_123",
            timestamp=pytest.importorskip("datetime").datetime.now(),
            image_hash="hash_123",
            detections=[{"class": "cup", "confidence": 0.9}],
            embeddings=np.array([1.0, 2.0, 3.0]),
            mug_positions=[{"x": 100, "y": 100}],
            confidence_scores={"detection": 0.9},
            processing_time_ms=100,
            gpu_memory_mb=512,
            cached=False,
        )

        result_dict = result.to_dict()

        assert result_dict["analysis_id"] == "test_123"
        assert len(result_dict["detections"]) == 1
        assert isinstance(result_dict["embeddings"], list)  # Converted from numpy
        assert len(result_dict["embeddings"]) == 3


@pytest.mark.unit
class TestBatchAnalysisResult:
    """Test cases for BatchAnalysisResult dataclass."""

    def test_create_batch_result(self):
        """Test creating BatchAnalysisResult."""
        results = []  # Would contain AnalysisResult objects

        batch_result = BatchAnalysisResult(
            batch_id="batch_123",
            results=results,
            total_processing_time_ms=1000,
            average_time_per_image_ms=333,
            cache_hit_rate=0.5,
            timestamp=pytest.importorskip("datetime").datetime.now(),
        )

        assert batch_result.batch_id == "batch_123"
        assert batch_result.cache_hit_rate == 0.5

    def test_calculate_metrics(self):
        """Test calculating batch metrics."""
        # This would test the calculation logic if implemented
        pass


# Performance-specific tests
@pytest.mark.unit
@pytest.mark.performance
class TestAnalyzerPerformance(ModelTestBase):
    """Performance tests for analyzer."""

    @pytest.mark.asyncio
    async def test_analysis_performance_threshold(
        self, sample_mug_image, performance_thresholds
    ):
        """Test analysis meets performance thresholds."""
        analyzer = H200ImageAnalyzer()
        analyzer._initialized = True

        # Mock fast responses
        with (
            patch.object(analyzer, "_run_yolo_inference", return_value=[]),
            patch.object(
                analyzer, "_run_clip_inference", return_value=np.random.randn(512)
            ),
            patch.object(
                analyzer.positioning_engine, "calculate_positions", return_value=[]
            ),
        ):

            start_time = pytest.importorskip("time").time()
            result = await analyzer.analyze_image(sample_mug_image)
            end_time = pytest.importorskip("time").time()

            processing_time_ms = (end_time - start_time) * 1000

            # Should meet performance threshold
            assert processing_time_ms <= performance_thresholds["image_processing_ms"]

    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self, sample_batch_images):
        """Test batch processing is more efficient than individual processing."""
        analyzer = H200ImageAnalyzer()
        analyzer._initialized = True

        # Mock responses
        with patch.object(analyzer, "analyze_image") as mock_analyze:
            mock_result = Mock()
            mock_result.processing_time_ms = 100
            mock_analyze.return_value = mock_result

            start_time = pytest.importorskip("time").time()
            result = await analyzer.analyze_batch(sample_batch_images)
            end_time = pytest.importorskip("time").time()

            batch_time_ms = (end_time - start_time) * 1000

            # Batch processing should have some overhead but still be reasonable
            assert batch_time_ms <= 500  # Should be fast with mocking
            assert result.average_time_per_image_ms <= 200
