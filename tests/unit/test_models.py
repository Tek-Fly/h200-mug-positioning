"""Unit tests for model components."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image

from src.core.models.manager import ModelManager
from src.core.models.yolo import YOLOv8Model
from src.core.models.clip import CLIPVisionModel
from src.core.models.base import BaseModel
from tests.base import ModelTestBase


@pytest.mark.unit
class TestModelManager(ModelTestBase):
    """Test cases for ModelManager."""
    
    def setup_method(self):
        """Setup test model manager."""
        super().setup_method()
        self.manager = ModelManager()
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test model manager initialization."""
        with patch.object(self.manager, '_load_default_models') as mock_load:
            await self.manager.initialize()
            mock_load.assert_called_once()
            assert self.manager.initialized
    
    @pytest.mark.asyncio
    async def test_register_model(self):
        """Test model registration."""
        mock_model = Mock(spec=BaseModel)
        mock_model.model_name = "test_model"
        mock_model.model_type = "test"
        
        self.manager.register_model("test_key", mock_model)
        
        assert "test_key" in self.manager._models
        assert self.manager._models["test_key"] == mock_model
    
    @pytest.mark.asyncio
    async def test_get_model_exists(self):
        """Test getting existing model."""
        mock_model = Mock(spec=BaseModel)
        self.manager._models["test_key"] = mock_model
        
        result = self.manager.get_model("test_key")
        
        assert result == mock_model
    
    def test_get_model_not_exists(self):
        """Test getting non-existent model."""
        with pytest.raises(KeyError, match="Model 'nonexistent' not found"):
            self.manager.get_model("nonexistent")
    
    def test_is_loaded(self):
        """Test checking if model is loaded."""
        mock_model = Mock(spec=BaseModel)
        mock_model.is_loaded.return_value = True
        self.manager._models["test_key"] = mock_model
        
        assert self.manager.is_loaded("test_key")
        assert not self.manager.is_loaded("nonexistent")
    
    @pytest.mark.asyncio
    async def test_load_model_success(self):
        """Test successful model loading."""
        mock_model = AsyncMock(spec=BaseModel)
        mock_model.load_model.return_value = True
        self.manager._models["test_key"] = mock_model
        
        result = await self.manager.load_model("test_key")
        
        assert result is True
        mock_model.load_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_model_failure(self):
        """Test model loading failure."""
        mock_model = AsyncMock(spec=BaseModel)
        mock_model.load_model.side_effect = Exception("Load failed")
        self.manager._models["test_key"] = mock_model
        
        with pytest.raises(Exception, match="Load failed"):
            await self.manager.load_model("test_key")
    
    @pytest.mark.asyncio
    async def test_unload_model(self):
        """Test model unloading."""
        mock_model = AsyncMock(spec=BaseModel)
        self.manager._models["test_key"] = mock_model
        
        await self.manager.unload_model("test_key")
        
        mock_model.unload_model.assert_called_once()
    
    def test_get_registry_info(self):
        """Test getting registry information."""
        mock_model = Mock(spec=BaseModel)
        mock_model.model_name = "Test Model"
        mock_model.model_type = "test"
        mock_model.is_loaded.return_value = True
        mock_model.get_model_info.return_value = {
            "description": "Test model",
            "version": "1.0"
        }
        self.manager._models["test_key"] = mock_model
        
        info = self.manager.get_registry_info()
        
        assert "test_key" in info
        assert info["test_key"]["loaded"] is True
        assert info["test_key"]["description"] == "Test model"
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test manager cleanup."""
        mock_model1 = AsyncMock(spec=BaseModel)
        mock_model2 = AsyncMock(spec=BaseModel)
        self.manager._models = {
            "model1": mock_model1,
            "model2": mock_model2
        }
        
        await self.manager.cleanup()
        
        mock_model1.unload_model.assert_called_once()
        mock_model2.unload_model.assert_called_once()
        assert len(self.manager._models) == 0
        assert not self.manager.initialized


@pytest.mark.unit
class TestYOLOv8Model(ModelTestBase):
    """Test cases for YOLOv8Model."""
    
    def setup_method(self):
        """Setup YOLO model test."""
        super().setup_method()
        self.model = YOLOv8Model(model_size="n")
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test YOLO model initialization."""
        model = YOLOv8Model(model_size="s", device="cpu")
        
        assert model.model_size == "s"
        assert model.device == "cpu"
        assert model.model_name == "YOLOv8s Object Detection"
        assert not model.is_loaded()
    
    @pytest.mark.asyncio
    async def test_load_model_success(self):
        """Test successful YOLO model loading."""
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_model = Mock()
            mock_yolo_class.return_value = mock_model
            
            result = await self.model.load_model()
            
            assert result is True
            assert self.model.is_loaded()
            assert self.model.model == mock_model
            mock_yolo_class.assert_called_with("yolov8n.pt")
    
    @pytest.mark.asyncio
    async def test_load_model_failure(self):
        """Test YOLO model loading failure."""
        with patch('ultralytics.YOLO', side_effect=Exception("Load failed")):
            result = await self.model.load_model()
            
            assert result is False
            assert not self.model.is_loaded()
    
    @pytest.mark.asyncio
    async def test_predict_success(self, sample_image):
        """Test successful YOLO prediction."""
        # Setup loaded model
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.data = torch.tensor([
            [100, 100, 200, 200, 0.9, 47]  # Cup detection
        ])
        mock_result.boxes.conf = torch.tensor([0.9])
        mock_result.boxes.cls = torch.tensor([47])
        mock_result.boxes.xyxy = torch.tensor([[100, 100, 200, 200]])
        
        mock_model.return_value = [mock_result]
        self.model.model = mock_model
        self.model._loaded = True
        
        result = await self.model.predict(sample_image)
        
        assert len(result) == 1
        detection = result[0]
        assert detection["bbox"] == [100, 100, 200, 200]
        assert detection["confidence"] == 0.9
        assert detection["class_id"] == 47
    
    @pytest.mark.asyncio
    async def test_predict_not_loaded(self, sample_image):
        """Test prediction when model not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await self.model.predict(sample_image)
    
    @pytest.mark.asyncio
    async def test_predict_empty_results(self, sample_image):
        """Test prediction with no detections."""
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.data = torch.empty((0, 6))
        mock_result.boxes.conf = torch.empty(0)
        mock_result.boxes.cls = torch.empty(0, dtype=torch.int64)
        mock_result.boxes.xyxy = torch.empty((0, 4))
        
        mock_model.return_value = [mock_result]
        self.model.model = mock_model
        self.model._loaded = True
        
        result = await self.model.predict(sample_image)
        
        assert len(result) == 0
    
    def test_format_prediction(self):
        """Test prediction formatting."""
        bbox_tensor = torch.tensor([100.5, 100.5, 200.7, 200.7])
        conf_tensor = torch.tensor(0.95)
        cls_tensor = torch.tensor(47, dtype=torch.int64)
        
        result = self.model._format_prediction(bbox_tensor, conf_tensor, cls_tensor)
        
        assert result["bbox"] == [100, 100, 200, 200]  # Rounded
        assert result["confidence"] == 0.95
        assert result["class_id"] == 47
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.model.get_model_info()
        
        assert info["model_size"] == "n"
        assert info["device"] == "cuda"
        assert "model_path" in info
        assert info["loaded"] is False
    
    @pytest.mark.asyncio
    async def test_unload_model(self):
        """Test model unloading."""
        self.model._loaded = True
        self.model.model = Mock()
        
        await self.model.unload_model()
        
        assert not self.model.is_loaded()
        assert self.model.model is None


@pytest.mark.unit
class TestCLIPVisionModel(ModelTestBase):
    """Test cases for CLIPVisionModel."""
    
    def setup_method(self):
        """Setup CLIP model test."""
        super().setup_method()
        self.model = CLIPVisionModel(model_name="ViT-B/32")
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test CLIP model initialization."""
        model = CLIPVisionModel(model_name="ViT-L/14", device="cpu")
        
        assert model.model_name_clip == "ViT-L/14"
        assert model.device == "cpu"
        assert model.model_name == "CLIP Vision ViT-L/14"
        assert not model.is_loaded()
    
    @pytest.mark.asyncio
    async def test_load_model_success(self):
        """Test successful CLIP model loading."""
        with patch('clip.load') as mock_clip_load:
            mock_model = Mock()
            mock_preprocess = Mock()
            mock_clip_load.return_value = (mock_model, mock_preprocess)
            
            result = await self.model.load_model()
            
            assert result is True
            assert self.model.is_loaded()
            assert self.model.model == mock_model
            assert self.model.preprocess == mock_preprocess
    
    @pytest.mark.asyncio
    async def test_load_model_failure(self):
        """Test CLIP model loading failure."""
        with patch('clip.load', side_effect=Exception("Load failed")):
            result = await self.model.load_model()
            
            assert result is False
            assert not self.model.is_loaded()
    
    @pytest.mark.asyncio
    async def test_encode_image_success(self, sample_image):
        """Test successful image encoding."""
        # Setup loaded model
        mock_model = Mock()
        mock_preprocess = Mock()
        
        # Mock preprocessing
        preprocessed_tensor = torch.randn(1, 3, 224, 224)
        mock_preprocess.return_value = preprocessed_tensor
        
        # Mock model encoding
        mock_features = torch.randn(1, 512)
        mock_model.encode_image.return_value = mock_features
        
        self.model.model = mock_model
        self.model.preprocess = mock_preprocess
        self.model._loaded = True
        
        with patch('torch.no_grad'):
            result = await self.model.encode_image(sample_image)
        
        assert result.shape == (1, 512)
        mock_preprocess.assert_called_once()
        mock_model.encode_image.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_encode_image_not_loaded(self, sample_image):
        """Test encoding when model not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await self.model.encode_image(sample_image)
    
    @pytest.mark.asyncio
    async def test_encode_image_batch(self, sample_batch_images):
        """Test batch image encoding."""
        # Setup loaded model
        mock_model = Mock()
        mock_preprocess = Mock()
        
        # Mock preprocessing - return stacked tensors
        preprocessed_tensor = torch.randn(3, 3, 224, 224)
        mock_preprocess.side_effect = [
            torch.randn(3, 224, 224),
            torch.randn(3, 224, 224),
            torch.randn(3, 224, 224)
        ]
        
        # Mock model encoding
        mock_features = torch.randn(3, 512)
        mock_model.encode_image.return_value = mock_features
        
        self.model.model = mock_model
        self.model.preprocess = mock_preprocess
        self.model._loaded = True
        
        with patch('torch.no_grad'), \
             patch('torch.stack', return_value=preprocessed_tensor):
            result = await self.model.encode_image_batch(sample_batch_images)
        
        assert result.shape == (3, 512)
        assert mock_preprocess.call_count == 3
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.model.get_model_info()
        
        assert info["clip_model_name"] == "ViT-B/32"
        assert info["device"] == "cuda"
        assert info["loaded"] is False
        assert "embedding_size" in info
    
    @pytest.mark.asyncio
    async def test_unload_model(self):
        """Test model unloading."""
        self.model._loaded = True
        self.model.model = Mock()
        self.model.preprocess = Mock()
        
        await self.model.unload_model()
        
        assert not self.model.is_loaded()
        assert self.model.model is None
        assert self.model.preprocess is None


@pytest.mark.unit
class TestBaseModel:
    """Test cases for BaseModel abstract class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel()
    
    def test_subclass_must_implement_methods(self):
        """Test that subclasses must implement abstract methods."""
        class IncompleteModel(BaseModel):
            pass
        
        with pytest.raises(TypeError):
            IncompleteModel()


# Performance tests for models
@pytest.mark.unit
@pytest.mark.performance
class TestModelPerformance(ModelTestBase):
    """Performance tests for model components."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    async def test_yolo_inference_performance(self, sample_image, performance_thresholds):
        """Test YOLO inference performance."""
        model = YOLOv8Model(model_size="n")  # Smallest/fastest model
        
        with patch('ultralytics.YOLO') as mock_yolo_class:
            mock_yolo = Mock()
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.data = torch.empty((0, 6))
            mock_result.boxes.conf = torch.empty(0)
            mock_result.boxes.cls = torch.empty(0, dtype=torch.int64)
            mock_result.boxes.xyxy = torch.empty((0, 4))
            
            # Simulate fast inference
            def fast_predict(*args, **kwargs):
                return [mock_result]
            
            mock_yolo.side_effect = fast_predict
            mock_yolo_class.return_value = mock_yolo
            
            await model.load_model()
            
            start_time = pytest.importorskip("time").time()
            result = await model.predict(sample_image)
            end_time = pytest.importorskip("time").time()
            
            inference_time_ms = (end_time - start_time) * 1000
            
            # Should be very fast with mocking
            assert inference_time_ms <= 100  # 100ms threshold for mocked inference
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    async def test_clip_encoding_performance(self, sample_image):
        """Test CLIP encoding performance."""
        model = CLIPVisionModel(model_name="ViT-B/32")
        
        with patch('clip.load') as mock_load:
            mock_clip = Mock()
            mock_preprocess = Mock()
            mock_preprocess.return_value = torch.randn(3, 224, 224)
            mock_clip.encode_image.return_value = torch.randn(1, 512)
            mock_load.return_value = (mock_clip, mock_preprocess)
            
            await model.load_model()
            
            start_time = pytest.importorskip("time").time()
            with patch('torch.no_grad'):
                result = await model.encode_image(sample_image)
            end_time = pytest.importorskip("time").time()
            
            encoding_time_ms = (end_time - start_time) * 1000
            
            # Should be fast with mocking
            assert encoding_time_ms <= 50  # 50ms threshold for mocked encoding
            assert result.shape == (1, 512)
    
    @pytest.mark.asyncio
    async def test_model_manager_batch_loading(self):
        """Test model manager can handle multiple models efficiently."""
        manager = ModelManager()
        
        # Mock models
        mock_yolo = AsyncMock(spec=YOLOv8Model)
        mock_yolo.load_model.return_value = True
        mock_clip = AsyncMock(spec=CLIPVisionModel)  
        mock_clip.load_model.return_value = True
        
        manager.register_model("yolo", mock_yolo)
        manager.register_model("clip", mock_clip)
        
        start_time = pytest.importorskip("time").time()
        
        # Load models concurrently
        import asyncio
        await asyncio.gather(
            manager.load_model("yolo"),
            manager.load_model("clip")
        )
        
        end_time = pytest.importorskip("time").time()
        
        loading_time_ms = (end_time - start_time) * 1000
        
        # Concurrent loading should be efficient
        assert loading_time_ms <= 100  # Should be fast with mocking
        mock_yolo.load_model.assert_called_once()
        mock_clip.load_model.assert_called_once()