"""Integration tests for analysis API endpoints."""

import pytest
import json
import io
from PIL import Image
from unittest.mock import patch, AsyncMock
from fastapi import status
from fastapi.testclient import TestClient

from tests.base import APITestBase, IntegrationTestBase


@pytest.mark.integration
class TestAnalysisAPI(APITestBase, IntegrationTestBase):
    """Test cases for analysis API endpoints."""
    
    async def setup_method_async(self):
        """Setup analysis API tests."""
        await super().setup_method_async()
        await self.mock_database_operations()
    
    def create_test_image_file(self, format="PNG"):
        """Create a test image file for upload."""
        image = Image.new('RGB', (640, 640), color='white')
        img_buffer = io.BytesIO()
        image.save(img_buffer, format=format)
        img_buffer.seek(0)
        return img_buffer
    
    def test_analyze_image_success(self, test_client, auth_headers, sample_analysis_result):
        """Test successful image analysis."""
        with patch('src.core.analyzer.H200ImageAnalyzer') as mock_analyzer_class:
            # Mock analyzer instance
            mock_analyzer = AsyncMock()
            mock_result = AsyncMock()
            mock_result.to_dict.return_value = sample_analysis_result
            mock_analyzer.analyze_image.return_value = mock_result
            mock_analyzer_class.return_value = mock_analyzer
            
            # Create test image file
            image_file = self.create_test_image_file()
            
            response = test_client.post(
                f"{self.base_url}/analyze/with-feedback",
                files={"image": ("test.png", image_file, "image/png")},
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            # Verify response structure
            required_fields = [
                "analysis_id", "detections", "mug_positions", 
                "confidence_scores", "processing_time_ms"
            ]
            self.assert_api_response_format(data, required_fields)
            
            # Verify specific values
            assert data["analysis_id"] == "test_analysis_123"
            assert len(data["detections"]) == 1
            assert data["detections"][0]["class"] == "cup"
            assert len(data["mug_positions"]) == 1
    
    def test_analyze_image_no_auth(self, test_client):
        """Test image analysis without authentication."""
        image_file = self.create_test_image_file()
        
        response = test_client.post(
            f"{self.base_url}/analyze/with-feedback",
            files={"image": ("test.png", image_file, "image/png")}
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_analyze_image_invalid_file(self, test_client, auth_headers):
        """Test analysis with invalid file."""
        # Create a text file instead of image
        text_file = io.BytesIO(b"This is not an image")
        
        response = test_client.post(
            f"{self.base_url}/analyze/with-feedback",
            files={"image": ("test.txt", text_file, "text/plain")},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        self.assert_error_response(data, "INVALID_IMAGE_FORMAT")
    
    def test_analyze_image_no_file(self, test_client, auth_headers):
        """Test analysis without file."""
        response = test_client.post(
            f"{self.base_url}/analyze/with-feedback",
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_analyze_image_large_file(self, test_client, auth_headers):
        """Test analysis with file too large."""
        # Create large image (simulate > 10MB)
        with patch('fastapi.UploadFile.size', 11 * 1024 * 1024):  # 11MB
            image_file = self.create_test_image_file()
            
            response = test_client.post(
                f"{self.base_url}/analyze/with-feedback",
                files={"image": ("large.png", image_file, "image/png")},
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    
    def test_analyze_batch_success(self, test_client, auth_headers, sample_analysis_result):
        """Test successful batch analysis."""
        with patch('src.core.analyzer.H200ImageAnalyzer') as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            
            # Mock batch result
            mock_batch_result = AsyncMock()
            mock_batch_result.batch_id = "batch_123"
            mock_batch_result.results = [AsyncMock(to_dict=lambda: sample_analysis_result)] * 2
            mock_batch_result.total_processing_time_ms = 500
            mock_batch_result.average_time_per_image_ms = 250
            mock_batch_result.cache_hit_rate = 0.5
            
            mock_analyzer.analyze_batch.return_value = mock_batch_result
            mock_analyzer_class.return_value = mock_analyzer
            
            # Create multiple test images
            files = [
                ("images", ("test1.png", self.create_test_image_file(), "image/png")),
                ("images", ("test2.png", self.create_test_image_file(), "image/png"))
            ]
            
            response = test_client.post(
                f"{self.base_url}/analyze/batch",
                files=files,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert data["batch_id"] == "batch_123"
            assert len(data["results"]) == 2
            assert data["total_processing_time_ms"] == 500
            assert data["cache_hit_rate"] == 0.5
    
    def test_analyze_batch_too_many_files(self, test_client, auth_headers):
        """Test batch analysis with too many files."""
        # Create more than allowed files (assume limit is 10)
        files = []
        for i in range(15):  # Exceed limit
            files.append(
                ("images", (f"test{i}.png", self.create_test_image_file(), "image/png"))
            )
        
        response = test_client.post(
            f"{self.base_url}/analyze/batch",
            files=files,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        self.assert_error_response(data, "TOO_MANY_FILES")
    
    def test_get_analysis_result_success(self, test_client, auth_headers, sample_analysis_result):
        """Test getting analysis result by ID."""
        analysis_id = "test_analysis_123"
        
        with patch('src.database.mongodb.get_mongodb_client') as mock_get_db:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_collection.find_one.return_value = sample_analysis_result
            mock_get_db.return_value = mock_db
            
            response = test_client.get(
                f"{self.base_url}/analyze/results/{analysis_id}",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["analysis_id"] == analysis_id
    
    def test_get_analysis_result_not_found(self, test_client, auth_headers):
        """Test getting non-existent analysis result."""
        analysis_id = "nonexistent_analysis"
        
        with patch('src.database.mongodb.get_mongodb_client') as mock_get_db:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_collection.find_one.return_value = None
            mock_get_db.return_value = mock_db
            
            response = test_client.get(
                f"{self.base_url}/analyze/results/{analysis_id}",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_list_analysis_history(self, test_client, auth_headers):
        """Test listing analysis history."""
        with patch('src.database.mongodb.get_mongodb_client') as mock_get_db:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_cursor = AsyncMock()
            
            # Mock query results
            history_items = [
                {
                    "_id": "analysis_1",
                    "timestamp": "2025-01-01T10:00:00Z",
                    "processing_time_ms": 250,
                    "cached": False
                },
                {
                    "_id": "analysis_2", 
                    "timestamp": "2025-01-01T11:00:00Z",
                    "processing_time_ms": 100,
                    "cached": True
                }
            ]
            
            mock_cursor.to_list.return_value = history_items
            mock_collection.find.return_value = mock_cursor
            mock_db.__getitem__.return_value = mock_collection
            mock_get_db.return_value = mock_db
            
            response = test_client.get(
                f"{self.base_url}/analyze/history",
                params={"limit": 10, "offset": 0},
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            
            assert "items" in data
            assert len(data["items"]) == 2
            assert data["items"][0]["analysis_id"] == "analysis_1"
            assert "pagination" in data
    
    def test_delete_analysis_result(self, test_client, auth_headers):
        """Test deleting analysis result."""
        analysis_id = "test_analysis_123"
        
        with patch('src.database.mongodb.get_mongodb_client') as mock_get_db:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_collection.delete_one.return_value = AsyncMock(deleted_count=1)
            mock_db.__getitem__.return_value = mock_collection
            mock_get_db.return_value = mock_db
            
            response = test_client.delete(
                f"{self.base_url}/analyze/results/{analysis_id}",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_204_NO_CONTENT
    
    def test_analyze_image_with_options(self, test_client, auth_headers, sample_analysis_result):
        """Test image analysis with custom options."""
        with patch('src.core.analyzer.H200ImageAnalyzer') as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_result = AsyncMock()
            mock_result.to_dict.return_value = sample_analysis_result
            mock_analyzer.analyze_image.return_value = mock_result
            mock_analyzer_class.return_value = mock_analyzer
            
            image_file = self.create_test_image_file()
            
            # Test with custom options
            form_data = {
                "confidence_threshold": "0.7",
                "enable_caching": "false",
                "positioning_strategy": "hybrid"
            }
            
            response = test_client.post(
                f"{self.base_url}/analyze/with-feedback",
                files={"image": ("test.png", image_file, "image/png")},
                data=form_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
    
    def test_reanalyze_existing_image(self, test_client, auth_headers, sample_analysis_result):
        """Test re-analyzing existing image with different parameters."""
        analysis_id = "existing_analysis_123"
        
        with patch('src.core.analyzer.H200ImageAnalyzer') as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_result = AsyncMock()
            mock_result.to_dict.return_value = sample_analysis_result
            mock_analyzer.analyze_image.return_value = mock_result
            mock_analyzer_class.return_value = mock_analyzer
            
            # Mock finding existing analysis
            with patch('src.database.mongodb.get_mongodb_client') as mock_get_db:
                mock_db = AsyncMock()
                mock_collection = AsyncMock()
                mock_collection.find_one.return_value = {
                    "_id": analysis_id,
                    "image_hash": "existing_hash",
                    "stored_image_path": "path/to/image.png"
                }
                mock_db.__getitem__.return_value = mock_collection
                mock_get_db.return_value = mock_db
                
                response = test_client.post(
                    f"{self.base_url}/analyze/reanalyze/{analysis_id}",
                    json={"confidence_threshold": 0.8},
                    headers=auth_headers
                )
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert "analysis_id" in data


@pytest.mark.integration
class TestAnalysisAPIPerformance(APITestBase):
    """Performance tests for analysis API."""
    
    def test_analyze_image_performance_threshold(self, test_client, auth_headers, performance_thresholds):
        """Test analysis API meets performance thresholds."""
        import time
        
        with patch('src.core.analyzer.H200ImageAnalyzer') as mock_analyzer_class:
            # Mock fast analysis
            mock_analyzer = AsyncMock()
            mock_result = AsyncMock()
            mock_result.to_dict.return_value = {
                "analysis_id": "perf_test_123",
                "processing_time_ms": 150,  # Under threshold
                "detections": [],
                "mug_positions": [],
                "confidence_scores": {},
                "cached": False
            }
            mock_analyzer.analyze_image.return_value = mock_result
            mock_analyzer_class.return_value = mock_analyzer
            
            image_file = self.create_test_image_file()
            
            start_time = time.time()
            response = test_client.post(
                f"{self.base_url}/analyze/with-feedback",
                files={"image": ("test.png", image_file, "image/png")},
                headers=auth_headers
            )
            end_time = time.time()
            
            # API should respond quickly
            api_latency_ms = (end_time - start_time) * 1000
            assert api_latency_ms <= performance_thresholds["api_latency_p95_ms"]
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["processing_time_ms"] <= performance_thresholds["image_processing_ms"]
    
    def test_batch_analysis_scaling(self, test_client, auth_headers):
        """Test batch analysis scales reasonably with number of images."""
        import time
        
        with patch('src.core.analyzer.H200ImageAnalyzer') as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            
            def mock_batch_analysis(images):
                # Simulate scaling processing time
                batch_time = len(images) * 50  # 50ms per image
                mock_batch_result = AsyncMock()
                mock_batch_result.batch_id = "batch_perf_123"
                mock_batch_result.results = [AsyncMock(to_dict=lambda: {}) for _ in images]
                mock_batch_result.total_processing_time_ms = batch_time
                mock_batch_result.average_time_per_image_ms = 50
                mock_batch_result.cache_hit_rate = 0.0
                return mock_batch_result
            
            mock_analyzer.analyze_batch.side_effect = mock_batch_analysis
            mock_analyzer_class.return_value = mock_analyzer
            
            # Test with different batch sizes
            for batch_size in [2, 5, 10]:
                files = []
                for i in range(batch_size):
                    files.append(
                        ("images", (f"test{i}.png", self.create_test_image_file(), "image/png"))
                    )
                
                start_time = time.time()
                response = test_client.post(
                    f"{self.base_url}/analyze/batch",
                    files=files,
                    headers=auth_headers
                )
                end_time = time.time()
                
                assert response.status_code == status.HTTP_200_OK
                
                # API latency should scale reasonably
                api_latency_ms = (end_time - start_time) * 1000
                expected_max_latency = batch_size * 100 + 200  # Base overhead
                assert api_latency_ms <= expected_max_latency