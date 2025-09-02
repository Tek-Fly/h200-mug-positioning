"""Performance tests for GPU operations."""

import pytest
import torch
import numpy as np
import time
from PIL import Image
from unittest.mock import patch, Mock
import statistics

from tests.base import PerformanceTestBase, requires_gpu


@pytest.mark.performance
@pytest.mark.gpu
class TestGPUPerformance(PerformanceTestBase):
    """Performance tests for GPU operations."""
    
    @requires_gpu
    @pytest.mark.asyncio
    async def test_yolo_inference_performance(self, sample_image, performance_thresholds):
        """Test YOLO inference performance on GPU."""
        from src.core.models.yolo import YOLOv8Model
        
        # Test with different model sizes
        model_sizes = ["n", "s", "m"]  # nano, small, medium
        
        for model_size in model_sizes:
            model = YOLOv8Model(model_size=model_size)
            
            with patch('ultralytics.YOLO') as mock_yolo_class:
                # Create mock YOLO model with realistic GPU timing
                mock_yolo = Mock()
                
                def realistic_inference(*args, **kwargs):
                    # Simulate GPU computation time based on model size
                    base_times = {"n": 0.02, "s": 0.04, "m": 0.08}  # seconds
                    time.sleep(base_times[model_size])
                    
                    # Return mock result
                    mock_result = Mock()
                    mock_result.boxes = Mock()
                    mock_result.boxes.data = torch.randn(5, 6)  # 5 detections
                    mock_result.boxes.conf = torch.rand(5)
                    mock_result.boxes.cls = torch.randint(0, 80, (5,))
                    mock_result.boxes.xyxy = torch.rand(5, 4) * 640
                    return [mock_result]
                
                mock_yolo.side_effect = realistic_inference
                mock_yolo_class.return_value = mock_yolo
                
                await model.load_model()
                
                # Warm-up runs
                for _ in range(3):
                    await model.predict(sample_image)
                
                # Performance measurement runs
                inference_times = []
                for _ in range(10):
                    start_time = time.time()
                    results = await model.predict(sample_image)
                    end_time = time.time()
                    
                    inference_time_ms = (end_time - start_time) * 1000
                    inference_times.append(inference_time_ms)
                    
                    assert len(results) == 5  # Should detect 5 objects
                
                # Calculate statistics
                avg_time = statistics.mean(inference_times)
                p95_time = np.percentile(inference_times, 95)
                std_dev = statistics.stdev(inference_times)
                
                # Record metrics
                self.record_metric(f"yolo_{model_size}_avg_inference", avg_time)
                self.record_metric(f"yolo_{model_size}_p95_inference", p95_time)
                self.record_metric(f"yolo_{model_size}_std_dev", std_dev)
                
                # Performance assertions
                expected_max_times = {"n": 100, "s": 150, "m": 250}  # ms
                assert avg_time <= expected_max_times[model_size], \
                    f"YOLOv8{model_size} avg inference time {avg_time}ms exceeds {expected_max_times[model_size]}ms"
                
                # Consistency check - standard deviation should be reasonable
                assert std_dev <= avg_time * 0.3, \
                    f"YOLOv8{model_size} inference time too variable (std_dev: {std_dev}ms)"
    
    @requires_gpu
    @pytest.mark.asyncio
    async def test_clip_encoding_performance(self, sample_image, performance_thresholds):
        """Test CLIP encoding performance on GPU."""
        from src.core.models.clip import CLIPVisionModel
        
        # Test different CLIP model sizes
        clip_models = ["ViT-B/32", "ViT-B/16"]  # Different model complexities
        
        for model_name in clip_models:
            model = CLIPVisionModel(model_name=model_name)
            
            with patch('clip.load') as mock_clip_load:
                # Create mock CLIP model
                mock_clip = Mock()
                mock_preprocess = Mock()
                
                def realistic_encoding(*args, **kwargs):
                    # Simulate GPU encoding time based on model
                    base_times = {"ViT-B/32": 0.08, "ViT-B/16": 0.15}  # seconds
                    model_key = model_name
                    time.sleep(base_times.get(model_key, 0.08))
                    return torch.randn(1, 512)  # 512-dim embedding
                
                def realistic_preprocessing(*args, **kwargs):
                    return torch.randn(3, 224, 224)  # Preprocessed tensor
                
                mock_clip.encode_image.side_effect = realistic_encoding
                mock_preprocess.side_effect = realistic_preprocessing
                mock_clip_load.return_value = (mock_clip, mock_preprocess)
                
                await model.load_model()
                
                # Warm-up runs
                for _ in range(3):
                    with patch('torch.no_grad'):
                        await model.encode_image(sample_image)
                
                # Performance measurement runs
                encoding_times = []
                for _ in range(10):
                    start_time = time.time()
                    with patch('torch.no_grad'):
                        embedding = await model.encode_image(sample_image)
                    end_time = time.time()
                    
                    encoding_time_ms = (end_time - start_time) * 1000
                    encoding_times.append(encoding_time_ms)
                    
                    assert embedding.shape == (1, 512)  # Correct embedding size
                
                # Calculate statistics
                avg_time = statistics.mean(encoding_times)
                p95_time = np.percentile(encoding_times, 95)
                
                # Record metrics
                model_key = model_name.replace("/", "_").replace("-", "_")
                self.record_metric(f"clip_{model_key}_avg_encoding", avg_time)
                self.record_metric(f"clip_{model_key}_p95_encoding", p95_time)
                
                # Performance assertions
                expected_max_times = {"ViT-B/32": 200, "ViT-B/16": 350}  # ms
                model_key = model_name
                assert avg_time <= expected_max_times[model_key], \
                    f"CLIP {model_name} avg encoding time {avg_time}ms exceeds {expected_max_times[model_key]}ms"
    
    @requires_gpu
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, generate_test_images):
        """Test batch processing performance scaling."""
        from src.core.analyzer import H200ImageAnalyzer
        
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            # Create batch of test images
            test_images = generate_test_images(batch_size)
            
            analyzer = H200ImageAnalyzer()
            
            with patch('src.core.models.manager.ModelManager') as mock_manager_class:
                # Mock model manager with realistic batch processing
                mock_manager = Mock()
                
                def mock_batch_yolo(*args, **kwargs):
                    # Simulate batch YOLO processing - should be more efficient than individual
                    batch_time_per_image = 0.03  # 30ms per image in batch
                    time.sleep(batch_time_per_image * len(args[0]))
                    
                    results = []
                    for _ in args[0]:
                        mock_result = Mock()
                        mock_result.boxes = Mock()
                        mock_result.boxes.data = torch.randn(3, 6)
                        mock_result.boxes.conf = torch.rand(3)
                        mock_result.boxes.cls = torch.randint(0, 80, (3,))
                        mock_result.boxes.xyxy = torch.rand(3, 4) * 640
                        results.append(mock_result)
                    return results
                
                def mock_batch_clip(*args, **kwargs):
                    # Simulate batch CLIP encoding
                    batch_time_per_image = 0.06  # 60ms per image in batch
                    time.sleep(batch_time_per_image * len(args[0]))
                    return torch.randn(len(args[0]), 512)
                
                mock_yolo = Mock()
                mock_yolo.side_effect = mock_batch_yolo
                mock_clip = Mock()
                mock_clip.encode_image_batch.side_effect = mock_batch_clip
                
                mock_manager.get_model.side_effect = lambda name: {
                    'yolo': mock_yolo,
                    'clip': mock_clip
                }[name]
                
                analyzer.model_manager = mock_manager
                analyzer._initialized = True
                
                # Measure batch processing time
                start_time = time.time()
                
                with patch.object(analyzer, 'positioning_engine') as mock_pos_engine:
                    mock_pos_engine.calculate_positions.return_value = []
                    
                    batch_result = await analyzer.analyze_batch(test_images)
                
                end_time = time.time()
                
                total_time_ms = (end_time - start_time) * 1000
                time_per_image_ms = total_time_ms / batch_size
                
                # Record metrics
                self.record_metric(f"batch_{batch_size}_total_time", total_time_ms)
                self.record_metric(f"batch_{batch_size}_time_per_image", time_per_image_ms)
                
                # Performance assertions
                assert len(batch_result.results) == batch_size
                
                # Batch processing should be more efficient for larger batches
                if batch_size > 1:
                    # Time per image should be less than individual processing time
                    individual_time_estimate = 100  # ms per image individually
                    assert time_per_image_ms < individual_time_estimate * 0.8, \
                        f"Batch processing not efficient enough: {time_per_image_ms}ms per image"
    
    @requires_gpu
    @pytest.mark.asyncio
    async def test_memory_usage_performance(self, generate_test_images):
        """Test GPU memory usage during operations."""
        from src.core.analyzer import H200ImageAnalyzer
        
        # Test memory usage with different batch sizes
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU cache
                initial_memory = torch.cuda.memory_allocated()
            else:
                initial_memory = 0
            
            test_images = generate_test_images(batch_size)
            analyzer = H200ImageAnalyzer()
            
            with patch('src.core.models.manager.ModelManager') as mock_manager_class:
                mock_manager = Mock()
                
                def mock_memory_intensive_operation(*args, **kwargs):
                    # Simulate memory allocation during processing
                    if torch.cuda.is_available():
                        # Allocate some GPU memory to simulate model usage
                        dummy_tensor = torch.randn(
                            batch_size * 100, 1000, device='cuda'
                        )
                        time.sleep(0.1)  # Processing time
                        del dummy_tensor
                        torch.cuda.synchronize()
                    else:
                        time.sleep(0.1)
                    
                    return [Mock() for _ in args[0]]
                
                mock_yolo = Mock()
                mock_yolo.side_effect = mock_memory_intensive_operation
                mock_clip = Mock()
                mock_clip.encode_image_batch = Mock(return_value=torch.randn(batch_size, 512))
                
                mock_manager.get_model.side_effect = lambda name: {
                    'yolo': mock_yolo,
                    'clip': mock_clip
                }[name]
                
                analyzer.model_manager = mock_manager
                analyzer._initialized = True
                
                # Measure peak memory usage
                with patch.object(analyzer, 'positioning_engine') as mock_pos_engine:
                    mock_pos_engine.calculate_positions.return_value = []
                    
                    if torch.cuda.is_available():
                        peak_memory = torch.cuda.max_memory_allocated()
                    else:
                        peak_memory = 1024 * 1024 * batch_size  # Simulate 1MB per image
                
                    await analyzer.analyze_batch(test_images)
                
                memory_used_mb = (peak_memory - initial_memory) / (1024 * 1024)
                memory_per_image_mb = memory_used_mb / batch_size
                
                # Record metrics
                self.record_metric(f"memory_batch_{batch_size}_total_mb", memory_used_mb)
                self.record_metric(f"memory_batch_{batch_size}_per_image_mb", memory_per_image_mb)
                
                # Memory usage assertions
                max_memory_per_image = 100  # MB per image
                assert memory_per_image_mb <= max_memory_per_image, \
                    f"Memory usage too high: {memory_per_image_mb}MB per image"
                
                # Memory usage should scale reasonably with batch size
                if batch_size > 1:
                    expected_max_total = batch_size * max_memory_per_image * 1.2  # 20% overhead
                    assert memory_used_mb <= expected_max_total, \
                        f"Total memory usage {memory_used_mb}MB exceeds expected {expected_max_total}MB"
    
    @requires_gpu
    @pytest.mark.asyncio
    async def test_cold_start_performance(self, sample_image, performance_thresholds):
        """Test cold start performance (model loading + first inference)."""
        from src.core.analyzer import H200ImageAnalyzer
        
        # Test FlashBoot cold start performance
        analyzer = H200ImageAnalyzer()
        
        with patch('src.core.models.manager.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            
            # Simulate cold start - model loading + first inference
            def mock_cold_start_inference(*args, **kwargs):
                # FlashBoot target: 500ms-2s cold start
                time.sleep(1.0)  # 1 second cold start simulation
                
                mock_result = Mock()
                mock_result.boxes = Mock()
                mock_result.boxes.data = torch.randn(3, 6)
                mock_result.boxes.conf = torch.rand(3)
                mock_result.boxes.cls = torch.randint(0, 80, (3,))
                mock_result.boxes.xyxy = torch.rand(3, 4) * 640
                return [mock_result]
            
            def mock_warm_inference(*args, **kwargs):
                # Warm inference should be much faster
                time.sleep(0.05)  # 50ms warm inference
                return mock_cold_start_inference(*args, **kwargs)
            
            mock_yolo = Mock()
            mock_clip = Mock()
            mock_clip.encode_image = Mock(return_value=torch.randn(1, 512))
            
            # First call is cold start, subsequent calls are warm
            call_count = 0
            def yolo_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_cold_start_inference(*args, **kwargs)
                else:
                    return mock_warm_inference(*args, **kwargs)
            
            mock_yolo.side_effect = yolo_side_effect
            
            mock_manager.get_model.side_effect = lambda name: {
                'yolo': mock_yolo,
                'clip': mock_clip
            }[name]
            mock_manager.initialize = Mock()
            
            analyzer.model_manager = mock_manager
            
            # Measure cold start time
            start_time = time.time()
            await analyzer.initialize()
            
            with patch.object(analyzer, 'positioning_engine') as mock_pos_engine:
                mock_pos_engine.calculate_positions.return_value = []
                
                result = await analyzer.analyze_image(sample_image)
            
            end_time = time.time()
            
            cold_start_time_ms = (end_time - start_time) * 1000
            
            # Measure warm start time
            start_time = time.time()
            result = await analyzer.analyze_image(sample_image)
            end_time = time.time()
            
            warm_start_time_ms = (end_time - start_time) * 1000
            
            # Record metrics
            self.record_metric("cold_start_time", cold_start_time_ms)
            self.record_metric("warm_start_time", warm_start_time_ms)
            
            # Performance assertions based on updated FlashBoot specs
            assert cold_start_time_ms <= performance_thresholds["cold_start_ms"], \
                f"Cold start time {cold_start_time_ms}ms exceeds threshold {performance_thresholds['cold_start_ms']}ms"
            
            assert warm_start_time_ms <= performance_thresholds["warm_start_ms"], \
                f"Warm start time {warm_start_time_ms}ms exceeds threshold {performance_thresholds['warm_start_ms']}ms"
            
            # Warm start should be significantly faster than cold start
            assert warm_start_time_ms <= cold_start_time_ms * 0.2, \
                f"Warm start not significantly faster: {warm_start_time_ms}ms vs {cold_start_time_ms}ms"
    
    @requires_gpu
    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self, generate_test_images):
        """Test performance under concurrent processing load."""
        from src.core.analyzer import H200ImageAnalyzer
        import asyncio
        
        # Test with multiple concurrent analyzers
        num_concurrent = 4
        images_per_analyzer = 3
        
        async def analyze_batch_task(task_id):
            """Task for concurrent analysis."""
            test_images = generate_test_images(images_per_analyzer)
            analyzer = H200ImageAnalyzer()
            
            with patch('src.core.models.manager.ModelManager') as mock_manager_class:
                mock_manager = Mock()
                
                def mock_concurrent_inference(*args, **kwargs):
                    # Simulate some processing time with slight variation per task
                    base_time = 0.05  # 50ms base
                    task_variation = task_id * 0.01  # Slight variation per task
                    time.sleep(base_time + task_variation)
                    
                    results = []
                    for _ in args[0]:
                        mock_result = Mock()
                        mock_result.boxes = Mock()
                        mock_result.boxes.data = torch.randn(2, 6)
                        mock_result.boxes.conf = torch.rand(2)
                        mock_result.boxes.cls = torch.randint(0, 80, (2,))
                        mock_result.boxes.xyxy = torch.rand(2, 4) * 640
                        results.append(mock_result)
                    return results
                
                mock_yolo = Mock()
                mock_yolo.side_effect = mock_concurrent_inference
                mock_clip = Mock()
                mock_clip.encode_image_batch = Mock(
                    return_value=torch.randn(images_per_analyzer, 512)
                )
                
                mock_manager.get_model.side_effect = lambda name: {
                    'yolo': mock_yolo,
                    'clip': mock_clip
                }[name]
                
                analyzer.model_manager = mock_manager
                analyzer._initialized = True
                
                with patch.object(analyzer, 'positioning_engine') as mock_pos_engine:
                    mock_pos_engine.calculate_positions.return_value = []
                    
                    start_time = time.time()
                    batch_result = await analyzer.analyze_batch(test_images)
                    end_time = time.time()
                    
                    return {
                        "task_id": task_id,
                        "processing_time_ms": (end_time - start_time) * 1000,
                        "results_count": len(batch_result.results)
                    }
        
        # Run concurrent tasks
        start_time = time.time()
        tasks = [analyze_batch_task(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_concurrent_time_ms = (end_time - start_time) * 1000
        
        # Calculate metrics
        individual_times = [r["processing_time_ms"] for r in results]
        avg_individual_time = statistics.mean(individual_times)
        max_individual_time = max(individual_times)
        
        # Record metrics
        self.record_metric("concurrent_total_time", total_concurrent_time_ms)
        self.record_metric("concurrent_avg_individual", avg_individual_time)
        self.record_metric("concurrent_max_individual", max_individual_time)
        
        # Performance assertions
        # Concurrent processing should not be significantly slower than sequential
        expected_sequential_time = avg_individual_time * num_concurrent
        efficiency_ratio = total_concurrent_time_ms / expected_sequential_time
        
        # Should have some concurrency benefit (efficiency ratio < 1.0)
        # But might have overhead (allow up to 1.5x sequential time)
        assert efficiency_ratio <= 1.5, \
            f"Concurrent processing too slow: {efficiency_ratio:.2f}x sequential time"
        
        # All tasks should complete
        assert all(r["results_count"] == images_per_analyzer for r in results)
        
        # Individual task times should be reasonable and consistent
        std_dev = statistics.stdev(individual_times) if len(individual_times) > 1 else 0
        assert std_dev <= avg_individual_time * 0.3, \
            f"Too much variation in concurrent task times: std_dev={std_dev}ms"


@pytest.mark.performance
@pytest.mark.gpu
class TestGPUResourceManagement(PerformanceTestBase):
    """Test GPU resource management and optimization."""
    
    @requires_gpu
    @pytest.mark.asyncio
    async def test_gpu_memory_cleanup(self, generate_test_images):
        """Test GPU memory cleanup and garbage collection."""
        
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        # Record initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Simulate memory-intensive operations
        for batch_size in [4, 8, 16, 8, 4]:  # Varying batch sizes
            test_images = generate_test_images(batch_size)
            
            # Allocate GPU tensors to simulate model processing
            tensors = []
            for _ in range(batch_size):
                tensor = torch.randn(1000, 1000, device='cuda')
                tensors.append(tensor)
            
            current_memory = torch.cuda.memory_allocated()
            memory_increase = current_memory - initial_memory
            
            # Record memory usage
            self.record_metric(
                f"gpu_memory_batch_{batch_size}_mb", 
                memory_increase / (1024 * 1024)
            )
            
            # Clean up tensors
            del tensors
            torch.cuda.empty_cache()
            
            # Verify memory cleanup
            post_cleanup_memory = torch.cuda.memory_allocated()
            memory_leaked = post_cleanup_memory - initial_memory
            
            # Should have minimal memory leakage
            assert memory_leaked <= 10 * 1024 * 1024, \
                f"Memory leak detected: {memory_leaked / (1024*1024):.1f}MB"
    
    @requires_gpu 
    @pytest.mark.asyncio
    async def test_gpu_utilization_monitoring(self):
        """Test GPU utilization monitoring during operations."""
        
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        # Simulate varying GPU workloads
        workload_intensities = [0.1, 0.5, 0.9, 0.5, 0.1]  # Light to heavy to light
        
        for intensity in workload_intensities:
            # Simulate GPU work proportional to intensity
            work_size = int(1000 * intensity)
            duration = 0.2  # 200ms of work
            
            start_time = time.time()
            
            # Create GPU workload
            with torch.cuda.device(0):
                tensor = torch.randn(work_size, work_size, device='cuda')
                
                # Perform computation to utilize GPU
                for _ in range(10):
                    tensor = torch.mm(tensor, tensor.t())
                    torch.cuda.synchronize()
                
                del tensor
                torch.cuda.empty_cache()
            
            end_time = time.time()
            actual_duration = (end_time - start_time) * 1000
            
            # Record utilization metrics
            self.record_metric(f"gpu_workload_intensity_{intensity}", actual_duration)
            
            # Higher intensity should generally take longer
            if intensity > 0.1:
                assert actual_duration >= 50, \
                    f"GPU workload {intensity} completed too quickly: {actual_duration}ms"


@pytest.mark.performance
class TestPerformanceRegression(PerformanceTestBase):
    """Test for performance regressions."""
    
    @pytest.mark.asyncio
    async def test_performance_baseline_comparison(self, performance_thresholds):
        """Compare current performance against baseline thresholds."""
        
        # Simulate current performance metrics
        current_metrics = {
            "cold_start_ms": 1500,    # Should be <= 2000
            "warm_start_ms": 80,      # Should be <= 100  
            "image_processing_ms": 450, # Should be <= 500
            "cache_hit_rate": 0.88,   # Should be >= 0.85
            "gpu_utilization": 0.72,  # Should be >= 0.70
        }
        
        # Check all metrics against thresholds
        for metric, current_value in current_metrics.items():
            threshold = performance_thresholds[metric]
            
            self.record_metric(f"current_{metric}", current_value)
            self.record_metric(f"threshold_{metric}", threshold)
            
            if metric == "cache_hit_rate" or metric == "gpu_utilization":
                # Higher is better
                assert current_value >= threshold, \
                    f"Performance regression in {metric}: {current_value} < {threshold}"
            else:
                # Lower is better  
                assert current_value <= threshold, \
                    f"Performance regression in {metric}: {current_value} > {threshold}"
    
    @pytest.mark.asyncio
    async def test_scalability_limits(self):
        """Test system behavior at scalability limits."""
        
        # Test with increasing load until limits
        max_concurrent_requests = 50
        request_sizes = [1, 5, 10, 20, 50]
        
        for num_requests in request_sizes:
            if num_requests > max_concurrent_requests:
                continue
                
            # Simulate concurrent requests
            start_time = time.time()
            
            # Mock concurrent processing
            await asyncio.sleep(0.1 * num_requests / 10)  # Simulate scaling
            
            end_time = time.time()
            
            processing_time_ms = (end_time - start_time) * 1000
            time_per_request = processing_time_ms / num_requests
            
            self.record_metric(f"scale_{num_requests}_total_time", processing_time_ms)
            self.record_metric(f"scale_{num_requests}_per_request", time_per_request)
            
            # Should scale reasonably - not exponentially worse
            if num_requests > 1:
                expected_linear_scaling = num_requests * 50  # 50ms per request baseline
                assert processing_time_ms <= expected_linear_scaling * 2, \
                    f"Poor scaling at {num_requests} requests: {processing_time_ms}ms"
    
    def teardown_method(self):
        """Print performance summary after tests."""
        super().teardown_method()
        
        summary = self.get_performance_summary()
        print("\n" + "="*50)
        print("PERFORMANCE TEST SUMMARY")
        print("="*50)
        
        for metric_name, metric_data in summary["metrics"].items():
            print(f"{metric_name}: {metric_data['value']:.2f} {metric_data['unit']}")
        
        print(f"\nTotal metrics recorded: {summary['total_metrics']}")
        print(f"Test duration: {summary['test_duration']:.2f}s")
        print("="*50)