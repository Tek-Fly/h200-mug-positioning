"""
Async processing pipeline for H200 Intelligent Mug Positioning System.

This module implements the main async processing pipeline with queue management,
result streaming, error recovery, and load balancing.
"""

# Standard library imports
import asyncio
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

# Third-party imports
import structlog

# First-party imports
from src.core.analyzer import AnalysisResult, BatchResult, H200ImageAnalyzer

# Initialize structured logger
logger = structlog.get_logger(__name__)


class JobStatus(Enum):
    """Processing job status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ProcessingJob:
    """Single processing job."""

    job_id: str
    images: List[Union[bytes, str]]
    metadata: Optional[Dict[str, Any]] = None
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Union[AnalysisResult, BatchResult]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None


@dataclass
class QueueStats:
    """Queue statistics."""

    pending_jobs: int
    processing_jobs: int
    completed_jobs: int
    failed_jobs: int
    average_wait_time_ms: float
    average_processing_time_ms: float
    throughput_per_second: float


class AsyncProcessingPipeline:
    """
    Main async processing pipeline with advanced queue management.

    Features:
    - Priority-based job scheduling
    - Automatic batching for GPU efficiency
    - Result streaming
    - Error recovery with exponential backoff
    - Load balancing across multiple workers
    - Real-time performance monitoring
    """

    def __init__(
        self,
        analyzer: Optional[H200ImageAnalyzer] = None,
        num_workers: int = 4,
        max_queue_size: int = 1000,
        batch_timeout_ms: int = 100,
        max_batch_size: int = 32,
        enable_auto_scaling: bool = True,
        min_workers: int = 1,
        max_workers: int = 8,
    ):
        """
        Initialize async processing pipeline.

        Args:
            analyzer: Image analyzer instance
            num_workers: Initial number of worker tasks
            max_queue_size: Maximum queue size
            batch_timeout_ms: Timeout for batch collection
            max_batch_size: Maximum batch size
            enable_auto_scaling: Enable dynamic worker scaling
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
        """
        self.analyzer = analyzer or H200ImageAnalyzer()
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.batch_timeout_ms = batch_timeout_ms
        self.max_batch_size = max_batch_size
        self.enable_auto_scaling = enable_auto_scaling
        self.min_workers = min_workers
        self.max_workers = max_workers

        # Job queues by priority
        self._queues: Dict[JobPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_queue_size) for priority in JobPriority
        }

        # Active jobs tracking
        self._active_jobs: Dict[str, ProcessingJob] = {}
        self._completed_jobs: deque = deque(maxlen=1000)
        self._job_lock = asyncio.Lock()

        # Worker management
        self._workers: List[asyncio.Task] = []
        self._worker_stats: Dict[int, Dict[str, Any]] = {}
        self._shutdown = False

        # Performance tracking
        self._total_processed = 0
        self._total_failed = 0
        self._processing_times: deque = deque(maxlen=1000)
        self._wait_times: deque = deque(maxlen=1000)

        logger.info(
            "AsyncProcessingPipeline initialized",
            num_workers=num_workers,
            max_batch_size=max_batch_size,
            enable_auto_scaling=enable_auto_scaling,
        )

    async def start(self) -> None:
        """Start the processing pipeline."""
        # Initialize analyzer
        await self.analyzer.initialize()

        # Start workers
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)
            self._worker_stats[i] = {"processed": 0, "failed": 0, "last_job_time": None}

        # Start auto-scaling monitor if enabled
        if self.enable_auto_scaling:
            asyncio.create_task(self._auto_scale_monitor())

        logger.info("Processing pipeline started", workers=len(self._workers))

    async def stop(self) -> None:
        """Stop the processing pipeline gracefully."""
        logger.info("Stopping processing pipeline...")
        self._shutdown = True

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)

        # Cleanup analyzer
        await self.analyzer.cleanup()

        logger.info("Processing pipeline stopped")

    async def submit_job(
        self,
        images: List[Union[bytes, str]],
        metadata: Optional[Dict[str, Any]] = None,
        priority: JobPriority = JobPriority.NORMAL,
        callback: Optional[Callable] = None,
    ) -> str:
        """
        Submit a processing job to the pipeline.

        Args:
            images: List of images to process
            metadata: Optional metadata
            priority: Job priority
            callback: Optional callback function

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        job = ProcessingJob(
            job_id=job_id,
            images=images,
            metadata=metadata,
            priority=priority,
            callback=callback,
        )

        # Add to appropriate queue
        queue = self._queues[priority]

        if queue.full():
            raise RuntimeError(f"Queue full for priority {priority.name}")

        await queue.put(job)

        async with self._job_lock:
            self._active_jobs[job_id] = job

        logger.info(
            "Job submitted",
            job_id=job_id,
            num_images=len(images),
            priority=priority.name,
        )

        return job_id

    async def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get current job status."""
        async with self._job_lock:
            # Check active jobs
            if job_id in self._active_jobs:
                return self._active_jobs[job_id]

            # Check completed jobs
            for job in self._completed_jobs:
                if job.job_id == job_id:
                    return job

        return None

    async def stream_results(
        self, job_id: Optional[str] = None, priority: Optional[JobPriority] = None
    ) -> AsyncIterator[ProcessingJob]:
        """
        Stream processing results as they complete.

        Args:
            job_id: Specific job ID to stream (None for all)
            priority: Filter by priority (None for all)

        Yields:
            Completed ProcessingJob instances
        """
        seen_jobs = set()

        while not self._shutdown:
            # Check completed jobs
            for job in list(self._completed_jobs):
                if job.job_id in seen_jobs:
                    continue

                if job_id and job.job_id != job_id:
                    continue

                if priority and job.priority != priority:
                    continue

                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    seen_jobs.add(job.job_id)
                    yield job

                    # If streaming specific job, we're done
                    if job_id and job.job_id == job_id:
                        return

            # Brief sleep to avoid busy waiting
            await asyncio.sleep(0.1)

    async def _worker_loop(self, worker_id: int) -> None:
        """Main worker processing loop."""
        logger.info(f"Worker {worker_id} started")

        while not self._shutdown:
            try:
                # Get next job from highest priority queue
                job = await self._get_next_job()
                if not job:
                    await asyncio.sleep(0.1)
                    continue

                # Process job
                await self._process_job(job, worker_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Worker {worker_id} error",
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
                await asyncio.sleep(1)

        logger.info(f"Worker {worker_id} stopped")

    async def _get_next_job(self) -> Optional[ProcessingJob]:
        """Get next job from priority queues."""
        # Check queues in priority order
        for priority in sorted(JobPriority, key=lambda p: p.value, reverse=True):
            queue = self._queues[priority]
            if not queue.empty():
                try:
                    return await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

        return None

    async def _process_job(self, job: ProcessingJob, worker_id: int) -> None:
        """Process a single job."""
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()

        # Calculate wait time
        wait_time = (job.started_at - job.created_at).total_seconds() * 1000
        self._wait_times.append(wait_time)

        try:
            # Process images
            start_time = time.perf_counter()

            if len(job.images) == 1:
                # Single image
                result = await self.analyzer.analyze_image(
                    job.images[0], metadata=job.metadata
                )
            else:
                # Batch processing
                result = await self.analyzer.analyze_batch(
                    job.images,
                    metadata_list=(
                        [job.metadata] * len(job.images) if job.metadata else None
                    ),
                )

            processing_time = (time.perf_counter() - start_time) * 1000
            self._processing_times.append(processing_time)

            # Update job
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result = result

            # Update stats
            self._total_processed += 1
            self._worker_stats[worker_id]["processed"] += 1
            self._worker_stats[worker_id]["last_job_time"] = processing_time

            # Execute callback if provided
            if job.callback:
                try:
                    await job.callback(job)
                except Exception as e:
                    logger.error("Job callback error", job_id=job.job_id, error=str(e))

            logger.info(
                "Job completed",
                job_id=job.job_id,
                worker_id=worker_id,
                processing_time_ms=processing_time,
                wait_time_ms=wait_time,
            )

        except Exception as e:
            # Handle failure
            job.error = str(e)
            job.retry_count += 1

            if job.retry_count < job.max_retries:
                # Retry with exponential backoff
                job.status = JobStatus.PENDING
                retry_delay = 2**job.retry_count

                logger.warning(
                    "Job failed, retrying",
                    job_id=job.job_id,
                    retry_count=job.retry_count,
                    retry_delay=retry_delay,
                    error=str(e),
                )

                await asyncio.sleep(retry_delay)
                await self._queues[job.priority].put(job)
            else:
                # Max retries exceeded
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()

                self._total_failed += 1
                self._worker_stats[worker_id]["failed"] += 1

                logger.error(
                    "Job failed permanently",
                    job_id=job.job_id,
                    error=str(e),
                    traceback=traceback.format_exc(),
                )

        finally:
            # Move to completed jobs
            async with self._job_lock:
                if job.job_id in self._active_jobs:
                    del self._active_jobs[job.job_id]
                self._completed_jobs.append(job)

    async def _auto_scale_monitor(self) -> None:
        """Monitor queue and scale workers automatically."""
        logger.info("Auto-scaling monitor started")

        while not self._shutdown:
            try:
                # Calculate queue pressure
                total_pending = sum(q.qsize() for q in self._queues.values())
                queue_pressure = total_pending / self.max_queue_size

                # Calculate worker utilization
                active_workers = len(self._workers)

                # Scale up if high pressure
                if queue_pressure > 0.7 and active_workers < self.max_workers:
                    new_worker_id = max(self._worker_stats.keys()) + 1
                    worker = asyncio.create_task(self._worker_loop(new_worker_id))
                    self._workers.append(worker)
                    self._worker_stats[new_worker_id] = {
                        "processed": 0,
                        "failed": 0,
                        "last_job_time": None,
                    }
                    logger.info(
                        "Scaled up workers",
                        new_count=len(self._workers),
                        queue_pressure=queue_pressure,
                    )

                # Scale down if low pressure
                elif queue_pressure < 0.3 and active_workers > self.min_workers:
                    # Find idle worker
                    idle_worker_id = None
                    for worker_id, stats in self._worker_stats.items():
                        if (
                            stats["last_job_time"] is None
                            or time.time() - stats["last_job_time"] > 60
                        ):
                            idle_worker_id = worker_id
                            break

                    if idle_worker_id is not None:
                        # Cancel idle worker
                        worker_index = idle_worker_id
                        if worker_index < len(self._workers):
                            self._workers[worker_index].cancel()
                            self._workers.pop(worker_index)
                            del self._worker_stats[idle_worker_id]

                            logger.info(
                                "Scaled down workers",
                                new_count=len(self._workers),
                                queue_pressure=queue_pressure,
                            )

                # Sleep before next check
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(
                    "Auto-scaling error", error=str(e), traceback=traceback.format_exc()
                )
                await asyncio.sleep(30)

    def get_stats(self) -> QueueStats:
        """Get pipeline statistics."""
        # Calculate averages
        avg_wait_time = (
            sum(self._wait_times) / len(self._wait_times) if self._wait_times else 0
        )
        avg_processing_time = (
            sum(self._processing_times) / len(self._processing_times)
            if self._processing_times
            else 0
        )

        # Calculate throughput
        if self._processing_times:
            total_time = sum(self._processing_times) / 1000  # Convert to seconds
            throughput = self._total_processed / total_time if total_time > 0 else 0
        else:
            throughput = 0

        # Count jobs by status
        pending_jobs = sum(q.qsize() for q in self._queues.values())
        processing_jobs = sum(
            1 for j in self._active_jobs.values() if j.status == JobStatus.PROCESSING
        )
        completed_jobs = sum(
            1 for j in self._completed_jobs if j.status == JobStatus.COMPLETED
        )
        failed_jobs = sum(
            1 for j in self._completed_jobs if j.status == JobStatus.FAILED
        )

        return QueueStats(
            pending_jobs=pending_jobs,
            processing_jobs=processing_jobs,
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            average_wait_time_ms=avg_wait_time,
            average_processing_time_ms=avg_processing_time,
            throughput_per_second=throughput,
        )

    def get_worker_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get individual worker statistics."""
        return self._worker_stats.copy()
