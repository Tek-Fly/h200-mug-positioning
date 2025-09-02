"""
High-level deployment management with state tracking
"""

# Standard library imports
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

# Local imports
from ..database.mongodb import get_database
from ..database.redis_client import get_redis_client
from .client import RunPodClient
from .config import DeploymentConfig, DeploymentMode, DeploymentStatus
from .deployer import RunPodDeployer
from .volume import VolumeManager

logger = logging.getLogger(__name__)


@dataclass
class DeploymentState:
    """Deployment state information"""

    deployment_id: str
    mode: DeploymentMode
    status: DeploymentStatus
    config: DeploymentConfig
    created_at: datetime
    updated_at: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    health_checks: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "deployment_id": self.deployment_id,
            "mode": self.mode.value,
            "status": self.status.value,
            "config": self.config.dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metrics": self.metrics,
            "events": self.events,
            "health_checks": self.health_checks,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentState":
        """Create from dictionary"""
        return cls(
            deployment_id=data["deployment_id"],
            mode=DeploymentMode(data["mode"]),
            status=DeploymentStatus(data["status"]),
            config=DeploymentConfig(**data["config"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metrics=data.get("metrics", {}),
            events=data.get("events", []),
            health_checks=data.get("health_checks", {}),
        )


class DeploymentManager:
    """High-level deployment manager with state tracking"""

    def __init__(self, api_key: str):
        """Initialize deployment manager"""
        self.client = RunPodClient(api_key)
        self.volume_manager = VolumeManager(self.client)
        self.deployer = RunPodDeployer(self.client, self.volume_manager)
        self._deployments: Dict[str, DeploymentState] = {}
        self._monitoring_tasks: Set[asyncio.Task] = set()
        self._initialized = False

    async def initialize(self):
        """Initialize manager and load existing deployments"""
        if self._initialized:
            return

        logger.info("Initializing deployment manager")

        try:
            # Load deployment states from database
            await self._load_deployment_states()

            # Start monitoring tasks
            await self._start_monitoring()

            self._initialized = True
            logger.info("Deployment manager initialized")

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to connect during initialization: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            raise RuntimeError(f"Failed to initialize deployment manager: {e}")

    async def shutdown(self):
        """Shutdown manager and cleanup"""
        logger.info("Shutting down deployment manager")

        # Cancel monitoring tasks
        for task in self._monitoring_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)

        # Save final states
        await self._save_deployment_states()

        # Close client connection
        if hasattr(self.client, "session") and self.client.session:
            await self.client.session.close()

    async def deploy(self, config: DeploymentConfig) -> DeploymentState:
        """Deploy with state tracking"""
        logger.info(f"Deploying {config.name} in {config.mode} mode")

        async with self.client:
            # Deploy
            result = await self.deployer.deploy(config)

            # Create deployment state
            if config.mode == DeploymentMode.BOTH:
                # Handle multiple deployments
                states = []
                for mode, deployment in result["deployments"].items():
                    state = DeploymentState(
                        deployment_id=deployment["id"],
                        mode=DeploymentMode(mode),
                        status=DeploymentStatus.RUNNING,
                        config=config,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    )
                    self._deployments[deployment["id"]] = state
                    states.append(state)

                # Save states
                await self._save_deployment_states()

                # Return primary state (serverless)
                return states[0] if states else None

            else:
                # Single deployment
                state = DeploymentState(
                    deployment_id=result["id"],
                    mode=config.mode,
                    status=DeploymentStatus.RUNNING,
                    config=config,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )

                # Store state
                self._deployments[result["id"]] = state

                # Save to database
                await self._save_deployment_states()

                # Add event
                await self._add_deployment_event(
                    state.deployment_id, "deployed", {"result": result}
                )

                return state

    async def update_deployment(
        self, deployment_id: str, updates: Dict[str, Any]
    ) -> DeploymentState:
        """Update deployment configuration"""
        state = self._deployments.get(deployment_id)
        if not state:
            raise ValueError(f"Deployment {deployment_id} not found")

        async with self.client:
            # Update deployment
            result = await self.deployer.update_deployment(
                deployment_id, state.mode, updates
            )

            # Update state
            state.updated_at = datetime.utcnow()

            # Add event
            await self._add_deployment_event(
                deployment_id, "updated", {"updates": updates, "result": result}
            )

            # Save state
            await self._save_deployment_states()

            return state

    async def stop_deployment(self, deployment_id: str) -> DeploymentState:
        """Stop deployment"""
        state = self._deployments.get(deployment_id)
        if not state:
            raise ValueError(f"Deployment {deployment_id} not found")

        async with self.client:
            # Stop deployment
            result = await self.deployer.stop_deployment(deployment_id, state.mode)

            # Update state
            state.status = DeploymentStatus.STOPPED
            state.updated_at = datetime.utcnow()

            # Add event
            await self._add_deployment_event(
                deployment_id, "stopped", {"result": result}
            )

            # Save state
            await self._save_deployment_states()

            return state

    async def start_deployment(self, deployment_id: str) -> DeploymentState:
        """Start stopped deployment"""
        state = self._deployments.get(deployment_id)
        if not state:
            raise ValueError(f"Deployment {deployment_id} not found")

        async with self.client:
            # Start deployment
            result = await self.deployer.start_deployment(
                deployment_id, state.mode, state.config
            )

            # Update state
            state.status = DeploymentStatus.RUNNING
            state.updated_at = datetime.utcnow()

            # Add event
            await self._add_deployment_event(
                deployment_id, "started", {"result": result}
            )

            # Save state
            await self._save_deployment_states()

            return state

    async def terminate_deployment(self, deployment_id: str) -> DeploymentState:
        """Terminate deployment permanently"""
        state = self._deployments.get(deployment_id)
        if not state:
            raise ValueError(f"Deployment {deployment_id} not found")

        async with self.client:
            # Terminate deployment
            result = await self.deployer.terminate_deployment(deployment_id, state.mode)

            # Update state
            state.status = DeploymentStatus.TERMINATED
            state.updated_at = datetime.utcnow()

            # Add event
            await self._add_deployment_event(
                deployment_id, "terminated", {"result": result}
            )

            # Save state
            await self._save_deployment_states()

            return state

    async def get_deployment(self, deployment_id: str) -> Optional[DeploymentState]:
        """Get deployment state"""
        return self._deployments.get(deployment_id)

    async def list_deployments(
        self,
        mode: Optional[DeploymentMode] = None,
        status: Optional[DeploymentStatus] = None,
    ) -> List[DeploymentState]:
        """List deployments with optional filtering"""
        deployments = list(self._deployments.values())

        if mode:
            deployments = [d for d in deployments if d.mode == mode]

        if status:
            deployments = [d for d in deployments if d.status == status]

        return deployments

    async def get_deployment_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment metrics"""
        state = self._deployments.get(deployment_id)
        if not state:
            raise ValueError(f"Deployment {deployment_id} not found")

        async with self.client:
            # Get current status
            status = await self.deployer.get_deployment_status(
                deployment_id, state.mode
            )

            # Get historical metrics
            if state.mode == DeploymentMode.SERVERLESS:
                metrics = await self.client.get_endpoint_metrics(
                    deployment_id, start_time=datetime.utcnow() - timedelta(hours=1)
                )
            else:
                metrics = await self.client.get_pod_metrics(
                    deployment_id, start_time=datetime.utcnow() - timedelta(hours=1)
                )

            # Combine with cached metrics
            combined = {
                "current": status,
                "historical": metrics,
                "cached": state.metrics,
            }

            return combined

    async def get_deployment_logs(self, deployment_id: str, lines: int = 100) -> str:
        """Get deployment logs"""
        state = self._deployments.get(deployment_id)
        if not state:
            raise ValueError(f"Deployment {deployment_id} not found")

        async with self.client:
            return await self.deployer.get_deployment_logs(
                deployment_id, state.mode, lines
            )

    async def scale_deployment(
        self, deployment_id: str, min_workers: int, max_workers: int
    ) -> DeploymentState:
        """Scale serverless deployment"""
        state = self._deployments.get(deployment_id)
        if not state:
            raise ValueError(f"Deployment {deployment_id} not found")

        if state.mode != DeploymentMode.SERVERLESS:
            raise ValueError("Only serverless deployments can be scaled")

        async with self.client:
            # Update scaling configuration
            updates = {"min_workers": min_workers, "max_workers": max_workers}

            result = await self.deployer.update_deployment(
                deployment_id, state.mode, updates
            )

            # Update state config
            state.config.scaling.min_workers = min_workers
            state.config.scaling.max_workers = max_workers
            state.updated_at = datetime.utcnow()

            # Add event
            await self._add_deployment_event(
                deployment_id,
                "scaled",
                {"min_workers": min_workers, "max_workers": max_workers},
            )

            # Save state
            await self._save_deployment_states()

            return state

    async def health_check(self, deployment_id: str) -> Dict[str, Any]:
        """Perform deployment health check"""
        state = self._deployments.get(deployment_id)
        if not state:
            raise ValueError(f"Deployment {deployment_id} not found")

        async with self.client:
            # Get current status
            status = await self.deployer.get_deployment_status(
                deployment_id, state.mode
            )

            # Perform health checks
            health = {
                "deployment_id": deployment_id,
                "timestamp": datetime.utcnow().isoformat(),
                "status": status["status"],
                "checks": {},
            }

            # Check deployment is running
            health["checks"]["running"] = status["status"] == DeploymentStatus.RUNNING

            # Mode-specific checks
            if state.mode == DeploymentMode.SERVERLESS:
                workers = status.get("workers", {})
                health["checks"]["has_workers"] = workers.get("running", 0) > 0
                health["checks"]["error_rate"] = self._calculate_error_rate(
                    status["metrics"]
                )

            else:  # Timed
                metrics = status.get("metrics", {})
                health["checks"]["cpu_healthy"] = (
                    metrics.get("cpu_usage_percent", 0) < 90
                )
                health["checks"]["memory_healthy"] = metrics.get(
                    "memory_usage_mb", 0
                ) < (
                    state.config.resources.memory_gb * 900
                )  # 90% threshold
                health["checks"]["gpu_healthy"] = (
                    metrics.get("gpu_usage_percent", 0) < 95
                )

            # Overall health
            health["healthy"] = all(health["checks"].values())

            # Update state
            state.health_checks[datetime.utcnow().isoformat()] = health

            return health

    def _calculate_error_rate(self, metrics: Dict[str, Any]) -> bool:
        """Calculate if error rate is acceptable"""
        total = metrics.get("requests_total", 0)
        failed = metrics.get("requests_failed", 0)

        if total == 0:
            return True

        error_rate = failed / total
        return error_rate < 0.05  # 5% error rate threshold

    async def _add_deployment_event(
        self, deployment_id: str, event_type: str, data: Dict[str, Any]
    ):
        """Add event to deployment history"""
        state = self._deployments.get(deployment_id)
        if not state:
            return

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "data": data,
        }

        state.events.append(event)

        # Keep only last 100 events
        if len(state.events) > 100:
            state.events = state.events[-100:]

    async def _start_monitoring(self):
        """Start monitoring tasks for active deployments"""
        # Create monitoring task
        task = asyncio.create_task(self._monitor_deployments())
        self._monitoring_tasks.add(task)
        task.add_done_callback(self._monitoring_tasks.discard)

    async def _monitor_deployments(self):
        """Monitor deployment health and metrics"""
        logger.info("Starting deployment monitoring")

        while True:
            try:
                async with self.client:
                    # Check each deployment
                    for deployment_id, state in list(self._deployments.items()):
                        if state.status not in [
                            DeploymentStatus.RUNNING,
                            DeploymentStatus.STARTING,
                        ]:
                            continue

                        try:
                            # Update status
                            status = await self.deployer.get_deployment_status(
                                deployment_id, state.mode
                            )

                            # Update state
                            state.status = status["status"]
                            state.metrics = status.get("metrics", {})
                            state.updated_at = datetime.utcnow()

                            # Perform health check
                            if state.status == DeploymentStatus.RUNNING:
                                await self.health_check(deployment_id)

                        except asyncio.TimeoutError:
                            logger.warning(
                                f"Timeout monitoring deployment {deployment_id}"
                            )
                        except ValueError as e:
                            logger.error(
                                f"Invalid deployment state for {deployment_id}: {e}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Unexpected error monitoring deployment {deployment_id}: {e}"
                            )

                    # Save states periodically
                    await self._save_deployment_states()

                # Wait before next check
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                logger.info("Deployment monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in deployment monitoring: {e}")
                await asyncio.sleep(60)

    async def _load_deployment_states(self):
        """Load deployment states from database"""
        try:
            db = await get_database()
            collection = db.deployments

            # Load all deployment documents
            async for doc in collection.find({"active": True}):
                try:
                    state = DeploymentState.from_dict(doc)
                    self._deployments[state.deployment_id] = state
                except (KeyError, ValueError) as e:
                    logger.error(f"Invalid deployment document format: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error loading deployment state: {e}")

            logger.info(f"Loaded {len(self._deployments)} deployment states")

        except ConnectionError as e:
            logger.error(f"Database connection failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading deployment states: {e}")

    async def _save_deployment_states(self):
        """Save deployment states to database"""
        try:
            db = await get_database()
            collection = db.deployments

            # Save each deployment state
            for deployment_id, state in self._deployments.items():
                doc = state.to_dict()
                doc["_id"] = deployment_id
                doc["active"] = state.status != DeploymentStatus.TERMINATED

                await collection.replace_one({"_id": deployment_id}, doc, upsert=True)

            logger.debug(f"Saved {len(self._deployments)} deployment states")

        except ConnectionError as e:
            logger.error(f"Database connection failed during save: {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving deployment states: {e}")

    async def get_deployment_costs(
        self,
        deployment_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Calculate deployment costs"""
        state = self._deployments.get(deployment_id)
        if not state:
            raise ValueError(f"Deployment {deployment_id} not found")

        # GPU hourly rates (example rates)
        gpu_rates = {
            "H100": 3.50,
            "A100": 2.50,
            "RTX4090": 1.20,
            "RTX3090": 0.80,
            "V100": 1.50,
        }

        # Calculate runtime hours
        if not start_date:
            start_date = state.created_at
        if not end_date:
            end_date = datetime.utcnow()

        runtime_hours = (end_date - start_date).total_seconds() / 3600

        # Calculate costs
        gpu_type = state.config.gpu.type.value
        gpu_count = state.config.gpu.count
        hourly_rate = gpu_rates.get(gpu_type, 1.0) * gpu_count

        # Additional costs
        storage_gb = state.config.resources.volume_size_gb + (
            state.config.resources.network_volume_size_gb or 0
        )
        storage_cost = storage_gb * 0.10 / 730  # $0.10/GB/month

        # Bandwidth (estimated)
        bandwidth_cost = 0.01 * runtime_hours  # Simplified estimate

        total_cost = (
            (hourly_rate * runtime_hours)
            + (storage_cost * runtime_hours)
            + bandwidth_cost
        )

        return {
            "deployment_id": deployment_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "hours": round(runtime_hours, 2),
            },
            "costs": {
                "gpu": round(hourly_rate * runtime_hours, 2),
                "storage": round(storage_cost * runtime_hours, 2),
                "bandwidth": round(bandwidth_cost, 2),
                "total": round(total_cost, 2),
            },
            "breakdown": {
                "gpu_type": gpu_type,
                "gpu_count": gpu_count,
                "hourly_rate": hourly_rate,
                "storage_gb": storage_gb,
            },
            "currency": "USD",
        }
