"""
Deployment orchestration for complex deployment scenarios
"""

# Standard library imports
import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Local imports
from .config import (
    DEPLOYMENT_PRESETS,
    DeploymentConfig,
    DeploymentMode,
    DeploymentStatus,
    EnvironmentType,
)
from .manager import DeploymentManager, DeploymentState

logger = logging.getLogger(__name__)


class DeploymentStrategy(str, Enum):
    """Deployment strategies"""

    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"


class DeploymentOrchestrator:
    """Orchestrates complex deployment scenarios"""

    def __init__(self, deployment_manager: DeploymentManager):
        """Initialize orchestrator"""
        self.manager = deployment_manager
        self._active_deployments: Dict[str, List[str]] = (
            {}
        )  # app_name -> deployment_ids

    async def deploy_application(
        self,
        app_name: str,
        config: DeploymentConfig,
        strategy: DeploymentStrategy = DeploymentStrategy.RECREATE,
    ) -> Dict[str, Any]:
        """Deploy application with specified strategy"""
        logger.info(f"Deploying application {app_name} with strategy {strategy}")

        if strategy == DeploymentStrategy.RECREATE:
            return await self._deploy_recreate(app_name, config)
        elif strategy == DeploymentStrategy.BLUE_GREEN:
            return await self._deploy_blue_green(app_name, config)
        elif strategy == DeploymentStrategy.CANARY:
            return await self._deploy_canary(app_name, config)
        elif strategy == DeploymentStrategy.ROLLING:
            return await self._deploy_rolling(app_name, config)
        else:
            raise ValueError(f"Unknown deployment strategy: {strategy}")

    async def _deploy_recreate(
        self, app_name: str, config: DeploymentConfig
    ) -> Dict[str, Any]:
        """Recreate deployment strategy - terminate old and create new"""
        logger.info(f"Recreate deployment for {app_name}")

        result = {
            "app_name": app_name,
            "strategy": DeploymentStrategy.RECREATE,
            "timestamp": datetime.utcnow().isoformat(),
            "old_deployments": [],
            "new_deployments": [],
        }

        # Terminate existing deployments
        existing_ids = self._active_deployments.get(app_name, [])
        for deployment_id in existing_ids:
            try:
                state = await self.manager.terminate_deployment(deployment_id)
                result["old_deployments"].append(
                    {"id": deployment_id, "status": "terminated"}
                )
            except Exception as e:
                logger.error(f"Failed to terminate {deployment_id}: {e}")
                result["old_deployments"].append(
                    {"id": deployment_id, "status": "failed", "error": str(e)}
                )

        # Deploy new version
        try:
            new_state = await self.manager.deploy(config)
            result["new_deployments"].append(
                {"id": new_state.deployment_id, "status": new_state.status.value}
            )

            # Update active deployments
            self._active_deployments[app_name] = [new_state.deployment_id]

        except Exception as e:
            logger.error(f"Failed to deploy new version: {e}")
            result["error"] = str(e)
            result["status"] = "failed"
            return result

        result["status"] = "success"
        return result

    async def _deploy_blue_green(
        self, app_name: str, config: DeploymentConfig
    ) -> Dict[str, Any]:
        """Blue-green deployment strategy"""
        logger.info(f"Blue-green deployment for {app_name}")

        result = {
            "app_name": app_name,
            "strategy": DeploymentStrategy.BLUE_GREEN,
            "timestamp": datetime.utcnow().isoformat(),
            "blue_deployment": None,
            "green_deployment": None,
            "switched": False,
        }

        # Get current (blue) deployment
        existing_ids = self._active_deployments.get(app_name, [])
        blue_id = existing_ids[0] if existing_ids else None

        if blue_id:
            blue_state = await self.manager.get_deployment(blue_id)
            result["blue_deployment"] = {
                "id": blue_id,
                "status": blue_state.status.value if blue_state else "unknown",
            }

        # Deploy green version
        green_config = config.copy()
        green_config.name = (
            f"{config.name}-green-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        )

        try:
            green_state = await self.manager.deploy(green_config)
            result["green_deployment"] = {
                "id": green_state.deployment_id,
                "status": green_state.status.value,
            }

            # Wait for green to be ready
            logger.info("Waiting for green deployment to be ready...")
            await asyncio.sleep(30)  # Give it time to start

            # Perform health check
            health = await self.manager.health_check(green_state.deployment_id)

            if health["healthy"]:
                logger.info("Green deployment is healthy, switching traffic")

                # In a real scenario, this would update load balancer or DNS
                self._active_deployments[app_name] = [green_state.deployment_id]
                result["switched"] = True

                # Terminate blue deployment after switch
                if blue_id:
                    await asyncio.sleep(60)  # Grace period
                    await self.manager.terminate_deployment(blue_id)
                    result["blue_deployment"]["status"] = "terminated"

                result["status"] = "success"
            else:
                logger.error("Green deployment health check failed")
                # Rollback - terminate green
                await self.manager.terminate_deployment(green_state.deployment_id)
                result["status"] = "rollback"
                result["error"] = "Health check failed"

        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    async def _deploy_canary(
        self,
        app_name: str,
        config: DeploymentConfig,
        canary_percentage: int = 10,
        validation_duration_minutes: int = 30,
    ) -> Dict[str, Any]:
        """Canary deployment strategy"""
        logger.info(
            f"Canary deployment for {app_name} with {canary_percentage}% traffic"
        )

        result = {
            "app_name": app_name,
            "strategy": DeploymentStrategy.CANARY,
            "timestamp": datetime.utcnow().isoformat(),
            "stable_deployment": None,
            "canary_deployment": None,
            "canary_percentage": canary_percentage,
            "validation_duration_minutes": validation_duration_minutes,
            "promoted": False,
        }

        # Get current stable deployment
        existing_ids = self._active_deployments.get(app_name, [])
        stable_id = existing_ids[0] if existing_ids else None

        if not stable_id:
            # No existing deployment, do regular deployment
            logger.info("No existing deployment, performing regular deployment")
            return await self._deploy_recreate(app_name, config)

        stable_state = await self.manager.get_deployment(stable_id)
        result["stable_deployment"] = {
            "id": stable_id,
            "status": stable_state.status.value,
        }

        # Deploy canary version
        canary_config = config.copy()
        canary_config.name = (
            f"{config.name}-canary-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        )

        # Scale down canary for initial deployment
        if canary_config.mode == DeploymentMode.SERVERLESS:
            canary_config.scaling.max_workers = max(
                1, canary_config.scaling.max_workers * canary_percentage // 100
            )

        try:
            canary_state = await self.manager.deploy(canary_config)
            result["canary_deployment"] = {
                "id": canary_state.deployment_id,
                "status": canary_state.status.value,
            }

            # Monitor canary for validation duration
            logger.info(
                f"Monitoring canary deployment for {validation_duration_minutes} minutes"
            )

            start_time = datetime.utcnow()
            validation_passed = True

            while (
                datetime.utcnow() - start_time
            ).total_seconds() < validation_duration_minutes * 60:
                # Check canary health
                health = await self.manager.health_check(canary_state.deployment_id)

                if not health["healthy"]:
                    logger.error("Canary health check failed")
                    validation_passed = False
                    break

                # Check metrics
                metrics = await self.manager.get_deployment_metrics(
                    canary_state.deployment_id
                )
                error_rate = self._calculate_error_rate(
                    metrics["current"].get("metrics", {})
                )

                if error_rate > 0.05:  # 5% error rate threshold
                    logger.error(f"Canary error rate too high: {error_rate}")
                    validation_passed = False
                    break

                await asyncio.sleep(60)  # Check every minute

            if validation_passed:
                logger.info("Canary validation passed, promoting to stable")

                # Scale up canary to full capacity
                if canary_config.mode == DeploymentMode.SERVERLESS:
                    await self.manager.scale_deployment(
                        canary_state.deployment_id,
                        config.scaling.min_workers,
                        config.scaling.max_workers,
                    )

                # Update active deployments
                self._active_deployments[app_name] = [canary_state.deployment_id]
                result["promoted"] = True

                # Terminate old stable
                await asyncio.sleep(300)  # 5 minute grace period
                await self.manager.terminate_deployment(stable_id)
                result["stable_deployment"]["status"] = "terminated"

                result["status"] = "success"
            else:
                logger.error("Canary validation failed, rolling back")

                # Terminate canary
                await self.manager.terminate_deployment(canary_state.deployment_id)
                result["canary_deployment"]["status"] = "terminated"
                result["status"] = "rollback"

        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    async def _deploy_rolling(
        self, app_name: str, config: DeploymentConfig, batch_size: int = 1
    ) -> Dict[str, Any]:
        """Rolling deployment strategy (for multiple instances)"""
        logger.info(f"Rolling deployment for {app_name} with batch size {batch_size}")

        result = {
            "app_name": app_name,
            "strategy": DeploymentStrategy.ROLLING,
            "timestamp": datetime.utcnow().isoformat(),
            "batch_size": batch_size,
            "deployments": [],
        }

        # Get existing deployments
        existing_ids = self._active_deployments.get(app_name, [])

        if not existing_ids:
            # No existing deployments, do regular deployment
            return await self._deploy_recreate(app_name, config)

        # Process in batches
        new_deployment_ids = []

        for i in range(0, len(existing_ids), batch_size):
            batch = existing_ids[i : i + batch_size]
            batch_result = {
                "batch": i // batch_size + 1,
                "old_deployments": [],
                "new_deployments": [],
            }

            # Deploy new instances
            for j, old_id in enumerate(batch):
                try:
                    # Create new deployment
                    new_config = config.copy()
                    new_config.name = f"{config.name}-rolling-{i+j}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

                    new_state = await self.manager.deploy(new_config)
                    new_deployment_ids.append(new_state.deployment_id)

                    batch_result["new_deployments"].append(
                        {
                            "id": new_state.deployment_id,
                            "status": new_state.status.value,
                        }
                    )

                    # Wait for health check
                    await asyncio.sleep(30)
                    health = await self.manager.health_check(new_state.deployment_id)

                    if health["healthy"]:
                        # Terminate old deployment
                        await self.manager.terminate_deployment(old_id)
                        batch_result["old_deployments"].append(
                            {"id": old_id, "status": "terminated"}
                        )
                    else:
                        raise Exception("Health check failed")

                except Exception as e:
                    logger.error(f"Failed to roll deployment {old_id}: {e}")
                    batch_result["error"] = str(e)
                    result["status"] = "partial_failure"

                    # Rollback this batch
                    for new_id in batch_result["new_deployments"]:
                        await self.manager.terminate_deployment(new_id["id"])

                    return result

            result["deployments"].append(batch_result)

            # Wait between batches
            if i + batch_size < len(existing_ids):
                await asyncio.sleep(60)

        # Update active deployments
        self._active_deployments[app_name] = new_deployment_ids
        result["status"] = "success"

        return result

    async def rollback_deployment(
        self, app_name: str, to_deployment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Rollback to previous deployment"""
        logger.info(f"Rolling back deployment for {app_name}")

        result = {
            "app_name": app_name,
            "action": "rollback",
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Get current deployments
        current_ids = self._active_deployments.get(app_name, [])

        if to_deployment_id:
            # Rollback to specific deployment
            try:
                state = await self.manager.get_deployment(to_deployment_id)
                if state.status == DeploymentStatus.TERMINATED:
                    # Need to redeploy
                    new_state = await self.manager.deploy(state.config)
                    result["rolled_back_to"] = new_state.deployment_id
                else:
                    # Just restart if stopped
                    if state.status == DeploymentStatus.STOPPED:
                        await self.manager.start_deployment(to_deployment_id)
                    result["rolled_back_to"] = to_deployment_id

                # Terminate current deployments
                for current_id in current_ids:
                    if current_id != to_deployment_id:
                        await self.manager.terminate_deployment(current_id)

                self._active_deployments[app_name] = [result["rolled_back_to"]]
                result["status"] = "success"

            except Exception as e:
                logger.error(f"Rollback failed: {e}")
                result["status"] = "failed"
                result["error"] = str(e)
        else:
            result["status"] = "failed"
            result["error"] = "No deployment ID specified for rollback"

        return result

    async def deploy_environment(
        self, environment: EnvironmentType, apps: List[str]
    ) -> Dict[str, Any]:
        """Deploy multiple apps to an environment"""
        logger.info(f"Deploying {len(apps)} apps to {environment} environment")

        result = {
            "environment": environment,
            "timestamp": datetime.utcnow().isoformat(),
            "deployments": {},
        }

        # Get preset for environment
        preset = DEPLOYMENT_PRESETS.get(environment.value)
        if not preset:
            result["status"] = "failed"
            result["error"] = f"No preset found for environment {environment}"
            return result

        # Deploy each app
        for app_name in apps:
            try:
                # Customize preset for app
                app_config = preset.copy()
                app_config.name = f"{app_name}-{environment.value}"

                # Deploy with appropriate strategy
                if environment == EnvironmentType.PRODUCTION:
                    deployment_result = await self.deploy_application(
                        app_name, app_config, DeploymentStrategy.CANARY
                    )
                else:
                    deployment_result = await self.deploy_application(
                        app_name, app_config, DeploymentStrategy.RECREATE
                    )

                result["deployments"][app_name] = deployment_result

            except Exception as e:
                logger.error(f"Failed to deploy {app_name}: {e}")
                result["deployments"][app_name] = {"status": "failed", "error": str(e)}

        # Overall status
        failed = [
            app
            for app, res in result["deployments"].items()
            if res.get("status") == "failed"
        ]

        if not failed:
            result["status"] = "success"
        elif len(failed) < len(apps):
            result["status"] = "partial_success"
            result["failed_apps"] = failed
        else:
            result["status"] = "failed"

        return result

    def _calculate_error_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate error rate from metrics"""
        total = metrics.get("requests_total", 0)
        failed = metrics.get("requests_failed", 0)

        if total == 0:
            return 0.0

        return failed / total

    async def schedule_deployment(
        self,
        app_name: str,
        config: DeploymentConfig,
        schedule_time: datetime,
        strategy: DeploymentStrategy = DeploymentStrategy.RECREATE,
    ) -> Dict[str, Any]:
        """Schedule a deployment for future execution"""
        delay = (schedule_time - datetime.utcnow()).total_seconds()

        if delay <= 0:
            # Execute immediately
            return await self.deploy_application(app_name, config, strategy)

        logger.info(f"Scheduling deployment of {app_name} in {delay} seconds")

        # Create scheduled task
        async def scheduled_deploy():
            await asyncio.sleep(delay)
            return await self.deploy_application(app_name, config, strategy)

        task = asyncio.create_task(scheduled_deploy())

        return {
            "app_name": app_name,
            "scheduled_time": schedule_time.isoformat(),
            "strategy": strategy.value,
            "status": "scheduled",
            "task_id": id(task),
        }
