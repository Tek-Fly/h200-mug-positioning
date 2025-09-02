#!/usr/bin/env python3
"""
Example script demonstrating the deployment module usage.

This script shows how to:
1. Deploy a serverless endpoint
2. Deploy a timed GPU instance
3. Manage deployment lifecycle
4. Use deployment orchestration strategies
"""

# Standard library imports
import asyncio
import logging
import os
from datetime import datetime

# First-party imports
from src.deployment import (
    DEPLOYMENT_PRESETS,
    DeploymentConfig,
    DeploymentManager,
    DeploymentMode,
    DeploymentOrchestrator,
    DeploymentStrategy,
    EnvironmentType,
    GPUConfig,
    GPUType,
    ResourceConfig,
    ScalingConfig,
)
from src.utils.secrets import get_secret

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_deployment_example():
    """Example of basic deployment operations."""
    print("\n=== Basic Deployment Example ===\n")

    # Get API key
    api_key = get_secret("RUNPOD_API_KEY")

    # Initialize manager
    manager = DeploymentManager(api_key)
    await manager.initialize()

    try:
        # Create serverless deployment config
        config = DeploymentConfig(
            name="h200-mug-positioning-demo",
            mode=DeploymentMode.SERVERLESS,
            environment=EnvironmentType.DEVELOPMENT,
            docker_image=os.getenv("DOCKER_USERNAME", "myuser")
            + "/h200-mug-positioning",
            docker_tag="latest",
            gpu=GPUConfig(type=GPUType.RTX3090, count=1),  # Use cheaper GPU for demo
            resources=ResourceConfig(
                cpu_cores=4,
                memory_gb=16,
                container_disk_gb=20,
                volume_size_gb=30,
            ),
            scaling=ScalingConfig(
                min_workers=0,
                max_workers=2,
                target_requests_per_second=5,
            ),
            env_vars={
                "MONGODB_ATLAS_URI": get_secret("MONGODB_ATLAS_URI"),
                "REDIS_HOST": get_secret("REDIS_HOST"),
                "REDIS_PASSWORD": get_secret("REDIS_PASSWORD"),
                "JWT_SECRET": get_secret("JWT_SECRET"),
                "LOG_LEVEL": "DEBUG",
            },
            idle_timeout_minutes=5,  # Short timeout for demo
        )

        # Deploy
        print("Deploying serverless endpoint...")
        state = await manager.deploy(config)
        print(f"✓ Deployed: {state.deployment_id}")
        print(f"  Status: {state.status}")
        print(f"  Mode: {state.mode}")

        # Wait a bit
        await asyncio.sleep(10)

        # Check health
        print("\nPerforming health check...")
        health = await manager.health_check(state.deployment_id)
        print(f"✓ Health check: {'Healthy' if health['healthy'] else 'Unhealthy'}")
        for check, result in health["checks"].items():
            print(f"  - {check}: {'✓' if result['passed'] else '✗'}")

        # Get metrics
        print("\nFetching metrics...")
        metrics = await manager.get_deployment_metrics(state.deployment_id)
        print(f"✓ Metrics retrieved")
        if "current" in metrics:
            print(f"  Workers: {metrics['current'].get('workers', {})}")

        # Scale deployment
        print("\nScaling deployment...")
        await manager.scale_deployment(
            state.deployment_id, min_workers=1, max_workers=3
        )
        print("✓ Scaled to min=1, max=3")

        # Get costs
        print("\nCalculating costs...")
        costs = await manager.get_deployment_costs(state.deployment_id)
        print(f"✓ Estimated cost: ${costs['costs']['total']:.2f}")

        # Stop deployment
        print("\nStopping deployment...")
        await manager.stop_deployment(state.deployment_id)
        print("✓ Deployment stopped")

    finally:
        await manager.shutdown()


async def orchestration_example():
    """Example of deployment orchestration strategies."""
    print("\n=== Orchestration Example ===\n")

    # Get API key
    api_key = get_secret("RUNPOD_API_KEY")

    # Initialize manager and orchestrator
    manager = DeploymentManager(api_key)
    await manager.initialize()

    orchestrator = DeploymentOrchestrator(manager)

    try:
        # Use a preset configuration
        config = DEPLOYMENT_PRESETS["development"]
        config.docker_image = (
            os.getenv("DOCKER_USERNAME", "myuser") + "/h200-mug-positioning"
        )
        config.env_vars = {
            "MONGODB_ATLAS_URI": get_secret("MONGODB_ATLAS_URI"),
            "REDIS_HOST": get_secret("REDIS_HOST"),
            "REDIS_PASSWORD": get_secret("REDIS_PASSWORD"),
            "JWT_SECRET": get_secret("JWT_SECRET"),
        }

        # Deploy with blue-green strategy
        print("Deploying with blue-green strategy...")
        result = await orchestrator.deploy_application(
            app_name="mug-positioning-demo",
            config=config,
            strategy=DeploymentStrategy.BLUE_GREEN,
        )

        print(f"✓ Blue-green deployment: {result['status']}")
        if result.get("blue_deployment"):
            print(f"  Blue: {result['blue_deployment']}")
        if result.get("green_deployment"):
            print(f"  Green: {result['green_deployment']}")
        print(f"  Switched: {result.get('switched', False)}")

    finally:
        await manager.shutdown()


async def volume_management_example():
    """Example of volume management."""
    print("\n=== Volume Management Example ===\n")

    # Get API key
    api_key = get_secret("RUNPOD_API_KEY")

    # Initialize manager
    manager = DeploymentManager(api_key)
    await manager.initialize()

    try:
        # Ensure volume exists
        print("Ensuring model storage volume...")
        volume_id = await manager.volume_manager.ensure_volume(
            name="h200-model-storage-demo",
            size_gb=50,
            labels={"purpose": "models", "env": "demo"},
        )
        print(f"✓ Volume ready: {volume_id}")

        # Get volume info
        info = await manager.volume_manager.get_volume_info(volume_id)
        print(f"  Size: {info.get('size', 0)}GB")
        print(f"  Usage: {info.get('usage_percent', 0)}%")

        # List volumes
        print("\nListing all volumes...")
        volumes = await manager.volume_manager.list_volumes()
        print(f"✓ Found {len(volumes)} volumes")
        for vol in volumes[:3]:  # Show first 3
            print(f"  - {vol['name']}: {vol.get('size', 0)}GB")

    finally:
        await manager.shutdown()


async def main():
    """Run all examples."""
    print("H200 Deployment Module Examples")
    print("================================")

    try:
        # Run basic deployment example
        await basic_deployment_example()

        # Run orchestration example
        # await orchestration_example()

        # Run volume management example
        # await volume_management_example()

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
