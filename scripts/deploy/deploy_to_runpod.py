#!/usr/bin/env python3
"""
Deploy H200 Intelligent Mug Positioning System to RunPod
Handles both serverless and timed GPU deployments

This script demonstrates how to use the deployment module from the command line.
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.deployment import (
    DeploymentManager,
    DeploymentConfig,
    DeploymentMode,
    DeploymentOrchestrator,
    DeploymentStrategy,
    GPUConfig,
    GPUType,
    ResourceConfig,
    ScalingConfig,
    EnvironmentType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables"""
    # Load from .env file
    env_path = os.path.join(os.path.dirname(__file__), '../../.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    
    # Required environment variables
    required_vars = [
        "RUNPOD_API_KEY",
        "DOCKER_USERNAME",
        "MONGODB_ATLAS_URI",
        "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "REDIS_HOST",
        "REDIS_PASSWORD",
        "GCP_PROJECT_ID",
        "JWT_SECRET"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        sys.exit(1)


def prepare_env_vars() -> Dict[str, str]:
    """Prepare environment variables for deployment"""
    return {
        "MONGODB_ATLAS_URI": os.getenv("MONGODB_ATLAS_URI"),
        "R2_ENDPOINT_URL": os.getenv("R2_ENDPOINT_URL"),
        "R2_ACCESS_KEY_ID": os.getenv("R2_ACCESS_KEY_ID"),
        "R2_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY"),
        "R2_BUCKET_NAME": os.getenv("R2_BUCKET_NAME", "h200-backup"),
        "REDIS_HOST": os.getenv("REDIS_HOST"),
        "REDIS_PORT": os.getenv("REDIS_PORT", "6379"),
        "REDIS_PASSWORD": os.getenv("REDIS_PASSWORD"),
        "GCP_PROJECT_ID": os.getenv("GCP_PROJECT_ID"),
        "JWT_SECRET": os.getenv("JWT_SECRET"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "WEBHOOK_URL": os.getenv("WEBHOOK_URL", ""),
        "N8N_WEBHOOK_URL": os.getenv("N8N_WEBHOOK_URL", ""),
        "TEMPLATED_API_KEY": os.getenv("TEMPLATED_API_KEY", "")
    }


async def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy H200 system to RunPod")
    parser.add_argument(
        "mode",
        choices=["serverless", "timed", "both"],
        help="Deployment mode"
    )
    parser.add_argument(
        "--gpu-type",
        default="H100",
        choices=["H100", "A100", "RTX4090", "RTX3090"],
        help="GPU type"
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="Number of GPUs"
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=0,
        help="Minimum workers for serverless"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum workers for serverless"
    )
    parser.add_argument(
        "--strategy",
        default="recreate",
        choices=["recreate", "blue_green", "canary"],
        help="Deployment strategy"
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for deployment to be ready"
    )
    parser.add_argument(
        "--tag",
        default="latest",
        help="Docker image tag"
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_environment()
    
    # Initialize deployment manager
    api_key = os.getenv("RUNPOD_API_KEY")
    docker_username = os.getenv("DOCKER_USERNAME")
    
    manager = DeploymentManager(api_key)
    await manager.initialize()
    
    try:
        # Prepare deployment configurations
        env_vars = prepare_env_vars()
        
        # Map GPU type string to enum
        gpu_type_map = {
            "H100": GPUType.H100,
            "A100": GPUType.A100,
            "RTX4090": GPUType.RTX4090,
            "RTX3090": GPUType.RTX3090,
        }
        gpu_type = gpu_type_map[args.gpu_type]
        
        deployments = []
        
        # Deploy serverless
        if args.mode in ["serverless", "both"]:
            config = DeploymentConfig(
                name="h200-mug-positioning-serverless",
                mode=DeploymentMode.SERVERLESS,
                environment=EnvironmentType.PRODUCTION,
                docker_image=f"{docker_username}/h200-mug-positioning",
                docker_tag=f"serverless-{args.tag}",
                gpu=GPUConfig(type=gpu_type, count=args.gpu_count),
                resources=ResourceConfig(
                    cpu_cores=8,
                    memory_gb=32,
                    container_disk_gb=50,
                    volume_size_gb=100,
                ),
                scaling=ScalingConfig(
                    min_workers=args.min_workers,
                    max_workers=args.max_workers,
                    target_requests_per_second=20,
                ),
                env_vars=env_vars,
            )
            
            # Deploy with strategy
            if args.strategy != "recreate":
                orchestrator = DeploymentOrchestrator(manager)
                strategy_map = {
                    "blue_green": DeploymentStrategy.BLUE_GREEN,
                    "canary": DeploymentStrategy.CANARY,
                }
                result = await orchestrator.deploy_application(
                    app_name="h200-serverless",
                    config=config,
                    strategy=strategy_map[args.strategy]
                )
                logger.info(f"Serverless deployment result: {result}")
            else:
                state = await manager.deploy(config)
                deployments.append(("serverless", state.deployment_id))
                
                # Save endpoint ID
                with open(".runpod_serverless_id", "w") as f:
                    f.write(state.deployment_id)
        
        # Deploy timed
        if args.mode in ["timed", "both"]:
            config = DeploymentConfig(
                name="h200-mug-positioning-timed",
                mode=DeploymentMode.TIMED,
                environment=EnvironmentType.PRODUCTION,
                docker_image=f"{docker_username}/h200-mug-positioning",
                docker_tag=f"timed-{args.tag}",
                gpu=GPUConfig(type=gpu_type, count=args.gpu_count),
                resources=ResourceConfig(
                    cpu_cores=8,
                    memory_gb=32,
                    container_disk_gb=50,
                    volume_size_gb=100,
                    network_volume_size_gb=200,  # For model storage
                ),
                env_vars=env_vars,
                idle_timeout_minutes=10,
            )
            
            state = await manager.deploy(config)
            deployments.append(("timed", state.deployment_id))
            
            # Save pod ID
            with open(".runpod_timed_id", "w") as f:
                f.write(state.deployment_id)
        
        # Wait for deployments to be ready
        if not args.no_wait and deployments:
            logger.info("Waiting for deployments to be ready...")
            await asyncio.sleep(10)  # Initial wait
            
            for mode, deployment_id in deployments:
                # Perform health check
                health = await manager.health_check(deployment_id)
                if health["healthy"]:
                    logger.info(f"{mode} deployment is healthy")
                    
                    # Get metrics
                    metrics = await manager.get_deployment_metrics(deployment_id)
                    logger.info(f"{mode} deployment metrics: {json.dumps(metrics['current'], indent=2)}")
                else:
                    logger.error(f"{mode} deployment failed health check")
                    logger.error(f"Health checks: {json.dumps(health['checks'], indent=2)}")
                    sys.exit(1)
        
        # Print summary
        logger.info("\n=== Deployment Summary ===")
        for mode, deployment_id in deployments:
            logger.info(f"{mode.capitalize()}: {deployment_id}")
            
            # Get deployment info
            state = await manager.get_deployment(deployment_id)
            if state:
                metrics = state.metrics
                if mode == "serverless" and "endpoint_url" in metrics:
                    logger.info(f"  Endpoint URL: {metrics['endpoint_url']}")
                elif mode == "timed" and "pod_ip" in metrics:
                    logger.info(f"  Pod IP: {metrics['pod_ip']}")
                
                # Get cost estimate
                costs = await manager.get_deployment_costs(deployment_id)
                logger.info(f"  Estimated hourly cost: ${costs['breakdown']['hourly_rate']:.2f}")
        
        logger.info("\nDeployment completed successfully!")
        
    finally:
        await manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())