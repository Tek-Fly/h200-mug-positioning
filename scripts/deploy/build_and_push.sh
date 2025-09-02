#!/bin/bash
# Build and push Docker images for H200 Intelligent Mug Positioning System
# Supports multi-platform builds with Docker Build Cloud

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
DOCKER_BUILD_CLOUD_TOKEN="${DOCKER_BUILD_CLOUD_TOKEN:-}"
REGISTRY="${REGISTRY:-docker.io}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
BUILD_CACHE_TAG="buildcache"

# Function to print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_color $BLUE "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_color $RED "Error: Docker is not installed"
        exit 1
    fi
    
    # Check if buildx is available
    if ! docker buildx version &> /dev/null; then
        print_color $RED "Error: Docker buildx is not available"
        exit 1
    fi
    
    # Check environment variables
    if [[ -z "$DOCKER_USERNAME" ]]; then
        print_color $RED "Error: DOCKER_USERNAME is not set"
        exit 1
    fi
    
    # Load .env file if exists
    if [[ -f "${PROJECT_ROOT}/.env" ]]; then
        print_color $YELLOW "Loading environment from .env file..."
        export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
    fi
    
    print_color $GREEN "Prerequisites check passed"
}

# Function to setup buildx
setup_buildx() {
    print_color $BLUE "Setting up Docker buildx..."
    
    # Create a new builder instance
    if ! docker buildx inspect h200-builder &> /dev/null; then
        docker buildx create --name h200-builder --driver docker-container --use
    else
        docker buildx use h200-builder
    fi
    
    # Start the builder
    docker buildx inspect --bootstrap
    
    # Setup Docker Build Cloud if token is provided
    if [[ -n "$DOCKER_BUILD_CLOUD_TOKEN" ]]; then
        print_color $YELLOW "Setting up Docker Build Cloud..."
        docker buildx create --use --driver cloud "cloud-${DOCKER_USERNAME}/h200-builder" || true
    fi
    
    print_color $GREEN "Buildx setup completed"
}

# Function to login to registry
docker_login() {
    print_color $BLUE "Logging into Docker registry..."
    
    if [[ -n "${DOCKER_PASSWORD:-}" ]]; then
        echo "$DOCKER_PASSWORD" | docker login "$REGISTRY" -u "$DOCKER_USERNAME" --password-stdin
    else
        docker login "$REGISTRY" -u "$DOCKER_USERNAME"
    fi
    
    print_color $GREEN "Successfully logged into registry"
}

# Function to build and push an image
build_and_push() {
    local dockerfile=$1
    local tag=$2
    local build_args="${3:-}"
    
    print_color $BLUE "Building ${tag}..."
    
    # Construct the full image name
    local full_tag="${REGISTRY}/${DOCKER_USERNAME}/h200-mug-positioning:${tag}"
    local cache_tag="${REGISTRY}/${DOCKER_USERNAME}/h200-mug-positioning:${BUILD_CACHE_TAG}-${tag}"
    
    # Build command with caching
    local build_cmd="docker buildx build \
        --platform ${PLATFORMS} \
        --file ${dockerfile} \
        --tag ${full_tag} \
        --cache-from type=registry,ref=${cache_tag} \
        --cache-to type=registry,ref=${cache_tag},mode=max \
        --push"
    
    # Add build args if provided
    if [[ -n "$build_args" ]]; then
        build_cmd="${build_cmd} ${build_args}"
    fi
    
    # Add build context
    build_cmd="${build_cmd} ${PROJECT_ROOT}"
    
    # Execute build
    print_color $YELLOW "Executing: ${build_cmd}"
    eval $build_cmd
    
    print_color $GREEN "Successfully built and pushed ${tag}"
}

# Function to build all images
build_all_images() {
    print_color $BLUE "Building all Docker images..."
    
    # Build base image first
    build_and_push \
        "${PROJECT_ROOT}/docker/Dockerfile.base" \
        "base-latest" \
        ""
    
    # Tag base image with timestamp
    local timestamp=$(date +%Y%m%d-%H%M%S)
    docker buildx imagetools create \
        "${REGISTRY}/${DOCKER_USERNAME}/h200-mug-positioning:base-latest" \
        --tag "${REGISTRY}/${DOCKER_USERNAME}/h200-mug-positioning:base-${timestamp}"
    
    # Build serverless image
    build_and_push \
        "${PROJECT_ROOT}/docker/Dockerfile.serverless" \
        "serverless-latest" \
        "--build-arg BASE_IMAGE=${REGISTRY}/${DOCKER_USERNAME}/h200-mug-positioning:base-latest"
    
    # Build timed image
    build_and_push \
        "${PROJECT_ROOT}/docker/Dockerfile.timed" \
        "timed-latest" \
        "--build-arg BASE_IMAGE=${REGISTRY}/${DOCKER_USERNAME}/h200-mug-positioning:base-latest"
    
    # Tag images with timestamp
    for image in serverless timed; do
        docker buildx imagetools create \
            "${REGISTRY}/${DOCKER_USERNAME}/h200-mug-positioning:${image}-latest" \
            --tag "${REGISTRY}/${DOCKER_USERNAME}/h200-mug-positioning:${image}-${timestamp}"
    done
    
    print_color $GREEN "All images built and pushed successfully"
    print_color $YELLOW "Images tagged with timestamp: ${timestamp}"
}

# Function to verify images
verify_images() {
    print_color $BLUE "Verifying pushed images..."
    
    for tag in base-latest serverless-latest timed-latest; do
        local full_tag="${REGISTRY}/${DOCKER_USERNAME}/h200-mug-positioning:${tag}"
        
        # Check if image exists in registry
        if docker manifest inspect "${full_tag}" &> /dev/null; then
            print_color $GREEN "✓ ${full_tag} exists in registry"
            
            # Show manifest details
            docker buildx imagetools inspect "${full_tag}" | grep -E "(Platform|Digest):" | head -6
        else
            print_color $RED "✗ ${full_tag} not found in registry"
            return 1
        fi
    done
    
    print_color $GREEN "All images verified successfully"
}

# Function to cleanup
cleanup() {
    print_color $BLUE "Cleaning up..."
    
    # Prune build cache (optional)
    if [[ "${PRUNE_CACHE:-false}" == "true" ]]; then
        docker buildx prune -f
    fi
    
    print_color $GREEN "Cleanup completed"
}

# Main execution
main() {
    print_color $BLUE "=== H200 Docker Build and Push Script ==="
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --platforms)
                PLATFORMS="$2"
                shift 2
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --prune-cache)
                PRUNE_CACHE=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --platforms PLATFORMS   Comma-separated list of platforms (default: linux/amd64,linux/arm64)"
                echo "  --registry REGISTRY     Docker registry (default: docker.io)"
                echo "  --prune-cache          Prune build cache after build"
                echo "  --help                 Show this help message"
                exit 0
                ;;
            *)
                print_color $RED "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute build steps
    check_prerequisites
    setup_buildx
    docker_login
    build_all_images
    verify_images
    cleanup
    
    print_color $GREEN "=== Build and push completed successfully ==="
    print_color $YELLOW "Images are available at: ${REGISTRY}/${DOCKER_USERNAME}/h200-mug-positioning"
}

# Run main function
main "$@"