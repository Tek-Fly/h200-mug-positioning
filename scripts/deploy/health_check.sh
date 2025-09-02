#!/bin/bash
# Health check script for H200 deployments

set -euo pipefail

# Default values
ENDPOINT="${HEALTH_ENDPOINT:-http://localhost:8000/api/v1/health}"
TIMEOUT="${HEALTH_TIMEOUT:-10}"
RETRIES="${HEALTH_RETRIES:-3}"
INTERVAL="${HEALTH_INTERVAL:-5}"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check health
check_health() {
    local attempt=1
    
    while [ $attempt -le $RETRIES ]; do
        echo -e "${YELLOW}Health check attempt $attempt/$RETRIES...${NC}"
        
        if curl -sf --max-time $TIMEOUT "$ENDPOINT" > /dev/null; then
            echo -e "${GREEN}✓ Service is healthy${NC}"
            
            # Get detailed health info
            HEALTH_INFO=$(curl -s "$ENDPOINT" 2>/dev/null || echo "{}")
            echo "Health details: $HEALTH_INFO"
            
            return 0
        else
            echo -e "${RED}✗ Health check failed${NC}"
            
            if [ $attempt -lt $RETRIES ]; then
                echo "Waiting ${INTERVAL}s before retry..."
                sleep $INTERVAL
            fi
        fi
        
        ((attempt++))
    done
    
    echo -e "${RED}Service is unhealthy after $RETRIES attempts${NC}"
    return 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --endpoint)
            ENDPOINT="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --retries)
            RETRIES="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --endpoint URL    Health check endpoint (default: http://localhost:8000/api/v1/health)"
            echo "  --timeout SECS    Request timeout in seconds (default: 10)"
            echo "  --retries NUM     Number of retries (default: 3)"
            echo "  --interval SECS   Interval between retries (default: 5)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run health check
echo "Checking health at: $ENDPOINT"
check_health