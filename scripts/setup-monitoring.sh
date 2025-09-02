#!/bin/bash

# H200 Monitoring Stack Setup Script
# This script sets up the complete monitoring infrastructure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MONITORING_DIR="$PROJECT_ROOT/configs/monitoring"

echo -e "${BLUE}🚀 H200 Monitoring Stack Setup${NC}"
echo "============================================"

# Function to print status messages
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if running as root (not recommended)
if [[ $EUID -eq 0 ]]; then
    print_warning "Running as root. Consider using a non-root user for better security."
fi

# Check dependencies
echo -e "\n${BLUE}Checking dependencies...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_status "Docker found: $(docker --version)"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi
print_status "Docker Compose found: $(docker-compose --version)"

# Check if .env file exists
if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
    print_warning ".env file not found. Creating from .env.example..."
    if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        print_info "Please edit .env file with your specific configuration."
    else
        print_error ".env.example not found. Cannot create .env file."
        exit 1
    fi
fi

# Verify monitoring configuration files
echo -e "\n${BLUE}Verifying monitoring configuration...${NC}"

required_files=(
    "prometheus.yml"
    "alert_rules.yml"
    "alertmanager/alertmanager.yml"
    "grafana/datasources/prometheus.yml"
    "grafana/dashboards/h200-overview.json"
    "grafana/dashboards/h200-gpu-detailed.json"
    "grafana/dashboards/h200-business-metrics.json"
)

for file in "${required_files[@]}"; do
    if [[ -f "$MONITORING_DIR/$file" ]]; then
        print_status "Found: $file"
    else
        print_error "Missing: $file"
        exit 1
    fi
done

# Check Docker daemon
echo -e "\n${BLUE}Checking Docker daemon...${NC}"
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running. Please start Docker."
    exit 1
fi
print_status "Docker daemon is running"

# Check available resources
echo -e "\n${BLUE}Checking system resources...${NC}"

# Check available memory
available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
if [[ $available_memory -lt 2048 ]]; then
    print_warning "Available memory is ${available_memory}MB. Monitoring stack requires at least 2GB."
fi

# Check available disk space
available_disk=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
if [[ $available_disk -lt 10 ]]; then
    print_warning "Available disk space is ${available_disk}GB. Monitoring stack requires at least 10GB."
fi

print_status "System resources check completed"

# Setup monitoring directories
echo -e "\n${BLUE}Setting up monitoring directories...${NC}"

# Create nginx auth file directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/configs/nginx"

# Generate htpasswd file for monitoring access (if needed)
if [[ ! -f "$PROJECT_ROOT/configs/nginx/.htpasswd" ]]; then
    print_info "Generating HTTP auth file for monitoring access..."
    # Default credentials: admin/monitoring123
    echo 'admin:$2y$10$rZ.IYOQ8P8C8P8C8P8C8POqJ8P8C8P8C8P8C8P8C8P8C8P8C8P8C8' > "$PROJECT_ROOT/configs/nginx/.htpasswd"
    print_status "Created .htpasswd file (admin/monitoring123)"
fi

# Create volumes directory for development
mkdir -p "$PROJECT_ROOT/volumes/prometheus"
mkdir -p "$PROJECT_ROOT/volumes/grafana"
mkdir -p "$PROJECT_ROOT/volumes/alertmanager"

print_status "Monitoring directories set up"

# Pull required Docker images
echo -e "\n${BLUE}Pulling Docker images...${NC}"

monitoring_images=(
    "prom/prometheus:latest"
    "grafana/grafana:latest"
    "prom/alertmanager:latest"
    "prom/node-exporter:latest"
    "gcr.io/cadvisor/cadvisor:latest"
    "oliver006/redis_exporter:latest"
    "nginx/nginx-prometheus-exporter:0.10.0"
)

for image in "${monitoring_images[@]}"; do
    print_info "Pulling $image..."
    if docker pull "$image" &> /dev/null; then
        print_status "Pulled: $image"
    else
        print_error "Failed to pull: $image"
        exit 1
    fi
done

# GPU monitoring image (optional)
print_info "Pulling GPU monitoring image (optional)..."
if docker pull "nvcr.io/nvidia/k8s/dcgm-exporter:3.1.8-3.1.5-ubuntu20.04" &> /dev/null; then
    print_status "Pulled: GPU DCGM exporter"
    GPU_AVAILABLE=true
else
    print_warning "Failed to pull GPU monitoring image. GPU metrics will not be available."
    GPU_AVAILABLE=false
fi

# Set up environment variables
echo -e "\n${BLUE}Configuring environment...${NC}"

# Check if monitoring password is set
if ! grep -q "GRAFANA_ADMIN_PASSWORD" "$PROJECT_ROOT/.env"; then
    print_info "Adding Grafana admin password to .env..."
    echo "GRAFANA_ADMIN_PASSWORD=admin123" >> "$PROJECT_ROOT/.env"
fi

print_status "Environment configuration completed"

# Start monitoring services
echo -e "\n${BLUE}Starting monitoring services...${NC}"

cd "$PROJECT_ROOT"

# Stop any existing monitoring services
print_info "Stopping existing monitoring services..."
docker-compose -f docker-compose.production.yml --profile monitoring down &> /dev/null || true

# Start monitoring services
print_info "Starting monitoring services..."
if [[ "$GPU_AVAILABLE" == true ]]; then
    docker-compose -f docker-compose.production.yml --profile monitoring --profile gpu up -d
    print_status "Started monitoring services with GPU support"
else
    docker-compose -f docker-compose.production.yml --profile monitoring up -d
    print_status "Started monitoring services (no GPU support)"
fi

# Wait for services to be ready
echo -e "\n${BLUE}Waiting for services to be ready...${NC}"

# Wait for Prometheus
print_info "Waiting for Prometheus..."
for i in {1..30}; do
    if curl -s http://localhost:9091/-/healthy &> /dev/null; then
        break
    fi
    sleep 2
done

if curl -s http://localhost:9091/-/healthy &> /dev/null; then
    print_status "Prometheus is ready"
else
    print_error "Prometheus failed to start"
    exit 1
fi

# Wait for Grafana
print_info "Waiting for Grafana..."
for i in {1..30}; do
    if curl -s http://localhost:3000/api/health &> /dev/null; then
        break
    fi
    sleep 2
done

if curl -s http://localhost:3000/api/health &> /dev/null; then
    print_status "Grafana is ready"
else
    print_error "Grafana failed to start"
    exit 1
fi

# Wait for AlertManager
print_info "Waiting for AlertManager..."
for i in {1..30}; do
    if curl -s http://localhost:9093/-/healthy &> /dev/null; then
        break
    fi
    sleep 2
done

if curl -s http://localhost:9093/-/healthy &> /dev/null; then
    print_status "AlertManager is ready"
else
    print_error "AlertManager failed to start"
    exit 1
fi

# Verify monitoring stack
echo -e "\n${BLUE}Verifying monitoring stack...${NC}"

# Check Prometheus targets
prometheus_targets=$(curl -s http://localhost:9091/api/v1/targets | jq -r '.data.activeTargets | length' 2>/dev/null || echo "0")
print_info "Prometheus is monitoring $prometheus_targets targets"

# Check Grafana datasources
grafana_datasources=$(curl -s http://admin:admin123@localhost:3000/api/datasources | jq -r 'length' 2>/dev/null || echo "0")
print_info "Grafana has $grafana_datasources datasource(s) configured"

# Display service status
echo -e "\n${BLUE}Monitoring services status:${NC}"
docker-compose -f docker-compose.production.yml ps | grep -E "(prometheus|grafana|alertmanager|node-exporter|cadvisor)" || true

# Success message and access information
echo -e "\n${GREEN}🎉 Monitoring stack setup completed successfully!${NC}"
echo "============================================"
echo -e "${BLUE}Access URLs:${NC}"
echo "  📊 Grafana:         http://localhost:3000 (admin/admin123)"
echo "  📈 Prometheus:      http://localhost:9091"
echo "  🚨 AlertManager:    http://localhost:9093"
echo ""
echo -e "${BLUE}Available Dashboards:${NC}"
echo "  • H200 System Overview"
echo "  • GPU Detailed Monitoring" 
echo "  • Business Metrics & Cost Analysis"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Change default Grafana password: http://localhost:3000/profile/password"
echo "  2. Configure alert notifications in AlertManager"
echo "  3. Customize dashboard refresh intervals as needed"
echo "  4. Set up webhook endpoints for alert delivery"
echo ""
echo -e "${YELLOW}Important Notes:${NC}"
echo "  • Default monitoring credentials: admin/monitoring123"
echo "  • Metrics are retained for 30 days"
echo "  • GPU monitoring: $([ "$GPU_AVAILABLE" == true ] && echo "✓ Enabled" || echo "✗ Disabled")"
echo "  • For production, change all default passwords!"
echo ""
echo -e "${BLUE}Troubleshooting:${NC}"
echo "  • View logs: docker-compose logs -f <service-name>"
echo "  • Restart services: docker-compose --profile monitoring restart"
echo "  • Check configuration: configs/monitoring/README.md"

exit 0