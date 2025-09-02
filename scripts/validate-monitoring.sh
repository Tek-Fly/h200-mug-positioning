#!/bin/bash

# H200 Monitoring Stack Validation Script
# Validates monitoring configuration files and connectivity

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

echo -e "${BLUE}🔍 H200 Monitoring Stack Validation${NC}"
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

# Validation counters
errors=0
warnings=0

# Function to increment error counter
report_error() {
    print_error "$1"
    ((errors++))
}

# Function to increment warning counter
report_warning() {
    print_warning "$1"
    ((warnings++))
}

# Validate configuration files
echo -e "\n${BLUE}Validating configuration files...${NC}"

# Check Prometheus configuration
if [[ -f "$MONITORING_DIR/prometheus.yml" ]]; then
    if command -v promtool &> /dev/null; then
        if promtool check config "$MONITORING_DIR/prometheus.yml" &> /dev/null; then
            print_status "Prometheus configuration is valid"
        else
            report_error "Prometheus configuration validation failed"
        fi
    else
        print_info "promtool not available, skipping Prometheus config validation"
    fi
else
    report_error "Prometheus configuration file missing"
fi

# Check alert rules
if [[ -f "$MONITORING_DIR/alert_rules.yml" ]]; then
    if command -v promtool &> /dev/null; then
        if promtool check rules "$MONITORING_DIR/alert_rules.yml" &> /dev/null; then
            print_status "Alert rules are valid"
        else
            report_error "Alert rules validation failed"
        fi
    else
        print_info "promtool not available, skipping alert rules validation"
    fi
else
    report_error "Alert rules file missing"
fi

# Check AlertManager configuration
if [[ -f "$MONITORING_DIR/alertmanager/alertmanager.yml" ]]; then
    if command -v amtool &> /dev/null; then
        if amtool check-config "$MONITORING_DIR/alertmanager/alertmanager.yml" &> /dev/null; then
            print_status "AlertManager configuration is valid"
        else
            report_error "AlertManager configuration validation failed"
        fi
    else
        print_info "amtool not available, skipping AlertManager config validation"
    fi
else
    report_error "AlertManager configuration file missing"
fi

# Check Grafana datasource configuration
if [[ -f "$MONITORING_DIR/grafana/datasources/prometheus.yml" ]]; then
    print_status "Grafana datasource configuration found"
else
    report_error "Grafana datasource configuration missing"
fi

# Check dashboard files
dashboard_files=(
    "h200-overview.json"
    "h200-gpu-detailed.json"
    "h200-business-metrics.json"
)

for dashboard in "${dashboard_files[@]}"; do
    if [[ -f "$MONITORING_DIR/grafana/dashboards/$dashboard" ]]; then
        # Basic JSON validation
        if jq empty "$MONITORING_DIR/grafana/dashboards/$dashboard" &> /dev/null; then
            print_status "Dashboard $dashboard is valid JSON"
        else
            report_error "Dashboard $dashboard contains invalid JSON"
        fi
    else
        report_error "Dashboard $dashboard missing"
    fi
done

# Check Docker Compose configuration
echo -e "\n${BLUE}Validating Docker Compose configuration...${NC}"

cd "$PROJECT_ROOT"

# Check if docker-compose.yml syntax is valid
if docker-compose -f docker-compose.production.yml config &> /dev/null; then
    print_status "Docker Compose configuration is valid"
else
    report_error "Docker Compose configuration has errors"
fi

# Check if monitoring services are defined
monitoring_services=(
    "prometheus"
    "grafana"
    "alertmanager"
    "node-exporter"
    "cadvisor"
    "redis-exporter"
    "nginx-prometheus-exporter"
)

for service in "${monitoring_services[@]}"; do
    if docker-compose -f docker-compose.production.yml config --services | grep -q "^$service$"; then
        print_status "Service $service is defined"
    else
        report_warning "Service $service is not defined"
    fi
done

# Check for GPU monitoring service
if docker-compose -f docker-compose.production.yml config --services | grep -q "^nvidia-dcgm-exporter$"; then
    print_status "GPU monitoring service is defined"
else
    report_warning "GPU monitoring service is not defined (optional)"
fi

# Check network connectivity (if services are running)
echo -e "\n${BLUE}Checking service connectivity...${NC}"

services_to_check=(
    "prometheus:9091:Prometheus"
    "grafana:3000:Grafana"
    "alertmanager:9093:AlertManager"
)

for service_check in "${services_to_check[@]}"; do
    IFS=':' read -r service port name <<< "$service_check"
    
    if docker-compose -f docker-compose.production.yml ps | grep -q "$service.*Up"; then
        if curl -s "http://localhost:$port/api/health" &> /dev/null || curl -s "http://localhost:$port/-/healthy" &> /dev/null; then
            print_status "$name is running and accessible"
        else
            report_warning "$name is running but not responding to health checks"
        fi
    else
        print_info "$name is not currently running"
    fi
done

# Check environment variables
echo -e "\n${BLUE}Checking environment variables...${NC}"

required_env_vars=(
    "GRAFANA_ADMIN_PASSWORD"
    "WEBHOOK_URL"
    "REDIS_PASSWORD"
)

for var in "${required_env_vars[@]}"; do
    if grep -q "^$var=" "$PROJECT_ROOT/.env" 2>/dev/null; then
        print_status "Environment variable $var is set"
    else
        report_warning "Environment variable $var is not set in .env"
    fi
done

# Check ports availability
echo -e "\n${BLUE}Checking port availability...${NC}"

monitoring_ports=(
    "3000:Grafana"
    "9091:Prometheus"
    "9093:AlertManager"
    "9100:Node Exporter"
    "9121:Redis Exporter"
    "9400:GPU Exporter"
)

for port_check in "${monitoring_ports[@]}"; do
    IFS=':' read -r port service <<< "$port_check"
    
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        print_info "$service port $port is in use"
    else
        print_status "$service port $port is available"
    fi
done

# Final validation summary
echo -e "\n${BLUE}Validation Summary${NC}"
echo "============================================"

if [[ $errors -eq 0 && $warnings -eq 0 ]]; then
    echo -e "${GREEN}🎉 All validations passed!${NC}"
    echo "The monitoring stack is properly configured and ready to use."
elif [[ $errors -eq 0 ]]; then
    echo -e "${YELLOW}⚠ Validation completed with $warnings warning(s)${NC}"
    echo "The monitoring stack should work, but consider addressing the warnings."
else
    echo -e "${RED}❌ Validation failed with $errors error(s) and $warnings warning(s)${NC}"
    echo "Please fix the errors before deploying the monitoring stack."
    exit 1
fi

# Additional recommendations
echo -e "\n${BLUE}Recommendations:${NC}"
echo "  • Test the monitoring stack with: docker-compose --profile monitoring up -d"
echo "  • Configure external alert endpoints in alertmanager.yml"
echo "  • Customize dashboard refresh rates based on your monitoring needs"
echo "  • Set up log aggregation for comprehensive monitoring"
echo "  • Consider enabling remote storage for long-term metrics retention"

exit 0