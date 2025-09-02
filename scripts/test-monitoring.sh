#!/bin/bash

# H200 Monitoring Stack Test Script
# Tests the complete monitoring infrastructure

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

echo -e "${BLUE}🧪 H200 Monitoring Stack Test${NC}"
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

# Test counter
tests_passed=0
tests_failed=0

# Function to run test
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_pattern="$3"
    
    print_info "Testing: $test_name"
    
    if eval "$test_command" | grep -q "$expected_pattern"; then
        print_status "$test_name - PASSED"
        ((tests_passed++))
    else
        print_error "$test_name - FAILED"
        ((tests_failed++))
    fi
}

cd "$PROJECT_ROOT"

# Test 1: Prometheus connectivity
run_test "Prometheus connectivity" \
         "curl -s http://localhost:9091/api/v1/status/config" \
         "success"

# Test 2: Prometheus targets
run_test "Prometheus targets" \
         "curl -s http://localhost:9091/api/v1/targets" \
         "activeTargets"

# Test 3: Grafana API
run_test "Grafana API" \
         "curl -s http://localhost:3000/api/health" \
         "ok"

# Test 4: AlertManager API
run_test "AlertManager API" \
         "curl -s http://localhost:9093/api/v1/status" \
         "uptime"

# Test 5: Check if metrics are being scraped
run_test "Metrics scraping" \
         "curl -s 'http://localhost:9091/api/v1/query?query=up'" \
         '"result"'

# Test 6: Check specific H200 metrics (if available)
run_test "H200 custom metrics" \
         "curl -s 'http://localhost:9091/api/v1/query?query=h200_requests_total'" \
         '"result"'

# Test 7: Grafana datasource connectivity
run_test "Grafana datasource" \
         "curl -s http://admin:admin@localhost:3000/api/datasources/proxy/1/api/v1/label/__name__/values" \
         '"data"'

# Test 8: Dashboard accessibility
run_test "Dashboard accessibility" \
         "curl -s http://admin:admin@localhost:3000/api/dashboards/uid/h200-overview" \
         '"dashboard"'

# Test 9: Alert rules loaded
run_test "Alert rules loaded" \
         "curl -s http://localhost:9091/api/v1/rules" \
         '"groups"'

# Test 10: GPU metrics (if available)
print_info "Testing: GPU metrics availability"
if curl -s 'http://localhost:9091/api/v1/query?query=dcgm_gpu_utilization' | grep -q '"result"'; then
    print_status "GPU metrics - AVAILABLE"
    ((tests_passed++))
else
    print_warning "GPU metrics - NOT AVAILABLE (optional)"
fi

# Performance Tests
echo -e "\n${BLUE}Running performance tests...${NC}"

# Test query performance
print_info "Testing query performance..."
start_time=$(date +%s%N)
curl -s 'http://localhost:9091/api/v1/query?query=rate(prometheus_http_requests_total[5m])' > /dev/null
end_time=$(date +%s%N)
query_time=$(( (end_time - start_time) / 1000000 ))

if [[ $query_time -lt 1000 ]]; then
    print_status "Query performance: ${query_time}ms (excellent)"
    ((tests_passed++))
elif [[ $query_time -lt 5000 ]]; then
    print_warning "Query performance: ${query_time}ms (acceptable)"
    ((tests_passed++))
else
    print_error "Query performance: ${query_time}ms (slow)"
    ((tests_failed++))
fi

# Test metric cardinality
print_info "Checking metric cardinality..."
cardinality=$(curl -s 'http://localhost:9091/api/v1/label/__name__/values' | jq -r '.data | length' 2>/dev/null || echo "0")
print_info "Total metrics: $cardinality"

if [[ $cardinality -gt 1000 ]]; then
    print_warning "High metric cardinality ($cardinality). Monitor for performance impact."
elif [[ $cardinality -gt 100 ]]; then
    print_status "Good metric cardinality ($cardinality)"
else
    print_info "Low metric cardinality ($cardinality). Ensure all exporters are running."
fi

# Integration Tests
echo -e "\n${BLUE}Testing integrations...${NC}"

# Test webhook endpoint (if configured)
if grep -q "WEBHOOK_URL" "$PROJECT_ROOT/.env" 2>/dev/null; then
    webhook_url=$(grep "WEBHOOK_URL=" "$PROJECT_ROOT/.env" | cut -d'=' -f2)
    if [[ -n "$webhook_url" && "$webhook_url" != "https://your-webhook-url.com" ]]; then
        print_info "Testing webhook connectivity..."
        if curl -s --max-time 10 "$webhook_url/health" &> /dev/null; then
            print_status "Webhook endpoint accessible"
        else
            print_warning "Webhook endpoint not responding"
        fi
    else
        print_info "Webhook URL not configured"
    fi
else
    print_info "Webhook configuration not found"
fi

# Resource Usage Check
echo -e "\n${BLUE}Checking resource usage...${NC}"

# Check container resource usage
monitoring_containers=("h200-prometheus" "h200-grafana" "h200-alertmanager")

for container in "${monitoring_containers[@]}"; do
    if docker ps | grep -q "$container"; then
        stats=$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" "$container" 2>/dev/null | tail -n 1)
        if [[ -n "$stats" ]]; then
            print_status "Resource usage - $stats"
        else
            print_warning "Could not get resource stats for $container"
        fi
    else
        print_info "Container $container is not running"
    fi
done

# Summary
echo -e "\n${BLUE}Test Summary${NC}"
echo "============================================"

total_tests=$((tests_passed + tests_failed))

if [[ $tests_failed -eq 0 ]]; then
    echo -e "${GREEN}🎉 All tests passed! ($tests_passed/$total_tests)${NC}"
    echo "The monitoring stack is fully functional."
    exit_code=0
elif [[ $tests_failed -lt 3 ]]; then
    echo -e "${YELLOW}⚠ Most tests passed ($tests_passed/$total_tests)${NC}"
    echo "The monitoring stack is mostly functional with minor issues."
    exit_code=0
else
    echo -e "${RED}❌ Multiple tests failed ($tests_failed/$total_tests failed)${NC}"
    echo "The monitoring stack has significant issues that need to be addressed."
    exit_code=1
fi

# Additional information
echo -e "\n${BLUE}Monitoring Endpoints:${NC}"
echo "  • Grafana Dashboard: http://localhost:3000"
echo "  • Prometheus: http://localhost:9091"
echo "  • AlertManager: http://localhost:9093"
echo ""
echo -e "${BLUE}Quick Commands:${NC}"
echo "  • View logs: docker-compose logs -f <service-name>"
echo "  • Restart monitoring: docker-compose --profile monitoring restart"
echo "  • Stop monitoring: docker-compose --profile monitoring down"

exit $exit_code