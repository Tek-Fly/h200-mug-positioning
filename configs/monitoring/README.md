# H200 Monitoring Stack

This directory contains the complete monitoring configuration for the H200 Intelligent Mug Positioning System. The monitoring stack provides comprehensive observability across system performance, GPU metrics, business KPIs, and cost analysis.

## Architecture

The monitoring stack consists of:

- **Prometheus**: Metrics collection and alerting rules engine
- **Grafana**: Visualization dashboards and analytics
- **AlertManager**: Alert routing and notification management
- **Exporters**: Specialized metric collectors for different components

## Quick Start

### Start Full Monitoring Stack (Recommended)

```bash
# Start all services including monitoring
docker-compose -f docker-compose.production.yml --profile monitoring up -d

# Or include GPU monitoring
docker-compose -f docker-compose.production.yml --profile monitoring --profile gpu up -d
```

### Start Monitoring Only

```bash
# Navigate to monitoring directory
cd configs/monitoring

# Start standalone monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d
```

## Access Points

After starting the services, access the monitoring interfaces:

- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9091
- **AlertManager**: http://localhost:9093
- **Via Nginx Proxy**: 
  - http://localhost/monitoring/grafana/
  - http://localhost/monitoring/prometheus/
  - http://localhost/monitoring/alertmanager/

## Dashboards

### 1. H200 System Overview (`h200-overview`)
**Main system health dashboard**
- Request rate and analysis throughput
- API latency (mean, P95, P99)
- GPU utilization and memory usage
- Error rates and cache hit ratios
- Service status and cost estimates
- System resource utilization

### 2. GPU Detailed Monitoring (`h200-gpu-detailed`)
**Comprehensive GPU metrics**
- Real-time GPU utilization
- Memory usage (used vs total)
- Temperature monitoring with thresholds
- Power consumption and limits
- Multi-GPU support with templating

### 3. Business Metrics & Cost Analysis (`h200-business-metrics`)
**Business intelligence and financial tracking**
- Analysis success vs failure rates
- Cache performance metrics
- Cost projections (per minute, hourly, daily)
- Rule execution statistics
- Model confidence scores
- Integration health monitoring

## Metrics Collected

### API Metrics
- `h200_requests_total`: Total API requests with status labels
- `h200_request_duration_seconds`: Request latency histograms
- `h200:request_rate:5m`: 5-minute request rate
- `h200:error_rate:5m`: 5-minute error rate
- `h200:p95_latency:5m`: 95th percentile latency

### GPU Metrics (via DCGM Exporter)
- `dcgm_gpu_utilization`: GPU compute utilization (%)
- `dcgm_gpu_memory_used/total`: GPU memory usage
- `dcgm_gpu_temperature`: GPU temperature (°C)
- `dcgm_gpu_power_usage/limit`: Power consumption (W)

### System Metrics (via Node Exporter)
- `node_cpu_seconds_total`: CPU usage by mode
- `node_memory_*`: Memory usage statistics
- `node_filesystem_*`: Disk usage and availability
- `node_network_*`: Network interface statistics

### Business Metrics
- `h200_analyses_total`: Total image analyses with status
- `h200_cache_hits/misses_total`: Cache performance
- `h200_model_confidence_score`: Model prediction confidence
- `h200_rule_executions_total`: Rule engine statistics
- `h200_webhook_*_total`: Integration success/failure rates

### Cost Metrics
- `h200:cost_per_minute:estimated`: Estimated operational cost
- `h200_scale_events_total`: Auto-scaling events
- `h200_cold_starts_total`: Cold start occurrences

## Alerting Rules

### Critical Alerts (Immediate Response)
- **ServiceDown**: Any service becomes unavailable
- **HighErrorRate**: Error rate exceeds 10%
- **ExtremeHighLatency**: P99 latency exceeds 5 seconds
- **GPUOverheating**: GPU temperature above 85°C
- **GPUNotResponding**: GPU metrics unavailable

### Warning Alerts (Proactive Monitoring)
- **HighLatency**: P95 latency exceeds 1 second
- **GPUHighUtilization**: GPU usage above 95%
- **GPUMemoryHigh**: GPU memory usage above 90%
- **LowCacheHitRate**: Cache hit rate below 80%
- **HighCostRate**: Estimated cost exceeds $5/minute

### Business Alerts
- **LowAnalysisSuccessRate**: Success rate below 95%
- **NoAnalysesProcessed**: No analyses for 30 minutes
- **ModelAccuracyDrop**: Model confidence below 70%

## Alert Routing

Alerts are categorized and routed to different channels:

- **Critical**: Immediate notifications to all channels
- **GPU**: GPU-specific alerts with thermal/performance data
- **Cost**: Financial threshold warnings
- **Performance**: Latency and throughput issues
- **Business**: Success rates and model performance

## Exporters Configuration

### NVIDIA DCGM Exporter
- Port: 9400
- Metrics: GPU utilization, memory, temperature, power
- Requires: NVIDIA drivers, GPU access

### Node Exporter
- Port: 9100
- Metrics: CPU, memory, disk, network
- Collectors: cpu, meminfo, loadavg, diskstats, filesystem, netdev

### Redis Exporter
- Port: 9121
- Metrics: Redis performance, memory usage, key statistics

### cAdvisor
- Port: 8080
- Metrics: Container resource usage, Docker statistics

### Nginx Exporter
- Port: 9113
- Metrics: HTTP request rates, response times, status codes

## Performance Tuning

### Prometheus Configuration
- **Scrape interval**: 15s (default), 10s (API), 5s (GPU)
- **Retention**: 30 days, 10GB maximum
- **Recording rules**: Pre-computed metrics for faster queries

### Grafana Optimization
- **Refresh rates**: 30s (overview), 10s (GPU), 30s (business)
- **Query timeout**: 60s
- **Caching**: Enabled for dashboards and datasources

### Storage Considerations
- **Prometheus data**: ~1GB per week with current metrics
- **Grafana data**: ~100MB for dashboards and users
- **AlertManager**: ~10MB for alert history

## Troubleshooting

### Common Issues

**Prometheus not scraping targets**
```bash
# Check Prometheus targets
curl http://localhost:9091/api/v1/targets

# Verify network connectivity
docker exec h200-prometheus wget -qO- http://h200-timed:9090/metrics
```

**Grafana dashboards not loading**
```bash
# Check Grafana logs
docker logs h200-grafana

# Verify datasource connection
curl http://localhost:3000/api/datasources/proxy/1/api/v1/query?query=up
```

**GPU metrics missing**
```bash
# Check DCGM exporter
docker logs h200-gpu-exporter

# Verify GPU access
docker exec h200-gpu-exporter nvidia-smi
```

**Alerts not firing**
```bash
# Check AlertManager status
curl http://localhost:9093/api/v1/status

# Verify webhook configuration
docker logs h200-alertmanager
```

### Health Checks

All monitoring services include health checks:

```bash
# Check all monitoring services
docker-compose ps

# Individual service health
curl http://localhost:9091/-/healthy  # Prometheus
curl http://localhost:3000/api/health  # Grafana  
curl http://localhost:9093/-/healthy  # AlertManager
```

## Environment Variables

Required environment variables in `.env`:

```env
# Grafana
GRAFANA_ADMIN_PASSWORD=secure_password

# Alert routing
WEBHOOK_URL=https://your-webhook-url.com
WEBHOOK_TOKEN=your_webhook_token
N8N_WEBHOOK_URL=https://your-n8n-instance.com/webhook
TEMPLATED_API_KEY=your_templated_api_key

# Redis (for Redis exporter)
REDIS_PASSWORD=your_redis_password
```

## Scaling Considerations

### High-Volume Deployments
- Increase Prometheus retention and storage
- Configure remote storage (e.g., Thanos, Cortex)
- Use Grafana clustering for high availability
- Implement metric sharding for large metric volumes

### Multi-Instance Monitoring
- Use service discovery for dynamic targets
- Implement federation for multi-cluster monitoring
- Configure alert deduplication across instances

## Security

### Authentication
- Grafana: Admin password required
- Nginx proxy: HTTP basic auth for /monitoring/
- Internal services: Network-level isolation

### Network Security
- All monitoring traffic on isolated Docker network
- Exporters only accessible within monitoring network
- External access via reverse proxy with authentication

### Data Privacy
- Metrics contain no PII or sensitive business data
- Alert messages sanitized for external delivery
- Retention policies limit historical data exposure

## Maintenance

### Regular Tasks
- Monitor disk usage for Prometheus data
- Update dashboard configurations as needed
- Review and update alert thresholds
- Backup Grafana dashboard configurations

### Updates
```bash
# Update monitoring stack
docker-compose -f docker-compose.production.yml --profile monitoring pull
docker-compose -f docker-compose.production.yml --profile monitoring up -d
```

## Integration with H200 System

The monitoring stack is fully integrated with the H200 application:

- **Automatic metrics**: Application automatically exposes Prometheus metrics
- **Custom metrics**: Business-specific KPIs tracked and visualized
- **Alert integration**: Alerts routed through existing webhook infrastructure
- **Dashboard embedding**: Monitoring widgets available in main dashboard
- **Performance correlation**: System metrics correlated with business outcomes

For more information about H200-specific metrics and integration, see the main project documentation.