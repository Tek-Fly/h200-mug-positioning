# H200 Intelligent Mug Positioning System

[![Build Status](https://github.com/tekfly/h200-mug-positioning/workflows/Deploy/badge.svg)](https://github.com/tekfly/h200-mug-positioning/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

Production-grade H200 GPU-accelerated intelligent mug positioning system with dynamic rule management, Cloudflare R2 storage optimization, Redis caching, and comprehensive control plane.

## Features

- 🚀 **Dual Deployment Modes**: Serverless (500ms-2s cold starts) and Timed GPU
- 🧠 **Dynamic Rule Management**: LangChain-powered natural language rule updates
- 🔌 **MCP/AG-UI Compliant**: Full protocol support for agentic integration
- 📊 **Universal Control Plane**: Complete system monitoring and management
- 🔐 **Enterprise Security**: Google Secret Manager + JWT authentication
- 📈 **Storage Optimized**: Cloudflare R2 + Redis dual-layer caching
- 🐳 **Docker Build Cloud**: Optimized CI/CD pipeline

## Quick Start

1. **Clone and setup**

git clone https://github.com/tekfly/h200-mug-positioning.git
cd h200-mug-positioning
cp .env.example .env
Edit .env with your credentials


2. **Deploy to RunPod**

python scripts/deploy_to_runpod.py
--serverless-image tekfly/h200:serverless-latest
--timed-image tekfly/h200:timed-latest
--enable-r2 --enable-redis

3. **Access Control Plane**

http://[POD_ID]-8002.proxy.runpod.net

## Architecture


Image Input → H200 GPU Analysis → Dynamic Rules → Positioning Calculation →
MongoDB Storage → Redis Caching → Templated.io Rendering → Final Design

### Storage Architecture
- **Primary**: RunPod network volumes
- **Secondary**: Cloudflare R2 (zero egress fees)
- **Tertiary**: GitHub LFS for model checkpoints
- **Caching**: Redis L1 + H200 L2 dual-layer

## 📚 Documentation

### Quick Links
- **[📖 Complete Documentation](./docs/README.md)** - Full documentation index
- **[🚀 Quick Start Guide](./docs/user-guides/quick-start.md)** - Get up and running in minutes
- **[🔧 API Reference](./docs/api/README.md)** - Complete API documentation
- **[🎛️ Dashboard Guide](./docs/user-guides/dashboard-guide.md)** - Web interface tutorial
- **[🛠️ Troubleshooting](./docs/operations/troubleshooting.md)** - Common issues and solutions

### User Documentation
- **[User Manual](./docs/user-guides/user-manual.md)** - Complete feature guide
- **[Rules Management](./docs/user-guides/rules-management.md)** - Create and manage positioning rules
- **[Configuration Reference](./docs/reference/configuration.md)** - All configuration options
- **[Error Codes](./docs/reference/error-codes.md)** - API error reference

### Developer Documentation  
- **[Architecture Overview](./docs/developer-guides/architecture.md)** - System design and components
- **[Development Setup](./docs/developer-guides/development.md)** - Local development environment
- **[Deployment Guide](./docs/developer-guides/deployment.md)** - Production deployment
- **[Contributing Guide](./docs/developer-guides/contributing.md)** - Development workflow

### Operations Documentation
- **[Performance Guide](./docs/operations/performance.md)** - Optimization and tuning
- **[Monitoring & Alerts](./docs/operations/monitoring.md)** - System monitoring setup
- **[Security Guide](./docs/operations/security.md)** - Security best practices

## API Endpoints

- `POST /api/v1/analyze/with-feedback` - Image analysis with optional feedback
- `POST /api/v1/rules/natural-language` - Natural language rule updates  
- `GET /api/v1/dashboard` - Complete dashboard data
- `POST /api/v1/servers/{type}/control` - Server management
- `WebSocket /ws/control-plane` - Real-time updates

**Full API Documentation**: [📋 API Reference](./docs/api/endpoints.md)

## Development

### Prerequisites
- Python 3.11+
- Docker Desktop
- MongoDB Atlas account
- RunPod account
- Cloudflare account (for R2)

### Local Setup

```bash
pip install -r requirements-dev.txt
pytest tests/
docker-compose -f docker-compose.yml up
```

**Detailed Setup**: [🔧 Development Guide](./docs/developer-guides/development.md)

## Cost Optimization

- **Cloudflare R2**: $1.35/month (100GB) vs $2.30+ for S3
- **Redis Enhancement**: 20-30% H200 efficiency gains
- **Intelligent Auto-shutdown**: 10-minute idle timeout
- **Monthly Budget**: ~$1,749 with all enhancements

## Security

- All secrets managed through Google Secret Manager
- JWT-based authentication for API access
- TLS encryption for all communications
- Regular security scans with Docker Scout

## Documentation

- **📖 [COMPREHENSIVE USER MANUAL](./USER_MANUAL.md)** - Complete guide with copy-paste examples
- **📚 [Full Documentation](./docs/README.md)** - Technical documentation
- **🚀 [Quick Start Guide](./docs/user-guides/quick-start.md)** - Get started quickly
- **🏗️ [API Reference](./docs/api/endpoints.md)** - Complete API documentation

## Support

- **Email**: support@tekfly.co.uk
- **Documentation**: [docs.tekfly.co.uk](https://docs.tekfly.co.uk)
- **Issues**: [GitHub Issues](https://github.com/tekfly/h200-mug-positioning/issues)

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**© 2025 Tekfly Ltd. All rights reserved.**
