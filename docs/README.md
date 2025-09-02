# H200 Intelligent Mug Positioning System - Documentation

Welcome to the comprehensive documentation for the H200 Intelligent Mug Positioning System. This documentation provides everything you need to understand, deploy, use, and contribute to the system.

## Documentation Structure

### User Documentation
- **[Quick Start Guide](./user-guides/quick-start.md)** - Get up and running in minutes
- **[User Manual](./user-guides/user-manual.md)** - Complete guide to using the system
- **[Dashboard Guide](./user-guides/dashboard-guide.md)** - Navigate and use the web dashboard
- **[Rules Management](./user-guides/rules-management.md)** - Create and manage positioning rules

### Developer Documentation
- **[API Reference](./api/README.md)** - Complete API documentation
- **[Architecture Overview](./developer-guides/architecture.md)** - System design and components
- **[Deployment Guide](./developer-guides/deployment.md)** - Deploy to production environments
- **[Development Setup](./developer-guides/development.md)** - Local development environment
- **[Contributing Guide](./developer-guides/contributing.md)** - Guidelines for contributors

### Operations Documentation
- **[Troubleshooting Guide](./operations/troubleshooting.md)** - Common issues and solutions
- **[Performance Tuning](./operations/performance.md)** - Optimize system performance
- **[Monitoring & Alerting](./operations/monitoring.md)** - Set up monitoring and alerts
- **[Security Guide](./operations/security.md)** - Security best practices

### Technical References
- **[Model Integration](./model_integration.md)** - Deep learning model integration details
- **[Configuration Reference](./reference/configuration.md)** - All configuration options
- **[Error Codes](./reference/error-codes.md)** - API error codes and meanings
- **[Changelog](./CHANGELOG.md)** - Version history and updates

## System Overview

The H200 Intelligent Mug Positioning System is a production-grade GPU-accelerated solution for analyzing mug positions in images and providing intelligent positioning feedback based on configurable rules.

### Key Features

- **Dual Deployment Modes**: Serverless (500ms-2s cold starts) and Timed GPU
- **Dynamic Rule Management**: LangChain-powered natural language rule updates
- **MCP/AG-UI Compliant**: Full protocol support for agentic integration
- **Universal Control Plane**: Complete system monitoring and management
- **Enterprise Security**: Google Secret Manager + JWT authentication
- **Storage Optimized**: Cloudflare R2 + Redis dual-layer caching

### Architecture Highlights

```
Image Input → H200 GPU Analysis → Dynamic Rules → Positioning Calculation →
MongoDB Storage → Redis Caching → Templated.io Rendering → Final Design
```

### Quick Links

- [System Requirements](./user-guides/quick-start.md#requirements)
- [Installation Instructions](./user-guides/quick-start.md#installation)
- [API Endpoints](./api/endpoints.md)
- [Performance Metrics](./operations/performance.md#metrics)
- [Common Issues](./operations/troubleshooting.md#common-issues)

## Getting Help

- **Documentation Issues**: Open an issue on GitHub
- **Technical Support**: support@tekfly.co.uk
- **Community**: [GitHub Discussions](https://github.com/tekfly/h200-mug-positioning/discussions)

## Version Information

This documentation is for version **1.0.0** of the H200 Intelligent Mug Positioning System.

Last updated: 2025-09-02