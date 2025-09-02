# Claude Code Implementation Instructions

CRITICAL UPDATE: FlashBoot cold start performance is 500ms-2s (not sub-200ms) - update all performance targets, expectations, and documentation accordingly while maintaining all other architecture decisions.

## Implementation Phases

### Phase 1: Infrastructure Setup
- Configure Cloudflare R2 replacing all S3 references
- Deploy Redis with clustering and LRU eviction
- Setup MongoDB Atlas with vector search indexes
- Implement Google Secret Manager integration

### Phase 2: Core System Development  
- Build H200ImageAnalyzer with dual caching (Redis L1 + H200 L2)
- Implement YOLOv8 + CLIP models with Redis state management
- Create embedded MCP server with A2A protocol compliance
- Develop LangChain rule management system

### Phase 3: Control Plane & Monitoring
- Build FastAPI control plane with WebSocket support
- Create Vue.js dashboard with real-time metrics
- Implement cost monitoring with progressive alerts
- Setup comprehensive logging and audit trails

### Phase 4: Deployment & Integration
- Configure GitHub Actions with Docker Build Cloud
- Setup RunPod deployment automation
- Implement N8N workflow integration examples
- Complete Templated.io API integration

## Success Criteria
- FlashBoot cold starts: 500ms-2s consistently
- Redis cache hit rate: >85%
- H200 GPU utilization: >70%
- R2 backup cost savings: $15-25/month vs S3
- Complete dual deployment: serverless + timed modes
