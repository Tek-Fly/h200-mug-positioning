// API Types based on FastAPI models

export interface Point2D {
  x: number
  y: number
}

export interface BoundingBox {
  top_left: Point2D
  bottom_right: Point2D
  confidence: number
}

export interface MugDetection {
  id: string
  bbox: BoundingBox
  class_name: string
  attributes: Record<string, any>
}

export interface PositioningResult {
  position: string
  confidence: number
  offset_pixels: Point2D
  offset_mm?: Point2D
  rule_violations: string[]
}

export interface AnalysisResponse {
  request_id: string
  timestamp: string
  processing_time_ms: number
  detections: MugDetection[]
  positioning: PositioningResult
  feedback?: string
  suggestions: string[]
  metadata: Record<string, any>
}

export interface AnalysisFeedback {
  request_id: string
  is_correct: boolean
  correct_position?: string
  comments?: string
}

export enum ServiceStatus {
  HEALTHY = 'healthy',
  DEGRADED = 'degraded',
  UNHEALTHY = 'unhealthy',
  UNKNOWN = 'unknown'
}

export interface ServiceHealth {
  name: string
  status: ServiceStatus
  uptime_seconds?: number
  last_check: string
  details: Record<string, any>
}

export interface PerformanceMetrics {
  cold_start_ms: number
  warm_start_ms: number
  image_processing_ms: number
  api_latency_p95_ms: number
  gpu_utilization_percent: number
  cache_hit_rate: number
  requests_per_second: number
}

export interface ResourceUsage {
  cpu_percent: number
  memory_used_mb: number
  memory_total_mb: number
  gpu_memory_used_mb?: number
  gpu_memory_total_mb?: number
  disk_used_gb: number
  disk_total_gb: number
}

export interface CostMetrics {
  period: string
  compute_cost: number
  storage_cost: number
  network_cost: number
  total_cost: number
  cost_per_request: number
  breakdown: Record<string, number>
}

export interface ActivityLog {
  timestamp: string
  type: string
  user_id?: string
  action: string
  details: Record<string, any>
  duration_ms?: number
}

export interface DashboardResponse {
  timestamp: string
  services?: ServiceHealth[]
  overall_health?: ServiceStatus
  performance?: PerformanceMetrics
  resources?: ResourceUsage
  costs?: CostMetrics
  recent_activity?: ActivityLog[]
  summary: Record<string, any>
}

export enum ServerType {
  SERVERLESS = 'serverless',
  TIMED = 'timed'
}

export enum ServerState {
  STOPPED = 'stopped',
  STARTING = 'starting',
  RUNNING = 'running',
  STOPPING = 'stopping',
  ERROR = 'error'
}

export enum ServerAction {
  START = 'start',
  STOP = 'stop',
  RESTART = 'restart',
  SCALE = 'scale'
}

export interface ServerInfo {
  id: string
  type: ServerType
  state: ServerState
  endpoint_url?: string
  created_at: string
  last_updated: string
  config: Record<string, any>
  metadata: Record<string, any>
}

export interface ServerControlRequest {
  action: ServerAction
  force?: boolean
  config?: {
    min_instances?: number
    max_instances?: number
  }
}

export interface ServerControlResponse {
  success: boolean
  server: ServerInfo
  message: string
  duration_seconds: number
  warnings: string[]
}

export interface Rule {
  id: string
  name: string
  description: string
  type: string
  priority: string
  conditions: Record<string, any>[]
  actions: Record<string, any>[]
  enabled: boolean
  created_at: string
  updated_at: string
  metadata: Record<string, any>
}

export interface NaturalLanguageRuleRequest {
  text: string
  context?: Record<string, any>
  auto_enable: boolean
}

export interface NaturalLanguageRuleResponse {
  rule: Rule
  interpretation: string
  confidence: number
  warnings: string[]
}

// WebSocket message types
export interface WebSocketMessage {
  type: string
  timestamp: string
  data?: any
}

export interface MetricsMessage extends WebSocketMessage {
  type: 'metrics'
  data: {
    gpu_utilization: number
    requests_per_second: number
    average_latency_ms: number
    cache_hit_rate: number
  }
}

export interface LogMessage extends WebSocketMessage {
  type: 'log'
  data: {
    level: string
    message: string
    context: Record<string, any>
  }
}

export interface AlertMessage extends WebSocketMessage {
  type: 'alert'
  data: {
    id: string
    severity: string
    message: string
    details: Record<string, any>
  }
}

export interface ActivityMessage extends WebSocketMessage {
  type: 'activity'
  data: ActivityLog
}