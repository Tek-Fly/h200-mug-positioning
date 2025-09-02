import axios, { type AxiosInstance, type AxiosResponse } from 'axios'
import type {
  AnalysisFeedback,
  AnalysisResponse,
  DashboardResponse,
  NaturalLanguageRuleRequest,
  NaturalLanguageRuleResponse,
  Rule,
  ServerControlRequest,
  ServerControlResponse,
  ServerInfo,
  ServerType
} from '@/types/api'

export class ApiClient {
  private client: AxiosInstance
  private authToken: string | null = null

  constructor(baseURL: string = '/api/v1') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    })

    // Request interceptor to add auth token
    this.client.interceptors.request.use((config) => {
      if (this.authToken) {
        config.headers.Authorization = `Bearer ${this.authToken}`
      }
      return config
    })

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          this.clearAuth()
          // Emit auth error event
          window.dispatchEvent(new CustomEvent('auth:unauthorized'))
        }
        return Promise.reject(error)
      }
    )
  }

  setAuth(token: string): void {
    this.authToken = token
    localStorage.setItem('auth_token', token)
  }

  clearAuth(): void {
    this.authToken = null
    localStorage.removeItem('auth_token')
  }

  initAuth(): string | null {
    const token = localStorage.getItem('auth_token')
    if (token) {
      this.authToken = token
    }
    return token
  }

  // Auth endpoints
  async login(username: string, password: string): Promise<{ access_token: string }> {
    const response = await this.client.post('/auth/login', {
      username,
      password
    })
    return response.data
  }

  async getCurrentUser(): Promise<any> {
    const response = await this.client.get('/auth/me')
    return response.data
  }

  // Dashboard endpoints
  async getDashboard(): Promise<DashboardResponse> {
    const response = await this.client.get('/dashboard')
    return response.data
  }

  async getMetricHistory(metricName: string, timeRange: string = '1h'): Promise<any> {
    const response = await this.client.get(`/dashboard/metrics/${metricName}`, {
      params: { time_range: timeRange }
    })
    return response.data
  }

  // Analysis endpoints
  async analyzeImage(
    imageFile: File,
    options: {
      includeFeedback?: boolean
      rulesContext?: string
      calibrationMmPerPixel?: number
      confidenceThreshold?: number
    } = {}
  ): Promise<AnalysisResponse> {
    const formData = new FormData()
    formData.append('image', imageFile)
    
    if (options.includeFeedback !== undefined) {
      formData.append('include_feedback', options.includeFeedback.toString())
    }
    if (options.rulesContext) {
      formData.append('rules_context', options.rulesContext)
    }
    if (options.calibrationMmPerPixel) {
      formData.append('calibration_mm_per_pixel', options.calibrationMmPerPixel.toString())
    }
    if (options.confidenceThreshold) {
      formData.append('confidence_threshold', options.confidenceThreshold.toString())
    }

    const response = await this.client.post('/analyze/with-feedback', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    return response.data
  }

  async submitAnalysisFeedback(feedback: AnalysisFeedback): Promise<{ success: boolean }> {
    const response = await this.client.post('/analyze/feedback', feedback)
    return response.data
  }

  // Server management endpoints
  async listServers(serverType?: ServerType): Promise<ServerInfo[]> {
    const response = await this.client.get('/servers', {
      params: serverType ? { server_type: serverType } : undefined
    })
    return response.data
  }

  async getServer(serverId: string): Promise<ServerInfo> {
    const response = await this.client.get(`/servers/${serverId}`)
    return response.data
  }

  async controlServer(
    serverType: ServerType,
    request: ServerControlRequest
  ): Promise<ServerControlResponse> {
    const response = await this.client.post(`/servers/${serverType}/control`, request)
    return response.data
  }

  async getServerMetrics(serverId: string): Promise<any> {
    const response = await this.client.get(`/servers/${serverId}/metrics`)
    return response.data
  }

  async getServerLogs(serverId: string, lines: number = 100): Promise<any> {
    const response = await this.client.post(`/servers/${serverId}/logs`, {
      lines,
      level: 'INFO'
    })
    return response.data
  }

  async getServerStatus(serverId: string): Promise<any> {
    const response = await this.client.get(`/servers/${serverId}/status`)
    return response.data
  }

  async protectServer(serverId: string): Promise<{ success: boolean }> {
    const response = await this.client.post(`/servers/${serverId}/protect`)
    return response.data
  }

  async unprotectServer(serverId: string): Promise<{ success: boolean }> {
    const response = await this.client.delete(`/servers/${serverId}/protect`)
    return response.data
  }

  // Rules management endpoints
  async listRules(enabledOnly: boolean = false, ruleType?: string): Promise<Rule[]> {
    const response = await this.client.get('/rules', {
      params: {
        enabled_only: enabledOnly,
        rule_type: ruleType
      }
    })
    return response.data
  }

  async getRule(ruleId: string): Promise<Rule> {
    const response = await this.client.get(`/rules/${ruleId}`)
    return response.data
  }

  async createRuleFromNaturalLanguage(
    request: NaturalLanguageRuleRequest
  ): Promise<NaturalLanguageRuleResponse> {
    const response = await this.client.post('/rules/natural-language', request)
    return response.data
  }

  async createRule(rule: Partial<Rule>): Promise<Rule> {
    const response = await this.client.post('/rules', rule)
    return response.data
  }

  async updateRule(ruleId: string, updates: Partial<Rule>): Promise<Rule> {
    const response = await this.client.patch(`/rules/${ruleId}`, updates)
    return response.data
  }

  async deleteRule(ruleId: string): Promise<{ success: boolean }> {
    const response = await this.client.delete(`/rules/${ruleId}`)
    return response.data
  }

  async evaluateRules(data: any, ruleIds?: string[]): Promise<any> {
    const response = await this.client.post('/rules/evaluate', {
      data,
      rule_ids: ruleIds,
      include_disabled: false
    })
    return response.data
  }
}

// Create singleton instance
export const apiClient = new ApiClient()

// Export types for convenience
export * from '@/types/api'