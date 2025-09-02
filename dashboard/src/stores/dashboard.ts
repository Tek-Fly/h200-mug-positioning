import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { apiClient } from '@/api/client'
import { useWebSocket } from '@/api/websocket'
import type {
  DashboardResponse,
  ServiceHealth,
  ServiceStatus,
  PerformanceMetrics,
  ResourceUsage,
  CostMetrics,
  ActivityLog,
  MetricsMessage,
  ActivityMessage
} from '@/types/api'
import { useNotificationStore } from './notifications'

export const useDashboardStore = defineStore('dashboard', () => {
  const loading = ref(false)
  const error = ref<string | null>(null)
  const lastUpdated = ref<Date | null>(null)
  
  // Dashboard data
  const services = ref<ServiceHealth[]>([])
  const overallHealth = ref<ServiceStatus>(ServiceStatus.UNKNOWN)
  const performance = ref<PerformanceMetrics | null>(null)
  const resources = ref<ResourceUsage | null>(null)
  const costs = ref<CostMetrics | null>(null)
  const recentActivity = ref<ActivityLog[]>([])
  const summary = ref<Record<string, any>>({})

  // Real-time metrics
  const realtimeMetrics = ref<{
    gpu_utilization: number
    requests_per_second: number
    average_latency_ms: number
    cache_hit_rate: number
  } | null>(null)

  // Computed values
  const healthyServicesCount = computed(() =>
    services.value.filter(s => s.status === ServiceStatus.HEALTHY).length
  )

  const totalServicesCount = computed(() => services.value.length)

  const healthPercentage = computed(() =>
    totalServicesCount.value > 0 ? (healthyServicesCount.value / totalServicesCount.value) * 100 : 0
  )

  const isSystemHealthy = computed(() => overallHealth.value === ServiceStatus.HEALTHY)

  // WebSocket integration
  const { subscribe, unsubscribe } = useWebSocket()
  
  let metricsUnsubscribe: (() => void) | null = null
  let activityUnsubscribe: (() => void) | null = null

  async function fetchDashboardData(): Promise<void> {
    loading.value = true
    error.value = null
    
    try {
      const data = await apiClient.getDashboard()
      updateDashboardData(data)
    } catch (err: any) {
      error.value = err.message || 'Failed to fetch dashboard data'
      const notificationStore = useNotificationStore()
      notificationStore.error('Dashboard Error', error.value)
    } finally {
      loading.value = false
    }
  }

  function updateDashboardData(data: DashboardResponse): void {
    if (data.services) services.value = data.services
    if (data.overall_health) overallHealth.value = data.overall_health
    if (data.performance) performance.value = data.performance
    if (data.resources) resources.value = data.resources
    if (data.costs) costs.value = data.costs
    if (data.recent_activity) recentActivity.value = data.recent_activity
    summary.value = data.summary || {}
    
    lastUpdated.value = new Date()
  }

  function startRealtimeUpdates(): void {
    // Subscribe to metrics updates
    metricsUnsubscribe = subscribe('metrics', (message) => {
      const metricsMessage = message as MetricsMessage
      if (metricsMessage.data) {
        realtimeMetrics.value = metricsMessage.data
        
        // Update performance metrics with real-time data
        if (performance.value) {
          performance.value.gpu_utilization_percent = metricsMessage.data.gpu_utilization
          performance.value.requests_per_second = metricsMessage.data.requests_per_second
          performance.value.api_latency_p95_ms = metricsMessage.data.average_latency_ms
          performance.value.cache_hit_rate = metricsMessage.data.cache_hit_rate
        }
      }
    })

    // Subscribe to activity updates
    activityUnsubscribe = subscribe('activity', (message) => {
      const activityMessage = message as ActivityMessage
      if (activityMessage.data) {
        // Add new activity to the beginning of the array
        recentActivity.value.unshift(activityMessage.data)
        
        // Keep only the last 50 activities
        if (recentActivity.value.length > 50) {
          recentActivity.value = recentActivity.value.slice(0, 50)
        }
      }
    })
  }

  function stopRealtimeUpdates(): void {
    if (metricsUnsubscribe) {
      metricsUnsubscribe()
      metricsUnsubscribe = null
    }
    
    if (activityUnsubscribe) {
      activityUnsubscribe()
      activityUnsubscribe = null
    }
  }

  async function refreshData(): Promise<void> {
    await fetchDashboardData()
  }

  function getServiceByName(name: string): ServiceHealth | undefined {
    return services.value.find(s => s.name === name)
  }

  function getMetricHistory(metricName: string, timeRange: string = '1h'): Promise<any> {
    return apiClient.getMetricHistory(metricName, timeRange)
  }

  // Auto-refresh data periodically
  let refreshInterval: NodeJS.Timeout | null = null

  function startAutoRefresh(intervalMs: number = 30000): void {
    stopAutoRefresh()
    refreshInterval = setInterval(fetchDashboardData, intervalMs)
  }

  function stopAutoRefresh(): void {
    if (refreshInterval) {
      clearInterval(refreshInterval)
      refreshInterval = null
    }
  }

  return {
    // State
    loading: readonly(loading),
    error: readonly(error),
    lastUpdated: readonly(lastUpdated),
    services: readonly(services),
    overallHealth: readonly(overallHealth),
    performance: readonly(performance),
    resources: readonly(resources),
    costs: readonly(costs),
    recentActivity: readonly(recentActivity),
    summary: readonly(summary),
    realtimeMetrics: readonly(realtimeMetrics),

    // Computed
    healthyServicesCount,
    totalServicesCount,
    healthPercentage,
    isSystemHealthy,

    // Methods
    fetchDashboardData,
    refreshData,
    getServiceByName,
    getMetricHistory,
    startRealtimeUpdates,
    stopRealtimeUpdates,
    startAutoRefresh,
    stopAutoRefresh
  }
})