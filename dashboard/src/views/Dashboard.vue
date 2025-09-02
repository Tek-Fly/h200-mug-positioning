<template>
  <AppLayout>
    <div class="space-y-6">
      <!-- Header -->
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 class="text-2xl font-bold text-gray-900 dark:text-white">
            System Dashboard
          </h1>
          <p class="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Real-time monitoring of the H200 Intelligent Mug Positioning System
          </p>
        </div>
        <div class="mt-4 sm:mt-0 flex items-center space-x-3">
          <div class="text-xs text-gray-500 dark:text-gray-400">
            Last updated: {{ formatTime(lastUpdated) }}
          </div>
          <button
            type="button"
            :disabled="loading"
            class="btn btn-primary btn-sm"
            @click="refreshData"
          >
            <ArrowPathIcon :class="['h-4 w-4 mr-2', { 'animate-spin': loading }]" />
            Refresh
          </button>
        </div>
      </div>

      <!-- Overall Health Status -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
            System Health
          </h2>
          <div :class="['px-3 py-1 rounded-full text-sm font-medium', getHealthStatusClass(overallHealth)]">
            <div class="flex items-center">
              <div :class="['w-2 h-2 rounded-full mr-2', getHealthStatusDotClass(overallHealth)]"></div>
              {{ overallHealth.toUpperCase() }}
            </div>
          </div>
        </div>
        
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div
            v-for="service in services"
            :key="service.name"
            class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4"
          >
            <div class="flex items-center justify-between mb-2">
              <h3 class="text-sm font-medium text-gray-900 dark:text-white">
                {{ service.name }}
              </h3>
              <div :class="['w-3 h-3 rounded-full', getHealthStatusDotClass(service.status)]"></div>
            </div>
            <p class="text-xs text-gray-600 dark:text-gray-400">
              {{ getServiceStatusText(service) }}
            </p>
          </div>
        </div>
      </div>

      <!-- Key Metrics Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <!-- GPU Utilization -->
        <MetricCard
          title="GPU Utilization"
          :value="performance?.gpu_utilization_percent || 0"
          suffix="%"
          :trend="getGpuTrend()"
          color="primary"
          :loading="loading"
        >
          <template #icon>
            <CpuChipIcon class="h-6 w-6" />
          </template>
        </MetricCard>

        <!-- Request Rate -->
        <MetricCard
          title="Requests/sec"
          :value="performance?.requests_per_second || 0"
          :decimals="1"
          :trend="getRequestTrend()"
          color="success"
          :loading="loading"
        >
          <template #icon>
            <BoltIcon class="h-6 w-6" />
          </template>
        </MetricCard>

        <!-- API Latency -->
        <MetricCard
          title="API Latency (p95)"
          :value="performance?.api_latency_p95_ms || 0"
          :decimals="0"
          suffix="ms"
          :trend="getLatencyTrend()"
          color="warning"
          :loading="loading"
        >
          <template #icon>
            <ClockIcon class="h-6 w-6" />
          </template>
        </MetricCard>

        <!-- Cache Hit Rate -->
        <MetricCard
          title="Cache Hit Rate"
          :value="performance?.cache_hit_rate || 0"
          suffix="%"
          :decimals="1"
          :trend="getCacheHitTrend()"
          color="success"
          :loading="loading"
        >
          <template #icon>
            <RocketLaunchIcon class="h-6 w-6" />
          </template>
        </MetricCard>
      </div>

      <!-- Charts and Details -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Performance Chart -->
        <div class="card p-6">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Performance Metrics
          </h3>
          <PerformanceChart :height="300" />
        </div>

        <!-- Resource Usage -->
        <div class="card p-6">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Resource Usage
          </h3>
          <ResourceUsageChart v-if="resources" :data="resources" :height="300" />
          <div v-else class="flex items-center justify-center h-[300px]">
            <div class="loading-spinner h-8 w-8"></div>
          </div>
        </div>
      </div>

      <!-- Cost and Activity -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Cost Breakdown -->
        <div class="card p-6">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Daily Costs
          </h3>
          <CostBreakdown v-if="costs" :data="costs" />
          <div v-else class="flex items-center justify-center h-32">
            <div class="loading-spinner h-6 w-6"></div>
          </div>
        </div>

        <!-- Recent Activity -->
        <div class="lg:col-span-2 card p-6">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Recent Activity
          </h3>
          <ActivityFeed :activities="recentActivity" />
        </div>
      </div>
    </div>
  </AppLayout>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import {
  ArrowPathIcon,
  CpuChipIcon,
  BoltIcon,
  ClockIcon,
  RocketLaunchIcon
} from '@heroicons/vue/24/outline'
import { format, formatDistanceToNow } from 'date-fns'
import AppLayout from '@/components/layout/AppLayout.vue'
import MetricCard from '@/components/dashboard/MetricCard.vue'
import PerformanceChart from '@/components/dashboard/PerformanceChart.vue'
import ResourceUsageChart from '@/components/dashboard/ResourceUsageChart.vue'
import CostBreakdown from '@/components/dashboard/CostBreakdown.vue'
import ActivityFeed from '@/components/dashboard/ActivityFeed.vue'
import { useDashboardStore } from '@/stores/dashboard'
import { useWebSocket } from '@/api/websocket'
import type { ServiceStatus } from '@/types/api'
import { logError } from '@/utils/logger'

const dashboardStore = useDashboardStore()
const { connect: connectWs, isConnected } = useWebSocket()

// Computed properties from store
const loading = computed(() => dashboardStore.loading)
const services = computed(() => dashboardStore.services)
const overallHealth = computed(() => dashboardStore.overallHealth)
const performance = computed(() => dashboardStore.performance)
const resources = computed(() => dashboardStore.resources)
const costs = computed(() => dashboardStore.costs)
const recentActivity = computed(() => dashboardStore.recentActivity)
const lastUpdated = computed(() => dashboardStore.lastUpdated)

function refreshData(): void {
  dashboardStore.refreshData()
}

function formatTime(date: Date | null): string {
  if (!date) return 'Never'
  return formatDistanceToNow(date, { addSuffix: true })
}

function getHealthStatusClass(status: ServiceStatus): string {
  switch (status) {
    case 'healthy':
      return 'bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200'
    case 'degraded':
      return 'bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-200'
    case 'unhealthy':
      return 'bg-danger-100 text-danger-800 dark:bg-danger-900 dark:text-danger-200'
    default:
      return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
  }
}

function getHealthStatusDotClass(status: ServiceStatus): string {
  switch (status) {
    case 'healthy':
      return 'bg-success-500'
    case 'degraded':
      return 'bg-warning-500'
    case 'unhealthy':
      return 'bg-danger-500'
    default:
      return 'bg-gray-400'
  }
}

function getServiceStatusText(service: any): string {
  if (service.uptime_seconds) {
    const hours = Math.floor(service.uptime_seconds / 3600)
    return `Uptime: ${hours}h`
  }
  return service.status.charAt(0).toUpperCase() + service.status.slice(1)
}

// Placeholder trend functions - in real app these would calculate actual trends
function getGpuTrend(): number {
  return Math.random() > 0.5 ? 5 : -3
}

function getRequestTrend(): number {
  return Math.random() > 0.5 ? 12 : -8
}

function getLatencyTrend(): number {
  return Math.random() > 0.5 ? -15 : 10
}

function getCacheHitTrend(): number {
  return Math.random() > 0.5 ? 2 : -1
}

onMounted(async () => {
  // Fetch initial data
  await dashboardStore.fetchDashboardData()
  
  // Connect WebSocket and start real-time updates
  try {
    if (!isConnected()) {
      await connectWs()
    }
    dashboardStore.startRealtimeUpdates()
  } catch (error) {
    logError('Failed to setup real-time updates', error, 'Dashboard')
  }
  
  // Start auto-refresh
  dashboardStore.startAutoRefresh(30000) // 30 seconds
})

onUnmounted(() => {
  // Clean up
  dashboardStore.stopRealtimeUpdates()
  dashboardStore.stopAutoRefresh()
})
</script>