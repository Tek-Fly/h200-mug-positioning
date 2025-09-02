<template>
  <AppLayout>
    <div class="space-y-6">
      <!-- Header -->
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 class="text-2xl font-bold text-gray-900 dark:text-white">
            Server Management
          </h1>
          <p class="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Manage and monitor your H200 inference servers
          </p>
        </div>
        <div class="mt-4 sm:mt-0 flex items-center space-x-3">
          <button
            type="button"
            :disabled="loading"
            class="btn btn-secondary btn-sm"
            @click="refreshServers"
          >
            <ArrowPathIcon :class="['h-4 w-4 mr-2', { 'animate-spin': loading }]" />
            Refresh
          </button>
        </div>
      </div>

      <!-- Server Overview -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <MetricCard
          title="Total Servers"
          :value="servers.length"
          color="primary"
          :loading="loading"
        >
          <template #icon>
            <ServerIcon class="h-6 w-6" />
          </template>
        </MetricCard>

        <MetricCard
          title="Running Servers"
          :value="runningServers"
          color="success"
          :loading="loading"
        >
          <template #icon>
            <PlayIcon class="h-6 w-6" />
          </template>
        </MetricCard>

        <MetricCard
          title="Estimated Cost/Day"
          :value="estimatedDailyCost"
          prefix="$"
          :decimals="2"
          color="warning"
          :loading="loading"
        >
          <template #icon>
            <CurrencyDollarIcon class="h-6 w-6" />
          </template>
        </MetricCard>
      </div>

      <!-- Servers List -->
      <div class="card p-6">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
            Servers
          </h2>
          
          <div class="flex items-center space-x-3">
            <select
              v-model="serverTypeFilter"
              class="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="">All Types</option>
              <option value="serverless">Serverless</option>
              <option value="timed">Timed</option>
            </select>
          </div>
        </div>

        <div v-if="loading" class="text-center py-8">
          <div class="loading-spinner h-8 w-8 mx-auto"></div>
          <div class="mt-2 text-sm text-gray-500 dark:text-gray-400">
            Loading servers...
          </div>
        </div>

        <div v-else-if="filteredServers.length === 0" class="text-center py-8">
          <ServerIcon class="mx-auto h-12 w-12 text-gray-400" />
          <h3 class="mt-2 text-sm font-medium text-gray-900 dark:text-white">
            No servers found
          </h3>
          <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
            {{ servers.length === 0 ? 'Deploy your first server to get started' : 'No servers match your filter' }}
          </p>
        </div>

        <div v-else class="space-y-4">
          <ServerCard
            v-for="server in filteredServers"
            :key="server.id"
            :server="server"
            @control="handleServerControl"
            @view-logs="handleViewLogs"
            @view-metrics="handleViewMetrics"
            @protect="handleProtectServer"
            @unprotect="handleUnprotectServer"
          />
        </div>
      </div>

      <!-- Server Logs Modal -->
      <ServerLogsModal
        v-if="showLogsModal"
        :server="selectedServer"
        @close="showLogsModal = false"
      />

      <!-- Server Metrics Modal -->
      <ServerMetricsModal
        v-if="showMetricsModal"
        :server="selectedServer"
        @close="showMetricsModal = false"
      />
    </div>
  </AppLayout>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import {
  ArrowPathIcon,
  ServerIcon,
  PlayIcon,
  CurrencyDollarIcon
} from '@heroicons/vue/24/outline'
import AppLayout from '@/components/layout/AppLayout.vue'
import MetricCard from '@/components/dashboard/MetricCard.vue'
import ServerCard from '@/components/servers/ServerCard.vue'
import ServerLogsModal from '@/components/servers/ServerLogsModal.vue'
import ServerMetricsModal from '@/components/servers/ServerMetricsModal.vue'
import { apiClient } from '@/api/client'
import { useNotificationStore } from '@/stores/notifications'
import type { ServerInfo, ServerType, ServerAction, ServerControlRequest } from '@/types/api'

const servers = ref<ServerInfo[]>([])
const loading = ref(false)
const serverTypeFilter = ref<string>('')
const showLogsModal = ref(false)
const showMetricsModal = ref(false)
const selectedServer = ref<ServerInfo | null>(null)

const notificationStore = useNotificationStore()

const filteredServers = computed(() => {
  if (!serverTypeFilter.value) return servers.value
  return servers.value.filter(server => server.type === serverTypeFilter.value)
})

const runningServers = computed(() => {
  return servers.value.filter(server => server.state === 'running').length
})

const estimatedDailyCost = computed(() => {
  // Rough cost estimation - in real app this would be more precise
  const serverlessRate = 0.50 // per hour when running
  const timedRate = 3.50 // per hour
  
  let totalCost = 0
  
  servers.value.forEach(server => {
    if (server.state === 'running') {
      const rate = server.type === 'serverless' ? serverlessRate : timedRate
      totalCost += rate * 24 // 24 hours
    }
  })
  
  return totalCost
})

async function refreshServers(): Promise<void> {
  loading.value = true
  
  try {
    servers.value = await apiClient.listServers()
  } catch (error: any) {
    notificationStore.error(
      'Load Error',
      error.response?.data?.detail || 'Failed to load servers'
    )
  } finally {
    loading.value = false
  }
}

async function handleServerControl(server: ServerInfo, action: ServerAction): Promise<void> {
  try {
    const request: ServerControlRequest = {
      action,
      force: action === 'stop' // Force stop for safety
    }
    
    const response = await apiClient.controlServer(server.type, request)
    
    notificationStore.success(
      'Server Control',
      response.message
    )
    
    // Update server in list
    const index = servers.value.findIndex(s => s.id === server.id)
    if (index !== -1) {
      servers.value[index] = response.server
    }
    
  } catch (error: any) {
    notificationStore.error(
      'Control Error',
      error.response?.data?.detail || `Failed to ${action} server`
    )
  }
}

function handleViewLogs(server: ServerInfo): void {
  selectedServer.value = server
  showLogsModal.value = true
}

function handleViewMetrics(server: ServerInfo): void {
  selectedServer.value = server
  showMetricsModal.value = true
}

async function handleProtectServer(server: ServerInfo): Promise<void> {
  try {
    await apiClient.protectServer(server.id)
    notificationStore.success(
      'Server Protected',
      `${server.id} is now protected from auto-shutdown`
    )
  } catch (error: any) {
    notificationStore.error(
      'Protection Error',
      error.response?.data?.detail || 'Failed to protect server'
    )
  }
}

async function handleUnprotectServer(server: ServerInfo): Promise<void> {
  try {
    await apiClient.unprotectServer(server.id)
    notificationStore.success(
      'Protection Removed',
      `${server.id} protection removed`
    )
  } catch (error: any) {
    notificationStore.error(
      'Unprotection Error',
      error.response?.data?.detail || 'Failed to remove server protection'
    )
  }
}

onMounted(() => {
  refreshServers()
})
</script>