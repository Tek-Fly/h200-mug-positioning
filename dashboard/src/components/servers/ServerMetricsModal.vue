<template>
  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
    <div class="bg-white dark:bg-gray-800 rounded-lg max-w-4xl w-full mx-4 max-h-[80vh] flex flex-col">
      <!-- Header -->
      <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-600 flex items-center justify-between">
        <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
          Server Metrics - {{ server?.id }}
        </h2>
        <button
          type="button"
          class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
          @click="$emit('close')"
        >
          <XMarkIcon class="h-6 w-6" />
        </button>
      </div>

      <!-- Content -->
      <div class="flex-1 overflow-auto p-6">
        <div v-if="loading" class="flex items-center justify-center py-8">
          <div class="loading-spinner h-8 w-8"></div>
        </div>

        <div v-else-if="!metrics" class="text-center py-8">
          <ChartBarIcon class="mx-auto h-12 w-12 text-gray-400" />
          <h3 class="mt-2 text-sm font-medium text-gray-900 dark:text-white">
            No metrics available
          </h3>
        </div>

        <div v-else class="space-y-6">
          <!-- Key Metrics -->
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <MetricCard
              title="GPU Utilization"
              :value="metrics.gpu_utilization || 0"
              suffix="%"
              color="primary"
            >
              <template #icon>
                <CpuChipIcon class="h-6 w-6" />
              </template>
            </MetricCard>

            <MetricCard
              title="Memory Usage"
              :value="metrics.memory_usage || 0"
              suffix="%"
              color="warning"
            >
              <template #icon>
                <CircleStackIcon class="h-6 w-6" />
              </template>
            </MetricCard>

            <MetricCard
              title="Requests/min"
              :value="metrics.requests_per_minute || 0"
              :decimals="1"
              color="success"
            >
              <template #icon>
                <BoltIcon class="h-6 w-6" />
              </template>
            </MetricCard>
          </div>

          <!-- Detailed Stats -->
          <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <h3 class="text-sm font-medium text-gray-900 dark:text-white mb-4">
              Server Statistics
            </h3>
            
            <div class="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span class="text-gray-500 dark:text-gray-400">Uptime:</span>
                <span class="ml-2 font-medium text-gray-900 dark:text-white">
                  {{ formatUptime(metrics.uptime_seconds || 0) }}
                </span>
              </div>
              
              <div>
                <span class="text-gray-500 dark:text-gray-400">Total Requests:</span>
                <span class="ml-2 font-medium text-gray-900 dark:text-white">
                  {{ (metrics.total_requests || 0).toLocaleString() }}
                </span>
              </div>
              
              <div>
                <span class="text-gray-500 dark:text-gray-400">Error Rate:</span>
                <span class="ml-2 font-medium text-gray-900 dark:text-white">
                  {{ ((metrics.error_rate || 0) * 100).toFixed(2) }}%
                </span>
              </div>
              
              <div>
                <span class="text-gray-500 dark:text-gray-400">Avg Response Time:</span>
                <span class="ml-2 font-medium text-gray-900 dark:text-white">
                  {{ (metrics.avg_response_time || 0).toFixed(0) }}ms
                </span>
              </div>
            </div>
          </div>

          <!-- Resource Usage Chart -->
          <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg p-4">
            <h3 class="text-sm font-medium text-gray-900 dark:text-white mb-4">
              Resource Usage Over Time
            </h3>
            <div class="h-48 flex items-center justify-center text-gray-500 dark:text-gray-400">
              <ChartBarIcon class="h-8 w-8 mr-2" />
              Chart would be rendered here
            </div>
          </div>
        </div>
      </div>

      <!-- Footer -->
      <div class="px-6 py-4 border-t border-gray-200 dark:border-gray-600 flex justify-end">
        <button
          type="button"
          class="btn btn-secondary"
          @click="refreshMetrics"
          :disabled="loading"
        >
          <ArrowPathIcon class="h-4 w-4 mr-2" />
          Refresh
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  XMarkIcon,
  ChartBarIcon,
  ArrowPathIcon,
  CpuChipIcon,
  CircleStackIcon,
  BoltIcon
} from '@heroicons/vue/24/outline'
import MetricCard from '@/components/dashboard/MetricCard.vue'
import { apiClient } from '@/api/client'
import { useNotificationStore } from '@/stores/notifications'
import type { ServerInfo } from '@/types/api'

interface Props {
  server: ServerInfo | null
}

const props = defineProps<Props>()

const emit = defineEmits<{
  close: []
}>()

const metrics = ref<any>(null)
const loading = ref(false)

const notificationStore = useNotificationStore()

function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  
  if (hours > 0) {
    return `${hours}h ${minutes}m`
  }
  return `${minutes}m`
}

async function refreshMetrics(): Promise<void> {
  if (!props.server) return
  
  loading.value = true
  
  try {
    metrics.value = await apiClient.getServerMetrics(props.server.id)
  } catch (error: any) {
    notificationStore.error(
      'Metrics Error',
      error.response?.data?.detail || 'Failed to fetch server metrics'
    )
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  refreshMetrics()
})
</script>