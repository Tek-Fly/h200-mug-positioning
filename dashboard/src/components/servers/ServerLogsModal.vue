<template>
  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
    <div class="bg-white dark:bg-gray-800 rounded-lg max-w-4xl w-full mx-4 max-h-[80vh] flex flex-col">
      <!-- Header -->
      <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-600 flex items-center justify-between">
        <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
          Server Logs - {{ server?.id }}
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
      <div class="flex-1 overflow-hidden p-6">
        <div v-if="loading" class="flex items-center justify-center py-8">
          <div class="loading-spinner h-8 w-8"></div>
        </div>

        <div v-else-if="logs.length === 0" class="text-center py-8">
          <DocumentTextIcon class="mx-auto h-12 w-12 text-gray-400" />
          <h3 class="mt-2 text-sm font-medium text-gray-900 dark:text-white">
            No logs available
          </h3>
        </div>

        <div v-else class="bg-black rounded-lg p-4 font-mono text-sm overflow-auto max-h-96">
          <div
            v-for="(log, index) in logs"
            :key="index"
            :class="[
              'mb-1',
              getLogColor(log.level)
            ]"
          >
            <span class="text-gray-500">{{ log.timestamp }}</span>
            <span class="ml-2 font-semibold">{{ log.level }}</span>
            <span class="ml-2">{{ log.message }}</span>
          </div>
        </div>
      </div>

      <!-- Footer -->
      <div class="px-6 py-4 border-t border-gray-200 dark:border-gray-600 flex justify-end space-x-3">
        <button
          type="button"
          class="btn btn-secondary"
          @click="refreshLogs"
          :disabled="loading"
        >
          <ArrowPathIcon class="h-4 w-4 mr-2" />
          Refresh
        </button>
        <button
          type="button"
          class="btn btn-primary"
          @click="downloadLogs"
        >
          <ArrowDownTrayIcon class="h-4 w-4 mr-2" />
          Download
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  XMarkIcon,
  DocumentTextIcon,
  ArrowPathIcon,
  ArrowDownTrayIcon
} from '@heroicons/vue/24/outline'
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

const logs = ref<any[]>([])
const loading = ref(false)

const notificationStore = useNotificationStore()

function getLogColor(level: string): string {
  switch (level.toUpperCase()) {
    case 'ERROR':
      return 'text-red-400'
    case 'WARN':
    case 'WARNING':
      return 'text-yellow-400'
    case 'INFO':
      return 'text-blue-400'
    case 'DEBUG':
      return 'text-gray-400'
    default:
      return 'text-white'
  }
}

async function refreshLogs(): Promise<void> {
  if (!props.server) return
  
  loading.value = true
  
  try {
    const response = await apiClient.getServerLogs(props.server.id, 100)
    logs.value = response.logs
  } catch (error: any) {
    notificationStore.error(
      'Logs Error',
      error.response?.data?.detail || 'Failed to fetch server logs'
    )
  } finally {
    loading.value = false
  }
}

function downloadLogs(): void {
  if (!props.server || logs.value.length === 0) return
  
  try {
    const logText = logs.value.map(log => 
      `${log.timestamp} ${log.level} ${log.message}`
    ).join('\n')
    
    const blob = new Blob([logText], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    
    const link = document.createElement('a')
    link.href = url
    link.download = `server-${props.server.id}-logs.txt`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    URL.revokeObjectURL(url)
    
    notificationStore.success('Downloaded', 'Server logs downloaded')
  } catch (error) {
    notificationStore.error('Download Error', 'Failed to download logs')
  }
}

onMounted(() => {
  refreshLogs()
})
</script>