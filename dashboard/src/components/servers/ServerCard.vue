<template>
  <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg p-6 hover:shadow-md transition-shadow duration-200">
    <div class="flex items-start justify-between mb-4">
      <!-- Server Info -->
      <div class="flex items-start space-x-3">
        <div
          :class="[
            'flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center',
            getServerTypeColor(server.type)
          ]"
        >
          <component :is="getServerTypeIcon(server.type)" class="h-5 w-5" />
        </div>
        
        <div>
          <h3 class="text-lg font-medium text-gray-900 dark:text-white">
            {{ server.id }}
          </h3>
          <p class="text-sm text-gray-500 dark:text-gray-400 capitalize">
            {{ server.type }} Server
          </p>
        </div>
      </div>

      <!-- Status Badge -->
      <div :class="['px-3 py-1 rounded-full text-sm font-medium flex items-center', getStatusColor(server.state)]">
        <div :class="['w-2 h-2 rounded-full mr-2', getStatusDotColor(server.state)]"></div>
        {{ server.state.toUpperCase() }}
      </div>
    </div>

    <!-- Server Details -->
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
      <div>
        <span class="text-xs text-gray-500 dark:text-gray-400">Created</span>
        <div class="text-sm font-medium text-gray-900 dark:text-white">
          {{ formatDate(server.created_at) }}
        </div>
      </div>
      
      <div>
        <span class="text-xs text-gray-500 dark:text-gray-400">Last Updated</span>
        <div class="text-sm font-medium text-gray-900 dark:text-white">
          {{ formatDate(server.last_updated) }}
        </div>
      </div>
      
      <div v-if="server.endpoint_url">
        <span class="text-xs text-gray-500 dark:text-gray-400">Endpoint</span>
        <div class="text-sm font-medium text-gray-900 dark:text-white truncate">
          {{ getEndpointDisplay(server.endpoint_url) }}
        </div>
      </div>
      
      <div v-if="server.config.gpu_type">
        <span class="text-xs text-gray-500 dark:text-gray-400">GPU</span>
        <div class="text-sm font-medium text-gray-900 dark:text-white">
          {{ server.config.gpu_type }}
        </div>
      </div>
    </div>

    <!-- Configuration -->
    <div v-if="Object.keys(server.config).length" class="mb-4">
      <div class="flex flex-wrap gap-2">
        <span
          v-for="(value, key) in getRelevantConfig(server.config)"
          :key="key"
          class="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200"
        >
          {{ formatConfigKey(key) }}: {{ formatConfigValue(value) }}
        </span>
      </div>
    </div>

    <!-- Actions -->
    <div class="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-600">
      <div class="flex items-center space-x-2">
        <!-- Start/Stop/Restart Controls -->
        <template v-if="server.state === 'stopped'">
          <button
            type="button"
            class="btn btn-success btn-sm"
            :disabled="controlLoading"
            @click="$emit('control', server, 'start')"
          >
            <PlayIcon class="h-4 w-4 mr-2" />
            Start
          </button>
        </template>
        
        <template v-else-if="server.state === 'running'">
          <button
            type="button"
            class="btn btn-warning btn-sm"
            :disabled="controlLoading"
            @click="$emit('control', server, 'restart')"
          >
            <ArrowPathIcon class="h-4 w-4 mr-2" />
            Restart
          </button>
          
          <button
            type="button"
            class="btn btn-danger btn-sm"
            :disabled="controlLoading"
            @click="$emit('control', server, 'stop')"
          >
            <StopIcon class="h-4 w-4 mr-2" />
            Stop
          </button>
        </template>
        
        <template v-else>
          <div class="text-sm text-gray-500 dark:text-gray-400">
            {{ getStateMessage(server.state) }}
          </div>
        </template>
      </div>

      <!-- Additional Actions -->
      <div class="flex items-center space-x-1">
        <!-- View Logs -->
        <button
          type="button"
          class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200"
          @click="$emit('viewLogs', server)"
          title="View logs"
        >
          <DocumentTextIcon class="h-4 w-4" />
        </button>

        <!-- View Metrics -->
        <button
          v-if="server.state === 'running'"
          type="button"
          class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200"
          @click="$emit('viewMetrics', server)"
          title="View metrics"
        >
          <ChartBarIcon class="h-4 w-4" />
        </button>

        <!-- Protection -->
        <button
          type="button"
          class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200"
          @click="isProtected ? $emit('unprotect', server) : $emit('protect', server)"
          :title="isProtected ? 'Remove protection' : 'Protect from auto-shutdown'"
        >
          <component :is="isProtected ? ShieldExclamationIcon : ShieldCheckIcon" class="h-4 w-4" />
        </button>

        <!-- More Actions Menu -->
        <div class="relative">
          <button
            type="button"
            class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200"
            @click="showMenu = !showMenu"
          >
            <EllipsisVerticalIcon class="h-4 w-4" />
          </button>

          <Transition
            enter-active-class="transition ease-out duration-100"
            enter-from-class="transform opacity-0 scale-95"
            enter-to-class="transform opacity-100 scale-100"
            leave-active-class="transition ease-in duration-75"
            leave-from-class="transform opacity-100 scale-100"
            leave-to-class="transform opacity-0 scale-95"
          >
            <div
              v-if="showMenu"
              class="origin-top-right absolute right-0 mt-2 w-48 rounded-md shadow-lg bg-white dark:bg-gray-800 ring-1 ring-black ring-opacity-5 focus:outline-none z-10"
              @click.outside="showMenu = false"
            >
              <div class="py-1">
                <button
                  v-if="server.endpoint_url"
                  type="button"
                  class="block w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                  @click="copyEndpoint"
                >
                  Copy Endpoint URL
                </button>
                
                <button
                  type="button"
                  class="block w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                  @click="downloadConfig"
                >
                  Download Config
                </button>
              </div>
            </div>
          </Transition>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { format, formatDistanceToNow } from 'date-fns'
import {
  PlayIcon,
  StopIcon,
  ArrowPathIcon,
  DocumentTextIcon,
  ChartBarIcon,
  ShieldCheckIcon,
  ShieldExclamationIcon,
  EllipsisVerticalIcon,
  BoltIcon,
  ClockIcon
} from '@heroicons/vue/24/outline'
import { useNotificationStore } from '@/stores/notifications'
import type { ServerInfo, ServerState, ServerType, ServerAction } from '@/types/api'

interface Props {
  server: ServerInfo
  controlLoading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  controlLoading: false
})

const emit = defineEmits<{
  control: [server: ServerInfo, action: ServerAction]
  viewLogs: [server: ServerInfo]
  viewMetrics: [server: ServerInfo]
  protect: [server: ServerInfo]
  unprotect: [server: ServerInfo]
}>()

const showMenu = ref(false)
const notificationStore = useNotificationStore()

const isProtected = computed(() => {
  return props.server.metadata?.protected === true
})

function getServerTypeIcon(type: ServerType) {
  return type === 'serverless' ? BoltIcon : ClockIcon
}

function getServerTypeColor(type: ServerType): string {
  return type === 'serverless' 
    ? 'bg-primary-100 text-primary-600 dark:bg-primary-900 dark:text-primary-400'
    : 'bg-success-100 text-success-600 dark:bg-success-900 dark:text-success-400'
}

function getStatusColor(state: ServerState): string {
  switch (state) {
    case 'running':
      return 'bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200'
    case 'stopped':
      return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
    case 'starting':
    case 'stopping':
      return 'bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-200'
    case 'error':
      return 'bg-danger-100 text-danger-800 dark:bg-danger-900 dark:text-danger-200'
    default:
      return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
  }
}

function getStatusDotColor(state: ServerState): string {
  switch (state) {
    case 'running':
      return 'bg-success-500 animate-pulse'
    case 'stopped':
      return 'bg-gray-400'
    case 'starting':
    case 'stopping':
      return 'bg-warning-500 animate-pulse'
    case 'error':
      return 'bg-danger-500'
    default:
      return 'bg-gray-400'
  }
}

function getStateMessage(state: ServerState): string {
  switch (state) {
    case 'starting':
      return 'Starting up...'
    case 'stopping':
      return 'Shutting down...'
    case 'error':
      return 'Error occurred'
    default:
      return state
  }
}

function formatDate(dateString: string): string {
  try {
    return formatDistanceToNow(new Date(dateString), { addSuffix: true })
  } catch {
    return dateString
  }
}

function getEndpointDisplay(url: string): string {
  try {
    const urlObj = new URL(url)
    return urlObj.hostname
  } catch {
    return url
  }
}

function getRelevantConfig(config: Record<string, any>): Record<string, any> {
  const relevant: Record<string, any> = {}
  const keys = ['gpu_type', 'memory_gb', 'min_instances', 'max_instances', 'timeout_seconds']
  
  keys.forEach(key => {
    if (config[key] !== undefined) {
      relevant[key] = config[key]
    }
  })
  
  return relevant
}

function formatConfigKey(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

function formatConfigValue(value: any): string {
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'number') return value.toString()
  return String(value)
}

function copyEndpoint(): void {
  if (props.server.endpoint_url) {
    navigator.clipboard.writeText(props.server.endpoint_url)
    notificationStore.success('Copied', 'Endpoint URL copied to clipboard')
  }
  showMenu.value = false
}

function downloadConfig(): void {
  try {
    const config = {
      id: props.server.id,
      type: props.server.type,
      config: props.server.config,
      metadata: props.server.metadata
    }

    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    
    const link = document.createElement('a')
    link.href = url
    link.download = `server-${props.server.id}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    URL.revokeObjectURL(url)
    
    notificationStore.success('Downloaded', 'Server configuration downloaded')
    
  } catch (error) {
    notificationStore.error('Download Error', 'Failed to download server configuration')
  }
  
  showMenu.value = false
}
</script>