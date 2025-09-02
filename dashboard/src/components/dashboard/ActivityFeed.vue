<template>
  <div class="space-y-4">
    <div v-if="activities.length === 0" class="text-center py-8">
      <div class="text-gray-500 dark:text-gray-400">
        No recent activity
      </div>
    </div>
    
    <div v-else class="max-h-80 overflow-y-auto space-y-3">
      <div
        v-for="activity in activities"
        :key="`${activity.timestamp}-${activity.action}`"
        class="flex items-start space-x-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors duration-200"
      >
        <!-- Activity Icon -->
        <div
          :class="[
            'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
            getActivityColor(activity.type)
          ]"
        >
          <component :is="getActivityIcon(activity.type)" class="h-4 w-4" />
        </div>

        <!-- Activity Details -->
        <div class="flex-1 min-w-0">
          <div class="flex items-start justify-between">
            <div class="flex-1">
              <p class="text-sm font-medium text-gray-900 dark:text-white">
                {{ activity.action }}
              </p>
              
              <div class="flex items-center space-x-2 mt-1">
                <span class="text-xs text-gray-500 dark:text-gray-400">
                  {{ formatTime(activity.timestamp) }}
                </span>
                
                <span
                  v-if="activity.user_id"
                  class="inline-flex items-center px-2 py-1 rounded-full text-xs bg-gray-200 text-gray-800 dark:bg-gray-600 dark:text-gray-200"
                >
                  {{ activity.user_id }}
                </span>
                
                <span
                  v-if="activity.duration_ms"
                  class="text-xs text-gray-500 dark:text-gray-400"
                >
                  {{ activity.duration_ms.toFixed(0) }}ms
                </span>
              </div>
              
              <!-- Activity Details -->
              <div v-if="Object.keys(activity.details).length > 0" class="mt-2">
                <div class="flex flex-wrap gap-1">
                  <span
                    v-for="(value, key) in getRelevantDetails(activity.details)"
                    :key="key"
                    class="inline-flex items-center px-2 py-1 rounded text-xs bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200"
                  >
                    {{ key }}: {{ formatDetailValue(value) }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Load More Button (if needed) -->
    <div v-if="activities.length >= 20" class="text-center pt-4">
      <button
        type="button"
        class="text-sm text-primary-600 hover:text-primary-800 dark:text-primary-400 dark:hover:text-primary-200 font-medium"
        @click="$emit('loadMore')"
      >
        Load More Activity
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { formatDistanceToNow } from 'date-fns'
import {
  PhotoIcon,
  ServerIcon,
  DocumentTextIcon,
  UserIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  CpuChipIcon
} from '@heroicons/vue/24/outline'
import type { ActivityLog } from '@/types/api'

interface Props {
  activities: ActivityLog[]
}

defineProps<Props>()
defineEmits<{
  loadMore: []
}>()

function formatTime(timestamp: string): string {
  try {
    return formatDistanceToNow(new Date(timestamp), { addSuffix: true })
  } catch {
    return timestamp
  }
}

function getActivityIcon(type: string) {
  switch (type) {
    case 'analysis':
      return PhotoIcon
    case 'server':
      return ServerIcon
    case 'rule':
      return DocumentTextIcon
    case 'user':
      return UserIcon
    case 'alert':
      return ExclamationTriangleIcon
    case 'system':
      return CpuChipIcon
    default:
      return InformationCircleIcon
  }
}

function getActivityColor(type: string): string {
  switch (type) {
    case 'analysis':
      return 'bg-primary-100 text-primary-600 dark:bg-primary-900 dark:text-primary-400'
    case 'server':
      return 'bg-success-100 text-success-600 dark:bg-success-900 dark:text-success-400'
    case 'rule':
      return 'bg-warning-100 text-warning-600 dark:bg-warning-900 dark:text-warning-400'
    case 'user':
      return 'bg-purple-100 text-purple-600 dark:bg-purple-900 dark:text-purple-400'
    case 'alert':
      return 'bg-danger-100 text-danger-600 dark:bg-danger-900 dark:text-danger-400'
    case 'system':
      return 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
    default:
      return 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
  }
}

function getRelevantDetails(details: Record<string, any>): Record<string, any> {
  // Filter out internal/verbose details and show only relevant ones
  const relevant: Record<string, any> = {}
  
  const importantKeys = ['request_id', 'server_id', 'rule_id', 'status', 'error', 'user', 'endpoint']
  
  for (const key of importantKeys) {
    if (details[key] !== undefined) {
      relevant[key] = details[key]
    }
  }
  
  // Limit to 3 most important details
  return Object.fromEntries(Object.entries(relevant).slice(0, 3))
}

function formatDetailValue(value: any): string {
  if (typeof value === 'string' && value.length > 20) {
    return value.substring(0, 20) + '...'
  }
  if (typeof value === 'object') {
    return JSON.stringify(value).substring(0, 20) + '...'
  }
  return String(value)
}
</script>