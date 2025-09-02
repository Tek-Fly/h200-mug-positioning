<template>
  <div>
    <div v-if="analyses.length === 0" class="text-center py-8">
      <PhotoIcon class="mx-auto h-12 w-12 text-gray-400" />
      <h3 class="mt-2 text-sm font-medium text-gray-900 dark:text-white">
        No analysis history
      </h3>
      <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
        Start analyzing images to see your history here
      </p>
    </div>

    <div v-else class="space-y-4">
      <div
        v-for="analysis in analyses"
        :key="analysis.request_id"
        class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg p-4 hover:shadow-md transition-shadow duration-200"
      >
        <div class="flex items-start justify-between">
          <div class="flex-1">
            <!-- Header -->
            <div class="flex items-center space-x-3 mb-2">
              <div class="flex-shrink-0">
                <PhotoIcon class="h-5 w-5 text-gray-400" />
              </div>
              <div>
                <h4 class="text-sm font-medium text-gray-900 dark:text-white">
                  Analysis {{ analysis.request_id.substring(0, 8) }}...
                </h4>
                <p class="text-xs text-gray-500 dark:text-gray-400">
                  {{ formatTimestamp(analysis.timestamp) }}
                </p>
              </div>
            </div>

            <!-- Summary -->
            <div class="flex items-center space-x-6 text-sm">
              <div class="flex items-center">
                <span class="text-gray-500 dark:text-gray-400 mr-2">Detections:</span>
                <span class="font-medium text-gray-900 dark:text-white">
                  {{ analysis.detections.length }}
                </span>
              </div>
              
              <div class="flex items-center">
                <span class="text-gray-500 dark:text-gray-400 mr-2">Position:</span>
                <span class="font-medium text-gray-900 dark:text-white">
                  {{ analysis.positioning.position }}
                </span>
              </div>
              
              <div class="flex items-center">
                <span class="text-gray-500 dark:text-gray-400 mr-2">Confidence:</span>
                <span
                  :class="[
                    'px-2 py-1 rounded-full text-xs font-medium',
                    getConfidenceColor(analysis.positioning.confidence)
                  ]"
                >
                  {{ (analysis.positioning.confidence * 100).toFixed(1) }}%
                </span>
              </div>

              <div class="flex items-center">
                <span class="text-gray-500 dark:text-gray-400 mr-2">Time:</span>
                <span class="font-medium text-gray-900 dark:text-white">
                  {{ (analysis.processing_time_ms / 1000).toFixed(2) }}s
                </span>
              </div>
            </div>

            <!-- Feedback and Suggestions -->
            <div v-if="analysis.feedback || analysis.suggestions?.length" class="mt-3">
              <div v-if="analysis.feedback" class="text-xs text-primary-700 dark:text-primary-300 mb-1">
                💡 {{ analysis.feedback }}
              </div>
              
              <div v-if="analysis.suggestions?.length" class="text-xs text-warning-700 dark:text-warning-300">
                📋 {{ analysis.suggestions[0] }}
                <span v-if="analysis.suggestions.length > 1" class="text-gray-500">
                  (+{{ analysis.suggestions.length - 1 }} more)
                </span>
              </div>
            </div>

            <!-- Rule Violations -->
            <div v-if="analysis.positioning.rule_violations?.length" class="mt-2">
              <div class="text-xs text-danger-700 dark:text-danger-300">
                ⚠️ {{ analysis.positioning.rule_violations.length }} rule violation(s)
              </div>
            </div>
          </div>

          <!-- Actions -->
          <div class="flex items-center space-x-2">
            <button
              type="button"
              class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200"
              @click="downloadResults(analysis)"
              title="Download results"
            >
              <ArrowDownTrayIcon class="h-4 w-4" />
            </button>
            
            <button
              type="button"
              class="p-2 text-gray-400 hover:text-danger-600 dark:hover:text-danger-400 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200"
              @click="deleteAnalysis(analysis.request_id)"
              title="Delete analysis"
            >
              <TrashIcon class="h-4 w-4" />
            </button>
          </div>
        </div>

        <!-- Expandable details -->
        <div v-if="expandedAnalysis === analysis.request_id" class="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
          <AnalysisDetails :analysis="analysis" />
        </div>

        <!-- Toggle details button -->
        <div class="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
          <button
            type="button"
            class="text-xs text-primary-600 hover:text-primary-800 dark:text-primary-400 dark:hover:text-primary-200 font-medium flex items-center"
            @click="toggleDetails(analysis.request_id)"
          >
            <component
              :is="expandedAnalysis === analysis.request_id ? ChevronUpIcon : ChevronDownIcon"
              class="h-4 w-4 mr-1"
            />
            {{ expandedAnalysis === analysis.request_id ? 'Hide' : 'Show' }} Details
          </button>
        </div>
      </div>

      <!-- Load More Button -->
      <div v-if="analyses.length >= 10" class="text-center pt-4">
        <button
          type="button"
          class="btn btn-secondary btn-sm"
          @click="$emit('loadMore')"
        >
          Load More History
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { format } from 'date-fns'
import {
  PhotoIcon,
  ArrowDownTrayIcon,
  TrashIcon,
  ChevronUpIcon,
  ChevronDownIcon
} from '@heroicons/vue/24/outline'
import AnalysisDetails from './AnalysisDetails.vue'
import { useNotificationStore } from '@/stores/notifications'
import type { AnalysisResponse } from '@/types/api'

interface Props {
  analyses: AnalysisResponse[]
}

defineProps<Props>()

const emit = defineEmits<{
  loadMore: []
  deleteAnalysis: [requestId: string]
}>()

const expandedAnalysis = ref<string | null>(null)
const notificationStore = useNotificationStore()

function formatTimestamp(timestamp: string): string {
  try {
    return format(new Date(timestamp), 'MMM dd, HH:mm')
  } catch {
    return timestamp
  }
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) {
    return 'bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200'
  } else if (confidence >= 0.6) {
    return 'bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-200'
  } else {
    return 'bg-danger-100 text-danger-800 dark:bg-danger-900 dark:text-danger-200'
  }
}

function toggleDetails(requestId: string): void {
  expandedAnalysis.value = expandedAnalysis.value === requestId ? null : requestId
}

function downloadResults(analysis: AnalysisResponse): void {
  try {
    const data = {
      request_id: analysis.request_id,
      timestamp: analysis.timestamp,
      processing_time_ms: analysis.processing_time_ms,
      detections: analysis.detections,
      positioning: analysis.positioning,
      feedback: analysis.feedback,
      suggestions: analysis.suggestions,
      metadata: analysis.metadata
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    
    const link = document.createElement('a')
    link.href = url
    link.download = `analysis-${analysis.request_id.substring(0, 8)}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    URL.revokeObjectURL(url)
    
    notificationStore.success('Download Complete', 'Analysis results downloaded')
    
  } catch (error) {
    notificationStore.error('Download Error', 'Failed to download analysis results')
  }
}

function deleteAnalysis(requestId: string): void {
  if (confirm('Are you sure you want to delete this analysis?')) {
    emit('deleteAnalysis', requestId)
    notificationStore.success('Analysis Deleted', 'Analysis removed from history')
  }
}
</script>