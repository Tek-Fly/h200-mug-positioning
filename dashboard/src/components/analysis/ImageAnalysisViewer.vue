<template>
  <div class="space-y-4">
    <!-- Image with Overlays -->
    <div class="relative bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden">
      <img
        ref="imageElement"
        :src="imageUrl"
        alt="Analysis image"
        class="w-full h-auto max-h-96 object-contain"
        @load="handleImageLoad"
      />
      
      <!-- Bounding boxes overlay -->
      <div
        v-if="analysis && imageLoaded"
        class="absolute inset-0 pointer-events-none"
      >
        <div
          v-for="(detection, index) in analysis.detections"
          :key="detection.id"
          :style="getBoundingBoxStyle(detection.bbox)"
          class="absolute border-2 border-primary-500"
        >
          <!-- Detection label -->
          <div class="absolute -top-8 left-0 bg-primary-500 text-white text-xs px-2 py-1 rounded whitespace-nowrap">
            Mug #{{ index + 1 }} ({{ (detection.bbox.confidence * 100).toFixed(1) }}%)
          </div>
          
          <!-- Center point -->
          <div class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
            <div class="w-2 h-2 bg-primary-500 rounded-full"></div>
          </div>
        </div>
        
        <!-- Positioning indicator -->
        <div
          v-if="analysis.positioning.offset_pixels"
          class="absolute top-4 left-4 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded"
        >
          Position: {{ analysis.positioning.position }} 
          ({{ analysis.positioning.confidence.toFixed(2) }} confidence)
        </div>
      </div>
    </div>

    <!-- Analysis Summary -->
    <div v-if="analysis" class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="text-center">
          <div class="text-2xl font-bold text-gray-900 dark:text-white">
            {{ analysis.detections.length }}
          </div>
          <div class="text-xs text-gray-500 dark:text-gray-400">Mugs Detected</div>
        </div>
        
        <div class="text-center">
          <div class="text-2xl font-bold text-gray-900 dark:text-white">
            {{ (analysis.processing_time_ms / 1000).toFixed(2) }}s
          </div>
          <div class="text-xs text-gray-500 dark:text-gray-400">Processing Time</div>
        </div>
        
        <div class="text-center">
          <div class="text-2xl font-bold text-gray-900 dark:text-white">
            {{ (analysis.positioning.confidence * 100).toFixed(1) }}%
          </div>
          <div class="text-xs text-gray-500 dark:text-gray-400">Position Confidence</div>
        </div>
      </div>
    </div>

    <!-- Positioning Feedback -->
    <div v-if="analysis?.feedback || analysis?.suggestions?.length" class="space-y-3">
      <div v-if="analysis.feedback" class="bg-primary-50 dark:bg-primary-900/20 rounded-lg p-4">
        <h4 class="text-sm font-medium text-primary-800 dark:text-primary-200 mb-2">
          💡 AI Feedback
        </h4>
        <p class="text-sm text-primary-700 dark:text-primary-300">
          {{ analysis.feedback }}
        </p>
      </div>

      <div v-if="analysis.suggestions?.length" class="bg-warning-50 dark:bg-warning-900/20 rounded-lg p-4">
        <h4 class="text-sm font-medium text-warning-800 dark:text-warning-200 mb-2">
          📋 Suggestions
        </h4>
        <ul class="space-y-1">
          <li
            v-for="suggestion in analysis.suggestions"
            :key="suggestion"
            class="text-sm text-warning-700 dark:text-warning-300 flex items-start"
          >
            <span class="mr-2">•</span>
            <span>{{ suggestion }}</span>
          </li>
        </ul>
      </div>
    </div>

    <!-- Rule Violations -->
    <div v-if="analysis?.positioning.rule_violations?.length" class="bg-danger-50 dark:bg-danger-900/20 rounded-lg p-4">
      <h4 class="text-sm font-medium text-danger-800 dark:text-danger-200 mb-2">
        ⚠️ Rule Violations
      </h4>
      <ul class="space-y-1">
        <li
          v-for="violation in analysis.positioning.rule_violations"
          :key="violation"
          class="text-sm text-danger-700 dark:text-danger-300 flex items-start"
        >
          <span class="mr-2">•</span>
          <span>{{ violation }}</span>
        </li>
      </ul>
    </div>

    <!-- Feedback Form -->
    <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg p-4">
      <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3">
        Was this analysis helpful?
      </h4>
      
      <div class="flex items-center space-x-4 mb-3">
        <button
          type="button"
          :class="[
            'px-3 py-1 rounded text-sm font-medium transition-colors duration-200',
            feedbackForm.isCorrect === true
              ? 'bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200'
              : 'bg-gray-100 text-gray-700 hover:bg-success-50 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-success-900/20'
          ]"
          @click="feedbackForm.isCorrect = true"
        >
          👍 Yes, accurate
        </button>
        
        <button
          type="button"
          :class="[
            'px-3 py-1 rounded text-sm font-medium transition-colors duration-200',
            feedbackForm.isCorrect === false
              ? 'bg-danger-100 text-danger-800 dark:bg-danger-900 dark:text-danger-200'
              : 'bg-gray-100 text-gray-700 hover:bg-danger-50 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-danger-900/20'
          ]"
          @click="feedbackForm.isCorrect = false"
        >
          👎 Not accurate
        </button>
      </div>

      <div v-if="feedbackForm.isCorrect === false" class="space-y-3">
        <div>
          <label class="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
            What's the correct position?
          </label>
          <input
            v-model="feedbackForm.correctPosition"
            type="text"
            placeholder="e.g., centered, left edge, etc."
            class="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500"
          />
        </div>
      </div>

      <div class="mt-3">
        <label class="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
          Additional comments (optional)
        </label>
        <textarea
          v-model="feedbackForm.comments"
          rows="2"
          placeholder="Any additional feedback..."
          class="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500"
        />
      </div>

      <div class="mt-4 flex justify-end">
        <button
          type="button"
          :disabled="feedbackForm.isCorrect === null || submittingFeedback"
          class="btn btn-primary btn-sm"
          @click="submitFeedback"
        >
          <span v-if="submittingFeedback" class="loading-spinner h-4 w-4 mr-2"></span>
          Submit Feedback
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import { apiClient } from '@/api/client'
import { useNotificationStore } from '@/stores/notifications'
import type { AnalysisResponse, BoundingBox } from '@/types/api'

interface Props {
  imageUrl: string
  analysis: AnalysisResponse | null
}

const props = defineProps<Props>()

const emit = defineEmits<{
  feedbackSubmitted: []
}>()

const imageElement = ref<HTMLImageElement>()
const imageLoaded = ref(false)
const submittingFeedback = ref(false)

const notificationStore = useNotificationStore()

const feedbackForm = reactive({
  isCorrect: null as boolean | null,
  correctPosition: '',
  comments: ''
})

function handleImageLoad(): void {
  imageLoaded.value = true
}

function getBoundingBoxStyle(bbox: BoundingBox): string {
  if (!imageElement.value) return ''
  
  const img = imageElement.value
  const imgRect = img.getBoundingClientRect()
  
  // Convert from image coordinates to display coordinates
  const scaleX = imgRect.width / img.naturalWidth
  const scaleY = imgRect.height / img.naturalHeight
  
  const left = bbox.top_left.x * scaleX
  const top = bbox.top_left.y * scaleY
  const width = (bbox.bottom_right.x - bbox.top_left.x) * scaleX
  const height = (bbox.bottom_right.y - bbox.top_left.y) * scaleY
  
  return `left: ${left}px; top: ${top}px; width: ${width}px; height: ${height}px;`
}

async function submitFeedback(): Promise<void> {
  if (!props.analysis || feedbackForm.isCorrect === null) return
  
  submittingFeedback.value = true
  
  try {
    await apiClient.submitAnalysisFeedback({
      request_id: props.analysis.request_id,
      is_correct: feedbackForm.isCorrect,
      correct_position: feedbackForm.correctPosition || undefined,
      comments: feedbackForm.comments || undefined
    })
    
    emit('feedbackSubmitted')
    
    // Reset form
    feedbackForm.isCorrect = null
    feedbackForm.correctPosition = ''
    feedbackForm.comments = ''
    
  } catch (error: any) {
    notificationStore.error(
      'Feedback Error',
      error.response?.data?.detail || 'Failed to submit feedback'
    )
  } finally {
    submittingFeedback.value = false
  }
}
</script>