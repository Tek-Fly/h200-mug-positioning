<template>
  <div class="space-y-4">
    <!-- Processing Info -->
    <div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
      <div class="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span class="text-gray-500 dark:text-gray-400">Request ID:</span>
          <span class="ml-2 font-mono text-xs">{{ analysis.request_id }}</span>
        </div>
        <div>
          <span class="text-gray-500 dark:text-gray-400">Timestamp:</span>
          <span class="ml-2">{{ formatTimestamp(analysis.timestamp) }}</span>
        </div>
        <div>
          <span class="text-gray-500 dark:text-gray-400">Processing Time:</span>
          <span class="ml-2 font-semibold">{{ analysis.processing_time_ms.toFixed(0) }}ms</span>
        </div>
        <div>
          <span class="text-gray-500 dark:text-gray-400">Model Version:</span>
          <span class="ml-2">{{ analysis.metadata.model_version || 'Unknown' }}</span>
        </div>
      </div>
    </div>

    <!-- Detection Details -->
    <div>
      <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3">
        Detection Details
      </h4>
      
      <div class="space-y-3">
        <div
          v-for="(detection, index) in analysis.detections"
          :key="detection.id"
          class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg p-4"
        >
          <div class="flex items-center justify-between mb-3">
            <h5 class="text-sm font-medium text-gray-900 dark:text-white">
              Mug #{{ index + 1 }}
            </h5>
            <span
              :class="[
                'px-2 py-1 rounded-full text-xs font-medium',
                getConfidenceColor(detection.bbox.confidence)
              ]"
            >
              {{ (detection.bbox.confidence * 100).toFixed(1) }}% confidence
            </span>
          </div>
          
          <div class="grid grid-cols-2 gap-4 text-xs">
            <div>
              <span class="text-gray-500 dark:text-gray-400">Position:</span>
              <div class="mt-1 font-mono">
                Top-left: ({{ detection.bbox.top_left.x.toFixed(0) }}, {{ detection.bbox.top_left.y.toFixed(0) }})<br>
                Bottom-right: ({{ detection.bbox.bottom_right.x.toFixed(0) }}, {{ detection.bbox.bottom_right.y.toFixed(0) }})
              </div>
            </div>
            <div>
              <span class="text-gray-500 dark:text-gray-400">Size:</span>
              <div class="mt-1 font-mono">
                {{ (detection.bbox.bottom_right.x - detection.bbox.top_left.x).toFixed(0) }} × 
                {{ (detection.bbox.bottom_right.y - detection.bbox.top_left.y).toFixed(0) }} px
              </div>
            </div>
          </div>

          <!-- Attributes -->
          <div v-if="Object.keys(detection.attributes).length" class="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
            <span class="text-xs text-gray-500 dark:text-gray-400">Attributes:</span>
            <div class="mt-1 flex flex-wrap gap-1">
              <span
                v-for="(value, key) in detection.attributes"
                :key="key"
                class="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200"
              >
                {{ key }}: {{ value }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Positioning Analysis -->
    <div>
      <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3">
        Positioning Analysis
      </h4>
      
      <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg p-4">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <span class="text-sm text-gray-500 dark:text-gray-400">Position:</span>
            <div class="mt-1 text-lg font-semibold text-gray-900 dark:text-white">
              {{ analysis.positioning.position }}
            </div>
          </div>
          
          <div>
            <span class="text-sm text-gray-500 dark:text-gray-400">Confidence:</span>
            <div class="mt-1">
              <div class="flex items-center">
                <div class="flex-1 bg-gray-200 dark:bg-gray-600 rounded-full h-2 mr-3">
                  <div
                    :class="[
                      'h-2 rounded-full transition-all duration-300',
                      getConfidenceBarColor(analysis.positioning.confidence)
                    ]"
                    :style="{ width: `${analysis.positioning.confidence * 100}%` }"
                  ></div>
                </div>
                <span class="text-sm font-medium text-gray-900 dark:text-white">
                  {{ (analysis.positioning.confidence * 100).toFixed(1) }}%
                </span>
              </div>
            </div>
          </div>
        </div>

        <div class="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <span class="text-gray-500 dark:text-gray-400">Offset (pixels):</span>
            <div class="mt-1 font-mono">
              X: {{ analysis.positioning.offset_pixels.x.toFixed(1) }}px<br>
              Y: {{ analysis.positioning.offset_pixels.y.toFixed(1) }}px
            </div>
          </div>
          
          <div v-if="analysis.positioning.offset_mm">
            <span class="text-gray-500 dark:text-gray-400">Offset (mm):</span>
            <div class="mt-1 font-mono">
              X: {{ analysis.positioning.offset_mm.x.toFixed(1) }}mm<br>
              Y: {{ analysis.positioning.offset_mm.y.toFixed(1) }}mm
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Metadata -->
    <div v-if="Object.keys(analysis.metadata).length" class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
      <h4 class="text-sm font-medium text-gray-900 dark:text-white mb-3">
        Additional Information
      </h4>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        <div v-if="analysis.metadata.image_size">
          <span class="text-gray-500 dark:text-gray-400">Image Size:</span>
          <span class="ml-2 font-mono">
            {{ analysis.metadata.image_size[0] }} × {{ analysis.metadata.image_size[1] }}px
          </span>
        </div>
        
        <div v-if="analysis.metadata.user_id">
          <span class="text-gray-500 dark:text-gray-400">Analyzed by:</span>
          <span class="ml-2">{{ analysis.metadata.user_id }}</span>
        </div>
      </div>
      
      <div v-if="analysis.metadata.model_version" class="mt-3 text-xs text-gray-500 dark:text-gray-400">
        Model: {{ analysis.metadata.model_version }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { format } from 'date-fns'
import type { AnalysisResponse } from '@/types/api'

interface Props {
  analysis: AnalysisResponse
}

defineProps<Props>()

function formatTimestamp(timestamp: string): string {
  try {
    return format(new Date(timestamp), 'MMM dd, yyyy HH:mm:ss')
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

function getConfidenceBarColor(confidence: number): string {
  if (confidence >= 0.8) {
    return 'bg-success-500'
  } else if (confidence >= 0.6) {
    return 'bg-warning-500'
  } else {
    return 'bg-danger-500'
  }
}
</script>