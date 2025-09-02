<template>
  <AppLayout>
    <div class="space-y-6">
      <!-- Header -->
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 class="text-2xl font-bold text-gray-900 dark:text-white">
            Mug Position Analysis
          </h1>
          <p class="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Upload images to analyze mug positioning with AI
          </p>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Image Upload Section -->
        <div class="space-y-6">
          <!-- Upload Area -->
          <div class="card p-6">
            <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Upload Image
            </h2>
            
            <ImageUpload
              @file-selected="handleFileSelected"
              @analysis-complete="handleAnalysisComplete"
              :loading="analyzing"
            />
          </div>

          <!-- Analysis Settings -->
          <div class="card p-6">
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Analysis Settings
            </h3>
            
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Confidence Threshold
                </label>
                <input
                  v-model.number="settings.confidenceThreshold"
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.1"
                  class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                />
                <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {{ (settings.confidenceThreshold * 100).toFixed(0) }}%
                </div>
              </div>

              <div>
                <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Calibration (mm per pixel)
                </label>
                <input
                  v-model.number="settings.calibrationMmPerPixel"
                  type="number"
                  step="0.1"
                  min="0"
                  placeholder="Optional"
                  class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500"
                />
                <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Leave blank for pixel measurements
                </div>
              </div>

              <div>
                <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Rules Context
                </label>
                <textarea
                  v-model="settings.rulesContext"
                  rows="3"
                  placeholder="e.g., 'Mug should be centered on the coaster'"
                  class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500"
                />
              </div>

              <div class="flex items-center">
                <input
                  v-model="settings.includeFeedback"
                  type="checkbox"
                  class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 dark:border-gray-600 rounded"
                />
                <label class="ml-2 text-sm text-gray-700 dark:text-gray-300">
                  Include positioning feedback
                </label>
              </div>
            </div>
          </div>
        </div>

        <!-- Analysis Results Section -->
        <div class="space-y-6">
          <!-- Image Preview and Analysis -->
          <div class="card p-6">
            <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Analysis Results
            </h2>
            
            <div v-if="!currentImage && !analysisResult" class="text-center py-12">
              <PhotoIcon class="mx-auto h-12 w-12 text-gray-400" />
              <h3 class="mt-2 text-sm font-medium text-gray-900 dark:text-white">
                No image uploaded
              </h3>
              <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
                Upload an image to start analysis
              </p>
            </div>

            <div v-else-if="analyzing" class="text-center py-12">
              <div class="loading-spinner h-12 w-12 mx-auto mb-4"></div>
              <h3 class="text-sm font-medium text-gray-900 dark:text-white">
                Analyzing image...
              </h3>
              <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
                Processing with AI models
              </p>
            </div>

            <div v-else-if="currentImage">
              <ImageAnalysisViewer
                :image-url="currentImage"
                :analysis="analysisResult"
                @feedback-submitted="handleFeedbackSubmitted"
              />
            </div>
          </div>

          <!-- Analysis Details -->
          <div v-if="analysisResult" class="card p-6">
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Analysis Details
            </h3>
            
            <AnalysisDetails :analysis="analysisResult" />
          </div>
        </div>
      </div>

      <!-- Analysis History -->
      <div class="card p-6">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
            Analysis History
          </h2>
          <button
            type="button"
            class="btn btn-secondary btn-sm"
            @click="loadAnalysisHistory"
          >
            <ArrowPathIcon class="h-4 w-4 mr-2" />
            Refresh
          </button>
        </div>
        
        <AnalysisHistory :analyses="analysisHistory" />
      </div>
    </div>
  </AppLayout>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { PhotoIcon, ArrowPathIcon } from '@heroicons/vue/24/outline'
import AppLayout from '@/components/layout/AppLayout.vue'
import ImageUpload from '@/components/analysis/ImageUpload.vue'
import ImageAnalysisViewer from '@/components/analysis/ImageAnalysisViewer.vue'
import AnalysisDetails from '@/components/analysis/AnalysisDetails.vue'
import AnalysisHistory from '@/components/analysis/AnalysisHistory.vue'
import { apiClient } from '@/api/client'
import { useNotificationStore } from '@/stores/notifications'
import type { AnalysisResponse } from '@/types/api'

const notificationStore = useNotificationStore()

const currentImage = ref<string>('')
const analysisResult = ref<AnalysisResponse | null>(null)
const analyzing = ref(false)
const analysisHistory = ref<AnalysisResponse[]>([])

const settings = reactive({
  confidenceThreshold: 0.7,
  calibrationMmPerPixel: null as number | null,
  rulesContext: '',
  includeFeedback: true
})

function handleFileSelected(file: File): void {
  // Create preview URL
  currentImage.value = URL.createObjectURL(file)
  analysisResult.value = null
}

async function handleAnalysisComplete(file: File): Promise<void> {
  analyzing.value = true
  
  try {
    const result = await apiClient.analyzeImage(file, {
      includeFeedback: settings.includeFeedback,
      rulesContext: settings.rulesContext || undefined,
      calibrationMmPerPixel: settings.calibrationMmPerPixel || undefined,
      confidenceThreshold: settings.confidenceThreshold
    })
    
    analysisResult.value = result
    
    notificationStore.success(
      'Analysis Complete',
      `Found ${result.detections.length} mug(s) with ${result.positioning.confidence.toFixed(2)} confidence`
    )
    
    // Add to history
    analysisHistory.value.unshift(result)
    
  } catch (error: any) {
    notificationStore.error(
      'Analysis Failed', 
      error.response?.data?.detail || error.message
    )
  } finally {
    analyzing.value = false
  }
}

function handleFeedbackSubmitted(): void {
  notificationStore.success(
    'Feedback Submitted',
    'Thank you for helping improve our AI!'
  )
}

async function loadAnalysisHistory(): Promise<void> {
  try {
    // In a real app, this would fetch from the API
    // For now, keep the existing history
    notificationStore.info('History Updated', 'Analysis history refreshed')
  } catch (error: any) {
    notificationStore.error('Load Error', 'Failed to load analysis history')
  }
}

onMounted(() => {
  loadAnalysisHistory()
})
</script>