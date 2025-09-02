<template>
  <div class="space-y-4">
    <!-- Drop Zone -->
    <div
      :class="[
        'upload-area relative',
        { 'dragover': isDragging }
      ]"
      @drop="handleDrop"
      @dragover.prevent="handleDragOver"
      @dragleave="handleDragLeave"
      @click="triggerFileInput"
    >
      <input
        ref="fileInput"
        type="file"
        accept="image/*"
        class="hidden"
        @change="handleFileSelect"
      />
      
      <div class="text-center">
        <PhotoIcon class="mx-auto h-12 w-12 text-gray-400" />
        <div class="mt-4">
          <p class="text-lg font-medium text-gray-900 dark:text-white">
            Drop image here or click to browse
          </p>
          <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
            PNG, JPG, GIF up to 10MB
          </p>
        </div>
      </div>
    </div>

    <!-- Selected File Info -->
    <div v-if="selectedFile" class="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-3">
          <div class="flex-shrink-0">
            <PhotoIcon class="h-8 w-8 text-primary-600" />
          </div>
          <div>
            <p class="text-sm font-medium text-gray-900 dark:text-white">
              {{ selectedFile.name }}
            </p>
            <p class="text-xs text-gray-500 dark:text-gray-400">
              {{ formatFileSize(selectedFile.size) }} • {{ getFileType(selectedFile.type) }}
            </p>
          </div>
        </div>
        
        <div class="flex items-center space-x-2">
          <button
            v-if="!loading"
            type="button"
            class="btn btn-primary btn-sm"
            @click="analyzeImage"
          >
            <CpuChipIcon class="h-4 w-4 mr-2" />
            Analyze
          </button>
          
          <button
            type="button"
            class="btn btn-secondary btn-sm"
            @click="clearSelection"
            :disabled="loading"
          >
            <XMarkIcon class="h-4 w-4" />
          </button>
        </div>
      </div>

      <!-- Progress bar when analyzing -->
      <div v-if="loading" class="mt-4">
        <div class="flex items-center justify-between text-sm mb-2">
          <span class="text-gray-600 dark:text-gray-400">Analyzing image...</span>
          <span class="text-gray-600 dark:text-gray-400">{{ progress }}%</span>
        </div>
        <div class="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
          <div
            class="bg-primary-600 h-2 rounded-full transition-all duration-300"
            :style="{ width: `${progress}%` }"
          ></div>
        </div>
      </div>
    </div>

    <!-- Upload Tips -->
    <div class="bg-primary-50 dark:bg-primary-900/20 rounded-lg p-4">
      <h4 class="text-sm font-medium text-primary-800 dark:text-primary-200 mb-2">
        📸 Best Results Tips
      </h4>
      <ul class="text-xs text-primary-700 dark:text-primary-300 space-y-1">
        <li>• Use clear, well-lit images</li>
        <li>• Ensure mugs are fully visible</li>
        <li>• Include surrounding context (table, coaster, etc.)</li>
        <li>• Higher resolution images work better</li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { PhotoIcon, CpuChipIcon, XMarkIcon } from '@heroicons/vue/24/outline'
import { useNotificationStore } from '@/stores/notifications'

interface Props {
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  loading: false
})

const emit = defineEmits<{
  fileSelected: [file: File]
  analysisComplete: [file: File]
}>()

const fileInput = ref<HTMLInputElement>()
const selectedFile = ref<File | null>(null)
const isDragging = ref(false)
const progress = ref(0)

const notificationStore = useNotificationStore()

function triggerFileInput(): void {
  if (props.loading) return
  fileInput.value?.click()
}

function handleFileSelect(event: Event): void {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    processFile(file)
  }
}

function handleDrop(event: DragEvent): void {
  event.preventDefault()
  isDragging.value = false
  
  if (props.loading) return
  
  const file = event.dataTransfer?.files[0]
  if (file) {
    processFile(file)
  }
}

function handleDragOver(event: DragEvent): void {
  event.preventDefault()
  if (!props.loading) {
    isDragging.value = true
  }
}

function handleDragLeave(): void {
  isDragging.value = false
}

function processFile(file: File): void {
  // Validate file
  if (!file.type.startsWith('image/')) {
    notificationStore.error('Invalid File', 'Please select an image file')
    return
  }
  
  if (file.size > 10 * 1024 * 1024) { // 10MB
    notificationStore.error('File Too Large', 'Please select a file smaller than 10MB')
    return
  }
  
  selectedFile.value = file
  emit('fileSelected', file)
}

function analyzeImage(): void {
  if (!selectedFile.value) return
  
  // Start progress animation
  progress.value = 0
  const interval = setInterval(() => {
    if (progress.value < 95) {
      progress.value += Math.random() * 15
    }
  }, 200)
  
  emit('analysisComplete', selectedFile.value)
  
  // Stop progress when loading ends
  const stopProgress = () => {
    clearInterval(interval)
    progress.value = 100
    setTimeout(() => {
      if (!props.loading) {
        progress.value = 0
      }
    }, 1000)
  }
  
  // Watch for loading to stop
  const unwatch = watch(() => props.loading, (loading) => {
    if (!loading) {
      stopProgress()
      unwatch()
    }
  })
}

function clearSelection(): void {
  selectedFile.value = null
  if (fileInput.value) {
    fileInput.value.value = ''
  }
  progress.value = 0
}

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B'
  
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

function getFileType(mimeType: string): string {
  return mimeType.split('/')[1].toUpperCase()
}
</script>