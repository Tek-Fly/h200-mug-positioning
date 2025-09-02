<template>
  <div class="space-y-4">
    <!-- CPU Usage -->
    <div>
      <div class="flex justify-between items-center mb-2">
        <span class="text-sm font-medium text-gray-700 dark:text-gray-300">CPU</span>
        <span class="text-sm text-gray-500 dark:text-gray-400">{{ data.cpu_percent.toFixed(1) }}%</span>
      </div>
      <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
        <div
          class="h-2 rounded-full transition-all duration-300"
          :class="getCpuBarColor(data.cpu_percent)"
          :style="{ width: `${Math.min(data.cpu_percent, 100)}%` }"
        ></div>
      </div>
    </div>

    <!-- Memory Usage -->
    <div>
      <div class="flex justify-between items-center mb-2">
        <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Memory</span>
        <span class="text-sm text-gray-500 dark:text-gray-400">
          {{ formatBytes(data.memory_used_mb * 1024 * 1024) }} / {{ formatBytes(data.memory_total_mb * 1024 * 1024) }}
        </span>
      </div>
      <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
        <div
          class="h-2 rounded-full transition-all duration-300"
          :class="getMemoryBarColor(memoryPercent)"
          :style="{ width: `${Math.min(memoryPercent, 100)}%` }"
        ></div>
      </div>
    </div>

    <!-- GPU Memory Usage (if available) -->
    <div v-if="data.gpu_memory_used_mb && data.gpu_memory_total_mb">
      <div class="flex justify-between items-center mb-2">
        <span class="text-sm font-medium text-gray-700 dark:text-gray-300">GPU Memory</span>
        <span class="text-sm text-gray-500 dark:text-gray-400">
          {{ formatBytes(data.gpu_memory_used_mb * 1024 * 1024) }} / {{ formatBytes(data.gpu_memory_total_mb * 1024 * 1024) }}
        </span>
      </div>
      <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
        <div
          class="h-2 rounded-full transition-all duration-300"
          :class="getGpuMemoryBarColor(gpuMemoryPercent)"
          :style="{ width: `${Math.min(gpuMemoryPercent, 100)}%` }"
        ></div>
      </div>
    </div>

    <!-- Disk Usage -->
    <div>
      <div class="flex justify-between items-center mb-2">
        <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Disk</span>
        <span class="text-sm text-gray-500 dark:text-gray-400">
          {{ data.disk_used_gb.toFixed(1) }}GB / {{ data.disk_total_gb.toFixed(1) }}GB
        </span>
      </div>
      <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
        <div
          class="h-2 rounded-full transition-all duration-300"
          :class="getDiskBarColor(diskPercent)"
          :style="{ width: `${Math.min(diskPercent, 100)}%` }"
        ></div>
      </div>
    </div>

    <!-- Resource Summary -->
    <div class="grid grid-cols-2 gap-4 mt-6 pt-4 border-t border-gray-200 dark:border-gray-600">
      <div class="text-center">
        <div class="text-lg font-semibold text-gray-900 dark:text-white">
          {{ data.cpu_percent.toFixed(1) }}%
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400">CPU Usage</div>
      </div>
      <div class="text-center">
        <div class="text-lg font-semibold text-gray-900 dark:text-white">
          {{ memoryPercent.toFixed(1) }}%
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400">Memory Usage</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ResourceUsage } from '@/types/api'

interface Props {
  data: ResourceUsage
  height?: number
}

const props = withDefaults(defineProps<Props>(), {
  height: 300
})

const memoryPercent = computed(() => (props.data.memory_used_mb / props.data.memory_total_mb) * 100)
const diskPercent = computed(() => (props.data.disk_used_gb / props.data.disk_total_gb) * 100)
const gpuMemoryPercent = computed(() => {
  if (!props.data.gpu_memory_used_mb || !props.data.gpu_memory_total_mb) return 0
  return (props.data.gpu_memory_used_mb / props.data.gpu_memory_total_mb) * 100
})

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

function getCpuBarColor(percent: number): string {
  if (percent >= 90) return 'bg-danger-500'
  if (percent >= 70) return 'bg-warning-500'
  return 'bg-success-500'
}

function getMemoryBarColor(percent: number): string {
  if (percent >= 90) return 'bg-danger-500'
  if (percent >= 80) return 'bg-warning-500'
  return 'bg-primary-500'
}

function getGpuMemoryBarColor(percent: number): string {
  if (percent >= 95) return 'bg-danger-500'
  if (percent >= 85) return 'bg-warning-500'
  return 'bg-purple-500'
}

function getDiskBarColor(percent: number): string {
  if (percent >= 95) return 'bg-danger-500'
  if (percent >= 85) return 'bg-warning-500'
  return 'bg-gray-500'
}
</script>