<template>
  <div class="metric-card bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
    <div class="flex items-center justify-between mb-4">
      <div :class="['p-3 rounded-lg', getColorClasses(color)]">
        <slot name="icon">
          <ChartBarIcon class="h-6 w-6" />
        </slot>
      </div>
      
      <div
        v-if="trend !== undefined && !loading"
        :class="[
          'flex items-center text-sm font-medium px-2 py-1 rounded-full',
          trend > 0 
            ? 'text-success-700 bg-success-100 dark:text-success-200 dark:bg-success-900'
            : trend < 0
            ? 'text-danger-700 bg-danger-100 dark:text-danger-200 dark:bg-danger-900'
            : 'text-gray-700 bg-gray-100 dark:text-gray-300 dark:bg-gray-700'
        ]"
      >
        <component
          :is="trend > 0 ? ArrowTrendingUpIcon : trend < 0 ? ArrowTrendingDownIcon : MinusIcon"
          class="h-4 w-4 mr-1"
        />
        {{ Math.abs(trend) }}{{ trend !== 0 ? '%' : '' }}
      </div>
    </div>
    
    <div>
      <h3 class="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
        {{ title }}
      </h3>
      
      <div class="flex items-baseline">
        <div v-if="loading" class="loading-spinner h-8 w-8"></div>
        <div v-else class="text-2xl font-bold text-gray-900 dark:text-white">
          {{ formatValue(value) }}{{ suffix }}
        </div>
      </div>
      
      <div v-if="subtitle" class="text-xs text-gray-500 dark:text-gray-400 mt-1">
        {{ subtitle }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import {
  ChartBarIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  MinusIcon
} from '@heroicons/vue/24/outline'

interface Props {
  title: string
  value: number
  decimals?: number
  suffix?: string
  subtitle?: string
  trend?: number
  color?: 'primary' | 'success' | 'warning' | 'danger' | 'gray'
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  decimals: 0,
  suffix: '',
  color: 'primary',
  loading: false
})

function formatValue(val: number): string {
  if (props.decimals === 0) {
    return Math.round(val).toLocaleString()
  }
  return val.toFixed(props.decimals)
}

function getColorClasses(color: string): string {
  switch (color) {
    case 'success':
      return 'bg-success-100 text-success-600 dark:bg-success-900 dark:text-success-400'
    case 'warning':
      return 'bg-warning-100 text-warning-600 dark:bg-warning-900 dark:text-warning-400'
    case 'danger':
      return 'bg-danger-100 text-danger-600 dark:bg-danger-900 dark:text-danger-400'
    case 'gray':
      return 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
    default:
      return 'bg-primary-100 text-primary-600 dark:bg-primary-900 dark:text-primary-400'
  }
}
</script>