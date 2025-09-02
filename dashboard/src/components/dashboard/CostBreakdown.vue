<template>
  <div class="space-y-4">
    <!-- Total Cost -->
    <div class="text-center py-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
      <div class="text-2xl font-bold text-gray-900 dark:text-white">
        ${{ data.total_cost.toFixed(2) }}
      </div>
      <div class="text-sm text-gray-500 dark:text-gray-400 capitalize">
        {{ data.period }} Total
      </div>
    </div>

    <!-- Cost Breakdown -->
    <div class="space-y-3">
      <!-- Compute Cost -->
      <div class="flex items-center justify-between">
        <div class="flex items-center">
          <div class="w-3 h-3 rounded-full bg-primary-500 mr-3"></div>
          <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Compute</span>
        </div>
        <div class="text-sm text-gray-900 dark:text-white font-medium">
          ${{ data.compute_cost.toFixed(2) }}
        </div>
      </div>
      <div class="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
        <div
          class="bg-primary-500 h-2 rounded-full transition-all duration-300"
          :style="{ width: `${(data.compute_cost / data.total_cost) * 100}%` }"
        ></div>
      </div>

      <!-- Storage Cost -->
      <div class="flex items-center justify-between">
        <div class="flex items-center">
          <div class="w-3 h-3 rounded-full bg-success-500 mr-3"></div>
          <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Storage</span>
        </div>
        <div class="text-sm text-gray-900 dark:text-white font-medium">
          ${{ data.storage_cost.toFixed(2) }}
        </div>
      </div>
      <div class="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
        <div
          class="bg-success-500 h-2 rounded-full transition-all duration-300"
          :style="{ width: `${(data.storage_cost / data.total_cost) * 100}%` }"
        ></div>
      </div>

      <!-- Network Cost -->
      <div class="flex items-center justify-between">
        <div class="flex items-center">
          <div class="w-3 h-3 rounded-full bg-warning-500 mr-3"></div>
          <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Network</span>
        </div>
        <div class="text-sm text-gray-900 dark:text-white font-medium">
          ${{ data.network_cost.toFixed(2) }}
        </div>
      </div>
      <div class="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
        <div
          class="bg-warning-500 h-2 rounded-full transition-all duration-300"
          :style="{ width: `${(data.network_cost / data.total_cost) * 100}%` }"
        ></div>
      </div>
    </div>

    <!-- Additional Metrics -->
    <div class="pt-4 border-t border-gray-200 dark:border-gray-600">
      <div class="grid grid-cols-1 gap-3 text-sm">
        <div class="flex justify-between">
          <span class="text-gray-600 dark:text-gray-400">Cost per Request</span>
          <span class="font-medium text-gray-900 dark:text-white">
            ${{ data.cost_per_request.toFixed(4) }}
          </span>
        </div>
        
        <div v-if="data.breakdown.gpu_hours" class="flex justify-between">
          <span class="text-gray-600 dark:text-gray-400">GPU Hours</span>
          <span class="font-medium text-gray-900 dark:text-white">
            {{ data.breakdown.gpu_hours }}h
          </span>
        </div>
        
        <div v-if="data.breakdown.storage_gb" class="flex justify-between">
          <span class="text-gray-600 dark:text-gray-400">Storage Used</span>
          <span class="font-medium text-gray-900 dark:text-white">
            {{ data.breakdown.storage_gb }}GB
          </span>
        </div>
        
        <div v-if="data.breakdown.network_gb" class="flex justify-between">
          <span class="text-gray-600 dark:text-gray-400">Data Transfer</span>
          <span class="font-medium text-gray-900 dark:text-white">
            {{ data.breakdown.network_gb }}GB
          </span>
        </div>
      </div>
    </div>

    <!-- Cost Optimization Tips -->
    <div class="mt-4 p-3 bg-primary-50 dark:bg-primary-900/20 rounded-lg">
      <h4 class="text-sm font-medium text-primary-800 dark:text-primary-200 mb-2">
        💡 Cost Optimization
      </h4>
      <ul class="text-xs text-primary-700 dark:text-primary-300 space-y-1">
        <li>• Enable auto-shutdown to reduce idle costs</li>
        <li>• Use serverless mode for variable workloads</li>
        <li>• Monitor cache hit rate to reduce GPU usage</li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { CostMetrics } from '@/types/api'

interface Props {
  data: CostMetrics
}

defineProps<Props>()
</script>