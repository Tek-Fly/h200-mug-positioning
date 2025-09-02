<template>
  <AppLayout>
    <div class="space-y-6">
      <!-- Header -->
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 class="text-2xl font-bold text-gray-900 dark:text-white">
            System Metrics
          </h1>
          <p class="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Detailed performance and usage analytics
          </p>
        </div>
        <div class="mt-4 sm:mt-0 flex items-center space-x-3">
          <select
            v-model="timeRange"
            class="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last Week</option>
            <option value="30d">Last Month</option>
          </select>
        </div>
      </div>

      <!-- Key Performance Indicators -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Average GPU Usage"
          :value="75.2"
          suffix="%"
          :trend="5.3"
          color="primary"
        >
          <template #icon>
            <CpuChipIcon class="h-6 w-6" />
          </template>
        </MetricCard>

        <MetricCard
          title="Requests Processed"
          :value="12847"
          :trend="12.1"
          color="success"
        >
          <template #icon>
            <BoltIcon class="h-6 w-6" />
          </template>
        </MetricCard>

        <MetricCard
          title="Average Latency"
          :value="156"
          suffix="ms"
          :trend="-8.2"
          color="warning"
        >
          <template #icon>
            <ClockIcon class="h-6 w-6" />
          </template>
        </MetricCard>

        <MetricCard
          title="Cache Hit Rate"
          :value="87.3"
          suffix="%"
          :trend="2.1"
          color="success"
        >
          <template #icon>
            <RocketLaunchIcon class="h-6 w-6" />
          </template>
        </MetricCard>
      </div>

      <!-- Charts Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- GPU Utilization Chart -->
        <div class="card p-6">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            GPU Utilization Over Time
          </h3>
          <div class="h-64 flex items-center justify-center text-gray-500 dark:text-gray-400">
            <ChartBarIcon class="h-8 w-8 mr-2" />
            Interactive chart would be rendered here
          </div>
        </div>

        <!-- Request Volume Chart -->
        <div class="card p-6">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Request Volume
          </h3>
          <div class="h-64 flex items-center justify-center text-gray-500 dark:text-gray-400">
            <ChartBarIcon class="h-8 w-8 mr-2" />
            Interactive chart would be rendered here
          </div>
        </div>

        <!-- Response Time Chart -->
        <div class="card p-6">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Response Time Distribution
          </h3>
          <div class="h-64 flex items-center justify-center text-gray-500 dark:text-gray-400">
            <ChartBarIcon class="h-8 w-8 mr-2" />
            Interactive chart would be rendered here
          </div>
        </div>

        <!-- Error Rate Chart -->
        <div class="card p-6">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Error Rate Trends
          </h3>
          <div class="h-64 flex items-center justify-center text-gray-500 dark:text-gray-400">
            <ChartBarIcon class="h-8 w-8 mr-2" />
            Interactive chart would be rendered here
          </div>
        </div>
      </div>

      <!-- Detailed Metrics Table -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Detailed Metrics
        </h3>
        
        <div class="overflow-x-auto">
          <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-600">
            <thead class="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Metric
                </th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Current
                </th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Average
                </th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Peak
                </th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody class="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-600">
              <tr v-for="metric in detailedMetrics" :key="metric.name">
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                  {{ metric.name }}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                  {{ metric.current }}{{ metric.unit }}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  {{ metric.average }}{{ metric.unit }}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  {{ metric.peak }}{{ metric.unit }}
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <span
                    :class="[
                      'px-2 inline-flex text-xs leading-5 font-semibold rounded-full',
                      getStatusColor(metric.status)
                    ]"
                  >
                    {{ metric.status }}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </AppLayout>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import {
  CpuChipIcon,
  BoltIcon,
  ClockIcon,
  RocketLaunchIcon,
  ChartBarIcon
} from '@heroicons/vue/24/outline'
import AppLayout from '@/components/layout/AppLayout.vue'
import MetricCard from '@/components/dashboard/MetricCard.vue'

const timeRange = ref('24h')

const detailedMetrics = [
  {
    name: 'GPU Utilization',
    current: 75.2,
    average: 68.5,
    peak: 94.1,
    unit: '%',
    status: 'healthy'
  },
  {
    name: 'Memory Usage',
    current: 12.4,
    average: 11.2,
    peak: 15.8,
    unit: 'GB',
    status: 'healthy'
  },
  {
    name: 'Request Rate',
    current: 45.2,
    average: 38.7,
    peak: 67.3,
    unit: '/min',
    status: 'healthy'
  },
  {
    name: 'Error Rate',
    current: 0.12,
    average: 0.08,
    peak: 0.45,
    unit: '%',
    status: 'healthy'
  },
  {
    name: 'Response Time P95',
    current: 156,
    average: 142,
    peak: 289,
    unit: 'ms',
    status: 'warning'
  },
  {
    name: 'Cache Hit Rate',
    current: 87.3,
    average: 85.1,
    peak: 92.4,
    unit: '%',
    status: 'healthy'
  }
]

function getStatusColor(status: string): string {
  switch (status) {
    case 'healthy':
      return 'bg-success-100 text-success-800'
    case 'warning':
      return 'bg-warning-100 text-warning-800'
    case 'critical':
      return 'bg-danger-100 text-danger-800'
    default:
      return 'bg-gray-100 text-gray-800'
  }
}
</script>