<template>
  <div class="chart-container">
    <canvas ref="chartCanvas"></canvas>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import {
  Chart,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  type ChartConfiguration
} from 'chart.js'
import { useDashboardStore } from '@/stores/dashboard'
import { useThemeStore } from '@/stores/theme'

// Register Chart.js components
Chart.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

interface Props {
  height?: number
}

const props = withDefaults(defineProps<Props>(), {
  height: 400
})

const chartCanvas = ref<HTMLCanvasElement>()
const chart = ref<Chart>()
const dashboardStore = useDashboardStore()
const themeStore = useThemeStore()

// Mock data - in real app this would come from API
const generateMockData = () => {
  const now = new Date()
  const labels = []
  const gpuData = []
  const requestData = []
  const latencyData = []
  
  for (let i = 23; i >= 0; i--) {
    const time = new Date(now.getTime() - i * 5 * 60 * 1000) // 5 minute intervals
    labels.push(time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))
    
    // Generate realistic mock data
    gpuData.push(Math.random() * 40 + 50) // 50-90% GPU usage
    requestData.push(Math.random() * 50 + 10) // 10-60 requests/sec
    latencyData.push(Math.random() * 100 + 100) // 100-200ms latency
  }
  
  return { labels, gpuData, requestData, latencyData }
}

const createChart = () => {
  if (!chartCanvas.value) return

  const { labels, gpuData, requestData, latencyData } = generateMockData()
  
  const isDark = themeStore.isDark
  const textColor = isDark ? '#f3f4f6' : '#374151'
  const gridColor = isDark ? '#374151' : '#e5e7eb'

  const config: ChartConfiguration = {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'GPU Utilization (%)',
          data: gpuData,
          borderColor: '#3b82f6',
          backgroundColor: '#3b82f6',
          tension: 0.4,
          yAxisID: 'y'
        },
        {
          label: 'Requests/sec',
          data: requestData,
          borderColor: '#22c55e',
          backgroundColor: '#22c55e',
          tension: 0.4,
          yAxisID: 'y'
        },
        {
          label: 'Latency (ms)',
          data: latencyData,
          borderColor: '#f59e0b',
          backgroundColor: '#f59e0b',
          tension: 0.4,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: textColor,
            usePointStyle: true,
            padding: 20
          }
        }
      },
      scales: {
        x: {
          display: true,
          grid: {
            color: gridColor
          },
          ticks: {
            color: textColor
          }
        },
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          grid: {
            color: gridColor
          },
          ticks: {
            color: textColor
          },
          title: {
            display: true,
            text: 'Percentage / Requests',
            color: textColor
          }
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          grid: {
            drawOnChartArea: false
          },
          ticks: {
            color: textColor
          },
          title: {
            display: true,
            text: 'Latency (ms)',
            color: textColor
          }
        }
      },
      elements: {
        point: {
          radius: 3,
          hoverRadius: 6
        }
      }
    }
  }

  chart.value = new Chart(chartCanvas.value, config)
}

const updateChart = () => {
  if (!chart.value) return

  const { labels, gpuData, requestData, latencyData } = generateMockData()
  
  chart.value.data.labels = labels
  chart.value.data.datasets[0].data = gpuData
  chart.value.data.datasets[1].data = requestData
  chart.value.data.datasets[2].data = latencyData
  
  chart.value.update('none')
}

const updateTheme = () => {
  if (!chart.value) return

  const isDark = themeStore.isDark
  const textColor = isDark ? '#f3f4f6' : '#374151'
  const gridColor = isDark ? '#374151' : '#e5e7eb'

  // Update colors
  if (chart.value.options.plugins?.legend?.labels) {
    chart.value.options.plugins.legend.labels.color = textColor
  }

  const scales = chart.value.options.scales
  if (scales) {
    Object.values(scales).forEach((scale: any) => {
      if (scale.grid) scale.grid.color = gridColor
      if (scale.ticks) scale.ticks.color = textColor
      if (scale.title) scale.title.color = textColor
    })
  }

  chart.value.update('none')
}

// Update chart data periodically
let updateInterval: NodeJS.Timeout | null = null

onMounted(() => {
  createChart()
  
  // Update chart every 5 seconds with new data
  updateInterval = setInterval(updateChart, 5000)
})

onUnmounted(() => {
  if (chart.value) {
    chart.value.destroy()
  }
  if (updateInterval) {
    clearInterval(updateInterval)
  }
})

// Watch for theme changes
watch(() => themeStore.isDark, updateTheme)
</script>