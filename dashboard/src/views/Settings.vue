<template>
  <AppLayout>
    <div class="space-y-6">
      <!-- Header -->
      <div>
        <h1 class="text-2xl font-bold text-gray-900 dark:text-white">
          Settings
        </h1>
        <p class="mt-1 text-sm text-gray-600 dark:text-gray-400">
          Configure your dashboard and system preferences
        </p>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Settings Form -->
        <div class="lg:col-span-2 space-y-6">
          <!-- Appearance Settings -->
          <div class="card p-6">
            <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Appearance
            </h2>
            
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Theme
                </label>
                <div class="flex space-x-3">
                  <button
                    v-for="option in themeOptions"
                    :key="option.value"
                    type="button"
                    :class="[
                      'flex-1 p-3 border rounded-lg text-sm font-medium transition-colors duration-200',
                      themeStore.theme === option.value
                        ? 'border-primary-500 bg-primary-50 text-primary-700 dark:bg-primary-900 dark:text-primary-200'
                        : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                    ]"
                    @click="themeStore.setTheme(option.value)"
                  >
                    <component :is="option.icon" class="h-5 w-5 mx-auto mb-1" />
                    {{ option.label }}
                  </button>
                </div>
              </div>
            </div>
          </div>

          <!-- Dashboard Settings -->
          <div class="card p-6">
            <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Dashboard Preferences
            </h2>
            
            <div class="space-y-4">
              <div class="flex items-center justify-between">
                <div>
                  <label class="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Auto-refresh Dashboard
                  </label>
                  <p class="text-xs text-gray-500 dark:text-gray-400">
                    Automatically refresh dashboard data every 30 seconds
                  </p>
                </div>
                <button
                  type="button"
                  :class="[
                    'relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
                    settings.autoRefresh ? 'bg-primary-600' : 'bg-gray-200 dark:bg-gray-600'
                  ]"
                  @click="settings.autoRefresh = !settings.autoRefresh"
                >
                  <span
                    :class="[
                      'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out',
                      settings.autoRefresh ? 'translate-x-5' : 'translate-x-0'
                    ]"
                  />
                </button>
              </div>

              <div class="flex items-center justify-between">
                <div>
                  <label class="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Show Advanced Metrics
                  </label>
                  <p class="text-xs text-gray-500 dark:text-gray-400">
                    Display detailed performance metrics in the dashboard
                  </p>
                </div>
                <button
                  type="button"
                  :class="[
                    'relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
                    settings.showAdvancedMetrics ? 'bg-primary-600' : 'bg-gray-200 dark:bg-gray-600'
                  ]"
                  @click="settings.showAdvancedMetrics = !settings.showAdvancedMetrics"
                >
                  <span
                    :class="[
                      'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out',
                      settings.showAdvancedMetrics ? 'translate-x-5' : 'translate-x-0'
                    ]"
                  />
                </button>
              </div>

              <div class="flex items-center justify-between">
                <div>
                  <label class="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Enable Notifications
                  </label>
                  <p class="text-xs text-gray-500 dark:text-gray-400">
                    Show browser notifications for important system events
                  </p>
                </div>
                <button
                  type="button"
                  :class="[
                    'relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
                    settings.enableNotifications ? 'bg-primary-600' : 'bg-gray-200 dark:bg-gray-600'
                  ]"
                  @click="settings.enableNotifications = !settings.enableNotifications"
                >
                  <span
                    :class="[
                      'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out',
                      settings.enableNotifications ? 'translate-x-5' : 'translate-x-0'
                    ]"
                  />
                </button>
              </div>
            </div>
          </div>

          <!-- Analysis Settings -->
          <div class="card p-6">
            <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Analysis Defaults
            </h2>
            
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Default Confidence Threshold
                </label>
                <input
                  v-model.number="settings.defaultConfidenceThreshold"
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.1"
                  class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                />
                <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {{ (settings.defaultConfidenceThreshold * 100).toFixed(0) }}%
                </div>
              </div>

              <div>
                <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Auto-save Analysis Results
                </label>
                <select
                  v-model="settings.autoSaveResults"
                  class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500"
                >
                  <option value="never">Never</option>
                  <option value="successful">Successful analyses only</option>
                  <option value="always">All analyses</option>
                </select>
              </div>
            </div>
          </div>

          <!-- Save Button -->
          <div class="flex justify-end">
            <button
              type="button"
              class="btn btn-primary"
              @click="saveSettings"
              :disabled="saving"
            >
              <span v-if="saving" class="loading-spinner h-4 w-4 mr-2"></span>
              {{ saving ? 'Saving...' : 'Save Settings' }}
            </button>
          </div>
        </div>

        <!-- System Info -->
        <div class="space-y-6">
          <div class="card p-6">
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              System Information
            </h3>
            
            <div class="space-y-3 text-sm">
              <div class="flex justify-between">
                <span class="text-gray-600 dark:text-gray-400">Dashboard Version</span>
                <span class="font-medium text-gray-900 dark:text-white">v1.0.0</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-600 dark:text-gray-400">API Version</span>
                <span class="font-medium text-gray-900 dark:text-white">v1.0.0</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-600 dark:text-gray-400">Last Updated</span>
                <span class="font-medium text-gray-900 dark:text-white">{{ new Date().toLocaleDateString() }}</span>
              </div>
            </div>
          </div>

          <div class="card p-6">
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Quick Actions
            </h3>
            
            <div class="space-y-3">
              <button
                type="button"
                class="w-full btn btn-secondary btn-sm"
                @click="clearCache"
              >
                Clear Browser Cache
              </button>
              
              <button
                type="button"
                class="w-full btn btn-secondary btn-sm"
                @click="exportSettings"
              >
                Export Settings
              </button>
              
              <button
                type="button"
                class="w-full btn btn-secondary btn-sm"
                @click="resetSettings"
              >
                Reset to Defaults
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </AppLayout>
</template>

<script setup lang="ts">
import { reactive, ref } from 'vue'
import { SunIcon, MoonIcon, ComputerDesktopIcon } from '@heroicons/vue/24/outline'
import AppLayout from '@/components/layout/AppLayout.vue'
import { useThemeStore } from '@/stores/theme'
import { useNotificationStore } from '@/stores/notifications'
import type { Theme } from '@/stores/theme'

const themeStore = useThemeStore()
const notificationStore = useNotificationStore()

const saving = ref(false)

const themeOptions = [
  { value: 'light' as Theme, label: 'Light', icon: SunIcon },
  { value: 'dark' as Theme, label: 'Dark', icon: MoonIcon },
  { value: 'system' as Theme, label: 'System', icon: ComputerDesktopIcon }
]

const settings = reactive({
  autoRefresh: true,
  showAdvancedMetrics: false,
  enableNotifications: true,
  defaultConfidenceThreshold: 0.7,
  autoSaveResults: 'successful' as 'never' | 'successful' | 'always'
})

async function saveSettings(): Promise<void> {
  saving.value = true
  
  try {
    // Save settings to localStorage
    localStorage.setItem('dashboard_settings', JSON.stringify(settings))
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    notificationStore.success('Settings Saved', 'Your preferences have been updated')
    
  } catch (error) {
    notificationStore.error('Save Error', 'Failed to save settings')
  } finally {
    saving.value = false
  }
}

function clearCache(): void {
  // Clear browser cache
  if ('caches' in window) {
    caches.keys().then(names => {
      names.forEach(name => {
        caches.delete(name)
      })
    })
  }
  
  // Clear localStorage except auth token
  const authToken = localStorage.getItem('auth_token')
  localStorage.clear()
  if (authToken) {
    localStorage.setItem('auth_token', authToken)
  }
  
  notificationStore.success('Cache Cleared', 'Browser cache has been cleared')
}

function exportSettings(): void {
  try {
    const exportData = {
      settings,
      theme: themeStore.theme,
      exported_at: new Date().toISOString()
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    
    const link = document.createElement('a')
    link.href = url
    link.download = 'dashboard-settings.json'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    URL.revokeObjectURL(url)
    
    notificationStore.success('Settings Exported', 'Settings file downloaded')
    
  } catch (error) {
    notificationStore.error('Export Error', 'Failed to export settings')
  }
}

function resetSettings(): void {
  if (confirm('Are you sure you want to reset all settings to defaults?')) {
    // Reset to defaults
    Object.assign(settings, {
      autoRefresh: true,
      showAdvancedMetrics: false,
      enableNotifications: true,
      defaultConfidenceThreshold: 0.7,
      autoSaveResults: 'successful'
    })
    
    themeStore.setTheme('system')
    
    notificationStore.success('Settings Reset', 'All settings have been reset to defaults')
  }
}

// Load settings from localStorage on mount
const savedSettings = localStorage.getItem('dashboard_settings')
if (savedSettings) {
  try {
    const parsed = JSON.parse(savedSettings)
    Object.assign(settings, parsed)
  } catch (error) {
    console.error('Failed to load saved settings:', error)
  }
}
</script>