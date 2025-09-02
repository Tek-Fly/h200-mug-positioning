<template>
  <AppLayout>
    <div class="space-y-6">
      <!-- Header -->
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 class="text-2xl font-bold text-gray-900 dark:text-white">
            Rule Management
          </h1>
          <p class="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Configure positioning rules using natural language
          </p>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Natural Language Rule Creator -->
        <div class="lg:col-span-2 space-y-6">
          <div class="card p-6">
            <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Create Rule with Natural Language
            </h2>
            
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Describe your rule in plain English
                </label>
                <textarea
                  v-model="newRule.text"
                  rows="3"
                  placeholder="e.g., 'The mug should be centered on the coaster with at least 2cm margin'"
                  class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500"
                />
              </div>

              <div class="flex items-center justify-between">
                <div class="flex items-center">
                  <input
                    v-model="newRule.autoEnable"
                    type="checkbox"
                    class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 dark:border-gray-600 rounded"
                  />
                  <label class="ml-2 text-sm text-gray-700 dark:text-gray-300">
                    Enable rule automatically
                  </label>
                </div>

                <button
                  type="button"
                  :disabled="!newRule.text.trim() || creating"
                  class="btn btn-primary"
                  @click="createRule"
                >
                  <span v-if="creating" class="loading-spinner h-4 w-4 mr-2"></span>
                  <PlusIcon v-else class="h-4 w-4 mr-2" />
                  Create Rule
                </button>
              </div>
            </div>
          </div>

          <!-- Existing Rules -->
          <div class="card p-6">
            <div class="flex items-center justify-between mb-4">
              <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
                Active Rules
              </h2>
              <div class="flex items-center space-x-3">
                <select
                  v-model="ruleTypeFilter"
                  class="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500"
                >
                  <option value="">All Types</option>
                  <option value="position">Position</option>
                  <option value="distance">Distance</option>
                  <option value="alignment">Alignment</option>
                </select>
                <button
                  type="button"
                  class="btn btn-secondary btn-sm"
                  @click="refreshRules"
                  :disabled="loading"
                >
                  <ArrowPathIcon :class="['h-4 w-4 mr-2', { 'animate-spin': loading }]" />
                  Refresh
                </button>
              </div>
            </div>

            <div v-if="loading" class="text-center py-8">
              <div class="loading-spinner h-8 w-8 mx-auto"></div>
            </div>

            <div v-else-if="filteredRules.length === 0" class="text-center py-8">
              <DocumentTextIcon class="mx-auto h-12 w-12 text-gray-400" />
              <h3 class="mt-2 text-sm font-medium text-gray-900 dark:text-white">
                No rules found
              </h3>
              <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
                Create your first rule to get started
              </p>
            </div>

            <div v-else class="space-y-4">
              <RuleCard
                v-for="rule in filteredRules"
                :key="rule.id"
                :rule="rule"
                @toggle="toggleRule"
                @edit="editRule"
                @delete="deleteRule"
              />
            </div>
          </div>
        </div>

        <!-- Rule Examples & Help -->
        <div class="space-y-6">
          <div class="card p-6">
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Rule Examples
            </h3>
            
            <div class="space-y-3 text-sm">
              <div class="p-3 bg-primary-50 dark:bg-primary-900/20 rounded-lg">
                <p class="text-primary-800 dark:text-primary-200">
                  "The mug should be centered on the coaster"
                </p>
                <p class="text-xs text-primary-600 dark:text-primary-400 mt-1">
                  Creates a positioning rule for center alignment
                </p>
              </div>

              <div class="p-3 bg-success-50 dark:bg-success-900/20 rounded-lg">
                <p class="text-success-800 dark:text-success-200">
                  "Keep at least 5cm between mugs"
                </p>
                <p class="text-xs text-success-600 dark:text-success-400 mt-1">
                  Creates a distance rule with measurements
                </p>
              </div>

              <div class="p-3 bg-warning-50 dark:bg-warning-900/20 rounded-lg">
                <p class="text-warning-800 dark:text-warning-200">
                  "Alert if mug is too close to the edge"
                </p>
                <p class="text-xs text-warning-600 dark:text-warning-400 mt-1">
                  Creates an alert rule for boundary detection
                </p>
              </div>
            </div>
          </div>

          <div class="card p-6">
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Rule Statistics
            </h3>
            
            <div class="space-y-3">
              <div class="flex justify-between">
                <span class="text-sm text-gray-600 dark:text-gray-400">Total Rules</span>
                <span class="text-sm font-medium text-gray-900 dark:text-white">{{ rules.length }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-sm text-gray-600 dark:text-gray-400">Active Rules</span>
                <span class="text-sm font-medium text-gray-900 dark:text-white">{{ activeRules }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-sm text-gray-600 dark:text-gray-400">Success Rate</span>
                <span class="text-sm font-medium text-gray-900 dark:text-white">92%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </AppLayout>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import {
  PlusIcon,
  ArrowPathIcon,
  DocumentTextIcon
} from '@heroicons/vue/24/outline'
import AppLayout from '@/components/layout/AppLayout.vue'
import RuleCard from '@/components/rules/RuleCard.vue'
import { apiClient } from '@/api/client'
import { useNotificationStore } from '@/stores/notifications'
import type { Rule } from '@/types/api'

const rules = ref<Rule[]>([])
const loading = ref(false)
const creating = ref(false)
const ruleTypeFilter = ref('')

const newRule = reactive({
  text: '',
  autoEnable: true
})

const notificationStore = useNotificationStore()

const filteredRules = computed(() => {
  if (!ruleTypeFilter.value) return rules.value
  return rules.value.filter(rule => rule.type === ruleTypeFilter.value)
})

const activeRules = computed(() => {
  return rules.value.filter(rule => rule.enabled).length
})

async function createRule(): Promise<void> {
  if (!newRule.text.trim()) return
  
  creating.value = true
  
  try {
    const response = await apiClient.createRuleFromNaturalLanguage({
      text: newRule.text,
      auto_enable: newRule.autoEnable
    })
    
    rules.value.unshift(response.rule)
    
    notificationStore.success(
      'Rule Created',
      `${response.rule.name} created successfully`
    )
    
    // Reset form
    newRule.text = ''
    
  } catch (error: any) {
    notificationStore.error(
      'Creation Error',
      error.response?.data?.detail || 'Failed to create rule'
    )
  } finally {
    creating.value = false
  }
}

async function refreshRules(): Promise<void> {
  loading.value = true
  
  try {
    rules.value = await apiClient.listRules()
  } catch (error: any) {
    notificationStore.error(
      'Load Error',
      error.response?.data?.detail || 'Failed to load rules'
    )
  } finally {
    loading.value = false
  }
}

async function toggleRule(rule: Rule): Promise<void> {
  try {
    const updatedRule = await apiClient.updateRule(rule.id, {
      enabled: !rule.enabled
    })
    
    const index = rules.value.findIndex(r => r.id === rule.id)
    if (index !== -1) {
      rules.value[index] = updatedRule
    }
    
    notificationStore.success(
      'Rule Updated',
      `${rule.name} ${updatedRule.enabled ? 'enabled' : 'disabled'}`
    )
    
  } catch (error: any) {
    notificationStore.error(
      'Update Error',
      error.response?.data?.detail || 'Failed to update rule'
    )
  }
}

function editRule(rule: Rule): void {
  // TODO: Implement rule editing modal
  notificationStore.info('Coming Soon', 'Rule editing will be available soon')
}

async function deleteRule(rule: Rule): Promise<void> {
  if (!confirm(`Are you sure you want to delete "${rule.name}"?`)) return
  
  try {
    await apiClient.deleteRule(rule.id)
    
    const index = rules.value.findIndex(r => r.id === rule.id)
    if (index !== -1) {
      rules.value.splice(index, 1)
    }
    
    notificationStore.success('Rule Deleted', `${rule.name} has been deleted`)
    
  } catch (error: any) {
    notificationStore.error(
      'Delete Error',
      error.response?.data?.detail || 'Failed to delete rule'
    )
  }
}

onMounted(() => {
  refreshRules()
})
</script>