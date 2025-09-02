<template>
  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
    <div class="bg-white dark:bg-gray-800 rounded-lg max-w-2xl w-full mx-4 max-h-[80vh] flex flex-col">
      <!-- Header -->
      <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-600 flex items-center justify-between">
        <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
          Edit Rule - {{ rule?.name }}
        </h2>
        <button
          type="button"
          class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
          @click="$emit('close')"
        >
          <XMarkIcon class="h-6 w-6" />
        </button>
      </div>

      <!-- Content -->
      <div class="flex-1 overflow-y-auto p-6">
        <form class="space-y-6" @submit.prevent="saveRule">
          <!-- Rule Name -->
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Rule Name
            </label>
            <input
              v-model="editedRule.name"
              type="text"
              required
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500"
            />
          </div>

          <!-- Description -->
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Description
            </label>
            <textarea
              v-model="editedRule.description"
              rows="3"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500"
            />
          </div>

          <!-- Rule Type -->
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Rule Type
            </label>
            <select
              v-model="editedRule.type"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="position">Position</option>
              <option value="distance">Distance</option>
              <option value="alignment">Alignment</option>
              <option value="custom">Custom</option>
            </select>
          </div>

          <!-- Priority -->
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Priority
            </label>
            <select
              v-model="editedRule.priority"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="critical">Critical</option>
            </select>
          </div>

          <!-- Conditions (JSON Editor) -->
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Conditions (JSON)
            </label>
            <textarea
              v-model="conditionsJson"
              rows="5"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-900 text-green-400 font-mono text-sm focus:ring-primary-500 focus:border-primary-500"
              @blur="validateConditions"
            />
            <p v-if="conditionsError" class="mt-1 text-sm text-red-600 dark:text-red-400">
              {{ conditionsError }}
            </p>
          </div>

          <!-- Actions (JSON Editor) -->
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Actions (JSON)
            </label>
            <textarea
              v-model="actionsJson"
              rows="5"
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-900 text-green-400 font-mono text-sm focus:ring-primary-500 focus:border-primary-500"
              @blur="validateActions"
            />
            <p v-if="actionsError" class="mt-1 text-sm text-red-600 dark:text-red-400">
              {{ actionsError }}
            </p>
          </div>

          <!-- Enable/Disable -->
          <div class="flex items-center">
            <input
              v-model="editedRule.enabled"
              type="checkbox"
              class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 dark:border-gray-600 rounded"
            />
            <label class="ml-2 text-sm text-gray-700 dark:text-gray-300">
              Rule is enabled
            </label>
          </div>
        </form>
      </div>

      <!-- Footer -->
      <div class="px-6 py-4 border-t border-gray-200 dark:border-gray-600 flex justify-end space-x-3">
        <button
          type="button"
          class="btn btn-secondary"
          @click="$emit('close')"
        >
          Cancel
        </button>
        <button
          type="button"
          class="btn btn-primary"
          @click="saveRule"
          :disabled="saving || !!conditionsError || !!actionsError"
        >
          <span v-if="saving" class="loading-spinner h-4 w-4 mr-2"></span>
          Save Changes
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { XMarkIcon } from '@heroicons/vue/24/outline'
import { apiClient } from '@/api/client'
import { useNotificationStore } from '@/stores/notifications'
import type { Rule } from '@/types/api'

interface Props {
  rule: Rule | null
}

const props = defineProps<Props>()

const emit = defineEmits<{
  close: []
  updated: [rule: Rule]
}>()

const notificationStore = useNotificationStore()

// Form state
const editedRule = ref<Rule>({
  id: '',
  name: '',
  description: '',
  type: 'position',
  priority: 'medium',
  conditions: [],
  actions: [],
  enabled: true,
  created_at: '',
  updated_at: ''
})

const conditionsJson = ref('')
const actionsJson = ref('')
const conditionsError = ref('')
const actionsError = ref('')
const saving = ref(false)

// Initialize form when rule changes
watch(() => props.rule, (newRule) => {
  if (newRule) {
    editedRule.value = { ...newRule }
    conditionsJson.value = JSON.stringify(newRule.conditions, null, 2)
    actionsJson.value = JSON.stringify(newRule.actions, null, 2)
  }
}, { immediate: true })

function validateConditions() {
  try {
    const parsed = JSON.parse(conditionsJson.value)
    if (!Array.isArray(parsed)) {
      conditionsError.value = 'Conditions must be an array'
      return false
    }
    editedRule.value.conditions = parsed
    conditionsError.value = ''
    return true
  } catch (e) {
    conditionsError.value = 'Invalid JSON format'
    return false
  }
}

function validateActions() {
  try {
    const parsed = JSON.parse(actionsJson.value)
    if (!Array.isArray(parsed)) {
      actionsError.value = 'Actions must be an array'
      return false
    }
    editedRule.value.actions = parsed
    actionsError.value = ''
    return true
  } catch (e) {
    actionsError.value = 'Invalid JSON format'
    return false
  }
}

async function saveRule() {
  // Validate JSON fields
  if (!validateConditions() || !validateActions()) {
    return
  }

  saving.value = true
  
  try {
    const updatedRule = await apiClient.updateRule(editedRule.value.id, {
      name: editedRule.value.name,
      description: editedRule.value.description,
      type: editedRule.value.type,
      priority: editedRule.value.priority,
      conditions: editedRule.value.conditions,
      actions: editedRule.value.actions,
      enabled: editedRule.value.enabled
    })
    
    notificationStore.success('Rule Updated', `${updatedRule.name} has been updated successfully`)
    emit('updated', updatedRule)
    emit('close')
    
  } catch (error: any) {
    notificationStore.error(
      'Update Error',
      error.response?.data?.detail || 'Failed to update rule'
    )
  } finally {
    saving.value = false
  }
}
</script>

<style scoped>
.loading-spinner {
  @apply animate-spin rounded-full border-2 border-gray-300 border-t-primary-600;
}
</style>