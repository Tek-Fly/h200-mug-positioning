<template>
  <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg p-4 hover:shadow-md transition-shadow duration-200">
    <div class="flex items-start justify-between mb-3">
      <div class="flex-1">
        <div class="flex items-center space-x-2 mb-1">
          <h3 class="text-sm font-medium text-gray-900 dark:text-white">
            {{ rule.name }}
          </h3>
          <span
            :class="[
              'px-2 py-1 rounded-full text-xs font-medium',
              getRuleTypeColor(rule.type)
            ]"
          >
            {{ rule.type }}
          </span>
          <span
            :class="[
              'px-2 py-1 rounded-full text-xs font-medium',
              getPriorityColor(rule.priority)
            ]"
          >
            {{ rule.priority }}
          </span>
        </div>
        
        <p class="text-sm text-gray-600 dark:text-gray-400 mb-2">
          {{ rule.description }}
        </p>
        
        <div class="text-xs text-gray-500 dark:text-gray-500">
          Created {{ formatDate(rule.created_at) }}
        </div>
      </div>

      <!-- Toggle Switch -->
      <div class="flex items-center ml-4">
        <button
          type="button"
          :class="[
            'relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
            rule.enabled ? 'bg-primary-600' : 'bg-gray-200 dark:bg-gray-600'
          ]"
          @click="$emit('toggle', rule)"
        >
          <span
            :class="[
              'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out',
              rule.enabled ? 'translate-x-5' : 'translate-x-0'
            ]"
          />
        </button>
      </div>
    </div>

    <!-- Rule Details -->
    <div class="text-xs text-gray-500 dark:text-gray-400 mb-3">
      <span>{{ rule.conditions.length }} condition(s)</span>
      <span class="mx-2">•</span>
      <span>{{ rule.actions.length }} action(s)</span>
    </div>

    <!-- Natural Language Source -->
    <div v-if="rule.metadata.natural_language_source" class="mb-3 p-2 bg-gray-50 dark:bg-gray-700 rounded text-xs">
      <span class="text-gray-500 dark:text-gray-400">Original:</span>
      <span class="ml-2 text-gray-700 dark:text-gray-300 italic">
        "{{ rule.metadata.natural_language_source }}"
      </span>
    </div>

    <!-- Actions -->
    <div class="flex items-center justify-end space-x-2 pt-3 border-t border-gray-200 dark:border-gray-600">
      <button
        type="button"
        class="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200"
        @click="$emit('edit', rule)"
        title="Edit rule"
      >
        <PencilIcon class="h-4 w-4" />
      </button>
      
      <button
        type="button"
        class="p-2 text-gray-400 hover:text-danger-600 dark:hover:text-danger-400 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200"
        @click="$emit('delete', rule)"
        title="Delete rule"
      >
        <TrashIcon class="h-4 w-4" />
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { formatDistanceToNow } from 'date-fns'
import { PencilIcon, TrashIcon } from '@heroicons/vue/24/outline'
import type { Rule } from '@/types/api'

interface Props {
  rule: Rule
}

defineProps<Props>()

defineEmits<{
  toggle: [rule: Rule]
  edit: [rule: Rule]
  delete: [rule: Rule]
}>()

function formatDate(dateString: string): string {
  try {
    return formatDistanceToNow(new Date(dateString), { addSuffix: true })
  } catch {
    return dateString
  }
}

function getRuleTypeColor(type: string): string {
  switch (type) {
    case 'position':
      return 'bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200'
    case 'distance':
      return 'bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200'
    case 'alignment':
      return 'bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-200'
    default:
      return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
  }
}

function getPriorityColor(priority: string): string {
  switch (priority) {
    case 'high':
      return 'bg-danger-100 text-danger-800 dark:bg-danger-900 dark:text-danger-200'
    case 'medium':
      return 'bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-200'
    case 'low':
      return 'bg-success-100 text-success-800 dark:bg-success-900 dark:text-success-200'
    default:
      return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
  }
}
</script>