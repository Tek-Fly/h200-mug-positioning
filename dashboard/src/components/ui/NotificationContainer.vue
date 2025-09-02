<template>
  <div class="fixed top-4 right-4 z-50 space-y-4 max-w-sm w-full">
    <TransitionGroup name="toast" tag="div">
      <div
        v-for="notification in notifications"
        :key="notification.id"
        :class="[
          'toast p-4 rounded-lg shadow-lg border-l-4 bg-white dark:bg-gray-800',
          {
            'border-success-500 bg-success-50 dark:bg-success-900/20': notification.type === 'success',
            'border-danger-500 bg-danger-50 dark:bg-danger-900/20': notification.type === 'error',
            'border-warning-500 bg-warning-50 dark:bg-warning-900/20': notification.type === 'warning',
            'border-primary-500 bg-primary-50 dark:bg-primary-900/20': notification.type === 'info'
          }
        ]"
      >
        <div class="flex items-start">
          <div class="flex-shrink-0">
            <component
              :is="getIcon(notification.type)"
              :class="[
                'h-5 w-5',
                {
                  'text-success-600': notification.type === 'success',
                  'text-danger-600': notification.type === 'error',
                  'text-warning-600': notification.type === 'warning',
                  'text-primary-600': notification.type === 'info'
                }
              ]"
            />
          </div>
          
          <div class="ml-3 flex-1">
            <h4 class="text-sm font-medium text-gray-900 dark:text-gray-100">
              {{ notification.title }}
            </h4>
            <p class="mt-1 text-sm text-gray-700 dark:text-gray-300">
              {{ notification.message }}
            </p>
            
            <div v-if="notification.actions?.length" class="mt-3 flex space-x-2">
              <button
                v-for="action in notification.actions"
                :key="action.label"
                type="button"
                :class="[
                  'text-xs font-medium px-3 py-1 rounded',
                  action.style === 'primary' 
                    ? 'bg-primary-600 text-white hover:bg-primary-700'
                    : 'bg-gray-200 text-gray-900 hover:bg-gray-300 dark:bg-gray-600 dark:text-gray-200 dark:hover:bg-gray-500'
                ]"
                @click="action.action"
              >
                {{ action.label }}
              </button>
            </div>
          </div>
          
          <div class="ml-4 flex-shrink-0">
            <button
              type="button"
              class="inline-flex text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-500"
              @click="removeNotification(notification.id)"
            >
              <XMarkIcon class="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
    </TransitionGroup>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import {
  CheckCircleIcon,
  ExclamationCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  XMarkIcon
} from '@heroicons/vue/20/solid'
import { useNotificationStore } from '@/stores/notifications'

const notificationStore = useNotificationStore()

const notifications = computed(() => notificationStore.notifications)

function removeNotification(id: string): void {
  notificationStore.removeNotification(id)
}

function getIcon(type: string) {
  switch (type) {
    case 'success':
      return CheckCircleIcon
    case 'error':
      return ExclamationCircleIcon
    case 'warning':
      return ExclamationTriangleIcon
    case 'info':
      return InformationCircleIcon
    default:
      return InformationCircleIcon
  }
}
</script>

<style scoped>
.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s ease;
}

.toast-enter-from {
  opacity: 0;
  transform: translateX(100%);
}

.toast-leave-to {
  opacity: 0;
  transform: translateX(100%);
}

.toast-move {
  transition: transform 0.3s ease;
}
</style>