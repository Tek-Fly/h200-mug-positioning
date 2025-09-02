import { defineStore } from 'pinia'
import { ref, readonly } from 'vue'

export interface Notification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message: string
  duration?: number
  persistent?: boolean
  actions?: {
    label: string
    action: () => void
    style?: 'primary' | 'secondary'
  }[]
}

export const useNotificationStore = defineStore('notifications', () => {
  const notifications = ref<Notification[]>([])
  let notificationId = 0

  function addNotification(notification: Omit<Notification, 'id'>): string {
    const id = `notification-${++notificationId}`
    const newNotification: Notification = {
      id,
      duration: notification.persistent ? undefined : (notification.duration || 5000),
      ...notification
    }

    notifications.value.push(newNotification)

    // Auto-remove after duration if not persistent
    if (newNotification.duration) {
      setTimeout(() => {
        removeNotification(id)
      }, newNotification.duration)
    }

    return id
  }

  function removeNotification(id: string): void {
    const index = notifications.value.findIndex(n => n.id === id)
    if (index > -1) {
      notifications.value.splice(index, 1)
    }
  }

  function clearAllNotifications(): void {
    notifications.value = []
  }

  // Convenience methods for different types
  function success(title: string, message: string, options?: Partial<Notification>): string {
    return addNotification({ type: 'success', title, message, ...options })
  }

  function error(title: string, message: string, options?: Partial<Notification>): string {
    return addNotification({ type: 'error', title, message, persistent: true, ...options })
  }

  function warning(title: string, message: string, options?: Partial<Notification>): string {
    return addNotification({ type: 'warning', title, message, ...options })
  }

  function info(title: string, message: string, options?: Partial<Notification>): string {
    return addNotification({ type: 'info', title, message, ...options })
  }

  return {
    notifications: readonly(notifications),
    addNotification,
    removeNotification,
    clearAllNotifications,
    success,
    error,
    warning,
    info
  }
})