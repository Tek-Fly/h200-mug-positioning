import { defineStore } from 'pinia'
import { ref, computed, readonly } from 'vue'
import { apiClient } from '@/api/client'
import { wsClient } from '@/api/websocket'
import { useNotificationStore } from './notifications'
import { logError } from '@/utils/logger'

export const useAuthStore = defineStore('auth', () => {
  const user = ref<any>(null)
  const token = ref<string | null>(null)
  const loading = ref(false)
  const initialized = ref(false)

  const isAuthenticated = computed(() => !!token.value && !!user.value)

  async function login(username: string, password: string): Promise<void> {
    loading.value = true
    const notificationStore = useNotificationStore()
    
    try {
      const response = await apiClient.login(username, password)
      token.value = response.access_token
      
      // Set token in API client
      apiClient.setAuth(response.access_token)
      
      // Set token for WebSocket
      wsClient.setAuthToken(response.access_token)
      
      // Get user info
      user.value = await apiClient.getCurrentUser()
      
      notificationStore.addNotification({
        type: 'success',
        title: 'Login Successful',
        message: `Welcome back, ${user.value.username || 'User'}!`
      })
      
    } catch (error: any) {
      notificationStore.addNotification({
        type: 'error',
        title: 'Login Failed',
        message: error.response?.data?.detail || 'Invalid credentials'
      })
      throw error
    } finally {
      loading.value = false
    }
  }

  async function logout(): Promise<void> {
    const notificationStore = useNotificationStore()
    
    try {
      // Clear auth data
      user.value = null
      token.value = null
      
      // Clear API client auth
      apiClient.clearAuth()
      
      // Disconnect WebSocket
      wsClient.disconnect()
      
      notificationStore.addNotification({
        type: 'info',
        title: 'Logged Out',
        message: 'You have been successfully logged out'
      })
      
    } catch (error) {
      logError('Logout error', error, 'AuthStore')
    }
  }

  async function initializeAuth(): Promise<void> {
    if (initialized.value) return
    
    try {
      // Try to restore token from storage
      const storedToken = apiClient.initAuth()
      
      if (storedToken) {
        token.value = storedToken
        wsClient.setAuthToken(storedToken)
        
        try {
          // Verify token is still valid
          user.value = await apiClient.getCurrentUser()
        } catch (error) {
          // Token is invalid, clear it
          await logout()
        }
      }
    } catch (error) {
      logError('Auth initialization error', error, 'AuthStore')
    } finally {
      initialized.value = true
    }
  }

  function handleUnauthorized(): void {
    logout()
  }

  // Listen for unauthorized events
  if (typeof window !== 'undefined') {
    window.addEventListener('auth:unauthorized', handleUnauthorized)
  }

  return {
    user: readonly(user),
    token: readonly(token),
    loading: readonly(loading),
    initialized: readonly(initialized),
    isAuthenticated,
    login,
    logout,
    initializeAuth
  }
})