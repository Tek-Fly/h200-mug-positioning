<template>
  <div id="app" class="min-h-screen transition-colors duration-200">
    <router-view />
    <NotificationContainer />
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useThemeStore } from '@/stores/theme'
import { useAuthStore } from '@/stores/auth'
import NotificationContainer from '@/components/ui/NotificationContainer.vue'

const themeStore = useThemeStore()
const authStore = useAuthStore()

onMounted(async () => {
  // Initialize theme
  themeStore.initializeTheme()
  
  // Try to restore auth session
  await authStore.initializeAuth()
})
</script>