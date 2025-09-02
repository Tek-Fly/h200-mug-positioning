import { defineStore } from 'pinia'
import { ref, computed, watch, readonly } from 'vue'

export type Theme = 'light' | 'dark' | 'system'

export const useThemeStore = defineStore('theme', () => {
  const theme = ref<Theme>('system')
  const systemTheme = ref<'light' | 'dark'>('light')

  const isDark = computed(() => {
    if (theme.value === 'system') {
      return systemTheme.value === 'dark'
    }
    return theme.value === 'dark'
  })

  const effectiveTheme = computed(() => {
    if (theme.value === 'system') {
      return systemTheme.value
    }
    return theme.value
  })

  function setTheme(newTheme: Theme): void {
    theme.value = newTheme
    localStorage.setItem('theme', newTheme)
    applyTheme()
  }

  function toggleTheme(): void {
    if (theme.value === 'light') {
      setTheme('dark')
    } else if (theme.value === 'dark') {
      setTheme('system')
    } else {
      setTheme('light')
    }
  }

  function applyTheme(): void {
    const root = document.documentElement
    
    if (isDark.value) {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
  }

  function detectSystemTheme(): void {
    if (typeof window !== 'undefined' && window.matchMedia) {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
      systemTheme.value = mediaQuery.matches ? 'dark' : 'light'
      
      // Listen for system theme changes
      mediaQuery.addEventListener('change', (e) => {
        systemTheme.value = e.matches ? 'dark' : 'light'
      })
    }
  }

  function initializeTheme(): void {
    // Detect system theme
    detectSystemTheme()
    
    // Load saved theme
    const savedTheme = localStorage.getItem('theme') as Theme
    if (savedTheme && ['light', 'dark', 'system'].includes(savedTheme)) {
      theme.value = savedTheme
    }
    
    // Apply initial theme
    applyTheme()
  }

  // Watch for theme changes
  watch([isDark], () => {
    applyTheme()
  }, { immediate: true })

  return {
    theme: readonly(theme),
    isDark,
    effectiveTheme,
    setTheme,
    toggleTheme,
    initializeTheme
  }
})