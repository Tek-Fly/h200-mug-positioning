<template>
  <div class="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full space-y-8">
      <div>
        <div class="flex justify-center">
          <CpuChipIcon class="h-12 w-12 text-primary-600" />
        </div>
        <h2 class="mt-6 text-center text-3xl font-extrabold text-gray-900 dark:text-white">
          H200 Dashboard
        </h2>
        <p class="mt-2 text-center text-sm text-gray-600 dark:text-gray-400">
          Sign in to your account
        </p>
      </div>
      
      <form class="mt-8 space-y-6" @submit.prevent="handleSubmit">
        <div class="rounded-md shadow-sm space-y-4">
          <div>
            <label for="username" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Username
            </label>
            <input
              id="username"
              v-model="form.username"
              name="username"
              type="text"
              required
              class="appearance-none relative block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 placeholder-gray-500 dark:placeholder-gray-400 text-gray-900 dark:text-white bg-white dark:bg-gray-800 rounded-md focus:outline-none focus:ring-primary-500 focus:border-primary-500 focus:z-10 sm:text-sm"
              placeholder="Enter your username"
            />
          </div>
          
          <div>
            <label for="password" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Password
            </label>
            <input
              id="password"
              v-model="form.password"
              name="password"
              type="password"
              required
              class="appearance-none relative block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 placeholder-gray-500 dark:placeholder-gray-400 text-gray-900 dark:text-white bg-white dark:bg-gray-800 rounded-md focus:outline-none focus:ring-primary-500 focus:border-primary-500 focus:z-10 sm:text-sm"
              placeholder="Enter your password"
            />
          </div>
        </div>

        <div class="flex items-center justify-between">
          <div class="flex items-center">
            <input
              id="remember-me"
              v-model="form.rememberMe"
              name="remember-me"
              type="checkbox"
              class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
            />
            <label for="remember-me" class="ml-2 block text-sm text-gray-900 dark:text-gray-300">
              Remember me
            </label>
          </div>
        </div>

        <div>
          <button
            type="submit"
            :disabled="loading"
            class="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span v-if="loading" class="absolute left-0 inset-y-0 flex items-center pl-3">
              <div class="loading-spinner h-5 w-5"></div>
            </span>
            {{ loading ? 'Signing in...' : 'Sign in' }}
          </button>
        </div>

        <!-- Demo credentials info -->
        <div class="mt-6 p-4 bg-primary-50 dark:bg-primary-900/20 rounded-lg">
          <h3 class="text-sm font-medium text-primary-800 dark:text-primary-200 mb-2">
            Demo Credentials
          </h3>
          <div class="text-xs text-primary-700 dark:text-primary-300 space-y-1">
            <div>Username: <code class="bg-primary-100 dark:bg-primary-800 px-1 rounded">admin</code></div>
            <div>Password: <code class="bg-primary-100 dark:bg-primary-800 px-1 rounded">admin123</code></div>
          </div>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { reactive, computed } from 'vue'
import { useRouter } from 'vue-router'
import { CpuChipIcon } from '@heroicons/vue/24/outline'
import { useAuthStore } from '@/stores/auth'
import { logError } from '@/utils/logger'

const router = useRouter()
const authStore = useAuthStore()

const loading = computed(() => authStore.loading)

const form = reactive({
  username: '',
  password: '',
  rememberMe: false
})

async function handleSubmit(): Promise<void> {
  if (!form.username || !form.password) return

  try {
    await authStore.login(form.username, form.password)
    
    // Redirect to dashboard or intended route
    const redirect = router.currentRoute.value.query.redirect as string
    router.push(redirect || '/')
    
  } catch (error) {
    // Error is handled by the auth store
    logError('Login failed', error, 'Login')
  }
}

// Auto-fill demo credentials (for development)
if (import.meta.env.DEV) {
  form.username = 'admin'
  form.password = 'admin123'
}
</script>