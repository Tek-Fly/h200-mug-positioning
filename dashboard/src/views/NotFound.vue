<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full text-center">
      <div class="mb-8">
        <ExclamationTriangleIcon class="mx-auto h-24 w-24 text-gray-400" />
      </div>
      
      <h1 class="text-6xl font-bold text-gray-900 dark:text-white mb-4">
        404
      </h1>
      
      <h2 class="text-2xl font-semibold text-gray-700 dark:text-gray-300 mb-4">
        Page Not Found
      </h2>
      
      <p class="text-gray-600 dark:text-gray-400 mb-8">
        The page you're looking for doesn't exist or has been moved.
      </p>
      
      <div class="space-y-4">
        <router-link
          to="/"
          class="btn btn-primary w-full"
        >
          <HomeIcon class="h-5 w-5 mr-2" />
          Go to Dashboard
        </router-link>
        
        <button
          type="button"
          class="btn btn-secondary w-full"
          @click="$router.back()"
        >
          <ArrowLeftIcon class="h-5 w-5 mr-2" />
          Go Back
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  ExclamationTriangleIcon,
  HomeIcon,
  ArrowLeftIcon
} from '@heroicons/vue/24/outline'
</script>