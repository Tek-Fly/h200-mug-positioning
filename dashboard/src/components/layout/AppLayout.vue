<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
    <!-- Navigation -->
    <nav class="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <!-- Logo and navigation -->
          <div class="flex items-center">
            <div class="flex-shrink-0 flex items-center">
              <CpuChipIcon class="h-8 w-8 text-primary-600" />
              <span class="ml-2 text-xl font-bold text-gray-900 dark:text-white">
                H200 Dashboard
              </span>
            </div>
            
            <div class="hidden md:ml-8 md:flex md:space-x-4">
              <router-link
                v-for="item in navigation"
                :key="item.name"
                :to="item.to"
                :class="[
                  'px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200',
                  $route.name === item.name
                    ? 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-200'
                    : 'text-gray-700 hover:text-primary-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:text-primary-300 dark:hover:bg-gray-700'
                ]"
              >
                <component :is="item.icon" class="h-4 w-4 inline mr-2" />
                {{ item.name }}
              </router-link>
            </div>
          </div>

          <!-- Right side controls -->
          <div class="flex items-center space-x-4">
            <!-- WebSocket status indicator -->
            <div class="flex items-center space-x-2">
              <div
                :class="[
                  'w-2 h-2 rounded-full',
                  wsConnected ? 'bg-success-500 animate-pulse-slow' : 'bg-gray-400'
                ]"
              ></div>
              <span class="text-xs text-gray-500 dark:text-gray-400 hidden sm:block">
                {{ wsConnected ? 'Connected' : 'Disconnected' }}
              </span>
            </div>

            <!-- Theme toggle -->
            <button
              type="button"
              class="p-2 rounded-md text-gray-500 hover:text-gray-700 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-gray-200 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-primary-500"
              @click="themeStore.toggleTheme()"
            >
              <SunIcon v-if="themeStore.isDark" class="h-5 w-5" />
              <MoonIcon v-else class="h-5 w-5" />
            </button>

            <!-- User menu -->
            <div class="relative">
              <button
                type="button"
                class="flex items-center space-x-2 p-2 rounded-md text-gray-700 hover:text-primary-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:text-primary-300 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-primary-500"
                @click="showUserMenu = !showUserMenu"
              >
                <UserCircleIcon class="h-6 w-6" />
                <span class="text-sm font-medium hidden sm:block">
                  {{ authStore.user?.username || 'User' }}
                </span>
                <ChevronDownIcon class="h-4 w-4" />
              </button>

              <!-- User menu dropdown -->
              <Transition
                enter-active-class="transition ease-out duration-100"
                enter-from-class="transform opacity-0 scale-95"
                enter-to-class="transform opacity-100 scale-100"
                leave-active-class="transition ease-in duration-75"
                leave-from-class="transform opacity-100 scale-100"
                leave-to-class="transform opacity-0 scale-95"
              >
                <div
                  v-if="showUserMenu"
                  class="origin-top-right absolute right-0 mt-2 w-48 rounded-md shadow-lg bg-white dark:bg-gray-800 ring-1 ring-black ring-opacity-5 focus:outline-none z-50"
                  @click.outside="showUserMenu = false"
                >
                  <div class="py-1">
                    <router-link
                      to="/settings"
                      class="block px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                      @click="showUserMenu = false"
                    >
                      Settings
                    </router-link>
                    <button
                      type="button"
                      class="block w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                      @click="handleLogout"
                    >
                      Sign out
                    </button>
                  </div>
                </div>
              </Transition>
            </div>

            <!-- Mobile menu button -->
            <button
              type="button"
              class="md:hidden p-2 rounded-md text-gray-500 hover:text-gray-700 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-gray-200 dark:hover:bg-gray-700"
              @click="showMobileMenu = !showMobileMenu"
            >
              <Bars3Icon v-if="!showMobileMenu" class="h-6 w-6" />
              <XMarkIcon v-else class="h-6 w-6" />
            </button>
          </div>
        </div>

        <!-- Mobile menu -->
        <Transition
          enter-active-class="transition ease-out duration-200"
          enter-from-class="opacity-0 translate-y-1"
          enter-to-class="opacity-100 translate-y-0"
          leave-active-class="transition ease-in duration-150"
          leave-from-class="opacity-100 translate-y-0"
          leave-to-class="opacity-0 translate-y-1"
        >
          <div v-if="showMobileMenu" class="md:hidden border-t border-gray-200 dark:border-gray-700">
            <div class="px-2 pt-2 pb-3 space-y-1">
              <router-link
                v-for="item in navigation"
                :key="item.name"
                :to="item.to"
                :class="[
                  'block px-3 py-2 rounded-md text-base font-medium transition-colors duration-200',
                  $route.name === item.name
                    ? 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-200'
                    : 'text-gray-700 hover:text-primary-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:text-primary-300 dark:hover:bg-gray-700'
                ]"
                @click="showMobileMenu = false"
              >
                <component :is="item.icon" class="h-5 w-5 inline mr-3" />
                {{ item.name }}
              </router-link>
            </div>
          </div>
        </Transition>
      </div>
    </nav>

    <!-- Main content -->
    <main class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import {
  CpuChipIcon,
  HomeIcon,
  PhotoIcon,
  ServerIcon,
  DocumentTextIcon,
  ChartBarIcon,
  CogIcon,
  UserCircleIcon,
  ChevronDownIcon,
  SunIcon,
  MoonIcon,
  Bars3Icon,
  XMarkIcon
} from '@heroicons/vue/24/outline'
import { useAuthStore } from '@/stores/auth'
import { useThemeStore } from '@/stores/theme'
import { useWebSocket } from '@/api/websocket'

const router = useRouter()
const authStore = useAuthStore()
const themeStore = useThemeStore()
const { client: wsClient, connect: connectWs, isConnected } = useWebSocket()

const showUserMenu = ref(false)
const showMobileMenu = ref(false)

const wsConnected = computed(() => isConnected())

const navigation = [
  { name: 'Dashboard', to: '/', icon: HomeIcon },
  { name: 'Analysis', to: '/analysis', icon: PhotoIcon },
  { name: 'Servers', to: '/servers', icon: ServerIcon },
  { name: 'Rules', to: '/rules', icon: DocumentTextIcon },
  { name: 'Metrics', to: '/metrics', icon: ChartBarIcon },
]

async function handleLogout(): Promise<void> {
  showUserMenu.value = false
  await authStore.logout()
  router.push('/login')
}

onMounted(async () => {
  // Connect WebSocket if authenticated
  if (authStore.isAuthenticated) {
    try {
      await connectWs()
    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
    }
  }
})

// Close menus when clicking outside
function handleClickOutside(event: MouseEvent): void {
  if (!(event.target as Element).closest('.relative')) {
    showUserMenu.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>