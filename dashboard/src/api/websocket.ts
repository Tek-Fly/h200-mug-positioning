import type {
  WebSocketMessage,
  MetricsMessage,
  LogMessage,
  AlertMessage,
  ActivityMessage
} from '@/types/api'

export type WebSocketMessageType = 'metrics' | 'logs' | 'alerts' | 'activity'

export interface WebSocketSubscription {
  topic: WebSocketMessageType
  callback: (message: WebSocketMessage) => void
}

export class WebSocketClient {
  private ws: WebSocket | null = null
  private subscriptions: Map<WebSocketMessageType, Set<(message: WebSocketMessage) => void>> = new Map()
  private reconnectInterval: number = 5000
  private maxReconnectAttempts: number = 10
  private reconnectAttempts: number = 0
  private isConnecting: boolean = false
  private authToken: string | null = null

  constructor() {
    this.subscriptions.set('metrics', new Set())
    this.subscriptions.set('logs', new Set())
    this.subscriptions.set('alerts', new Set())
    this.subscriptions.set('activity', new Set())
  }

  setAuthToken(token: string): void {
    this.authToken = token
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.CONNECTING)) {
        return
      }

      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        resolve()
        return
      }

      this.isConnecting = true

      const wsUrl = this.buildWebSocketUrl()
      this.ws = new WebSocket(wsUrl)

      this.ws.onopen = () => {
        console.log('WebSocket connected')
        this.isConnecting = false
        this.reconnectAttempts = 0
        resolve()
      }

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          this.handleMessage(message)
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }

      this.ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason)
        this.isConnecting = false
        this.ws = null

        if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect()
        }
      }

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        this.isConnecting = false
        
        if (this.reconnectAttempts === 0) {
          reject(error)
        }
      }
    })
  }

  private buildWebSocketUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const path = '/ws/control-plane'
    
    let url = `${protocol}//${host}${path}`
    
    if (this.authToken) {
      url += `?token=${encodeURIComponent(this.authToken)}`
    }
    
    return url
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached')
      return
    }

    this.reconnectAttempts++
    console.log(`Reconnecting in ${this.reconnectInterval}ms (attempt ${this.reconnectAttempts})`)

    setTimeout(() => {
      this.connect().catch(error => {
        console.error('Reconnection failed:', error)
      })
    }, this.reconnectInterval)
  }

  private handleMessage(message: WebSocketMessage): void {
    // Handle connection messages
    if (message.type === 'connection' || message.type === 'subscription' || message.type === 'pong') {
      console.log('WebSocket system message:', message)
      return
    }

    // Dispatch to appropriate subscribers
    const callbacks = this.subscriptions.get(message.type as WebSocketMessageType)
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(message)
        } catch (error) {
          console.error('Error in WebSocket callback:', error)
        }
      })
    }
  }

  subscribe(topic: WebSocketMessageType, callback: (message: WebSocketMessage) => void): () => void {
    const callbacks = this.subscriptions.get(topic)
    if (callbacks) {
      callbacks.add(callback)

      // Send subscription message if connected
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.sendMessage({
          action: 'subscribe',
          topic
        })
      }

      // Return unsubscribe function
      return () => {
        callbacks.delete(callback)
        
        // Send unsubscribe message if no more callbacks for this topic
        if (callbacks.size === 0 && this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.sendMessage({
            action: 'unsubscribe',
            topic
          })
        }
      }
    }

    return () => {}
  }

  unsubscribe(topic: WebSocketMessageType, callback?: (message: WebSocketMessage) => void): void {
    const callbacks = this.subscriptions.get(topic)
    if (callbacks) {
      if (callback) {
        callbacks.delete(callback)
      } else {
        callbacks.clear()
      }

      // Send unsubscribe message if connected
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.sendMessage({
          action: 'unsubscribe',
          topic
        })
      }
    }
  }

  private sendMessage(message: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
    }
  }

  ping(): void {
    this.sendMessage({ action: 'ping' })
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect')
      this.ws = null
    }
    this.isConnecting = false
    this.reconnectAttempts = this.maxReconnectAttempts // Prevent reconnection
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }

  getConnectionState(): string {
    if (!this.ws) return 'disconnected'
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING: return 'connecting'
      case WebSocket.OPEN: return 'connected'
      case WebSocket.CLOSING: return 'closing'
      case WebSocket.CLOSED: return 'closed'
      default: return 'unknown'
    }
  }
}

// Create singleton instance
export const wsClient = new WebSocketClient()

// Helper composable for Vue components
export function useWebSocket() {
  return {
    client: wsClient,
    connect: () => wsClient.connect(),
    disconnect: () => wsClient.disconnect(),
    subscribe: (topic: WebSocketMessageType, callback: (message: WebSocketMessage) => void) => 
      wsClient.subscribe(topic, callback),
    unsubscribe: (topic: WebSocketMessageType, callback?: (message: WebSocketMessage) => void) =>
      wsClient.unsubscribe(topic, callback),
    isConnected: () => wsClient.isConnected(),
    getConnectionState: () => wsClient.getConnectionState()
  }
}