/**
 * Production-ready logging utility for the dashboard
 */

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
}

interface LogEntry {
  timestamp: string
  level: string
  message: string
  data?: any
  source?: string
}

class Logger {
  private level: LogLevel
  private isDevelopment: boolean
  private logBuffer: LogEntry[] = []
  private maxBufferSize = 100

  constructor() {
    this.isDevelopment = import.meta.env.DEV
    this.level = this.isDevelopment ? LogLevel.DEBUG : LogLevel.INFO
  }

  private formatMessage(level: string, message: string, data?: any, source?: string): LogEntry {
    return {
      timestamp: new Date().toISOString(),
      level,
      message,
      data,
      source,
    }
  }

  private shouldLog(level: LogLevel): boolean {
    return level >= this.level
  }

  private addToBuffer(entry: LogEntry): void {
    this.logBuffer.push(entry)
    if (this.logBuffer.length > this.maxBufferSize) {
      this.logBuffer.shift()
    }
  }

  private log(level: LogLevel, levelName: string, message: string, data?: any, source?: string): void {
    if (!this.shouldLog(level)) return

    const entry = this.formatMessage(levelName, message, data, source)
    this.addToBuffer(entry)

    // In development, also log to console
    if (this.isDevelopment) {
      const consoleMethod = levelName.toLowerCase() as 'debug' | 'info' | 'warn' | 'error'
      const consoleFn = console[consoleMethod] || console.log
      
      if (data) {
        consoleFn(`[${entry.timestamp}] ${source ? `[${source}] ` : ''}${message}`, data)
      } else {
        consoleFn(`[${entry.timestamp}] ${source ? `[${source}] ` : ''}${message}`)
      }
    }

    // In production, you could send logs to a remote logging service
    if (!this.isDevelopment && level >= LogLevel.ERROR) {
      // TODO: Send to remote logging service
      this.sendToRemote(entry)
    }
  }

  private async sendToRemote(entry: LogEntry): Promise<void> {
    // Placeholder for sending logs to remote service
    // This could be integrated with your backend logging endpoint
    try {
      // Example: await fetch('/api/v1/logs', { method: 'POST', body: JSON.stringify(entry) })
    } catch (error) {
      // Fail silently to avoid infinite error loops
    }
  }

  debug(message: string, data?: any, source?: string): void {
    this.log(LogLevel.DEBUG, 'DEBUG', message, data, source)
  }

  info(message: string, data?: any, source?: string): void {
    this.log(LogLevel.INFO, 'INFO', message, data, source)
  }

  warn(message: string, data?: any, source?: string): void {
    this.log(LogLevel.WARN, 'WARN', message, data, source)
  }

  error(message: string, data?: any, source?: string): void {
    this.log(LogLevel.ERROR, 'ERROR', message, data, source)
  }

  getLogBuffer(): LogEntry[] {
    return [...this.logBuffer]
  }

  clearBuffer(): void {
    this.logBuffer = []
  }

  setLevel(level: LogLevel): void {
    this.level = level
  }
}

// Create singleton instance
export const logger = new Logger()

// Export convenience functions
export const logDebug = (message: string, data?: any, source?: string) => logger.debug(message, data, source)
export const logInfo = (message: string, data?: any, source?: string) => logger.info(message, data, source)
export const logWarn = (message: string, data?: any, source?: string) => logger.warn(message, data, source)
export const logError = (message: string, data?: any, source?: string) => logger.error(message, data, source)