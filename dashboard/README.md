# H200 Mug Positioning Dashboard

A modern, responsive Vue.js dashboard for the H200 Intelligent Mug Positioning System.

## Features

- **Real-time Dashboard**: Live metrics and system monitoring with WebSocket updates
- **Image Analysis**: Upload and analyze mug positioning with AI-powered feedback
- **Server Management**: Control H200 inference servers (serverless and timed instances)
- **Rule Management**: Create positioning rules using natural language
- **Dark/Light Theme**: Automatic theme switching based on system preference
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices

## Tech Stack

- **Vue 3** with Composition API
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **Chart.js** for data visualization
- **Pinia** for state management
- **Vite** for development and building

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- H200 API server running on `localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Development Scripts

```bash
# Type checking
npm run type-check

# Linting
npm run lint

# Build and type check
npm run build
```

## Project Structure

```
src/
├── api/              # API client and WebSocket handling
├── assets/           # CSS and static assets
├── components/       # Vue components
│   ├── analysis/     # Image analysis components
│   ├── dashboard/    # Dashboard widgets and charts
│   ├── layout/       # Layout components
│   ├── rules/        # Rule management components
│   ├── servers/      # Server management components
│   └── ui/           # Reusable UI components
├── stores/           # Pinia state management
├── types/            # TypeScript type definitions
├── views/            # Page components
└── router/           # Vue Router configuration
```

## Key Components

### Dashboard
- Real-time system health monitoring
- Performance metrics with trend indicators
- Resource usage visualization
- Cost tracking and breakdown
- Recent activity feed

### Image Analysis
- Drag-and-drop image upload
- Real-time positioning analysis
- Interactive bounding box visualization
- Feedback submission for AI improvement
- Analysis history with export functionality

### Server Management
- Start/stop/restart server controls
- Real-time server metrics
- Log viewing and download
- Auto-shutdown protection
- Server configuration management

### Rule Management
- Natural language rule creation
- Rule activation/deactivation
- Rule performance statistics
- Pre-built rule examples

## Configuration

### API Endpoint

The dashboard expects the H200 API server to be running on `localhost:8000`. To change this, update the proxy configuration in `vite.config.ts`:

```typescript
server: {
  proxy: {
    '/api': {
      target: 'http://your-api-server:8000',
      changeOrigin: true,
    }
  }
}
```

### Authentication

The dashboard uses JWT tokens for authentication. Demo credentials are:
- Username: `admin`
- Password: `admin123`

### Theme Configuration

The dashboard supports three theme modes:
- **Light**: Always light theme
- **Dark**: Always dark theme  
- **System**: Follows system preference

Theme selection is persisted in localStorage.

## API Integration

The dashboard integrates with the following H200 API endpoints:

- `GET /api/v1/dashboard` - System metrics and health
- `POST /api/v1/analyze/with-feedback` - Image analysis
- `GET /api/v1/servers` - Server management
- `POST /api/v1/rules/natural-language` - Rule creation
- `WebSocket /ws/control-plane` - Real-time updates

## WebSocket Integration

Real-time updates are handled via WebSocket connections:

```javascript
// Subscribe to metrics updates
wsClient.subscribe('metrics', (message) => {
  // Handle real-time metrics
})

// Subscribe to activity updates  
wsClient.subscribe('activity', (message) => {
  // Handle new activity logs
})
```

## Performance Features

- **Lazy Loading**: Route-based code splitting
- **Image Optimization**: Automatic image compression
- **Caching**: Aggressive caching for static assets
- **Bundle Analysis**: Webpack bundle analyzer integration

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Development

### Hot Module Replacement
The development server supports HMR for instant updates during development.

### Type Checking
TypeScript strict mode is enabled for better type safety.

### Linting
ESLint configuration includes Vue 3 and TypeScript rules.

### Testing
Test setup ready with Vitest (tests not implemented yet).

## Deployment

Build the application for production:

```bash
npm run build
```

The built files will be in the `dist/` directory, ready for deployment to any static file server.

### Docker Deployment

```dockerfile
FROM nginx:alpine
COPY dist/ /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
```

## Contributing

1. Follow the existing code style and conventions
2. Add TypeScript types for all new code
3. Test components thoroughly across different screen sizes
4. Update documentation for new features

## License

This project is part of the H200 Intelligent Mug Positioning System.