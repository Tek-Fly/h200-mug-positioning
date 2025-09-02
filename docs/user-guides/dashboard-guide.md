# Dashboard Guide

Complete guide to navigating and using the H200 System web dashboard.

## Dashboard Overview

The web dashboard provides a comprehensive interface for monitoring, managing, and analyzing the H200 Intelligent Mug Positioning System. It's built with Vue.js and offers real-time updates, intuitive navigation, and powerful management capabilities.

### Accessing the Dashboard

**URLs:**
- **Local Development**: http://localhost:3000
- **Production**: https://your-domain.com
- **RunPod**: http://[POD_ID]-3000.proxy.runpod.net

**Login Requirements:**
- Valid user credentials
- Active session (24-hour JWT token)
- Appropriate role permissions

## Navigation Structure

### Main Navigation Menu

The dashboard uses a responsive sidebar navigation:

#### 📊 Dashboard (Overview)
- System health summary
- Key performance metrics
- Recent activity feed
- Cost tracking overview

#### 🔍 Analysis
- Image upload and analysis
- Analysis history
- Batch processing
- Result comparison

#### 📋 Rules
- Rule management
- Natural language rule creation
- Rule testing and validation
- Performance analytics

#### 🖥️ Servers
- Server status and control
- Performance monitoring
- Log viewing
- Deployment management

#### 📈 Metrics
- Detailed performance charts
- Historical data analysis
- Custom metric views
- Export capabilities

#### ⚙️ Settings
- User preferences
- System configuration
- Theme selection
- Notification settings

## Dashboard (Overview Page)

### System Health Panel

**Health Indicators:**
- 🟢 **Healthy**: All systems operational
- 🟡 **Degraded**: Some issues detected
- 🔴 **Unhealthy**: Critical systems down
- ⚪ **Unknown**: Status cannot be determined

**Service Status Cards:**
Each service shows:
- Service name and current status
- Uptime duration
- Key metrics (connections, memory, etc.)
- Last health check timestamp

### Performance Metrics

**Key Performance Indicators (KPIs):**

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Cold Start      │ Warm Start      │ Processing Time │ API Latency     │
│ 1,500ms         │ 50ms           │ 450ms          │ 180ms (p95)     │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ GPU Utilization │ Cache Hit Rate  │ Requests/sec    │ Success Rate    │
│ 75.5%          │ 87.3%          │ 25.5           │ 99.2%          │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

**Performance Charts:**
- Real-time response time graph
- GPU utilization over time
- Request volume and success rates
- Cache performance metrics

### Resource Usage

**System Resources:**
- **CPU**: Current usage percentage and trends
- **Memory**: Used vs. available with breakdown
- **Disk**: Storage utilization and remaining space
- **GPU**: Memory usage and processing capacity

**Visual Indicators:**
- Progress bars for usage percentages
- Color coding (green < 70%, yellow 70-90%, red > 90%)
- Trend arrows (↑ increasing, ↓ decreasing, → stable)

### Cost Tracking

**Cost Breakdown:**
```
Daily Cost Summary
├── Compute (GPU): $84.00 (98.8%)
├── Storage (R2): $0.08 (0.1%)
└── Network: $0.90 (1.1%)
Total: $84.98
```

**Cost Metrics:**
- Cost per request
- Monthly projection
- Optimization opportunities
- Budget alerts and warnings

### Activity Feed

**Recent Activities:**
- Image analysis completed
- Rules created/modified
- Server operations
- User login/logout events

**Activity Details:**
Each entry shows:
- Timestamp
- User who performed action
- Action description
- Duration (where applicable)
- Status (success/failure)

## Analysis Page

### Image Upload Interface

**Upload Methods:**
1. **Drag & Drop**: Drag images directly onto upload area
2. **File Browser**: Click to open file selection dialog
3. **URL Input**: Provide direct image URL
4. **Camera Capture**: Use device camera (mobile/webcam)

**Upload Progress:**
- File size validation
- Upload progress bar
- Preview thumbnail
- Upload completion confirmation

### Analysis Configuration

**Settings Panel:**

**Basic Settings:**
- **Confidence Threshold**: 0.1 - 1.0 (default: 0.7)
- **Include Feedback**: Toggle positioning suggestions
- **Save Results**: Store analysis in history

**Advanced Settings:**
- **Rules Context**: Specify rule application context
- **Calibration**: Manual pixel-to-millimeter ratio
- **Model Selection**: Choose detection model version
- **Output Format**: JSON, detailed report, or summary

### Analysis Results

**Detection Results:**
```
┌─────────────────────────────────────────────────────────────┐
│ 🏷️  Mug Detection Results                                    │
├─────────────────────────────────────────────────────────────┤
│ Detected: 2 mugs                                           │
│ Confidence: 95.2% average                                  │
│ Processing Time: 485ms                                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────┬─────────────┬─────────────┬─────────────────┐
│ Mug ID      │ Confidence  │ Position    │ Attributes      │
├─────────────┼─────────────┼─────────────┼─────────────────┤
│ mug_001     │ 97.3%      │ (150,200)   │ White, Medium   │
│ mug_002     │ 93.1%      │ (450,180)   │ Blue, Large     │
└─────────────┴─────────────┴─────────────┴─────────────────┘
```

**Visual Analysis:**
- Original image with detection boxes
- Positioning overlay with guidelines
- Rule violation highlights
- Suggested improvement arrows

**Positioning Feedback:**
- Position quality assessment
- Specific offset measurements
- Rule compliance status
- Actionable improvement suggestions

### Analysis History

**History Table:**
- Chronological list of all analyses
- Searchable by date, user, or results
- Filterable by success/failure status
- Sortable by any column

**Result Details:**
Click any history entry to view:
- Original uploaded image
- Complete detection results
- Applied rules and violations
- User feedback (if provided)

## Rules Management Page

### Rules Overview

**Rules List:**
```
┌─────────────────────────────────────────────────────────────┐
│ Active Rules: 5 │ Inactive: 2 │ Total Evaluations Today: 247 │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────┬─────────────┬──────────┬──────────────┐
│ Rule Name           │ Type        │ Priority │ Status       │
├──────────────────────┼─────────────┼──────────┼──────────────┤
│ Center on coaster   │ Positioning │ High     │ ✅ Active    │
│ Edge clearance      │ Safety      │ Critical │ ✅ Active    │  
│ Multi-mug spacing   │ Spacing     │ Medium   │ ⚠️ Warning   │
│ Size validation     │ Quality     │ Low      │ ❌ Inactive  │
└──────────────────────┴─────────────┴──────────┴──────────────┘
```

**Rule Categories:**
- **Positioning**: Mug placement and alignment
- **Spacing**: Distance between multiple mugs  
- **Safety**: Edge distance and stability
- **Quality**: Size, orientation, and appearance

### Creating Rules

**Natural Language Interface:**

1. **Rule Description Box:**
   ```
   📝 Describe your positioning rule in plain English:
   
   "The mug should be centered on the coaster with at least 
   2 inches clearance from all edges"
   ```

2. **Context Selection:**
   - Coffee shop setup
   - Formal dining
   - Photography session
   - Custom context

3. **Preview Interpretation:**
   ```
   🤖 System Interpretation:
   
   Rule Type: Positioning + Safety
   Conditions: 
   • Distance from center < 1cm
   • Distance to edges > 5cm
   
   Confidence: 92%
   ```

4. **Rule Configuration:**
   - Priority level selection
   - Auto-enable toggle
   - Notification preferences

**Advanced Rule Builder:**

For precise control:

1. **Condition Builder:**
   - Field selection (distance, alignment, size)
   - Operator choice (greater than, less than, equals)
   - Value input with units
   - Multiple condition logic (AND/OR)

2. **Action Configuration:**
   - Alert messages
   - Severity levels
   - Notification methods
   - Custom responses

### Rule Testing

**Test Interface:**
1. Upload test images
2. Select rules to evaluate
3. Run simulation
4. Review results and accuracy

**Test Results:**
- True positives/negatives
- False positives/negatives  
- Accuracy percentage
- Performance metrics

### Rule Analytics

**Performance Metrics:**
- Evaluation frequency
- Match rate
- User feedback scores
- Performance impact

**Usage Statistics:**
- Most/least triggered rules
- Accuracy trends over time
- User satisfaction ratings
- Optimization opportunities

## Servers Page

### Server Overview

**Server Status Cards:**
```
┌─────────────────────────────────────────────────────────────┐
│ 🖥️ Serverless                        🔄 Timed Instance      │
├─────────────────────────────────────┬─────────────────────────┤
│ Status: ✅ Running                   │ Status: ⏹️ Stopped      │
│ Instances: 1/3                      │ Type: H100 80GB        │
│ Requests: 2,847 today              │ Uptime: 0h 0m          │
│ Avg Response: 145ms                 │ Cost: $0.00 today      │
│ Cost: $45.23 today                  │                        │
└─────────────────────────────────────┴─────────────────────────┘
```

**Quick Actions:**
- Start/Stop servers
- Restart with confirmation
- Scale serverless instances
- View detailed metrics

### Server Control

**Control Panel:**
```
┌─────────────────────────────────────────────────────────────┐
│ 🎮 Server Control                                           │
├─────────────────────────────────────────────────────────────┤
│ [🟢 Start] [🔴 Stop] [🔄 Restart] [⚙️ Configure]            │
│                                                             │
│ Force Stop: ☐ (terminates immediately)                     │
│ Safe Mode: ☑ (completes current requests)                  │
└─────────────────────────────────────────────────────────────┘
```

**Configuration Options:**

**Serverless Settings:**
- Min/Max instances (0-10)
- Idle timeout (5-30 minutes)
- Auto-scaling triggers
- Resource allocation

**Timed Instance Settings:**
- GPU type selection
- Duration settings
- Auto-renewal options
- Resource specifications

### Server Monitoring

**Real-Time Metrics:**
- Live performance graphs
- Request rate and latency
- Resource utilization
- Error rate tracking

**Historical Data:**
- Performance trends
- Usage patterns
- Cost analysis
- Optimization recommendations

### Log Viewer

**Log Interface:**
```
┌─────────────────────────────────────────────────────────────┐
│ 📄 Server Logs                    [📥 Download] [🔄 Refresh] │
├─────────────────────────────────────────────────────────────┤
│ 2025-09-02 10:30:15 | INFO  | Request processed: req_123   │
│ 2025-09-02 10:30:10 | DEBUG | Model loaded successfully    │
│ 2025-09-02 10:29:55 | WARN  | High GPU utilization: 89%    │
│ 2025-09-02 10:29:45 | INFO  | Health check passed          │
└─────────────────────────────────────────────────────────────┘
```

**Log Features:**
- Real-time streaming
- Level filtering (ERROR, WARN, INFO, DEBUG)
- Search and highlighting
- Export to file
- Contextual details on click

## Metrics Page

### Performance Charts

**Interactive Visualizations:**

**Response Time Chart:**
- Line graph showing API response times
- P50, P95, P99 percentile lines
- Configurable time ranges
- Zoom and pan capabilities

**GPU Utilization:**
- Area chart of GPU usage over time
- Memory usage overlay
- Temperature monitoring (if available)
- Efficiency calculations

**Request Volume:**
- Bar chart of requests per time period
- Success/failure breakdown
- Peak usage identification
- Capacity planning insights

### Custom Dashboards

**Dashboard Builder:**
1. **Metric Selection**: Choose from available metrics
2. **Visualization Type**: Line, bar, area, gauge charts
3. **Time Range**: Minutes to months
4. **Refresh Rate**: Real-time to daily updates

**Saved Dashboards:**
- Personal dashboards
- Team shared views
- System templates
- Export/import capability

### Data Export

**Export Options:**
- CSV for spreadsheet analysis
- JSON for programmatic use
- PNG/PDF for reports
- Real-time API endpoints

**Scheduled Reports:**
- Daily/weekly/monthly reports
- Email delivery
- Custom formatting
- Automated insights

## Settings Page

### User Preferences

**Profile Settings:**
- Display name and email
- Password change
- Profile picture
- Contact information

**Dashboard Preferences:**
- Default page on login
- Refresh rates
- Chart preferences
- Notification settings

### Theme and Appearance

**Theme Options:**
- 🌞 Light mode
- 🌙 Dark mode
- 🎨 Auto (system preference)
- High contrast accessibility

**Customization:**
- Color scheme selection
- Font size adjustment
- Layout density
- Animation preferences

### Notification Settings

**Alert Channels:**
- Email notifications
- In-dashboard alerts
- Webhook integrations
- Mobile push (if available)

**Alert Preferences:**
- System health alerts
- Performance warnings
- Cost threshold notifications
- Rule violation alerts

### System Configuration

**API Settings:**
- Rate limiting preferences
- Timeout configurations
- Retry policies
- Cache settings

**Integration Settings:**
- Webhook URLs
- External service tokens
- MCP client configuration
- Third-party integrations

## Mobile Experience

### Responsive Design

The dashboard is fully responsive and optimized for:
- **📱 Mobile phones**: Collapsible navigation, touch-friendly controls
- **📱 Tablets**: Optimized layout with touch gestures
- **💻 Desktop**: Full feature set with keyboard shortcuts

### Mobile-Specific Features

**Touch Gestures:**
- Swipe navigation between sections
- Pull-to-refresh on data views
- Pinch-to-zoom on charts
- Long-press for context menus

**Mobile Optimizations:**
- Simplified upload interface
- Voice-to-text rule creation
- Camera integration for analysis
- Offline result caching

## Keyboard Shortcuts

**Global Shortcuts:**
- `Ctrl/Cmd + K`: Quick search
- `Ctrl/Cmd + D`: Go to Dashboard
- `Ctrl/Cmd + A`: Go to Analysis
- `Ctrl/Cmd + R`: Go to Rules
- `Ctrl/Cmd + S`: Go to Servers
- `Ctrl/Cmd + M`: Go to Metrics

**Page-Specific Shortcuts:**
- `Space`: Start/pause real-time updates
- `F5`: Refresh current page
- `Esc`: Close modals/overlays
- `Tab/Shift+Tab`: Navigate controls

## Troubleshooting Dashboard Issues

### Common Issues

**Dashboard Won't Load:**
1. Check internet connection
2. Verify correct URL
3. Clear browser cache
4. Disable browser extensions
5. Try incognito/private mode

**Performance Issues:**
1. Close unnecessary browser tabs
2. Disable real-time updates temporarily
3. Reduce chart time ranges
4. Check system resources

**Authentication Problems:**
1. Verify credentials
2. Check token expiration
3. Clear stored tokens
4. Contact administrator

**Data Not Updating:**
1. Check WebSocket connection
2. Verify server status
3. Refresh page
4. Check network connectivity

### Getting Help

**In-Dashboard Support:**
- Help tooltips on hover
- Contextual help panels
- Built-in tutorial mode
- Error message guidance

**External Support:**
- Documentation links
- Support ticket system
- Community forums
- Video tutorials

This completes the comprehensive dashboard guide. The interface is designed to be intuitive, but this guide provides detailed information for users who want to maximize their productivity with the system.