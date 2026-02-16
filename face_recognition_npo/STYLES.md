# CSS Classes Reference

**Last Updated**: February 16, 2026

---

## SCSS

The styles are written in SCSS for maintainability.

**Source**: `electron-ui/styles/design-system.scss`  
**Compiled**: `electron-ui/styles/design-system.css`

**Build**:
```bash
cd electron-ui
npm run scss
```

Or use `./start.sh` which auto-compiles.

---

## macOS Tahoe Liquid Glass Design System

The UI uses macOS Tahoe style with glassmorphism effects.

### Glass Variables

```scss
$glass-bg: rgba(255, 255, 255, 0.72);
$glass-bg-strong: rgba(255, 255, 255, 0.85);
$glass-border: rgba(255, 255, 255, 0.5);
$glass-shadow: rgba(0, 0, 0, 0.1);
$glass-highlight: rgba(255, 255, 255, 0.8);
```

---

## Utility Classes

| Class | CSS | Purpose |
|-------|-----|---------|
| `.hidden` | `max-height: 0; opacity: 0; overflow: hidden` | Hide element with animation |
| `.visible` | `max-height: 500px; opacity: 1` | Show element with animation |
| `.button-group` | `display: flex; gap: 8px` | Flex container for buttons |
| `.button-group-center` | `display: flex; gap: 8px; justify-content: center` | Centered button group |
| `.text-center` | `text-align: center` | Center text |
| `.text-error` | `color: var(--color-error)` | Error color from variable |
| `.text-error-inline` | `color: #cc0000` | Red error text |
| `.text-muted` | `color: var(--color-muted)` | Muted text color |
| `.webcam-video` | `max-width: 100%; max-height: 400px; border: 1px solid #000; border-radius: 4px` | Webcam video styling |
| `.btn-margin-top` | `margin-top: 16px` | Button top margin |
| `.empty-state` | `color: #666; padding: 8px` | Empty state styling |

---

## Buttons

| Class | Purpose |
|-------|---------|
| `.btn` | Default button |
| `.btn:hover` | Button hover state |
| `.btn:disabled` | Disabled button |
| `.btn-primary` | Primary button (black background) |
| `.btn-primary:hover` | Primary button hover |
| `.btn-primary:disabled` | Primary button disabled |
| `.btn-success` | Success button (green background) |
| `.btn-success:hover` | Success button hover |
| `.btn-small` | Small button |

---

## Steps & Workflow

| Class | Purpose |
|-------|---------|
| `.container` | Main container (max-width: 1200px, padding-top: 76px for titlebar) |
| `.steps-row` | Flex container for grouping steps |
| `.step` | Workflow step card with glassmorphism |
| `.step-header` | Step header (flex, space-between) |
| `.step-number` | Step number badge (pill-shaped, glass effect) |
| `.step-number.webcam` | Blue webcam badge (#5856D6) |
| `.step-number.optional` | Optional badge |
| `.step-hint` | Step hint text |
| `.step-complete` | Completed step (green border + shadow) |
| `.step-checkmark` | Checkmark circle for completed steps |

---

## Preview & Gallery

| Class | Purpose |
|-------|---------|
| `.preview-container` | Preview images container (animated) |
| `.preview-box` | Single preview box (24px border-radius) |
| `.preview-box img` | Preview image |
| `.preview-label` | Preview label text |
| `.gallery` | Gallery container (flex, wrap) |
| `.gallery-item` | Gallery item |
| `.gallery-item img` | Gallery image (80x80) |
| `.gallery-item span` | Gallery label |

---

## Display/Toggle

| Class | Purpose |
|-------|---------|
| `.reference-details` | Reference details panel |
| `.reference-details.active` | Show reference details |
| `.comparison-result` | Comparison result (hidden by default) |
| `.comparison-result.active` | Show comparison result |

---

## Status Messages

| Class | Purpose |
|-------|---------|
| `.status` | Default status text |
| `.status-success` | Success status (green) |
| `.status-warning` | Warning status (yellow) |
| `.status-error` | Error status (red) |
| `.status-info` | Info status (blue) |

---

## Badges

| Class | Purpose |
|-------|---------|
| `.badge` | Default badge |
| `.badge-success` | Success badge (green) |
| `.badge-warning` | Warning badge (yellow) |
| `.badge-error` | Error badge (red) |

---

## Toast Notifications

| Class | Purpose |
|-------|---------|
| `.toast-container` | Fixed container (top: 76px, right: 24px) |
| `.toast` | Toast with liquid glass effect |
| `.toast.success` | Green tinted toast |
| `.toast.error` | Red tinted toast |
| `.toast.warning` | Yellow tinted toast |

**Toast Features**:
- Liquid glass effect with backdrop blur
- Color tint for type indication
- 5 second display duration
- Smooth fade-out animation

---

## Comparison Display

| Class | Purpose |
|-------|---------|
| `.comparison-result` | Comparison result box |
| `.comparison-content` | Flex container for comparison |
| `.comparison-side` | Left/right comparison image |
| `.comparison-side img` | Comparison image (150x150) |
| `.comparison-center` | Center score display |
| `.comparison-status` | Match/no-match status badge |
| `.comparison-status.match` | Match status (green) |
| `.comparison-status.possible` | Possible match (yellow) |
| `.comparison-status.no-match` | No match (red) |
| `.comparison-scores` | Scores container |
| `.score-row` | Individual score row |
| `.score-row.final` | Final combined score |
| `.score-label` | Score label |
| `.score-value` | Score value (monospace) |
| `.match-reasons` | Match reasons list |

---

## Reference List

| Class | Purpose |
|-------|---------|
| `.reference-list` | Reference list container |
| `.reference-item` | Reference item |
| `.reference-item img` | Reference thumbnail |
| `.reference-item span` | Reference name |
| `.ref-remove-btn` | Remove reference button (red circle) |
| `.ref-remove-btn:hover` | Remove button hover |
| `.ref-remove-btn:disabled` | Remove button disabled |

---

## Reference Details

| Class | Purpose |
|-------|---------|
| `.ref-details-header` | Details header (flex, space-between) |
| `.ref-viz-tabs` | Visualization tabs (horizontal scroll) |
| `.ref-viz-tab` | Visualization tab |
| `.ref-viz-tab.active` | Active tab |
| `.ref-viz-content` | Visualization content area |
| `.ref-info-grid` | Info grid (auto-fill) |
| `.ref-info-item` | Info item box |
| `.ref-info-label` | Info label |
| `.ref-info-value` | Info value |

---

## Visualization Tabs

| Class | Purpose |
|-------|---------|
| `.viz-tabs` | Tabs container (horizontal scroll) |
| `.viz-tab` | Tab button |
| `.viz-tab:hover` | Tab hover |
| `.viz-tab.active` | Active tab |
| `.viz-content` | Visualization content area |
| `.viz-content img` | Visualization image |
| `.viz-placeholder` | Placeholder when no content |
| `.viz-placeholder p` | Placeholder text |

---

## Loading

| Class | Purpose |
|-------|---------|
| `.loading-overlay` | Full-screen overlay |
| `.loading-overlay.active` | Show overlay |
| `.loading-content` | Loading content center |
| `.loading-spinner` | Spinner animation |
| `.loading-text` | Loading text |

---

## Terminal

| Class | Purpose |
|-------|---------|
| `.terminal-footer` | Fixed footer at bottom |
| `.terminal-header` | Terminal header |
| `.terminal-title` | Terminal title |
| `.terminal-toggle` | Toggle button |
| `.terminal-log` | Log content area (32px collapsed) |
| `.terminal-log.expanded` | Expanded log (140px) |
| `.terminal-log-content` | Log content padding |
| `.terminal-line` | Log line |
| `.terminal-line.command` | Command line (cyan) |
| `.terminal-line.success` | Success line (green) |
| `.terminal-line.info` | Info line (yellow) |
| `.terminal-line.error` | Error line (red) |

---

## Data Table

| Class | Purpose |
|-------|---------|
| `.viz-data-table` | Data table container |
| `.viz-data-table table` | Table styling |
| `.viz-data-table td.label` | Label column (40%, gray bg) |
| `.viz-data-table td.value` | Value column (monospace) |

---

## Responsive

| Breakpoint | CSS Variable Changes |
|------------|---------------------|
| `@media (max-width: 1200px)` | `--sidebar-width: 240px` |
| `@media (max-width: 992px)` | Flex direction column for comparison |
| `@media (max-width: 768px)` | `--header-height: 50px` |
| `@media (max-width: 900px)` | `.steps-row` flex-direction: column |

---

## CSS Variables

### Colors

```css
/* Primary */
--color-primary: #007AFF

/* Semantic */
--color-success: #30D158
--color-warning: #FFD60A
--color-error: #FF453A

/* Background */
--bg-primary: #F5F5F7
--bg-secondary: #f5f5f5
--bg-card: #FFFFFF

/* Text */
--text-primary: #1D1D1F
--text-secondary: #86868B
--color-muted: #666

/* Border */
--border-light: #E5E5EA
--border-default: #D2D2D7
--color-border: #000
```

### Spacing

```css
--space-1: 4px
--space-2: 8px
--space-3: 12px
--space-4: 16px
--space-8: 32px
```

### Typography

```css
/* Size */
--text-xs: 11px
--text-sm: 12px
--text-base: 14px
--text-lg: 16px
--text-xl: 20px
--text-2xl: 24px
--text-3xl: 32px
--text-4xl: 40px

/* Weight */
--font-normal: 400
--font-medium: 500
--font-semibold: 600
--font-bold: 700
```

---

## Animations

### Step Container Transition
```scss
.step {
  transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
```

### Preview Container Animation
```scss
.preview-container, .hidden, .visible {
  transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
```

### Toast Animation
- Display: 5 seconds
- Fade out: 300ms

---

*Document created: February 15, 2026*
*Last updated: February 16, 2026*
*Added: macOS Tahoe glassmorphism, step completion indicators, animated containers*
