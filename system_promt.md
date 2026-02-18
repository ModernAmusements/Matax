# SYSTEM PROMPT - MANTAX (NGO Facial Image Analysis System)

## Project Overview

MANTAX is an ethical, consent-based facial recognition system designed for NGO use cases such as documentation verification, missing persons investigations, and trafficking victim identification. Version 0.5.2 (February 2026).

**Core ethical principles:**
- No automated identification decisions -- human review always required
- Confidence bands instead of binary match/no-match
- Non-reversible embeddings (cannot reconstruct faces)
- Consent tracking and audit trails

---

## Architecture

### Three-tier Architecture
```
Electron Desktop App (or Browser)
        |
        | HTTP REST (port 3000)
        v
Flask API Server (api_server.py)
        |
        v
Core ML Pipeline (src/)
```

### Backend (api_server.py - 1,512 lines)
- Flask server on port 3000
- In-memory session state
- Dual-model: Always loads both ArcFace (512-dim) and FaceNet (128-dim)
- Persistence: References saved to reference_images/embeddings.json

### ML Pipeline (src/)
- `src/detection/__init__.py` - FaceDetector (OpenCV DNN, MediaPipe landmarks)
- `src/embedding/__init__.py` - FaceNetExtractor + SimilarityComparator
- `src/embedding/arcface_extractor.py` - ArcFace ONNX implementation

### Frontend (electron-ui/)
- Electron Desktop App (frameless with custom titlebar)
- HTML/CSS/JavaScript with macOS Tahoe Liquid Glass design
- Custom titlebar with traffic lights, sidebar toggle, step navigation
- Three themes: light, dim, dark

---

## NEW: Apple Tahoe Sidebar (v0.5.2)

### Sidebar Features
- Slides in/out with smooth animation (260px width)
- Overlay mode: slides under titlebar (z-index: 998)
- Content pushes main container when open
- Glass liquid effect with backdrop-filter blur

### Sidebar HTML Structure
```html
<div class="sidebar" id="sidebar">
    <div class="sidebar-content">
        <div class="sidebar-step active" onclick="jumpToStep(1)">
            <img src="person.fill.svg" class="step-icon">
            <span>Choose Photo</span>
        </div>
    </div>
</div>
```

### Sidebar CSS
- `.sidebar`: position: fixed, width: 260px, z-index: 998
- `.sidebar.open`: transform: translateX(0)
- `.container.sidebar-open`: margin-left: 260px

---

## NEW: Custom Titlebar (v0.5.2)

### Configuration
- Electron: `frame: false, titleBarStyle: 'hidden', trafficLightPosition: { x: -100, y: 0 }`
- Custom traffic lights in HTML (connected via IPC)

### Titlebar Layout
```
┌─────────────────────────────────────────────────────────┐
│ [☰] [●○○] [Step1] [Step2] [Step3] [Step4]   [⋮]    │
│  ↑     ↑       ↑                              ↑        │
│ Side  Traffic Step nav icons (click→jump)   Menu     │
│ toggle Lights (liquid glass style)          (nothing) │
└─────────────────────────────────────────────────────────┘
```

### HTML Structure
```html
<div class="titlebar">
    <div class="titlebar-left">
        <div class="traffic-lights">
            <div class="light close" onclick="closeWindow()"></div>
            <div class="light minimize" onclick="minimizeWindow()"></div>
            <div class="light maximize" onclick="maximizeWindow()"></div>
        </div>
        <img src="sidebar.left.svg" onclick="toggleSidebar()">
    </div>
    <div class="titlebar-center">
        <img src="person.fill.svg" class="step-nav-icon" onclick="jumpToStep(1)">
        <img src="face.dashed.fill.svg" class="step-nav-icon" onclick="jumpToStep(2)">
        <img src="photo.badge.plus.svg" class="step-nav-icon" onclick="jumpToStep(3)">
        <img src="list.number.svg" class="step-nav-icon" onclick="jumpToStep(4)">
    </div>
    <div class="titlebar-right">
        <img src="ellipsis.svg" class="menu-icon">
    </div>
</div>
```

### Step Navigation
- Click step icon → jumpToStep(n) → scrolls to step AND highlights icon
- Active state: `.step-nav-icon.active` with background highlight
- Liquid glass container style in `.titlebar-center`

### Traffic Light IPC
- preload.js exposes: close(), minimize(), maximize()
- main.js handles: ipcMain.on('window-close/minimize/maximize')

### Available Icons (sf_icons/)
| Icon | Use |
|------|-----|
| sidebar.left.svg | Sidebar toggle |
| person.fill.svg | Step 1 - Choose Photo |
| face.dashed.fill.svg | Step 2 - Find Faces |
| photo.badge.plus.svg | Step 3 - Create Signature |
| list.number.svg | Step 4 - Compare |
| ellipsis.svg | 3-dot menu |

---

## API Endpoints (20+)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/health` | Health check |
| GET | `/api/embedding-info` | Model info |
| GET | `/api/diagnostics` | System diagnostics |
| POST | `/api/detect` | Face detection |
| POST | `/api/extract` | Embedding extraction |
| POST | `/api/add-reference` | Add reference |
| POST | `/api/add-reference-pose/<id>` | Add pose variant |
| GET | `/api/references` | List references |
| DELETE | `/api/references/<id>` | Remove reference |
| POST | `/api/compare` | Multi-factor comparison |
| GET | `/api/visualizations/<type>` | Get visualization |
| GET | `/api/visualizations/<type>/reference/<id>` | Reference visualization |
| GET | `/api/quality` | Quality metrics |
| GET | `/api/eyewear` | Eyewear detection |
| POST | `/api/clear` | Clear session |
| GET | `/api/status` | Debug state |
| GET | `/api/webcam/available` | Webcam availability |
| POST | `/api/webcam/capture` | Capture from webcam |
| POST | `/api/webcam/detect` | Detect from webcam |

---

## Frontend Structure

### HTML (index.html - 350+ lines)
- 5-step workflow: Choose Photo, Find Faces, Create Signature, Compare
- Sidebar with step navigation
- Custom titlebar with traffic lights, toggle, step icons, menu
- 19+ visualization tabs (detection, landmarks, mesh3d, embedding, etc.)
- 9 test tabs
- Comparison results with expandable score breakdown
- Reference details panel
- Terminal footer, toast notifications

### JavaScript (app.js - 1,400+ lines)
- All API communication via fetch()
- Sidebar: toggleSidebar(), jumpToStep(n)
- Traffic lights: closeWindow(), minimizeWindow(), maximizeWindow()
- Radio-button-based visualization tabs with sliding indicator animation
- Key functions: handleImageSelect, detectFaces, extractFeatures, compareFaces, saveReference, showVisualization

### CSS (design-system.scss - 2,100+ lines)
- macOS Tahoe Liquid Glass design system
- Glassmorphism effects with backdrop-filter and saturate
- Layered box shadows for glass reflections
- CSS custom properties for theming
- Sidebar styles (.sidebar, .sidebar.open, .container.sidebar-open)
- Titlebar styles (.titlebar-center, .step-nav-icon, .traffic-lights)

---

## Testing Approach

1. **E2E Tests** (test_e2e_pipeline.py): 6 tests
2. **Edge Case Tests** (test_edge_cases.py): 16 tests
3. **Frontend Tests** (test_frontend_integration.py): 9 tests
4. **Unit Tests** (tests/): 30 tests via pytest

**MANDATORY**: All tests must pass after code changes.

---

## Key Patterns & Conventions

1. **Startup**: Use start.sh (kills old processes, clears cache, starts Flask)
2. **Session-based state**: Single-user design, in-memory state
3. **Dual-model**: ArcFace primary (better discrimination), FaceNet secondary
4. **Fire-and-forget**: Non-blocking API calls use .catch() not await
5. **Human-in-loop**: Confidence bands (Very High/High/Moderate/Insufficient)
6. **Real embeddings only**: Random/placeholder embeddings forbidden

---

## Strict Coding Rules (from CONTEXT.md)

### Rule 1: Syntax Check
Run `python -m py_compile <file>` before submitting.

### Rule 2: Check for Duplicate Code
Use grep to find duplicate function definitions.

### Rule 3: Import Verification
Run module to verify imports work.

### Rule 4: Read Before Edit
Read 50+ lines around edit location.

### Rule 5: Function Preservation (JavaScript)
Always verify HTML onclick/onchange handlers exist in app.js:
```bash
grep -E 'onclick=|onchange=' electron-ui/index.html
```

### Rule 6: Atomic Edits
Make ONE edit per function, verify each individually.

### Rule 7: Fire-and-Forget for Non-Critical APIs
Use .catch() instead of await for non-essential calls.

### Rule 8: HTML-JS Cross-Check (MANDATORY)
Before every commit:
```bash
for func in $(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u); do
    grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js || echo "MISSING: $func"
done
```

---

## Common Mistakes (from CONTEXT.md)

| # | Mistake | Solution |
|---|---------|----------|
| 1 | Missing closing paren | Count parentheses |
| 2 | Duplicate code left behind | grep function definitions |
| 3 | Not running syntax check | python -m py_compile |
| 4 | Not reading context | Read 50+ lines before edit |
| 5 | Not testing edge cases | Run tests after edits |
| 6 | Blocking UI with async/await | Use .catch() for fire-and-forget |
| 7 | Missing HTML-JS cross-check | Verify onclick handlers exist |
| 8 | References not persisting | Call save_references() |
| 9 | Old process caching code | Use start.sh |
| 10 | Wrong test image paths | Use test_subject.jpg |
| 11 | Test viz returning empty data | Return data dict as second tuple |

---

## Build Commands

```bash
# Compile SCSS
cd electron-ui && npm run scss

# Start application
cd face_recognition_npo && ./start.sh

# Run tests
python test_e2e_pipeline.py
python test_edge_cases.py
python test_frontend_integration.py
```

---

## File Locations

| File | Purpose |
|------|---------|
| api_server.py | Flask API (backend) |
| src/detection/__init__.py | Face detection & landmarks |
| src/embedding/__init__.py | Embedding extraction & comparison |
| electron-ui/renderer/app.js | Frontend logic |
| electron-ui/index.html | Frontend HTML |
| electron-ui/styles/design-system.scss | Design system (source) |
| electron-ui/styles/design-system.css | Design system (compiled) |
| electron-ui/main.js | Electron main process |
| electron-ui/preload.js | Electron preload (IPC) |
| start.sh | Startup script |
| test_e2e_pipeline.py | E2E tests |
| test_edge_cases.py | Edge case tests |

---

*Last updated: February 18, 2026*
