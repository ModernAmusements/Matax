# MASTER PROMPT

You are about to work on an existing project repository.

Your first and most important task is full contextual preparation before making any assumptions, suggestions, or implementations.

Follow these instructions strictly:

## Phase 1: Contextual Preparation

### Step 1: Read System Prompt
First, read `system_promt.md` to understand the project context.

### Step 2: Model
Systematically explore Build Mental and understand:
- Project structure and directory layout
- Architecture (backend, frontend, ML pipeline)
- API endpoints and contracts
- Frontend components and state management
- Testing approach
- Key patterns and conventions

### Step 3: Identify Key Files
Locate and understand:
- API server (api_server.py)
- Frontend (electron-ui/renderer/app.js, index.html)
- Design system (electron-ui/styles/design-system.scss)
- ML modules (src/detection/, src/embedding/)
- Test files

---

## Important Rules

### Preparation Quality
- Preparation quality determines output quality
- If context is incomplete, explicitly state what is missing
- If files reference other files not provided, flag them
- Do NOT hallucinate missing code
- Do NOT invent files

### Code Changes
- After every code change, verify with tests
- Run: `python test_e2e_pipeline.py`, `python test_edge_cases.py`
- Verify JavaScript with: `node --check electron-ui/renderer/app.js`
- Compile SCSS: `cd electron-ui && npm run scss`

### HTML-JS Cross-Check (MANDATORY)
Before every commit:
```bash
for func in $(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u); do
    grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js || echo "MISSING: $func"
done
```

### Syntax Verification
- Python: `python -m py_compile <file>`
- JavaScript: `node --check electron-ui/renderer/app.js`
- SCSS: `cd electron-ui && npm run scss`

---

## Project: MANTAX (NGO Facial Image Analysis System)

### Technology Stack
- **Backend**: Python Flask (api_server.py)
- **Frontend**: Electron + Vanilla JS + SCSS (frameless with custom titlebar)
- **ML**: ArcFace (ONNX) + FaceNet (PyTorch)
- **Design**: macOS Tahoe Liquid Glass

### Key Commands
```bash
# Start application
cd face_recognition_npo && ./start.sh

# Run tests
python test_e2e_pipeline.py
python test_edge_cases.py
python test_frontend_integration.py

# Build frontend
cd electron-ui && npm run scss
```

---

## NEW Features (v0.5.2)

### Apple Tahoe Sidebar
- Slides in/out (260px width)
- Overlay mode under titlebar
- Glass liquid effect

### Custom Titlebar
- Frameless window with custom traffic lights
- Step navigation icons in center (liquid glass style)
- Sidebar toggle on left, 3-dot menu on right

### Available Icons (electron-ui/styles/sf_icons/)
- sidebar.left.svg - Sidebar toggle
- person.fill.svg - Step 1
- face.dashed.fill.svg - Step 2
- photo.badge.plus.svg - Step 3
- list.number.svg - Step 4
- ellipsis.svg - Menu

---

*Last updated: February 18, 2026*
