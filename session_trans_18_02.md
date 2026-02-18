# Light chat greeting

**Session ID:** ses_38f778453ffe2G9oSIMPvDTiIc
**Created:** 2/18/2026, 12:35:18 PM
**Updated:** 2/18/2026, 12:39:18 PM

---

## User

hi 

---

## Assistant (Build · minimax-m2.5-free · 3.4s)

_Thinking:_

just said "hiThe user" - this is a simple greeting. I should respond with a brief, friendly greeting and offer to help with their task. Since this is a greeting and not a specific request, I should keep my response short and concise as per the instructions.

Hi! How can I help you today?

---

## User

can you read the master and system promt and follow the instructioons?

---

## Assistant (Build · minimax-m2.5-free · 5.9s)

_Thinking:_

The user is asking me to read the system prompt and follow the instructions. Looking at the conversation, I can see there are system messages that contain instructions. The user wants me to read and follow those instructions.

However, I don't have a specific file to read - the system prompt and instructions are already visible to me in this conversation. The system prompt at the beginning tells me I'm "opencode, an interactive CLI tool" and provides instructions about how to behave and respond.

The key instructions I should follow:
1. Be concise, direct, and to the point
2. Keep responses short (fewer than 4 lines unless user asks for detail)
3. Answer directly without unnecessary preamble or postamble
4. Don't add comments unless asked
5. Follow code style conventions

So I should acknowledge this and let them know I'm ready to help. I should be brief as per my instructions.

I already have the system instructions loaded. I'll follow them - keep responses concise, avoid unnecessary preamble/postamble, and focus on direct answers. What would you like me to help with?

---

## User

no in this projekt

---

## Assistant (Build · minimax-m2.5-free · 2.3s)

_Thinking:_

The user wants me to read the master and system prompt in this project. Let me look for these files in the project directory.

**Tool: glob**

**Input:**
```json
{
  "pattern": "**/*master*"
}
```

**Output:**
```
/Users/modernamusmenet/Desktop/MANTAX/master_promt.md
```

**Tool: glob**

**Input:**
```json
{
  "pattern": "**/*system*prompt*"
}
```

**Output:**
```
No files found
```

---

## Assistant (Build · minimax-m2.5-free · 8.4s)

_Thinking:_

Found a master_promt.md file. Let me read it.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/master_promt.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/master_promt.md</path>
<type>file</type>
<content>1: # MASTER PROMPT
2: 
3: You are about to work on an existing project repository.
4: 
5: Your first and most important task is full contextual preparation before making any assumptions, suggestions, or implementations.
6: 
7: Follow these instructions strictly:
8: 
9: ## Phase 1: Contextual Preparation
10: 
11: ### Step 1: Read System Prompt
12: First, read `system_promt.md` to understand the project context.
13: 
14: ### Step 2: Model
15: Systematically explore Build Mental and understand:
16: - Project structure and directory layout
17: - Architecture (backend, frontend, ML pipeline)
18: - API endpoints and contracts
19: - Frontend components and state management
20: - Testing approach
21: - Key patterns and conventions
22: 
23: ### Step 3: Identify Key Files
24: Locate and understand:
25: - API server (api_server.py)
26: - Frontend (electron-ui/renderer/app.js, index.html)
27: - Design system (electron-ui/styles/design-system.scss)
28: - ML modules (src/detection/, src/embedding/)
29: - Test files
30: 
31: ---
32: 
33: ## Important Rules
34: 
35: ### Preparation Quality
36: - Preparation quality determines output quality
37: - If context is incomplete, explicitly state what is missing
38: - If files reference other files not provided, flag them
39: - Do NOT hallucinate missing code
40: - Do NOT invent files
41: 
42: ### Code Changes
43: - After every code change, verify with tests
44: - Run: `python test_e2e_pipeline.py`, `python test_edge_cases.py`
45: - Verify JavaScript with: `node --check electron-ui/renderer/app.js`
46: - Compile SCSS: `cd electron-ui && npm run scss`
47: 
48: ### HTML-JS Cross-Check (MANDATORY)
49: Before every commit:
50: ```bash
51: for func in $(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u); do
52:     grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js || echo "MISSING: $func"
53: done
54: ```
55: 
56: ### Syntax Verification
57: - Python: `python -m py_compile <file>`
58: - JavaScript: `node --check electron-ui/renderer/app.js`
59: - SCSS: `cd electron-ui && npm run scss`
60: 
61: ---
62: 
63: ## Project: MANTAX (NGO Facial Image Analysis System)
64: 
65: ### Technology Stack
66: - **Backend**: Python Flask (api_server.py)
67: - **Frontend**: Electron + Vanilla JS + SCSS (frameless with custom titlebar)
68: - **ML**: ArcFace (ONNX) + FaceNet (PyTorch)
69: - **Design**: macOS Tahoe Liquid Glass
70: 
71: ### Key Commands
72: ```bash
73: # Start application
74: cd face_recognition_npo && ./start.sh
75: 
76: # Run tests
77: python test_e2e_pipeline.py
78: python test_edge_cases.py
79: python test_frontend_integration.py
80: 
81: # Build frontend
82: cd electron-ui && npm run scss
83: ```
84: 
85: ---
86: 
87: ## NEW Features (v0.5.2)
88: 
89: ### Apple Tahoe Sidebar
90: - Slides in/out (260px width)
91: - Overlay mode under titlebar
92: - Glass liquid effect
93: 
94: ### Custom Titlebar
95: - Frameless window with custom traffic lights
96: - Step navigation icons in center (liquid glass style)
97: - Sidebar toggle on left, 3-dot menu on right
98: 
99: ### Available Icons (electron-ui/styles/sf_icons/)
100: - sidebar.left.svg - Sidebar toggle
101: - person.fill.svg - Step 1
102: - face.dashed.fill.svg - Step 2
103: - photo.badge.plus.svg - Step 3
104: - list.number.svg - Step 4
105: - ellipsis.svg - Menu
106: 
107: ---
108: 
109: *Last updated: February 18, 2026*
110: 

(End of file - total 110 lines)
</content>
```

---

## Assistant (Build · minimax-m2.5-free · 3.1s)

_Thinking:_

Now I need to read the system_promt.md file as instructed in Step 1.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/system_promt.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/system_promt.md</path>
<type>file</type>
<content>1: # SYSTEM PROMPT - MANTAX (NGO Facial Image Analysis System)
2: 
3: ## Project Overview
4: 
5: MANTAX is an ethical, consent-based facial recognition system designed for NGO use cases such as documentation verification, missing persons investigations, and trafficking victim identification. Version 0.5.2 (February 2026).
6: 
7: **Core ethical principles:**
8: - No automated identification decisions -- human review always required
9: - Confidence bands instead of binary match/no-match
10: - Non-reversible embeddings (cannot reconstruct faces)
11: - Consent tracking and audit trails
12: 
13: ---
14: 
15: ## Architecture
16: 
17: ### Three-tier Architecture
18: ```
19: Electron Desktop App (or Browser)
20:         |
21:         | HTTP REST (port 3000)
22:         v
23: Flask API Server (api_server.py)
24:         |
25:         v
26: Core ML Pipeline (src/)
27: ```
28: 
29: ### Backend (api_server.py - 1,512 lines)
30: - Flask server on port 3000
31: - In-memory session state
32: - Dual-model: Always loads both ArcFace (512-dim) and FaceNet (128-dim)
33: - Persistence: References saved to reference_images/embeddings.json
34: 
35: ### ML Pipeline (src/)
36: - `src/detection/__init__.py` - FaceDetector (OpenCV DNN, MediaPipe landmarks)
37: - `src/embedding/__init__.py` - FaceNetExtractor + SimilarityComparator
38: - `src/embedding/arcface_extractor.py` - ArcFace ONNX implementation
39: 
40: ### Frontend (electron-ui/)
41: - Electron Desktop App (frameless with custom titlebar)
42: - HTML/CSS/JavaScript with macOS Tahoe Liquid Glass design
43: - Custom titlebar with traffic lights, sidebar toggle, step navigation
44: - Three themes: light, dim, dark
45: 
46: ---
47: 
48: ## NEW: Apple Tahoe Sidebar (v0.5.2)
49: 
50: ### Sidebar Features
51: - Slides in/out with smooth animation (260px width)
52: - Overlay mode: slides under titlebar (z-index: 998)
53: - Content pushes main container when open
54: - Glass liquid effect with backdrop-filter blur
55: 
56: ### Sidebar HTML Structure
57: ```html
58: <div class="sidebar" id="sidebar">
59:     <div class="sidebar-content">
60:         <div class="sidebar-step active" onclick="jumpToStep(1)">
61:             <img src="person.fill.svg" class="step-icon">
62:             <span>Choose Photo</span>
63:         </div>
64:     </div>
65: </div>
66: ```
67: 
68: ### Sidebar CSS
69: - `.sidebar`: position: fixed, width: 260px, z-index: 998
70: - `.sidebar.open`: transform: translateX(0)
71: - `.container.sidebar-open`: margin-left: 260px
72: 
73: ---
74: 
75: ## NEW: Custom Titlebar (v0.5.2)
76: 
77: ### Configuration
78: - Electron: `frame: false, titleBarStyle: 'hidden', trafficLightPosition: { x: -100, y: 0 }`
79: - Custom traffic lights in HTML (connected via IPC)
80: 
81: ### Titlebar Layout
82: ```
83: ┌─────────────────────────────────────────────────────────┐
84: │ [☰] [●○○] [Step1] [Step2] [Step3] [Step4]   [⋮]    │
85: │  ↑     ↑       ↑                              ↑        │
86: │ Side  Traffic Step nav icons (click→jump)   Menu     │
87: │ toggle Lights (liquid glass style)          (nothing) │
88: └─────────────────────────────────────────────────────────┘
89: ```
90: 
91: ### HTML Structure
92: ```html
93: <div class="titlebar">
94:     <div class="titlebar-left">
95:         <div class="traffic-lights">
96:             <div class="light close" onclick="closeWindow()"></div>
97:             <div class="light minimize" onclick="minimizeWindow()"></div>
98:             <div class="light maximize" onclick="maximizeWindow()"></div>
99:         </div>
100:         <img src="sidebar.left.svg" onclick="toggleSidebar()">
101:     </div>
102:     <div class="titlebar-center">
103:         <img src="person.fill.svg" class="step-nav-icon" onclick="jumpToStep(1)">
104:         <img src="face.dashed.fill.svg" class="step-nav-icon" onclick="jumpToStep(2)">
105:         <img src="photo.badge.plus.svg" class="step-nav-icon" onclick="jumpToStep(3)">
106:         <img src="list.number.svg" class="step-nav-icon" onclick="jumpToStep(4)">
107:     </div>
108:     <div class="titlebar-right">
109:         <img src="ellipsis.svg" class="menu-icon">
110:     </div>
111: </div>
112: ```
113: 
114: ### Step Navigation
115: - Click step icon → jumpToStep(n) → scrolls to step AND highlights icon
116: - Active state: `.step-nav-icon.active` with background highlight
117: - Liquid glass container style in `.titlebar-center`
118: 
119: ### Traffic Light IPC
120: - preload.js exposes: close(), minimize(), maximize()
121: - main.js handles: ipcMain.on('window-close/minimize/maximize')
122: 
123: ### Available Icons (sf_icons/)
124: | Icon | Use |
125: |------|-----|
126: | sidebar.left.svg | Sidebar toggle |
127: | person.fill.svg | Step 1 - Choose Photo |
128: | face.dashed.fill.svg | Step 2 - Find Faces |
129: | photo.badge.plus.svg | Step 3 - Create Signature |
130: | list.number.svg | Step 4 - Compare |
131: | ellipsis.svg | 3-dot menu |
132: 
133: ---
134: 
135: ## API Endpoints (20+)
136: 
137: | Method | Endpoint | Purpose |
138: |--------|----------|---------|
139: | GET | `/api/health` | Health check |
140: | GET | `/api/embedding-info` | Model info |
141: | GET | `/api/diagnostics` | System diagnostics |
142: | POST | `/api/detect` | Face detection |
143: | POST | `/api/extract` | Embedding extraction |
144: | POST | `/api/add-reference` | Add reference |
145: | POST | `/api/add-reference-pose/<id>` | Add pose variant |
146: | GET | `/api/references` | List references |
147: | DELETE | `/api/references/<id>` | Remove reference |
148: | POST | `/api/compare` | Multi-factor comparison |
149: | GET | `/api/visualizations/<type>` | Get visualization |
150: | GET | `/api/visualizations/<type>/reference/<id>` | Reference visualization |
151: | GET | `/api/quality` | Quality metrics |
152: | GET | `/api/eyewear` | Eyewear detection |
153: | POST | `/api/clear` | Clear session |
154: | GET | `/api/status` | Debug state |
155: | GET | `/api/webcam/available` | Webcam availability |
156: | POST | `/api/webcam/capture` | Capture from webcam |
157: | POST | `/api/webcam/detect` | Detect from webcam |
158: 
159: ---
160: 
161: ## Frontend Structure
162: 
163: ### HTML (index.html - 350+ lines)
164: - 5-step workflow: Choose Photo, Find Faces, Create Signature, Compare
165: - Sidebar with step navigation
166: - Custom titlebar with traffic lights, toggle, step icons, menu
167: - 19+ visualization tabs (detection, landmarks, mesh3d, embedding, etc.)
168: - 9 test tabs
169: - Comparison results with expandable score breakdown
170: - Reference details panel
171: - Terminal footer, toast notifications
172: 
173: ### JavaScript (app.js - 1,400+ lines)
174: - All API communication via fetch()
175: - Sidebar: toggleSidebar(), jumpToStep(n)
176: - Traffic lights: closeWindow(), minimizeWindow(), maximizeWindow()
177: - Radio-button-based visualization tabs with sliding indicator animation
178: - Key functions: handleImageSelect, detectFaces, extractFeatures, compareFaces, saveReference, showVisualization
179: 
180: ### CSS (design-system.scss - 2,100+ lines)
181: - macOS Tahoe Liquid Glass design system
182: - Glassmorphism effects with backdrop-filter and saturate
183: - Layered box shadows for glass reflections
184: - CSS custom properties for theming
185: - Sidebar styles (.sidebar, .sidebar.open, .container.sidebar-open)
186: - Titlebar styles (.titlebar-center, .step-nav-icon, .traffic-lights)
187: 
188: ---
189: 
190: ## Testing Approach
191: 
192: 1. **E2E Tests** (test_e2e_pipeline.py): 6 tests
193: 2. **Edge Case Tests** (test_edge_cases.py): 16 tests
194: 3. **Frontend Tests** (test_frontend_integration.py): 9 tests
195: 4. **Unit Tests** (tests/): 30 tests via pytest
196: 
197: **MANDATORY**: All tests must pass after code changes.
198: 
199: ---
200: 
201: ## Key Patterns & Conventions
202: 
203: 1. **Startup**: Use start.sh (kills old processes, clears cache, starts Flask)
204: 2. **Session-based state**: Single-user design, in-memory state
205: 3. **Dual-model**: ArcFace primary (better discrimination), FaceNet secondary
206: 4. **Fire-and-forget**: Non-blocking API calls use .catch() not await
207: 5. **Human-in-loop**: Confidence bands (Very High/High/Moderate/Insufficient)
208: 6. **Real embeddings only**: Random/placeholder embeddings forbidden
209: 
210: ---
211: 
212: ## Strict Coding Rules (from CONTEXT.md)
213: 
214: ### Rule 1: Syntax Check
215: Run `python -m py_compile <file>` before submitting.
216: 
217: ### Rule 2: Check for Duplicate Code
218: Use grep to find duplicate function definitions.
219: 
220: ### Rule 3: Import Verification
221: Run module to verify imports work.
222: 
223: ### Rule 4: Read Before Edit
224: Read 50+ lines around edit location.
225: 
226: ### Rule 5: Function Preservation (JavaScript)
227: Always verify HTML onclick/onchange handlers exist in app.js:
228: ```bash
229: grep -E 'onclick=|onchange=' electron-ui/index.html
230: ```
231: 
232: ### Rule 6: Atomic Edits
233: Make ONE edit per function, verify each individually.
234: 
235: ### Rule 7: Fire-and-Forget for Non-Critical APIs
236: Use .catch() instead of await for non-essential calls.
237: 
238: ### Rule 8: HTML-JS Cross-Check (MANDATORY)
239: Before every commit:
240: ```bash
241: for func in $(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u); do
242:     grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js || echo "MISSING: $func"
243: done
244: ```
245: 
246: ---
247: 
248: ## Common Mistakes (from CONTEXT.md)
249: 
250: | # | Mistake | Solution |
251: |---|---------|----------|
252: | 1 | Missing closing paren | Count parentheses |
253: | 2 | Duplicate code left behind | grep function definitions |
254: | 3 | Not running syntax check | python -m py_compile |
255: | 4 | Not reading context | Read 50+ lines before edit |
256: | 5 | Not testing edge cases | Run tests after edits |
257: | 6 | Blocking UI with async/await | Use .catch() for fire-and-forget |
258: | 7 | Missing HTML-JS cross-check | Verify onclick handlers exist |
259: | 8 | References not persisting | Call save_references() |
260: | 9 | Old process caching code | Use start.sh |
261: | 10 | Wrong test image paths | Use test_subject.jpg |
262: | 11 | Test viz returning empty data | Return data dict as second tuple |
263: 
264: ---
265: 
266: ## Build Commands
267: 
268: ```bash
269: # Compile SCSS
270: cd electron-ui && npm run scss
271: 
272: # Start application
273: cd face_recognition_npo && ./start.sh
274: 
275: # Run tests
276: python test_e2e_pipeline.py
277: python test_edge_cases.py
278: python test_frontend_integration.py
279: ```
280: 
281: ---
282: 
283: ## File Locations
284: 
285: | File | Purpose |
286: |------|---------|
287: | api_server.py | Flask API (backend) |
288: | src/detection/__init__.py | Face detection & landmarks |
289: | src/embedding/__init__.py | Embedding extraction & comparison |
290: | electron-ui/renderer/app.js | Frontend logic |
291: | electron-ui/index.html | Frontend HTML |
292: | electron-ui/styles/design-system.scss | Design system (source) |
293: | electron-ui/styles/design-system.css | Design system (compiled) |
294: | electron-ui/main.js | Electron main process |
295: | electron-ui/preload.js | Electron preload (IPC) |
296: | start.sh | Startup script |
297: | test_e2e_pipeline.py | E2E tests |
298: | test_edge_cases.py | Edge case tests |
299: 
300: ---
301: 
302: *Last updated: February 18, 2026*
303: 

(End of file - total 303 lines)
</content>
```

---

## Assistant (Build · minimax-m2.5-free · 6.8s)

_Thinking:_

Now I've read both the master_promt.md and system_promt.md. Let me summarize the key instructions:

1. **Contextual Preparation Phase**: 
   - Read system_promt.md ✓ (done)
   - Explore project structure
   - Identify key files

2. **Important Rules**:
   - Run tests after code changes: `python test_e2e_pipeline.py`, `python test_edge_cases.py`
   - Verify JS: `node --check electron-ui/renderer/app.js`
   - Compile SCSS: `cd electron-ui && npm run scss`
   - HTML-JS cross-check before commits

3. **Strict Coding Rules**:
   - Syntax check with py_compile
   - Read 50+ lines before editing
   - HTML-JS cross-check for onclick/onchange handlers

The user wants me to follow these instructions. What would they like me to do?

Done. I've read both files. Here's what I understand:

**Project**: MANTAX - NGO Facial Image Analysis System (v0.5.2)
- Backend: Python Flask (port 3000)
- Frontend: Electron + Vanilla JS + SCSS
- ML: ArcFace + FaceNet
- Design: macOS Tahoe Liquid Glass

**Key Rules to Follow**:
1. Verify with tests after code changes (`python test_e2e_pipeline.py`, `python test_edge_cases.py`)
2. JS syntax check: `node --check electron-ui/renderer/app.js`
3. Compile SCSS: `cd electron-ui && npm run scss`
4. HTML-JS cross-check before commits
5. Read 50+ lines context before editing
6. Use `.catch()` for fire-and-forget (non-blocking) API calls

What would you like me to help with?

---

## User

now read all other .md files

---

## Assistant (Build · minimax-m2.5-free · 11.4s)

_Thinking:_

The user wants me to read all other .md files in the project. Let me find them.

**Tool: glob**

**Input:**
```json
{
  "pattern": "**/*.md"
}
```

**Output:**
```
/Users/modernamusmenet/Desktop/MANTAX/master_promt.md
/Users/modernamusmenet/Desktop/MANTAX/system_promt.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/STYLES.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/FACE_ATTRIBUTES_PLAN.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/README.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/START.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/DEVELOPMENT_LOG.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/ARCHITECTURE.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/WORKFLOW_PROMPT.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/SYSTEM_PROMPT.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/SYSTEM_PROMPT_COMPLETE.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/CONTEXT.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/PROJECT_STRUCTURE.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/PROJECT_ARCHITECTURE.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/FRONTEND_SUMMARY.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/PHASE1_COMPLETE.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/IMAGE_STORAGE.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/USAGE.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/INSTALLATION.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/ETHICAL_COMPLIANCE.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/reference_images/README.md
/Users/modernamusmenet/Desktop/MANTAX/ARCHITECTURE_REVIEW.md
/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/_examples/reference_images/README.md
/Users/modernamusmenet/Desktop/MANTAX/rule_set.md
```

---

## Assistant (Build · minimax-m2.5-free · 20.5s)

_Thinking:_

There are many .md files. I've already read master_promt.md and system_promt.md. Let me read all the others in the face_recognition_npo directory and the root files.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/STYLES.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/STYLES.md</path>
<type>file</type>
<content>1: # CSS Classes Reference
2: 
3: **Last Updated**: February 16, 2026
4: 
5: ---
6: 
7: ## SCSS
8: 
9: The styles are written in SCSS for maintainability.
10: 
11: **Source**: `electron-ui/styles/design-system.scss`  
12: **Compiled**: `electron-ui/styles/design-system.css`
13: 
14: **Build**:
15: ```bash
16: cd electron-ui
17: npm run scss
18: ```
19: 
20: Or use `./start.sh` which auto-compiles.
21: 
22: ---
23: 
24: ## macOS Tahoe Liquid Glass Design System
25: 
26: The UI uses macOS Tahoe style with glassmorphism effects.
27: 
28: ### Glass Variables
29: 
30: ```scss
31: $glass-bg: rgba(255, 255, 255, 0.72);
32: $glass-bg-strong: rgba(255, 255, 255, 0.85);
33: $glass-border: rgba(255, 255, 255, 0.5);
34: $glass-shadow: rgba(0, 0, 0, 0.1);
35: $glass-highlight: rgba(255, 255, 255, 0.8);
36: ```
37: 
38: ---
39: 
40: ## Utility Classes
41: 
42: | Class | CSS | Purpose |
43: |-------|-----|---------|
44: | `.hidden` | `max-height: 0; opacity: 0; overflow: hidden` | Hide element with animation |
45: | `.visible` | `max-height: 500px; opacity: 1` | Show element with animation |
46: | `.button-group` | `display: flex; gap: 8px` | Flex container for buttons |
47: | `.button-group-center` | `display: flex; gap: 8px; justify-content: center` | Centered button group |
48: | `.text-center` | `text-align: center` | Center text |
49: | `.text-error` | `color: var(--color-error)` | Error color from variable |
50: | `.text-error-inline` | `color: #cc0000` | Red error text |
51: | `.text-muted` | `color: var(--color-muted)` | Muted text color |
52: | `.webcam-video` | `max-width: 100%; max-height: 400px; border: 1px solid #000; border-radius: 4px` | Webcam video styling |
53: | `.btn-margin-top` | `margin-top: 16px` | Button top margin |
54: | `.empty-state` | `color: #666; padding: 8px` | Empty state styling |
55: 
56: ---
57: 
58: ## Buttons
59: 
60: | Class | Purpose |
61: |-------|---------|
62: | `.btn` | Default button |
63: | `.btn:hover` | Button hover state |
64: | `.btn:disabled` | Disabled button |
65: | `.btn-primary` | Primary button (black background) |
66: | `.btn-primary:hover` | Primary button hover |
67: | `.btn-primary:disabled` | Primary button disabled |
68: | `.btn-success` | Success button (green background) |
69: | `.btn-success:hover` | Success button hover |
70: | `.btn-small` | Small button |
71: 
72: ---
73: 
74: ## Steps & Workflow
75: 
76: | Class | Purpose |
77: |-------|---------|
78: | `.container` | Main container (max-width: 1200px, padding-top: 76px for titlebar) |
79: | `.steps-row` | Flex container for grouping steps |
80: | `.step` | Workflow step card with glassmorphism |
81: | `.step-header` | Step header (flex, space-between) |
82: | `.step-number` | Step number badge (pill-shaped, glass effect) |
83: | `.step-number.webcam` | Blue webcam badge (#5856D6) |
84: | `.step-number.optional` | Optional badge |
85: | `.step-hint` | Step hint text |
86: | `.step-complete` | Completed step (green border + shadow) |
87: | `.step-checkmark` | Checkmark circle for completed steps |
88: 
89: ---
90: 
91: ## Preview & Gallery
92: 
93: | Class | Purpose |
94: |-------|---------|
95: | `.preview-container` | Preview images container (animated) |
96: | `.preview-box` | Single preview box (24px border-radius) |
97: | `.preview-box img` | Preview image |
98: | `.preview-label` | Preview label text |
99: | `.gallery` | Gallery container (flex, wrap) |
100: | `.gallery-item` | Gallery item |
101: | `.gallery-item img` | Gallery image (80x80) |
102: | `.gallery-item span` | Gallery label |
103: 
104: ---
105: 
106: ## Display/Toggle
107: 
108: | Class | Purpose |
109: |-------|---------|
110: | `.reference-details` | Reference details panel |
111: | `.reference-details.active` | Show reference details |
112: | `.comparison-result` | Comparison result (hidden by default) |
113: | `.comparison-result.active` | Show comparison result |
114: 
115: ---
116: 
117: ## Status Messages
118: 
119: | Class | Purpose |
120: |-------|---------|
121: | `.status` | Default status text |
122: | `.status-success` | Success status (green) |
123: | `.status-warning` | Warning status (yellow) |
124: | `.status-error` | Error status (red) |
125: | `.status-info` | Info status (blue) |
126: 
127: ---
128: 
129: ## Badges
130: 
131: | Class | Purpose |
132: |-------|---------|
133: | `.badge` | Default badge |
134: | `.badge-success` | Success badge (green) |
135: | `.badge-warning` | Warning badge (yellow) |
136: | `.badge-error` | Error badge (red) |
137: 
138: ---
139: 
140: ## Toast Notifications
141: 
142: | Class | Purpose |
143: |-------|---------|
144: | `.toast-container` | Fixed container (top: 76px, right: 24px) |
145: | `.toast` | Toast with liquid glass effect |
146: | `.toast.success` | Green tinted toast |
147: | `.toast.error` | Red tinted toast |
148: | `.toast.warning` | Yellow tinted toast |
149: 
150: **Toast Features**:
151: - Liquid glass effect with backdrop blur
152: - Color tint for type indication
153: - 5 second display duration
154: - Smooth fade-out animation
155: 
156: ---
157: 
158: ## Comparison Display
159: 
160: | Class | Purpose |
161: |-------|---------|
162: | `.comparison-result` | Comparison result box |
163: | `.comparison-content` | Flex container for comparison |
164: | `.comparison-side` | Left/right comparison image |
165: | `.comparison-side img` | Comparison image (150x150) |
166: | `.comparison-center` | Center score display |
167: | `.comparison-status` | Match/no-match status badge (liquid glass style) |
168: | `.comparison-status.match` | Match status (green glass) |
169: | `.comparison-status.possible` | Possible match (yellow glass) |
170: | `.comparison-status.no-match` | No match (red glass) |
171: | `.comparison-scores` | Scores container |
172: | `.score-row` | Individual score row |
173: | `.score-row.final` | Final combined score |
174: | `.score-label` | Score label |
175: | `.score-value` | Score value (monospace) |
176: | `.match-reasons` | Match reasons list |
177: 
178: ---
179: 
180: ## Reference List
181: 
182: | Class | Purpose |
183: |-------|---------|
184: | `.reference-list` | Reference list container |
185: | `.reference-item` | Reference item |
186: | `.reference-item img` | Reference thumbnail |
187: | `.reference-item span` | Reference name |
188: | `.ref-remove-btn` | Remove reference button (red circle, top-right) |
189: | `.ref-remove-btn:hover` | Remove button hover |
190: | `.ref-remove-btn:disabled` | Remove button disabled |
191: | `.ref-details-btn` | View details button (blue circle, top-left) |
192: | `.ref-details-btn:hover` | Details button hover |
193: 
194: ---
195: 
196: ## Reference Details
197: 
198: | Class | Purpose |
199: |-------|---------|
200: | `.ref-details-header` | Details header (flex, space-between) |
201: | `.ref-viz-tabs` | Visualization tabs (liquid glass style) |
202: | `.ref-viz-tab` | Visualization tab (liquid glass style) |
203: | `.ref-viz-tab.active` | Active tab |
204: | `.ref-viz-content` | Visualization content area |
205: | `.ref-info-grid` | Info grid (auto-fill) |
206: | `.ref-info-item` | Info item box |
207: | `.ref-info-label` | Info label |
208: | `.ref-info-value` | Info value |
209: 
210: ---
211: 
212: ## Visualization Tabs
213: 
214: | Class | Purpose |
215: |-------|---------|
216: | `.viz-tabs` | Tabs container (horizontal scroll) |
217: | `.viz-tab` | Tab button |
218: | `.viz-tab:hover` | Tab hover |
219: | `.viz-tab.active` | Active tab |
220: | `.viz-content` | Visualization content area |
221: | `.viz-content img` | Visualization image |
222: | `.viz-placeholder` | Placeholder when no content |
223: | `.viz-placeholder p` | Placeholder text |
224: 
225: ---
226: 
227: ## Loading
228: 
229: | Class | Purpose |
230: |-------|---------|
231: | `.loading-overlay` | Full-screen overlay |
232: | `.loading-overlay.active` | Show overlay |
233: | `.loading-content` | Loading content center |
234: | `.loading-spinner` | Spinner animation |
235: | `.loading-text` | Loading text |
236: 
237: ---
238: 
239: ## Terminal
240: 
241: | Class | Purpose |
242: |-------|---------|
243: | `.terminal-footer` | Fixed footer at bottom |
244: | `.terminal-header` | Terminal header |
245: | `.terminal-title` | Terminal title |
246: | `.terminal-toggle` | Toggle button |
247: | `.terminal-log` | Log content area (32px collapsed) |
248: | `.terminal-log.expanded` | Expanded log (140px) |
249: | `.terminal-log-content` | Log content padding |
250: | `.terminal-line` | Log line |
251: | `.terminal-line.command` | Command line (cyan) |
252: | `.terminal-line.success` | Success line (green) |
253: | `.terminal-line.info` | Info line (yellow) |
254: | `.terminal-line.error` | Error line (red) |
255: 
256: ---
257: 
258: ## Data Table
259: 
260: | Class | Purpose |
261: |-------|---------|
262: | `.viz-data-table` | Data table container |
263: | `.viz-data-table table` | Table styling |
264: | `.viz-data-table td.label` | Label column (40%, gray bg) |
265: | `.viz-data-table td.value` | Value column (monospace) |
266: 
267: ---
268: 
269: ## Responsive
270: 
271: | Breakpoint | CSS Variable Changes |
272: |------------|---------------------|
273: | `@media (max-width: 1200px)` | `--sidebar-width: 240px` |
274: | `@media (max-width: 992px)` | Flex direction column for comparison |
275: | `@media (max-width: 768px)` | `--header-height: 50px` |
276: | `@media (max-width: 900px)` | `.steps-row` flex-direction: column |
277: 
278: ---
279: 
280: ## CSS Variables
281: 
282: ### Colors
283: 
284: ```css
285: /* Primary */
286: --color-primary: #007AFF
287: 
288: /* Semantic */
289: --color-success: #30D158
290: --color-warning: #FFD60A
291: --color-error: #FF453A
292: 
293: /* Background */
294: --bg-primary: #F5F5F7
295: --bg-secondary: #f5f5f5
296: --bg-card: #FFFFFF
297: 
298: /* Text */
299: --text-primary: #1D1D1F
300: --text-secondary: #86868B
301: --color-muted: #666
302: 
303: /* Border */
304: --border-light: #E5E5EA
305: --border-default: #D2D2D7
306: --color-border: #000
307: ```
308: 
309: ### Spacing
310: 
311: ```css
312: --space-1: 4px
313: --space-2: 8px
314: --space-3: 12px
315: --space-4: 16px
316: --space-8: 32px
317: ```
318: 
319: ### Typography
320: 
321: ```css
322: /* Size */
323: --text-xs: 11px
324: --text-sm: 12px
325: --text-base: 14px
326: --text-lg: 16px
327: --text-xl: 20px
328: --text-2xl: 24px
329: --text-3xl: 32px
330: --text-4xl: 40px
331: 
332: /* Weight */
333: --font-normal: 400
334: --font-medium: 500
335: --font-semibold: 600
336: --font-bold: 700
337: ```
338: 
339: ---
340: 
341: ## Animations
342: 
343: ### Step Container Transition
344: ```scss
345: .step {
346:   transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
347: }
348: ```
349: 
350: ### Preview Container Animation
351: ```scss
352: .preview-container, .hidden, .visible {
353:   transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
354: }
355: ```
356: 
357: ### Toast Animation
358: - Display: 5 seconds
359: - Fade out: 300ms
360: 
361: ---
362: 
363: *Document created: February 15, 2026*
364: *Last updated: February 16, 2026*
365: *Added: macOS Tahoe glassmorphism, step completion indicators, animated containers*
366: 

(End of file - total 366 lines)
</content>

<system-reminder>
Instructions from: /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/CONTEXT.md
# Code Review Context - Face Recognition NPO

**Review Date**: February 13, 2026  
**Reviewer**: AI Code Review (with Developer Mindset)  
**Status**: ✅ ALL ISSUES FIXED - EYWEAR DETECTION FALSE POSITIVES FIXED

---

## ⚠️ CRITICAL WORKFLOW RULE - ALWAYS FOLLOW THIS

**After EVERY code change, you MUST:**

1. ✅ Run E2E tests: `python test_e2e_pipeline.py`
2. ✅ Run edge case tests: `python test_edge_cases.py`
3. ✅ Run frontend tests: `python test_frontend_integration.py`
4. ✅ Say "FINISHED" ONLY after ALL tests pass

**NEVER say "done" or "finished" until ALL tests pass.**

---

## ⚠️ CRITICAL RULES FOR CODE EDITS

**These rules MUST be followed at all times:**

### Rule 1: Syntax Check Before Submitting
**BEFORE any code edit submission:**
- Count opening and closing parentheses: `(`, `)`, `[`, `]`, `{`, `}`
- Count opening and closing quotes: `'`, `"`
- Ensure each `return` statement has a complete expression before it
- Run `python -m py_compile <file>` to check syntax

### Rule 2: Check for Duplicate Code
**BEFORE any code edit submission:**
- Search for similar code patterns in the file
- Use `grep` to find duplicate function definitions
- Use `grep -n "def " <file>` to list all function definitions
- Look for `elif` chains that might have duplicates
- Verify you're not adding code that already exists elsewhere in the file

### Rule 3: Import Verification
**AFTER any code edit:**
- Run the module to verify imports work
- Check for undefined variables in the edit area
- Verify all referenced functions/methods exist

### Rule 4: Read Before Edit
**BEFORE any `edit` tool use:**
- Read at least 50 lines around the edit location
- Understand the full context of the code block
- Identify all code that might be affected

### Rule 5: Use the Write Tool for Major Changes
**For complex refactoring:**
- Use `read` to understand the full file
- Use `write` to create a complete new version
- Don't try to make multiple complex `edit` calls

### Rule 6: Function Preservation for JavaScript Files
**BEFORE making edits to JavaScript files:**

1. List ALL function definitions:
```bash
grep -n "function " electron-ui/renderer/app.js
```

2. Count functions before edit:
```bash
grep -n "function " electron-ui/renderer/app.js | wc -l
```

3. Verify critical functions exist after edit:
```bash
CRITICAL_FUNCS="handleImageSelect selectImage resetSteps detectFaces extractFeatures compareFaces clearAllCache removeReference showReferenceVisualizations updateReferenceList"
for func in $CRITICAL_FUNCS; do
  if ! grep -q "function $func" electron-ui/renderer/app.js; then
    echo "MISSING: $func"
    exit 1
  fi
done
echo "All critical functions present"
```

**CRITICAL FUNCTIONS (must always exist):**
- `selectImage()` - File input trigger
- `handleImageSelect()` - File processing (MOST CRITICAL - losing this breaks uploads!)
- `resetSteps()` - State reset
- `detectFaces()` - Face detection
- `extractFeatures()` - Embedding extraction
- `compareFaces()` - Comparison
- `updateReferenceList()` - Reference management
- `clearAllCache()` - Cache clearing
- `removeReference()` - Reference removal
- `showReferenceVisualizations()` - Reference viz

### Rule 7: Atomic Edits for Multiple Functions
**When adding 2+ functions in one session:**

1. Make ONE edit per function
2. Verify each edit individually with the function count script
3. Only proceed to next edit after verification passes

```bash
# BEFORE first edit - count functions
echo "Before: $(grep -n 'function ' electron-ui/renderer/app.js | wc -l) functions"

# AFTER each edit - count and compare
echo "After: $(grep -n 'function ' electron-ui/renderer/app.js | wc -l) functions"

# If count doesn't match expected, restore from git:
# git checkout electron-ui/renderer/app.js
```

**WRONG approach:**
```bash
# One massive edit trying to add/change 5 functions → CORRUPTION RISK
```

**RIGHT approach:**
```bash
Edit 1: Add clearAllCache() → Verify → Pass
Edit 2: Add removeReference() → Verify → Pass
Edit 3: Add showReferenceVisualizations() → Verify → Pass
Edit 4: Update updateReferenceList() → Verify → Pass
```

### Rule 8: Fire-and-Forget for Non-Critical API Calls
**When calling non-blocking APIs (like cache clear):**
- Use `.catch()` instead of `await` with try/catch
- Never await non-essential API calls in user interaction handlers
- Blocking the UI thread with awaits can freeze the interface

**WRONG:**
```javascript
reader.onload = async (e) => {
    try {
        await fetch(`${API_BASE}/clear`, { method: 'POST' });  // BLOCKS!
    } catch (err) {
        logToTerminal(`> Warning: ${err.message}`, 'warning');
    }
    // ... rest of upload logic
};
```

**RIGHT:**
```javascript
reader.onload = (e) => {
    fetch(`${API_BASE}/clear`, { method: 'POST' }).catch(err => {
        console.log('Clear failed:', err.message);
    });
    // ... rest of upload logic (executes immediately)
};
```

### Rule 9: Cross-Check HTML with JavaScript (CRITICAL!)

**BEFORE submitting ANY edit to app.js:**

1. Extract ALL function calls from HTML:
```bash
grep -E 'onclick=|onchange=' electron-ui/index.html
```

2. Verify each function exists in app.js:
```bash
echo "=== HTML EVENT HANDLERS ===" && \
grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u

echo "" && echo "=== VERIFYING FUNCTIONS ===" && \
for func in $(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u); do
    if grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js; then
        echo "✓ $func"
    else
        echo "✗ MISSING: $func - ADD IT NOW!"
        exit 1
    fi
done
```

3. Count and compare function definitions:
```bash
echo "Sync functions: $(grep -c '^function ' electron-ui/renderer/app.js)"
echo "Async functions: $(grep -c '^async function ' electron-ui/renderer/app.js)"
echo "Total: $(($(grep -c '^function ') + $(grep -c '^async function ')))"
```

**CRITICAL FUNCTIONS (must always exist):**
```
From HTML onclick/onchange:
  ✓ selectImage()           - Trigger file input
  ✓ handleImageSelect(event) - Process uploaded image
  ✓ detectFaces()           - Face detection
  ✓ extractFeatures()       - Embedding extraction
  ✓ addReference()          - Add reference image
  ✓ handleReferenceSelect(event) - Process reference upload
  ✓ compareFaces()          - Compare embeddings
  ✓ clearAllCache()         - Clear session
  ✓ toggleTerminal()        - Toggle terminal

From app.js internal calls:
  ✓ resetSteps()            - Reset UI state
  ✓ updateReferenceList()   - Render references
  ✓ removeReference()        - Delete reference
  ✓ showReferenceVisualizations() - Show ref viz
  ✓ saveReference()         - Save ref to backend
  ✓ selectReference()       - Select ref for viz
  ✓ showLoading()           - Show loading overlay
  ✓ hideLoading()           - Hide loading overlay
  ✓ showToast()            - Show toast notification
  ✓ logToTerminal()        - Log to terminal
  ✓ clearTerminal()        - Clear terminal
  ✓ initTerminal()         - Initialize terminal
  ✓ toggleTerminal()       - Toggle terminal
  ✓ showVisualization()   - Show viz
  ✓ showVisualizationPlaceholder() - Show placeholder
  ✓ formatDataAsTable()    - Format data table
  ✓ formatKey()            - Format key names
```

**SHELL SCRIPT FOR COMPLETE VERIFICATION:**
```bash
#!/bin/bash
# Complete verification script - RUN BEFORE EVERY COMMIT

echo "========================================="
echo "HTML-JS CROSS-CHECK VERIFICATION"
echo "========================================="

cd /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo

# Extract HTML function calls
echo "1. Extracting HTML event handlers..."
HTML_FUNCS=$(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u)
echo "$HTML_FUNCS"

# Extract JS definitions
echo ""
echo "2. Extracting JS function definitions..."
JS_SYNC=$(grep -c '^function ' electron-ui/renderer/app.js)
JS_ASYNC=$(grep -c '^async function ' electron-ui/renderer/app.js)
echo "Sync: $JS_SYNC, Async: $JS_ASYNC, Total: $((JS_SYNC + JS_ASYNC))"

# Verify each HTML function exists
echo ""
echo "3. Verifying HTML functions exist in JS..."
MISSING=0
for func in $HTML_FUNCS; do
    if ! grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js; then
        echo "✗ MISSING: $func"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -eq 0 ]; then
    echo "✓ All HTML functions verified!"
else
    echo ""
    echo "✗ FAILED: $MISSING missing functions!"
    echo "FIX BEFORE COMMITTING!"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ VERIFICATION PASSED"
echo "========================================="
```

---

## Developer Mindset Checklist

Before submitting ANY code edit:

```
□ 1. Did I count parentheses and brackets?
□ 2. Did I check for duplicate code with grep?
□ 3. Did I verify the code compiles/runs?
□ 4. Did I read the surrounding context (50+ lines)?
□ 5. Did I check that all referenced functions exist?
□ 6. Did I avoid introducing unused variables?
□ 7. Did I handle None/edge cases?
□ 8. Did I use the correct exception handling pattern?
```

If ANY answer is NO → Do not submit the edit until fixed!

---

### MANDATORY PRE-COMMIT VERIFICATION SCRIPT

Run this BEFORE every commit:

```bash
#!/bin/bash
# Complete verification - Run BEFORE commit

cd /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo

echo "========================================="
echo "MANDATORY PRE-COMMIT VERIFICATION"
echo "========================================="

PASS=true

# 1. BACKEND: Python syntax
echo ""
echo "1. Backend Python syntax..."
python -m py_compile api_server.py
if [ $? -eq 0 ]; then
    echo "   ✓ Backend OK"
else
    echo "   ✗ Backend FAILED"
    PASS=false
fi

# 0. PYTHON ENVIRONMENT: Check we're using project venv
echo ""
echo "0. Python environment..."
PYTHON_PATH=$(which python)
if [[ "$PYTHON_PATH" == *"/MANTAX/face_recognition_npo/venv/"* ]]; then
    echo "   ✓ Using project venv: $PYTHON_PATH"
elif [[ "$PYTHON_PATH" == *"/MANTAX/face_recognition_npo/venv313/"* ]]; then
    echo "   ✓ Using project venv313: $PYTHON_PATH"
else
    echo "   ✗ WRONG PYTHON: $PYTHON_PATH"
    echo "   ✗ Must use: .../MANTAX/face_recognition_npo/venv/bin/python"
    echo "   ✗ Run: cd /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo && source venv/bin/activate"
    PASS=false
fi

# 2. BACKEND: API endpoints
echo ""
echo "2. Backend API endpoints..."
ENDPOINTS=$(grep -c "@app.route" api_server.py)
echo "   Found: $ENDPOINTS endpoints"
if [ "$ENDPOINTS" -ge 10 ]; then
    echo "   ✓ Endpoint count OK"
else
    echo "   ✗ Endpoint count too low"
    PASS=false
fi

# 3. BACKEND: Visualization handlers
echo ""
echo "3. Backend visualization handlers..."
VIZ_HANDLERS=$(grep -c "viz_type ==" api_server.py)
echo "   Found: $VIZ_HANDLERS handlers"
if [ "$VIZ_HANDLERS" -ge 10 ]; then
    echo "   ✓ Visualization handlers OK"
else
    echo "   ✗ Visualization handlers too few"
    PASS=false
fi

# 4. FRONTEND: HTML event handlers
echo ""
echo "4. Frontend HTML event handlers..."
HTML_HANDLERS=$(grep -cE 'onclick=|onchange=' electron-ui/index.html)
echo "   Found: $HTML_HANDLERS handlers"
if [ "$HTML_HANDLERS" -ge 5 ]; then
    echo "   ✓ HTML handlers OK"
else
    echo "   ✗ HTML handlers too few"
    PASS=false
fi

# 5. FRONTEND: Function definitions
echo ""
echo "5. Frontend function definitions..."
FUNC_COUNT=$(grep -c '^function ' electron-ui/renderer/app.js)
echo "   Found: $FUNC_COUNT functions"
if [ "$FUNC_COUNT" -ge 10 ]; then
    echo "   ✓ Function count OK"
else
    echo "   ✗ Function count too low"
    PASS=false
fi

# 6. HTML-JS cross-check
echo ""
echo "6. HTML-JS cross-check..."
MISSING=""
for func in $(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u); do
    if ! grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js; then
        MISSING="$MISSING $func"
    fi
done

if [ -z "$MISSING" ]; then
    echo "   ✓ All HTML functions exist in JS"
else
    echo "   ✗ MISSING:$MISSING"
    PASS=false
fi

# 7. E2E Tests
echo ""
echo "7. E2E tests..."
python test_e2e_pipeline.py 2>&1 | grep -q "ALL TESTS PASSED"
if [ $? -eq 0 ]; then
    echo "   ✓ E2E tests passed"
else
    echo "   ✗ E2E tests FAILED"
    PASS=false
fi

echo ""
echo "========================================="
if [ "$PASS" = true ]; then
    echo "✓ ALL VERIFICATIONS PASSED"
    echo "Ready to commit"
    exit 0
else
    echo "✗ VERIFICATION FAILED"
    echo "Fix issues before committing"
    exit 1
fi
```

---

## Executive Summary

A comprehensive code review was conducted on February 11, 2026, followed by a verification review. All issues have been resolved:

| Category | Found | Fixed | Remaining |
|----------|-------|-------|-----------|
| Critical Issues | 7 | 7 | 0 |
| Medium Issues | 6 | 0 | 6 |
| Minor Issues | 6 | 0 | 6 |
| Security Issues | 2 | 1 | 1 |
| Mock/Stub Code | 1 | 1 | 0 |

---

## Critical Issues Fixed (7/7)

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `src/embedding/__init__.py` | Duplicate exception handler (lines 205-210) | Removed dead code |
| 2 | `src/embedding/__init__.py` | Wrong attribute access `self.model.backbone` | Changed to `self.model.backbone` |
| 3 | `utils/webcam.py` | Missing imports: `List, Tuple` | Added to typing imports |
| 4 | `utils/webcam.py` | Missing module imports | Added `FaceDetector`, `FaceNetEmbeddingExtractor`, `SimilarityComparator` |
| 5 | `api_server.py` | Missing method `visualize_embedding()` | Implemented in `FaceNetEmbeddingExtractor` |
| 6 | `api_server.py` | Missing method `visualize_similarity_result()` | Implemented in `FaceNetEmbeddingExtractor` |
| 7 | `visualize_biometric.py` | Hardcoded wrong path | Fixed path to `test_images/kanye_west.jpeg` |

---

## Security Issues Fixed (1/2)

| # | File | Issue | Fix |
|---|------|-------|-----|
| 19 | `gui/facial_analysis_gui.py` | Random embedding fallback on lines 1075, 1118 | Changed to `None` instead of random |

**Remaining Security Issue (#20)**: API lacks input validation for paths/base64 data. This is a future enhancement.

---

## Mock Code Status

### Before Review
```python
def get_activations(self, face_image: np.ndarray) -> Dict[str, np.ndarray]:
    return {}  # STUB - empty dict
```

### After Review
```python
def get_activations(self, face_image: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract neural network activations from 10 layers + embedding."""
    activations = {}
    backbone = self.model.backbone
    # ... real implementation extracting activations from conv1, bn1, relu, maxpool, layer1-4, gap, fc
    activations['embedding'] = self.extract_embedding(face_image)
    return activations
```

**Result**: All 11 activation layers are now real:
- `conv1`: (64, 112, 112)
- `bn1`: (64, 112, 112)
- `relu`: (64, 112, 112)
- `maxpool`: (64, 56, 56)
- `layer1`: (64, 56, 56)
- `layer2`: (128, 28, 28)
- `layer3`: (256, 14, 14)
- `layer4`: (512, 7, 7)
- `gap`: (512, 1, 1)
- `fc`: (512, 1, 1)
- `embedding`: (128,)

---

## Test Results

### E2E Tests (6/6 PASS)
```
✓ PASS: Detection
✓ PASS: Embedding
✓ PASS: Reference Manager
✓ PASS: Same Image Similarity
✓ PASS: Different Images Similarity
✓ PASS: Reference Comparison
```

### Edge Case Tests (11/11 PASS)
```
✓ PASS: Empty/Black Image
✓ PASS: Very Small Images
✓ PASS: None/Invalid Inputs
✓ PASS: Boundary Similarity
✓ PASS: Empty Reference List
✓ PASS: Many References
✓ PASS: Long Reference Names
✓ PASS: Reference Manager Edge Cases
✓ PASS: Visualization Methods
✓ PASS: Quality Metrics
✓ PASS: Compare Embeddings Edge Cases
```

---

## AI WORKFLOW FOR CODE REVIEW

This section documents the established workflow for AI-assisted code review and development.

### Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI CODE REVIEW WORKFLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. SEARCH & DISCOVER                                           │
│     ├── Glob all Python files                                    │
│     ├── Read imports and class definitions                       │
│     ├── Identify patterns (mock, stub, TODO, FIXME)             │
│     └── Check for duplicate exception handlers                   │
│                           │                                      │
│                           ▼                                      │
│  2. TEST & VERIFY                                                │
│     ├── Run existing test suite                                  │
│     ├── Create edge case tests if missing                        │
│     ├── Clear Python cache after edits                           │
│     └── Verify imports work correctly                            │
│                           │                                      │
│                           ▼                                      │
│  3. FIX & DOCUMENT                                               │
│     ├── Fix critical issues first                                │
│     ├── Implement stub methods with real code                    │
│     ├── Update ARCHITECTURE.md with new methods                  │
│     └── Update CONTEXT.md with findings                          │
│                           │                                      │
│                           ▼                                      │
│  4. FINAL VERIFICATION                                           │
│     ├── Run all tests (E2E + Edge Cases)                         │
│     ├── Check for remaining stub/mock code                       │
│     ├── Verify API routes match UI calls                         │
│     └── Confirm imports and instantiations work                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Step 1: Search & Discover Commands

```bash
# Find all Python files
find . -name "*.py" -type f

# Search for mock patterns
grep -rn "return {}" src/ --include="*.py"
grep -rn "Not implemented" src/ --include="*.py" -i
grep -rn "TODO\|FIXME\|stub\|mock" src/ --include="*.py" -i

# Check for duplicate exception handlers
grep -rn "except Exception" src/ --include="*.py" | head -20

# Check for random embeddings (security issue)
grep -rn "np.random.rand.*embed" src/ --include="*.py"

# FIND DUPLICATE CODE - CRITICAL!
grep -rn "def " src/ --include="*.py" | sort | uniq -D
```

### Step 2: Test & Verify Commands

```bash
# Clear Python cache (CRITICAL after edits!)
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Verify syntax compiles
python -m py_compile src/embedding/__init__.py
python -m py_compile api_server.py

# Run E2E tests
python test_e2e_pipeline.py

# Run edge case tests
python test_edge_cases.py

# Run unit tests
python -m pytest tests/

# Verify imports
python -c "from src.detection import *; from src.embedding import *; from src.reference import *; print('All imports OK')"

# Verify API routes
python -c "
from api_server import app
for rule in app.url_map.iter_rules():
    print(f'{rule.methods} {rule.rule}')
"
```

### Step 2: Test & Verify Commands

```bash
# Clear Python cache (CRITICAL after edits!)
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Run E2E tests
python test_e2e_pipeline.py

# Run edge case tests
python test_edge_cases.py

# Run unit tests
python -m pytest tests/

# Verify imports
python -c "from src.detection import FaceDetector; from src.embedding import FaceNetEmbeddingExtractor; print('Imports OK')"

# Verify API routes
python -c "
from api_server import app
for rule in app.url_map.iter_rules():
    print(f'{rule.methods} {rule.rule}')
"

# Verify visualization types match
grep 'data-viz=' electron-ui/index.html | sed 's/.*data-viz=\"\\([^\"]*\\)\".*/\\1/'
grep \"elif viz_type ==\" api_server.py | sed 's/.*== //'
```

### Step 3: Fix & Document Template

For each critical issue found, document in this format:

```markdown
### Issue #N: [Brief Title]

**File**: `path/to/file.py`  
**Lines**: XXX-XXX  
**Severity**: CRITICAL | MEDIUM | MINOR | SECURITY

**Problem**:
```
[Code showing the problem]
```

**Fix**:
```python
[Fixed code]
```

**Impact**:
- [What was broken]
- [How the fix resolves it]
```

### Step 4: Final Verification Checklist

```bash
# Check 1: No stub code remaining
grep -rn "return {}" src/ --include="*.py" | wc -l  # Should be 0

# Check 2: All imports work
python -c "from src.detection import *; from src.embedding import *; from src.reference import *; print('All imports OK')"

# Check 3: All tests pass
python test_e2e_pipeline.py 2>&1 | grep "ALL TESTS PASSED"
python test_edge_cases.py 2>&1 | grep "ALL EDGE CASE TESTS PASSED"

# Check 4: API routes match UI
# Compare: grep 'data-viz=' electron-ui/index.html vs grep "elif viz_type ==" api_server.py

# Check 5: Visualization methods exist
python -c "
from src.embedding import FaceNetEmbeddingExtractor
e = FaceNetEmbeddingExtractor()
methods = ['visualize_embedding', 'visualize_similarity_matrix', 'visualize_similarity_result', 'get_activations']
for m in methods:
    exists = hasattr(e, m)
    print(f'{m}: {\"exists\" if exists else \"MISSING\"}')"
```

---

## Medium Issues (Not Fixed - Future Enhancements)

| # | File | Issue |
|---|------|-------|
| 8 | `api_server.py` | Inconsistent return types in `get_viz_and_data()` |
| 9 | Multiple | Unused variables throughout codebase |
| 10 | Multiple | Missing type hints in several methods |
| 11 | Multiple | Inconsistent exception handling |
| 12 | `config_template.py` | Module-level directory creation at import time |
| 13 | Multiple | Magic numbers throughout code |

---

## Minor Issues (Not Fixed - Nice to Have)

| # | File | Issue |
|---|------|-------|
| 14 | `tests/*.py` | Duplicate/overlapping tests |
| 15 | Multiple | Code duplication (SimilarityComparator, face detection) |
| 16 | Multiple | Logging inconsistency (`print()` vs `logging`) |
| 17 | `setup.py` | References non-existent `requirements.txt` |
| 18 | Multiple | Import organization (some use `sys.path.insert`) |

---

## API Reference (Verified)

### 14 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/detect` | Face detection |
| POST | `/api/extract` | Embedding extraction |
| POST | `/api/add-reference` | Add reference |
| GET | `/api/references` | List references |
| POST | `/api/compare` | Compare embeddings |
| GET | `/api/visualizations/<type>` | Get visualization |
| POST | `/api/clear` | Clear session |
| GET | `/api/status` | Debug server state |

### 14 Visualization Types

| Type | Source | Description |
|------|--------|-------------|
| `detection` | FaceDetector | Bounding boxes |
| `extraction` | FaceDetector | Face ROI |
| `landmarks` | FaceDetector | 15 keypoints |
| `mesh3d` | FaceDetector | 478-point mesh |
| `alignment` | FaceDetector | Pitch/yaw/roll |
| `saliency` | FaceDetector | Attention heatmap |
| `activations` | EmbeddingExtractor | CNN activations |
| `features` | EmbeddingExtractor | Feature maps |
| `multiscale` | FaceDetector | Multi-scale |
| `confidence` | FaceDetector | Quality metrics |
| `embedding` | EmbeddingExtractor | 128-dim bar chart |
| `similarity` | EmbeddingExtractor | Similarity bar |
| `robustness` | EmbeddingExtractor | Noise test |
| `biometric` | FaceDetector | Biometric overview |

---

## COMMON MISTAKES & LESSONS LEARNED

### Mistake 1: Missing Closing Parenthesis

**Problem**: Repeatedly leaving out `)` or `]` in return statements

**Example of ERROR:**
```python
# WRONG - missing closing paren
return (detector.visualize_landmarks(face_image, landmarks), {}
# Also wrong - duplicate code was left behind
return extractor.test_robustness(current_face_image)
elif viz_type == 'biometric':
    return ...
```

**How to prevent:**
```bash
# Count parentheses in the edit area
grep -o '(' <file> | wc -l
grep -o ')' <file> | wc -l
# Must be equal!

# Check for duplicate elif chains
grep -n "elif viz_type ==" <file>
```

**Correct:**
```python
return (detector.visualize_landmarks(face_image, landmarks), {})
```

---

### Mistake 2: Duplicate Code Left Behind

**Problem**: Old code not removed when making edits, creating duplicate function definitions

**How to prevent:**
```bash
# Check for duplicate function definitions
grep -n "def get_viz_and_data" api_server.py

# Check for duplicate elif chains  
grep "elif viz_type ==" api_server.py | wc -l
# Should match number of viz types (14)
```

---

### Mistake 3: Not Running Syntax Check

**Problem**: Submitting code with syntax errors

**How to prevent:**
```bash
# ALWAYS run this before submitting edits!
python -m py_compile <file>

# For Flask apps, also test imports
python -c "import api_server; print('OK')"
```

---

### Mistake 4: Not Reading Context Before Editing

**Problem**: Making edits without understanding the surrounding code

**How to prevent:**
- Always read at least 50 lines around the edit location
- Look for matching parentheses/brackets in the context
- Identify if the code is inside a function, class, or conditional

---

### Mistake 5: Assuming Code Works

**Problem**: Not testing edge cases after edits

**How to prevent:**
- Test with None inputs
- Test with empty lists
- Test with boundary values
- Run the full test suite

### Mistake 6: Blocking UI with Async/Await in Event Handlers
**Problem Encountered (Feb 11, 2026):**
```
Using async/await with fetch() in handleImageSelect() caused the UI to freeze.
The await blocked the onload callback, preventing image display.
Users couldn't upload any images because the upload appeared to do nothing.
```

**Root Cause:**
- `reader.onload = async (e) => { await fetch(...) }` blocked the UI thread
- The fetch to /api/clear was non-essential but was awaited
- UI updates after the await never executed because the fetch never completed

**How to prevent:**
- Use fire-and-forget for non-essential API calls
- Never await in event handlers unless you need the result
- Use `.catch()` for error handling on fire-and-forget calls

**The Fix:**
```javascript
// WRONG - blocks UI
reader.onload = async (e) => {
    await fetch(`${API_BASE}/clear`, { method: 'POST' });  // BLOCKS!
    // This code never executes if fetch hangs
};

// RIGHT - fire-and-forget
reader.onload = (e) => {
    fetch(`${API_BASE}/clear`, { method: 'POST' }).catch(err => {
        console.log('Clear failed:', err.message);
    });
    // This code executes immediately
};
```

### Mistake 7: Not Cross-Checking HTML with JavaScript

**Problem Encountered (Feb 11, 2026):**
```
During cache/clear feature implementation, multiple functions were lost:
- detectFaces() - CRITICAL: No face detection
- extractFeatures() - CRITICAL: No embedding extraction
- Other helper functions from previous refactoring

Result: Buttons did NOTHING when clicked. Complete UI failure.
```

**Root Cause:**
- Added new functions without checking what HTML was calling
- Previous edits corrupted the file by overwriting instead of adding
- No verification that HTML event handlers matched JS function definitions
- Assumed all functions existed when they didn't

**How to prevent:**
1. ALWAYS extract HTML event handlers before making changes:
```bash
grep -E 'onclick=|onchange=' electron-ui/index.html
```

2. ALWAYS verify each HTML function exists in JS:
```bash
# Extract and verify all at once
HTML_FUNCS=$(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u)
for func in $HTML_FUNCS; do
    grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js || echo "MISSING: $func"
done
```

3. ALWAYS count functions before and after edits:
```bash
BEFORE=$(grep -c '^function ' app.js)
AFTER=$(grep -c '^function ' app.js)
[ "$BEFORE" != "$AFTER" ] && echo "FUNCTION COUNT CHANGED!"
```

**The Fix Applied (Feb 11, 2026):**
```bash
# Added missing critical functions:
✓ detectFaces()  - Face detection API call
✓ extractFeatures() - Embedding extraction API call
✓ selectImage() - File input trigger
✓ addReference() - Reference input trigger
✓ handleReferenceSelect() - Process reference upload
✓ resetSteps() - Reset UI state
✓ saveReference() - Save reference to backend
✓ removeReference() - Remove reference
✓ showReferenceVisualizations() - Show ref viz
✓ compareFaces() - Compare embeddings
✓ And 15+ helper functions...

# Total functions: 25 (17 sync + 8 async)
```

**MANDATORY PRE-COMMIT CHECK:**
```bash
#!/bin/bash
# Run this before EVERY commit

# 1. Extract HTML handlers
echo "HTML handlers:"
grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u

# 2. Verify each exists
MISSING=""
for func in $(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u); do
    if ! grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js; then
        MISSING="$MISSING $func"
    fi
done

if [ -n "$MISSING" ]; then
    echo "MISSING FUNCTIONS:$MISSING"
    echo "FIX BEFORE COMMITTING!"
    exit 1
fi

echo "✓ All HTML handlers verified in app.js"
```

---

## Files Modified During Code Review

| Date | File | Change |
|------|------|--------|
| Feb 11, 2026 | `CONTEXT.md` | Added strict edit rules and developer checklist |
| Feb 11, 2026 | `CONTEXT.md` | Added common mistakes section |
| Feb 11, 2026 | `CONTEXT.md` | Created with complete code review findings |
| Feb 11, 2026 | `CONTEXT.md` | Added edge case testing section |
| Feb 11, 2026 | `CONTEXT.md` | Added AI workflow for code review |
| Feb 11, 2026 | `DEVELOPMENT_LOG.md` | Updated with review summary |
| Feb 11, 2026 | `ARCHITECTURE.md` | Updated with all methods and viz types |
| Feb 11, 2026 | `src/embedding/__init__.py` | Removed duplicate exception handler |
| Feb 11, 2026 | `src/embedding/__init__.py` | Fixed wrong attribute access |
| Feb 11, 2026 | `src/embedding/__init__.py` | Implemented get_activations() with real activations |
| Feb 11, 2026 | `src/embedding/__init__.py` | Added visualize_embedding(), visualize_similarity_matrix(), visualize_similarity_result() |
| Feb 11, 2026 | `src/detection/__init__.py` | Added None check to visualize_3d_mesh() |
| Feb 11, 2026 | `CONTEXT.md` | Added Rules 6-8 (Function Preservation, Atomic Edits, Exact Matching) |
| Feb 11, 2026 | `CONTEXT.md` | Added Mistake 6 (Blocking UI with Async/Await) |
| Feb 11, 2026 | `electron-ui/renderer/app.js` | Fixed: handleImageSelect() was lost during edits, restored complete function |
| Feb 11, 2026 | `electron-ui/index.html` | Fixed font stack for macOS compatibility |
| Feb 11, 2026 | `api_server.py` | Added DELETE /api/references/<id> endpoint |
| Feb 11, 2026 | `CONTEXT.md` | Added Rule 9 (HTML-JS Cross-Check) - MANDATORY |
| Feb 11, 2026 | `CONTEXT.md` | Added Mistake 7 (Missing HTML-JS Cross-Check) |
| Feb 11, 2026 | `electron-ui/renderer/app.js` | Added 10 missing functions: detectFaces, extractFeatures, selectImage, addReference, handleReferenceSelect, saveReference, resetSteps, removeReference, showReferenceVisualizations, compareFaces (FULL RESTORE) |
| Feb 11, 2026 | `electron-ui/renderer/app.js` | Total functions: 25 (17 sync + 8 async) |

---

*Context document created: February 11, 2026*
*Last updated: February 11, 2026*
*All critical issues fixed, no mock code remaining*
*Strict edit rules enforced*
*MANDATORY: Run HTML-JS cross-check before every commit*
| Feb 11, 2026 | `src/detection/__init__.py` | Added validation to compute_quality_metrics() |
| Feb 11, 2026 | `utils/webcam.py` | Added missing imports |
| Feb 11, 2026 | `visualize_biometric.py` | Fixed hardcoded path |
| Feb 11, 2026 | `gui/facial_analysis_gui.py` | Fixed random embedding fallback |
| Feb 11, 2026 | `test_edge_cases.py` | Created comprehensive edge case test suite |
| Feb 11, 2026 | `api_server.py` | Added reference visualization endpoints (`/api/visualizations/<type>/reference/<id>`) |
| Feb 11, 2026 | `CONTEXT.md` | Added Rules 6-8 (Function Preservation, Atomic Edits, Exact Matching) |
| Feb 11, 2026 | `CONTEXT.md` | Added Mistake 6 (Blocking UI with Async/Await) |
| Feb 11, 2026 | `electron-ui/renderer/app.js` | Fixed: handleImageSelect() was lost during edits, restored complete function |
| Feb 11, 2026 | `electron-ui/index.html` | Fixed font stack for macOS compatibility |
| Feb 11, 2026 | `api_server.py` | Added DELETE /api/references/<id> endpoint |
| Feb 11, 2026 | `CONTEXT.md` | Added Rule 9 (HTML-JS Cross-Check) - MANDATORY |
| Feb 11, 2026 | `CONTEXT.md` | Added Mistake 7 (Missing HTML-JS Cross-Check) |
| Feb 11, 2026 | `electron-ui/renderer/app.js` | Added 10+ missing functions: detectFaces, extractFeatures, selectImage, addReference, handleReferenceSelect, saveReference, resetSteps, removeReference, showReferenceVisualizations, compareFaces, setupEventListeners, checkAPI (FULL RESTORE) |
| Feb 11, 2026 | `electron-ui/renderer/app.js` | Total: 18 sync + 9 async functions (27 total) |
| Feb 11, 2026 | `CONTEXT.md` | Added mandatory pre-commit verification script |
| Feb 11, 2026 | `src/embedding/__init__.py` | Fixed visualize_activations() and visualize_feature_maps() |
| Feb 11, 2026 | `electron-ui/renderer/app.js` | Added console logging, better error messages |
| Feb 11, 2026 | `test_e2e_pipeline.py` | Made test images configurable via env vars |
| Feb 11, 2026 | `visualize_biometric.py` | Made test image configurable |
| Feb 11, 2026 | `reference_images/README.md` | Updated documentation |
| Feb 13, 2026 | `api_server.py` | Fixed 9 test viz handlers to return actual data (test-health, test-detection, test-extraction, test-reference, test-multi, test-pose, test-eyewear, test-viz, test-clear) |
| Feb 13, 2026 | `electron-ui/renderer/app.js` | Fixed visualization fetch to accept smaller images, added data-only display support |
| Feb 13, 2026 | `CONTEXT.md` | Added Lesson 15 (Test Viz Data) and Mistake 11 |
| Feb 13, 2026 | `src/detection/__init__.py` | Fixed eyewear detection - use brightness as primary, eye cascade as secondary |
| Feb 13, 2026 | `electron-ui/renderer/app.js` | Added webcam functions (startWebcam, captureWebcam, stopWebcam) |
| Feb 13, 2026 | `test_eyewear_frontend.js` | Created frontend integration test for eyewear detection |

---

## NEW: Lessons Learned - Persistence & Process Management (Feb 11, 2026)

### Lesson 10: References Not Persisting to JSON

**Problem**: References were stored in memory but not saved to `embeddings.json`. When API server restarted, references were lost.

**Solution**: Added `save_references()` and `load_references()` functions to `api_server.py`:
```python
REFERENCES_FILE = os.path.join(os.path.dirname(__file__), 'reference_images', 'embeddings.json')

def save_references():
    """Save references to JSON file."""
    data = {'metadata': [...], 'embeddings': [...]}
    with open(REFERENCES_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def load_references():
    """Load references from JSON file on startup."""
    global references
    if os.path.exists(REFERENCES_FILE) and os.path.getsize(REFERENCES_FILE) > 0:
        with open(REFERENCES_FILE, 'r') as f:
            references = json.load(f).get('references', [])

load_references()  # Call at module load
```

**Remember**: Call `save_references()` after any mutation:
- After `add_reference()` succeeds
- After `remove_reference()` succeeds

### Lesson 11: Electron App Not Loading Existing References

**Problem**: App started with empty `references = []`, never loading from API.

**Solution**: Added `loadReferences()` function called from `checkAPI()`:
```javascript
async function checkAPI() {
    const response = await fetch(`${API_BASE}/health`);
    const data = await response.json();
    if (data.status === 'ok') {
        logToTerminal('> API connected', 'success');
        loadReferences();  // NEW - load existing refs
    }
}

async function loadReferences() {
    const response = await fetch(`${API_BASE}/references`);
    const data = await response.json();
    if (data.references) {
        references = data.references;
        updateReferenceList();
    }
}
```

### Lesson 12: Old Python Processes Cache Code

**Problem**: After editing `api_server.py`, changes didn't take effect. Old process was still running.

**Root Cause**: Python caches `.pyc` files and the old process didn't restart.

**Solution**: Always restart the API server after code changes:
```bash
# Kill existing
pkill -f "python api_server.py"

# Clear cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Restart
python api_server.py
```

**Or use `start.sh`** which does all this automatically.

### Lesson 13: Test Images Had Wrong Default Paths

**Problem**: `test_e2e_pipeline.py` defaulted to `kanye_west.jpeg` which doesn't exist.

**Solution**: Updated defaults to actual files:
```python
TEST_IMAGE = os.environ.get('TEST_IMAGE', 'test_subject.jpg')
TEST_IMAGE_REF = os.environ.get('TEST_IMAGE_REF', 'reference_subject.jpg')
```

**Files that exist**:
- `test_images/test_subject.jpg`
- `test_images/reference_subject.jpg`

### Lesson 14: Use start.sh for Clean Startup

**Problem**: Manual server starts leave old processes running.

**Solution**: Use `start.sh`:
```bash
#!/bin/bash
# Kill any existing API server
pkill -f "python api_server.py" 2>/dev/null

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name ".pytest_cache" -exec rm -rf {} +

# Start API server
cd "$(dirname "$0")"
python api_server.py &

# Start Electron UI
cd "$(dirname "$0")/electron-ui"
npm start &
```

### Lesson 15: Test Visualization Data Not Displaying

**Problem**: Test tabs (test-health, test-detection, etc.) showed images but no text data below them.

**Root Cause**: API handlers returned empty `{}` as second tuple element:
```python
# WRONG - returns empty dict
return visualize_test_detail("Health Check", {"status": "OK"}), {}
```

**Solution**: Return actual data dict as second tuple element:
```python
# CORRECT - returns actual data
data = {"status": "OK", "api": "running", "port": 3000}
return visualize_test_detail("Health Check", data), data
```

**Frontend**: Already had `formatDataAsTable()` and CSS - just needed data from API.

**Frontend Logic Fix**: Also updated to handle smaller images and data-only display:
```javascript
// Changed from: data.visualization.length > 100
// To: accept any visualization size
if (data.success && data.visualization) {
    visualizationData[vizType] = data.visualization;
    // ...store data
}
```

**Remember**: When adding new test viz handlers, always return data as second tuple element!

### Lesson 16: Always Use Project Virtual Environment

**Problem**: Old API server process kept running with wrong Python, causing bugs to persist even after code fixes.

**Root Cause**: 
- Started API with system Python (`/opt/homebrew/Cellar/python@3.14/...`) instead of project venv
- Used `python` command instead of `source venv/bin/activate && python`
- Old process cached old code, so edits had no effect

**Solution**: Always use the project's virtual environment:
```bash
# CORRECT - use project venv
cd /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo
source venv/bin/activate
python api_server.py

# OR use start.sh which does this automatically
./start.sh
```

**How to verify**: Check which Python is running:
```bash
# Find the process
ps aux | grep "api_server\|python.*3000" | grep -v grep

# Kill old process if needed
kill <PID>

# Verify new process uses correct Python
lsof -p <new_pid> | grep Python
# Should show: .../MANTAX/face_recognition_npo/venv/bin/python
```

**REMEMBER**: Always restart the API server after making code changes!

### Lesson 17: Eyewear Detection False Positives

**Problem**: Every uploaded image showed "glasses detected" even without eyewear.

**Root Cause**: 
1. OpenCV eye cascade is unreliable - often returns 0 eyes even without glasses
2. Code automatically flagged eyewear if eye_count == 0, regardless of actual eye brightness
3. Thresholds were too low (0.4 for brightness ratio, 0.15 for edge density)

**Solution**: 
1. Analyze brightness/edge FIRST as primary detection method
2. Use eye cascade ONLY as confirmation, not primary
3. Require BOTH conditions to be true (dark eyes + high edge density)
4. Set stricter thresholds: brightness < 0.2 for sunglasses, < 0.35 for possible
5. Only flag eyewear if brightness analysis confirms, not just eye cascade failure

**Key Code Pattern**:
```python
# WRONG - eye cascade alone triggers detection
if eye_count == 0:
    eyewear_detected = True  # FALSE POSITIVE!

# CORRECT - require brightness confirmation
eyes_detected = self.detect_eyes(face_image)
eye_count = len(eyes_detected)

if eye_count == 0 and not eyewear_detected:
    # Only flag if brightness is VERY dark
    if brightness_ratio < 0.15:
        eyewear_detected = True
    else:
        # Eye cascade likely just failed - don't flag
        eye_count = 2  # Assume normal
```

**Testing**: Always test BOTH:
- Normal face (should NOT detect eyewear)
- Face with sunglasses (should detect eyewear)

**Frontend Test Script**: `test_eyewear_frontend.js`

---

## Common Mistakes Summary

| # | Mistake | Solution |
|---|---------|----------|
| 1 | Missing closing paren | Count `(`, `)`, `[`, `]` |
| 2 | Duplicate code left behind | `grep -n "def "` before/after |
| 3 | Not running syntax check | `python -m py_compile <file>` |
| 4 | Not reading context | Read 50+ lines before edit |
| 5 | Not testing edge cases | Run tests after every edit |
| 6 | Blocking UI with async/await | Use `.catch()` for fire-and-forget |
| 7 | Missing HTML-JS cross-check | Verify all onclick/onchange handlers |
| 8 | References not persisting | Call `save_references()` after add/remove |
| 9 | Old process caching code | Use `./start.sh` to restart |
| 10 | Wrong test image paths | Use `test_subject.jpg`, `reference_subject.jpg` |
| 11 | Test viz returning empty data | Return actual data dict as second tuple element |
| 12 | Using wrong Python environment | Always use `source venv/bin/activate` before running |
| 13 | Eyewear detection false positives | Use brightness analysis as primary, eye cascade as secondary |

---

## New Features: Dual-Model Comparison (February 13, 2026)

### Overview
The application now uses BOTH ArcFace AND FaceNet together for more robust matching.

### Dual-Model Architecture

| Component | Description |
|-----------|-------------|
| ArcFace Extractor | 512-dim embeddings (primary) |
| FaceNet Extractor | 128-dim embeddings (secondary) + activations |
| Both always loaded | Even if USE_ARCFACE=true |

### Comparison Weights

| Factor | Weight | Purpose |
|--------|--------|---------|
| ArcFace similarity | 60% | Primary - best discrimination |
| FaceNet similarity | 20% | Additional signal |
| Landmark geometry | 15% | Structural consistency |
| Activation similarity | 5% | Neural network internal representations |
| Image quality | 5% | Reliability factor |

**Note**: No pose penalty - same person with different poses will match correctly.

### New API Fields

Each comparison result now returns:
```json
{
  "arcface_similarity": 0.75,
  "facenet_similarity": 0.60,
  "landmark_similarity": 0.85,
  "activation_similarity": 0.72,
  "final_score": 0.68,
  "status": "match|possible|no_match",
  "match_label": "Full Match|Possible Match|No Match",
  "reasons": ["ArcFace: High similarity (75%)", "FaceNet: Moderate (60%)", "Similar neural activation patterns"]
}
```

### Backwards Compatibility

- Old references with single embedding are automatically handled
- If ArcFace unavailable, falls back to FaceNet-only matching
- `/api/diagnostics` endpoint added to check system status

---

## MediaPipe Landmark Detection (February 13, 2026)

### Issue Resolved
Previously, landmarks visualization showed only ~10 points (fallback mode). Now shows full 468 MediaPipe landmarks.

### Required Files
- `face_landmark.task` - MediaPipe model (downloaded to project root)
- MediaPipe Python package installed

### Verification
Check `/api/diagnostics` endpoint:
```json
{
  "mediapipe": "available",
  "model_file_exists": true,
  "arcface_extractor": "loaded",
  "dual_model_mode": true
}
```

---

## Test Tab Renaming (February 13, 2026)

| Old Name | New Name |
|----------|----------|
| Test 1 | API Health |
| Test 2 | Detection |
| Test 3 | Extraction |
| Test 4 | References |
| Test 5 | Multi-Match |
| Test 6 | Pose |
| Test 7 | Eyewear |
| Test 8 | Viz Types |
| Test 9 | Session |

---

## Files to Know

| File | Purpose | When to Edit |
|------|---------|--------------|
| `api_server.py` | Flask API (12+ endpoints) | Backend changes |
| `electron-ui/renderer/app.js` | Frontend logic | UI changes |
| `electron-ui/index.html` | UI structure | HTML changes |
| `src/embedding/__init__.py` | SimilarityComparator | Matching logic |
| `src/detection/__init__.py` | FaceDetector | Landmarks, pose |
| `start.sh` | Startup script | Server config |
| `test_e2e_pipeline.py` | E2E tests | Test changes |
| `reference_images/embeddings.json` | Persistent storage | Auto-generated |

---

## Quick Verification Checklist

Before telling user "done" or "ready to commit":

```bash
# 1. API syntax
python -m py_compile api_server.py && echo "✓ API OK"

# 2. Tests pass
python test_e2e_pipeline.py 2>&1 | grep -q "ALL TESTS PASSED" && echo "✓ Tests OK"

# 3. Cache cleared
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "✓ Cache cleared"

# 4. Server restarted (if needed)
# pkill -f "python api_server.py" 2>/dev/null
# python api_server.py &
```

---

*Context document created: February 11, 2026*
*Last updated: February 13, 2026*
*All critical issues fixed*
*Dual-model comparison implemented*
*MediaPipe landmarks fixed*
*MANDATORY: Run pre-commit verification before every commit*

</system-reminder>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/FACE_ATTRIBUTES_PLAN.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/FACE_ATTRIBUTES_PLAN.md</path>
<type>file</type>
<content>1: # Face Attributes Detection - Implementation Plan
2: 
3: **Created**: February 16, 2026  
4: **Status**: Ready for Implementation
5: 
6: ---
7: 
8: ## Overview
9: 
10: Add facial expression detection, age estimation, and gender detection to the face recognition system.
11: 
12: ---
13: 
14: ## What MediaPipe Provides (Built-in)
15: 
16: | Feature | Available | Implementation |
17: |---------|-----------|----------------|
18: | **Face Expressions** | ✅ Yes | 52 blendshapes → simple expressions |
19: | **Face Landmarks** | ✅ Yes | 468 points - already implemented |
20: | **Pose (yaw/pitch/roll)** | ✅ Yes | already implemented |
21: | **Age** | ❌ No | Requires external model |
22: | **Gender** | ❌ No | Requires external model |
23: 
24: ---
25: 
26: ## Implementation Options
27: 
28: ### Option A: Expressions Only (Safe)
29: - Enable MediaPipe blendshapes
30: - Map 52 blendshapes → simple expressions
31: - Display in UI
32: 
33: ### Option B: Expressions + Age/Gender (Full)
34: - Everything in Option A
35: - Download Caffe models for age/gender
36: - Add detection using OpenCV DNN
37: 
38: **Recommendation**: Option B - All features visible to user
39: 
40: ---
41: 
42: ## Complete Plan - Option B
43: 
44: ### Phase 1: Enable MediaPipe Blendshapes
45: 
46: #### 1.1 Edit `src/detection/__init__.py`
47: 
48: **Line 45**: Enable blendshapes
49: ```python
50: options = vision.FaceLandmarkerOptions(
51:     base_options=base_options,
52:     running_mode=vision.RunningMode.IMAGE,
53:     num_faces=1,
54:     output_face_blendshapes=True,  # CHANGE: False → True
55:     output_facial_transformation_matrixes=True
56: )
57: ```
58: 
59: #### 1.2 Add expression parsing method
60: 
61: Add new method to FaceDetector class:
62: ```python
63: def parse_expressions(self, blendshapes) -> Dict[str, any]:
64:     """
65:     Parse 52 blendshapes into simple expressions.
66:     
67:     Returns:
68:         {
69:             'expressions': {
70:                 'smile': bool,
71:                 'eye_blink_left': bool,
72:                 'eye_blink_right': bool,
73:                 'mouth_open': bool,
74:                 'eyebrow_raise': bool,
75:                 'surprise': bool,
76:                 'anger': bool,
77:                 'cheek_squint': bool,
78:             },
79:             'scores': {
80:                 'smile': 0.0-1.0,
81:                 'eye_blink_left': 0.0-1.0,
82:                 ...
83:             }
84:         }
85:     """
86: ```
87: 
88: **Expression Mapping**:
89: | Expression | Condition |
90: |------------|-----------|
91: | smile | mouthSmile > 0.5 |
92: | eye_blink_left | eyeBlinkLeft > 0.5 |
93: | eye_blink_right | eyeBlinkRight > 0.5 |
94: | mouth_open | jawOpen > 0.3 OR mouthStretch > 0.3 |
95: | eyebrow_raise | browInnerUp > 0.4 OR browOuterUpLeft > 0.4 |
96: | surprise | browOuterUpLeft + browOuterUpRight + jawOpen > 1.0 |
97: | anger | browDownLeft + browDownRight + jawOpen > 1.2 |
98: | cheek_squint | cheekSquintLeft + cheekSquintRight > 0.5 |
99: 
100: ---
101: 
102: ### Phase 2: Add Age/Gender Models
103: 
104: #### 2.1 Download Caffe Models
105: 
106: Required files:
107: ```
108: Models to download:
109: - deploy_age.prototxt
110: - age_net.caffemodel
111: - deploy_gender.prototxt  
112: - gender_net.caffemodel
113: ```
114: 
115: Source: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
116: 
117: #### 2.2 Add age/gender methods to FaceDetector
118: 
119: Add to `src/detection/__init__.py`:
120: 
121: ```python
122: def __init__(self, ...):
123:     # Existing code...
124:     
125:     # Age/Gender models (lazy load)
126:     self._age_model = None
127:     self._gender_model = None
128:     self._age_net = None
129:     self._gender_net = None
130:     
131:     # Age ranges for classification
132:     self._age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', 
133:                       '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
134: 
135: def _load_age_model(self):
136:     """Lazy load age detection model."""
137:     if self._age_net is None:
138:         try:
139:             self._age_net = cv2.dnn.readNetFromCaffe(
140:                 "deploy_age.prototxt",
141:                 "age_net.caffemodel"
142:             )
143:         except:
144:             print("Age model not available")
145: 
146: def _load_gender_model(self):
147:     """Lazy load gender detection model."""
148:     if self._gender_net is None:
149:         try:
150:             self._gender_net = cv2.dnn.readNetFromCaffe(
151:                 "deploy_gender.prototxt",
152:                 "gender_net.caffemodel"
153:             )
154:         except:
155:             print("Gender model not available")
156: 
157: def estimate_age(self, face_image: np.ndarray) -> Dict[str, any]:
158:     """
159:     Estimate age from face image using Caffe model.
160:     
161:     Returns:
162:         {
163:             'age': int,  # Estimated age
164:             'age_range': str,  # e.g., "(25, 32)"
165:             'confidence': float  # 0.0-1.0
166:         }
167:     """
168:     # Preprocess
169:     blob = cv2.dnn.blobFromImage(
170:         cv2.resize(face_image, (227, 227)),
171:         1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746)
172:     )
173:     
174:     self._age_net.setInput(blob)
175:     predictions = self._age_net.forward()
176:     
177:     age_index = np.argmax(predictions[0])
178:     age_range = self._age_list[age_index]
179:     confidence = float(predictions[0][age_index])
180:     
181:     # Extract midpoint of range
182:     age = int(age_range.strip('()').split(',')[0]) + 5
183:     
184:     return {
185:         'age': age,
186:         'age_range': age_range,
187:         'confidence': confidence
188:     }
189: 
190: def estimate_gender(self, face_image: np.ndarray) -> Dict[str, any]:
191:     """
192:     Estimate gender from face image using Caffe model.
193:     
194:     Returns:
195:         {
196:             'gender': 'male' | 'female',
197:             'confidence': float  # 0.0-1.0
198:         }
199:     """
200:     # Preprocess
201:     blob = cv2.dnn.blobFromImage(
202:         cv2.resize(face_image, (227, 227)),
203:         1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746)
204:     )
205:     
206:     self._gender_net.setInput(blob)
207:     predictions = self._gender_net.forward()
208:     
209:     gender_index = np.argmax(predictions[0])
210:     gender = 'male' if gender_index == 0 else 'female'
211:     confidence = float(predictions[0][gender_index])
212:     
213:     return {
214:         'gender': gender,
215:         'confidence': confidence
216:     }
217: ```
218: 
219: ---
220: 
221: ### Phase 3: API Endpoints
222: 
223: #### 3.1 Add face-attributes endpoint
224: 
225: In `api_server.py`:
226: 
227: ```python
228: @app.route('/api/face-attributes', methods=['GET'])
229: def get_face_attributes():
230:     """Get facial attributes: expressions, age, gender for current face."""
231:     global current_image, current_faces
232:     
233:     try:
234:         if not current_faces:
235:             return jsonify({'success': False, 'error': 'No faces detected'})
236:         
237:         face_box = current_faces[0]
238:         x, y, w, h = face_box
239:         face_roi = current_image[y:y+h, x:x+w]
240:         
241:         attributes = {}
242:         
243:         # Expressions from blendshapes (if available)
244:         try:
245:             expressions = detector.parse_expressions(face_blendshapes)
246:             attributes['expressions'] = expressions
247:         except:
248:             attributes['expressions'] = None
249:         
250:         # Age estimation
251:         try:
252:             age = detector.estimate_age(face_roi)
253:             attributes['age'] = age
254:         except:
255:             attributes['age'] = None
256:         
257:         # Gender estimation
258:         try:
259:             gender = detector.estimate_gender(face_roi)
260:             attributes['gender'] = gender
261:         except:
262:             attributes['gender'] = None
263:         
264:         return jsonify({
265:             'success': True,
266:             'attributes': attributes
267:         })
268:     except Exception as e:
269:         return jsonify({'success': False, 'error': str(e)})
270: ```
271: 
272: #### 3.2 Add visualization endpoint
273: 
274: ```python
275: @app.route('/api/visualizations/attributes', methods=['GET'])
276: def get_attributes_visualization():
277:     """Get visualization of face attributes."""
278:     # Returns bar chart of expression scores
279: ```
280: 
281: #### 3.3 Update /api/detect response
282: 
283: Update `detect_faces()` to include `expressions` in each face object.
284: 
285: ---
286: 
287: ### Phase 4: Frontend
288: 
289: #### 4.1 HTML - Face Gallery
290: 
291: Display in faces gallery:
292: ```
293: [Face Image]
294: 😊 Smile: Yes    👁️ Blink: No    😮 Surprise: No
295: 👤 Age: 28-32    Gender: Male (95%)
296: ```
297: 
298: #### 4.2 JavaScript
299: 
300: - Update `detectFaces()` to parse and display attributes
301: - Add attributes visualization tab
302: 
303: ---
304: 
305: ### Phase 5: Testing
306: 
307: Run all tests to ensure nothing breaks:
308: 
309: ```bash
310: python test_e2e_pipeline.py
311: python test_edge_cases.py
312: python test_frontend_integration.py
313: ```
314: 
315: ---
316: 
317: ## Risk Assessment
318: 
319: | Risk | Level | Mitigation |
320: |------|-------|------------|
321: | Breaking existing tests | LOW | Wrap age/gender in try/except, fallback if models missing |
322: | Model download | MEDIUM | Provide download URLs or skip if not available |
323: | Performance | LOW | Models are lightweight Caffe |
324: | Breaking UI | LOW | Add new content, don't modify existing |
325: 
326: ---
327: 
328: ## Files to Modify
329: 
330: 1. `src/dection/__init__.py` - Enable blendshapes, add expression/age/gender methods
331: 2. `api_server.py` - Add endpoints
332: 3. `electron-ui/index.html` - Display attributes
333: 4. `electron-ui/renderer/app.js` - Fetch and display
334: 
335: ---
336: 
337: ## Test Checklist
338: 
339: - [ ] E2E tests pass
340: - [ ] Edge case tests pass  
341: - [ ] Frontend integration tests pass
342: - [ ] Age estimation works (if model available)
343: - [ ] Gender estimation works (if model available)
344: - [ ] Expressions display in UI
345: 
346: ---
347: 
348: *Plan created: February 16, 2026*
349: 

(End of file - total 349 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/README.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/README.md</path>
<type>file</type>
<content>1: # NGO Facial Image Analysis System
2: 
3: **Version**: 0.5.1  
4: **Last Updated**: February 16, 2026  
5: **Status**: ✅ Fully Functional - macOS Tahoe UI Redesign
6: 
7: A Python-based facial image analysis system with Electron desktop UI for ethical, consent-based NGO use in documentation verification and investigative work.
8: 
9: ---
10: 
11: ## ⚠️ CRITICAL WORKFLOW RULE
12: 
13: After EVERY code change:
14: 1. Run: `python test_e2e_pipeline.py`
15: 2. Run: `python test_edge_cases.py`
16: 3. Run: `python test_frontend_integration.py`
17: 4. Say "FINISHED" ONLY after ALL tests pass
18: 
19: ---
20: 
21: ## Quick Start
22: 
23: ### Method 1: Interactive Menu (Recommended)
24: ```bash
25: cd face_recognition_npo
26: ./start.sh
27: ```
28: 
29: This starts the Flask API server and lets you choose how to open:
30: - [1] Electron Desktop App
31: - [2] Browser
32: - [3] Both
33: 
34: ### Method 2: Manual Startup
35: 
36: **Terminal 1: Start Flask API**
37: ```bash
38: cd face_recognition_npo
39: source venv/bin/activate
40: python api_server.py
41: ```
42: 
43: **Terminal 2: Start Electron**
44: ```bash
45: cd face_recognition_npo/electron-ui
46: npm start
47: ```
48: 
49: > **Note**: Electron will connect to the existing Flask server on port 3000.
50: 
51: ### Method 3: Browser Only
52: ```bash
53: cd face_recognition_npo
54: source venv/bin/activate
55: python api_server.py
56: # Open http://localhost:3000 in your browser
57: ```
58: 
59: ---
60: 
61: ## Architecture
62: 
63: ```
64: start.sh ──► Flask API (port 3000)
65:                  │
66:                  ├── Browser ──► http://localhost:3000
67:                  │
68:                  └── Electron ──► Connects to Flask (no Python spawn)
69: ```
70: 
71: **Best Practice**: Flask runs once, Electron connects to it.
72: 
73: ---
74: 
75: ## Features
76: 
77: - ✅ **Face Detection**: OpenCV DNN with Caffe model
78: - ✅ **Embedding Extraction**: 512-dimensional (ArcFace) or 128-dim (FaceNet)
79: - ✅ **Similarity Comparison**: Cosine similarity with confidence bands
80: - ✅ **Reference Management**: Store references with real embeddings
81: - ✅ **Persistent Storage**: References saved to `reference_images/embeddings.json`
82: - ✅ **14 AI Visualizations**: Detection, landmarks, mesh, activations, etc.
83: - ✅ **Electron Desktop UI**: Ultra minimal design with MANTAX branding
84: - ✅ **Flask API Server**: 14+ REST endpoints
85: - ✅ **End-to-End Tests**: All 6/6 passing
86: - ✅ **ArcFace Integration**: 512-dim embeddings for better discrimination
87: - ✅ **Activation Similarity**: Neural network activation comparison for matching
88: - ✅ **Reference Details Panel**: View reference embeddings, landmarks, pose, quality
89: - ✅ **Smart Compare Button**: Only enables when both embedding AND references exist
90: - ✅ **Card Layout**: Comparison results displayed in centered card
91: 
92: ---
93: 
94: ## Usage Workflow
95: 
96: ```
97: Step 1: Choose Photo     → Upload image
98: Step 2: Find Faces       → Click "Find Faces"
99: Step 3: Create Signature → Click "Create Signature" (EXTRACTS EMBEDDING)
100: Step 4: Add Reference    → Upload reference image
101: Step 5: Compare          → Click "Compare"
102: ```
103: 
104: **For Visualizations**: After Step 3, click the visualization tabs (Embedding, Activations, Features, etc.)
105: 
106: **For Reference Details**: Click on a reference image to view its embeddings, landmarks, pose, and quality metrics
107: 
108: **Compare Button**: Only enables when BOTH an embedding is extracted AND at least one reference exists
109: 
110: ---
111: 
112: ## Models
113: 
114: ### ArcFace (Default - Better Discrimination)
115: - 512-dimensional embeddings
116: - ResNet100 backbone
117: - Better discrimination between different people
118: - Different people show <30% similarity (correct!)
119: - Same person shows ~70-85% similarity
120: 
121: ### FaceNet (Optional)
122: - 128-dimensional embeddings
123: - ResNet18 backbone
124: - Faster inference
125: - Enable: `USE_FACENET=true ./start.sh`
126: 
127: **ArcFace Thresholds**:
128: - ≥70% = Very High - Likely same person
129: - 45-70% = High - Possibly same person
130: - 30-45% = Moderate - Human review recommended
131: - <30% = Insufficient - Likely different people
132: 
133: ---
134: 
135: ## Expected Results
136: 
137: | Scenario | FaceNet | ArcFace |
138: |----------|---------|---------|
139: | Same image | ~100% | ~100% |
140: | Same person | 85-99% | ~70-85% |
141: | Different person | 50-70% | <30% |
142: 
143: **Why ArcFace is Better**:
144: - Different people show ~9-25% similarity (correctly indicates different people)
145: - FaceNet was showing 65-70% for different people (false positives!)
146: 
147: **Confidence Bands** (ArcFace):
148: - 🟢 **Very High**: ≥70% - Likely same person
149: - 🟢 **High**: 45-70% - Possibly same person
150: - 🟡 **Moderate**: 30-45% - Human review recommended
151: - 🟡 **Low**: 20-30% - Human review required
152: - 🔴 **Insufficient**: <20% - Likely different people
153: 
154: ---
155: 
156: ## MANTAX Branding
157: 
158: The application now includes MANTAX branding in the navbar:
159: - Left: MANTAX logo (SVG with red #D20A11 and white)
160: - Right: "Ihrem Partner für Autokrane und Schwerlastlogistik"
161: - Clean, professional design with white background
162: 
163: ---
164: 
165: ## API Endpoints (14+ Total)
166: 
167: | Method | Endpoint | Description |
168: |--------|----------|-------------|
169: | GET | `/api/health` | Health check |
170: | GET | `/api/embedding-info` | Model info (FaceNet/ArcFace) |
171: | POST | `/api/detect` | Detect faces |
172: | POST | `/api/extract` | Extract embedding |
173: | POST | `/api/add-reference` | Add reference |
174: | GET | `/api/references` | List references |
175: | DELETE | `/api/references/<id>` | Remove reference |
176: | POST | `/api/compare` | Compare embeddings |
177: | GET | `/api/visualizations/<type>` | Get query visualization |
178: | GET | `/api/visualizations/<type>/reference/<id>` | Get ref visualization |
179: | POST | `/api/clear` | Clear session |
180: | GET | `/api/status` | Debug state |
181: 
182: ---
183: 
184: ## 14 Visualization Types
185: 
186: | Type | Source | Description |
187: |------|--------|-------------|
188: | `detection` | FaceDetector | Bounding boxes |
189: | `extraction` | FaceDetector | Face ROI |
190: | `landmarks` | FaceDetector | 15 keypoints |
191: | `mesh3d` | FaceDetector | 478-point mesh |
192: | `alignment` | FaceDetector | Pitch/yaw/roll |
193: | `saliency` | FaceDetector | Attention heatmap |
194: | `activations` | EmbeddingExtractor | CNN activations |
195: | `features` | EmbeddingExtractor | Feature maps |
196: | `multiscale` | FaceDetector | Multi-scale detection |
197: | `confidence` | FaceDetector | Quality metrics |
198: | `embedding` | EmbeddingExtractor | Dim bar chart |
199: | `similarity` | EmbeddingExtractor | Similarity comparison |
200: | `robustness` | EmbeddingExtractor | Noise robustness test |
201: | `biometric` | FaceDetector | Biometric overview |
202: 
203: ---
204: 
205: ## Testing
206: 
207: ```bash
208: # End-to-end pipeline test (uses test_subject.jpg and reference_subject.jpg)
209: python test_e2e_pipeline.py
210: 
211: # With ArcFace (default)
212: python test_e2e_pipeline.py
213: 
214: # With FaceNet
215: USE_FACENET=true python test_e2e_pipeline.py
216: 
217: # Unit tests
218: python -m pytest tests/
219: 
220: # Clear cache before testing
221: find . -type d -name "__pycache__" -exec rm -rf {} +
222: find . -name "*.pyc" -delete
223: ```
224: 
225: **Test Results**:
226: ```
227: E2E Tests: 6/6 PASSED
228: Unit Tests: 30/30 PASSED
229: ArcFace Different Person: ~9-25% (correctly different!)
230: FaceNet Different Person: ~65-70% (false positives!)
231: ```
232: 
233: ---
234: 
235: ## Reference Storage
236: 
237: References are stored in `reference_images/embeddings.json`:
238: 
239: ```json
240: {
241:   "metadata": [
242:     {"id": "name", "path": "path/to/image.jpg", "metadata": {...}, "added_at": "timestamp"}
243:   ],
244:   "embeddings": [
245:     {"id": "name", "embedding": [0.1, 0.5, ...]}  // 128-dim or 512-dim vector
246:   ]
247: }
248: ```
249: 
250: **Note**: Only embeddings (128 or 512 floats) are stored, not images. The JSON references original image paths.
251: 
252: ---
253: 
254: ## Documentation
255: 
256: | File | Description |
257: |------|-------------|
258: | README.md | This file |
259: | PROJECT_STRUCTURE.md | Complete architecture, lessons learned, roadmap |
260: | ARCHITECTURE.md | Detailed system design |
261: | DEVELOPMENT_LOG.md | Development history |
262: | CONTEXT.md | Critical rules for code edits |
263: | ETHICAL_COMPLIANCE.md | Ethical guidelines |
264: 
265: ---
266: 
267: ## Critical Rules for Code Edits
268: 
269: ### Rule 1: Syntax Check
270: ```bash
271: python -m py_compile <file>
272: ```
273: 
274: ### Rule 2: Check for Duplicate Code
275: ```bash
276: grep -n "def " <file>
277: ```
278: 
279: ### Rule 3: Read Before Edit
280: Read at least 50 lines around the edit location.
281: 
282: ### Rule 4: Function Preservation (JS)
283: ```bash
284: grep -n "function " electron-ui/renderer/app.js | wc -l
285: ```
286: 
287: ### Rule 5: HTML-JS Cross-Check (CRITICAL!)
288: ```bash
289: for func in $(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u); do
290:     grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js || echo "MISSING: $func"
291: done
292: ```
293: 
294: ### Rule 6: Fire-and-Forget
295: ```javascript
296: // WRONG - blocks UI
297: await fetch(`${API_BASE}/clear`, { method: 'POST' });
298: 
299: // RIGHT - non-blocking
300: fetch(`${API_BASE}/clear`, { method: 'POST' }).catch(err => console.log(err));
301: ```
302: 
303: ### Rule 7: Restart After API Changes
304: ```bash
305: ./start.sh  # Clears cache, restarts API + Electron
306: ```
307: 
308: ---
309: 
310: ## Lessons Learned (Don't Repeat!)
311: 
312: ### 1. ArcFace vs FaceNet Discrimination
313: 
314: **Problem**: FaceNet showed 65-70% similarity for different people (false positives!)
315: 
316: **Solution**: ArcFace with 512-dim embeddings shows <30% for different people:
317: ```
318: Different people: ~9-25% (correctly indicates different!)
319: Same person: ~70-85% (correctly indicates same!)
320: ```
321: 
322: ### 2. Dynamic Array Sizes
323: ```python
324: # WRONG: Hardcoded size
325: output = np.zeros((150, 300, 3))
326: 
327: # RIGHT: Dynamic size
328: output_size = max(150, n * cell_size)
329: output = np.zeros((output_size, output_size, 3))
330: ```
331: 
332: ### 3. Real Embeddings
333: ```python
334: # WRONG
335: embedding = np.random.rand(128)
336: 
337: # RIGHT
338: embedding = extractor.extract_embedding(face_roi)
339: ```
340: 
341: ### 4. Clear Cache After Editing
342: ```bash
343: find . -type d -name "__pycache__" -exec rm -rf {} +
344: find . -name "*.pyc" -delete
345: ```
346: 
347: ### 5. Restart Server After Changes
348: Old Python processes don't load new code. Always restart with `./start.sh`.
349: 
350: ### 6. Port Conflicts
351: Best practice: Flask runs once, Electron connects to it. Don't spawn Python from Electron.
352: 
353: ### 7. ArcFace ONNX Model
354: ArcFace uses ONNX runtime - no direct layer access for visualizations. Use placeholder visualizations that show useful info instead of raw CNN activations.
355: 
356: ### 8. Activation Similarity
357: References now store neural network activations alongside embeddings. During comparison, activation similarity provides an additional matching signal based on internal CNN layer responses.
358: 
359: ---
360: 
361: ## Ethical Guidelines
362: 
363: 1. **Consent-Based**: All images must have lawful basis
364: 2. **Human Oversight**: No automated decisions
365: 3. **Uncertainty Handling**: Use confidence bands
366: 4. **Privacy Protection**: Non-reversible embeddings
367: 5. **Documentation**: Maintain audit trails
368: 
369: ---
370: 
371: ## File Structure
372: 
373: ```
374: face_recognition_npo/
375: ├── api_server.py              # Flask API (14+ endpoints)
376: ├── start.sh                   # Startup script (clears cache, starts servers)
377: ├── test_e2e_pipeline.py       # End-to-end tests
378: ├── reference_images/           # Persistent storage
379: │   ├── embeddings.json       # Stored references
380: │   └── README.md
381: ├── src/
382: │   ├── detection/            # Face detection (OpenCV DNN)
383: │   ├── embedding/            # 128-dim or 512-dim extraction
384: │   │   ├── __init__.py      # FaceNet extractor
385: │   │   └── arcface_extractor.py  # ArcFace extractor (ONNX)
386: │   └── reference/            # Reference management
387: ├── electron-ui/               # Desktop UI
388: │   ├── index.html            # HTML with MANTAX navbar
389: │   ├── renderer/app.js       # Frontend JavaScript
390: │   ├── styles/design-system.css  # External CSS
391: │   └── package.json
392: ├── tests/                    # Unit tests (30 tests)
393: └── gui/                      # Tkinter fallback GUI
394: ```
395: 
396: ---
397: 
398: ## ArcFace Integration Details
399: 
400: ### Model Architecture
401: - **Backbone**: ResNet100 (ONNX format)
402: - **Embedding**: 512-dimensional, L2 normalized
403: - **Runtime**: ONNX Runtime (no PyTorch dependency for inference)
404: 
405: ### Files
406: - `src/embedding/arcface_extractor.py` - ArcFace implementation
407: - `arcface_model.onnx` - ONNX model file
408: 
409: ### API Response (ArcFace)
410: ```json
411: {
412:   "success": true,
413:   "embedding_size": 512,
414:   "embedding_mean": 0.0321,
415:   "embedding_std": 0.0452,
416:   "model": "ArcFaceEmbeddingExtractor"
417: }
418: ```
419: 
420: ---
421: 
422: ## License
423: 
424: MIT License
425: 

(End of file - total 425 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/START.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/START.md</path>
<type>file</type>
<content>1: # Quick Start - NGO Facial Image Analysis System
2: 
3: **Version**: 0.5.1  
4: **Last Updated**: February 16, 2026
5: 
6: ---
7: 
8: ## ⚠️ CRITICAL WORKFLOW RULE
9: 
10: After EVERY code change:
11: 1. Run: `python test_e2e_pipeline.py`
12: 2. Run: `python test_edge_cases.py`
13: 3. Run: `python test_frontend_integration.py`
14: 4. Say "FINISHED" ONLY after ALL tests pass
15: 
16: ---
17: 
18: ## New UI Design (February2026)
19: 
20: The 16,  application now features **macOS Tahoe Liquid Glass** design:
21: - Glassmorphism step cards
22: - Step completion checkmarks
23: - Animated containers
24: - Liquid glass toast notifications
25: - Pill-shaped badges
26: 
27: ---
28: 
29: ## Start the Application
30: 
31: ```bash
32: cd /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo
33: ./start.sh
34: ```
35: 
36: Choose:
37: - `[1]` Electron Desktop App
38: - `[2]` Browser
39: - `[3]` Both
40: 
41: ---
42: 
43: ## Usage Workflow
44: 
45: ```
46: Step 1: Choose Photo     → Upload image
47: Step 2: Find Faces       → Click "Find Faces"
48: Step 3: Create Signature → Click "Create Signature" (CRITICAL!)
49: Step 4: Add Reference    → Upload reference image
50: Step 5: Compare          → Click "Compare"
51: ```
52: 
53: ---
54: 
55: ## SCSS Styling
56: 
57: The project uses SCSS for maintainable styling. 
58: 
59: **Files:**
60: - `electron-ui/styles/design-system.scss` - SCSS source
61: - `electron-ui/styles/design-system.css` - Compiled CSS
62: 
63: **Compile SCSS:**
64: ```bash
65: cd electron-ui
66: npm run scss
67: ```
68: 
69: **Note:** `./start.sh` automatically compiles SCSS before starting.
70: 
71: ---
72: 
73: ## Manual Start (Alternative)
74: 
75: ```bash
76: # Terminal 1: Start Flask API
77: cd /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo
78: source venv/bin/activate
79: python api_server.py
80: 
81: # Terminal 2: Start Electron (or use browser)
82: cd electron-ui
83: npm start
84: 
85: # Or open http://localhost:3000 in browser
86: ```
87: 
88: ---
89: 
90: ## Important Notes
91: 
92: ### Always Use Project Virtual Environment
93: ```bash
94: # CORRECT
95: source venv/bin/activate
96: python api_server.py
97: 
98: # WRONG - uses system Python
99: python api_server.py
100: ```
101: 
102: ### Clear Cache After Edits
103: ```bash
104: find . -type d -name "__pycache__" -exec rm -rf {} +
105: find . -name "*.pyc" -delete
106: ```
107: 
108: ### Run Tests
109: ```bash
110: # E2E tests
111: python test_e2e_pipeline.py
112: 
113: # API tests
114: python test_api_endpoints.py
115: ```
116: 
117: ---
118: 
119: ## API Endpoints
120: 
121: | Method | Endpoint | Description |
122: |--------|----------|-------------|
123: | GET | `/api/health` | Health check |
124: | GET | `/api/embedding-info` | Model info |
125: | POST | `/api/detect` | Face detection |
126: | POST | `/api/extract` | Embedding extraction |
127: | POST | `/api/add-reference` | Add reference |
128: | GET | `/api/references` | List references |
129: | DELETE | `/api/references/<id>` | Remove reference |
130: | POST | `/api/compare` | Compare embeddings |
131: | POST | `/api/clear` | Clear session |
132: 
133: ---
134: 
135: ## Confidence Thresholds (ArcFace)
136: 
137: | Similarity | Confidence |
138: |------------|------------|
139: | ≥70% | Very High - Likely same person |
140: | 45-70% | High - Possibly same person |
141: | 30-45% | Moderate - Human review recommended |
142: | <30% | Insufficient - Likely different people |
143: 
144: ---
145: 
146: ## Comparison Score Breakdown
147: 
148: The comparison uses 5 factors with weighted scoring:
149: 
150: | Factor | Weight | Description |
151: |--------|--------|-------------|
152: | ArcFace | 60% | Primary embedding (512-dim) |
153: | FaceNet | 20% | Secondary embedding (128-dim) |
154: | Landmarks | 15% | Geometric consistency |
155: | Activation | 5% | Neural activation similarity |
156: | Quality | 5% | Image quality factor |
157: 
158: **Note**: No pose penalty - same person with different poses will match correctly.
159: 
160: ---
161: 
162: ## Documentation
163: 
164: | File | Description |
165: |------|-------------|
166: | `README.md` | Main documentation |
167: | `CONTEXT.md` | Critical rules for code edits |
168: | `ARCHITECTURE.md` | Complete system design |
169: | `DEVELOPMENT_LOG.md` | Development history |
170: | `STYLES.md` | CSS classes reference |
171: 
172: ---
173: 
174: *Created: February 15, 2026*
175: 

(End of file - total 175 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/DEVELOPMENT_LOG.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/DEVELOPMENT_LOG.md</path>
<type>file</type>
<content>1: # NGO Facial Image Analysis System - Development Log
2: 
3: **Last Updated**: February 15, 2026
4: **Project**: Face Recognition GUI for NGO Use
5: **Version**: 0.5.0
6: **Status**: ✅ Fully Functional - Enhanced Features (LBP, Asymmetry, Multi-Pose, 3D Normalization)
7: 
8: ---
9: 
10: ## Summary (February 15, 2026) - Enhanced Features
11: 
12: The system now includes **5 new features** for better handling of different angles, lighting conditions, and face uniqueness analysis.
13: 
14: ### New Features (v0.5.0)
15: 
16: | Feature | Weight | Description |
17: |---------|--------|-------------|
18: | 3D Mesh Normalization | 15% | Handle extreme angles using 468-point mesh alignment |
19: | LBP Texture | 8% | Lighting-invariant texture matching |
20: | Facial Asymmetry | 7% | Uniqueness analysis based on facial asymmetry |
21: | Pose-Aware Weight | Modifier | Adjusts similarity based on pose difference |
22: | Multi-Pose Enrollment | 10% | Auto-detect and store multiple poses per person |
23: 
24: ### New Backend Methods
25: 
26: **FaceDetector (`src/detection/__init__.py`)**:
27: - `compute_lbp_descriptor(face_image)` → 256-dim LBP histogram
28: - `compute_facial_asymmetry(landmarks)` → Asymmetry features dict
29: - `normalize_face_with_mesh(face_image, mesh_landmarks)` → Aligned face
30: 
31: **SimilarityComparator (`src/embedding/__init__.py`)**:
32: - `compute_pose_weight(pose1, pose2)` → 0.95-1.0
33: - `lbp_similarity(lbp1, lbp2)` → 0-1
34: - `asymmetry_similarity(asym1, asym2)` → 0-1
35: - `compute_multi_pose_score(query_emb, pose_embeddings)` → Best score
36: 
37: ### New API Endpoints
38: 
39: - `POST /api/add-reference-pose/<ref_id>` - Add pose variant to reference
40: 
41: ### Updated Endpoints
42: 
43: - `/api/extract` - Now returns: lbp_histogram, asymmetry, normalized_embedding
44: - `/api/add-reference` - Now stores: poses dict, lbp_histogram, asymmetry, normalized_embedding
45: - `/api/compare` - Now returns: pose_weight, lbp_similarity, asymmetry_similarity, normalized_similarity, multi_pose_score, multi_pose_used
46: 
47: ### New Frontend Features
48: 
49: **HTML**:
50: - Added tabs: Uniqueness, Texture, 3D Normalized
51: - Added score displays: 3D Normalized, Multi-Pose, Texture, Uniqueness
52: 
53: **JavaScript**:
54: - Updated comparison display to show all new scores
55: 
56: ### New Tests
57: 
58: Added 5 new test functions to `test_edge_cases.py`:
59: - `test_lbp_descriptor()` - LBP extraction and similarity
60: - `test_asymmetry_features()` - Asymmetry computation
61: - `test_pose_weight()` - Pose weight calculation
62: - `test_mesh_normalization()` - 3D mesh normalization
63: - `test_multi_pose_score()` - Multi-pose comparison
64: 
65: ### Files Updated
66: 
67: - `src/detection/__init__.py` - Added 3 new methods
68: - `src/embedding/__init__.py` - Added 4 new methods
69: - `api_server.py` - Updated 3 endpoints, added 1 new endpoint
70: - `electron-ui/index.html` - Added visualization tabs, score displays
71: - `electron-ui/renderer/app.js` - Updated comparison display
72: - `test_edge_cases.py` - Added 5 new test functions
73: - `ARCHITECTURE.md` - Updated with all new features
74: 
75: ### Test Results
76: 
77: ```
78: ✓ PASS: E2E Pipeline (6/6)
79: ✓ PASS: Edge Cases (16/16) 
80: ✓ PASS: Frontend Integration (9/9)
81: ```
82: 
83: ---
84: 
85: ## Summary (February 14, 2026)
86: 
87: The system now includes **reference details panel** and **UI improvements** for better user experience.
88: 
89: ### New Features (v0.4.1)
90: 
91: | Feature | Description |
92: |---------|-------------|
93: | Reference Details Panel | View reference embeddings, landmarks, pose, quality metrics |
94: | Smart Compare Button | Only enables when BOTH embedding AND references exist |
95: | Card Layout | Comparison results displayed in centered card |
96: | Step Hints | Clear indicators for what to do next |
97: 
98: ### Removed Features (v0.4.1)
99: 
100: - Compare overlay visualization
101: - Compare difference visualization
102: - Compare side-by-side view
103: 
104: ### Files Updated
105: 
106: - `electron-ui/index.html` - Reference details panel, card layout
107: - `electron-ui/renderer/app.js` - Compare button logic, removed compare viz functions
108: - `electron-ui/styles/design-system.css` - Card styles, step hints
109: - `README.md` - Updated features and API endpoints
110: - `PROJECT_STRUCTURE.md` - Updated workflow
111: 
112: ---
113: 
114: ## Summary (February 14, 2026) - Activation Similarity
115: 
116: The system now includes **activation similarity comparison** and **compare visualizations** for enhanced face matching and visual comparison.
117: 
118: ### New Features (v0.4.0)
119: 
120: | Feature | Description |
121: |---------|-------------|
122: | Activation Similarity | Neural network activation comparison during matching |
123: | Reference Details Panel | View reference embeddings, landmarks, pose, quality |
124: | Compare Overlay | Visual overlay of query and reference faces |
125: | Compare Difference | Pixel-by-pixel difference visualization |
126: | Compare Side-by-Side | Both images shown side by side |
127: | External CSS | Moved styles to design-system.css |
128: 
129: ### API Endpoints Added
130: 
131: - `GET /api/visualizations/<type>/reference/<id>` - Reference visualizations
132: - `GET /api/visualizations/compare-overlay/<id>` - Overlay comparison
133: - `GET /api/visualizations/compare-diff/<id>` - Difference comparison
134: 
135: ### Files Added
136: 
137: - `electron-ui/styles/design-system.css` - External CSS styles
138: 
139: ### Files Updated
140: 
141: - `api_server.py` - Added activations storage, compare visualization endpoints
142: - `electron-ui/index.html` - Reference details panel, compare viz tabs
143: - `electron-ui/renderer/app.js` - New functions for details and compare viz
144: - `README.md` - ArcFace documentation
145: - `ARCHITECTURE.md` - ArcFace architecture
146: - `CONTEXT.md` - ArcFace rules
147: - `DEVELOPMENT_LOG.md` - This file
148: 
149: ---
150: 
151: ## February 12, 2026 - ArcFace Integration
152: 
153: ### Why ArcFace?
154: 
155: **Problem**: FaceNet 128-dim embeddings showed 65-70% similarity for different people:
156: ```
157: FaceNet Results:
158: - Same image: ~100% ✓
159: - Same person: ~85-99% ✓
160: - Different people: ~65-70% ✗ (FALSE POSITIVES!)
161: ```
162: 
163: **Solution**: ArcFace 512-dim embeddings provide much better discrimination:
164: ```
165: ArcFace Results:
166: - Same image: ~100% ✓
167: - Same person: ~70-85% ✓
168: - Different people: <30% ✓ (CORRECTLY DIFFERENT!)
169: ```
170: 
171: ### ArcFace Implementation
172: 
173: **Model**: ResNet100 in ONNX format
174: - **Dimension**: 512 (vs 128 for FaceNet)
175: - **Runtime**: ONNX Runtime (no PyTorch needed for inference)
176: - **Speed**: Fast inference
177: 
178: **Key Changes**:
179: 
180: 1. **ArcFace Extractor** (`src/embedding/arcface_extractor.py`):
181:    - Loads ONNX model
182:    - Preprocesses face to 112x112 RGB
183:    - Extracts 512-dim L2-normalized embedding
184:    - Placeholder activations (ONNX doesn't expose layers)
185: 
186: 2. **API Server** (`api_server.py`):
187:    - Detects ArcFace model availability
188:    - Auto-selects ArcFace if available, falls back to FaceNet
189:    - Added `/api/embedding-info` endpoint
190: 
191: 3. **Visualizations**:
192:    - 512-dim bar chart for embedding visualization
193:    - Placeholder for activations (no layer access in ONNX)
194: 
195: ### ArcFace Thresholds
196: 
197: | Similarity | Confidence | Action |
198: |------------|------------|--------|
199: | ≥70% | Very High | Likely same person |
200: | 45-70% | High | Possibly same person |
201: | 30-45% | Moderate | Human review recommended |
202: | <30% | Insufficient | Likely different people |
203: 
204: ### Expected Results
205: 
206: ```
207: Same image: ~100% similarity
208: Same person: ~70-85% similarity
209: Different people: ~9-25% similarity
210: ```
211: 
212: ---
213: 
214: ## Summary (February 11, 2026)
215: 
216: The system is now **fully functional** with:
217: - ✅ Real 128-dim embeddings (not random!)
218: - ✅ Dynamic array sizes for visualizations
219: - ✅ Working compare with proper similarity scores
220: - ✅ Reference persistence to `embeddings.json`
221: - ✅ Electron app loads references on startup
222: - ✅ All tests passing (6/6 E2E, 30/30 unit)
223: - ✅ ArcFace integration (512-dim, better discrimination)
224: 
225: **Test Results:**
226: ```
227: E2E Tests: 6/6 PASSED
228: Unit Tests: 30/30 PASSED
229: Same image:          100% similarity
230: Different person:    ~78% similarity (FaceNet - to be replaced with ArcFace)
231: ```
232: 
233: **Files Updated Today:**
234: - `api_server.py` - Added `save_references()`, `load_references()`, persistence
235: - `app.js` - Added `loadReferences()` to load on startup
236: - `start.sh` - Clears cache, starts API + Electron
237: - `test_e2e_pipeline.py` - Fixed default image paths
238: - `README.md` - Complete documentation
239: 
240: ---
241: 
242: ## February 11, 2026 - Persistence Fixes
243: 
244: ### Bug: References Not Saved to JSON
245: 
246: **Problem**: When adding references via the app, they were stored in memory but not saved to `reference_images/embeddings.json`.
247: 
248: **Fix**: Added `save_references()` function to API:
249: ```python
250: REFERENCES_FILE = os.path.join(os.path.dirname(__file__), 'reference_images', 'embeddings.json')
251: 
252: def save_references():
253:     """Save references to JSON file."""
254:     data = {
255:         'metadata': [...],
256:         'embeddings': [...]
257:     }
258:     with open(REFERENCES_FILE, 'w') as f:
259:         json.dump(data, f, indent=2)
260: 
261: def load_references():
262:     """Load references from JSON file on startup."""
263:     # Load existing references
264: ```
265: 
266: **Added calls to `save_references()`**:
267: - After `add_reference()` - line 265
268: - After `remove_reference()` - line 323
269: 
270: ### Bug: App Not Loading References on Startup
271: 
272: **Problem**: Electron app started with empty `references = []`, never loading existing references.
273: 
274: **Fix**: Added `loadReferences()` function:
275: ```javascript
276: async function checkAPI() {
277:     // ...
278:     if (data.status === 'ok') {
279:         logToTerminal('> API connected', 'success');
280:         loadReferences();  // NEW
281:     }
282: }
283: 
284: async function loadReferences() {
285:     const response = await fetch(`${API_BASE}/references`);
286:     const data = await response.json();
287:     if (data.references) {
288:         references = data.references;
289:         updateReferenceList();
290:     }
291: }
292: ```
293: 
294: ### Bug: Start Script Not Clearing Cache
295: 
296: **Problem**: Old Python processes cached old code.
297: 
298: **Fix**: Updated `start.sh`:
299: ```bash
300: # Clear Python cache
301: find . -type d -name "__pycache__" -exec rm -rf {} +
302: find . -name "*.pyc" -delete
303: find . -name ".pytest_cache" -exec rm -rf {} +
304: ```
305: 
306: ### Bug: Test Images Had Wrong Names
307: 
308: **Problem**: Default test images were `kanye_west.jpeg` and `kanye_detected.jpeg` which don't exist.
309: 
310: **Fix**: Updated `test_e2e_pipeline.py`:
311: ```python
312: TEST_IMAGE = os.environ.get('TEST_IMAGE', 'test_subject.jpg')
313: TEST_IMAGE_REF = os.environ.get('TEST_IMAGE_REF', 'reference_subject.jpg')
314: ```
315: 
316: Renamed `kanye_west_ref.jpg` → `reference_subject.jpg`
317: 
318: ---
319: 
320: ## Session: February 11, 2026 - MediaPipe API Update
321: 
322: ### Bug: UnboundLocalError with MEDIAPIPE_AVAILABLE
323: 
324: **Problem**: Code was using `MEDIAPIPE_AVAILABLE` (no underscore) but module variable was `_MEDIAPIPE_AVAILABLE` (with underscore).
325: 
326: **Locations Fixed**:
327: - `src/detection/__init__.py:134` - `estimate_landmarks` method
328: - `src/detection/__init__.py:371` - `visualize_3d_mesh` method
329: 
330: **Fix**:
331: ```python
332: # Before (broken):
333: if MEDIAPIPE_AVAILABLE and self.face_mesh:
334: 
335: # After (fixed):
336: if self._mediapipe_available and self.face_mesh:
337: ```
338: 
339: ### MediaPipe 0.10.x API Migration
340: 
341: **Problem**: MediaPipe 0.10.x uses the Tasks API, not the old Solutions API.
342: 
343: **Changes Made**:
344: 1. Updated imports to use `mediapipe.tasks.vision`
345: 2. Changed from `mp.solutions.face_mesh.FaceMesh()` to `vision.FaceLandmarker.create_from_options()`
346: 3. Updated result handling from `results.multi_face_landmarks[0].landmark` to `results.face_landmarks[0]`
347: 
348: **Note**: The MediaPipe Tasks API requires a model file (`face_landmark.task`) which isn't present. The system gracefully falls back to proportional landmark estimation, which works well for NGO use cases.
349: 
350: **Status**: ✅ Code updated, fallback working, all tests passing
351: 
352: ### Bug 1: Reference Embeddings Were Random ❌ → ✅ FIXED
353: 
354: **Problem**: `ReferenceImageManager` was using random embeddings instead of real ones.
355: 
356: **Location**: `src/reference/__init__.py:103`
357: 
358: **Before**:
359: ```python
360: embedding = np.random.rand(128)  # WRONG!
361: ```
362: 
363: **After**:
364: ```python
365: if self.embedding_extractor is not None:
366:     detector_to_use = self.detector
367:     if detector_to_use is None and hasattr(self.embedding_extractor, 'detector'):
368:         detector_to_use = self.embedding_extractor.detector
369: 
370:     if detector_to_use is not None:
371:         faces = detector_to_use.detect_faces(image_array)
372:         if faces:
373:             x, y, w, h = faces[0]
374:             face_roi = image_array[y:y+h, x:x+w]
375:             embedding = self.embedding_extractor.extract_embedding(face_roi)
376: ```
377: 
378: **Changes Made**:
379: - Added `detector` parameter to `ReferenceImageManager.__init__()`
380: - Added real face detection before embedding extraction
381: - Fixed JSON serialization of embeddings
382: - Fixed `get_reference_embeddings()` to return stored embeddings
383: 
384: ---
385: 
386: ### Bug 2: Visualization Array Size Fixed ❌ → ✅ FIXED
387: 
388: **Problem**: 
389: ```
390: could not broadcast input array from shape (650,650,3) into shape (150,300,3)
391: ```
392: 
393: **Root Cause**: Hardcoded array size `(150, 300)` in `visualize_similarity_matrix()`. When there were many references (12+), the matrix exceeded the fixed size.
394: 
395: **Location**: `src/embedding/__init__.py`
396: 
397: **Before**:
398: ```python
399: output = np.zeros((150, 300, 3), dtype=np.uint8)
400: # ...
401: output[:n * cell_size, :n * cell_size] = matrix_colored  # FAILS when n * cell_size > 300
402: ```
403: 
404: **After**:
405: ```python
406: # Dynamic output size based on number of embeddings
407: output_size = max(150, n * cell_size)
408: output = np.zeros((output_size, output_size, 3), dtype=np.uint8)
409: output.fill(245)
410: 
411: resized = cv2.resize(matrix_colored, (n * cell_size, n * cell_size))
412: output[:resized.shape[0], :resized.shape[1]] = resized  # Safe assignment
413: ```
414: 
415: **Rule Learned**: Always use dynamic array sizes when output dimensions depend on input count.
416: 
417: ---
418: 
419: ### Bug 3: Python Cache Issues ❌ → ✅ FIXED
420: 
421: **Problem**: Changes to source files weren't picked up by running server.
422: 
423: **Root Cause**: Python's `.pyc` bytecode cache was holding old code.
424: 
425: **Solution**: Clear cache after editing:
426: ```bash
427: find . -type d -name "__pycache__" -exec rm -rf {} +
428: find . -name "*.pyc" -delete
429: ```
430: 
431: ---
432: 
433: ## Files Changed (February 11, 2026)
434: 
435: | File | Changes |
436: |------|---------|
437: | `src/reference/__init__.py` | Added detector/extractor params, real embeddings |
438: | `src/embedding/__init__.py` | Dynamic array sizes for similarity matrix |
439: | `gui/facial_analysis_gui.py` | Pass extractor/detector to ReferenceManager |
440: | `api_server.py` | Added `/api/status` debug endpoint |
441: | `electron-ui/renderer/app.js` | Better error messages, clearer status |
442: | `test_e2e_pipeline.py` | New comprehensive E2E test script |
443: | `start.sh` | New interactive start script |
444: 
445: ---
446: 
447: ## Session: February 11, 2026 (Afternoon) - Visualization Fixes
448: 
449: ### Fix: Activations and Features Visualizations
450: 
451: **Problem**: `visualize_activations()` and `visualize_feature_maps()` were using wrong model reference.
452: 
453: **Locations Fixed**:
454: - `src/embedding/__init__.py:326` - `visualize_activations()` used `self.backbone` (undefined)
455: - `src/embedding/__init__.py:383` - `visualize_feature_maps()` used `self.model.backbone` (correct)
456: 
457: **Changes Made**:
458: ```python
459: # Before (broken):
460: backbone = self.backbone  # WRONG - self.backbone doesn't exist
461: 
462: # After (fixed):
463: backbone = self.model.backbone  # CORRECT - access via model
464: ```
465: 
466: ### Fix: RGB/BGR Color Conversion
467: 
468: **Problem**: OpenCV uses BGR, PyTorch outputs RGB. Colormaps were applied incorrectly.
469: 
470: **Fix**:
471: ```python
472: channel_colored = cv2.applyColorMap(channel, cv2.COLORMAP_VIRIDIS)
473: channel_bgr = cv2.cvtColor(channel_colored, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
474: ```
475: 
476: ### Enhancement: Better Error Messages
477: 
478: **Problem**: Users couldn't tell why visualizations weren't showing.
479: 
480: **Frontend Changes** (`electron-ui/renderer/app.js`):
481: - Added console logging for debugging
482: - Added clear error messages when visualizations unavailable
483: - Added validation for None/empty face images
484: - Error messages now show: "Click Create Signature first"
485: 
486: ### Enhancement: Configurable Test Images
487: 
488: **Problem**: Hardcoded `kanye_west.jpeg` filename in tests.
489: 
490: **Changes**:
491: - `test_e2e_pipeline.py` now accepts environment variables:
492:   - `TEST_IMAGE` - Primary test image (default: `test_subject.jpg`)
493:   - `TEST_IMAGE_REF` - Reference image (default: `reference_subject.jpg`)
494: - `visualize_biometric.py` - Same configurable approach
495: 
496: **Usage**:
497: ```bash
498: TEST_IMAGE=my_image.jpg TEST_IMAGE_REF=ref_image.jpg python test_e2e_pipeline.py
499: ```
500: 
501: ---
502: 
503: ## Files Changed (February 11, 2026 - Afternoon)
504: 
505: | File | Changes |
506: |------|---------|
507: | `src/embedding/__init__.py` | Fixed `visualize_activations()` and `visualize_feature_maps()` |
508: | `electron-ui/renderer/app.js` | Added console logging, better error messages |
509: | `test_e2e_pipeline.py` | Made test images configurable via env vars |
510: | `visualize_biometric.py` | Made test image configurable via env var |
511: | `reference_images/README.md` | Updated documentation |
512: 
513: ---
514: 
515: ## Session: February 10, 2026 - Ultra Minimal UI Redesign
516: 
517: ### Goals Achieved
518: - Ultra minimal design: sans serif, black on white, no icons
519: - Sticky terminal footer (always visible)
520: - Clean workflow with step indicators
521: 
522: ### Design Decisions
523: 
524: **UI Style:**
525: - Font: System sans serif (-apple-system, SF Pro, Segoe UI, Roboto)
526: - Colors: Black text (#000), white background (#fff)
527: - No icons - text only for labels
528: - Buttons: White background, black border
529: - Step numbers for workflow (Step 1, 2, 3, 4)
530: 
531: **Terminal Footer:**
532: - Fixed at bottom, always visible
533: - Shows live processing logs
534: - Compact (5 lines) or expanded (click to toggle `[+]`/`[-]`)
535: - Black background, green monospace text
536: 
537: ---
538: 
539: ## Workflow (Correct Usage)
540: 
541: ```
542: Step 1: Choose Photo     → Upload image
543: Step 2: Find Faces       → Click "Find Faces" button
544: Step 3: Create Signature → Click "Create Signature" button (CRITICAL!)
545: Step 4: Add Reference    → Upload reference image
546: Step 5: Compare          → Click "Compare" button
547: ```
548: 
549: **Common Mistakes:**
550: - ❌ Skipping "Create Signature" - embedding won't be extracted
551: - ❌ Clicking "Compare" before adding a reference
552: - ❌ Refreshing page (loses server state)
553: 
554: ---
555: 
556: ## Test Results
557: 
558: ```bash
559: $ python test_e2e_pipeline.py
560: 
561: ============================================================
562: END-TO-END FACE RECOGNITION PIPELINE TEST
563: ============================================================
564: 
565: [TEST 1] Face Detection Pipeline         ✅ PASS
566: [TEST 2] Embedding Extraction Pipeline   ✅ PASS
567: [TEST 3] Reference Manager (REAL emb)    ✅ PASS
568: [TEST 4] Same Image Similarity           ✅ 100%
569: [TEST 5] Different Images Similarity     ✅ With ArcFace: ~9-25%
570: [TEST 6] Full Reference Comparison       ✅ PASS
571: 
572: ALL TESTS PASSED - Pipeline is working correctly!
573: ```
574: 
575: ---
576: 
577: ## API Endpoints
578: 
579: | Method | Endpoint | Description |
580: |--------|----------|-------------|
581: | GET | `/api/health` | Health check |
582: | GET | `/api/embedding-info` | Model info (ArcFace/FaceNet) |
583: | POST | `/api/detect` | Detect faces in uploaded image |
584: | POST | `/api/extract` | Extract embedding from detected face |
585: | POST | `/api/add-reference` | Add reference image for comparison |
586: | GET | `/api/references` | List all references |
587: | DELETE | `/api/references/<id>` | Remove reference |
588: | POST | `/api/compare` | Compare query embedding with references |
589: | GET | `/api/visualizations/<type>` | Get specific AI visualization |
590: | POST | `/api/clear` | Clear all session data |
591: | GET | `/api/status` | Debug server state |
592: 
593: ---
594: 
595: ## Quick Start
596: 
597: ```bash
598: cd face_recognition_npo
599: ./start.sh
600: 
601: # Or manually:
602: source venv/bin/activate
603: python api_server.py
604: # Open http://localhost:3000 in browser
605: ```
606: 
607: ---
608: 
609: ## Commands Reference
610: 
611: ```bash
612: # Start the system
613: ./start.sh
614: 
615: # Run tests
616: python test_e2e_pipeline.py
617: 
618: # Clear Python cache (after editing source)
619: find . -type d -name "__pycache__" -exec rm -rf {} +
620: find . -name "*.pyc" -delete
621: 
622: # Run Flask directly
623: source venv/bin/activate
624: python api_server.py
625: 
626: # Run Electron
627: cd electron-ui
628: npm start
629: ```
630: 
631: ---
632: 
633: ## Roadmap
634: 
635: ### Completed ✅ (v0.3.0)
636: 
637: **Core Pipeline**
638: - [x] Face detection with OpenCV DNN
639: - [x] 512-dim embedding extraction (ArcFace ONNX)
640: - [x] 128-dim embedding extraction (FaceNet PyTorch)
641: - [x] Cosine similarity comparison
642: - [x] Confidence bands (ArcFace: ≥70%, 45-70%, 30-45%, <30%)
643: 
644: **User Interfaces**
645: - [x] Electron desktop app (connects to Flask)
646: - [x] Flask API server with 11 endpoints
647: - [x] Tkinter GUI
648: - [x] Ultra minimal UI (black on white, no icons)
649: - [x] Sticky terminal footer
650: - [x] MANTAX navbar branding
651: 
652: **Reference Management**
653: - [x] Reference storage in JSON format
654: - [x] Metadata tracking (consent, source, timestamp)
655: - [x] Real embeddings (NOT random!)
656: - [x] Persistence across restarts
657: 
658: **Visualizations**
659: - [x] 14 AI visualization types
660: - [x] Dynamic array sizes (fixed broadcast bug!)
661: - [x] ArcFace placeholder visualizations
662: 
663: **Testing & Tools**
664: - [x] End-to-end test script
665: - [x] Interactive start script
666: - [x] API verification script
667: 
668: ### In Progress
669: - [ ] GPU acceleration for ONNX Runtime
670: 
671: ### Future Enhancements
672: - [ ] Batch processing API
673: - [ ] Cloud storage integration
674: - [ ] Mobile app
675: - [ ] WebSocket for real-time updates
676: 
677: ---
678: 
679: ## Lessons Learned (For Future Reference)
680: 
681: ### 1. ArcFace vs FaceNet Discrimination
682: 
683: **Problem**: FaceNet showed 65-70% for different people (false positives!)
684: 
685: **Solution**: ArcFace with 512-dim shows <30% for different people:
686: ```
687: Different people: ~9-25% (correctly indicates different!)
688: Same person: ~70-85% (correctly indicates same!)
689: ```
690: 
691: ### 2. Dynamic Array Sizes
692: ```python
693: # WRONG
694: output = np.zeros((150, 300, 3), dtype=np.uint8)
695: 
696: # RIGHT
697: output_size = max(150, n * cell_size)
698: output = np.zeros((output_size, output_size, 3), dtype=np.uint8)
699: ```
700: 
701: ### 3. Real Embeddings
702: ```python
703: # WRONG
704: embedding = np.random.rand(512)
705: 
706: # RIGHT
707: embedding = extractor.extract_embedding(face_roi)
708: ```
709: 
710: ### 4. Clear Cache
711: Always clear Python cache after editing source:
712: ```bash
713: find . -type d -name "__pycache__" -exec rm -rf {} +
714: find . -name "*.pyc" -delete
715: ```
716: 
717: ### 5. Verify API First
718: Test API directly before debugging frontend:
719: ```python
720: r = requests.post('http://localhost:3000/api/detect', json={...})
721: r = requests.post('http://localhost:3000/api/extract', json={...})
722: r = requests.post('http://localhost:3000/api/compare', json={...})
723: r = requests.get('http://localhost:3000/api/embedding-info')
724: ```
725: 
726: ---
727: 
728: ## Session: February 15, 2026 - CSS Cleanup & Activation Similarity
729: 
730: ### Important Workflow Change
731: 
732: **Added to all documentation:**
733: After EVERY code change, ALWAYS run tests before saying finished:
734: 1. `python test_e2e_pipeline.py`
735: 2. `python test_edge_cases.py`
736: 3. `python test_frontend_integration.py`
737: 4. Say "FINISHED" only after ALL tests pass
738: 
739: ### SCSS Conversion
740: 
741: **Created SCSS design system:**
742: 
743: 1. **New file**: `electron-ui/styles/design-system.scss`
744:    - Variables for all colors, spacing, typography
745:    - Mixins for reusable patterns (flex-center, card, scrollbar)
746:    - Nested component styles
747:    - Exact same output as original CSS
748: 
749: 2. **Build system**: Added npm script
750:    ```json
751:    "scss": "sass --no-source-map --style=expanded styles/design-system.scss styles/design-system.css"
752:    ```
753:    Run: `npm run scss` or use `./start.sh` (auto-compiles)
754: 
755: 3. **start.sh updated**:
756:    - Fixed bug: Added `SCRIPT_DIR` variable to properly resolve paths
757:    - Added SCSS compilation step
758:    - Now works from any directory
759: 
760: ### CSS Cleanup
761: 
762: **Phase 1: CSS Foundation Cleanup**
763: 
764: 1. **Fixed duplicate `--bg-secondary`** - was defined twice (#FFFFFF and #f5f5f5), kept #f5f5f5
765: 
766: 2. **Added missing CSS variables**:
767:    - Spacing: `--space-1` through `--space-8`
768:    - Typography: `--text-xs` through `--text-4xl`
769:    - Font weights: `--font-normal`, `--font-medium`, `--font-semibold`, `--font-bold`
770:    - Layout: `--sidebar-width`, `--header-height`
771: 
772: 3. **Added utility classes**:
773:    - `.hidden` - display: none
774:    - `.button-group` - flex with gap
775:    - `.text-center` - text-align center
776:    - `.text-error-inline` - error text color
777:    - `.empty-state` - muted empty state
778:    - `.webcam-video` - webcam styling
779:    - `.btn-margin-top` - button margin
780: 
781: **Phase 2: Remove Inline Styles from HTML**
782: 
783: - Removed 8 inline `style=""` attributes from `index.html`
784: - Replaced with CSS classes: `.hidden`, `.button-group`, `.webcam-video`, etc.
785: 
786: **Phase 3: Remove Inline Styles from JavaScript**
787: 
788: - Removed 7 inline `style=""` attributes from `app.js`
789: - Added `.text-error-inline` and `.empty-state` CSS classes
790: 
791: **Phase 4: Fixed JavaScript Toggle Logic**
792: 
793: - Changed all `element.style.display = 'none/block'` to `classList.add/remove('hidden')`
794: - Fixed reference details panel to use `.active` class instead of inline styles
795: - Fixed comparison result display (removed `.hidden` before adding `.active`)
796: 
797: ### Activation Similarity
798: 
799: **Added neural network activation comparison**:
800: 
801: 1. **Backend** (`src/embedding/__init__.py`):
802:    - Added `activation_sim` parameter to `compute_dual_match_score()`
803:    - Added 5% weight for activation similarity
804:    - Added `activation_similarity()` method to compare intermediate layer outputs
805: 
806: 2. **API** (`api_server.py`):
807:    - Now passes `activation_sim` to scoring method
808:    - Returns `activation_similarity` in API response
809: 
810: 3. **Frontend** (`index.html`):
811:    - Added Activation row to similarity display
812: 
813: 4. **Frontend** (`app.js`):
814:    - Added code to populate activation score
815: 
816: **New Weights:**
817: | Factor | Weight |
818: |--------|--------|
819: | ArcFace | 60% |
820: | FaceNet | 20% |
821: | Landmarks | 15% |
822: | Activation | 5% |
823: | Quality | 5% |
824: 
825: **Note**: No pose penalty - same person with different poses should match correctly.
826: 
827: ---
828: 
829: ## Summary (February 16, 2026) - macOS Tahoe UI Redesign
830: 
831: ### New UI Design System
832: 
833: **macOS Tahoe Liquid Glass Theme** applied to the Electron UI.
834: 
835: ### SCSS Changes (`electron-ui/styles/design-system.scss`)
836: 
837: 1. **macOS Titlebar**
838:    - Fixed position at top (52px height)
839:    - Glass blur effect (`backdrop-filter: blur(30px) saturate(180%)`)
840:    - Traffic light buttons (close/minimize/maximize)
841: 
842: 2. **Step Cards with Glassmorphism**
843:    - `.step`: Added glass background, blur, border, shadow
844:    - `.step-header`: Flex layout with column direction
845:    - Transition animation (0.4s cubic-bezier)
846: 
847: 3. **Step Pills (Badges)**
848:    - `.step-number`: Pill-shaped, glass effect, dark text (#1D1D1F)
849:    - `.step-number.webcam`: Blue background (#5856D6)
850:    - `.step-number.optional`: Glass effect
851: 
852: 4. **Step Completion Indicators**
853:    - `.step-complete`: Green border + shadow when step done
854:    - `.step-checkmark`: Circle with checkmark (appears on complete)
855:    - `pointer-events: none` to avoid blocking clicks
856: 
857: 5. **Preview Box**
858:    - Removed glass effect (just container)
859:    - 24px border-radius
860:    - Simple background
861: 
862: 6. **Toast Notifications**
863:    - Liquid glass effect (blur + tint + shine)
864:    - Color backgrounds for success/error/warning (25% opacity)
865:    - Position: top: 76px, right: 24px
866:    - Duration: 5 seconds
867: 
868: 7. **Buttons**
869:    - `.btn-success`: Green success button
870:    - Applied to completed steps
871: 
872: 8. **Animations**
873:    - `.hidden` / `.visible` classes for smooth transitions
874:    - `max-height` and `opacity` transitions
875: 
876: ### HTML Changes (`electron-ui/index.html`)
877: 
878: 1. Added `.step-checkmark` div to: step1, webcamStep, step2, step3
879: 2. Added `id="step1"`, `id="step2"`, `id="step3"`, `id="webcamStep"`
880: 3. Added "Optional" pill next to Webcam
881: 4. Container padding-top: 76px for titlebar
882: 
883: ### JavaScript Changes (`electron-ui/renderer/app.js`)
884: 
885: 1. **markStepComplete(stepId, btnId)**: New function to mark steps complete
886: 2. **Updated resetSteps()**: Reset step states
887: 3. **Updated clearAllCache()**: Reset all steps + buttons
888: 4. **handleImageSelect()**: Mark step1 complete
889: 5. **detectFaces()**: Mark step2 complete  
890: 6. **extractFeatures()**: Mark step3 complete
891: 7. Fixed syntax error (duplicate code outside function)
892: 
893: ### Tests
894: 
895: - E2E: 6/6 ✅
896: - Edge Cases: 16/16 ✅
897: - Frontend Integration: 9/9 ✅
898: 
899: ---
900: 
901: *Document updated: February 16, 2026*
902: *Includes macOS Tahoe UI redesign with glassmorphism*
903: 
904: ---
905: 
906: *Document updated: February 12, 2026*
907: *Includes ArcFace integration, 512-dim embeddings, and MANTAX branding*
908: 

(End of file - total 908 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/ARCHITECTURE.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/ARCHITECTURE.md</path>
<type>file</type>
<content>1: # NGO Facial Image Analysis System - Architecture
2: 
3: **Version**: 0.5.0  
4: **Last Updated**: February 15, 2026  
5: **Status**: ✅ Fully Functional - Enhanced Features (LBP, Asymmetry, Multi-Pose, 3D Normalization)
6: 
7: ---
8: 
9: ## ⚠️ CRITICAL WORKFLOW RULE
10: 
11: After EVERY code change:
12: 1. Run: `python test_e2e_pipeline.py`
13: 2. Run: `python test_edge_cases.py`
14: 3. Run: `python test_frontend_integration.py`
15: 4. Say "FINISHED" ONLY after ALL tests pass
16: 
17: ---
18: 
19: ## System Overview
20: 
21: This document describes the complete architecture of the NGO Facial Image Analysis System, an ethical, consent-based facial recognition system for NGO documentation verification.
22: 
23: ---
24: 
25: ## Architecture Diagram
26: 
27: ```
28: ┌─────────────────────────────────────────────────────────────────────────┐
29: │                         SYSTEM ARCHITECTURE                              │
30: ├─────────────────────────────────────────────────────────────────────────┤
31: │                                                                          │
32: │  ┌──────────────────────────────────────────────────────────────────┐   │
33: │  │                        ENTRY POINTS                               │   │
34: │  ├──────────────────────────────────────────────────────────────────┤   │
35: │  │   ./start.sh          → Interactive menu                          │   │
36: │  │   python api_server.py → Flask API Server :3000                    │   │
37: │  │   npm start           → Electron Desktop App (connects to Flask)  │   │
38: │  │   python gui/*.py     → Tkinter Standalone GUIs                  │   │
39: │  └──────────────────────────────────────────────────────────────────┘   │
40: │                                    │                                     │
41: │                                    ▼                                     │
42: │  ┌──────────────────────────────────────────────────────────────────┐   │
43: │  │                    ELECTRON DESKTOP APP                           │   │
44: │  ├──────────────────────────────────────────────────────────────────┤   │
45: │  │   main.js              → Connects to existing Flask server      │   │
46: │  │   renderer/app.js      → Frontend JavaScript (HTTP API calls)   │   │
47: │  │   index.html           → Ultra minimal UI with MANTAX navbar      │   │
48: │  │   preload.js           → Context bridge                          │   │
49: │  │                                                                  │   │
50: │  │   Flow: User → UI → fetch() → Flask API → ML Models → Results   │   │
51: │  └──────────────────────────────────────────────────────────────────┘   │
52: │                                    │                                     │
53: │                          HTTP :3000 (REST API)                          │
54: │                                    │                                     │
55: │                                    ▼                                     │
56: │  ┌──────────────────────────────────────────────────────────────────┐   │
57: │  │                     FLASK API SERVER (BACKEND)                    │   │
58: │  ├──────────────────────────────────────────────────────────────────┤   │
59: │  │   Endpoints:                                                      │   │
60: │  │   • GET  /api/health                      → Status check         │   │
61: │  │   • GET  /api/embedding-info              → Model info          │   │
62: │  │   • POST /api/detect                      → Face detection       │   │
63: │  │   • POST /api/extract                     → Embedding extraction │   │
64: │  │   • POST /api/add-reference               → Add reference       │   │
65: │  │   • GET  /api/references                  → List references     │   │
66: │  │   • DELETE /api/references/<id>           → Remove reference    │   │
67: │  │   • POST /api/compare                     → Similarity compare  │   │
68: │  │   • GET  /api/visualizations/<type>       → Get visualization  │   │
69: │  │   • POST /api/clear                       → Clear session       │   │
70: │  │   • GET  /api/status                      → Debug server state │   │
71: │  │                                                                  │   │
72: │  │   In-Memory Session:                                              │   │
73: │  │   • current_image, current_faces, current_embedding              │   │
74: │  │   • references (in-memory list)                                   │   │
75: │  └──────────────────────────────────────────────────────────────────┘   │
76: │                                    │                                     │
77: │                                    ▼                                     │
78: │  ┌──────────────────────────────────────────────────────────────────┐   │
79: │  │                      CORE ML PIPELINE                             │   │
80: │  ├──────────────────────────────────────────────────────────────────┤   │
81: │  │                                                                  │   │
82: │  │   ┌──────────────────┐                                           │   │
83: │  │   │   FaceDetector   │                                           │   │
84: │  │   │  (OpenCV DNN)    │                                           │   │
85: │  │   │                  │                                           │   │
86: │  │   │ Input: Image     │                                           │   │
87: │  │   │ Output: BBoxes   │                                           │   │
88: │  │   └────────┬─────────┘                                           │   │
89: │  │            │                                                     │   │
90: │  │            ▼                                                     │   │
91: │  │   ┌─────────────────────────────────────────────────────┐        │   │
92: │  │   │           Embedding Extractor                        │        │   │
93: │  │   │  ┌─────────────────────┐  ┌─────────────────────┐   │        │   │
94: │  │   │  │  ArcFace (Default)  │  │  FaceNet (Option)  │   │        │   │
95: │  │   │  │  ONNX / ResNet100   │  │  PyTorch / ResNet18│   │        │   │
96: │  │   │  │  512-dimensional    │  │  128-dimensional   │   │        │   │
97: │  │   │  └─────────────────────┘  └─────────────────────┘   │        │   │
98: │  │   │                                                      │        │   │
99: │  │   │ Input: Face ROI                                      │        │   │
100: │  │   │ Output: 512-dim or 128-dim embedding                 │        │   │
101: │  │   └────────────────────────────┬──────────────────────────┘        │   │
102: │  │                                │                                     │   │
103: │  │                                ▼                                     │   │
104: │  │   ┌──────────────────┐                                           │   │
105: │  │   │ Similarity       │                                           │   │
106: │  │   │ Comparator       │                                           │   │
107: │  │   │                  │                                           │   │
108: │  │   │ Input: 2 emb.   │                                           │   │
109: │  │   │ Output: Score    │                                           │   │
110: │  │   └──────────────────┘                                           │   │
111: │  │                                                                  │   │
112: │  └──────────────────────────────────────────────────────────────────┘   │
113: │                                    │                                     │
114: │                                    ▼                                     │
115: │  ┌──────────────────────────────────────────────────────────────────┐   │
116: │  │                     REFERENCE STORAGE                             │   │
117: │  ├──────────────────────────────────────────────────────────────────┤   │
118: │  │   ReferenceImageManager                                          │   │
119: │  │   • Stores references in reference_images/embeddings.json        │   │
120: │  │   • Metadata: id, path, consent, timestamp                     │   │
121: │  │   • Embeddings: 512-dim (ArcFace) or 128-dim (FaceNet)        │   │
122: │  │   • Persistence: Saved to JSON on add/remove                     │   │
123: │  │                                                                  │   │
124: │  │   HumanReviewInterface                                           │   │
125: │  │   • Side-by-side comparison display                             │   │
126: │  │   • Review history tracking                                      │   │
127: │  └──────────────────────────────────────────────────────────────────┘   │
128: │                                                                          │
129: └─────────────────────────────────────────────────────────────────────────┘
130: ```
131: 
132: ---
133: 
134: ## Data Flow
135: 
136: ```
137: ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
138: │   Image     │ →  │   Detect    │ →  │   Extract   │ →  │  Compare    │
139: │   Upload    │    │   Faces     │    │   Embedding │    │  References │
140: └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
141: ```
142: 
143: ### Step 1: Image Upload
144: - Frontend reads image as Base64
145: - POST /api/detect with Base64 image
146: 
147: ### Step 2: Face Detection
148: - OpenCV DNN with Caffe model (`res10_300x300_ssd_iter_140000.caffemodel`)
149: - Returns bounding boxes (x, y, w, h) for each face
150: 
151: ### Step 3: Embedding Extraction (ArcFace vs FaceNet)
152: 
153: #### ArcFace (Default - Recommended)
154: - **Model**: ONNX format ResNet100
155: - **Dimension**: 512-dimensional
156: - **L2 Normalized**: Yes
157: - **Discrimination**: Excellent - different people show <30% similarity
158: - **Thresholds**:
159:   - ≥70% = Very High - Likely same person
160:   - 45-70% = High - Possibly same person
161:   - 30-45% = Moderate - Human review recommended
162:   - <30% = Insufficient - Likely different people
163: 
164: #### FaceNet (Optional - Legacy)
165: - **Model**: PyTorch ResNet18
166: - **Dimension**: 128-dimensional
167: - **L2 Normalized**: Yes
168: - **Discrimination**: Poor - different people show ~65-70% similarity
169: - **Note**: Use `USE_FACENET=true` to enable
170: 
171: ### Step 4: Comparison
172: - Cosine similarity: `dot(a, b) / (|a| * |b|)`
173: - Confidence bands (ArcFace):
174:   - Very High (>0.7): High confidence match
175:   - High (0.45-0.7): Moderate confidence
176:   - Moderate (0.3-0.45): Low confidence, human review required
177:   - Insufficient (<0.3): Likely different people
178: 
179: ---
180: 
181: ## Core Components
182: 
183: ### 1. FaceDetector (`src/detection/__init__.py`)
184: 
185: **Purpose**: Detect faces in images using OpenCV DNN
186: 
187: **Model**: Caffe-based SSD detector
188: - Config: `deploy.prototxt.txt`
189: - Weights: `res10_300x300_ssd_iter_140000.caffemodel`
190: 
191: **Detection Methods**:
192: - `detect_faces(image)` → List[Tuple[int, int, int, int]]
193: - `detect_faces_with_confidence(image)` → List[Tuple[Tuple, float]]
194: - `detect_eyes(face_image)` → List[Tuple[int, int, int, int]]
195: - `estimate_landmarks(face_image, face_box)` → Dict with 15 keypoints + 468 full landmarks
196: - `compute_alignment(face_image, landmarks)` → Dict with pitch, yaw, roll
197: - `compute_quality_metrics(face_image, face_box)` → Dict[brightness, contrast, sharpness, eye_detection, centering, overall]
198: - `compute_lbp_descriptor(face_image)` → np.ndarray (256-dim histogram) - NEW
199: - `compute_facial_asymmetry(landmarks)` → Dict with asymmetry features - NEW
200: - `normalize_face_with_mesh(face_image, mesh_landmarks)` → np.ndarray - NEW (3D mesh-based alignment)
201: 
202: **Visualization Methods**:
203: - `visualize_detection(image, faces)` → Bounding boxes with confidence
204: - `visualize_extraction(image, faces)` → Face ROI extraction
205: - `visualize_landmarks(face_image, landmarks)` → 15 keypoints + facial regions
206: - `visualize_3d_mesh(face_image)` → 478-point mesh (MediaPipe or fallback)
207: - `visualize_alignment(face_image, landmarks, alignment)` → Orientation indicator
208: - `visualize_saliency(face_image)` → Attention/gradient visualization
209: - `visualize_biometric_capture(image, faces)` → Biometric capture overview
210: - `visualize_multiscale(face_image)` → Multi-scale detection
211: - `visualize_quality(face_image, face_box)` → Quality metrics overlay
212: - `visualize_confidence_levels(face_image, similarity)` → Confidence bands
213: 
214: ### 2. Embedding Extractor (`src/embedding/`)
215: 
216: #### ArcFaceEmbeddingExtractor (`arcface_extractor.py`)
217: 
218: **Purpose**: Extract 512-dimensional face embeddings using ONNX Runtime
219: 
220: **Architecture**:
221: - Backbone: ResNet100 (ONNX format)
222: - Embedding: 512-dimensional
223: - L2 normalized
224: 
225: **Core Methods**:
226: - `extract_embedding(face_image)` → np.ndarray (512,)
227: - `preprocess(face_image)` → np.ndarray (112, 112)
228: - `get_activations(face_image)` → Dict[str, np.ndarray] (placeholder for ONNX)
229: - `get_embedding_info()` → Dict with model info
230: 
231: **Visualization Methods**:
232: - `visualize_embedding(embedding)` → (np.ndarray, Dict) - Bar chart of 512 values
233: - `visualize_similarity_matrix(query, references, ids)` → (np.ndarray, Dict)
234: - `visualize_similarity_result(query, ref, similarity)` → np.ndarray
235: - `test_robustness(face_image)` → (np.ndarray, Dict) - Noise robustness test
236: 
237: **Note**: ArcFace ONNX model doesn't expose internal layers, so activations visualization uses placeholder that shows useful info.
238: 
239: #### FaceNetEmbeddingExtractor (`__init__.py`)
240: 
241: **Purpose**: Extract 128-dimensional face embeddings (legacy)
242: 
243: **Architecture**:
244: - Backbone: torchvision ResNet18 (pretrained on ImageNet)
245: - Custom head: FC(512→512) → BatchNorm → ReLU → Dropout → FC(512→128) → BatchNorm
246: - L2 normalization
247: 
248: **Core Methods**:
249: - `extract_embedding(face_image)` → np.ndarray (128,)
250: - `preprocess(face_image)` → torch.Tensor
251: - `extract_embeddings(face_images)` → List[np.ndarray]
252: - `get_activations(face_image)` → Dict[str, np.ndarray] (11 layers!)
253: 
254: **Visualization Methods**:
255: - `visualize_embedding(embedding)` → (np.ndarray, Dict) - Bar chart of 128 values
256: - `visualize_similarity_matrix(query, references, ids)` → (np.ndarray, Dict)
257: - `visualize_similarity_result(query, ref, similarity)` → np.ndarray
258: - `visualize_activations(face_image, max_channels)` → CNN layer activations grid
259: - `visualize_feature_maps(face_image)` → Feature map visualization
260: - `test_robustness(face_image)` → (np.ndarray, Dict) - Noise robustness test
261: 
262: ### 3. SimilarityComparator (`src/embedding/__init__.py`)
263: 
264: **Purpose**: Compare embeddings and return similarity scores
265: 
266: **Methods**:
267: - `cosine_similarity(embedding1, embedding2)` → float
268: - `compare_embeddings(query, references, ids)` → List[Tuple[str, float]]
269: - `get_confidence_band(similarity, model='arcface')` → str
270: - `compute_pose_weight(pose1, pose2)` → float - NEW (pose-aware matching)
271: - `lbp_similarity(lbp1, lbp2)` → float - NEW (texture matching)
272: - `asymmetry_similarity(asym1, asym2)` → float - NEW (uniqueness analysis)
273: - `compute_multi_pose_score(query_emb, pose_embeddings)` → Tuple[float, Dict] - NEW (multi-pose matching)
274: 
275: ### 4. ReferenceImageManager (`src/reference/__init__.py`)
276: 
277: **Purpose**: Manage reference images and their embeddings
278: 
279: **Features**:
280: - Stores references in `reference_images/embeddings.json`
281: - Extracts REAL embeddings (not random!)
282: - Metadata: id, path, consent info, timestamp
283: - Auto-saves on add/remove
284: 
285: **Methods**:
286: - `__init__(reference_dir, embedding_extractor, detector)`
287: - `add_reference_image(image_path, reference_id, metadata)` → (bool, np.ndarray)
288: - `get_reference_embeddings()` → (List[np.ndarray], List[str])
289: - `list_references()` → List[dict]
290: - `remove_reference(reference_id)` → bool
291: 
292: ### 5. HumanReviewInterface (`src/reference/__init__.py`)
293: 
294: **Purpose**: Human-in-the-loop review workflow
295: 
296: **Features**:
297: - Side-by-side comparison display
298: - Confidence-based decision making
299: - Review history tracking
300: 
301: ---
302: 
303: ## Model Files
304: 
305: | File | Size | Purpose |
306: |------|------|---------|
307: | `deploy.prototxt.txt` | 28KB | OpenCV DNN config |
308: | `res10_300x300_ssd_iter_140000.caffemodel` | 10MB | Face detection weights |
309: | `arcface_model.onnx` | ~117MB | ArcFace embedding extractor (ONNX) |
310: | torchvision ResNet18 | ~44MB | PyTorch embedding backbone (FaceNet) |
311: 
312: ---
313: 
314: ## API Reference
315: 
316: ### GET /api/embedding-info
317: 
318: **Response**:
319: ```json
320: {
321:   "model": "ArcFaceEmbeddingExtractor",
322:   "dimension": 512,
323:   "discrimination": "Excellent - different people show <30% similarity"
324: }
325: ```
326: 
327: ### POST /api/detect
328: 
329: **Request**:
330: ```json
331: {
332:   "image": "base64_encoded_image_string"
333: }
334: ```
335: 
336: **Response**:
337: ```json
338: {
339:   "success": true,
340:   "count": 2,
341:   "faces": [
342:     {"id": 0, "bbox": [x, y, w, h], "thumbnail": "base64..."}
343:   ],
344:   "visualizations": {
345:     "detection": "base64...",
346:     "extraction": "base64...",
347:     "biometric": "base64..."
348:   }
349: }
350: ```
351: 
352: ### POST /api/extract
353: 
354: **Request**:
355: ```json
356: {
357:   "face_id": 0
358: }
359: ```
360: 
361: **Response**:
362: ```json
363: {
364:   "success": true,
365:   "embedding_size": 512,
366:   "model": "ArcFaceEmbeddingExtractor",
367:   "embedding_mean": 0.0321,
368:   "embedding_std": 0.0452,
369:   "visualizations": {...},
370:   "visualization_data": {...}
371: }
372: ```
373: 
374: ### POST /api/add-reference
375: 
376: **Request**:
377: ```json
378: {
379:   "image": "base64_encoded_image_string",
380:   "name": "reference_name"
381: }
382: ```
383: 
384: **Response**:
385: ```json
386: {
387:   "success": true,
388:   "reference": {
389:     "id": 0,
390:     "name": "reference_name",
391:     "embedding": [...512 values...],
392:     "thumbnail": "base64..."
393:   },
394:   "count": 1
395: }
396: ```
397: 
398: ### POST /api/compare
399: 
400: **Response**:
401: ```json
402: {
403:   "success": true,
404:   "results": [
405:     {
406:       "id": 0,
407:       "name": "reference_name",
408:       "similarity": 0.75,
409:       "confidence": "High confidence",
410:       "verdict": "Likely same person",
411:       "thumbnail": "base64..."
412:     }
413:   ],
414:   "best_match": {...},
415:   "similarity_viz": "base64...",
416:   "similarity_data": {...}
417: }
418: ```
419: 
420: ---
421: 
422: ## CRITICAL IMPLEMENTATION NOTES
423: 
424: ### ArcFace ONNX Model (No Layer Access)
425: 
426: ArcFace uses ONNX Runtime which doesn't expose internal layer activations like PyTorch. This means:
427: 
428: **For ArcFace**:
429: - `get_activations()` returns placeholder with model info
430: - Visualizations show useful info instead of raw CNN activations
431: - `visualize_activations()` shows embedding channel groups
432: 
433: **For FaceNet** (if enabled):
434: - Full layer activations available (11 layers)
435: - Raw CNN feature maps accessible
436: 
437: ### Dynamic Array Sizes (Fixed Bug!)
438: 
439: When visualization output size depends on input count, use dynamic allocation:
440: 
441: ```python
442: # WRONG - Hardcoded size causes broadcast error with many references
443: output = np.zeros((150, 300, 3), dtype=np.uint8)
444: output[:n * cell_size, :n * cell_size] = matrix_colored  # FAILS!
445: 
446: # RIGHT - Dynamic sizing based on input count
447: output_size = max(150, n * cell_size)
448: output = np.zeros((output_size, output_size, 3), dtype=np.uint8)
449: output.fill(245)
450: resized = cv2.resize(matrix_colored, (n * cell_size, n * cell_size))
451: output[:resized.shape[0], :resized.shape[1]] = resized  # Safe!
452: ```
453: 
454: ### Real Embeddings (Fixed Bug!)
455: 
456: Never use random values for embeddings:
457: 
458: ```python
459: # WRONG - Random embeddings cause incorrect comparisons
460: embedding = np.random.rand(512)
461: 
462: # RIGHT - Extract real embeddings from actual images
463: if self.embedding_extractor is not None:
464:     faces = self.detector.detect_faces(image_array)
465:     if faces:
466:         x, y, w, h = faces[0]
467:         face_roi = image_array[y:y+h, x:x+w]
468:         embedding = self.embedding_extractor.extract_embedding(face_roi)
469: ```
470: 
471: ---
472: 
473: ## ArcFace vs FaceNet Comparison
474: 
475: | Metric | ArcFace (Default) | FaceNet (Optional) |
476: |--------|-------------------|-------------------|
477: | **Dimension** | 512 | 128 |
478: | **Backbone** | ResNet100 (ONNX) | ResNet18 (PyTorch) |
479: | **Discrimination** | Excellent | Poor |
480: | **Same Person** | ~70-85% | ~85-99% |
481: | **Different Person** | <30% | ~65-70% |
482: | **False Positive Risk** | Low | High |
483: | **Inference Speed** | Fast (ONNX) | Slower (PyTorch) |
484: 
485: **Why ArcFace is Default**:
486: FaceNet showed 65-70% similarity for different people - this caused false positives! ArcFace correctly shows <30% for different people, making it much safer for NGO use cases.
487: 
488: ---
489: 
490: ## Testing
491: 
492: ### End-to-End Test
493: ```bash
494: python test_e2e_pipeline.py
495: ```
496: 
497: Tests:
498: 1. Face Detection Pipeline
499: 2. Embedding Extraction Pipeline
500: 3. Reference Manager with Real Embeddings
501: 4. Same Image Similarity (~100%)
502: 5. Different Images Similarity (~9-25% with ArcFace)
503: 6. Full Reference Comparison Pipeline
504: 
505: ### Unit Tests
506: ```bash
507: python -m unittest discover tests/
508: ```
509: 
510: ---
511: 
512: ## Performance
513: 
514: | Operation | Time Complexity | Notes |
515: |-----------|-----------------|-------|
516: | Face Detection | O(n) | n = image pixels |
517: | ArcFace Embedding | O(1) | Fixed network size (ONNX) |
518: | FaceNet Embedding | O(1) | Fixed network size (PyTorch) |
519: | Similarity Comparison | O(m) | m = number of references |
520: | Embedding Storage | O(k) | k = number of stored refs |
521: 
522: ---
523: 
524: ## Complete API Reference
525: 
526: ### All Endpoints
527: 
528: | Method | Endpoint | Description |
529: |--------|----------|-------------|
530: | GET | `/api/health` | Health check |
531: | GET | `/api/embedding-info` | Model info (ArcFace/FaceNet) |
532: | GET | `/api/diagnostics` | System diagnostics (MediaPipe, models) |
533: | POST | `/api/detect` | Face detection |
534: | POST | `/api/extract` | Embedding extraction (both models) |
535: | POST | `/api/add-reference` | Add reference image |
536: | POST | `/api/add-reference-pose/<id>` | Add pose variant to reference - NEW |
537: | GET | `/api/references` | List references |
538: | DELETE | `/api/references/<id>` | Remove reference |
539: | POST | `/api/compare` | Compare embeddings (enhanced) |
540: | GET | `/api/visualizations/<type>` | Get visualization |
541: | GET | `/api/visualizations/<type>/reference/<id>` | Get ref visualization |
542: | GET | `/api/quality` | Quality metrics |
543: | GET | `/api/eyewear` | Eyewear detection |
544: | POST | `/api/clear` | Clear session |
545: | GET | `/api/status` | Debug server state |
546: 
547: ### Enhanced Comparison (Version 0.5.0)
548: 
549: The `/api/compare` endpoint now returns with all new features:
550: 
551: | Field | Description |
552: |-------|-------------|
553: | `arcface_similarity` | ArcFace cosine similarity (0-1) |
554: | `facenet_similarity` | FaceNet cosine similarity (0-1) |
555: | `landmark_similarity` | Landmark geometry similarity (0-1) |
556: | `activation_similarity` | Neural network activation similarity (0-1) |
557: | `pose_weight` | NEW - Pose-aware weight adjustment (0.95-1.0) |
558: | `lbp_similarity` | NEW - LBP texture similarity (lighting-invariant) |
559: | `asymmetry_similarity` | NEW - Facial asymmetry similarity (uniqueness) |
560: | `normalized_similarity` | NEW - 3D mesh-normalized embedding similarity |
561: | `multi_pose_score` | NEW - Best match across multiple pose variants |
562: | `multi_pose_used` | NEW - Whether multiple poses were available |
563: | `final_score` | Weighted combination (see below) |
564: | `status` | "match", "possible", or "no_match" |
565: | `match_label` | "Full Match", "Possible Match", "No Match" |
566: | `reasons` | List of match reasoning strings |
567: 
568: #### New Weights (v0.5.0)
569: 
570: | Factor | Weight | Purpose |
571: |--------|--------|---------|
572: | ArcFace | 40% | Primary embedding (reduced from 60%) |
573: | 3D Normalized | 15% | Handle extreme angles (NEW) |
574: | Multi-Pose Best | 10% | Best pose variant match (NEW) |
575: | LBP Texture | 8% | Lighting-invariant matching (NEW) |
576: | Asymmetry | 7% | Uniqueness analysis (NEW) |
577: | Landmark Geometry | 8% | Geometric consistency |
578: | FaceNet | 10% | Secondary embedding |
579: | Activation | 1% | Neural patterns |
580: | Quality | 1% | Reliability factor |
581: 
582: **Pose Weight Modifier**: 0.95-1.0x applied to all scores based on pose similarity
583: 
584: ### Visualization Types (17 + 9 Test = 26 total)
585: 
586: | Type | Source | Description |
587: |------|--------|-------------|
588: | `detection` | FaceDetector | Bounding boxes with confidence |
589: | `extraction` | FaceDetector | Face ROI extraction |
590: | `preprocessing` | FaceDetector | CLAHE enhancement |
591: | `landmarks` | FaceDetector | **468 MediaPipe landmarks** |
592: | `mesh3d` | FaceDetector | 478-point 3D mesh |
593: | `alignment` | FaceDetector | Pitch/yaw/roll orientation |
594: | `saliency` | FaceDetector | Attention visualization |
595: | `activations` | EmbeddingExtractor | CNN activations (FaceNet) |
596: | `features` | EmbeddingExtractor | Feature map grid |
597: | `multiscale` | FaceDetector | Multi-scale detection |
598: | `confidence` | FaceDetector | Quality metrics overlay |
599: | `embedding` | EmbeddingExtractor | 512-dim or 128-dim bar chart |
600: | `similarity` | EmbeddingExtractor | Similarity result bar |
601: | `robustness` | EmbeddingExtractor | Noise robustness test |
602: | `biometric` | FaceDetector | Biometric capture overview |
603: | `eyewear` | FaceDetector | Eyewear detection visualization |
604: | `asymmetry` | FaceDetector | NEW - Uniqueness analysis |
605: | `texture` | FaceDetector | NEW - LBP texture visualization |
606: | `normalized` | FaceDetector | NEW - 3D normalized face |
607: 
608: ### Test Tabs (9)
609: 
610: | Tab | Description |
611: |-----|-------------|
612: | API Health | Health check status |
613: | Detection | Face detection info |
614: | Extraction | Embedding extraction info |
615: | References | Reference management |
616: | Multi-Match | Multi-reference matching |
617: | Pose | Pose detection status |
618: | Eyewear | Eyewear detection |
619: | Viz Types | Available visualizations |
620: | Session | Session management |
621: 
622: ---
623: 
624: ## Edge Case Handling
625: 
626: The system handles various edge cases gracefully:
627: 
628: - **Empty/black images**: Returns 0 faces detected
629: - **Very small images (1x1)**: Extracts embedding (upsamples)
630: - **None inputs**: Returns placeholder/error image
631: - **NaN/Inf values**: Handled with safe defaults
632: - **Zero embeddings**: Returns 0.0 similarity
633: - **Empty reference list**: Returns empty results
634: - **Many references (50+)**: Dynamic array sizing
635: - **Long Unicode names**: Truncated in display
636: - **Zero-sized face boxes**: Returns error in quality metrics
637: 
638: Run `python test_edge_cases.py` to verify all edge cases.
639: 
640: ---
641: 
642: ## New Features (v0.5.0 - February 15, 2026)
643: 
644: ### 1. 3D Mesh Normalization
645: **Purpose**: Handle extreme angles using 468-point MediaPipe mesh
646: 
647: Method: `normalize_face_with_mesh(face_image, mesh_landmarks)`
648: - Uses MediaPipe's 468-point facial mesh
649: - Computes eye centers from multiple mesh points (more accurate than 2-point)
650: - Rotates face to align eyes horizontally
651: - Extracts embedding from aligned face for better matching
652: 
653: **Weight in comparison**: 15%
654: 
655: ### 2. LBP Texture Features
656: **Purpose**: Lighting-invariant matching
657: 
658: Method: `compute_lbp_descriptor(face_image)`
659: - Extracts Local Binary Pattern histogram (256 bins)
660: - Compares texture patterns between faces
661: - Robust to lighting variations
662: 
663: Method: `lbp_similarity(lbp1, lbp2)`
664: - Compares LBP histograms using intersection
665: - Returns 0-1 similarity score
666: 
667: **Weight in comparison**: 8%
668: 
669: ### 3. Facial Asymmetry Analysis
670: **Purpose**: Uniqueness analysis
671: 
672: Method: `compute_facial_asymmetry(landmarks)`
673: - Computes distances between bilateral facial points
674: - Measures left-right asymmetry
675: - Each face has unique asymmetry pattern
676: 
677: Method: `asymmetry_similarity(asym1, asym2)`
678: - Compares asymmetry features between two faces
679: - Returns 0-1 similarity score
680: 
681: **Weight in comparison**: 7%
682: 
683: ### 4. Pose-Aware Weight
684: **Purpose**: Adjust similarity based on pose difference
685: 
686: Method: `compute_pose_weight(pose1, pose2)`
687: - Computes pose difference (yaw + pitch)
688: - If poses are similar (< 15°): weight = 1.0
689: - If poses differ (15-30°): weight = 0.98
690: - If poses differ greatly (> 30°): weight = 0.95
691: 
692: **Modifier**: 0.95-1.0x applied to final score
693: 
694: ### 5. Multi-Pose Enrollment
695: **Purpose**: Store and match against multiple poses of same person
696: 
697: New endpoint: `POST /api/add-reference-pose/<ref_id>`
698: - Adds additional pose variant to existing reference
699: - Automatically categorizes pose (frontal, left, right, up, down)
700: - Stores in reference's `poses` dictionary
701: 
702: Method: `compute_multi_pose_score(query_emb, pose_embeddings)`
703: - Compares query against all pose variants
704: - Returns best match score
705: 
706: **Weight in comparison**: 10%
707: 
708: ---
709: 
710: ## Storage Format
711: 
712: Reference storagejson
713: {
714:   "id": 0,
715:   " now includes:
716: 
717: ```name": "John Doe",
718:   "embedding": {"arcface": [...], "facenet": [...]},
719:   "lbp_histogram": [...],
720:   "asymmetry": {"left_eye_right_eye_dist": 10.5, ...},
721:   "normalized_embedding": [...],
722:   "poses": {
723:     "frontal": {"embedding": {...}, "yaw": 0, "pitch": 0},
724:     "left": {"embedding": {...}, "yaw": -25, "pitch": 0},
725:     "right": {"embedding": {...}, "yaw": 25, "pitch": 0}
726:   },
727:   "pose": {"yaw": 0, "pitch": 0, "roll": 0},
728:   "pose_category": "frontal"
729: }
730: ```
731: 
732: ---
733: 
734: ## Testing New Features
735: 
736: Run `python test_edge_cases.py` to verify:
737: 
738: - `test_lbp_descriptor()` - LBP extraction and similarity
739: - `test_asymmetry_features()` - Asymmetry computation
740: - `test_pose_weight()` - Pose weight calculation
741: - `test_mesh_normalization()` - 3D mesh normalization
742: - `test_multi_pose_score()` - Multi-pose comparison
743: 
744: ---
745: 
746: ## MANTAX Branding
747: 
748: The Electron UI includes MANTAX branding:
749: - **Navbar**: White background with subtle border
750: - **Logo**: SVG with red (#D20A11) and white colors
751: - **Tagline**: "Ihrem Partner für Autokrane und Schwerlastlogistik" (right side)
752: - **Compact Design**: 16px padding, clean typography
753: 
754: ---
755: 
756: ## Ethical Design
757: 
758: This system is built with ethical principles:
759: 
760: 1. **Consent-Based**: All images require documented consent
761: 2. **Human Oversight**: No automated decisions - human review required
762: 3. **Uncertainty Handling**: Confidence bands instead of binary decisions
763: 4. **Privacy Protection**: Non-reversible embeddings only
764: 5. **Documentation**: Complete audit trail of all operations
765: 
766: ---
767: 
768: ## Support
769: 
770: - **Documentation**: See all .md files in project root
771: - **Architecture**: PROJECT_STRUCTURE.md
772: - **Development Log**: DEVELOPMENT_LOG.md
773: - **Testing**: test_e2e_pipeline.py
774: - **Code Review**: CONTEXT.md
775: 
776: ---
777: 
778: *Architecture documentation updated: February 15, 2026*
779: *Version 0.5.0 - Includes LBP texture, facial asymmetry, 3D mesh normalization, multi-pose enrollment, pose-aware matching*
780: 

(End of file - total 780 lines)
</content>
```

---

## Assistant (Build · minimax-m2.5-free · 7.6s)

_Thinking:_

This is a lot of content. Let me continue reading the remaining .md files.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/WORKFLOW_PROMPT.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/WORKFLOW_PROMPT.md</path>
<type>file</type>
<content>1: # Comprehensive System Prompt - READ BEFORE EVERY TASK
2: 
3: You are working on the **NGO Facial Image Analysis System** - a face recognition application with Flask API + Electron UI.
4: 
5: ---
6: 
7: ## ⚠️ MANDATORY WORKFLOW - NEVER DEVIATE FROM THIS
8: 
9: ### For EVERY code change, you MUST follow this exact sequence:
10: 
11: 1. **Read relevant files first** - Understand the existing code before making changes
12: 2. **Make your code changes**
13: 3. **Check syntax** - Run `python -m py_compile <file>` for Python files
14: 4. **Test locally** - Run the relevant test(s)
15: 5. **Run ALL tests** (non-negotiable):
16:    ```bash
17:    python test_e2e_pipeline.py
18:    python test_edge_cases.py
19:    python test_frontend_integration.py
20:    ```
21: 6. **Verify startup works** - Run `./start.sh` and confirm it works
22: 7. **Say "FINISHED"** - ONLY after ALL tests pass
23: 
24: **NEVER say "done", "finished", or "ready to commit" until step 6 is complete.**
25: 
26: ---
27: 
28: ## PROJECT STRUCTURE - DETAILED VIEW
29: 
30: ```
31: face_recognition_npo/
32: ├── ROOT FILES
33: │   ├── api_server.py           # Flask API - main entry point (25+ endpoints)
34: │   ├── start.sh                # Startup script (kills servers, compiles SCSS, starts app)
35: │   ├── config_template.py      # Configuration template
36: │   ├── setup.py                # Setup script
37: │   └── visualize_biometric.py # Standalone visualization script
38: │
39: ├── SRC/                        # Core ML modules
40: │   ├── detection/
41: │   │   ├── __init__.py        # FaceDetector class (OpenCV DNN + MediaPipe)
42: │   │   └── preprocessing.py    # ImagePreprocessor
43: │   │
44: │   ├── embedding/
45: │   │   ├── __init__.py        # FaceNetEmbeddingExtractor, SimilarityComparator
46: │   │   └── arcface_extractor.py # ArcFaceEmbeddingExtractor
47: │   │
48: │   └── reference/
49: │       └── __init__.py         # ReferenceImageManager, HumanReviewInterface
50: │
51: ├── ELECTRON-UI/                # Frontend
52: │   ├── index.html             # Main UI HTML
53: │   ├── main.js               # Electron main process
54: │   ├── preload.js            # Electron preload script
55: │   ├── package.json          # NPM config (has sass)
56: │   │
57: │   ├── renderer/
58: │   │   └── app.js            # Frontend JavaScript (calls API)
59: │   │
60: │   └── styles/
61: │       ├── design-system.scss # SCSS source
62: │       └── design-system.css  # Compiled CSS
63: │
64: ├── GUI/                        # Alternative GUI (not main)
65: │   ├── facial_analysis_gui.py
66: │   └── user_friendly_gui.py
67: │
68: ├── UTILS/
69: │   └── webcam.py             # Webcam utilities
70: │
71: ├── TESTS/                     # Unit tests
72: │   ├── test_detection.py
73: │   ├── test_embedding.py
74: │   ├── test_comparison.py
75: │   ├── test_reference.py
76: │   └── test_review.py
77: │
78: ├── TEST FILES (root)          # Integration tests
79: │   ├── test_e2e_pipeline.py
80: │   ├── test_edge_cases.py
81: │   ├── test_frontend_integration.py
82: │   ├── test_eyewear.py
83: │   └── test_api_endpoints.py
84: │
85: ├── REFERENCE_IMAGES/
86: │   ├── embeddings.json        # Stored references
87: │   └── README.md
88: │
89: ├── TEST_IMAGES/               # Test images
90: │   ├── test_subject.jpg
91: │   └── reference_subject.jpg
92: │
93: ├── _EXAMPLES/                 # Example scripts
94: │   ├── basic_usage.py
95: │   ├── webcam_demo.py
96: │   └── reference_images/
97: │
98: └── DOCUMENTATION (*.md)
99:     ├── README.md               # Main docs
100:     ├── CONTEXT.md             # Critical rules
101:     ├── ARCHITECTURE.md        # System design
102:     ├── DEVELOPMENT_LOG.md     # History
103:     ├── WORKFLOW_PROMPT.md     # This file
104:     └── ... (15+ more .md files)
105: ```
106: 
107: ---
108: 
109: ## FILE INTERACTIONS - HOW THINGS CONNECT
110: 
111: ### Backend Flow (api_server.py)
112: 
113: ```
114: api_server.py (Flask)
115:     │
116:     ├── imports → src.detection.FaceDetector
117:     │            src.embedding.FaceNetEmbeddingExtractor
118:     │            src.embedding.ArcFaceEmbeddingExtractor
119:     │            src.embedding.SimilarityComparator
120:     │            src.reference.ReferenceImageManager
121:     │
122:     └── defines endpoints → /api/health, /api/detect, /api/extract,
123:                             /api/add-reference, /api/references, /api/compare, etc.
124: ```
125: 
126: ### Frontend Flow (electron-ui/)
127: 
128: ```
129: index.html
130:     │
131:     ├── loads → app.js (JavaScript)
132:     │        design-system.css (compiled SCSS)
133:     │
134:     └── app.js calls → http://localhost:3000/api/* (Flask endpoints)
135: ```
136: 
137: ### Test Files → Source Files
138: 
139: ```
140: test_e2e_pipeline.py
141:     │
142:     ├── imports → src.detection.FaceDetector
143:     │            src.embedding.FaceNetEmbeddingExtractor
144:     │            src.embedding.SimilarityComparator
145:     │            src.reference.ReferenceImageManager
146:     │
147:     └── Tests → Full pipeline (detect → extract → compare)
148: 
149: test_edge_cases.py
150:     │
151:     └── Tests → Edge cases (None inputs, empty lists, boundaries)
152: 
153: test_frontend_integration.py
154:     │
155:     └── Tests → API endpoints via HTTP requests
156: 
157: gui/facial_analysis_gui.py
158:     │
159:     └── Alternative GUI that uses same src modules
160: ```
161: 
162: ### Import Dependency Graph
163: 
164: ```
165: api_server.py
166:     ├── src.detection (FaceDetector)
167:     ├── src.embedding (FaceNetEmbeddingExtractor, SimilarityComparator)
168:     └── src.reference (ReferenceImageManager)
169: 
170: test_e2e_pipeline.py
171:     ├── src.detection
172:     ├── src.embedding
173:     └── src.reference
174: 
175: gui/facial_analysis_gui.py
176:     ├── src.detection
177:     ├── src.embedding
178:     └── src.reference
179: 
180: tests/test_*.py
181:     └── (each imports relevant src modules)
182: ```
183: 
184: ---
185: 
186: ## CRITICAL RULES - NEVER FORGET
187: 
188: ### Rule 1: Always Use Project Virtual Environment
189: ```bash
190: # WRONG - uses system Python
191: python api_server.py
192: 
193: # CORRECT - uses project venv
194: source venv/bin/activate
195: python api_server.py
196: 
197: # Or use start.sh which does this automatically
198: ./start.sh
199: ```
200: 
201: ### Rule 2: Always Check start.sh Works
202: - start.sh MUST use `SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"` to resolve paths correctly
203: - start.sh MUST compile SCSS before starting
204: - NEVER use `$(dirname "$0")` alone - it returns "." when script is run with "./"
205: 
206: ### Rule 3: SCSS Compilation
207: - SCSS source: `electron-ui/styles/design-system.scss`
208: - Compiled CSS: `electron-ui/styles/design-system.css`
209: - Always compile after changing SCSS: `npm run scss`
210: - start.sh should auto-compile SCSS
211: 
212: ### Rule 4: Check for Duplicate Code
213: Before submitting, always check:
214: ```bash
215: # Check for duplicate function definitions
216: grep -n "def " <file>
217: 
218: # Check for duplicate selectors in CSS
219: grep -n "\.class" <file>
220: ```
221: 
222: ### Rule 5: Count Parentheses and Brackets
223: Before any edit submission:
224: - Count opening and closing: `(`, `)`, `[`, `]`, `{`, `}`
225: - Ensure each `return` statement has a complete expression
226: 
227: ---
228: 
229: ## TEST COMMANDS
230: 
231: ```bash
232: # Activate venv first
233: source venv/bin/activate
234: 
235: # Run all tests (REQUIRED after any code change)
236: python test_e2e_pipeline.py
237: python test_edge_cases.py
238: python test_frontend_integration.py
239: 
240: # Run specific test
241: python -m pytest tests/
242: python test_eyewear.py
243: ```
244: 
245: ---
246: 
247: ## COMMON MISTAKES TO AVOID
248: 
249: | Mistake | Solution |
250: |---------|----------|
251: | Using system Python instead of venv | Always `source venv/bin/activate` first |
252: | Forgetting to compile SCSS | Run `npm run scss` or use `./start.sh` |
253: | start.sh path issues | Use `SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"` |
254: | Saying "done" before testing | ALWAYS run tests first |
255: | Breaking CSS with SCSS nesting | Verify compiled CSS matches original |
256: | Blocking UI with async/await | Use `.catch()` for fire-and-forget calls |
257: | Missing HTML-JS cross-check | Verify all onclick handlers exist in JS |
258: | Wrong import paths | Use `from src.detection import ...` not `from .src...` |
259: 
260: ---
261: 
262: ## CURRENT CONFIGURATION
263: 
264: - **Backend**: Flask on port 3000
265: - **Frontend**: Electron desktop app
266: - **ML Models**: ArcFace (512-dim) + FaceNet (128-dim)
267: - **Weights**: ArcFace 60%, FaceNet 20%, Landmarks 15%, Activation 1%, Quality 1%
268: - **Testing**: 3 test suites (E2E, Edge Cases, Frontend)
269: - **Persistence**: JSON file (`reference_images/embeddings.json`)
270: 
271: ---
272: 
273: ## API ENDPOINTS (from api_server.py)
274: 
275: | Method | Endpoint | Description |
276: |--------|----------|-------------|
277: | GET | `/api/health` | Health check |
278: | GET | `/api/embedding-info` | Model info |
279: | POST | `/api/detect` | Face detection |
280: | POST | `/api/extract` | Embedding extraction |
281: | POST | `/api/add-reference` | Add reference |
282: | GET | `/api/references` | List references |
283: | DELETE | `/api/references/<id>` | Remove reference |
284: | POST | `/api/compare` | Compare embeddings |
285: | POST | `/api/clear` | Clear session |
286: | GET | `/api/visualizations/<type>` | Get visualization |
287: | POST | `/api/eyewear` | Eyewear detection |
288: 
289: ---
290: 
291: ## VERIFICATION CHECKLIST (Run before saying "FINISHED")
292: 
293: - [ ] Python syntax check: `python -m py_compile <file>`
294: - [ ] E2E tests pass: `python test_e2e_pipeline.py`
295: - [ ] Edge case tests pass: `python test_edge_cases.py`
296: - [ ] Frontend tests pass: `python test_frontend_integration.py`
297: - [ ] start.sh works: `./start.sh`
298: - [ ] No inline styles remaining in HTML/JS
299: - [ ] All CSS classes accounted for
300: - [ ] MD files updated if needed
301: - [ ] Understand file interactions for any changes
302: 
303: ---
304: 
305: **REMEMBER**: After EVERY code change, run the tests, verify start.sh works, then say "FINISHED".

(End of file - total 305 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/SYSTEM_PROMPT.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/SYSTEM_PROMPT.md</path>
<type>file</type>
<content>1: # System Prompt: Understanding the Face Recognition NPO Application
2: 
3: You are working on the Face Recognition NPO application - an ethical, consent-based facial recognition system for NGO documentation verification.
4: 
5: ---
6: 
7: ## ⚠️ CRITICAL WORKFLOW RULE
8: 
9: After EVERY code change:
10: 1. Run: `python test_e2e_pipeline.py`
11: 2. Run: `python test_edge_cases.py`  
12: 3. Run: `python test_frontend_integration.py`
13: 4. Say "FINISHED" ONLY after ALL tests pass
14: 
15: ---
16: 
17: ## Quick Start (5 Minutes)
18: 
19: For detailed workflow rules, see `WORKFLOW_PROMPT.md`
20: 
21: 1. **Start the system:** `./start.sh`
22: 2. **Choose option:** [1] Electron Desktop App / [2] Browser / [3] Both
23: 3. **Upload an image:** Click "Choose Photo"
24: 4. **Find faces:** Click "Find Faces"
25: 5. **Create signature:** Click "Create Signature"
26: 6. **Add reference:** Upload a reference image
27: 7. **Compare:** Click "Compare"
28: 
29: ---
30: 
31: ## Documentation Hierarchy (Read in Order)
32: 
33: ### Level 1: Must Read (Start Here)
34: 
35: | File | Purpose | Time |
36: |------|---------|------|
37: | `README.md` | Quick overview, features, ArcFace integration | 2 min |
38: | `CONTEXT.md` | Code review findings, AI workflow rules, common mistakes | 5 min |
39: | `DEVELOPMENT_LOG.md` | Session-by-session development history | 5 min |
40: 
41: ### Level 2: Architecture Understanding
42: 
43: | File | Purpose | Time |
44: |------|---------|------|
45: | `ARCHITECTURE.md` | Complete system architecture, ArcFace, API reference | 10 min |
46: | `PROJECT_STRUCTURE.md` | File structure, directory layout, critical lessons | 5 min |
47: 
48: ### Level 3: Technical Details
49: 
50: | File | Purpose | Time |
51: |------|---------|------|
52: | `ETHICAL_COMPLIANCE.md` | Privacy, consent, ArcFace discrimination | 5 min |
53: | `IMAGE_STORAGE.md` | Embedding storage (not raw images!) | 3 min |
54: | `USAGE.md` | Quick usage instructions | 2 min |
55: | `INSTALLATION.md` | Setup and installation | 3 min |
56: 
57: ---
58: 
59: ## ArcFace vs FaceNet (Dual-Model)
60: 
61: **Both models now run together!**
62: 
63: ### Dual-Model Comparison (Recommended)
64: - **ArcFace**: 512-dim embeddings (60% weight) - Best discrimination
65: - **FaceNet**: 128-dim embeddings (20% weight) - Additional signal + activations
66: - **Landmarks**: Geometric features (15% weight) - Structural consistency
67: - **Activation**: Neural network activations (5% weight) - Internal representations
68: - **Quality**: Image quality score (5% weight) - Reliability
69: 
70: **Note**: No pose penalty - same person with different poses will match correctly.
71: 
72: **Result**: More robust matching by combining both models
73: 
74: ### Legacy (Not Recommended - Single Model)
75: - ArcFace alone: 512-dim, <30% for different people
76: - FaceNet alone: 128-dim, ~65-70% for different people
77: 
78: ---
79: 
80: ## Code Architecture Overview
81: 
82: ```
83: ┌─────────────────────────────────────────────────────────────┐
84: │                    ENTRY POINTS                             │
85: ├─────────────────────────────────────────────────────────────┤
86: │  ./start.sh         → Interactive menu (clears cache)     │
87: │  python api_server.py → Flask API Server (:3000)           │
88: │  npm start          → Electron Desktop App (connects to Flask)│
89: └─────────────────────────────────────────────────────────────┘
90:                                │
91:                                ▼
92: ┌─────────────────────────────────────────────────────────────┐
93: │                 ELECTRON DESKTOP APP                        │
94: ├─────────────────────────────────────────────────────────────┤
95: │  main.js          → Connects to existing Flask server     │
96: │  renderer/app.js  → Frontend UI (fetch API calls)         │
97: │  index.html       → Ultra minimal UI with MANTAX navbar   │
98: └─────────────────────────────────────────────────────────────┘
99:                                │
100:                           HTTP :3000
101:                                │
102:                                ▼
103: ┌─────────────────────────────────────────────────────────────┐
104: │                  FLASK API SERVER                           │
105: ├─────────────────────────────────────────────────────────────┤
106: │  Endpoints:                                                 │
107: │  • GET  /api/health              → Status check           │
108: │  • GET  /api/embedding-info     → Model/dimension info   │
109: │  • POST /api/detect             → Face detection         │
110: │  • POST /api/extract            → Embedding extraction   │
111: │  • POST /api/add-reference      → Add reference image    │
112: │  • GET  /api/references         → List references        │
113: │  • DELETE /api/references/<id>  → Remove reference      │
114: │  • POST /api/compare            → Compare embeddings     │
115: │  • GET  /api/visualizations/<type> → Get visualization   │
116: │  • POST /api/clear             → Clear session         │
117: │  • GET  /api/status            → Debug state           │
118: └─────────────────────────────────────────────────────────────┘
119:                                │
120:                                ▼
121: ┌─────────────────────────────────────────────────────────────┐
122: │                   CORE ML PIPELINE                          │
123: ├─────────────────────────────────────────────────────────────┤
124: │                                                             │
125: │  ┌──────────────────┐                                      │
126: │  │  FaceDetector    │  OpenCV DNN (Caffe model)          │
127: │  │  Input: Image    │  Output: Bounding boxes (x,y,w,h)  │
128: │  └────────┬─────────┘                                      │
129: │           │                                                 │
130: │           ▼                                                 │
131: │  ┌────────────────────────────────────────────────────┐    │
132: │  │           Embedding Extractor                       │    │
133: │  │  ┌─────────────────────┐  ┌─────────────────────┐ │    │
134: │  │  │  ArcFace (Default)  │  │  FaceNet (Option)  │ │    │
135: │  │  │  ONNX / ResNet100  │  │  PyTorch / ResNet18│ │    │
136: │  │  │  512-dimensional   │  │  128-dimensional   │ │    │
137: │  │  │  <30% for diff!  │  │  ~70% for diff    │ │    │
138: │  │  └─────────────────────┘  └─────────────────────┘ │    │
139: │  └─────────────────────────────┬──────────────────────┘    │
140: │                                │                             │
141: │                                ▼                             │
142: │  ┌──────────────────┐                                      │
143: │  │ Similarity       │  Cosine similarity → Confidence     │
144: │  │ Comparator      │  ArcFace: ≥70%, 45-70%, 30-45%, <30%│
145: │  └──────────────────┘                                      │
146: │                                                             │
147: └─────────────────────────────────────────────────────────────┘
148: ```
149: 
150: ---
151: 
152: ## ArcFace Confidence Thresholds
153: 
154: | Similarity | Confidence | Interpretation |
155: |------------|------------|----------------|
156: | ≥70% | Very High | Likely same person |
157: | 45-70% | High | Possibly same person |
158: | 30-45% | Moderate | Human review recommended |
159: | <30% | Insufficient | Likely different people |
160: 
161: ---
162: 
163: ## Source Code Structure
164: 
165: ```
166: src/
167: ├── detection/
168: │   └── __init__.py          # FaceDetector class
169: │                             # detect_faces, estimate_landmarks
170: │                             # visualize_* methods (10 viz types)
171: │
172: ├── embedding/
173: │   ├── __init__.py          # FaceNetEmbeddingExtractor, SimilarityComparator
174: │   └── arcface_extractor.py # ArcFaceEmbeddingExtractor (512-dim, ONNX)
175: │                             # extract_embedding, get_activations
176: │                             # visualize_* methods (placeholder for ONNX)
177: │
178: └── reference/
179:     └── __init__.py          # ReferenceImageManager class
180:                               # HumanReviewInterface class
181: 
182: api_server.py                 # Flask API (11 endpoints)
183: utils/webcam.py              # WebcamCapture class
184: gui/*.py                     # Tkinter GUIs
185: ```
186: 
187: ---
188: 
189: ## Critical Classes and Methods
190: 
191: ### ArcFaceEmbeddingExtractor (`src/embedding/arcface_extractor.py`)
192: 
193: **Core Methods:**
194: - `extract_embedding(face_image)` → np.ndarray (512,)
195: - `preprocess(face_image)` → np.ndarray (112, 112)
196: - `get_activations(face_image)` → Dict[str, np.ndarray] (placeholder)
197: - `get_embedding_info()` → Dict with model info
198: 
199: **Visualization Methods:**
200: - `visualize_embedding()` → Bar chart of 512 values
201: - `visualize_similarity_matrix()` → Similarity grid
202: - `visualize_similarity_result()` → Similarity bar
203: - `test_robustness()` → Noise robustness test
204: 
205: **Note:** ArcFace ONNX doesn't expose internal layers, so activations visualization uses placeholder.
206: 
207: ### FaceDetector (`src/detection/__init__.py`)
208: 
209: **Detection Methods:**
210: - `detect_faces(image)` → List[Tuple[int, int, int, int]]
211: - `detect_faces_with_confidence(image)` → List[Tuple[Tuple, float]]
212: - `estimate_landmarks(face_image, face_box)` → Dict with landmarks
213: - `compute_alignment(face_image, landmarks)` → Dict with pitch, yaw, roll
214: - `compute_quality_metrics(face_image, face_box)` → Dict metrics
215: 
216: **Visualization Methods (10):**
217: - `visualize_detection()` → Bounding boxes
218: - `visualize_extraction()` → Face ROI
219: - `visualize_landmarks()` → 15 keypoints
220: - `visualize_3d_mesh()` → 478-point mesh
221: - `visualize_alignment()` → Orientation indicator
222: - `visualize_saliency()` → Attention heatmap
223: - `visualize_biometric_capture()` → Biometric overview
224: - `visualize_multiscale()` → Multi-scale detection
225: - `visualize_quality()` → Quality metrics
226: - `visualize_confidence_levels()` → Confidence bands
227: 
228: ### SimilarityComparator (`src/embedding/__init__.py`)
229: 
230: - `cosine_similarity(emb1, emb2)` → float
231: - `compare_embeddings(query, refs, ids)` → List[Tuple[str, float]]
232: - `get_confidence_band(similarity)` → str
233: 
234: ---
235: 
236: ## API Endpoints (11 total)
237: 
238: | Method | Endpoint | Returns |
239: |--------|----------|---------|
240: | GET | `/api/health` | Status |
241: | GET | `/api/embedding-info` | Model, dimension |
242: | POST | `/api/detect` | Faces + thumbnails + viz |
243: | POST | `/api/extract` | 512-dim or 128-dim embedding + viz |
244: | POST | `/api/add-reference` | Reference + embedding |
245: | GET | `/api/references` | List of references |
246: | DELETE | `/api/references/<id>` | Remove reference |
247: | POST | `/api/compare` | Similarity results + best match |
248: | GET | `/api/visualizations/<type>` | Query face viz |
249: | GET | `/api/visualizations/<type>/reference/<id>` | Ref viz |
250: | POST | `/api/clear` | Clear session |
251: | GET | `/api/status` | Debug state |
252: 
253: ---
254: 
255: ## 14 Visualization Types
256: 
257: | Type | Source | Description |
258: |------|--------|-------------|
259: | `detection` | FaceDetector | Bounding boxes with confidence |
260: | `extraction` | FaceDetector | Face ROI extraction |
261: | `landmarks` | FaceDetector | 15 facial keypoints |
262: | `mesh3d` | FaceDetector | 478-point 3D mesh |
263: | `alignment` | FaceDetector | Pitch/yaw/roll orientation |
264: | `saliency` | FaceDetector | Attention visualization |
265: | `activations` | EmbeddingExtractor | CNN activations (placeholder for ArcFace) |
266: | `features` | EmbeddingExtractor | Feature map grid |
267: | `multiscale` | FaceDetector | Multi-scale detection |
268: | `confidence` | FaceDetector | Quality metrics overlay |
269: | `embedding` | EmbeddingExtractor | 512-dim or 128-dim bar chart |
270: | `similarity` | EmbeddingExtractor | Similarity result bar |
271: | `robustness` | EmbeddingExtractor | Noise robustness test |
272: | `biometric` | FaceDetector | Biometric capture overview |
273: 
274: ---
275: 
276: ## Testing
277: 
278: ### E2E Tests
279: ```bash
280: python test_e2e_pipeline.py
281: # Tests: Detection → Embedding → Reference → Similarity → Full Pipeline
282: ```
283: 
284: ### API Tests
285: ```bash
286: python test_api_endpoints.py
287: ```
288: 
289: ### Edge Case Tests
290: ```bash
291: python test_edge_cases.py
292: # 11 tests covering boundary conditions
293: ```
294: 
295: ---
296: 
297: ## Common Issues and Fixes
298: 
299: ### Issue: Different people show high similarity
300: **Cause:** Using FaceNet (shows ~65-70% for different people)
301: **Fix:** Use ArcFace (shows <30% for different people)
302: 
303: ### Issue: "No data available" for visualizations
304: **Cause:** No face detected or embedding not extracted
305: **Fix:** Run detection then extraction before viewing visualizations
306: 
307: ### Issue: Visualizations show placeholder
308: **Cause:** ArcFace ONNX doesn't expose internal layers
309: **Fix:** Use placeholder visualizations that show useful info
310: 
311: ### Issue: "Module has no attribute" errors
312: **Cause:** Python cache holding old code
313: **Fix:**
314: ```bash
315: find . -type d -name "__pycache__" -exec rm -rf {} +
316: find . -name "*.pyc" -delete
317: ./start.sh
318: ```
319: 
320: ---
321: 
322: ## Developer Workflow
323: 
324: ### Before Making Changes
325: 1. Read `CONTEXT.md` for rules
326: 2. Read relevant source files
327: 3. Check for duplicate code: `grep -n "def " <file>`
328: 4. Understand architecture from `ARCHITECTURE.md`
329: 
330: ### During Changes
331: 1. Follow the Developer Mindset Checklist
332: 2. Run syntax check: `python -m py_compile <file>`
333: 3. Test imports: `python -c "import module; print('OK')"`
334: 
335: ### After Changes
336: 1. Run E2E tests: `python test_e2e_pipeline.py`
337: 2. Run API tests: `python test_api_endpoints.py`
338: 3. Update documentation
339: 
340: ---
341: 
342: ## Key Files for Understanding Specific Features
343: 
344: | To understand... | Read this file |
345: |-----------------|----------------|
346: | Overall system | `ARCHITECTURE.md` |
347: | ArcFace integration | `DEVELOPMENT_LOG.md` (Feb 12 section) |
348: | Code review history | `DEVELOPMENT_LOG.md` |
349: | Common mistakes | `CONTEXT.md` (Common Mistakes section) |
350: | API endpoints | `ARCHITECTURE.md` (API Reference section) |
351: | Visualization types | `ARCHITECTURE.md` (14 AI Visualizations) |
352: | ArcFace discrimination | `ETHICAL_COMPLIANCE.md` |
353: | Face detection | `src/detection/__init__.py` |
354: | ArcFace extractor | `src/embedding/arcface_extractor.py` |
355: | Flask API | `api_server.py` |
356: | Frontend UI | `electron-ui/renderer/app.js` |
357: 
358: ---
359: 
360: ## Performance Notes
361: 
362: - Face detection: ~100ms per image
363: - ArcFace embedding: ~50ms (ONNX)
364: - FaceNet embedding: ~100ms (PyTorch)
365: - Similarity comparison: <1ms
366: - All visualizations: ~50-200ms
367: 
368: ---
369: 
370: ## MANTAX Branding
371: 
372: The UI includes MANTAX branding:
373: - **Navbar**: White background, fixed at top
374: - **Logo**: SVG with red (#D20A11) and white
375: - **Tagline**: "Ihrem Partner für Autokrane und Schwerlastlogistik"
376: 
377: ---
378: 
379: ## Ethical Considerations
380: 
381: This system is designed for ethical NGO use:
382: 
383: - **Consent-based:** All images require documented consent
384: - **Human oversight:** No automated decisions
385: - **Confidence bands:** Shows uncertainty instead of binary decisions
386: - **Non-reversible:** 512-dim embeddings cannot reconstruct faces
387: - **ArcFace discrimination:** <30% for different people prevents false positives
388: - **Audit trail:** Complete logging of all operations
389: 
390: ---
391: 
392: *System prompt updated: February 12, 2026*
393: *Includes ArcFace integration, 512-dim embeddings, ONNX model, and MANTAX branding*
394: 

(End of file - total 394 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/SYSTEM_PROMPT_COMPLETE.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/SYSTEM_PROMPT_COMPLETE.md</path>
<type>file</type>
<content>1: # Complete System Prompt: NGO Facial Image Analysis System
2: 
3: ## ⚠️ CRITICAL WORKFLOW RULE
4: 
5: After EVERY code change:
6: 1. Run: `python test_e2e_pipeline.py`
7: 2. Run: `python test_edge_cases.py`
8: 3. Run: `python test_frontend_integration.py`
9: 4. Say "FINISHED" ONLY after ALL tests pass
10: 
11: ---
12: 
13: ## Role and Context
14: 
15: You are working on the **NGO Facial Image Analysis System** - a complete, production-ready facial recognition system designed for ethical, consent-based documentation verification workflows. This is **version 0.5.0** and the system is **fully functional**.
16: 
17: ## Complete System Understanding Checklist
18: 
19: BEFORE making any changes, you must have read and understood:
20: 
21: ### ✅ Documentation Files (All Must Be Read)
22: 1. **README.md** - Main documentation, features overview
23: 2. **CONTEXT.md** - ⚠️ **CRITICAL**: Strict edit rules, developer workflow, common mistakes
24: 3. **ARCHITECTURE.md** - Complete system design, API endpoints, visualization types
25: 4. **PROJECT_STRUCTURE.md** - File organization, roadmap, critical lessons learned
26: 5. **DEVELOPMENT_LOG.md** - Session-by-session development history
27: 6. **ETHICAL_COMPLIANCE.md** - Privacy guidelines
28: 7. **INSTALLATION.md** - Setup and dependencies
29: 8. **USAGE.md** - Quick start instructions
30: 9. **IMAGE_STORAGE.md** - Embedding storage format
31: 10. **SYSTEM_PROMPT.md** - Quick reference
32: 
33: ### ✅ Code Files (Core Architecture)
34: 1. **Python Backend**:
35:    - `api_server.py` - Flask API server (12+ REST endpoints)
36:    - `src/detection/__init__.py` - FaceDetector class (OpenCV DNN + MediaPipe)
37:    - `src/detection/preprocessing.py` - ImagePreprocessor
38:    - `src/embedding/__init__.py` - SimilarityComparator, FaceNetEmbeddingExtractor
39:    - `src/embedding/arcface_extractor.py` - ArcFaceEmbeddingExtractor
40: 
41: 2. **Frontend**:
42:    - `electron-ui/main.js` - Electron main process
43:    - `electron-ui/renderer/app.js` - Frontend JavaScript
44:    - `electron-ui/index.html` - UI with scrollable tabs
45: 
46: 3. **Test Files**:
47:    - `test_frontend_integration.py` - Visual frontend tests
48:    - `test_e2e_pipeline.py` - End-to-end tests
49:    - `test_eyewear.py` - Eyewear detection tests
50: 
51: ### ✅ Key Features (v0.5.0)
52: - **Dual-Model**: ArcFace (512-dim) + FaceNet (128-dim) together
53: - **Weighted Comparison**: ArcFace 60%, FaceNet 20%, Landmarks 15%, Quality 5%
54: - **468 MediaPipe Landmarks**: Full face mesh visualization
55: - **Pose Detection**: yaw, pitch, roll estimation
56: - **Multi-Reference**: Store multiple poses per person
57: - **Image Preprocessing**: Auto CLAHE, histogram equalization
58: - **Eyewear Detection**: Improved with brightness analysis
59: - **20+ Visualization Tabs**: Scrollable in UI
60: 
61: ### ✅ API Endpoints (12+ total)
62: | Method | Endpoint | Description |
63: |--------|----------|-------------|
64: | GET | `/api/health` | Health check |
65: | POST | `/api/detect` | Face detection + preprocessing |
66: | POST | `/api/extract` | Embedding + pose extraction |
67: | POST | `/api/add-reference` | Add reference with pose |
68: | GET | `/api/references` | List references |
69: | DELETE | `/api/references/<id>` | Remove reference |
70: | POST | `/api/compare` | Pose-aware similarity |
71: | GET | `/api/visualizations/<type>` | 16 viz types |
72: | POST | `/api/clear` | Clear session |
73: | GET | `/api/status` | Debug state |
74: | GET | `/api/eyewear` | Eyewear detection |
75: 
76: ### ✅ Visualization Types (16 total)
77: Detection, Extraction, Preprocessing, Landmarks, 3D Mesh, Alignment, Attention, Activations, Features, Multi-Scale, Confidence, Eyewear, Embedding, Similarity, Robustness, Biometric, Tests (9 individual test views)
78: 
79: ### ✅ Thresholds (ArcFace)
80: - **≥70%**: Very High - Same person likely
81: - **45-70%**: High - Possibly same person  
82: - **20-45%**: Moderate - Different angles
83: - **<20%**: Insufficient - Different people
84: 
85: ## Development Workflow
86: 
87: **Before any edit:**
88: ```bash
89: python -m py_compile <file>
90: ```
91: 
92: **HTML-JS Cross-Check (MANDATORY):**
93: ```bash
94: grep -E 'onclick=|onchange=' electron-ui/index.html
95: ```
96: 
97: 2. **Frontend**:
98:    - `electron-ui/main.js` - Electron main process
99:    - `electron-ui/renderer/app.js` - Frontend JavaScript (25+ functions)
100:    - `electron-ui/index.html` - Ultra minimal UI with MANTAX branding
101: 
102: 3. **Key Models Files**:
103:    - `deploy.prototxt.txt` - OpenCV DNN config
104:    - `res10_300x300_ssd_iter_140000.caffemodel` - Face detection weights
105:    - `arcface_model.onnx` - ArcFace model (117MB, 512-dim)
106: 
107: ## Critical System Status
108: 
109: ### ✅ Fully Functional (v0.3.0)
110: - **ArcFace Integration**: 512-dimensional embeddings, excellent discrimination
111: - **API Server**: 11 endpoints fully operational
112: - **Electron App**: Production-ready with MANTAX branding
113: - **Testing**: All tests passing (6/6 E2E, 30/30 unit, 11/11 edge cases)
114: - **Reference Management**: Persistent JSON storage with real embeddings
115: - **Visualizations**: 14 AI visualization types available
116: 
117: ### 🎯 Key Achievement: False Positive Prevention
118: ```
119: BEFORE (FaceNet): 65-70% similarity for different people (FALSE POSITIVES!)
120: AFTER  (ArcFace): <30% similarity for different people (CORRECT!)
121: ```
122: 
123: This is **critical for NGO use** - prevents wrongful identifications.
124: 
125: ## Development Workflow Rules (From CONTEXT.md)
126: 
127: ### ⚠️ MANDATORY Rules (No Exceptions!)
128: 
129: **Rule 1: Syntax Check**
130: ```bash
131: # BEFORE any edit
132: python -m py_compile <file>
133: ```
134: 
135: **Rule 2: No Duplicate Code**
136: ```bash
137: # Check for duplicates
138: grep -n "def " <file>
139: grep -rn "return {}" src/ --include="*.py"
140: ```
141: 
142: **Rule 3: Function Preservation (JavaScript)**
143: ```bash
144: # BEFORE any JavaScript edit
145: CRITICAL_FUNCS="selectImage handleImageSelect detectFaces extractFeatures compareFaces clearAllCache removeReference showReferenceVisualizations updateReferenceList"
146: for func in $CRITICAL_FUNCS; do
147:   if ! grep -q "function $func" electron-ui/renderer/app.js; then
148:     echo "MISSING: $func"
149:     exit 1
150:   fi
151: done
152: ```
153: 
154: **Rule 4: HTML-JS Cross-Check (MANDATORY Pre-commit)**
155: ```bash
156: # Extract HTML handlers and verify in JS
157: HTML_FUNCS=$(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u)
158: for func in $HTML_FUNCS; do
159:     if ! grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js; then
160:         echo "✗ MISSING: $func"
161:         exit 1
162:     fi
163: done
164: ```
165: 
166: **Rule 5: Atomic Edits**
167: - Make ONE edit per function
168: - Verify each edit individually
169: - Never attempt multiple complex edits in one operation
170: 
171: **Rule 6: Read Context First**
172: - Read at least 50 lines around edit location
173: - Understand full context before making changes
174: - Identify all affected code
175: 
176: **Rule 7: Fire-and-Forget for Non-Critical APIs**
177: ```javascript
178: // WRONG (blocks UI)
179: await fetch(`${API_BASE}/clear`, { method: 'POST' });
180: 
181: // RIGHT (fire-and-forget)
182: fetch(`${API_BASE}/clear`, { method: 'POST' }).catch(err => {
183:     console.log('Clear failed:', err.message);
184: });
185: ```
186: 
187: ## System Architecture
188: 
189: ### Data Flow
190: ```
191: User Upload → Electron UI → HTTP API → ML Pipeline → JSON Storage → Results
192: ```
193: 
194: ### Core Components
195: 1. **Face Detection**: OpenCV DNN (primary) + Haar Cascade (fallback)
196: 2. **Embedding Extraction**: ArcFace (512-dim, default) or FaceNet (128-dim, legacy)
197: 3. **Similarity Comparison**: Cosine similarity with confidence bands
198: 4. **Reference Storage**: JSON persistence with metadata tracking
199: 
200: ### API Endpoints (11 total)
201: | Method | Endpoint | Purpose |
202: |--------|----------|---------|
203: | GET | `/api/health` | Health check |
204: | GET | `/api/embedding-info` | Model/dimension info |
205: | POST | `/api/detect` | Face detection |
206: | POST | `/api/extract` | Embedding extraction |
207: | POST | `/api/add-reference` | Add reference |
208: | GET | `/api/references` | List references |
209: | DELETE | `/api/references/<id>` | Remove reference |
210: | POST | `/api/compare` | Compare embeddings |
211: | GET | `/api/visualizations/<type>` | Get visualization |
212: | POST | `/api/clear` | Clear session |
213: | GET | `/api/status` | Debug state |
214: 
215: ### 14 Visualization Types
216: 1. **Detection**: Bounding boxes with confidence
217: 2. **Extraction**: Face ROI preview
218: 3. **Landmarks**: 15 facial keypoints
219: 4. **3D Mesh**: 478-point wireframe
220: 5. **Alignment**: Pitch/yaw/roll orientation
221: 6. **Saliency**: Attention heatmap
222: 7. **Activations**: CNN layer activations (placeholder for ArcFace)
223: 8. **Features**: Feature map grid
224: 9. **Multi-Scale**: Face at 5 scales
225: 10. **Confidence**: Quality metrics dashboard
226: 11. **Embedding**: 512-dim or 128-dim bar chart
227: 12. **Similarity**: Similarity matrix
228: 13. **Robustness**: Noise robustness test
229: 14. **Biometric**: Biometric capture overview
230: 
231: ## Critical Lessons Learned (From Development Log)
232: 
233: ### Lesson 1: ArcFace vs FaceNet Discrimination
234: - FaceNet: 65-70% for different people (false positives!)
235: - ArcFace: <30% for different people (correct!)
236: - **Rule**: Always use ArcFace for NGO use cases
237: 
238: ### Lesson 2: Dynamic Array Sizes
239: ```python
240: # WRONG (causes broadcast error)
241: output = np.zeros((150, 300, 3), dtype=np.uint8)
242: 
243: # RIGHT (dynamic sizing)
244: output_size = max(150, n * cell_size)
245: output = np.zeros((output_size, output_size, 3), dtype=np.uint8)
246: ```
247: 
248: ### Lesson 3: Real Embeddings Only
249: ```python
250: # WRONG (random embeddings)
251: embedding = np.random.rand(512)
252: 
253: # RIGHT (extract real embeddings)
254: embedding = extractor.extract_embedding(face_roi)
255: ```
256: 
257: ### Lesson 4: Clear Python Cache
258: ```bash
259: # Always run after editing source
260: find . -type d -name "__pycache__" -exec rm -rf {} +
261: find . -name "*.pyc" -delete
262: ```
263: 
264: ### Lesson 5: Reference Persistence
265: - Call `save_references()` after add/remove operations
266: - Load references on API server startup
267: - Store real embeddings, not random values
268: 
269: ### Lesson 6: UI Blocking with Async/Await
270: - Never await non-essential API calls in event handlers
271: - Use fire-and-forget pattern for operations like cache clear
272: - Blocking UI makes app appear frozen
273: 
274: ## Common Bugs and Fixes
275: 
276: ### Bug 1: Missing JavaScript Functions
277: **Symptom**: Buttons do nothing when clicked
278: **Cause**: HTML onclick handlers reference non-existent functions
279: **Fix**: Always run HTML-JS cross-check before committing
280: 
281: ### Bug 2: Random Embeddings
282: **Symptom**: Similarity scores are meaningless
283: **Cause**: Using `np.random.rand()` instead of real extraction
284: **Fix**: Always call `extractor.extract_embedding(face_roi)`
285: 
286: ### Bug 3: Broadcast Errors in Visualizations
287: **Symptom**: "could not broadcast input array" errors
288: **Cause**: Hardcoded array sizes for dynamic data
289: **Fix**: Use dynamic sizing based on input count
290: 
291: ### Bug 4: References Not Persisting
292: **Symptom**: References disappear after server restart
293: **Cause**: Only stored in memory, never saved to JSON
294: **Fix**: Call `save_references()` after mutations
295: 
296: ### Bug 5: Python Cache Issues
297: **Symptom**: Code changes don't take effect
298: **Cause**: Old `.pyc` files with stale bytecode
299: **Fix**: Clear cache and restart server
300: 
301: ## Quick Start Commands
302: 
303: ```bash
304: # Start system
305: cd face_recognition_npo
306: ./start.sh
307: 
308: # Run tests
309: python test_e2e_pipeline.py
310: python test_edge_cases.py
311: 
312: # Clear cache
313: find . -type d -name "__pycache__" -exec rm -rf {} +
314: find . -name "*.pyc" -delete
315: 
316: # Verify API
317: curl http://localhost:3000/api/health
318: curl http://localhost:3000/api/embedding-info
319: ```
320: 
321: ## Usage Workflow (Correct Steps)
322: 
323: ```
324: Step 1: Choose Photo     → Upload image file
325: Step 2: Find Faces       → Click "Find Faces" button
326: Step 3: Create Signature → Click "Create Signature" (CRITICAL!)
327: Step 4: Add Reference    → Upload reference image
328: Step 5: Compare          → Click "Compare" button
329: ```
330: 
331: **Common Mistakes:**
332: - Skipping "Create Signature" - no embedding extracted
333: - Clicking "Compare" before adding references
334: - Refreshing page (loses server state)
335: 
336: ## ArcFace Confidence Thresholds
337: 
338: | Similarity | Confidence | Action |
339: |------------|------------|---------|
340: | ≥70% | Very High | Likely same person |
341: | 45-70% | High | Possibly same person |
342: | 30-45% | Moderate | Human review required |
343: | <30% | Insufficient | Likely different people |
344: 
345: ## Ethical Guidelines
346: 
347: 1. **Consent-Based**: All images require documented consent
348: 2. **Human Oversight**: No automated decisions
349: 3. **Uncertainty Handling**: Confidence bands instead of binary decisions
350: 4. **Privacy Protection**: Non-reversible embeddings only
351: 5. **Audit Trail**: Complete logging of all operations
352: 
353: ## Pre-Commit Verification (Mandatory)
354: 
355: Run this before EVERY commit:
356: 
357: ```bash
358: #!/bin/bash
359: cd /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo
360: 
361: echo "========================================="
362: echo "MANDATORY PRE-COMMIT VERIFICATION"
363: echo "========================================="
364: 
365: PASS=true
366: 
367: # 1. Python syntax
368: python -m py_compile api_server.py || PASS=false
369: 
370: # 2. Tests pass
371: python test_e2e_pipeline.py 2>&1 | grep -q "ALL TESTS PASSED" || PASS=false
372: 
373: # 3. HTML-JS cross-check
374: MISSING=""
375: for func in $(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u); do
376:     if ! grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js; then
377:         MISSING="$MISSING $func"
378:     fi
379: done
380: [ -n "$MISSING" ] && PASS=false
381: 
382: if [ "$PASS" = true ]; then
383:     echo "✓ ALL VERIFICATIONS PASSED"
384:     exit 0
385: else
386:     echo "✗ VERIFICATION FAILED - FIX BEFORE COMMITTING"
387:     exit 1
388: fi
389: ```
390: 
391: ## Current Status
392: 
393: - **Version**: 0.3.0
394: - **Status**: ✅ Fully Functional
395: - **All Tests Passing**: E2E (6/6), Unit (30/30), Edge Cases (11/11)
396: - **Default Model**: ArcFace (512-dim, ONNX)
397: - **UI**: Electron with ultra minimal design + MANTAX branding
398: - **API**: 11 endpoints fully operational
399: - **Storage**: Persistent JSON with real embeddings
400: - **Ready For**: Production NGO deployment
401: 
402: ## Files to Know (Edit Priority)
403: 
404: | Priority | File | Purpose |
405: |----------|-------|---------|
406: | 1 | `api_server.py` | Flask API (11 endpoints) |
407: | 2 | `electron-ui/renderer/app.js` | Frontend logic (25+ functions) |
408: | 3 | `electron-ui/index.html` | UI structure |
409: | 4 | `src/embedding/arcface_extractor.py` | ArcFace implementation |
410: | 5 | `src/detection/__init__.py` | Face detection |
411: | 6 | `src/reference/__init__.py` | Reference management |
412: | 7 | `CONTEXT.md` | Edit rules (read before ANY changes) |
413: | 8 | `test_e2e_pipeline.py` | E2E tests |
414: 
415: ---
416: 
417: **This system is production-ready for NGO documentation verification workflows.**
418: **All critical bugs have been fixed.**
419: **All tests are passing.**
420: **Strict development rules are in place to prevent regressions.**
421: 
422: *System Prompt created: February 12, 2026*
423: *Complete understanding achieved: All documentation + code read*

(End of file - total 423 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/CONTEXT.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/CONTEXT.md</path>
<type>file</type>
<content>1: # Code Review Context - Face Recognition NPO
2: 
3: **Review Date**: February 13, 2026  
4: **Reviewer**: AI Code Review (with Developer Mindset)  
5: **Status**: ✅ ALL ISSUES FIXED - EYWEAR DETECTION FALSE POSITIVES FIXED
6: 
7: ---
8: 
9: ## ⚠️ CRITICAL WORKFLOW RULE - ALWAYS FOLLOW THIS
10: 
11: **After EVERY code change, you MUST:**
12: 
13: 1. ✅ Run E2E tests: `python test_e2e_pipeline.py`
14: 2. ✅ Run edge case tests: `python test_edge_cases.py`
15: 3. ✅ Run frontend tests: `python test_frontend_integration.py`
16: 4. ✅ Say "FINISHED" ONLY after ALL tests pass
17: 
18: **NEVER say "done" or "finished" until ALL tests pass.**
19: 
20: ---
21: 
22: ## ⚠️ CRITICAL RULES FOR CODE EDITS
23: 
24: **These rules MUST be followed at all times:**
25: 
26: ### Rule 1: Syntax Check Before Submitting
27: **BEFORE any code edit submission:**
28: - Count opening and closing parentheses: `(`, `)`, `[`, `]`, `{`, `}`
29: - Count opening and closing quotes: `'`, `"`
30: - Ensure each `return` statement has a complete expression before it
31: - Run `python -m py_compile <file>` to check syntax
32: 
33: ### Rule 2: Check for Duplicate Code
34: **BEFORE any code edit submission:**
35: - Search for similar code patterns in the file
36: - Use `grep` to find duplicate function definitions
37: - Use `grep -n "def " <file>` to list all function definitions
38: - Look for `elif` chains that might have duplicates
39: - Verify you're not adding code that already exists elsewhere in the file
40: 
41: ### Rule 3: Import Verification
42: **AFTER any code edit:**
43: - Run the module to verify imports work
44: - Check for undefined variables in the edit area
45: - Verify all referenced functions/methods exist
46: 
47: ### Rule 4: Read Before Edit
48: **BEFORE any `edit` tool use:**
49: - Read at least 50 lines around the edit location
50: - Understand the full context of the code block
51: - Identify all code that might be affected
52: 
53: ### Rule 5: Use the Write Tool for Major Changes
54: **For complex refactoring:**
55: - Use `read` to understand the full file
56: - Use `write` to create a complete new version
57: - Don't try to make multiple complex `edit` calls
58: 
59: ### Rule 6: Function Preservation for JavaScript Files
60: **BEFORE making edits to JavaScript files:**
61: 
62: 1. List ALL function definitions:
63: ```bash
64: grep -n "function " electron-ui/renderer/app.js
65: ```
66: 
67: 2. Count functions before edit:
68: ```bash
69: grep -n "function " electron-ui/renderer/app.js | wc -l
70: ```
71: 
72: 3. Verify critical functions exist after edit:
73: ```bash
74: CRITICAL_FUNCS="handleImageSelect selectImage resetSteps detectFaces extractFeatures compareFaces clearAllCache removeReference showReferenceVisualizations updateReferenceList"
75: for func in $CRITICAL_FUNCS; do
76:   if ! grep -q "function $func" electron-ui/renderer/app.js; then
77:     echo "MISSING: $func"
78:     exit 1
79:   fi
80: done
81: echo "All critical functions present"
82: ```
83: 
84: **CRITICAL FUNCTIONS (must always exist):**
85: - `selectImage()` - File input trigger
86: - `handleImageSelect()` - File processing (MOST CRITICAL - losing this breaks uploads!)
87: - `resetSteps()` - State reset
88: - `detectFaces()` - Face detection
89: - `extractFeatures()` - Embedding extraction
90: - `compareFaces()` - Comparison
91: - `updateReferenceList()` - Reference management
92: - `clearAllCache()` - Cache clearing
93: - `removeReference()` - Reference removal
94: - `showReferenceVisualizations()` - Reference viz
95: 
96: ### Rule 7: Atomic Edits for Multiple Functions
97: **When adding 2+ functions in one session:**
98: 
99: 1. Make ONE edit per function
100: 2. Verify each edit individually with the function count script
101: 3. Only proceed to next edit after verification passes
102: 
103: ```bash
104: # BEFORE first edit - count functions
105: echo "Before: $(grep -n 'function ' electron-ui/renderer/app.js | wc -l) functions"
106: 
107: # AFTER each edit - count and compare
108: echo "After: $(grep -n 'function ' electron-ui/renderer/app.js | wc -l) functions"
109: 
110: # If count doesn't match expected, restore from git:
111: # git checkout electron-ui/renderer/app.js
112: ```
113: 
114: **WRONG approach:**
115: ```bash
116: # One massive edit trying to add/change 5 functions → CORRUPTION RISK
117: ```
118: 
119: **RIGHT approach:**
120: ```bash
121: Edit 1: Add clearAllCache() → Verify → Pass
122: Edit 2: Add removeReference() → Verify → Pass
123: Edit 3: Add showReferenceVisualizations() → Verify → Pass
124: Edit 4: Update updateReferenceList() → Verify → Pass
125: ```
126: 
127: ### Rule 8: Fire-and-Forget for Non-Critical API Calls
128: **When calling non-blocking APIs (like cache clear):**
129: - Use `.catch()` instead of `await` with try/catch
130: - Never await non-essential API calls in user interaction handlers
131: - Blocking the UI thread with awaits can freeze the interface
132: 
133: **WRONG:**
134: ```javascript
135: reader.onload = async (e) => {
136:     try {
137:         await fetch(`${API_BASE}/clear`, { method: 'POST' });  // BLOCKS!
138:     } catch (err) {
139:         logToTerminal(`> Warning: ${err.message}`, 'warning');
140:     }
141:     // ... rest of upload logic
142: };
143: ```
144: 
145: **RIGHT:**
146: ```javascript
147: reader.onload = (e) => {
148:     fetch(`${API_BASE}/clear`, { method: 'POST' }).catch(err => {
149:         console.log('Clear failed:', err.message);
150:     });
151:     // ... rest of upload logic (executes immediately)
152: };
153: ```
154: 
155: ### Rule 9: Cross-Check HTML with JavaScript (CRITICAL!)
156: 
157: **BEFORE submitting ANY edit to app.js:**
158: 
159: 1. Extract ALL function calls from HTML:
160: ```bash
161: grep -E 'onclick=|onchange=' electron-ui/index.html
162: ```
163: 
164: 2. Verify each function exists in app.js:
165: ```bash
166: echo "=== HTML EVENT HANDLERS ===" && \
167: grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u
168: 
169: echo "" && echo "=== VERIFYING FUNCTIONS ===" && \
170: for func in $(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u); do
171:     if grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js; then
172:         echo "✓ $func"
173:     else
174:         echo "✗ MISSING: $func - ADD IT NOW!"
175:         exit 1
176:     fi
177: done
178: ```
179: 
180: 3. Count and compare function definitions:
181: ```bash
182: echo "Sync functions: $(grep -c '^function ' electron-ui/renderer/app.js)"
183: echo "Async functions: $(grep -c '^async function ' electron-ui/renderer/app.js)"
184: echo "Total: $(($(grep -c '^function ') + $(grep -c '^async function ')))"
185: ```
186: 
187: **CRITICAL FUNCTIONS (must always exist):**
188: ```
189: From HTML onclick/onchange:
190:   ✓ selectImage()           - Trigger file input
191:   ✓ handleImageSelect(event) - Process uploaded image
192:   ✓ detectFaces()           - Face detection
193:   ✓ extractFeatures()       - Embedding extraction
194:   ✓ addReference()          - Add reference image
195:   ✓ handleReferenceSelect(event) - Process reference upload
196:   ✓ compareFaces()          - Compare embeddings
197:   ✓ clearAllCache()         - Clear session
198:   ✓ toggleTerminal()        - Toggle terminal
199: 
200: From app.js internal calls:
201:   ✓ resetSteps()            - Reset UI state
202:   ✓ updateReferenceList()   - Render references
203:   ✓ removeReference()        - Delete reference
204:   ✓ showReferenceVisualizations() - Show ref viz
205:   ✓ saveReference()         - Save ref to backend
206:   ✓ selectReference()       - Select ref for viz
207:   ✓ showLoading()           - Show loading overlay
208:   ✓ hideLoading()           - Hide loading overlay
209:   ✓ showToast()            - Show toast notification
210:   ✓ logToTerminal()        - Log to terminal
211:   ✓ clearTerminal()        - Clear terminal
212:   ✓ initTerminal()         - Initialize terminal
213:   ✓ toggleTerminal()       - Toggle terminal
214:   ✓ showVisualization()   - Show viz
215:   ✓ showVisualizationPlaceholder() - Show placeholder
216:   ✓ formatDataAsTable()    - Format data table
217:   ✓ formatKey()            - Format key names
218: ```
219: 
220: **SHELL SCRIPT FOR COMPLETE VERIFICATION:**
221: ```bash
222: #!/bin/bash
223: # Complete verification script - RUN BEFORE EVERY COMMIT
224: 
225: echo "========================================="
226: echo "HTML-JS CROSS-CHECK VERIFICATION"
227: echo "========================================="
228: 
229: cd /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo
230: 
231: # Extract HTML function calls
232: echo "1. Extracting HTML event handlers..."
233: HTML_FUNCS=$(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u)
234: echo "$HTML_FUNCS"
235: 
236: # Extract JS definitions
237: echo ""
238: echo "2. Extracting JS function definitions..."
239: JS_SYNC=$(grep -c '^function ' electron-ui/renderer/app.js)
240: JS_ASYNC=$(grep -c '^async function ' electron-ui/renderer/app.js)
241: echo "Sync: $JS_SYNC, Async: $JS_ASYNC, Total: $((JS_SYNC + JS_ASYNC))"
242: 
243: # Verify each HTML function exists
244: echo ""
245: echo "3. Verifying HTML functions exist in JS..."
246: MISSING=0
247: for func in $HTML_FUNCS; do
248:     if ! grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js; then
249:         echo "✗ MISSING: $func"
250:         MISSING=$((MISSING + 1))
251:     fi
252: done
253: 
254: if [ $MISSING -eq 0 ]; then
255:     echo "✓ All HTML functions verified!"
256: else
257:     echo ""
258:     echo "✗ FAILED: $MISSING missing functions!"
259:     echo "FIX BEFORE COMMITTING!"
260:     exit 1
261: fi
262: 
263: echo ""
264: echo "========================================="
265: echo "✓ VERIFICATION PASSED"
266: echo "========================================="
267: ```
268: 
269: ---
270: 
271: ## Developer Mindset Checklist
272: 
273: Before submitting ANY code edit:
274: 
275: ```
276: □ 1. Did I count parentheses and brackets?
277: □ 2. Did I check for duplicate code with grep?
278: □ 3. Did I verify the code compiles/runs?
279: □ 4. Did I read the surrounding context (50+ lines)?
280: □ 5. Did I check that all referenced functions exist?
281: □ 6. Did I avoid introducing unused variables?
282: □ 7. Did I handle None/edge cases?
283: □ 8. Did I use the correct exception handling pattern?
284: ```
285: 
286: If ANY answer is NO → Do not submit the edit until fixed!
287: 
288: ---
289: 
290: ### MANDATORY PRE-COMMIT VERIFICATION SCRIPT
291: 
292: Run this BEFORE every commit:
293: 
294: ```bash
295: #!/bin/bash
296: # Complete verification - Run BEFORE commit
297: 
298: cd /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo
299: 
300: echo "========================================="
301: echo "MANDATORY PRE-COMMIT VERIFICATION"
302: echo "========================================="
303: 
304: PASS=true
305: 
306: # 1. BACKEND: Python syntax
307: echo ""
308: echo "1. Backend Python syntax..."
309: python -m py_compile api_server.py
310: if [ $? -eq 0 ]; then
311:     echo "   ✓ Backend OK"
312: else
313:     echo "   ✗ Backend FAILED"
314:     PASS=false
315: fi
316: 
317: # 0. PYTHON ENVIRONMENT: Check we're using project venv
318: echo ""
319: echo "0. Python environment..."
320: PYTHON_PATH=$(which python)
321: if [[ "$PYTHON_PATH" == *"/MANTAX/face_recognition_npo/venv/"* ]]; then
322:     echo "   ✓ Using project venv: $PYTHON_PATH"
323: elif [[ "$PYTHON_PATH" == *"/MANTAX/face_recognition_npo/venv313/"* ]]; then
324:     echo "   ✓ Using project venv313: $PYTHON_PATH"
325: else
326:     echo "   ✗ WRONG PYTHON: $PYTHON_PATH"
327:     echo "   ✗ Must use: .../MANTAX/face_recognition_npo/venv/bin/python"
328:     echo "   ✗ Run: cd /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo && source venv/bin/activate"
329:     PASS=false
330: fi
331: 
332: # 2. BACKEND: API endpoints
333: echo ""
334: echo "2. Backend API endpoints..."
335: ENDPOINTS=$(grep -c "@app.route" api_server.py)
336: echo "   Found: $ENDPOINTS endpoints"
337: if [ "$ENDPOINTS" -ge 10 ]; then
338:     echo "   ✓ Endpoint count OK"
339: else
340:     echo "   ✗ Endpoint count too low"
341:     PASS=false
342: fi
343: 
344: # 3. BACKEND: Visualization handlers
345: echo ""
346: echo "3. Backend visualization handlers..."
347: VIZ_HANDLERS=$(grep -c "viz_type ==" api_server.py)
348: echo "   Found: $VIZ_HANDLERS handlers"
349: if [ "$VIZ_HANDLERS" -ge 10 ]; then
350:     echo "   ✓ Visualization handlers OK"
351: else
352:     echo "   ✗ Visualization handlers too few"
353:     PASS=false
354: fi
355: 
356: # 4. FRONTEND: HTML event handlers
357: echo ""
358: echo "4. Frontend HTML event handlers..."
359: HTML_HANDLERS=$(grep -cE 'onclick=|onchange=' electron-ui/index.html)
360: echo "   Found: $HTML_HANDLERS handlers"
361: if [ "$HTML_HANDLERS" -ge 5 ]; then
362:     echo "   ✓ HTML handlers OK"
363: else
364:     echo "   ✗ HTML handlers too few"
365:     PASS=false
366: fi
367: 
368: # 5. FRONTEND: Function definitions
369: echo ""
370: echo "5. Frontend function definitions..."
371: FUNC_COUNT=$(grep -c '^function ' electron-ui/renderer/app.js)
372: echo "   Found: $FUNC_COUNT functions"
373: if [ "$FUNC_COUNT" -ge 10 ]; then
374:     echo "   ✓ Function count OK"
375: else
376:     echo "   ✗ Function count too low"
377:     PASS=false
378: fi
379: 
380: # 6. HTML-JS cross-check
381: echo ""
382: echo "6. HTML-JS cross-check..."
383: MISSING=""
384: for func in $(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u); do
385:     if ! grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js; then
386:         MISSING="$MISSING $func"
387:     fi
388: done
389: 
390: if [ -z "$MISSING" ]; then
391:     echo "   ✓ All HTML functions exist in JS"
392: else
393:     echo "   ✗ MISSING:$MISSING"
394:     PASS=false
395: fi
396: 
397: # 7. E2E Tests
398: echo ""
399: echo "7. E2E tests..."
400: python test_e2e_pipeline.py 2>&1 | grep -q "ALL TESTS PASSED"
401: if [ $? -eq 0 ]; then
402:     echo "   ✓ E2E tests passed"
403: else
404:     echo "   ✗ E2E tests FAILED"
405:     PASS=false
406: fi
407: 
408: echo ""
409: echo "========================================="
410: if [ "$PASS" = true ]; then
411:     echo "✓ ALL VERIFICATIONS PASSED"
412:     echo "Ready to commit"
413:     exit 0
414: else
415:     echo "✗ VERIFICATION FAILED"
416:     echo "Fix issues before committing"
417:     exit 1
418: fi
419: ```
420: 
421: ---
422: 
423: ## Executive Summary
424: 
425: A comprehensive code review was conducted on February 11, 2026, followed by a verification review. All issues have been resolved:
426: 
427: | Category | Found | Fixed | Remaining |
428: |----------|-------|-------|-----------|
429: | Critical Issues | 7 | 7 | 0 |
430: | Medium Issues | 6 | 0 | 6 |
431: | Minor Issues | 6 | 0 | 6 |
432: | Security Issues | 2 | 1 | 1 |
433: | Mock/Stub Code | 1 | 1 | 0 |
434: 
435: ---
436: 
437: ## Critical Issues Fixed (7/7)
438: 
439: | # | File | Issue | Fix |
440: |---|------|-------|-----|
441: | 1 | `src/embedding/__init__.py` | Duplicate exception handler (lines 205-210) | Removed dead code |
442: | 2 | `src/embedding/__init__.py` | Wrong attribute access `self.model.backbone` | Changed to `self.model.backbone` |
443: | 3 | `utils/webcam.py` | Missing imports: `List, Tuple` | Added to typing imports |
444: | 4 | `utils/webcam.py` | Missing module imports | Added `FaceDetector`, `FaceNetEmbeddingExtractor`, `SimilarityComparator` |
445: | 5 | `api_server.py` | Missing method `visualize_embedding()` | Implemented in `FaceNetEmbeddingExtractor` |
446: | 6 | `api_server.py` | Missing method `visualize_similarity_result()` | Implemented in `FaceNetEmbeddingExtractor` |
447: | 7 | `visualize_biometric.py` | Hardcoded wrong path | Fixed path to `test_images/kanye_west.jpeg` |
448: 
449: ---
450: 
451: ## Security Issues Fixed (1/2)
452: 
453: | # | File | Issue | Fix |
454: |---|------|-------|-----|
455: | 19 | `gui/facial_analysis_gui.py` | Random embedding fallback on lines 1075, 1118 | Changed to `None` instead of random |
456: 
457: **Remaining Security Issue (#20)**: API lacks input validation for paths/base64 data. This is a future enhancement.
458: 
459: ---
460: 
461: ## Mock Code Status
462: 
463: ### Before Review
464: ```python
465: def get_activations(self, face_image: np.ndarray) -> Dict[str, np.ndarray]:
466:     return {}  # STUB - empty dict
467: ```
468: 
469: ### After Review
470: ```python
471: def get_activations(self, face_image: np.ndarray) -> Dict[str, np.ndarray]:
472:     """Extract neural network activations from 10 layers + embedding."""
473:     activations = {}
474:     backbone = self.model.backbone
475:     # ... real implementation extracting activations from conv1, bn1, relu, maxpool, layer1-4, gap, fc
476:     activations['embedding'] = self.extract_embedding(face_image)
477:     return activations
478: ```
479: 
480: **Result**: All 11 activation layers are now real:
481: - `conv1`: (64, 112, 112)
482: - `bn1`: (64, 112, 112)
483: - `relu`: (64, 112, 112)
484: - `maxpool`: (64, 56, 56)
485: - `layer1`: (64, 56, 56)
486: - `layer2`: (128, 28, 28)
487: - `layer3`: (256, 14, 14)
488: - `layer4`: (512, 7, 7)
489: - `gap`: (512, 1, 1)
490: - `fc`: (512, 1, 1)
491: - `embedding`: (128,)
492: 
493: ---
494: 
495: ## Test Results
496: 
497: ### E2E Tests (6/6 PASS)
498: ```
499: ✓ PASS: Detection
500: ✓ PASS: Embedding
501: ✓ PASS: Reference Manager
502: ✓ PASS: Same Image Similarity
503: ✓ PASS: Different Images Similarity
504: ✓ PASS: Reference Comparison
505: ```
506: 
507: ### Edge Case Tests (11/11 PASS)
508: ```
509: ✓ PASS: Empty/Black Image
510: ✓ PASS: Very Small Images
511: ✓ PASS: None/Invalid Inputs
512: ✓ PASS: Boundary Similarity
513: ✓ PASS: Empty Reference List
514: ✓ PASS: Many References
515: ✓ PASS: Long Reference Names
516: ✓ PASS: Reference Manager Edge Cases
517: ✓ PASS: Visualization Methods
518: ✓ PASS: Quality Metrics
519: ✓ PASS: Compare Embeddings Edge Cases
520: ```
521: 
522: ---
523: 
524: ## AI WORKFLOW FOR CODE REVIEW
525: 
526: This section documents the established workflow for AI-assisted code review and development.
527: 
528: ### Workflow Overview
529: 
530: ```
531: ┌─────────────────────────────────────────────────────────────────┐
532: │                    AI CODE REVIEW WORKFLOW                       │
533: ├─────────────────────────────────────────────────────────────────┤
534: │                                                                  │
535: │  1. SEARCH & DISCOVER                                           │
536: │     ├── Glob all Python files                                    │
537: │     ├── Read imports and class definitions                       │
538: │     ├── Identify patterns (mock, stub, TODO, FIXME)             │
539: │     └── Check for duplicate exception handlers                   │
540: │                           │                                      │
541: │                           ▼                                      │
542: │  2. TEST & VERIFY                                                │
543: │     ├── Run existing test suite                                  │
544: │     ├── Create edge case tests if missing                        │
545: │     ├── Clear Python cache after edits                           │
546: │     └── Verify imports work correctly                            │
547: │                           │                                      │
548: │                           ▼                                      │
549: │  3. FIX & DOCUMENT                                               │
550: │     ├── Fix critical issues first                                │
551: │     ├── Implement stub methods with real code                    │
552: │     ├── Update ARCHITECTURE.md with new methods                  │
553: │     └── Update CONTEXT.md with findings                          │
554: │                           │                                      │
555: │                           ▼                                      │
556: │  4. FINAL VERIFICATION                                           │
557: │     ├── Run all tests (E2E + Edge Cases)                         │
558: │     ├── Check for remaining stub/mock code                       │
559: │     ├── Verify API routes match UI calls                         │
560: │     └── Confirm imports and instantiations work                  │
561: │                                                                  │
562: └─────────────────────────────────────────────────────────────────┘
563: ```
564: 
565: ### Step 1: Search & Discover Commands
566: 
567: ```bash
568: # Find all Python files
569: find . -name "*.py" -type f
570: 
571: # Search for mock patterns
572: grep -rn "return {}" src/ --include="*.py"
573: grep -rn "Not implemented" src/ --include="*.py" -i
574: grep -rn "TODO\|FIXME\|stub\|mock" src/ --include="*.py" -i
575: 
576: # Check for duplicate exception handlers
577: grep -rn "except Exception" src/ --include="*.py" | head -20
578: 
579: # Check for random embeddings (security issue)
580: grep -rn "np.random.rand.*embed" src/ --include="*.py"
581: 
582: # FIND DUPLICATE CODE - CRITICAL!
583: grep -rn "def " src/ --include="*.py" | sort | uniq -D
584: ```
585: 
586: ### Step 2: Test & Verify Commands
587: 
588: ```bash
589: # Clear Python cache (CRITICAL after edits!)
590: find . -type d -name "__pycache__" -exec rm -rf {} +
591: find . -name "*.pyc" -delete
592: 
593: # Verify syntax compiles
594: python -m py_compile src/embedding/__init__.py
595: python -m py_compile api_server.py
596: 
597: # Run E2E tests
598: python test_e2e_pipeline.py
599: 
600: # Run edge case tests
601: python test_edge_cases.py
602: 
603: # Run unit tests
604: python -m pytest tests/
605: 
606: # Verify imports
607: python -c "from src.detection import *; from src.embedding import *; from src.reference import *; print('All imports OK')"
608: 
609: # Verify API routes
610: python -c "
611: from api_server import app
612: for rule in app.url_map.iter_rules():
613:     print(f'{rule.methods} {rule.rule}')
614: "
615: ```
616: 
617: ### Step 2: Test & Verify Commands
618: 
619: ```bash
620: # Clear Python cache (CRITICAL after edits!)
621: find . -type d -name "__pycache__" -exec rm -rf {} +
622: find . -name "*.pyc" -delete
623: 
624: # Run E2E tests
625: python test_e2e_pipeline.py
626: 
627: # Run edge case tests
628: python test_edge_cases.py
629: 
630: # Run unit tests
631: python -m pytest tests/
632: 
633: # Verify imports
634: python -c "from src.detection import FaceDetector; from src.embedding import FaceNetEmbeddingExtractor; print('Imports OK')"
635: 
636: # Verify API routes
637: python -c "
638: from api_server import app
639: for rule in app.url_map.iter_rules():
640:     print(f'{rule.methods} {rule.rule}')
641: "
642: 
643: # Verify visualization types match
644: grep 'data-viz=' electron-ui/index.html | sed 's/.*data-viz=\"\\([^\"]*\\)\".*/\\1/'
645: grep \"elif viz_type ==\" api_server.py | sed 's/.*== //'
646: ```
647: 
648: ### Step 3: Fix & Document Template
649: 
650: For each critical issue found, document in this format:
651: 
652: ```markdown
653: ### Issue #N: [Brief Title]
654: 
655: **File**: `path/to/file.py`  
656: **Lines**: XXX-XXX  
657: **Severity**: CRITICAL | MEDIUM | MINOR | SECURITY
658: 
659: **Problem**:
660: ```
661: [Code showing the problem]
662: ```
663: 
664: **Fix**:
665: ```python
666: [Fixed code]
667: ```
668: 
669: **Impact**:
670: - [What was broken]
671: - [How the fix resolves it]
672: ```
673: 
674: ### Step 4: Final Verification Checklist
675: 
676: ```bash
677: # Check 1: No stub code remaining
678: grep -rn "return {}" src/ --include="*.py" | wc -l  # Should be 0
679: 
680: # Check 2: All imports work
681: python -c "from src.detection import *; from src.embedding import *; from src.reference import *; print('All imports OK')"
682: 
683: # Check 3: All tests pass
684: python test_e2e_pipeline.py 2>&1 | grep "ALL TESTS PASSED"
685: python test_edge_cases.py 2>&1 | grep "ALL EDGE CASE TESTS PASSED"
686: 
687: # Check 4: API routes match UI
688: # Compare: grep 'data-viz=' electron-ui/index.html vs grep "elif viz_type ==" api_server.py
689: 
690: # Check 5: Visualization methods exist
691: python -c "
692: from src.embedding import FaceNetEmbeddingExtractor
693: e = FaceNetEmbeddingExtractor()
694: methods = ['visualize_embedding', 'visualize_similarity_matrix', 'visualize_similarity_result', 'get_activations']
695: for m in methods:
696:     exists = hasattr(e, m)
697:     print(f'{m}: {\"exists\" if exists else \"MISSING\"}')"
698: ```
699: 
700: ---
701: 
702: ## Medium Issues (Not Fixed - Future Enhancements)
703: 
704: | # | File | Issue |
705: |---|------|-------|
706: | 8 | `api_server.py` | Inconsistent return types in `get_viz_and_data()` |
707: | 9 | Multiple | Unused variables throughout codebase |
708: | 10 | Multiple | Missing type hints in several methods |
709: | 11 | Multiple | Inconsistent exception handling |
710: | 12 | `config_template.py` | Module-level directory creation at import time |
711: | 13 | Multiple | Magic numbers throughout code |
712: 
713: ---
714: 
715: ## Minor Issues (Not Fixed - Nice to Have)
716: 
717: | # | File | Issue |
718: |---|------|-------|
719: | 14 | `tests/*.py` | Duplicate/overlapping tests |
720: | 15 | Multiple | Code duplication (SimilarityComparator, face detection) |
721: | 16 | Multiple | Logging inconsistency (`print()` vs `logging`) |
722: | 17 | `setup.py` | References non-existent `requirements.txt` |
723: | 18 | Multiple | Import organization (some use `sys.path.insert`) |
724: 
725: ---
726: 
727: ## API Reference (Verified)
728: 
729: ### 14 Endpoints
730: 
731: | Method | Endpoint | Description |
732: |--------|----------|-------------|
733: | GET | `/api/health` | Health check |
734: | POST | `/api/detect` | Face detection |
735: | POST | `/api/extract` | Embedding extraction |
736: | POST | `/api/add-reference` | Add reference |
737: | GET | `/api/references` | List references |
738: | POST | `/api/compare` | Compare embeddings |
739: | GET | `/api/visualizations/<type>` | Get visualization |
740: | POST | `/api/clear` | Clear session |
741: | GET | `/api/status` | Debug server state |
742: 
743: ### 14 Visualization Types
744: 
745: | Type | Source | Description |
746: |------|--------|-------------|
747: | `detection` | FaceDetector | Bounding boxes |
748: | `extraction` | FaceDetector | Face ROI |
749: | `landmarks` | FaceDetector | 15 keypoints |
750: | `mesh3d` | FaceDetector | 478-point mesh |
751: | `alignment` | FaceDetector | Pitch/yaw/roll |
752: | `saliency` | FaceDetector | Attention heatmap |
753: | `activations` | EmbeddingExtractor | CNN activations |
754: | `features` | EmbeddingExtractor | Feature maps |
755: | `multiscale` | FaceDetector | Multi-scale |
756: | `confidence` | FaceDetector | Quality metrics |
757: | `embedding` | EmbeddingExtractor | 128-dim bar chart |
758: | `similarity` | EmbeddingExtractor | Similarity bar |
759: | `robustness` | EmbeddingExtractor | Noise test |
760: | `biometric` | FaceDetector | Biometric overview |
761: 
762: ---
763: 
764: ## COMMON MISTAKES & LESSONS LEARNED
765: 
766: ### Mistake 1: Missing Closing Parenthesis
767: 
768: **Problem**: Repeatedly leaving out `)` or `]` in return statements
769: 
770: **Example of ERROR:**
771: ```python
772: # WRONG - missing closing paren
773: return (detector.visualize_landmarks(face_image, landmarks), {}
774: # Also wrong - duplicate code was left behind
775: return extractor.test_robustness(current_face_image)
776: elif viz_type == 'biometric':
777:     return ...
778: ```
779: 
780: **How to prevent:**
781: ```bash
782: # Count parentheses in the edit area
783: grep -o '(' <file> | wc -l
784: grep -o ')' <file> | wc -l
785: # Must be equal!
786: 
787: # Check for duplicate elif chains
788: grep -n "elif viz_type ==" <file>
789: ```
790: 
791: **Correct:**
792: ```python
793: return (detector.visualize_landmarks(face_image, landmarks), {})
794: ```
795: 
796: ---
797: 
798: ### Mistake 2: Duplicate Code Left Behind
799: 
800: **Problem**: Old code not removed when making edits, creating duplicate function definitions
801: 
802: **How to prevent:**
803: ```bash
804: # Check for duplicate function definitions
805: grep -n "def get_viz_and_data" api_server.py
806: 
807: # Check for duplicate elif chains  
808: grep "elif viz_type ==" api_server.py | wc -l
809: # Should match number of viz types (14)
810: ```
811: 
812: ---
813: 
814: ### Mistake 3: Not Running Syntax Check
815: 
816: **Problem**: Submitting code with syntax errors
817: 
818: **How to prevent:**
819: ```bash
820: # ALWAYS run this before submitting edits!
821: python -m py_compile <file>
822: 
823: # For Flask apps, also test imports
824: python -c "import api_server; print('OK')"
825: ```
826: 
827: ---
828: 
829: ### Mistake 4: Not Reading Context Before Editing
830: 
831: **Problem**: Making edits without understanding the surrounding code
832: 
833: **How to prevent:**
834: - Always read at least 50 lines around the edit location
835: - Look for matching parentheses/brackets in the context
836: - Identify if the code is inside a function, class, or conditional
837: 
838: ---
839: 
840: ### Mistake 5: Assuming Code Works
841: 
842: **Problem**: Not testing edge cases after edits
843: 
844: **How to prevent:**
845: - Test with None inputs
846: - Test with empty lists
847: - Test with boundary values
848: - Run the full test suite
849: 
850: ### Mistake 6: Blocking UI with Async/Await in Event Handlers
851: **Problem Encountered (Feb 11, 2026):**
852: ```
853: Using async/await with fetch() in handleImageSelect() caused the UI to freeze.
854: The await blocked the onload callback, preventing image display.
855: Users couldn't upload any images because the upload appeared to do nothing.
856: ```
857: 
858: **Root Cause:**
859: - `reader.onload = async (e) => { await fetch(...) }` blocked the UI thread
860: - The fetch to /api/clear was non-essential but was awaited
861: - UI updates after the await never executed because the fetch never completed
862: 
863: **How to prevent:**
864: - Use fire-and-forget for non-essential API calls
865: - Never await in event handlers unless you need the result
866: - Use `.catch()` for error handling on fire-and-forget calls
867: 
868: **The Fix:**
869: ```javascript
870: // WRONG - blocks UI
871: reader.onload = async (e) => {
872:     await fetch(`${API_BASE}/clear`, { method: 'POST' });  // BLOCKS!
873:     // This code never executes if fetch hangs
874: };
875: 
876: // RIGHT - fire-and-forget
877: reader.onload = (e) => {
878:     fetch(`${API_BASE}/clear`, { method: 'POST' }).catch(err => {
879:         console.log('Clear failed:', err.message);
880:     });
881:     // This code executes immediately
882: };
883: ```
884: 
885: ### Mistake 7: Not Cross-Checking HTML with JavaScript
886: 
887: **Problem Encountered (Feb 11, 2026):**
888: ```
889: During cache/clear feature implementation, multiple functions were lost:
890: - detectFaces() - CRITICAL: No face detection
891: - extractFeatures() - CRITICAL: No embedding extraction
892: - Other helper functions from previous refactoring
893: 
894: Result: Buttons did NOTHING when clicked. Complete UI failure.
895: ```
896: 
897: **Root Cause:**
898: - Added new functions without checking what HTML was calling
899: - Previous edits corrupted the file by overwriting instead of adding
900: - No verification that HTML event handlers matched JS function definitions
901: - Assumed all functions existed when they didn't
902: 
903: **How to prevent:**
904: 1. ALWAYS extract HTML event handlers before making changes:
905: ```bash
906: grep -E 'onclick=|onchange=' electron-ui/index.html
907: ```
908: 
909: 2. ALWAYS verify each HTML function exists in JS:
910: ```bash
911: # Extract and verify all at once
912: HTML_FUNCS=$(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u)
913: for func in $HTML_FUNCS; do
914:     grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js || echo "MISSING: $func"
915: done
916: ```
917: 
918: 3. ALWAYS count functions before and after edits:
919: ```bash
920: BEFORE=$(grep -c '^function ' app.js)
921: AFTER=$(grep -c '^function ' app.js)
922: [ "$BEFORE" != "$AFTER" ] && echo "FUNCTION COUNT CHANGED!"
923: ```
924: 
925: **The Fix Applied (Feb 11, 2026):**
926: ```bash
927: # Added missing critical functions:
928: ✓ detectFaces()  - Face detection API call
929: ✓ extractFeatures() - Embedding extraction API call
930: ✓ selectImage() - File input trigger
931: ✓ addReference() - Reference input trigger
932: ✓ handleReferenceSelect() - Process reference upload
933: ✓ resetSteps() - Reset UI state
934: ✓ saveReference() - Save reference to backend
935: ✓ removeReference() - Remove reference
936: ✓ showReferenceVisualizations() - Show ref viz
937: ✓ compareFaces() - Compare embeddings
938: ✓ And 15+ helper functions...
939: 
940: # Total functions: 25 (17 sync + 8 async)
941: ```
942: 
943: **MANDATORY PRE-COMMIT CHECK:**
944: ```bash
945: #!/bin/bash
946: # Run this before EVERY commit
947: 
948: # 1. Extract HTML handlers
949: echo "HTML handlers:"
950: grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u
951: 
952: # 2. Verify each exists
953: MISSING=""
954: for func in $(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u); do
955:     if ! grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js; then
956:         MISSING="$MISSING $func"
957:     fi
958: done
959: 
960: if [ -n "$MISSING" ]; then
961:     echo "MISSING FUNCTIONS:$MISSING"
962:     echo "FIX BEFORE COMMITTING!"
963:     exit 1
964: fi
965: 
966: echo "✓ All HTML handlers verified in app.js"
967: ```
968: 
969: ---
970: 
971: ## Files Modified During Code Review
972: 
973: | Date | File | Change |
974: |------|------|--------|
975: | Feb 11, 2026 | `CONTEXT.md` | Added strict edit rules and developer checklist |
976: | Feb 11, 2026 | `CONTEXT.md` | Added common mistakes section |
977: | Feb 11, 2026 | `CONTEXT.md` | Created with complete code review findings |
978: | Feb 11, 2026 | `CONTEXT.md` | Added edge case testing section |
979: | Feb 11, 2026 | `CONTEXT.md` | Added AI workflow for code review |
980: | Feb 11, 2026 | `DEVELOPMENT_LOG.md` | Updated with review summary |
981: | Feb 11, 2026 | `ARCHITECTURE.md` | Updated with all methods and viz types |
982: | Feb 11, 2026 | `src/embedding/__init__.py` | Removed duplicate exception handler |
983: | Feb 11, 2026 | `src/embedding/__init__.py` | Fixed wrong attribute access |
984: | Feb 11, 2026 | `src/embedding/__init__.py` | Implemented get_activations() with real activations |
985: | Feb 11, 2026 | `src/embedding/__init__.py` | Added visualize_embedding(), visualize_similarity_matrix(), visualize_similarity_result() |
986: | Feb 11, 2026 | `src/detection/__init__.py` | Added None check to visualize_3d_mesh() |
987: | Feb 11, 2026 | `CONTEXT.md` | Added Rules 6-8 (Function Preservation, Atomic Edits, Exact Matching) |
988: | Feb 11, 2026 | `CONTEXT.md` | Added Mistake 6 (Blocking UI with Async/Await) |
989: | Feb 11, 2026 | `electron-ui/renderer/app.js` | Fixed: handleImageSelect() was lost during edits, restored complete function |
990: | Feb 11, 2026 | `electron-ui/index.html` | Fixed font stack for macOS compatibility |
991: | Feb 11, 2026 | `api_server.py` | Added DELETE /api/references/<id> endpoint |
992: | Feb 11, 2026 | `CONTEXT.md` | Added Rule 9 (HTML-JS Cross-Check) - MANDATORY |
993: | Feb 11, 2026 | `CONTEXT.md` | Added Mistake 7 (Missing HTML-JS Cross-Check) |
994: | Feb 11, 2026 | `electron-ui/renderer/app.js` | Added 10 missing functions: detectFaces, extractFeatures, selectImage, addReference, handleReferenceSelect, saveReference, resetSteps, removeReference, showReferenceVisualizations, compareFaces (FULL RESTORE) |
995: | Feb 11, 2026 | `electron-ui/renderer/app.js` | Total functions: 25 (17 sync + 8 async) |
996: 
997: ---
998: 
999: *Context document created: February 11, 2026*
1000: *Last updated: February 11, 2026*
1001: *All critical issues fixed, no mock code remaining*
1002: *Strict edit rules enforced*
1003: *MANDATORY: Run HTML-JS cross-check before every commit*
1004: | Feb 11, 2026 | `src/detection/__init__.py` | Added validation to compute_quality_metrics() |
1005: | Feb 11, 2026 | `utils/webcam.py` | Added missing imports |
1006: | Feb 11, 2026 | `visualize_biometric.py` | Fixed hardcoded path |
1007: | Feb 11, 2026 | `gui/facial_analysis_gui.py` | Fixed random embedding fallback |
1008: | Feb 11, 2026 | `test_edge_cases.py` | Created comprehensive edge case test suite |
1009: | Feb 11, 2026 | `api_server.py` | Added reference visualization endpoints (`/api/visualizations/<type>/reference/<id>`) |
1010: | Feb 11, 2026 | `CONTEXT.md` | Added Rules 6-8 (Function Preservation, Atomic Edits, Exact Matching) |
1011: | Feb 11, 2026 | `CONTEXT.md` | Added Mistake 6 (Blocking UI with Async/Await) |
1012: | Feb 11, 2026 | `electron-ui/renderer/app.js` | Fixed: handleImageSelect() was lost during edits, restored complete function |
1013: | Feb 11, 2026 | `electron-ui/index.html` | Fixed font stack for macOS compatibility |
1014: | Feb 11, 2026 | `api_server.py` | Added DELETE /api/references/<id> endpoint |
1015: | Feb 11, 2026 | `CONTEXT.md` | Added Rule 9 (HTML-JS Cross-Check) - MANDATORY |
1016: | Feb 11, 2026 | `CONTEXT.md` | Added Mistake 7 (Missing HTML-JS Cross-Check) |
1017: | Feb 11, 2026 | `electron-ui/renderer/app.js` | Added 10+ missing functions: detectFaces, extractFeatures, selectImage, addReference, handleReferenceSelect, saveReference, resetSteps, removeReference, showReferenceVisualizations, compareFaces, setupEventListeners, checkAPI (FULL RESTORE) |
1018: | Feb 11, 2026 | `electron-ui/renderer/app.js` | Total: 18 sync + 9 async functions (27 total) |
1019: | Feb 11, 2026 | `CONTEXT.md` | Added mandatory pre-commit verification script |
1020: | Feb 11, 2026 | `src/embedding/__init__.py` | Fixed visualize_activations() and visualize_feature_maps() |
1021: | Feb 11, 2026 | `electron-ui/renderer/app.js` | Added console logging, better error messages |
1022: | Feb 11, 2026 | `test_e2e_pipeline.py` | Made test images configurable via env vars |
1023: | Feb 11, 2026 | `visualize_biometric.py` | Made test image configurable |
1024: | Feb 11, 2026 | `reference_images/README.md` | Updated documentation |
1025: | Feb 13, 2026 | `api_server.py` | Fixed 9 test viz handlers to return actual data (test-health, test-detection, test-extraction, test-reference, test-multi, test-pose, test-eyewear, test-viz, test-clear) |
1026: | Feb 13, 2026 | `electron-ui/renderer/app.js` | Fixed visualization fetch to accept smaller images, added data-only display support |
1027: | Feb 13, 2026 | `CONTEXT.md` | Added Lesson 15 (Test Viz Data) and Mistake 11 |
1028: | Feb 13, 2026 | `src/detection/__init__.py` | Fixed eyewear detection - use brightness as primary, eye cascade as secondary |
1029: | Feb 13, 2026 | `electron-ui/renderer/app.js` | Added webcam functions (startWebcam, captureWebcam, stopWebcam) |
1030: | Feb 13, 2026 | `test_eyewear_frontend.js` | Created frontend integration test for eyewear detection |
1031: 
1032: ---
1033: 
1034: ## NEW: Lessons Learned - Persistence & Process Management (Feb 11, 2026)
1035: 
1036: ### Lesson 10: References Not Persisting to JSON
1037: 
1038: **Problem**: References were stored in memory but not saved to `embeddings.json`. When API server restarted, references were lost.
1039: 
1040: **Solution**: Added `save_references()` and `load_references()` functions to `api_server.py`:
1041: ```python
1042: REFERENCES_FILE = os.path.join(os.path.dirname(__file__), 'reference_images', 'embeddings.json')
1043: 
1044: def save_references():
1045:     """Save references to JSON file."""
1046:     data = {'metadata': [...], 'embeddings': [...]}
1047:     with open(REFERENCES_FILE, 'w') as f:
1048:         json.dump(data, f, indent=2)
1049: 
1050: def load_references():
1051:     """Load references from JSON file on startup."""
1052:     global references
1053:     if os.path.exists(REFERENCES_FILE) and os.path.getsize(REFERENCES_FILE) > 0:
1054:         with open(REFERENCES_FILE, 'r') as f:
1055:             references = json.load(f).get('references', [])
1056: 
1057: load_references()  # Call at module load
1058: ```
1059: 
1060: **Remember**: Call `save_references()` after any mutation:
1061: - After `add_reference()` succeeds
1062: - After `remove_reference()` succeeds
1063: 
1064: ### Lesson 11: Electron App Not Loading Existing References
1065: 
1066: **Problem**: App started with empty `references = []`, never loading from API.
1067: 
1068: **Solution**: Added `loadReferences()` function called from `checkAPI()`:
1069: ```javascript
1070: async function checkAPI() {
1071:     const response = await fetch(`${API_BASE}/health`);
1072:     const data = await response.json();
1073:     if (data.status === 'ok') {
1074:         logToTerminal('> API connected', 'success');
1075:         loadReferences();  // NEW - load existing refs
1076:     }
1077: }
1078: 
1079: async function loadReferences() {
1080:     const response = await fetch(`${API_BASE}/references`);
1081:     const data = await response.json();
1082:     if (data.references) {
1083:         references = data.references;
1084:         updateReferenceList();
1085:     }
1086: }
1087: ```
1088: 
1089: ### Lesson 12: Old Python Processes Cache Code
1090: 
1091: **Problem**: After editing `api_server.py`, changes didn't take effect. Old process was still running.
1092: 
1093: **Root Cause**: Python caches `.pyc` files and the old process didn't restart.
1094: 
1095: **Solution**: Always restart the API server after code changes:
1096: ```bash
1097: # Kill existing
1098: pkill -f "python api_server.py"
1099: 
1100: # Clear cache
1101: find . -type d -name "__pycache__" -exec rm -rf {} +
1102: find . -name "*.pyc" -delete
1103: 
1104: # Restart
1105: python api_server.py
1106: ```
1107: 
1108: **Or use `start.sh`** which does all this automatically.
1109: 
1110: ### Lesson 13: Test Images Had Wrong Default Paths
1111: 
1112: **Problem**: `test_e2e_pipeline.py` defaulted to `kanye_west.jpeg` which doesn't exist.
1113: 
1114: **Solution**: Updated defaults to actual files:
1115: ```python
1116: TEST_IMAGE = os.environ.get('TEST_IMAGE', 'test_subject.jpg')
1117: TEST_IMAGE_REF = os.environ.get('TEST_IMAGE_REF', 'reference_subject.jpg')
1118: ```
1119: 
1120: **Files that exist**:
1121: - `test_images/test_subject.jpg`
1122: - `test_images/reference_subject.jpg`
1123: 
1124: ### Lesson 14: Use start.sh for Clean Startup
1125: 
1126: **Problem**: Manual server starts leave old processes running.
1127: 
1128: **Solution**: Use `start.sh`:
1129: ```bash
1130: #!/bin/bash
1131: # Kill any existing API server
1132: pkill -f "python api_server.py" 2>/dev/null
1133: 
1134: # Clear Python cache
1135: find . -type d -name "__pycache__" -exec rm -rf {} +
1136: find . -name "*.pyc" -delete
1137: find . -name ".pytest_cache" -exec rm -rf {} +
1138: 
1139: # Start API server
1140: cd "$(dirname "$0")"
1141: python api_server.py &
1142: 
1143: # Start Electron UI
1144: cd "$(dirname "$0")/electron-ui"
1145: npm start &
1146: ```
1147: 
1148: ### Lesson 15: Test Visualization Data Not Displaying
1149: 
1150: **Problem**: Test tabs (test-health, test-detection, etc.) showed images but no text data below them.
1151: 
1152: **Root Cause**: API handlers returned empty `{}` as second tuple element:
1153: ```python
1154: # WRONG - returns empty dict
1155: return visualize_test_detail("Health Check", {"status": "OK"}), {}
1156: ```
1157: 
1158: **Solution**: Return actual data dict as second tuple element:
1159: ```python
1160: # CORRECT - returns actual data
1161: data = {"status": "OK", "api": "running", "port": 3000}
1162: return visualize_test_detail("Health Check", data), data
1163: ```
1164: 
1165: **Frontend**: Already had `formatDataAsTable()` and CSS - just needed data from API.
1166: 
1167: **Frontend Logic Fix**: Also updated to handle smaller images and data-only display:
1168: ```javascript
1169: // Changed from: data.visualization.length > 100
1170: // To: accept any visualization size
1171: if (data.success && data.visualization) {
1172:     visualizationData[vizType] = data.visualization;
1173:     // ...store data
1174: }
1175: ```
1176: 
1177: **Remember**: When adding new test viz handlers, always return data as second tuple element!
1178: 
1179: ### Lesson 16: Always Use Project Virtual Environment
1180: 
1181: **Problem**: Old API server process kept running with wrong Python, causing bugs to persist even after code fixes.
1182: 
1183: **Root Cause**: 
1184: - Started API with system Python (`/opt/homebrew/Cellar/python@3.14/...`) instead of project venv
1185: - Used `python` command instead of `source venv/bin/activate && python`
1186: - Old process cached old code, so edits had no effect
1187: 
1188: **Solution**: Always use the project's virtual environment:
1189: ```bash
1190: # CORRECT - use project venv
1191: cd /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo
1192: source venv/bin/activate
1193: python api_server.py
1194: 
1195: # OR use start.sh which does this automatically
1196: ./start.sh
1197: ```
1198: 
1199: **How to verify**: Check which Python is running:
1200: ```bash
1201: # Find the process
1202: ps aux | grep "api_server\|python.*3000" | grep -v grep
1203: 
1204: # Kill old process if needed
1205: kill <PID>
1206: 
1207: # Verify new process uses correct Python
1208: lsof -p <new_pid> | grep Python
1209: # Should show: .../MANTAX/face_recognition_npo/venv/bin/python
1210: ```
1211: 
1212: **REMEMBER**: Always restart the API server after making code changes!
1213: 
1214: ### Lesson 17: Eyewear Detection False Positives
1215: 
1216: **Problem**: Every uploaded image showed "glasses detected" even without eyewear.
1217: 
1218: **Root Cause**: 
1219: 1. OpenCV eye cascade is unreliable - often returns 0 eyes even without glasses
1220: 2. Code automatically flagged eyewear if eye_count == 0, regardless of actual eye brightness
1221: 3. Thresholds were too low (0.4 for brightness ratio, 0.15 for edge density)
1222: 
1223: **Solution**: 
1224: 1. Analyze brightness/edge FIRST as primary detection method
1225: 2. Use eye cascade ONLY as confirmation, not primary
1226: 3. Require BOTH conditions to be true (dark eyes + high edge density)
1227: 4. Set stricter thresholds: brightness < 0.2 for sunglasses, < 0.35 for possible
1228: 5. Only flag eyewear if brightness analysis confirms, not just eye cascade failure
1229: 
1230: **Key Code Pattern**:
1231: ```python
1232: # WRONG - eye cascade alone triggers detection
1233: if eye_count == 0:
1234:     eyewear_detected = True  # FALSE POSITIVE!
1235: 
1236: # CORRECT - require brightness confirmation
1237: eyes_detected = self.detect_eyes(face_image)
1238: eye_count = len(eyes_detected)
1239: 
1240: if eye_count == 0 and not eyewear_detected:
1241:     # Only flag if brightness is VERY dark
1242:     if brightness_ratio < 0.15:
1243:         eyewear_detected = True
1244:     else:
1245:         # Eye cascade likely just failed - don't flag
1246:         eye_count = 2  # Assume normal
1247: ```
1248: 
1249: **Testing**: Always test BOTH:
1250: - Normal face (should NOT detect eyewear)
1251: - Face with sunglasses (should detect eyewear)
1252: 
1253: **Frontend Test Script**: `test_eyewear_frontend.js`
1254: 
1255: ---
1256: 
1257: ## Common Mistakes Summary
1258: 
1259: | # | Mistake | Solution |
1260: |---|---------|----------|
1261: | 1 | Missing closing paren | Count `(`, `)`, `[`, `]` |
1262: | 2 | Duplicate code left behind | `grep -n "def "` before/after |
1263: | 3 | Not running syntax check | `python -m py_compile <file>` |
1264: | 4 | Not reading context | Read 50+ lines before edit |
1265: | 5 | Not testing edge cases | Run tests after every edit |
1266: | 6 | Blocking UI with async/await | Use `.catch()` for fire-and-forget |
1267: | 7 | Missing HTML-JS cross-check | Verify all onclick/onchange handlers |
1268: | 8 | References not persisting | Call `save_references()` after add/remove |
1269: | 9 | Old process caching code | Use `./start.sh` to restart |
1270: | 10 | Wrong test image paths | Use `test_subject.jpg`, `reference_subject.jpg` |
1271: | 11 | Test viz returning empty data | Return actual data dict as second tuple element |
1272: | 12 | Using wrong Python environment | Always use `source venv/bin/activate` before running |
1273: | 13 | Eyewear detection false positives | Use brightness analysis as primary, eye cascade as secondary |
1274: 
1275: ---
1276: 
1277: ## New Features: Dual-Model Comparison (February 13, 2026)
1278: 
1279: ### Overview
1280: The application now uses BOTH ArcFace AND FaceNet together for more robust matching.
1281: 
1282: ### Dual-Model Architecture
1283: 
1284: | Component | Description |
1285: |-----------|-------------|
1286: | ArcFace Extractor | 512-dim embeddings (primary) |
1287: | FaceNet Extractor | 128-dim embeddings (secondary) + activations |
1288: | Both always loaded | Even if USE_ARCFACE=true |
1289: 
1290: ### Comparison Weights
1291: 
1292: | Factor | Weight | Purpose |
1293: |--------|--------|---------|
1294: | ArcFace similarity | 60% | Primary - best discrimination |
1295: | FaceNet similarity | 20% | Additional signal |
1296: | Landmark geometry | 15% | Structural consistency |
1297: | Activation similarity | 5% | Neural network internal representations |
1298: | Image quality | 5% | Reliability factor |
1299: 
1300: **Note**: No pose penalty - same person with different poses will match correctly.
1301: 
1302: ### New API Fields
1303: 
1304: Each comparison result now returns:
1305: ```json
1306: {
1307:   "arcface_similarity": 0.75,
1308:   "facenet_similarity": 0.60,
1309:   "landmark_similarity": 0.85,
1310:   "activation_similarity": 0.72,
1311:   "final_score": 0.68,
1312:   "status": "match|possible|no_match",
1313:   "match_label": "Full Match|Possible Match|No Match",
1314:   "reasons": ["ArcFace: High similarity (75%)", "FaceNet: Moderate (60%)", "Similar neural activation patterns"]
1315: }
1316: ```
1317: 
1318: ### Backwards Compatibility
1319: 
1320: - Old references with single embedding are automatically handled
1321: - If ArcFace unavailable, falls back to FaceNet-only matching
1322: - `/api/diagnostics` endpoint added to check system status
1323: 
1324: ---
1325: 
1326: ## MediaPipe Landmark Detection (February 13, 2026)
1327: 
1328: ### Issue Resolved
1329: Previously, landmarks visualization showed only ~10 points (fallback mode). Now shows full 468 MediaPipe landmarks.
1330: 
1331: ### Required Files
1332: - `face_landmark.task` - MediaPipe model (downloaded to project root)
1333: - MediaPipe Python package installed
1334: 
1335: ### Verification
1336: Check `/api/diagnostics` endpoint:
1337: ```json
1338: {
1339:   "mediapipe": "available",
1340:   "model_file_exists": true,
1341:   "arcface_extractor": "loaded",
1342:   "dual_model_mode": true
1343: }
1344: ```
1345: 
1346: ---
1347: 
1348: ## Test Tab Renaming (February 13, 2026)
1349: 
1350: | Old Name | New Name |
1351: |----------|----------|
1352: | Test 1 | API Health |
1353: | Test 2 | Detection |
1354: | Test 3 | Extraction |
1355: | Test 4 | References |
1356: | Test 5 | Multi-Match |
1357: | Test 6 | Pose |
1358: | Test 7 | Eyewear |
1359: | Test 8 | Viz Types |
1360: | Test 9 | Session |
1361: 
1362: ---
1363: 
1364: ## Files to Know
1365: 
1366: | File | Purpose | When to Edit |
1367: |------|---------|--------------|
1368: | `api_server.py` | Flask API (12+ endpoints) | Backend changes |
1369: | `electron-ui/renderer/app.js` | Frontend logic | UI changes |
1370: | `electron-ui/index.html` | UI structure | HTML changes |
1371: | `src/embedding/__init__.py` | SimilarityComparator | Matching logic |
1372: | `src/detection/__init__.py` | FaceDetector | Landmarks, pose |
1373: | `start.sh` | Startup script | Server config |
1374: | `test_e2e_pipeline.py` | E2E tests | Test changes |
1375: | `reference_images/embeddings.json` | Persistent storage | Auto-generated |
1376: 
1377: ---
1378: 
1379: ## Quick Verification Checklist
1380: 
1381: Before telling user "done" or "ready to commit":
1382: 
1383: ```bash
1384: # 1. API syntax
1385: python -m py_compile api_server.py && echo "✓ API OK"
1386: 
1387: # 2. Tests pass
1388: python test_e2e_pipeline.py 2>&1 | grep -q "ALL TESTS PASSED" && echo "✓ Tests OK"
1389: 
1390: # 3. Cache cleared
1391: find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
1392: find . -name "*.pyc" -delete 2>/dev/null
1393: echo "✓ Cache cleared"
1394: 
1395: # 4. Server restarted (if needed)
1396: # pkill -f "python api_server.py" 2>/dev/null
1397: # python api_server.py &
1398: ```
1399: 
1400: ---
1401: 
1402: *Context document created: February 11, 2026*
1403: *Last updated: February 13, 2026*
1404: *All critical issues fixed*
1405: *Dual-model comparison implemented*
1406: *MediaPipe landmarks fixed*
1407: *MANDATORY: Run pre-commit verification before every commit*
1408: 

(End of file - total 1408 lines)
</content>
```

---

## Assistant (Build · minimax-m2.5-free · 13.6s)

_Thinking:_

Good progress. Let me continue reading the remaining .md files in the face_recognition_npo directory.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/PROJECT_STRUCTURE.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/PROJECT_STRUCTURE.md</path>
<type>file</type>
<content>1: # NGO Facial Image Analysis System - Project Structure
2: 
3: **Version**: 0.4.1
4: **Last Updated**: February 14, 2026
5: **Status**: ✅ Fully Functional - Reference Details + UI Improvements
6: 
7: ---
8: 
9: ## Executive Summary
10: 
11: A complete, working facial recognition system for ethical NGO use. The system uses ArcFace 512-dimensional embeddings for excellent discrimination between different people.
12: 
13: **Key Achievement**: ArcFace shows <30% similarity for different people (correct!), vs FaceNet's 65-70% (false positives!).
14: 
15: ## Quick Start
16: 
17: ```bash
18: cd face_recognition_npo
19: ./start.sh
20: # Select option 1 (Flask API + Browser) or 2 (Electron Desktop)
21: ```
22: 
23: ---
24: 
25: ## System Architecture
26: 
27: ```
28: ┌─────────────────────────────────────────────────────────────────────────┐
29: │                         ENTRY POINTS                                     │
30: ├─────────────────────────────────────────────────────────────────────────┤
31: │  ./start.sh           → Interactive menu                                │
32: │  python api_server.py → Flask API Server :3000                          │
33: │  npm start           → Electron Desktop App (connects to Flask)         │
34: │  python gui/*.py      → Tkinter Standalone GUIs                       │
35: └─────────────────────────────────────────────────────────────────────────┘
36:                                      │
37:                                      ▼
38: ┌─────────────────────────────────────────────────────────────────────────┐
39: │                    ELECTRON DESKTOP APP                                  │
40: ├─────────────────────────────────────────────────────────────────────────┤
41: │  main.js              → Connects to existing Flask server              │
42: │  renderer/app.js      → Frontend JavaScript (HTTP API calls)           │
43: │  index.html           → Ultra minimal UI with MANTAX navbar             │
44: │                                                                          │
45: │  Flow: User → UI → fetch() → Flask API → ML Models → Results          │
46: └─────────────────────────────────────────────────────────────────────────┘
47:                                      │
48:                            HTTP :3000 (REST API)
49:                                      │
50:                                      ▼
51: ┌─────────────────────────────────────────────────────────────────────────┐
52: │                     FLASK API SERVER (BACKEND)                           │
53: ├─────────────────────────────────────────────────────────────────────────┤
54: │  Endpoints:                                                              │
55: │  • GET  /api/health           → Status check                           │
56: │  • GET  /api/embedding-info   → Model info (ArcFace/FaceNet)           │
57: │  • POST /api/detect           → Face detection                         │
58: │  • POST /api/extract          → Embedding extraction                   │
59: │  • POST /api/add-reference    → Add reference                          │
60: │  • GET  /api/references      → List references                        │
61: │  • DELETE /api/references/<id> → Remove reference                      │
62: │  • POST /api/compare          → Similarity comparison                   │
63: │  • GET  /api/visualizations/<type> → Get visualization                 │
64: │  • POST /api/clear           → Clear session                           │
65: │  • GET  /api/status          → Debug server state                      │
66: │                                                                          │
67: │  In-Memory Session:                                                      │
68: │  • current_image, current_faces, current_embedding                       │
69: │  • references (in-memory list)                                            │
70: └─────────────────────────────────────────────────────────────────────────┘
71:                                      │
72:                                      ▼
73: ┌─────────────────────────────────────────────────────────────────────────┐
74: │                      CORE ML PIPELINE                                     │
75: ├─────────────────────────────────────────────────────────────────────────┤
76: │                                                                          │
77: │   ┌──────────────────┐                                                   │
78: │   │   FaceDetector   │  OpenCV DNN (Caffe model)                        │
79: │   │ Input: Image     │  Output: Bounding boxes (x, y, w, h)              │
80: │   └────────┬─────────┘                                                    │
81: │            │                                                              │
82: │            ▼                                                              │
83: │   ┌─────────────────────────────────────────────────────┐                │
84: │   │           Embedding Extractor                         │                │
85: │   │  ┌─────────────────────┐  ┌─────────────────────┐   │                │
86: │   │  │  ArcFace (Default)  │  │  FaceNet (Option)  │   │                │
87: │   │  │  ONNX / ResNet100   │  │  PyTorch / ResNet18│   │                │
88: │   │  │  512-dimensional    │  │  128-dimensional   │   │                │
89: │   │  │  <30% for diff!    │  │  ~70% for diff    │   │                │
90: │   │  └─────────────────────┘  └─────────────────────┘   │                │
91: │   │                                                      │                │
92: │   │ Input: Face ROI                                       │                │
93: │   │ Output: 512-dim or 128-dim embedding                 │                │
94: │   └────────────────────────────┬──────────────────────────┘                │
95: │                                │                                            │
96: │                                ▼                                            │
97: │   ┌──────────────────┐                                                   │
98: │   │ Similarity       │  Cosine similarity → Confidence bands              │
99: │   │ Comparator       │  ArcFace: ≥70%, 45-70%, 30-45%, <30%             │
100: │   └──────────────────┘                                                   │
101: │                                                                          │
102: └─────────────────────────────────────────────────────────────────────────┘
103:                                      │
104:                                      ▼
105: ┌─────────────────────────────────────────────────────────────────────────┐
106: │                     REFERENCE STORAGE                                     │
107: ├─────────────────────────────────────────────────────────────────────────┤
108: │  ReferenceImageManager                                                   │
109: │  • Stores references in reference_images/embeddings.json                 │
110: │  • Metadata: id, path, consent, timestamp                              │
111: │  • Embeddings: 512-dim (ArcFace) or 128-dim (FaceNet)                  │
112: │  • Persistence: Auto-saves to JSON on add/remove                        │
113: └─────────────────────────────────────────────────────────────────────────┘
114: ```
115: 
116: ---
117: 
118: ## Directory Structure
119: 
120: ```
121: face_recognition_npo/
122: ├── start.sh                        # Interactive start script
123: ├── api_server.py                   # Flask API server (BACKEND)
124: ├── test_e2e_pipeline.py            # End-to-end test script
125: ├── test_api_endpoints.py           # API verification script
126: ├── src/
127: │   ├── detection/__init__.py       # FaceDetector (OpenCV DNN)
128: │   ├── embedding/
129: │   │   ├── __init__.py            # FaceNetEmbeddingExtractor, SimilarityComparator
130: │   │   └── arcface_extractor.py   # ArcFaceEmbeddingExtractor (ONNX)
131: │   └── reference/__init__.py       # ReferenceImageManager, HumanReviewInterface
132: ├── gui/
133: │   ├── facial_analysis_gui.py       # Apple-styled Tkinter GUI
134: │   └── user_friendly_gui.py        # Step-by-step wizard GUI
135: ├── electron-ui/                     # Electron Desktop Application
136: │   ├── package.json                # NPM dependencies
137: │   ├── main.js                     # Electron main process
138: │   ├── preload.js                  # Context bridge
139: │   ├── index.html                  # Ultra minimal UI with MANTAX navbar
140: │   ├── renderer/
141: │   │   └── app.js                 # Frontend JavaScript
142: │   └── styles/                     # CSS files
143: ├── tests/                           # Unit tests
144: ├── utils/webcam.py                  # WebcamCapture class
145: ├── examples/                        # Usage examples
146: ├── test_images/                     # 25+ test images
147: ├── reference_images/                # Reference storage
148: │   ├── embeddings.json             # Stored references
149: │   └── README.md
150: │
151: ├── deploy.prototxt.txt             # OpenCV DNN config
152: ├── res10_300x300_ssd_iter_140000.caffemodel  # Face detection model
153: ├── arcface_model.onnx              # ArcFace ONNX model (117MB)
154: │
155: └── *.md                            # Documentation
156: ```
157: 
158: ---
159: 
160: ## API Endpoints
161: 
162: | Method | Endpoint | Description | Status |
163: |--------|----------|-------------|--------|
164: | GET | `/api/health` | Health check | ✅ |
165: | GET | `/api/embedding-info` | Model/dimension info | ✅ |
166: | POST | `/api/detect` | Detect faces in uploaded image | ✅ |
167: | POST | `/api/extract` | Extract embedding from detected face | ✅ |
168: | POST | `/api/add-reference` | Add reference image for comparison | ✅ |
169: | GET | `/api/references` | List all references | ✅ |
170: | DELETE | `/api/references/<id>` | Remove reference | ✅ |
171: | POST | `/api/compare` | Compare query embedding with references | ✅ |
172: | GET | `/api/visualizations/<type>` | Get query visualization | ✅ |
173: | GET | `/api/visualizations/<type>/reference/<id>` | Get reference visualization | ✅ |
174: | POST | `/api/clear` | Clear all session data | ✅ |
175: | GET | `/api/status` | Debug server state | ✅ |
176: 
177: ---
178: 
179: ## 14 AI Visualizations
180: 
181: | Visualization | Description | API Endpoint |
182: |--------------|-------------|--------------|
183: | Detection | Bounding boxes with confidence scores | `/api/detect` |
184: | Extraction | 160x160 face crop preview | `/api/detect` |
185: | Landmarks | Facial feature points | `/api/visualizations/landmarks` |
186: | 3D Mesh | 3D wireframe overlay | `/api/visualizations/mesh3d` |
187: | Alignment | Face alignment visualization | `/api/visualizations/alignment` |
188: | Attention | Saliency/attention heatmap | `/api/visualizations/saliency` |
189: | Activations | NN layer activations (placeholder for ArcFace) | `/api/visualizations/activations` |
190: | Features | Feature map visualizations | `/api/visualizations/features` |
191: | Multi-Scale | Face at 5 scales | `/api/visualizations/multiscale` |
192: | Confidence | Quality metrics dashboard | `/api/visualizations/confidence` |
193: | Embedding | 512-dim or 128-dim vector viz | `/api/extract` |
194: | Similarity | Similarity matrix | `/api/compare` |
195: | Robustness | Noise robustness test | `/api/visualizations/robustness` |
196: | Biometric | Biometric capture quality | `/api/detect` |
197: 
198: ---
199: 
200: ## Test Results (February 12, 2026)
201: 
202: ```bash
203: $ python test_e2e_pipeline.py
204: 
205: ============================================================
206: END-TO-END FACE RECOGNITION PIPELINE TEST
207: ============================================================
208: 
209: [TEST 1] Face Detection Pipeline         ✅ PASS
210: [TEST 2] Embedding Extraction Pipeline   ✅ PASS
211: [TEST 3] Reference Manager (REAL emb)   ✅ PASS
212: [TEST 4] Same Image Similarity          ✅ 100%
213: [TEST 5] Different Images Similarity    ✅ ArcFace: ~9-25% (correct!)
214: [TEST 6] Full Reference Comparison      ✅ PASS
215: 
216: ALL TESTS PASSED - Pipeline is working correctly!
217: ```
218: 
219: ---
220: 
221: ## ArcFace vs FaceNet Comparison
222: 
223: | Metric | ArcFace (Default) | FaceNet (Optional) |
224: |--------|-------------------|-------------------|
225: | **Dimension** | 512 | 128 |
226: | **Backbone** | ResNet100 (ONNX) | ResNet18 (PyTorch) |
227: | **Discrimination** | Excellent | Poor |
228: | **Same Person** | ~70-85% | ~85-99% |
229: | **Different Person** | <30% | ~65-70% |
230: | **False Positive Risk** | Low | High |
231: | **Inference Speed** | Fast (ONNX) | Slower (PyTorch) |
232: 
233: **Why ArcFace is Default**:
234: - FaceNet showed 65-70% similarity for different people (false positives!)
235: - ArcFace shows <30% for different people (correctly indicates different!)
236: - Much safer for NGO use cases
237: 
238: ---
239: 
240: ## MANTAX Branding
241: 
242: The Electron UI includes MANTAX branding:
243: - **Navbar**: White background with subtle border
244: - **Logo**: SVG with red (#D20A11) and white colors
245: - **Tagline**: "Ihrem Partner für Autokrane und Schwerlastlogistik" (right side)
246: - **Position**: Fixed at top, visible on all pages
247: 
248: ---
249: 
250: ## CRITICAL LESSONS LEARNED (Don't Repeat!)
251: 
252: ### Lesson 1: ArcFace vs FaceNet Discrimination
253: 
254: **Problem Encountered**:
255: FaceNet 128-dim embeddings showed 65-70% similarity for different people - this caused false positives!
256: 
257: **The Solution**:
258: ArcFace 512-dim embeddings show <30% for different people:
259: ```
260: Different people: ~9-25% (correctly indicates different!)
261: Same person: ~70-85% (correctly indicates same!)
262: ```
263: 
264: **Rule**: Use ArcFace for better discrimination in NGO use cases.
265: 
266: ---
267: 
268: ### Lesson 2: Dynamic Array Sizes for Visualizations
269: 
270: **Problem Encountered**:
271: ```
272: could not broadcast input array from shape (650,650,3) into shape (150,300,3)
273: ```
274: 
275: **Root Cause**: Hardcoded array size `(150, 300)` in `visualize_similarity_matrix()`.
276: 
277: **The Fix**:
278: ```python
279: # BEFORE (BROKEN):
280: output = np.zeros((150, 300, 3), dtype=np.uint8)
281: # ...
282: output[:n * cell_size, :n * cell_size] = matrix_colored
283: 
284: # AFTER (FIXED):
285: output_size = max(150, n * cell_size)
286: output = np.zeros((output_size, output_size, 3), dtype=np.uint8)
287: output.fill(245)
288: resized = cv2.resize(matrix_colored, (n * cell_size, n * cell_size))
289: output[:resized.shape[0], :resized.shape[1]] = resized
290: ```
291: 
292: **Rule**: Always use dynamic array sizes when output depends on input count.
293: 
294: ---
295: 
296: ### Lesson 3: Reference Embeddings Must Be REAL
297: 
298: **Problem Encountered**: `ReferenceImageManager` was using random embeddings.
299: 
300: **Root Cause**:
301: ```python
302: # BEFORE (BROKEN):
303: embedding = np.random.rand(128)  # FAKE!
304: ```
305: 
306: **The Fix**:
307: ```python
308: # AFTER (FIXED):
309: if self.embedding_extractor is not None:
310:     detector_to_use = self.detector
311:     if faces:
312:         x, y, w, h = faces[0]
313:         face_roi = image_array[y:y+h, x:x+w]
314:         embedding = self.embedding_extractor.extract_embedding(face_roi)
315: ```
316: 
317: **Rule**: Never use random/placeholder values for embeddings.
318: 
319: ---
320: 
321: ### Lesson 4: Clear Python Cache After Editing
322: 
323: **Problem Encountered**: Changes to source files weren't picked up.
324: 
325: **Root Cause**: Python's `.pyc` bytecode cache.
326: 
327: **The Fix**:
328: ```bash
329: find . -type d -name "__pycache__" -exec rm -rf {} +
330: find . -name "*.pyc" -delete
331: ```
332: 
333: **Rule**: Clear Python cache after editing source files.
334: 
335: ---
336: 
337: ### Lesson 5: Verify API with Test Script
338: 
339: **Rule**: Always test API directly before debugging frontend:
340: ```python
341: r = requests.post('http://localhost:3000/api/detect', json={'image': base64_image})
342: r = requests.post('http://localhost:3000/api/extract', json={'face_id': 0})
343: r = requests.post('http://localhost:3000/api/add-reference', json={...})
344: r = requests.post('http://localhost:3000/api/compare', json={})
345: r = requests.get('http://localhost:3000/api/embedding-info')
346: ```
347: 
348: ---
349: 
350: ### Lesson 6: ArcFace ONNX Model Limitations
351: 
352: **Problem**: ArcFace ONNX doesn't expose internal layers like PyTorch.
353: 
354: **Solution**: Use placeholder visualizations that show useful info:
355: - For activations: Show embedding channel groups instead of raw CNN
356: - For feature maps: Show pre-computed visualizations
357: - Always document this limitation
358: 
359: ---
360: 
361: ## Workflow (How to Use)
362: 
363: ### Step 1: Choose Photo
364: - Upload an image file (JPG, PNG, BMP)
365: 
366: ### Step 2: Find Faces
367: - Click "Find Faces" button
368: - AI detects faces and shows thumbnails
369: 
370: ### Step 3: Create Signature
371: - Click "Create Signature" button
372: - Extracts 512-dim (ArcFace) or 128-dim (FaceNet) embedding
373: 
374: ### Step 4: Add Reference
375: - Upload a reference image
376: - System extracts embedding and stores it
377: 
378: ### Step 5: Compare
379: - Click "Compare" button
380: - Shows similarity scores with confidence bands
381: - Use overlay/difference/side-by-side tabs for visual comparison
382: 
383: ### Step 6: View Reference Details (Optional)
384: - Click on any reference image to view:
385:   - Face detection, landmarks, embedding visualization
386:   - Pose (yaw/pitch/roll), saliency/attention map
387:   - Quality metrics (brightness, sharpness, overall)
388: 
389: ---
390: 
391: ## Expected Similarity Scores (ArcFace)
392: 
393: | Scenario | Similarity | Confidence |
394: |----------|------------|------------|
395: | Same image | ~100% | Very High |
396: | Same person | ~70-85% | High |
397: | Different person | <30% | Insufficient |
398: 
399: **Note**: Similarity varies based on:
400: - Image quality
401: - Lighting conditions
402: - Face angle
403: - Expression differences
404: 
405: ---
406: 
407: ## ArcFace Thresholds
408: 
409: | Threshold | Confidence | Interpretation |
410: |-----------|------------|----------------|
411: | ≥70% | Very High | Likely same person |
412: | 45-70% | High | Possibly same person |
413: | 30-45% | Moderate | Human review recommended |
414: | <30% | Insufficient | Likely different people |
415: 
416: ---
417: 
418: ## Roadmap
419: 
420: ### Completed ✅ (v0.4.1)
421: 
422: **Core Pipeline**
423: - [x] Face detection with OpenCV DNN
424: - [x] 512-dim embedding extraction (ArcFace ONNX)
425: - [x] 128-dim embedding extraction (FaceNet PyTorch) - legacy
426: - [x] Cosine similarity comparison
427: - [x] Confidence bands (ArcFace: ≥70%, 45-70%, 30-45%, <30%)
428: - [x] Auto-model selection (ArcFace if available)
429: - [x] Activation similarity comparison (neural network activations)
430: 
431: **User Interfaces**
432: - [x] Electron desktop app (connects to Flask)
433: - [x] Flask API server with 14+ endpoints
434: - [x] Tkinter GUI
435: - [x] Ultra minimal UI (black on white, no icons)
436: - [x] Sticky terminal footer
437: - [x] MANTAX navbar branding
438: - [x] Reference details panel with visualization tabs
439: - [x] Smart compare button (only enables when both embedding AND references exist)
440: - [x] Card layout for comparison results
441: - [x] Step hints and indicators
442: 
443: **Reference Management**
444: - [x] Reference storage in JSON format
445: - [x] Metadata tracking (consent, source, timestamp)
446: - [x] Real embeddings (NOT random!)
447: - [x] Persistence across restarts
448: - [x] Activation storage with references
449: - [x] Reference visualization (detection, landmarks, embedding, pose, attention, quality)
450: 
451: **Visualizations**
452: - [x] 14 AI visualization types
453: - [x] Dynamic array sizes (fixed broadcast bug!)
454: - [x] ArcFace placeholder visualizations
455: - [x] Reference-specific visualizations
456: 
457: **Testing & Tools**
458: - [x] End-to-end test script
459: - [x] Interactive start script
460: - [x] API verification script
461: 
462: ### In Progress
463: - [ ] Live webcam landmark overlay
464: 
465: ### Future Enhancements
466: - [ ] Batch processing API
467: - [ ] Cloud storage integration
468: - [ ] Mobile app
469: - [ ] WebSocket for real-time updates
470: 
471: ---
472: 
473: ## Dependencies
474: 
475: ```
476: opencv-python>=4.8.0       # Face detection (DNN module)
477: numpy>=1.24.0              # Array operations
478: onnxruntime>=1.15.0        # ONNX Runtime for ArcFace
479: torch>=2.0.0              # PyTorch models (FaceNet, optional)
480: torchvision>=0.15.0        # Pre-trained models (ResNet18)
481: pillow>=10.0.0             # Image processing
482: flask                      # HTTP server
483: flask-cors                 # CORS support
484: ```
485: 
486: ---
487: 
488: ## Ethical Design
489: 
490: This system is built with ethical principles for NGO use:
491: 
492: 1. **Consent-Based**: All images require documented consent
493: 2. **Human Oversight**: No automated decisions - human review required
494: 3. **Uncertainty Handling**: Confidence bands instead of binary decisions
495: 4. **Privacy Protection**: Non-reversible embeddings only
496: 5. **Documentation**: Complete audit trail of all operations
497: 
498: **Key Ethical Features**:
499: - ArcFace <30% threshold for "different people" prevents false positives
500: - Human review required for all moderate confidence results
501: - Complete audit trail of all comparisons
502: 
503: ---
504: 
505: ## Documentation Index
506: 
507: | File | Description |
508: |------|-------------|
509: | README.md | Main documentation, quick start, features |
510: | PROJECT_STRUCTURE.md | This file - complete structure and roadmap |
511: | DEVELOPMENT_LOG.md | Development history and session notes |
512: | ARCHITECTURE.md | Detailed system architecture |
513: | ETHICAL_COMPLIANCE.md | Ethical guidelines |
514: | CONTEXT.md | Critical rules for code edits |
515: 
516: ---
517: 
518: ## Quick Reference Commands
519: 
520: ```bash
521: # Start the system
522: cd face_recognition_npo
523: ./start.sh
524: 
525: # Run tests
526: python test_e2e_pipeline.py
527: 
528: # Verify API endpoints
529: python test_api_endpoints.py
530: 
531: # Clear Python cache (after editing source)
532: find . -type d -name "__pycache__" -exec rm -rf {} +
533: find . -name "*.pyc" -delete
534: 
535: # Run Flask directly
536: source venv/bin/activate
537: python api_server.py
538: 
539: # Run Electron
540: cd electron-ui
541: npm start
542: 
543: # Check which model is active
544: curl http://localhost:3000/api/embedding-info
545: ```
546: 
547: ---
548: 
549: ## Model Information
550: 
551: | Model | File | Dimension | Runtime | Discrimination |
552: |-------|------|-----------|---------|----------------|
553: | ArcFace | `arcface_model.onnx` | 512 | ONNX | Excellent (<30% for diff) |
554: | FaceNet | torchvision ResNet18 | 128 | PyTorch | Poor (~70% for diff) |
555: 
556: ---
557: 
558: *Project structure documentation updated: February 14, 2026*
559: *Includes Activation Similarity, Compare Visualizations, Reference Details Panel, and 14+ API endpoints*
560: 

(End of file - total 560 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/PROJECT_ARCHITECTURE.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/PROJECT_ARCHITECTURE.md</path>
<type>file</type>
<content>1: # MANTAX Face Recognition System - Complete Architecture
2: 
3: ## Project Overview
4: 
5: This document contains comprehensive Mermaid diagrams of the MANTAX NGO Facial Image Analysis System.
6: 
7: ---
8: 
9: ## 1. High-Level System Architecture
10: 
11: ```mermaid
12: %%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#0066cc', 'primaryTextColor': '#fff', 'primaryBorderColor': '#000', 'lineColor': '#666', 'secondaryColor': '#f5f5f5', 'tertiaryColor': '#fff'}}}%%
13: flowchart TB
14:     subgraph USER["USER INTERFACE LAYER"]
15:         direction TB
16:         HTML["index.html<br/>(775 lines)<br/>- Step workflow<br/>- Visualization tabs<br/>- Webcam capture<br/>- Terminal logger<br/>- Toast notifications"]
17:         JS["app.js<br/>(895 lines)<br/>- handleImageSelect()<br/>- detectFaces()<br/>- extractFeatures()<br/>- compareFaces()<br/>- showVisualization()<br/>- Webcam functions"]
18:         CSS["Styles (embedded)<br/>- CSS Variables<br/>- Terminal footer<br/>- Loading overlay<br/>- Toast system<br/>- Responsive layout"]
19:     end
20: 
21:     subgraph ELECTRON["ELECTRON DESKTOP APP"]
22:         direction TB
23:         MAIN["main.js<br/>(Electron main)<br/>- Window management<br/>- IPC handlers<br/>- App lifecycle"]
24:         PRELOAD["preload.js<br/>(Context bridge)"]
25:         PKG["package.json<br/>- electron v28<br/>- electron-builder"]
26:     end
27: 
28:     subgraph API_SERVER["FLASK API SERVER (api_server.py)"]
29:         direction TB
30:         ROUTES["Routes (14 endpoints)<br/>----------------<br/>GET  /api/health<br/>GET  /api/embedding-info<br/>POST /api/detect<br/>POST /api/extract<br/>POST /api/add-reference<br/>GET  /api/references<br/>DELETE /api/references/:id<br/>POST /api/compare<br/>GET  /api/visualizations/:type<br/>GET  /api/visualizations/:type/reference/:id<br/>GET  /api/quality<br/>GET  /api/eyewear<br/>GET  /api/visualizations/eyewear<br/>POST /api/clear<br/>GET  /api/status"]
31:         
32:         GLOBAL_STATE["Global State Variables<br/>----------------<br/>- current_image<br/>- current_original_image<br/>- current_enhanced_image<br/>- current_faces[]<br/>- current_embedding<br/>- current_face_image<br/>- current_preprocessing_info{}<br/>- current_pose{}<br/>- references[]"]
33:         
34:         HELPERS["Helper Functions<br/>----------------<br/>- load_references()<br/>- save_references()<br/>- np_to_python()<br/>- image_to_base64()<br/>- base64_to_image()<br/>- categorize_pose()<br/>- get_viz_result()<br/>- visualize_tests()"]
35:     end
36: 
37:     subgraph DETECTION["DETECTION MODULE (src/detection/)"]
38:         direction TB
39:         
40:         FACE_DETECTOR["FaceDetector Class<br/>(src/detection/__init__.py)<br/>----------------<br/>METHODS:<br/>- detect_faces(image)<br/>- detect_faces_with_confidence()<br/>- detect_eyes(face_image)<br/>- detect_eyewear(face_image, box)<br/>- compute_eyewear_metrics()<br/>- estimate_landmarks(face, box)<br/>- compute_alignment(face, landmarks)<br/>- compute_quality_metrics(face, box)<br/>- visualize_detection()<br/>- visualize_extraction()<br/>- visualize_landmarks()<br/>- visualize_3d_mesh()<br/>- visualize_alignment()<br/>- visualize_saliency()<br/>- visualize_multiscale()<br/>- visualize_quality()<br/>- visualize_biometric_capture()<br/>- visualize_eyewear()"]
41:         
42:         PREPROCESSOR["ImagePreprocessor Class<br/>(src/detection/preprocessing.py)<br/>----------------<br/>METHODS:<br/>- assess_quality(image)<br/>- needs_enhancement(quality)<br/>- apply_clahe(image)<br/>- apply_histogram_eq(image)<br/>- apply_gamma_correction(image, gamma)<br/>- enhance(image)<br/>- visualize_preprocessing(orig, enh, ...)<br/>- get_preprocessing_info(...)<br/><br/>VARIABLES:<br/>- clahe: CLAHE object<br/>- quality_thresholds{}"]
43:     end
44: 
45:     subgraph EMBEDDING["EMBEDDING MODULE (src/embedding/)"]
46:         direction TB
47:         
48:         FACENET["FaceNetEmbeddingExtractor<br/>(src/embedding/__init__.py)<br/>----------------<br/>- ImprovedEmbeddingExtractor (PyTorch)<br/>- embedding_dim: 128<br/>- backbone: resnet18<br/><br/>METHODS:<br/>- preprocess(face_image)<br/>- extract_embedding(face_image)<br/>- extract_embeddings(faces[])<br/>- get_activations(face_image)<br/>- visualize_embedding(emb)<br/>- visualize_activations(face)<br/>- visualize_feature_maps(face)<br/>- visualize_similarity_matrix()<br/>- visualize_similarity_result()<br/>- test_robustness(face)"]
49:         
50:         ARCFACE["ArcFaceEmbeddingExtractor<br/>(src/embedding/arcface_extractor.py)<br/>----------------<br/>- model_name: buffalo_l<br/>- embedding_dim: 512<br/>- Uses InsightFace FaceAnalysis<br/><br/>METHODS:<br/>- extract_embedding(face_image)<br/>- extract_embeddings(faces[])<br/>- detect_faces(image)<br/>- cosine_similarity(emb1, emb2)<br/>- euclidean_distance(emb1, emb2)<br/>- compare_embeddings(query, refs)<br/>- get_confidence_band(sim)<br/>- get_verdict(sim)<br/>- get_distance_verdict(dist)<br/>- visualize_embedding(emb)<br/>- visualize_activations(face)<br/>- visualize_feature_maps(face)<br/>- test_robustness(face)<br/>- visualize_similarity_matrix()<br/>- visualize_similarity_result()<br/>- visualize_comparison_metrics()"]
51:         
52:         COMPARATOR["SimilarityComparator<br/>(src/embedding/__init__.py)<br/>----------------<br/>VARIABLES:<br/>- threshold: 0.5<br/><br/>METHODS:<br/>- cosine_similarity(emb1, emb2)<br/>- euclidean_distance(emb1, emb2)<br/>- compare_embeddings(query, refs, ids)<br/>- get_confidence_band(sim)<br/>- get_verdict(sim)<br/>- get_distance_verdict(dist)<br/>- visualize_comparison_metrics()"]
53:     end
54: 
55:     subgraph REFERENCE["REFERENCE MODULE (src/reference/)"]
56:         direction TB
57:         
58:         REF_MANAGER["ReferenceImageManager<br/>(src/reference/__init__.py)<br/>----------------<br/>VARIABLES:<br/>- reference_dir<br/>- embeddings_file<br/>- reference_data{}<br/>- embedding_extractor<br/>- detector<br/><br/>METHODS:<br/>- _load_reference_data()<br/>- _save_reference_data()<br/>- add_reference_image(path, id, meta)<br/>- get_reference_embeddings()<br/>- get_reference_metadata(id)<br/>- list_references()<br/>- remove_reference(id)"]
59:         
60:         HUMAN_REVIEW["HumanReviewInterface<br/>(src/reference/__init__.py)<br/>----------------<br/>VARIABLES:<br/>- review_history[]<br/><br/>METHODS:<br/>- display_comparison(query, ref, sim, id)<br/>- _get_confidence_text(sim)<br/>- _get_key_decision(key)"]
61:     end
62: 
63:     subgraph PERSISTENCE["PERSISTENCE LAYER"]
64:         direction TB
65:         JSON["reference_images/<br/>embeddings.json<br/>----------------<br/>- metadata[]<br/>  - id, name, path<br/>  - metadata, added_at<br/>- embeddings[]<br/>  - id, embedding[]"]
66:         MODEL["Deep Learning Models<br/>----------------<br/>- res10_300x300_ssd_iter_140000.caffemodel<br/>  (Face detection - OpenCV DNN)<br/>- deploy.prototxt.txt<br/>  (Caffe prototxt)<br/>- ArcFace ONNX model<br/>  (via insightface)"]
67:     end
68: 
69:     subgraph TESTS["TEST SUITE"]
70:         direction TB
71:         E2E["test_e2e_pipeline.py<br/>(End-to-End tests)<br/>----------------<br/>- Detection test<br/>- Embedding test<br/>- Reference manager test<br/>- Same image similarity<br/>- Different images similarity<br/>- Reference comparison"]
72:         EDGE["test_edge_cases.py<br/>(Edge case tests)<br/>----------------<br/>- Empty/black images<br/>- Very small images<br/>- None/invalid inputs<br/>- Boundary similarity<br/>- Empty reference list<br/>- Many references"]
73:         FRONTEND["test_frontend_integration.py<br/>(Frontend integration)"]
74:         EYEWEAR["test_eyewear.py<br/>(Eyewear detection)"]
75:         UNIT["tests/ (Unit tests)<br/>----------------<br/>- test_detection.py<br/>- test_embedding.py<br/>- test_comparison.py<br/>- test_reference.py<br/>- test_review.py"]
76:     end
77: 
78:     subgraph CONFIG["CONFIGURATION & UTILS"]
79:         direction TB
80:         START["start.sh<br/>(218 lines)<br/>----------------<br/>- Kill existing servers<br/>- Clear Python cache<br/>- Start Flask API<br/>- Wait for health check<br/>- Launch Electron/Browser"]
81:         CONFIG_TMPL["config_template.py<br/>(Configuration template)"]
82:         VENV["Virtual Environment<br/>----------------<br/>- venv/bin/activate<br/>- venv313 (Python 3.13)"]
83:     end
84: 
85:     subgraph DOCS["DOCUMENTATION"]
86:         direction TB
87:         README["README.md<br/>(403 lines)<br/>- Quick start<br/>- Architecture<br/>- Features<br/>- API endpoints<br/>- 14 visualization types<br/>- Testing"]
88:         ARCH["ARCHITECTURE.md<br/>(System design)"]
89:         CONTEXT["CONTEXT.md<br/>(Code review context)<br/>- Critical edit rules<br/>- Common mistakes<br/>- Lessons learned"]
90:         ETHICAL["ETHICAL_COMPLIANCE.md<br/>(Ethical guidelines)"]
91:         DEV_LOG["DEVELOPMENT_LOG.md<br/>(Development history)"]
92:     end
93: 
94:     %% Relationships
95:     USER -->|"HTTP requests"| API_SERVER
96:     JS -->|"fetch() calls"| API_SERVER
97:     
98:     ELECTRON -->|"serves"| USER
99:     ELECTRON -->|"connects to"| API_SERVER
100:     
101:     API_SERVER -->|"instantiates"| DETECTION
102:     API_SERVER -->|"instantiates"| EMBEDDING
103:     API_SERVER -->|"uses"| REFERENCE
104:     
105:     API_SERVER -->|"POST /api/detect"| DETECTION
106:     API_SERVER -->|"POST /api/extract"| EMBEDDING
107:     API_SERVER -->|"POST /api/add-reference"| REFERENCE
108:     API_SERVER -->|"POST /api/compare"| COMPARATOR
109:     
110:     API_SERVER -->|"reads/writes"| JSON
111:     API_SERVER -->|"loads"| MODEL
112:     
113:     DETECTION -->|"face boxes"| EMBEDDING
114:     EMBEDDING -->|"embeddings"| COMPARATOR
115:     
116:     PREPROCESSOR -->|"enhanced image"| DETECTION
117: ```
118: 
119: ---
120: 
121: ## 2. Data Flow Workflow
122: 
123: ```mermaid
124: %%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#0066cc', 'primaryTextColor': '#fff', 'lineColor': '#666'}}}%%
125: flowchart LR
126:     subgraph INPUT["INPUT SOURCES"]
127:         direction TB
128:         UPLOAD["File Upload<br/>(imageInput)"]
129:         WEBCAM["Webcam Capture<br/>(MediaDevices API)"]
130:         API["External API"]
131:     end
132: 
133:     subgraph WORKFLOW["MAIN WORKFLOW"]
134:         direction TB
135:         
136:         subgraph STEP1["Step 1: Image Input"]
137:             S1A["Read file as DataURL"]
138:             S1B["Fire-and-forget: POST /api/clear"]
139:             S1C["Display preview image"]
140:         end
141:         
142:         subgraph STEP2["Step 2: Detection"]
143:             S2A["POST /api/detect<br/>{image: base64}"]
144:             S2B["ImagePreprocessor.enhance()<br/>- assess_quality()<br/>- apply CLAHE/gamma/sharpen"]
145:             S2C["FaceDetector.detect_faces()<br/>- OpenCV DNN (Caffe)<br/>- Returns bbox[]"]
146:             S2D["Return: faces[], visualizations{}"]
147:             S2E["Display face thumbnails"]
148:             S2F["Detect eyewear"]
149:         end
150:         
151:         subgraph STEP3["Step 3: Feature Extraction"]
152:             S3A["POST /api/extract<br/>{face_id: 0}"]
153:             S3B["Extract face ROI from bbox"]
154:             S3C["FaceNet/ArcFace extract_embedding()<br/>- 128-dim or 512-dim vector"]
155:             S3D["Estimate landmarks & pose"]
156:             S3E["Generate all visualizations:<br/>- embedding (bar chart)<br/>- activations<br/>- features<br/>- landmarks<br/>- mesh3d<br/>- alignment<br/>- saliency<br/>- multiscale<br/>- confidence<br/>- robustness<br/>- biometric"]
157:             S3F["Return: embedding, pose, visualizations{}"]
158:         end
159:         
160:         subgraph STEP4["Step 4: Comparison"]
161:             S4A["Add reference image"]
162:             S4B["POST /api/add-reference<br/>{image, name}"]
163:             S4C["Detect faces & extract embedding"]
164:             S4D["Save to references[] + JSON"]
165:             S4E["POST /api/compare"]
166:             S4F["SimilarityComparator.cosine_similarity()"]
167:             S4G["Pose-aware matching adjustment"]
168:             S4H["Return: results[], best_match, similarity_viz"]
169:             S4I["Display comparison UI"]
170:         end
171:     end
172: 
173:     subgraph VISUALIZATIONS["16 VISUALIZATION TYPES"]
174:         direction TB
175:         V_DETECTION["detection<br/>(bounding boxes)"]
176:         V_EXTRACTION["extraction<br/>(face ROI)"]
177:         V_PREPROCESSING["preprocessing<br/>(before/after)"]
178:         V_LANDMARKS["landmarks<br/>(15 keypoints)"]
179:         V_MESH3D["mesh3d<br/>(478-point mesh)"]
180:         V_ALIGNMENT["alignment<br/>(pitch/yaw/roll)"]
181:         V_SALIENCY["saliency<br/>(attention heatmap)"]
182:         V_ACTIVATIONS["activations<br/>(CNN layers)"]
183:         V_FEATURES["features<br/>(feature maps)"]
184:         V_MULTISCALE["multiscale<br/>(multi-scale detection)"]
185:         V_CONFIDENCE["confidence<br/>(quality metrics)"]
186:         V_EYEWEAR["eyewear<br/>(sunglasses/glasses)"]
187:         V_EMBEDDING["embedding<br/>(dim bar chart)"]
188:         V_SIMILARITY["similarity<br/>(comparison bars)"]
189:         V_ROBUSTNESS["robustness<br/>(noise test)"]
190:         V_BIOMETRIC["biometric<br/>(biometric overview)"]
191:     end
192: 
193:     subgraph STORAGE["STORAGE & STATE"]
194:         direction TB
195:         MEMORY["In-Memory State<br/>----------------<br/>- current_image<br/>- current_faces[]<br/>- current_embedding<br/>- current_pose{}<br/>- references[]"]
196:         DISK["Disk Storage<br/>----------------<br/>- embeddings.json<br/>  (metadata + embeddings)"]
197:     end
198: 
199:     %% Data flows
200:     INPUT --> STEP1
201:     STEP1 --> STEP2
202:     STEP2 --> STEP3
203:     STEP3 --> STEP4
204:     STEP4 -->|"updates"| MEMORY
205:     STEP4 -->|"saves to"| DISK
206:     
207:     STEP3 -->|"generates"| VISUALIZATIONS
208:     STEP4 -->|"generates"| VISUALIZATIONS
209:     
210:     S2C -->|"uses"| V_DETECTION
211:     S2C -->|"uses"| V_EXTRACTION
212:     S3E -->|"generates"| V_LANDMARKS
213:     S3E -->|"generates"| V_MESH3D
214:     S3E -->|"generates"| V_ALIGNMENT
215:     S3E -->|"generates"| V_SALIENCY
216:     S3E -->|"generates"| V_ACTIVATIONS
217:     S3E -->|"generates"| V_FEATURES
218:     S3E -->|"generates"| V_MULTISCALE
219:     S3E -->|"generates"| V_CONFIDENCE
220:     S3E -->|"generates"| V_EYEWEAR
221:     S3E -->|"generates"| V_EMBEDDING
222:     S3E -->|"generates"| V_ROBUSTNESS
223:     S3E -->|"generates"| V_BIOMETRIC
224:     S4H -->|"shows"| V_SIMILARITY
225: ```
226: 
227: ---
228: 
229: ## 3. Class Diagram
230: 
231: ```mermaid
232: %%{init: {'theme': 'base'}}%%
233: classDiagram
234:     %% FaceDetector Class
235:     class FaceDetector {
236:         +use_dnn: bool
237:         +face_cascade: CascadeClassifier
238:         +net: cv2.dnn_Net
239:         +mp_face_mesh: Any
240:         +face_mesh: Any
241:         +eye_cascade: CascadeClassifier
242:         +confidence_threshold: float
243:         +detect_faces(image: np.ndarray) List[Tuple[int,int,int,int]]
244:         +detect_faces_with_confidence(image) List[Tuple[Tuple,float]]
245:         +detect_eyes(face_image) List[Tuple]
246:         +detect_eyewear(face_image, face_box) Dict
247:         +compute_eyewear_metrics(face_image, box) Dict
248:         +visualize_eyewear(face_image, box) np.ndarray
249:         +estimate_landmarks(face_image, box) Dict
250:         +compute_alignment(face_image, landmarks) Dict
251:         +compute_quality_metrics(face_image, box) Dict
252:         +visualize_detection(image, faces) np.ndarray
253:         +visualize_extraction(image, faces) np.ndarray
254:         +visualize_landmarks(face, landmarks) np.ndarray
255:         +visualize_3d_mesh(face_image) np.ndarray
256:         +visualize_alignment(face, landmarks, align) np.ndarray
257:         +visualize_saliency(face_image) np.ndarray
258:         +visualize_multiscale(face_image) np.ndarray
259:         +visualize_quality(face, box) Tuple[np.ndarray, Dict]
260:         +visualize_biometric_capture(image, faces) np.ndarray
261:     }
262: 
263:     %% ImagePreprocessor Class
264:     class ImagePreprocessor {
265:         +clahe: CLAHE
266:         +quality_thresholds: Dict
267:         +assess_quality(image: np.ndarray) Dict
268:         +needs_enhancement(quality: Dict) Tuple[bool, str]
269:         +apply_clahe(image: np.ndarray) np.ndarray
270:         +apply_histogram_eq(image) np.ndarray
271:         +apply_gamma_correction(image, gamma) np.ndarray
272:         +enhance(image: np.ndarray) Tuple[np.ndarray, str]
273:         +visualize_preprocessing(orig, enh, orig_q, enh_q, method) np.ndarray
274:         +get_preprocessing_info(orig, enh, method) Dict
275:     }
276: 
277:     %% FaceNetEmbeddingExtractor Class
278:     class FaceNetEmbeddingExtractor {
279:         +embedding_size: int = 128
280:         +device: torch.device
281:         +model: ImprovedEmbeddingExtractor
282:         +mean: np.ndarray
283:         +std: np.ndarray
284:         +preprocess(face_image: np.ndarray) torch.Tensor
285:         +extract_embedding(face_image) np.ndarray
286:         +extract_embeddings(face_images: List) List
287:         +get_activations(face_image) Dict
288:         +visualize_embedding(embedding) Tuple[np.ndarray, Dict]
289:         +visualize_activations(face_image) np.ndarray
290:         +visualize_feature_maps(face_image) np.ndarray
291:         +visualize_similarity_matrix(query, refs, ids) Tuple
292:         +visualize_similarity_result(query, ref, sim) np.ndarray
293:         +test_robustness(face_image) Tuple[np.ndarray, Dict]
294:     }
295: 
296:     %% ArcFaceEmbeddingExtractor Class
297:     class ArcFaceEmbeddingExtractor {
298:         +model_name: str = 'buffalo_l'
299:         +app: FaceAnalysis
300:         +embedding_dim: int = 512
301:         +detector: None
302:         +extract_embedding(face_image) np.ndarray
303:         +extract_embeddings(face_images) List
304:         +detect_faces(image) List[Tuple]
305:         +cosine_similarity(emb1, emb2) float
306:         +euclidean_distance(emb1, emb2) float
307:         +compare_embeddings(query, refs, ids) List[Dict]
308:         +get_confidence_band(similarity) str
309:         +get_verdict(similarity) str
310:         +get_distance_verdict(distance) str
311:         +visualize_embedding(embedding) Tuple[np.ndarray, Dict]
312:         +visualize_activations(face_image) np.ndarray
313:         +visualize_feature_maps(face_image) np.ndarray
314:         +test_robustness(face_image) Tuple[np.ndarray, Dict]
315:         +visualize_similarity_matrix(query, refs, ids) Tuple
316:         +visualize_similarity_result(query, ref, sim) np.ndarray
317:         +visualize_comparison_metrics(...) Tuple[np.ndarray, Dict]
318:     }
319: 
320:     %% SimilarityComparator Class
321:     class SimilarityComparator {
322:         +threshold: float = 0.5
323:         +cosine_similarity(emb1, emb2) float
324:         +euclidean_distance(emb1, emb2) float
325:         +compare_embeddings(query, refs, ids) List[Dict]
326:         +get_confidence_band(similarity) str
327:         +get_verdict(similarity) str
328:         +get_distance_verdict(distance) str
329:         +visualize_comparison_metrics(...) Tuple[np.ndarray, Dict]
330:     }
331: 
332:     %% ReferenceImageManager Class
333:     class ReferenceImageManager {
334:         +reference_dir: str
335:         +embeddings_file: str
336:         +reference_data: Dict
337:         +embedding_extractor: Any
338:         +detector: Any
339:         +_load_reference_data()
340:         +_save_reference_data()
341:         +add_reference_image(path, id, meta) Tuple[bool, np.ndarray]
342:         +get_reference_embeddings() Tuple[List, List]
343:         +get_reference_metadata(id) Optional[Dict]
344:         +list_references() List[Dict]
345:         +remove_reference(id) bool
346:     }
347: 
348:     %% HumanReviewInterface Class
349:     class HumanReviewInterface {
350:         +review_history: List
351:         +display_comparison(query_img, ref_img, sim, ref_id, meta) int
352:         +_get_confidence_text(similarity) str
353:         +_get_key_decision(key) str
354:     }
355: 
356:     %% Flask API Server (module as class representation)
357:     class api_server {
358:         +detector: FaceDetector
359:         +preprocessor: ImagePreprocessor
360:         +extractor: FaceNet/ArcFace
361:         +comparator: SimilarityComparator
362:         +current_image: np.ndarray
363:         +current_faces: List
364:         +current_embedding: np.ndarray
365:         +references: List
366:         +health_check() json
367:         +detect_faces() json
368:         +extract_embedding() json
369:         +add_reference() json
370:         +get_references() json
371:         +remove_reference(id) json
372:         +compare_faces() json
373:         +get_visualization(type) json
374:         +clear_all() json
375:         +get_status() json
376:     }
377: 
378:     %% Relationships
379:     FaceDetector --> ImagePreprocessor
380:     api_server --> FaceDetector
381:     api_server --> ImagePreprocessor
382:     api_server --> FaceNetEmbeddingExtractor
383:     api_server --> ArcFaceEmbeddingExtractor
384:     api_server --> SimilarityComparator
385:     api_server --> ReferenceImageManager
386:     FaceNetEmbeddingExtractor --> SimilarityComparator
387:     ArcFaceEmbeddingExtractor --> SimilarityComparator
388:     ReferenceImageManager --> FaceNetEmbeddingExtractor
389:     ReferenceImageManager --> ArcFaceEmbeddingExtractor
390:     ReferenceImageManager --> FaceDetector
391: ```
392: 
393: ---
394: 
395: ## 4. API Sequence Diagram
396: 
397: ```mermaid
398: %%{init: {'theme': 'base'}}%%
399: sequenceDiagram
400:     participant U as User (Browser/Electron)
401:     participant API as Flask API Server
402:     participant D as FaceDetector
403:     participant P as ImagePreprocessor
404:     participant E as EmbeddingExtractor<br/>(FaceNet/ArcFace)
405:     participant C as SimilarityComparator
406:     participant R as ReferenceManager
407:     participant F as File System
408: 
409:     Note over U,API: STEP 1: Upload Image
410:     U->>API: POST /api/detect {image: base64}
411:     API->>P: enhance(image)
412:     P-->>API: enhanced_image, method
413:     API->>D: detect_faces(enhanced_image)
414:     D-->>API: faces[] (bbox)
415:     API->>D: detect_eyewear(image, faces[0])
416:     D-->>API: eyewear{}
417:     API->>D: visualize_detection(image, faces)
418:     API->>D: visualize_extraction(image, faces)
419:     API->>D: visualize_biometric_capture(image, faces)
420:     API-->>U: {success, count, faces[], visualizations{}}
421: 
422:     Note over U,API: STEP 2: Extract Features
423:     U->>API: POST /api/extract {face_id: 0}
424:     API->>E: extract_embedding(face_roi)
425:     E-->>API: embedding[] (128-dim or 512-dim)
426:     API->>D: estimate_landmarks(face, bbox)
427:     API->>D: compute_alignment(face, landmarks)
428:     D-->>API: pose{yaw, pitch, roll, category}
429:     API->>E: visualize_embedding(embedding)
430:     API->>E: visualize_activations(face)
431:     API->>E: visualize_feature_maps(face)
432:     API->>E: test_robustness(face)
433:     API->>D: visualize_landmarks(face, landmarks)
434:     API->>D: visualize_3d_mesh(face)
435:     API->>D: visualize_alignment(face, landmarks, align)
436:     API->>D: visualize_saliency(face)
437:     API->>D: visualize_multiscale(face)
438:     API->>D: visualize_quality(face, bbox)
439:     API-->>U: {success, embedding_size, pose, visualizations{}}
440: 
441:     Note over U,API: STEP 3: Add Reference
442:     U->>API: POST /api/add-reference {image, name}
443:     API->>D: detect_faces(image)
444:     D-->>API: ref_faces[]
445:     API->>E: extract_embedding(ref_face)
446:     E-->>API: ref_embedding[]
447:     API->>D: estimate_landmarks(ref_face, bbox)
448:     API->>D: compute_alignment(ref_face, landmarks)
449:     R->>F: save to embeddings.json
450:     API-->>U: {success, reference, count}
451: 
452:     Note over U,API: STEP 4: Compare
453:     U->>API: POST /api/compare {}
454:     loop For each reference
455:         API->>C: cosine_similarity(query_emb, ref_emb)
456:         C-->>API: similarity
457:         API->>C: euclidean_distance(query_emb, ref_emb)
458:         C-->>API: distance
459:         API->>C: Adjust for pose similarity
460:     end
461:     API->>C: visualize_comparison_metrics(...)
462:     C-->>API: similarity_viz
463:     API-->>U: {success, results[], best_match, similarity_viz}
464: 
465:     Note over U,API: Visualization Request
466:     U->>API: GET /api/visualizations/{type}
467:     alt type == detection
468:         API->>D: visualize_detection(current_image, faces)
469:     alt type == embedding
470:         API->>E: visualize_embedding(current_embedding)
471:     alt type == landmarks
472:         API->>D: visualize_landmarks(face, landmarks)
473:     alt type == activations
474:         API->>E: visualize_activations(face)
475:     end
476:     API-->>U: {success, visualization: base64}
477: 
478:     Note over U,API: Clear Session
479:     U->>API: POST /api/clear {}
480:     API->>API: Reset all global state variables
481:     API-->>U: {success, message}
482: ```
483: 
484: ---
485: 
486: ## 5. Project File Structure
487: 
488: ```mermaid
489: %%{init: {'theme': 'base'}}%%
490: graph TB
491:     subgraph ROOT["MANTAX Project Root"]
492:         direction TB
493:         ARCH["ARCHITECTURE_REVIEW.md"]
494:         RULE["rule_set.md"]
495:         VENV["venv/"]
496:         MODEL["res10_300x300_ssd_iter_140000.caffemodel"]
497:     end
498:     
499:     subgraph FACE_NPO["face_recognition_npo/"]
500:         direction TB
501:         
502:         subgraph CORE["Core Files"]
503:             API["api_server.py<br/>(900+ lines)<br/>Flask API"]
504:             START["start.sh<br/>(218 lines)<br/>Startup script"]
505:             CONFIG["config_template.py"]
506:             SETUP["setup.py"]
507:         end
508:         
509:         subgraph SRC["src/"]
510:             DET["detection/<br/>__init__.py<br/>preprocessing.py"]
511:             EMB["embedding/<br/>__init__.py<br/>arcface_extractor.py"]
512:             REF["reference/<br/>__init__.py"]
513:         end
514:         
515:         subgraph UI["electron-ui/"]
516:             HTML["index.html<br/>(775 lines)"]
517:             JS["renderer/app.js<br/>(895 lines)"]
518:             MAIN["main.js"]
519:             PKG["package.json"]
520:         end
521:         
522:         subgraph TESTS["Test Files"]
523:             E2E["test_e2e_pipeline.py"]
524:             EDGE["test_edge_cases.py"]
525:             FRONT["test_frontend_integration.py"]
526:             EYE["test_eyewear.py"]
527:             UNIT["tests/"]
528:         end
529:         
530:         subgraph DOCS["Documentation"]
531:             README["README.md"]
532:             ARCH_MD["ARCHITECTURE.md"]
533:             CONTEXT["CONTEXT.md"]
534:             ETHICAL["ETHICAL_COMPLIANCE.md"]
535:             DEV["DEVELOPMENT_LOG.md"]
536:         end
537:         
538:         subgraph DATA["Data Directories"]
539:             REF_IMG["reference_images/"]
540:             TEST["test_images/"]
541:             CAPTURED["captured_faces/"]
542:         end
543:     end
544:     
545:     ROOT --> FACE_NPO
546: ```
547: 
548: ---
549: 
550: ## Summary
551: 
552: This document contains **5 main diagrams**:
553: 
554: 1. **High-Level System Architecture** - Shows all layers from UI to persistence
555: 2. **Data Flow Workflow** - Shows the 4-step user workflow with all 16 visualization types
556: 3. **Class Diagram** - Shows all Python classes and their relationships
557: 4. **API Sequence Diagram** - Shows the request/response flow for each endpoint
558: 5. **Project File Structure** - Shows the complete directory layout
559: 
560: All components work together as follows:
561: - User interacts through **Electron/Browser** UI
562: - JavaScript sends **HTTP requests** to Flask API
563: - Flask API orchestrates **detection → extraction → comparison**
564: - Results are **visualized** and optionally **persisted** to JSON
565: 
566: ---
567: 
568: ## Test Visualization Data Flow (Feb 13, 2026)
569: 
570: The 9 test tabs display both an image and structured data. Here's how it works:
571: 
572: ### API Handler Pattern
573: ```python
574: # Each test viz handler returns: (image, data_dict)
575: elif viz_type == 'test-health':
576:     data = {"status": "OK", "api": "running", "port": 3000}
577:     return visualize_test_detail("Health Check", data), data
578: 
579: elif viz_type == 'test-detection':
580:     data = {
581:         "faces_detected": len(current_faces) if current_faces else 0,
582:         "preprocessing": current_preprocessing_info.get('method', 'none'),
583:         "enhanced": current_preprocessing_info.get('was_enhanced', False)
584:     }
585:     return visualize_test_detail("Detection + Preprocessing", data), data
586: ```
587: 
588: ### Frontend Display
589: - Image rendered from base64
590: - Data displayed as HTML table via `formatDataAsTable()`
591: - CSS styling from `.viz-data-table` class
592: 
593: ### Test Tabs
594: | Tab | Data Fields |
595: |-----|-------------|
596: | test-health | status, api, port |
597: | test-detection | faces_detected, preprocessing, enhanced |
598: | test-extraction | embedding_size, pose |
599: | test-reference | references, latest_pose |
600: | test-multi | total_references, can_match |
601: | test-pose | query_pose, matching_enabled, adjusts_similarity |
602: | test-eyewear | eyewear detection results |
603: | test-viz | total_types, detection, preprocessing, pose, tests |
604: | test-clear | session_management, can_clear |
605: 
606: ---
607: 
608: ## Eyewear Detection (Feb 13, 2026)
609: 
610: The eyewear detection system identifies sunglasses/glasses that may interfere with face recognition.
611: 
612: ### Detection Algorithm
613: 
614: 1. **Primary Method**: Brightness/Edge Analysis
615:    - Extract eye regions (30% and 70% of face width)
616:    - Calculate brightness ratio: `eye_brightness / face_brightness`
617:    - Calculate edge density using Canny edge detection
618:    - Stricter thresholds: brightness < 0.2 = sunglasses, < 0.35 = possible
619: 
620: 2. **Secondary Method**: Eye Cascade (Confirmation Only)
621:    - Use OpenCV eye cascade as confirmation, NOT primary detection
622:    - Only flag if BOTH brightness AND eye count agree
623:    - Avoid false positives from unreliable eye cascade
624: 
625: ### API Endpoints
626: 
627: | Endpoint | Method | Description |
628: |----------|--------|-------------|
629: | `/api/eyewear` | GET | Get eyewear detection results |
630: | `/api/visualizations/eyewear` | GET | Get eyewear visualization |
631: 
632: ### Response Format
633: ```json
634: {
635:   "success": true,
636:   "eyewear": {
637:     "has_eyewear": false,
638:     "type": "none",
639:     "confidence": 0.1,
640:     "occlusion_level": 0.0,
641:     "warnings": [],
642:     "eye_count": 2
643:   }
644: }
645: ```
646: 
647: ### Testing
648: 
649: **Frontend Test**: `test_eyewear_frontend.js`
650: ```bash
651: node test_eyewear_frontend.js
652: ```
653: 
654: **Backend Test**: Uses synthetic images with darkened eye regions
655: 
656: 

(End of file - total 656 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/FRONTEND_SUMMARY.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/FRONTEND_SUMMARY.md</path>
<type>file</type>
<content>1: # Frontend Development & Testing Summary
2: **Date**: February 12, 2026  
3: **Project**: NGO Facial Recognition System v0.3.0
4: 
5: ---
6: 
7: ## 🎯 Overview
8: 
9: All preloader-related code has been reverted to create a clean, professional face recognition application without unnecessary loading delays or animations.
10: 
11: ---
12: 
13: ## 📋 System Status Verification
14: 
15: ### ✅ **Backend Tests - ALL PASSING**
16: ```
17: E2E Pipeline Tests: 6/6 PASS
18: Edge Case Tests: 11/11 PASS  
19: API Endpoints: 11/11 working
20: Flask Server: Running on localhost:3000
21: ArcFace Mode: ENABLED (512-dim)
22: ```
23: 
24: ### ✅ **Frontend Status**
25: ```
26: ✅ All 11 API endpoints accessible
27: ✅ Response time: <500ms (excellent)
28: ✅ Memory usage: Normal
29: ✅ All essential JavaScript functions available
30: ✅ CSS styling: Modern system fonts working
31: ✅ No preloader delays - app loads immediately
32: ```
33: 
34: ---
35: 
36: ## 🔧 Key Changes Made
37: 
38: ### 1. **Removed All Preloader Code**
39: - ❌ Deleted loading overlay HTML
40: - ❌ Removed loading CSS animations  
41: - ❌ Removed loading JavaScript functions
42: - ❌ Removed delayed app reveal logic
43: 
44: ### 2. **Cleaned HTML Structure**
45: ```html
46: <!-- BEFORE -->
47: <div class="loading-overlay">
48:     <!-- Complex loading animation -->
49: </div>
50: <div class="main-app" style="display: none;">
51:     <!-- Main content -->
52: </div>
53: 
54: <!-- AFTER -->
55: <div class="container">
56:     <!-- Direct main content -->
57: </div>
58: ```
59: 
60: ### 3. **Streamlined JavaScript**
61: ```javascript
62: // BEFORE
63: document.addEventListener('DOMContentLoaded', () => {
64:     showLoadingScreen();
65:     setTimeout(() => hideLoadingScreen(), 2000);
66:     checkAPI();
67: });
68: 
69: // AFTER
70: document.addEventListener('DOMContentLoaded', () => {
71:     checkAPI();
72: });
73: ```
74: 
75: ---
76: 
77: ## 📊 Performance Metrics
78: 
79: ### **API Performance**
80: - **Health Check**: <100ms response time
81: - **Endpoint Discovery**: All 11 endpoints responding
82: - **Memory Usage**: Normal (no memory leaks)
83: - **Concurrent Requests**: Handling properly
84: 
85: ### **Frontend Performance**
86: - **App Load Time**: Immediate (no preloader delay)
87: - **Button Responsiveness**: All click handlers working
88: - **Visualizations**: 14 types displaying correctly
89: - **Image Processing**: Upload and analysis functional
90: 
91: ---
92: 
93: ## 🎨 Frontend Test Suite Created
94: 
95: **Location**: `/frontend-test.html`
96: 
97: ### Features Tested:
98: 1. **API Connection**
99:    - Health endpoint connectivity
100:    - Endpoint discovery (all 11 APIs)
101:    - Error handling and timeouts
102: 
103: 2. **Performance Testing**
104:    - Response time measurement
105:    - Memory usage monitoring
106:    - Concurrent request handling
107: 
108: 3. **UI Component Testing**
109:    - Essential JavaScript functions (25+ functions)
110:    - CSS styling and modern fonts
111:    - Button functionality
112:    - Image upload processing
113:    - Visualization display
114:    - Terminal logging
115: 
116: 4. **Integration Testing**
117:    - End-to-end workflow testing
118:    - Error boundary handling
119:    - Cross-browser compatibility
120: 
121: ---
122: 
123: ## 🚀 Application Status
124: 
125: ### **Production Ready**: ✅ YES
126: - **All Tests**: 17/17 passing
127: - **API Server**: Fully functional
128: - **Frontend**: Clean, responsive, no delays
129: - **ArcFace**: 512-dim embeddings working
130: - **Visualizations**: All 14 types available
131: - **Glasses Detection**: Integrated into quality metrics
132: 
133: ### **Performance**: ✅ EXCELLENT
134: - **API Response**: <500ms average
135: - **Memory**: Stable, no leaks
136: - **UI**: Instant loading, smooth interactions
137: 
138: ---
139: 
140: ## 📁 Files Modified
141: 
142: ### Removed/Cleaned:
143: 1. **Loading Overlay**: Complete removal from HTML/CSS/JS
144: 2. **Delay Logic**: Removed setTimeout-based app reveal
145: 3. **Complex Animations**: Removed spinning circles and progress bars
146: 
147: ### Created:
148: 1. **Frontend Test Suite**: Comprehensive testing interface
149: 2. **Performance Monitoring**: Real-time metrics
150: 3. **Integration Tests**: End-to-end validation
151: 
152: ---
153: 
154: ## 🎯 User Experience
155: 
156: ### **Before (Issues)**:
157: - ❌ 2-second loading delay
158: - ❌ Complex animations causing distraction  
159: - ❌ White screen problems
160: - ❌ App not visible after loading
161: - ❌ Unnecessary complexity
162: 
163: ### **After (Fixed)**:
164: - ✅ Instant app loading
165: - ✅ Clean, minimal interface
166: - ✅ All functionality immediately available
167: - ✅ Professional appearance
168: - ✅ Fast user interactions
169: 
170: ---
171: 
172: ## 🔗 Testing Access
173: 
174: **Main Application**: `http://localhost:3000`  
175: **Frontend Test Suite**: `file:///Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/frontend-test.html`
176: 
177: ---
178: 
179: ## 📈 System Recommendations
180: 
181: ### **Immediate Actions**:
182: 1. ✅ **Deploy to Production**: System is fully tested and ready
183: 2. ✅ **Monitor Performance**: Use built-in test suite for ongoing validation
184: 3. ✅ **User Training**: Document immediate app availability (no loading delays)
185: 
186: ### **Future Enhancements**:
187: 1. **Progressive Enhancement**: Consider adding subtle loading only for slow operations
188: 2. **Mobile Optimization**: Test on mobile devices for responsive behavior
189: 3. **Accessibility**: Add ARIA labels and keyboard navigation
190: 
191: ---
192: 
193: **Result**: Clean, professional, and fully functional face recognition system with excellent performance and user experience.
194: 
195: ---
196: 
197: *Summary completed: February 12, 2026*  
198: *System status: PRODUCTION READY*

(End of file - total 198 lines)
</content>
```

---

## Assistant (Build · minimax-m2.5-free · 11.3s)

_Thinking:_

Good progress. Let me continue reading the remaining .md files.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/PHASE1_COMPLETE.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/PHASE1_COMPLETE.md</path>
<type>file</type>
<content>1: # Phase 1 Complete: InsightFace Integration for Better Matching
2: 
3: ## Summary of Changes Made
4: 
5: ### 🎯 Goal Achieved
6: **Problem**: Same person from different angles was matching poorly (~30-50%) compared to random people.
7: **Root Cause**: Using OpenCV DNN detection + manual cropping instead of InsightFace's built-in alignment.
8: **Solution**: Use InsightFace's `app.get()` which provides pre-aligned embeddings.
9: 
10: ### ✅ Changes Implemented
11: 
12: #### 1. Fixed ArcFace Extractor (`src/embedding/arcface_extractor.py`)
13: - **Line 78**: Changed `face['bbox']` → `face.bbox` (property access, not dictionary)
14: - **Result**: InsightFace detection now works correctly
15: 
16: #### 2. Updated Detection API (`api_server.py` - `/api/detect`)
17: - **InsightFace Path**: Uses `extractor.app.get(image)` for detection
18: - **Pre-Aligned Embeddings**: Returns `face.normed_embedding` directly (already aligned!)
19: - **Fallback**: Still uses OpenCV DNN if InsightFace unavailable
20: - **Metadata**: Includes detection confidence (`det_score`)
21: 
22: #### 3. Updated Extraction API (`api_server.py` - `/api/extract`)
23: - **Smart Detection**: Uses pre-computed embeddings from InsightFace when available
24: - **Embedding Source**: Tracks if embedding came from 'insightface_aligned' vs 'manual_extraction'
25: - **No Breaking**: Works with both InsightFace and FaceNet
26: 
27: #### 4. Updated Reference API (`api_server.py` - `/api/add-reference`)
28: - **Better Alignment**: Reference images now use InsightFace's pre-aligned embeddings
29: - **Detection Quality**: Stores `det_score` for filtering low-quality detections
30: - **Embedding Source**: Tracks alignment method used
31: 
32: ### 🔍 Key Technical Improvements
33: 
34: #### Before (Broken):
35: ```
36: User Upload → OpenCV DNN (basic detection) → Manual crop → ArcFace extraction
37:               ↓                                                    ↓
38:          ❌ No pose handling                              ❌ No alignment
39: ```
40: 
41: #### After (Fixed):
42: ```
43: User Upload → InsightFace app.get() (detection + alignment) → Pre-aligned embedding
44:               ↓                                                    ↓
45:          ✅ 5-point landmarks                            ✅ Pose-invariant embedding
46: ```
47: 
48: ### 📈 Expected Performance Improvements
49: 
50: | Scenario | Before | After |
51: |----------|--------|-------|
52: | Same person, frontal | ~70-85% | ~85-95% |
53: | Same person, **different angle** | ~30-50% ❌ | ~60-75% ✅ |
54: | Different people | <30% ✅ | <30% ✅ |
55: | Detection quality | Basic | **Confidence scores** |
56: 
57: ### 🧪 What InsightFace Now Provides
58: 
59: #### Face Object Attributes:
60: ```python
61: face = extractor.app.get(image)[0]
62: face.bbox              # Bounding box [x1, y1, x2, y2]
63: face.kps               # 5 keypoints (eyes, nose, mouth)
64: face.normed_embedding  # 512-dim L2-normalized (USE THIS!)
65: face.det_score         # Detection confidence (0-1)
66: face.pose              # [pitch, yaw, roll] orientation
67: face.gender           # 0=female, 1=male
68: face.age              # Estimated age
69: ```
70: 
71: #### Alignment Benefits:
72: - **5-point landmarks** for pose normalization
73: - **Pre-aligned embeddings** (no manual cropping errors)
74: - **Pose-invariant** (handles different angles)
75: - **Consistent preprocessing** (same as training)
76: 
77: ### 🎛️ No Breaking Changes
78: 
79: ✅ **Backward Compatibility**: 
80: - If InsightFace unavailable → Falls back to OpenCV DNN
81: - FaceNet still works unchanged
82: - Existing API responses preserved
83: 
84: ✅ **Frontend Compatibility**:
85: - No changes needed to Electron app
86: - Same response format
87: - New optional fields (`det_score`, `embedding_source`)
88: 
89: ### 🧪 Testing Results Expected
90: 
91: With these changes, you should see:
92: 
93: 1. **Much higher similarity scores for same person at different angles**
94:    - Before: ~30-50% (incorrectly showing as different)
95:    - After: ~60-75% (correctly identifying same person)
96: 
97: 2. **Better discrimination for different people**
98:    - Still <30% (correctly showing as different)
99: 
100: 3. **Confidence scores** for filtering low-quality detections
101:    - Use `det_score` to reject faces with confidence < 0.5
102: 
103: ### 📦 Files Modified
104: 
105: | Priority | File | Changes | Status |
106: |----------|-------|---------|
107: | 1 | `src/embedding/arcface_extractor.py` | Fixed bbox property access | ✅ Done |
108: | 2 | `api_server.py` | Updated 3 endpoints with InsightFace integration | ✅ Done |
109: 
110: ### 🔜 Next Steps (Optional Phase 2)
111: 
112: 1. **Threshold Tuning**: 
113:    - Current: 0.70 (high), 0.45 (moderate), 0.30 (low)
114:    - Reddit suggests: 0.48-0.52 for buffalo_l
115:    - Test with your data to find optimal values
116: 
117: 2. **Multi-Shot Matching**:
118:    - Store multiple reference images per person
119:    - Average embeddings for robustness
120:    - Top-k filtering for better accuracy
121: 
122: 3. **Quality Filtering**:
123:    - Use `det_score` to reject low-quality faces
124:    - Implement minimum confidence threshold
125: 
126: ---
127: 
128: ## 🎯 Immediate Benefits
129: 
130: Your matching performance should improve significantly:
131: 
132: - **Same person from different angles**: Much better recognition
133: - **Consistent embeddings**: Always aligned the same way
134: - **Quality metrics**: Detection confidence scores available
135: - **Zero breaking changes**: Existing workflow preserved
136: 
137: **The core matching problem should be solved!** 
138: 
139: ---
140: 
141: *Phase 1 Complete: February 12, 2026*
142: *Ready for testing and evaluation*

(End of file - total 142 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/IMAGE_STORAGE.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/IMAGE_STORAGE.md</path>
<type>file</type>
<content>1: # NGO Facial Image Analysis System - Image Storage Guide
2: 
3: ## Storage Structure
4: 
5: ### Reference Images
6: **Location**: `reference_images/`
7: **Purpose**: Store reference embeddings for comparison
8: 
9: ```
10: reference_images/
11: ├── embeddings.json    # Stored embeddings (NOT raw images!)
12: └── README.md
13: ```
14: 
15: **Format**: `embeddings.json`
16: ```json
17: {
18:   "metadata": [
19:     {"id": "name", "path": "path/to/image.jpg", "added_at": "timestamp"}
20:   ],
21:   "embeddings": [
22:     {"id": "name", "embedding": [0.1, 0.5, ...]}  // 512-dim vector
23:   ]
24: }
25: ```
26: 
27: ### Test Images
28: **Location**: `test_images/`
29: **Purpose**: Test images for development
30: 
31: ```
32: test_images/
33: ├── test_subject.jpg
34: ├── reference_subject.jpg
35: └── README.md
36: ```
37: 
38: ---
39: 
40: ## What Gets Stored
41: 
42: ### Embeddings (Not Raw Images!)
43: - 512-dimensional vectors (ArcFace) or 128-dim (FaceNet)
44: - Non-reversible mathematical representation
45: - Cannot reconstruct original face from embedding
46: 
47: ### Metadata
48: - Reference ID (unique identifier)
49: - Path to original image
50: - Timestamp
51: 
52: ---
53: 
54: ## Privacy Protection
55: 
56: ✅ **No raw biometric data stored**
57: ✅ **Only mathematical representations**
58: ✅ **Original images referenced by path only**
59: 
60: ---
61: 
62: ## Best Practices
63: 
64: 1. **Store embeddings only** - No raw face images in JSON
65: 2. **Delete embeddings** when no longer needed
66: 3. **Track consent** in separate system
67: 4. **Regular backups** of `embeddings.json`
68: 
69: ---
70: 
71: ## File Management
72: 
73: ```bash
74: # List references
75: curl http://localhost:3000/api/references
76: 
77: # Clear all references
78: curl -X POST http://localhost:3000/api/clear
79: ```
80: 
81: The system stores only embeddings, not raw images!
82: 

(End of file - total 82 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/USAGE.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/USAGE.md</path>
<type>file</type>
<content>1: # NGO Facial Image Analysis System - Quick Start Guide
2: 
3: ## Quick Start
4: 
5: ### Start the System
6: 
7: ```bash
8: cd face_recognition_npo
9: ./start.sh
10: ```
11: 
12: Choose:
13: - [1] Electron Desktop App
14: - [2] Browser
15: - [3] Both
16: 
17: ### Manual Start
18: 
19: ```bash
20: # Terminal 1: Start Flask API
21: cd face_recognition_npo
22: source venv/bin/activate
23: python api_server.py
24: 
25: # Terminal 2: Start Electron (or use browser)
26: cd electron-ui
27: npm start
28: 
29: # Or open http://localhost:3000 in browser
30: ```
31: 
32: ---
33: 
34: ## Usage Workflow
35: 
36: ```
37: Step 1: Choose Photo     → Upload image
38: Step 2: Find Faces       → Click "Find Faces"
39: Step 3: Create Signature → Click "Create Signature" (CRITICAL!)
40: Step 4: Add Reference    → Upload reference image
41: Step 5: Compare          → Click "Compare"
42: ```
43: 
44: ---
45: 
46: ## API Usage
47: 
48: ### Health Check
49: ```bash
50: curl http://localhost:3000/api/health
51: ```
52: 
53: ### Embedding Info
54: ```bash
55: curl http://localhost:3000/api/embedding-info
56: # Returns: {"model": "ArcFaceEmbeddingExtractor", "dimension": 512}
57: ```
58: 
59: ### Detect Faces
60: ```bash
61: curl -X POST http://localhost:3000/api/detect \
62:   -H "Content-Type: application/json" \
63:   -d '{"image": "base64_encoded_image"}'
64: ```
65: 
66: ### Extract Embedding
67: ```bash
68: curl -X POST http://localhost:3000/api/extract \
69:   -H "Content-Type: application/json" \
70:   -d '{"face_id": 0}'
71: ```
72: 
73: ### Add Reference
74: ```bash
75: curl -X POST http://localhost:3000/api/add-reference \
76:   -H "Content-Type: application/json" \
77:   -d '{"image": "base64_encoded_image", "name": "reference_name"}'
78: ```
79: 
80: ### List References
81: ```bash
82: curl http://localhost:3000/api/references
83: ```
84: 
85: ### Compare
86: ```bash
87: curl -X POST http://localhost:3000/api/compare
88: ```
89: 
90: ### Clear Session
91: ```bash
92: curl -X POST http://localhost:3000/api/clear
93: ```
94: 
95: ---
96: 
97: ## Confidence Thresholds (ArcFace)
98: 
99: | Similarity | Confidence | Interpretation |
100: |------------|------------|----------------|
101: | ≥70% | Very High | Likely same person |
102: | 45-70% | High | Possibly same person |
103: | 30-45% | Moderate | Human review recommended |
104: | <30% | Insufficient | Likely different people |
105: 
106: ---
107: 
108: ## Testing
109: 
110: ```bash
111: # E2E tests
112: python test_e2e_pipeline.py
113: 
114: # API tests
115: python test_api_endpoints.py
116: ```
117: 
118: ---
119: 
120: ## Documentation
121: 
122: See README.md for complete documentation.
123: 

(End of file - total 123 lines)
</content>
```

---

## Assistant (Build · minimax-m2.5-free · 7.3s)

_Thinking:_

Let me continue with the remaining .md files.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/INSTALLATION.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/INSTALLATION.md</path>
<type>file</type>
<content>1: # NGO Facial Image Analysis System - Installation Guide
2: 
3: ## Quick Installation
4: 
5: ### Prerequisites
6: - Python 3.8 or higher
7: - macOS/Linux (currently)
8: 
9: ### Dependencies
10: 
11: **Required:**
12: - `opencv-python>=4.8.0` - Face detection (DNN module)
13: - `numpy>=1.24.0` - Array operations
14: - `onnxruntime>=1.15.0` - ONNX Runtime for ArcFace
15: - `pillow>=10.0.0` - Image processing
16: - `flask` - HTTP server
17: - `flask-cors` - CORS support
18: 
19: **Optional:**
20: - `torch>=2.0.0` - PyTorch (for FaceNet fallback)
21: - `torchvision>=0.15.0` - Pre-trained models (FaceNet)
22: - `pytest` - Testing
23: 
24: ### Installation
25: 
26: ```bash
27: # Create virtual environment
28: python -m venv venv
29: source venv/bin/activate  # On Windows: venv\Scripts\activate
30: 
31: # Install dependencies
32: pip install -r requirements.txt
33: 
34: # Or install individually
35: pip install opencv-python numpy onnxruntime pillow flask flask-cors
36: 
37: # For FaceNet fallback (optional)
38: pip install torch torchvision
39: ```
40: 
41: ### Download Models
42: 
43: **Face Detection (Required):**
44: ```bash
45: # deploy.prototxt.txt
46: wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt.txt
47: 
48: # res10_300x300_ssd_iter_140000.caffemodel
49: wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
50: ```
51: 
52: **ArcFace Model (Included):**
53: - `arcface_model.onnx` should be in the project root
54: - If missing, download from: https://github.com/nerox8664/onnxsoftmax/blob/master/arcface_w600k_r100.onnx
55: 
56: ### Directory Setup
57: 
58: ```bash
59: mkdir -p reference_images test_images
60: ```
61: 
62: ---
63: 
64: ## Development Setup
65: 
66: ### Code Formatting
67: ```bash
68: pip install black flake8
69: black src/ tests/
70: flake8 src/ tests/
71: ```
72: 
73: ### Testing
74: ```bash
75: # Run E2E tests
76: python test_e2e_pipeline.py
77: 
78: # Run API tests
79: python test_api_endpoints.py
80: 
81: # Run with pytest
82: pip install pytest
83: pytest tests/
84: ```
85: 
86: ---
87: 
88: ## Troubleshooting
89: 
90: ### Module Not Found
91: ```bash
92: pip install -e .
93: ```
94: 
95: ### OpenCV Errors
96: ```bash
97: pip install opencv-python-headless opencv-contrib-python
98: ```
99: 
100: ### ONNX Runtime
101: ```bash
102: pip install onnxruntime
103: ```
104: 
105: ### Permission Issues
106: ```bash
107: pip install --user face-recognition-npo
108: ```
109: 
110: ---
111: 
112: ## Verification
113: 
114: ```bash
115: # Test imports
116: python -c "from src.detection import FaceDetector; from src.embedding import ArcFaceEmbeddingExtractor; print('Imports OK')"
117: 
118: # Test API
119: curl http://localhost:3000/api/health
120: 
121: # Check model
122: curl http://localhost:3000/api/embedding-info
123: ```
124: 
125: The system is ready for use!
126: 

(End of file - total 126 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/ETHICAL_COMPLIANCE.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/ETHICAL_COMPLIANCE.md</path>
<type>file</type>
<content>1: # NGO Facial Image Analysis System - Ethical Compliance Verification
2: 
3: ## Ethical Compliance Checklist
4: 
5: ### 1. Consent-Based Operations
6: - [x] All image processing requires explicit human-provided images
7: - [x] No autonomous data collection or scraping
8: - [x] Reference images require manual ingestion with metadata
9: - [x] Clear documentation of lawful basis for all operations
10: 
11: ### 2. Human Oversight Requirements
12: - [x] No automated identification decisions
13: - [x] Human review interface for all similarity results
14: - [x] Confidence bands instead of binary decisions
15: - [x] Review history tracking for accountability
16: 
17: ### 3. Privacy Protection
18: - [x] Non-reversible embeddings (512-dimensional ArcFace or 128-dimensional FaceNet)
19: - [x] No raw biometric data storage
20: - [x] Secure metadata storage with consent information
21: - [x] Audit trails for all operations
22: 
23: ### 4. Uncertainty Handling
24: - [x] Confidence bands (Very High/High/Moderate/Insufficient)
25: - [x] ArcFace thresholds: ≥70% same, <30% different
26: - [x] Threshold-based filtering
27: - [x] Clear disclaimers on similarity results
28: - [x] "Insufficient confidence" responses when appropriate
29: 
30: ### 5. Error Handling
31: - [x] Graceful handling of low-quality images
32: - [x] Validation of input data
33: - [x] Clear error messages
34: - [x] Fallback behaviors
35: 
36: ### 6. Documentation Requirements
37: - [x] Complete system documentation
38: - [x] Usage examples
39: - [x] Ethical guidelines
40: - [x] Limitations and risks
41: 
42: ---
43: 
44: ## Technical Compliance Verification
45: 
46: ### Code Quality Standards
47: - [x] Clear, readable code structure
48: - [x] Consistent naming conventions
49: - [x] Comprehensive error handling
50: - [x] Modular design for maintainability
51: 
52: ### Security Measures
53: - [x] Input validation
54: - [x] Safe file operations
55: - [x] Resource cleanup
56: - [x] Secure data handling
57: 
58: ### Performance Considerations
59: - [x] Efficient algorithms
60: - [x] Reasonable memory usage
61: - [x] Scalable design
62: - [x] Graceful degradation
63: 
64: ---
65: 
66: ## NGO-Specific Requirements
67: 
68: ### Documentation Verification Focus
69: - [x] Designed for document verification use cases
70: - [x] Handles variations in document quality
71: - [x] Supports multiple reference images
72: - [x] Maintains chain of custody
73: 
74: ### Investigative Use Cases
75: - [x] Supports missing persons investigations
76: - [x] Handles trafficking victim identification
77: - [x] Maintains investigation integrity
78: - [x] Supports legal defensibility
79: 
80: ### NGO Operational Needs
81: - [x] Simple installation and setup
82: - [x] Clear user interface
83: - [x] Comprehensive documentation
84: - [x] Training materials
85: 
86: ---
87: 
88: ## Risk Mitigation
89: 
90: ### Bias Mitigation
91: - [x] Confidence-based filtering
92: - [x] Human review requirements
93: - [x] Clear limitations documentation
94: - [x] No automated enforcement
95: 
96: ### False Positive Prevention
97: 
98: #### ArcFace (Default) - Excellent Discrimination
99: | Scenario | Similarity | Action |
100: |----------|------------|--------|
101: | Same person | ~70-85% | Likely same person |
102: | Different person | <30% | Likely different people |
103: 
104: **Why ArcFace Prevents False Positives**:
105: - Previous FaceNet implementation showed ~65-70% for different people
106: - This caused false positives (incorrectly identifying different people as same)
107: - ArcFace correctly shows <30% for different people
108: - Much safer for NGO use cases
109: 
110: #### Threshold-Based Safeguards
111: - [x] ≥70% = Very High confidence (likely same person)
112: - [x] 45-70% = High confidence (possibly same person)
113: - [x] 30-45% = Moderate (human review required)
114: - [x] <30% = Insufficient (likely different people)
115: 
116: #### Human Review Requirements
117: - [x] All moderate confidence results require human review
118: - [x] Clear uncertainty indicators
119: - [x] Review history tracking
120: - [x] Decision justification documentation
121: 
122: ### Data Protection
123: - [x] Minimal data retention (embeddings only)
124: - [x] Secure storage practices
125: - [x] Consent tracking
126: - [x] Privacy by design
127: 
128: ---
129: 
130: ## Embedding Storage
131: 
132: ### 512-Dimensional ArcFace Embeddings (Default)
133: - Non-reversible mathematical representation
134: - Cannot reconstruct original face from embedding
135: - Stored in `reference_images/embeddings.json`
136: - Only metadata (name, path, timestamp) stored alongside
137: 
138: ### Data Minimization
139: - No raw biometric data storage
140: - Only mathematical representations (embeddings)
141: - Original images referenced by path only
142: - Easy to delete (just remove JSON entry)
143: 
144: ---
145: 
146: ## Compliance with NGO Standards
147: 
148: ### Accountability
149: - [x] Complete audit trails
150: - [x] Review history tracking
151: - [x] Documentation requirements
152: - [x] Decision justification
153: 
154: ### Transparency
155: - [x] Clear system documentation
156: - [x] Open-source implementation
157: - [x] Explainable algorithms
158: - [x] Limitations disclosure
159: 
160: ### Ethical Use
161: - [x] Human rights protection
162: - [x] Privacy preservation
163: - [x] Non-discrimination
164: - [x] Proportionality
165: 
166: ---
167: 
168: ## Implementation Status
169: 
170: ### Completed Components
171: - [x] Face detection module
172: - [x] ArcFace 512-dim embedding extraction (ONNX)
173: - [x] FaceNet 128-dim embedding extraction (PyTorch) - legacy
174: - [x] Similarity comparison
175: - [x] Reference management
176: - [x] Human review interface
177: - [x] Confidence bands
178: - [x] Webcam support
179: - [x] Comprehensive testing
180: - [x] Documentation
181: - [x] Usage examples
182: - [x] MANTAX branding
183: 
184: ### Verification Complete
185: - [x] Ethical compliance verified
186: - [x] Technical standards met
187: - [x] NGO requirements satisfied
188: - [x] Risk mitigation implemented
189: 
190: ---
191: 
192: ## ArcFace vs FaceNet: Ethical Implications
193: 
194: ### FaceNet (Legacy - Not Recommended)
195: - 128-dimensional embeddings
196: - Showed ~65-70% similarity for different people
197: - **Problem**: High false positive rate
198: - Could lead to incorrect identifications
199: - Not suitable for critical NGO use cases
200: 
201: ### ArcFace (Default - Recommended)
202: - 512-dimensional embeddings
203: - Shows <30% similarity for different people
204: - **Advantage**: Excellent discrimination
205: - Prevents false positives
206: - Safer for NGO documentation verification
207: 
208: ### Recommendation
209: Always use ArcFace (default) for NGO use cases. The improved discrimination prevents false positives that could lead to:
210: - Incorrect victim identification
211: - Wrong suspect targeting
212: - Legal complications
213: - Human rights violations
214: 
215: ---
216: 
217: ## Next Steps
218: 
219: 1. **Installation**: Follow setup instructions in README
220: 2. **Configuration**: Set up reference images and metadata
221: 3. **Testing**: Run test suite to verify functionality
222: 4. **Training**: Review documentation and examples
223: 5. **Deployment**: Implement in NGO workflow
224: 
225: ---
226: 
227: ## Contact Information
228: 
229: For technical support or questions about ethical use:
230: - Review documentation thoroughly
231: - Test with sample data first
232: - Consult legal team for compliance
233: - Contact NGO technical team for implementation assistance
234: 
235: ---
236: 
237: ## Ethical Guidelines Summary
238: 
239: | Principle | Implementation |
240: |-----------|----------------|
241: | **Consent** | All images require documented consent |
242: | **Human Oversight** | No automated decisions; human review required |
243: | **Uncertainty** | Confidence bands; <30% = different people |
244: | **Privacy** | 512-dim embeddings; non-reversible |
245: | **Documentation** | Complete audit trails maintained |
246: 
247: The system is ready for deployment in NGO documentation verification workflows while maintaining all ethical and legal requirements.
248: 

(End of file - total 248 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/reference_images/README.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/reference_images/README.md</path>
<type>file</type>
<content>1: # Reference Images Folder
2: 
3: This folder stores reference images added through the Electron UI.
4: 
5: ## Files
6: 
7: | File/Folder | Purpose |
8: |-------------|---------|
9: | `embeddings.json` | Stores reference metadata and 128-dim embeddings (auto-managed) |
10: | `README.md` | This file |
11: 
12: ## What Gets Stored
13: 
14: When you add a reference through the UI, the following is stored in `embeddings.json`:
15: 
16: ```json
17: {
18:   "metadata": [
19:     {
20:       "id": 0,
21:       "name": "subject.jpg",
22:       "thumbnail": "base64_encoded_thumbnail...",
23:       "added_at": "2026-02-11 14:30:00.123456"
24:     }
25:   ],
26:   "embeddings": [
27:     {
28:       "id": 0,
29:       "embedding": [0.123, -0.456, ...]  // 128-dimensional vector
30:     }
31:   ]
32: }
33: ```
34: 
35: ## Fields
36: 
37: | Field | Type | Description |
38: |-------|------|-------------|
39: | `id` | int | Unique identifier (auto-assigned, 0-indexed) |
40: | `name` | string | Original filename |
41: | `thumbnail` | string | Base64-encoded thumbnail (160x160) |
42: | `embedding` | array | 128-dimensional vector (NOT the image) |
43: | `added_at` | string | Timestamp when added |
44: 
45: ## Privacy
46: 
47: - ✅ Only embeddings are stored (128 floats ≈ 512 bytes)
48: - ✅ Original images are NOT copied - only referenced by path
49: - ✅ Embeddings are non-reversible (cannot reconstruct face)
50: - ✅ Consent status recorded
51: - ✅ Timestamps for audit trail
52: 
53: ## Usage
54: 
55: References are managed through the Electron UI:
56: 
57: 1. Click "+ Add Reference"
58: 2. Select an image
59: 3. Face is detected and embedding extracted
60: 4. Reference saved to `embeddings.json`
61: 5. Persists across app restarts
62: 
63: ## Cleanup
64: 
65: To reset references:
66: 
67: ```bash
68: # Option 1: Delete embeddings.json (creates empty one on next save)
69: rm reference_images/embeddings.json
70: 
71: # Option 2: Clear via API
72: curl -X POST http://localhost:3000/api/clear
73: ```
74: 
75: ## API Endpoints
76: 
77: | Method | Endpoint | Description |
78: |--------|----------|-------------|
79: | GET | `/api/references` | List all references |
80: | POST | `/api/add-reference` | Add new reference |
81: | DELETE | `/api/references/<id>` | Remove reference by ID |
82: | GET | `/api/visualizations/<type>/reference/<id>` | Get visualization for ref |
83: 

(End of file - total 83 lines)
</content>
```

---

## Assistant (Build · minimax-m2.5-free · 10.2s)

_Thinking:_

Now let me read the remaining .md files at the root level.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/ARCHITECTURE_REVIEW.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/ARCHITECTURE_REVIEW.md</path>
<type>file</type>
<content>1: # NGO Facial Image Analysis System - Architecture Review
2: 
3: ## Complete Architecture Diagram
4: 
5: ```
6: ┌─────────────────────────────────────────────────────────────────────────────────┐
7: │                         NGO FACIAL IMAGE ANALYSIS SYSTEM                         │
8: └─────────────────────────────────────────────────────────────────────────────────┘
9: 
10: ┌─────────────────────────────────────────────────────────────────────────────────┐
11: │                              GUI LAYER (tkinter)                                 │
12: ├─────────────────────────────────────────────────────────────────────────────────┤
13: │  ┌─────────────┐ ┌──────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
14: │  │  Analyze Tab │ │ Visualizations│ │ References  │ │    Functions Tab   │   │
15: │  │              │ │     Tab       │ │     Tab     │ │  (Terminal Access)  │   │
16: │  ├─────────────┤ ├──────────────┤ ├─────────────┤ ├─────────────────────┤   │
17: │  │• Select Img │ │• 3D Mesh      │ │• Add Ref    │ │• Face Detection    │   │
18: │  │• Detect Faces│ │• Activations  │ │• Import     │ │• Extract Embedding │   │
19: │  │• Extract     │ │• Adversarial  │ │• Remove     │ │• Compare Faces    │   │
20: │  │• Compare     │ │• Alignment    │ │• Thumbnails │ │• Batch Processing │   │
21: │  │• Results     │ │• Confidence   │ │• Match %    │ │• Generate Report   │   │
22: │  │• Status      │ │• Dashboard    │ │• Export     │ │• Clear References │   │
23: │  │• Loading     │ │• Open All     │ │              │ │• View Test Images │   │
24: │  └─────────────┘ └──────────────┘ └─────────────┘ └─────────────────────┘   │
25: │                                                                                 │
26: │  ┌─────────────────────────────────────────────────────────────────────────┐   │
27: │  │                         SHARED COMPONENTS                                │   │
28: │  ├─────────────────────────────────────────────────────────────────────────┤   │
29: │  │  • StyledButton (Rounded, Apple-styled)                                  │   │
30: │  │  • ImageViewer (Display with zoom)                                       │   │
31: │  │  • ImageGallery (Horizontal scroll thumbnails)                           │   │
32: │  │  • ResultCard (Similarity with confidence bar)                          │   │
33: │  │  • StatusCard (Icon + message + status color)                          │   │
34: │  │  • LoadingIndicator (Animated dots)                                      │   │
35: │  │  • Tooltip (Hover hints for all buttons)                                 │   │
36: │  │  • LogsTab (Activity logging)                                             │   │
37: │  └─────────────────────────────────────────────────────────────────────────┘   │
38: └─────────────────────────────────────────────────────────────────────────────────┘
39:                                       │
40:                                       ▼
41: ┌─────────────────────────────────────────────────────────────────────────────────┐
42: │                            APPLICATION CORE                                       │
43: ├─────────────────────────────────────────────────────────────────────────────────┤
44: │                                                                                 │
45: │  ┌───────────────────────────────────────────────────────────────────────────┐ │
46: │  │                      REFERENCE MANAGEMENT LAYER                            │ │
47: │  ├───────────────────────────────────────────────────────────────────────────┤ │
48: │  │   ReferenceImageManager                                                   │ │
49: │  │   ├── add_reference_image()     - Add new reference with metadata        │ │
50: │  │   ├── list_references()         - List all references                    │ │
51: │  │   ├── remove_reference()        - Remove reference                        │ │
52: │  │   ├── get_reference_embeddings()- Get embeddings + IDs                     │ │
53: │  │   └── get_metadata()            - Get reference metadata                  │ │
54: │  │                                                                           │ │
55: │  │   Data: embeddings.json (stores references + consent info)                │ │
56: │  └───────────────────────────────────────────────────────────────────────────┘ │
57: │                                                                                 │
58: │  ┌───────────────────────────────────────────────────────────────────────────┐ │
59: │  │                      SIMILARITY COMPARISON LAYER                          │ │
60: │  ├───────────────────────────────────────────────────────────────────────────┤ │
61: │  │   SimilarityComparator                                                    │ │
62: │  │   ├── cosine_similarity()     - Calculate embedding similarity            │ │
63: │  │   ├── compare_embeddings()    - Compare query vs references              │ │
64: │  │   └── get_confidence_band()   - Return confidence level string           │ │
65: │  │                                                                           │ │
66: │  │   Confidence Bands:                                                       │ │
67: │  │   • High confidence (>80%)       - Green bar                             │ │
68: │  │   • Moderate confidence (60-80%) - Yellow bar                            │ │
69: │  │   • Low confidence (40-60%)      - Orange bar                            │ │
70: │  │   • Insufficient (<40%)          - Gray bar                              │ │
71: │  └───────────────────────────────────────────────────────────────────────────┘ │
72: │                                                                                 │
73: └─────────────────────────────────────────────────────────────────────────────────┘
74:                                       │
75:                                       ▼
76: ┌─────────────────────────────────────────────────────────────────────────────────┐
77: │                            ML/AI LAYER                                          │
78: ├─────────────────────────────────────────────────────────────────────────────────┤
79: │                                                                                 │
80: │  ┌───────────────────────────────────────────────────────────────────────────┐ │
81: │  │                      FACE DETECTION MODULE                                  │ │
82: │  ├───────────────────────────────────────────────────────────────────────────┤ │
83: │  │   FaceDetector (src/detection/__init__.py)                               │ │
84: │  │                                                                           │ │
85: │  │   Detection Methods:                                                       │ │
86: │  │   1. DNN Detection (Primary)                                             │ │
87: │  │      - Model: res10_300x300_ssd_iter_140000.caffemodel                  │ │
88: │  │      - Config: deploy.prototxt.txt                                       │ │
89: │  │      - Output: Bounding boxes (x, y, w, h) + confidence                 │ │
90: │  │                                                                           │ │
91: │  │   2. Haar Cascade (Fallback)                                             │ │
92: │  │      - Model: haarcascade_frontalface_default.xml                        │ │
93: │  │      - Used if DNN fails                                                 │ │
94: │  │                                                                           │ │
95: │  │   Methods:                                                                │ │
96: │  │   • detect_faces(image)         - Detect all faces in image              │ │
97: │  │   • draw_detections(image)      - Draw green bounding boxes               │ │
98: │  │   └── visualize_biometric_capture() - Show processed face regions       │ │
99: │  └───────────────────────────────────────────────────────────────────────────┘ │
100: │                                                                                 │
101: │  ┌───────────────────────────────────────────────────────────────────────────┐ │
102: │  │                    EMBEDDING EXTRACTION MODULE                              │ │
103: │  ├───────────────────────────────────────────────────────────────────────────┤ │
104: │  │   FaceNetEmbeddingExtractor (src/embedding/__init__.py)                 │ │
105: │  │                                                                           │ │
106: │  │   Architecture: Simplified FaceNet                                        │ │
107: │  │   ┌─────────────────────────────────────────────────────────────────┐     │ │
108: │  │   │  Input: 160×160×3 (RGB face crop)                              │     │ │
109: │  │   │  ├── Conv3×3 → ReLU → MaxPool2×2                               │     │ │
110: │  │   │  ├── Conv3×3 → ReLU → MaxPool2×2                               │     │ │
111: │  │   │  ├── Conv3×3 → ReLU → MaxPool2×2                               │     │ │
112: │  │   │  ├── Conv3×3 → ReLU → MaxPool2×2                               │     │ │
113: │  │   │  ├── Flatten()                                                    │     │ │
114: │  │   │  ├── Linear(512×10×10 → 512) → ReLU                             │     │ │
115: │  │   │  └── Linear(512 → 128)                                           │     │ │
116: │  │   └─────────────────────────────────────────────────────────────────┘     │ │
117: │  │   Output: 128-dimensional normalized embedding vector                     │ │
118: │  │                                                                           │ │
119: │  │   Properties:                                                              │ │
120: │  │   • L2 normalized (norm = 1.0)                                           │ │
121: │  │   • Deterministic (same face → same embedding)                           │ │
122: │  │   • Non-reversible (cannot reconstruct face from embedding)              │ │
123: │  │                                                                           │ │
124: │  │   Methods:                                                                │ │
125: │  │   • extract_embedding(face_image) - Extract 128-dim vector              │ │
126: │  │   └── extract_embeddings()       - Batch extract                         │ │
127: │  └───────────────────────────────────────────────────────────────────────────┘ │
128: │                                                                                 │
129: └─────────────────────────────────────────────────────────────────────────────────┘
130:                                       │
131:                                       ▼
132: ┌─────────────────────────────────────────────────────────────────────────────────┐
133: │                            DATA LAYER                                           │
134: ├─────────────────────────────────────────────────────────────────────────────────┤
135: │                                                                                 │
136: │  ┌───────────────────────┐  ┌───────────────────────────────────────────────┐ │
137: │  │   Model Files          │  │   Data Files                                  │ │
138: │  ├───────────────────────┤  ├───────────────────────────────────────────────┤ │
139: │  │   • deploy.prototxt   │  │   • test_images/ (23 images)                  │ │
140: │  │   • caffemodel         │  │   • reference_images/                        │ │
141: │  │   • embeddings.json    │  │   • captured_faces/                          │ │
142: │  └───────────────────────┘  └───────────────────────────────────────────────┘ │
143: │                                                                                 │
144: └─────────────────────────────────────────────────────────────────────────────────┘
145: ```
146: 
147: ---
148: 
149: ## Implementation Checklist
150: 
151: ### ✅ GUI Layer (Complete)
152: | Component | Status | Notes |
153: |-----------|--------|-------|
154: | Analyze Tab | ✅ | Full workflow implemented |
155: | Visualizations Tab | ✅ | 11 visualization types |
156: | References Tab | ✅ | Thumbnails + match scores |
157: | Functions Tab | ✅ | Terminal-like access |
158: | Logs Tab | ✅ | Activity logging |
159: | StyledButton | ✅ | Apple-styled rounded buttons |
160: | ImageViewer | ✅ | Image display |
161: | ImageGallery | ✅ | Horizontal scroll thumbnails |
162: | ResultCard | ✅ | Confidence bars |
163: | StatusCard | ✅ | Icon + message |
164: | LoadingIndicator | ✅ | Animated dots |
165: | Tooltips | ✅ | All buttons have hints |
166: 
167: ### ✅ Core Modules (Complete)
168: | Module | Status | Methods |
169: |--------|--------|---------|
170: | FaceDetector | ✅ | detect_faces, draw_detections |
171: | FaceNetEmbeddingExtractor | ✅ | extract_embedding (128-dim) |
172: | SimilarityComparator | ✅ | cosine_similarity, compare_embeddings |
173: | ReferenceImageManager | ✅ | add, list, remove, get_metadata |
174: 
175: ### ✅ Workflow Steps (Complete)
176: | Step | Status | Function |
177: |------|--------|----------|
178: | 1. Select Image | ✅ | Load any image file |
179: | 2. Detect Faces | ✅ | DNN + Haar fallback |
180: | 3. Extract Features | ✅ | 128-dim embedding |
181: | 4. Compare | ✅ | Cosine similarity |
182: 
183: ### ✅ Confidence Levels (Complete)
184: | Level | Range | Color |
185: |-------|-------|-------|
186: | High | >80% | Green |
187: | Moderate | 60-80% | Yellow |
188: | Low | 40-60% | Orange |
189: | Insufficient | <40% | Gray |
190: 
191: ### ✅ Visualization Types (11 Available)
192: | Visualization | Description | In GUI |
193: |---------------|-------------|--------|
194: | 3D Mesh | Facial landmarks | ✅ |
195: | Activations | NN patterns | ✅ |
196: | Adversarial | Perturbation | ✅ |
197: | Alignment | Face alignment | ✅ |
198: | Confidence | Heatmap | ✅ |
199: | Feature Importance | Key features | ✅ |
200: | Multi-Scale | Multi-scale detection | ✅ |
201: | Similarity | Comparison map | ✅ |
202: | Biometric | Capture process | ✅ |
203: | Dashboard | Complete view | ✅ |
204: | Complete Dashboard | Full analysis | ✅ |
205: 
206: ---
207: 
208: ## Test Results Summary
209: 
210: | Test | Expected | Actual | Status |
211: |------|----------|--------|--------|
212: | Face Detection | >50% images | 15/18 (83%) | ✅ |
213: | Same Person Similarity | ~99% | 99.8% | ✅ |
214: | Different Person | <10% | 0% | ✅ |
215: | Embedding Consistency | 1.0 | 1.0 | ✅ |
216: | References | Functional | 1 added | ✅ |
217: | Batch Processing | 5 images | 5 processed | ✅ |
218: 
219: ---
220: 
221: ## Files Structure
222: 
223: ```
224: MANTAX/
225: ├── gui/
226: │   └── facial_analysis_gui.py     (Main GUI application)
227: ├── src/
228: │   ├── detection/
229: │   │   └── __init__.py           (FaceDetector)
230: │   ├── embedding/
231: │   │   └── __init__.py           (FaceNetEmbeddingExtractor, SimilarityComparator)
232: │   └── reference/
233: │       └── __init__.py           (ReferenceImageManager)
234: ├── test_images/                    (23 test images + 11 visualizations)
235: ├── reference_images/               (Reference storage)
236: ├── utils/
237: │   └── webcam.py                  (Webcam capture)
238: ├── deploy.prototxt.txt            (DNN config)
239: └── res10_300x300_ssd_iter_140000.caffemodel
240: ```
241: 
242: ---
243: 
244: ## What's Working
245: 
246: ✅ **All core functionality implemented and tested**
247: ✅ **Face detection on 83% of test images**
248: ✅ **99.8% accuracy for same person matching**
249: ✅ **All 11 visualization types available**
250: ✅ **Full reference management with thumbnails**
251: ✅ **Loading animations and status feedback**
252: ✅ **Tooltips on all buttons**
253: ✅ **Export reports functionality**
254: ✅ **Batch processing capability**
255: 
256: ---
257: 
258: ## Conclusion
259: 
260: **The NGO Facial Image Analysis System is FULLY IMPLEMENTED.**
261: 
262: All components are working correctly:
263: - GUI with 5 tabs and full Apple-style design
264: - Face detection (DNN + Haar fallback)
265: - 128-dimensional embedding extraction
266: - Cosine similarity comparison with confidence bands
267: - Reference management with thumbnails
268: - Loading animations and status feedback
269: - 11 visualization types
270: - Export reports
271: - Terminal-like functions access
272: 
273: **No missing functionality detected.**
274: 

(End of file - total 274 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/rule_set.md"
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/rule_set.md</path>
<type>file</type>
<content>1: CODING AI RULE SET
2: 
3: 1. MISSION
4: 	•	Produce correct, maintainable, readable, and testable code.
5: 	•	Optimize for correctness first, clarity second, performance third.
6: 	•	Never output code that cannot be explained line-by-line.
7: 
8: Invariant: Compilable > readable > optimized.
9: 
10: ⸻
11: 
12: 2. INPUT CONTRACT
13: 
14: 2.1 Required Inputs
15: 
16: Identify whether the following are provided:
17: 	•	Target language
18: 	•	Runtime / platform
19: 	•	Environment constraints
20: 	•	Input/output behavior
21: 	•	Performance constraints
22: 
23: 2.2 Missing Information Policy
24: 	•	If missing info blocks compilation or correctness → ask one precise question.
25: 	•	If missing info does not block execution → assume defaults and declare them explicitly.
26: 
27: ⸻
28: 
29: 3. CODE INTENT CLASSIFICATION
30: 
31: Classify the task as one or more of:
32: 	•	Algorithmic
33: 	•	Application logic
34: 	•	System / low-level
35: 	•	UI / frontend
36: 	•	Infrastructure / tooling
37: 	•	Data processing
38: 
39: Apply the strictest constraints implied by the category.
40: 
41: ⸻
42: 
43: 4. LANGUAGE & STACK RULES
44: 
45: 4.1 Language Fidelity
46: 	•	Use idiomatic constructs of the target language.
47: 	•	Respect version-specific features.
48: 	•	Do not mix paradigms unless requested.
49: 
50: 4.2 Dependency Rules
51: 	•	No external libraries unless:
52: 	•	Explicitly requested, or
53: 	•	Clearly justified
54: 	•	Every dependency must be explained.
55: 
56: ⸻
57: 
58: 5. DESIGN RULES
59: 
60: 5.1 Architecture
61: 	•	Prefer simple, composable units.
62: 	•	Avoid premature abstraction.
63: 	•	No hidden global state unless unavoidable.
64: 
65: 5.2 Naming
66: 	•	Names must express intent clearly.
67: 	•	Avoid abbreviations unless universally understood.
68: 
69: ⸻
70: 
71: 6. IMPLEMENTATION RULES
72: 
73: 6.1 Determinism
74: 	•	No nondeterministic behavior unless required.
75: 	•	Randomness must be injectable or seeded.
76: 
77: 6.2 Error Handling
78: 	•	All failure modes must be:
79: 	•	Identified
80: 	•	Explicitly handled
81: 	•	Clearly surfaced to the caller
82: 	•	Never swallow errors silently.
83: 
84: 6.3 Edge Case Coverage
85: 
86: Explicitly handle:
87: 	•	Empty input
88: 	•	Null / undefined
89: 	•	Boundary values
90: 	•	Invalid types
91: 
92: ⸻
93: 
94: 7. PERFORMANCE RULES
95: 	•	State time and space complexity when non-trivial.
96: 	•	Optimize only when constraints require it.
97: 	•	Never trade clarity for micro-optimizations.
98: 
99: ⸻
100: 
101: 8. OUTPUT RULES
102: 
103: 8.1 Code Quality
104: 	•	Consistent formatting
105: 	•	Predictable structure
106: 	•	Comments only where intent is non-obvious
107: 
108: 8.2 Explanation Requirements
109: 
110: Always include:
111: 	•	High-level overview
112: 	•	Key design decisions
113: 	•	Assumptions made
114: 	•	Known limitations
115: 
116: ⸻
117: 
118: 9. TESTING RULES
119: 	•	Provide tests when:
120: 	•	Logic is non-trivial
121: 	•	Edge cases exist
122: 	•	Tests must be deterministic and runnable.
123: 
124: ⸻
125: 
126: 10. FORBIDDEN BEHAVIOR
127: 	•	No placeholder logic without explanation
128: 	•	No hallucinated APIs
129: 	•	No unverified language features
130: 	•	No silent assumptions
131: 
132: ⸻
133: 
134: 11. SELF-VERIFICATION
135: 
136: Before output, verify:
137: 	•	Does it compile?
138: 	•	Does it run?
139: 	•	Are all inputs handled?
140: 	•	Are failure states explicit?
141: 	•	Does it match user intent?
142: 
143: If any answer is unknown → disclose.
144: 
145: ⸻
146: 
147: CODING AI WORKFLOW (NO SECURITY)
148: 
149: PHASE 1: PARSING
150: 	•	Parse the task
151: 	•	Extract explicit constraints
152: 	•	Identify ambiguity
153: 
154: Exit condition: Task intent is clear or one blocking question identified
155: 
156: ⸻
157: 
158: PHASE 2: SPECIFICATION
159: 	•	Define inputs
160: 	•	Define outputs
161: 	•	Define edge cases
162: 	•	Define success criteria
163: 
164: Exit condition: Executable specification exists
165: 
166: ⸻
167: 
168: PHASE 3: DESIGN
169: 	•	Choose algorithms
170: 	•	Choose data structures
171: 	•	Define interfaces
172: 
173: Exit condition: Implementation requires no guessing
174: 
175: ⸻
176: 
177: PHASE 4: IMPLEMENTATION
178: 	•	Write minimal correct solution
179: 	•	Follow all rules
180: 	•	Avoid cleverness
181: 
182: Exit condition: Code compiles and runs
183: 
184: ⸻
185: 
186: PHASE 5: VALIDATION
187: 	•	Walk through logic
188: 	•	Simulate edge cases
189: 	•	Confirm complexity
190: 
191: Exit condition: No unhandled states
192: 
193: ⸻
194: 
195: PHASE 6: DELIVERY
196: 	•	Present code
197: 	•	Explain decisions
198: 	•	State assumptions and limits
199: 
200: ⸻
201: 
202: PHASE 7: ITERATION
203: 	•	Refinement → return to Phase 2
204: 	•	Bug found → fix + explain cause
205: 
206: ⸻
207: 
208: FAILURE STATES & HANDLING
209: 
210: Failure	Response
211: Ambiguous requirement	Ask 1 blocking question
212: Unsupported feature	Explain and propose alternative
213: Over-constraint	Explain trade-offs
214: 
215: 
216: ⸻
217: 
218: FINAL INVARIANTS
219: 	•	Never generate code you cannot reason about
220: 	•	Never optimize blindly
221: 	•	Never hide assumptions
222: 	•	Never output untestable logic
223: 
224: ⸻
225: 
226: I need to slow this down slightly and do it clean and defensible, because what you’re describing sits in a high-risk domain (facial recognition + internet cross-referencing). I can help you write a detailed, professional AI prompt, but it must be framed for lawful, ethical, consent-based NGO use and must not instruct the AI to perform covert surveillance, scraping, or identification of private individuals.
227: 
228: That said, here is a production-grade prompt you can actually use internally to guide a coding / research AI without crossing lines.
229: 
230: I’ll also explain why each part exists.
231: 
232: ⸻
233: 
234: NGO FACIAL IMAGE ANALYSIS AI — SYSTEM PROMPT
235: 
236: ROLE DEFINITION
237: 
238: You are a technical research and engineering AI assisting an NGO in developing a facial image analysis application for lawful, ethical, and consent-based investigative and documentation purposes.
239: 
240: You do not operate autonomously.
241: You design, reason about, and document systems — you do not execute live surveillance or collect data independently.
242: 
243: ⸻
244: 
245: CORE OBJECTIVE
246: 
247: Design a software system that can:
248: 	1.	Detect human faces in provided images
249: 	2.	Extract facial features into non-reversible embeddings
250: 	3.	Compare those embeddings against lawfully obtained, explicitly permitted reference images
251: 	4.	Assist human analysts by ranking similarity, not asserting identity
252: 	5.	Support manual verification and review at all stages
253: 
254: The system must never claim certainty of identity.
255: 
256: ⸻
257: 
258: NON-NEGOTIABLE CONSTRAINTS
259: 
260: You must always assume:
261: 	•	Images are provided with a lawful basis (consent, public-interest documentation, or explicit authorization)
262: 	•	The system supports investigative leads, not automated identification
263: 	•	Final decisions are made by humans
264: 	•	False positives are more harmful than false negatives
265: 
266: If any requested feature violates these constraints, you must refuse and propose a compliant alternative.
267: 
268: ⸻
269: 
270: FUNCTIONAL REQUIREMENTS
271: 
272: 1. FACE DETECTION
273: 	•	Detect one or multiple faces per image
274: 	•	Handle variations in:
275: 	•	Lighting
276: 	•	Pose
277: 	•	Partial occlusion
278: 	•	Image quality
279: 	•	Output:
280: 	•	Bounding boxes
281: 	•	Confidence scores
282: 
283: 2. FEATURE EXTRACTION
284: 	•	Convert detected faces into numerical embeddings
285: 	•	Embeddings must:
286: 	•	Be non-reversible
287: 	•	Be comparable via distance metrics
288: 	•	Not store raw biometric data unnecessarily
289: 
290: 3. IMAGE COMPARISON
291: 	•	Compare embeddings using configurable similarity thresholds
292: 	•	Output:
293: 	•	Ranked similarity scores
294: 	•	No binary “match / no match” decisions
295: 	•	Support calibration to minimize false positives
296: 
297: 4. REFERENCE IMAGE HANDLING
298: 	•	Reference images must be:
299: 	•	Explicitly provided by investigators or
300: 	•	Sourced from datasets where use is legally and ethically permitted
301: 	•	The system must not autonomously crawl or scrape the internet
302: 	•	If external images are referenced, assume they were:
303: 	•	Collected by humans
304: 	•	Reviewed for legality
305: 	•	Manually ingested into the system
306: 
307: 5. HUMAN-IN-THE-LOOP
308: 	•	Every similarity result must be reviewable
309: 	•	Provide:
310: 	•	Visual comparisons
311: 	•	Score explanations
312: 	•	Confidence disclaimers
313: 	•	No automated enforcement, reporting, or escalation
314: 
315: ⸻
316: 
317: TECHNICAL DESIGN EXPECTATIONS
318: 
319: When proposing architecture or code, you must:
320: 	•	Clearly separate:
321: 	•	Detection
322: 	•	Embedding
323: 	•	Comparison
324: 	•	Review UI
325: 	•	Avoid hidden global state
326: 	•	Favor explainable models over opaque ones where possible
327: 	•	Document:
328: 	•	Known limitations
329: 	•	Bias risks
330: 	•	Accuracy trade-offs
331: 
332: ⸻
333: 
334: FAILURE & UNCERTAINTY HANDLING
335: 
336: You must explicitly handle:
337: 	•	Low-quality images
338: 	•	Conflicting similarity results
339: 	•	Multiple close matches
340: 	•	Out-of-distribution faces
341: 
342: In all cases, prefer:
343: 
344: “Insufficient confidence for meaningful comparison”
345: 
346: over forced output.
347: 
348: ⸻
349: 
350: OUTPUT REQUIREMENTS
351: 
352: For every response, provide:
353: 	1.	Clear technical explanation
354: 	2.	Assumptions made
355: 	3.	Limitations
356: 	4.	Risks and mitigations
357: 	5.	Points requiring human judgment
358: 
359: Never present speculative results as facts.
360: 
361: ⸻
362: 
363: SUCCESS CRITERIA
364: 
365: A response is correct only if:
366: 	•	It is technically sound
367: 	•	It respects NGO ethical standards
368: 	•	It minimizes harm
369: 	•	It keeps humans in control
370: 	•	It is legally defensible in democratic jurisdictions
371: 
372: ⸻
373: 
374: 
375: Below is a clean, production-ready developer system prompt you can drop directly into your coding AI.
376: It is written to be unambiguous, legally defensible, and implementation-focused, while keeping humans in control.
377: 
378: ⸻
379: 
380: DEVELOPER SYSTEM PROMPT
381: 
382: Role: Facial Image Analysis Engineering AI (NGO Context)
383: 
384: ROLE & OPERATING MODE
385: 
386: You are a software engineering and system-design AI assisting an NGO in building a facial image analysis application.
387: 
388: You:
389: 	•	Design architectures
390: 	•	Write and review code
391: 	•	Propose algorithms and data flows
392: 	•	Document limitations and assumptions
393: 
394: You do not:
395: 	•	Perform autonomous surveillance
396: 	•	Collect data on your own
397: 	•	Identify real persons with certainty
398: 	•	Make final decisions without human review
399: 
400: ⸻
401: 
402: CORE OBJECTIVE
403: 
404: Design and implement a system that:
405: 	1.	Detects human faces in provided images
406: 	2.	Extracts facial features into non-reversible numerical embeddings
407: 	3.	Compares embeddings against a lawfully provided reference image set
408: 	4.	Produces ranked similarity results, never identity claims
409: 	5.	Supports human-in-the-loop review at every stage
410: 
411: The system must support investigation and documentation, not automated identification.
412: 
413: ⸻
414: 
415: ASSUMPTIONS (ALWAYS TRUE)
416: 	•	All images are provided by humans with a lawful basis for use
417: 	•	Reference images are curated, reviewed, and ingested manually
418: 	•	The application assists analysts; it does not replace judgment
419: 	•	False positives are more harmful than false negatives
420: 
421: If a request violates these assumptions, you must refuse and propose a compliant alternative.
422: 
423: ⸻
424: 
425: FUNCTIONAL REQUIREMENTS
426: 
427: 1. FACE DETECTION
428: 	•	Detect zero, one, or multiple faces per image
429: 	•	Handle:
430: 	•	Variable lighting
431: 	•	Pose changes
432: 	•	Partial occlusion
433: 	•	Low resolution
434: 	•	Output:
435: 	•	Bounding boxes
436: 	•	Detection confidence scores
437: 
438: ⸻
439: 
440: 2. FEATURE EXTRACTION
441: 	•	Convert detected faces into embeddings
442: 	•	Embeddings must:
443: 	•	Be non-reversible
444: 	•	Be consistent across images
445: 	•	Support distance-based comparison
446: 	•	Avoid storing raw face images unless strictly required
447: 
448: ⸻
449: 
450: 3. SIMILARITY COMPARISON
451: 	•	Compare embeddings using configurable distance metrics
452: 	•	Output:
453: 	•	Ranked similarity scores
454: 	•	Threshold-based confidence bands
455: 	•	Never output binary match / no-match decisions
456: 
457: ⸻
458: 
459: 4. REFERENCE IMAGE HANDLING
460: 	•	Assume all reference images are:
461: 	•	Lawfully obtained
462: 	•	Explicitly approved for use
463: 	•	Manually added to the system
464: 	•	The system must not autonomously crawl, scrape, or discover images
465: 	•	Internet-sourced images, if used, are assumed to be human-collected and reviewed before ingestion
466: 
467: ⸻
468: 
469: 5. HUMAN-IN-THE-LOOP DESIGN
470: 
471: Every result must be reviewable by a human, including:
472: 	•	Side-by-side image comparisons
473: 	•	Similarity scores with explanation
474: 	•	Clear uncertainty indicators
475: 
476: No automated enforcement, reporting, or irreversible action.
477: 
478: ⸻
479: 
480: TECHNICAL DESIGN RULES
481: 
482: When proposing or implementing solutions, you must:
483: 	•	Clearly separate:
484: 	•	Face detection
485: 	•	Embedding generation
486: 	•	Similarity comparison
487: 	•	Analyst review interface
488: 	•	Prefer simple, inspectable architectures
489: 	•	Avoid hidden global state
490: 	•	Document:
491: 	•	Known accuracy limits
492: 	•	Bias risks
493: 	•	Failure modes
494: 
495: ⸻
496: 
497: FAILURE & UNCERTAINTY HANDLING
498: 
499: Explicitly handle:
500: 	•	Low-quality images
501: 	•	Multiple close matches
502: 	•	Conflicting similarity scores
503: 	•	Out-of-distribution inputs
504: 
505: When confidence is insufficient, output:
506: 
507: “Insufficient confidence for meaningful comparison.”
508: 
509: Never force a result.
510: 
511: ⸻
512: 
513: OUTPUT REQUIREMENTS
514: 
515: For every technical response, include:
516: 	1.	Clear explanation of the approach
517: 	2.	Assumptions made
518: 	3.	Limitations
519: 	4.	Trade-offs
520: 	5.	Points requiring human judgment
521: 
522: Do not present probabilistic results as facts.
523: 
524: ⸻
525: 
526: 
527: SUCCESS CRITERIA
528: 
529: A response is correct only if it is:
530: 	•	Technically sound
531: 	•	Ethically constrained
532: 	•	Human-reviewable
533: 	•	Minimally harmful
534: 	•	Defensible under NGO accountability standards
535: 
536: ⸻
537: 
538: 

(End of file - total 538 lines)
</content>
```

---

## Assistant (Build · minimax-m2.5-free · 12.2s)

_Thinking:_

Now I've read all the .md files in the project. Let me summarize what I've learned.

Done. I've read all the .md files. Here's the summary:

**Project**: MANTAX - NGO Facial Image Analysis System (v0.5.2)

**Key Documentation Files**:
- `master_promt.md` / `system_promt.md` - Project context and workflow rules
- `CONTEXT.md` - Critical edit rules (mandatory pre-commit checks, HTML-JS cross-check)
- `ARCHITECTURE.md` - Complete system design with 14+ API endpoints
- `README.md` - Features, quick start, testing
- `rule_set.md` - Ethical AI coding rules for facial recognition

**Critical Rules to Follow**:
1. After EVERY code change: run `python test_e2e_pipeline.py`, `python test_edge_cases.py`, `python test_frontend_integration.py`
2. Run syntax checks: `python -m py_compile <file>`, `node --check electron-ui/renderer/app.js`
3. HTML-JS cross-check before commits
4. Use project venv: `source venv/bin/activate`
5. Use `./start.sh` to restart (clears cache)

**Stack**:
- Backend: Flask (port 3000) with 14+ endpoints
- Frontend: Electron + Vanilla JS + SCSS (macOS Tahoe Liquid Glass design)
- ML: ArcFace (512-dim, ONNX) + FaceNet (128-dim, PyTorch)

What would you like me to help with?

---

## User

you need to make a very small and easy change do not break anything. the step switcher in the title bar is not right i have an example code for you. adapt it to the step swichterconst switcher = document.querySelector(".switcher");
const trackPrevious = (el) => {
  const radios = el.querySelectorAll('input[type="radio"]');
  let previousValue = null;
  // init first select
  const initiallyChecked = el.querySelector('input[type="radio"]:checked');
  if (initiallyChecked) {
    previousValue = initiallyChecked.getAttribute("c-option");
    el.setAttribute("c-previous", previousValue);
  }
  radios.forEach((radio) => {
    radio.addEventListener("change", () => {
      if (radio.checked) {
        el.setAttribute("c-previous", previousValue ?? "");
        previousValue = radio.getAttribute("c-option");
      }
    });
  });
};
trackPrevious(switcher); <fieldset class="switcher">
  <legend class="switcher__legend">Choose theme</legend>
  <label class="switcher__option">
    <input class="switcher__input" type="radio" name="theme" value="light" c-option="1" checked />
    <svg class="switcher__icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 36 36">
      <path fill="var(--c)" fill-rule="evenodd" d="M18 12a6 6 0 1 1 0 12 6 6 0 0 1 0-12Zm0 2a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z" clip-rule="evenodd" />
      <path fill="var(--c)" d="M17 6.038a1 1 0 1 1 2 0v3a1 1 0 0 1-2 0v-3ZM24.244 7.742a1 1 0 1 1 1.618 1.176L24.1 11.345a1 1 0 1 1-1.618-1.176l1.763-2.427ZM29.104 13.379a1 1 0 0 1 .618 1.902l-2.854.927a1 1 0 1 1-.618-1.902l2.854-.927ZM29.722 20.795a1 1 0 0 1-.619 1.902l-2.853-.927a1 1 0 1 1 .618-1.902l2.854.927ZM25.862 27.159a1 1 0 0 1-1.618 1.175l-1.763-2.427a1 1 0 1 1 1.618-1.175l1.763 2.427ZM19 30.038a1 1 0 0 1-2 0v-3a1 1 0 1 1 2 0v3ZM11.755 28.334a1 1 0 0 1-1.618-1.175l1.764-2.427a1 1 0 1 1 1.618 1.175l-1.764 2.427ZM6.896 22.697a1 1 0 1 1-.618-1.902l2.853-.927a1 1 0 1 1 .618 1.902l-2.853.927ZM6.278 15.28a1 1 0 1 1 .618-1.901l2.853.927a1 1 0 1 1-.618 1.902l-2.853-.927ZM10.137 8.918a1 1 0 0 1 1.618-1.176l1.764 2.427a1 1 0 0 1-1.618 1.176l-1.764-2.427Z" />
    </svg>
  </label>
  <label class="switcher__option">
    <input class="switcher__input" type="radio" name="theme" value="dark" c-option="2" />
    <svg class="switcher__icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 36 36">
      <path fill="var(--c)" d="M12.5 8.473a10.968 10.968 0 0 1 8.785-.97 7.435 7.435 0 0 0-3.737 4.672l-.09.373A7.454 7.454 0 0 0 28.732 20.4a10.97 10.97 0 0 1-5.232 7.125l-.497.27c-5.014 2.566-11.175.916-14.234-3.813l-.295-.483C5.53 18.403 7.13 11.93 12.017 8.77l.483-.297Zm4.234.616a8.946 8.946 0 0 0-2.805.883l-.429.234A9 9 0 0 0 10.206 22.5l.241.395A9 9 0 0 0 22.5 25.794l.416-.255a8.94 8.94 0 0 0 2.167-1.99 9.433 9.433 0 0 1-2.782-.313c-5.043-1.352-8.036-6.535-6.686-11.578l.147-.491c.242-.745.573-1.44.972-2.078Z" />
    </svg>
  </label>
  <label class="switcher__option">
    <input class="switcher__input" type="radio" name="theme" value="dim" c-option="3" />
    <svg class="switcher__icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 36 36">
      <path fill="var(--c)" d="M5 21a1 1 0 0 1 1-1h24a1 1 0 1 1 0 2H6a1 1 0 0 1-1-1ZM12 25a1 1 0 0 1 1-1h10a1 1 0 1 1 0 2H13a1 1 0 0 1-1-1ZM15 29a1 1 0 0 1 1-1h4a1 1 0 1 1 0 2h-4a1 1 0 0 1-1-1ZM18 13a6 6 0 0 1 5.915 7h-2.041A4.005 4.005 0 0 0 18 15a4 4 0 0 0-3.874 5h-2.041A6 6 0 0 1 18 13ZM17 7.038a1 1 0 1 1 2 0v3a1 1 0 0 1-2 0v-3ZM24.244 8.742a1 1 0 1 1 1.618 1.176L24.1 12.345a1 1 0 1 1-1.618-1.176l1.763-2.427ZM29.104 14.379a1 1 0 0 1 .618 1.902l-2.854.927a1 1 0 1 1-.618-1.902l2.854-.927ZM6.278 16.28a1 1 0 1 1 .618-1.901l2.853.927a1 1 0 1 1-.618 1.902l-2.853-.927ZM10.137 9.918a1 1 0 0 1 1.618-1.176l1.764 2.427a1 1 0 0 1-1.618 1.176l-1.764-2.427Z" />
    </svg>
  </label>
  <!--   <div class="switcher__toggle"></div> -->
  <div class="switcher__filter">
    <svg>
      <filter id="switcher" primitiveUnits="objectBoundingBox">
        <feImage result="map" width="100%" height="100%" x="0" y="0" href="data:image/webp;base64,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" />
        <feGaussianBlur in="SourceGraphic" stdDeviation="0.04" result="blur" />
        <feDisplacementMap id="disp" in="blur" in2="map" scale="0.5" xChannelSelector="R" yChannelSelector="G">
        </feDisplacementMap>
      </filter>
      <filter id="toggler" primitiveUnits="objectBoundingBox">
        <feImage result="map" width="100%" height="100%" x="0" y="0" href="data:image/webp;base64,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" />
        <feGaussianBlur in="SourceGraphic" stdDeviation="0.01" result="blur" />
        <feDisplacementMap id="disp" in="blur" in2="map" scale="0.5" xChannelSelector="R" yChannelSelector="G">
        </feDisplacementMap>
      </filter>
    </svg>
  </div>
</fieldset>
<article class="article">
  <h1>Liquid glass</h1>
  <p>
    Lorem ipsum dolor sit amet consectetur adipisicing elit. Qui maxime optio quam debitis autem, maiores odio tenetur dicta aperiam aliquam, iusto nisi ipsum tempore dolore doloremque facere non culpa sint sequi ducimus corporis veritatis cumque corrupti sed. Ipsa dolor quod alias dicta dolores. Ducimus pariatur nostrum quo, impedit <a href="">facilis voluptatibus</a>! Non doloremque, facere neque dolorem animi earum odio placeat quae voluptatem nisi nihil deleniti voluptatibus harum magnam adipisci tenetur. Ab aliquid reprehenderit accusamus id quae velit quasi nostrum possimus ipsa vel. Voluptate nisi impedit placeat nemo neque asperiores iure saepe, eius eligendi odit quisquam delectus veniam autem, sequi laborum atque
  </p>
  <figure>
    <img src="https://images.unsplash.com/photo-1706720094773-d91e070e4b90?q=80&w=2515&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Photo by Neeqolah Creative Works on Unsplash">
    <figcaption>Photo by Neeqolah Creative Works on <a href="https://unsplash.com/photos/a-close-up-of-a-green-object-on-a-yellow-background-A2kyFEwh3zo">Unsplash</a></figcaption>
  </figure>
  <p>
    Sit amet consectetur adipisicing elit. Quibusdam illum in voluptates omnis reprehenderit inventore perferendis dolores, architecto doloribus. Quam error qui nam quis! Dolorum, dolore saepe ipsam quae aliquam tenetur dolores dolor repellendus facere a quasi soluta voluptate provident earum cum. Nostrum consequuntur corporis quibusdam tempora amet, animi inventore dicta voluptas nisi placeat ut illum explicabo! Consequatur, accusamus. Labore quo perspiciatis quos est molestiae illo possimus deserunt sed amet maxime, dolorum praesentium nihil doloribus similique cumque eveniet qui vero eius magni reprehenderit aspernatur, et, quod voluptatum? Voluptatem, eveniet vel sunt nesciunt recusandae alias maiores optio labore, dolorum aut possimus iusto quia. Non quas excepturi ea, maxime mollitia fugit, animi alias ad repudiandae dolores facere molestiae id, nemo temporibus explicabo.
  </p>
  <blockquote>
    Et aliquam libero deserunt maxime! Perspiciatis neque deserunt sequi deleniti!
  </blockquote>
  <p>
    Recusandae doloribus, ullam inventore esse culpa cupiditate dignissimos qui ducimus possimus ipsum reprehenderit, suscipit debitis nihil sit. Animi eligendi sed molestiae. Repellat, est ut eos voluptates tempora quisquam corporis mollitia, excepturi commodi cum dolore asperiores eaque debitis fuga quidem! A laborum ab reiciendis saepe rerum iste. Nisi quos earum ex nulla neque <a href="">tempora explicabo reiciendis</a> vitae sapiente accusamus tempore consequatur nemo, placeat magni quasi corrupti nobis alias, rerum ipsam fuga quisquam tenetur repellendus! Sit laborum aut odit pariatur quos cupiditate iure maiores. Quis nemo dicta numquam autem debitis blanditiis, odio explicabo maxime officiis magni soluta dolor suscipit, quo nisi eaque deleniti dolorem aperiam deserunt. Animi dolor expedita consectetur deserunt, quae id ab.
  </p>
  <p>
    Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi aperiam enim eos repellendus molestiae, ad ab maxime et, aut nam culpa eveniet. Eaque consequuntur laudantium ex dolore voluptas rem cumque, quos voluptatibus aut maxime. Animi enim odio ea porro molestiae laudantium ipsam nulla quia distinctio quod incidunt pariatur accusamus possimus laborum veritatis maxime, asperiores quasi!
  </p>
  <h2>
    Doloremque nisi eius quis
  </h2>
  <p>Magnam quo voluptate vitae voluptatem expedita vel illum ut. Tempore, sed? Sunt distinctio minus dolore, consequuntur eos qui eveniet error rerum tempora, autem et quaerat, ea repellendus unde iure. Fuga ad tempore cupiditate animi iste, eius nam beatae, aliquid quae id iusto perspiciatis. Quos quis cum itaque facilis libero sequi quaerat ipsam hic blanditiis doloribus modi sapiente tempora ipsum fugiat similique cupiditate beatae provident eos natus doloremque alias, sed praesentium.
  </p>
  <figure>
    <img src="https://images.unsplash.com/photo-1734606901283-489b25f7aa9b?q=80&w=2360&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Photo by Neeqolah Creative Works on Unsplash">
    <figcaption>Photo by Irene Demetri on <a href="https://unsplash.com/photos/a-close-up-of-a-green-object-on-a-yellow-background-A2kyFEwh3zo">Unsplash</a></figcaption>
  </figure>
  <p>
    Quod iste recusandae sed labore corporis ea provident debitis hic maxime placeat alias rem cumque animi explicabo laboriosam, dicta molestias? Corporis quibusdam, aliquam asperiores quo officia reiciendis nemo aspernatur similique voluptatibus in tempora? Laborum temporibus ipsa at exercitationem ullam labore tempore neque. Porro earum distinctio rem perferendis amet nihil maiores ab id cumque quae nesciunt, velit fugiat in aspernatur unde. Ut dolor eaque, vero quo sunt sed exercitationem cum accusamus, beatae, quasi rerum. Id eius dolorem nihil porro, voluptatem exercitationem tenetur impedit laudantium odit aliquam vel, facilis eos quasi voluptatibus optio. Ipsum voluptatum labore consequatur dolorum veritatis qui placeat magnam aut expedita. Error facere tempora ratione sapiente illo laudantium, ut nihil. Mollitia vitae, eveniet blanditiis corrupti at molestias aliquam ipsa esse optio temporibus dolorem, fugiat obcaecati culpa fuga numquam sapiente libero. Impedit maiores doloremque iusto autem temporibus! Quam cupiditate officiis possimus, ratione quasi nobis rem quidem quibusdam? Ut earum possimus expedita culpa nobis suscipit, non minima at dignissimos aut optio eum? Cumque?
  </p>
  <p>
    Lorem ipsum dolor sit amet consectetur adipisicing elit. Excepturi, aut commodi aliquid, blanditiis doloribus suscipit perferendis hic assumenda dolorum magnam enim. Error minima magnam modi aliquam quia at obcaecati sunt quas velit numquam, voluptatem inventore blanditiis necessitatibus harum possimus nostrum vel expedita quam dignissimos tenetur maxime provident aspernatur nam iste. At veritatis nemo quisquam, facilis numquam nostrum veniam perferendis repudiandae cum ipsa eos eius est doloremque excepturi delectus quidem repellat suscipit et tenetur vitae porro nisi iure? Ipsa facere at blanditiis nobis temporibus labore architecto nulla deleniti quod hic minus consequuntur, unde, quasi eius consequatur est! Laborum neque illo ab.
  </p>
  <blockquote>
    Consequatur itaque sit amet consectetur adipisicing elit.
  </blockquote>
  <p>
    Ab eveniet recusandae incidunt id cumque porro? Dignissimos nostrum iure possimus mollitia sed provident esse optio odio consequatur! Recusandae dicta ab atque? Eveniet molestias autem accusantium vero, tenetur placeat ratione voluptates nesciunt, adipisci ullam cum quae eius eaque quibusdam nihil dolorem! Ab eveniet recusandae incidunt id cumque porro? Sit tenetur quasi odit, facere dolore quae voluptatibus fugiat, quibusdam vitae labore dignissimos. Unde, maiores cumque dolore quas repellat vel adipisci harum incidunt consequatur ad!
  </p>
  <p>
    Ipsum explicabo quod laborum non provident, consectetur sint unde hic dolor quisquam ullam nulla veniam! Sapiente impedit sint expedita officiis fugit est quia. Neque, ea aspernatur? Quod voluptas ab aliquid saepe id, eos suscipit, autem quae reiciendis omnis veritatis officiis culpa doloremque? Reiciendis ea modi quod ducimus culpa omnis aliquid non minus ipsam necessitatibus distinctio, aspernatur iure. Nihil cupiditate amet officiis eligendi nostrum harum labore porro optio obcaecati reiciendis blanditiis quae, perspiciatis voluptatibus suscipit enim.
  </p>
  <p class="box">
    Perspiciatis sapiente eum velit inventore illum accusamus eos at esse mollitia debitis quae rem odit, ipsam nam. Voluptas beatae, velit voluptatum dolor obcaecati a nobis consequuntur quis id eaque! Sapiente nostrum rerum esse quo laborum excepturi explicabo, perspiciatis dicta corporis atque?
  </p>
  <h2>Quod voluptas aliquid id saepe</h2>
  <p>
    Et minima quas amet! Debitis commodi consectetur laborum fugit voluptatum qui distinctio, natus odit obcaecati. Voluptate suscipit consectetur, aspernatur ratione impedit minus facilis voluptatum tempora nesciunt pariatur ipsa provident qui distinctio ad quasi magnam exercitationem itaque. Recusandae maiores dolor saepe commodi a eligendi aut adipisci neque dolorem tempora perspiciatis odio asperiores quos iusto animi quia tempore, molestiae deleniti vel. Earum, atque? Ipsa reiciendis cum laborum fugiat doloremque asperiores, voluptatem exercitationem in sequi!
  </p>
  <p>
    Dolor sit, amet consectetur adipisicing elit. Earum, amet ea quaerat, repellendus ipsum sunt blanditiis ab tempora doloribus quas voluptas magni ut perferendis unde dolor tenetur repudiandae, porro dolore et nisi eius. Eos, esse minus molestiae maiores praesentium dolorum dolor eaque repellendus laudantium provident adipisci alias inventore ratione distinctio libero facilis voluptas hic illum odio laboriosam quos minima officia natus. Maxime eveniet repellendus non iste exercitationem aliquam velit, corrupti ut, id provident reprehenderit accusamus nemo reiciendis fugiat saepe quas quasi ad tempora! Qui earum eligendi architecto delectus animi similique laborum dolorum, recusandae beatae voluptatibus! Maiores ad libero quibusdam temporibus, quis possimus perferendis corporis modi, culpa voluptatum maxime. Earum, rem.
  </p>
  <p>
    Eos dolore dicta delectus dolorum porro fuga modi, perferendis natus ratione repellendus sit harum sint! Numquam quas facere quod doloribus enim omnis earum maxime perferendis ipsam magnam quos culpa, vitae, tenetur quisquam, unde quis maiores. Eos quos earum illo animi exercitationem nemo, at asperiores soluta similique aliquid cum quam quo corrupti accusantium iure praesentium. Omnis aliquam harum reiciendis necessitatibus temporibus expedita voluptate, dignissimos ratione dolorem perferendis ut aliquid magni nam nisi deserunt eum nulla excepturi quos sunt accusamus earum cupiditate minus. Expedita, corporis suscipit mollitia qui repellendus harum cumque id quis architecto quidem, aliquid vero dolores necessitatibus animi eos optio? Perspiciatis!
  </p>
  <p>
    Maxime in officiis fugiat corporis cum doloremque atque, at architecto dicta quod inventore corrupti facilis repellendus pariatur possimus error! At aperiam aliquam fugiat eveniet sapiente soluta hic ab ipsam, corrupti temporibus tempore? Velit, sed quos mollitia deleniti fugiat tempora fuga reprehenderit neque tempore doloremque ducimus quasi? Sit facere porro esse vero accusantium praesentium odit. Quisquam sint nobis dolorum labore distinctio ducimus animi alias quod necessitatibus rem non error consectetur voluptate tempora libero facilis ratione voluptates expedita, unde a blanditiis odio. Nostrum sapiente dolor enim?
  </p>
  Made with love by <a href="//codepen.io/DedaloD">Den Dionigi | UX Designer</a>
</article> <fieldset class="switcher">
  <legend class="switcher__legend">Choose theme</legend>
  <label class="switcher__option">
    <input class="switcher__input" type="radio" name="theme" value="light" c-option="1" checked />
    <svg class="switcher__icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 36 36">
      <path fill="var(--c)" fill-rule="evenodd" d="M18 12a6 6 0 1 1 0 12 6 6 0 0 1 0-12Zm0 2a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z" clip-rule="evenodd" />
      <path fill="var(--c)" d="M17 6.038a1 1 0 1 1 2 0v3a1 1 0 0 1-2 0v-3ZM24.244 7.742a1 1 0 1 1 1.618 1.176L24.1 11.345a1 1 0 1 1-1.618-1.176l1.763-2.427ZM29.104 13.379a1 1 0 0 1 .618 1.902l-2.854.927a1 1 0 1 1-.618-1.902l2.854-.927ZM29.722 20.795a1 1 0 0 1-.619 1.902l-2.853-.927a1 1 0 1 1 .618-1.902l2.854.927ZM25.862 27.159a1 1 0 0 1-1.618 1.175l-1.763-2.427a1 1 0 1 1 1.618-1.175l1.763 2.427ZM19 30.038a1 1 0 0 1-2 0v-3a1 1 0 1 1 2 0v3ZM11.755 28.334a1 1 0 0 1-1.618-1.175l1.764-2.427a1 1 0 1 1 1.618 1.175l-1.764 2.427ZM6.896 22.697a1 1 0 1 1-.618-1.902l2.853-.927a1 1 0 1 1 .618 1.902l-2.853.927ZM6.278 15.28a1 1 0 1 1 .618-1.901l2.853.927a1 1 0 1 1-.618 1.902l-2.853-.927ZM10.137 8.918a1 1 0 0 1 1.618-1.176l1.764 2.427a1 1 0 0 1-1.618 1.176l-1.764-2.427Z" />
    </svg>
  </label>
  <label class="switcher__option">
    <input class="switcher__input" type="radio" name="theme" value="dark" c-option="2" />
    <svg class="switcher__icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 36 36">
      <path fill="var(--c)" d="M12.5 8.473a10.968 10.968 0 0 1 8.785-.97 7.435 7.435 0 0 0-3.737 4.672l-.09.373A7.454 7.454 0 0 0 28.732 20.4a10.97 10.97 0 0 1-5.232 7.125l-.497.27c-5.014 2.566-11.175.916-14.234-3.813l-.295-.483C5.53 18.403 7.13 11.93 12.017 8.77l.483-.297Zm4.234.616a8.946 8.946 0 0 0-2.805.883l-.429.234A9 9 0 0 0 10.206 22.5l.241.395A9 9 0 0 0 22.5 25.794l.416-.255a8.94 8.94 0 0 0 2.167-1.99 9.433 9.433 0 0 1-2.782-.313c-5.043-1.352-8.036-6.535-6.686-11.578l.147-.491c.242-.745.573-1.44.972-2.078Z" />
    </svg>
  </label>
  <label class="switcher__option">
    <input class="switcher__input" type="radio" name="theme" value="dim" c-option="3" />
    <svg class="switcher__icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 36 36">
      <path fill="var(--c)" d="M5 21a1 1 0 0 1 1-1h24a1 1 0 1 1 0 2H6a1 1 0 0 1-1-1ZM12 25a1 1 0 0 1 1-1h10a1 1 0 1 1 0 2H13a1 1 0 0 1-1-1ZM15 29a1 1 0 0 1 1-1h4a1 1 0 1 1 0 2h-4a1 1 0 0 1-1-1ZM18 13a6 6 0 0 1 5.915 7h-2.041A4.005 4.005 0 0 0 18 15a4 4 0 0 0-3.874 5h-2.041A6 6 0 0 1 18 13ZM17 7.038a1 1 0 1 1 2 0v3a1 1 0 0 1-2 0v-3ZM24.244 8.742a1 1 0 1 1 1.618 1.176L24.1 12.345a1 1 0 1 1-1.618-1.176l1.763-2.427ZM29.104 14.379a1 1 0 0 1 .618 1.902l-2.854.927a1 1 0 1 1-.618-1.902l2.854-.927ZM6.278 16.28a1 1 0 1 1 .618-1.901l2.853.927a1 1 0 1 1-.618 1.902l-2.853-.927ZM10.137 9.918a1 1 0 0 1 1.618-1.176l1.764 2.427a1 1 0 0 1-1.618 1.176l-1.764-2.427Z" />
    </svg>
  </label>
  <!--   <div class="switcher__toggle"></div> -->
  <div class="switcher__filter">
    <svg>
      <filter id="switcher" primitiveUnits="objectBoundingBox">
        <feImage result="map" width="100%" height="100%" x="0" y="0" href="data:image/webp;base64,UklGRq4vAABXRUJQVlA4WAoAAAAQAAAA5wEAhwAAQUxQSOYWAAABHAVpGzCrf9t7EiJCYdIGTDpvURGm9n7K+YS32rZ1W8q0LSSEBCQgAQlIwEGGA3CQOAAHSEDCJSEk4KDvUmL31vrYkSX3ufgXEb4gSbKt2LatxlqIgNBBzbM3ikHVkvUvq7btKpaOBCQgIRIiAQeNg46DwgE4oB1QDuKgS0IcXBykXieHkwdjX/4iAhZtK3ErSBYGEelp+4aM/5/+z14+//jLlz/++s/Xr4//kl9C8Ns8DaajU+lPX/74+viv/eWxOXsO+eHL3/88/ut/2b0zref99evjX8NLmNt1fP7178e/jJcw9k3G//XP49/Iy2qaa7328Xkk9ZnWx0VUj3bcyCY4Pi7C6reeEagEohnRCbQQwFmUp9ggYQj8MChjTSI0Ck7G/bh6P5ykNU9yP+10G8I2UAwXeQ96DQwNjqyPu/c4tK+5CtGOK0oM7AH5f767lHpotXVYYI66B+HjMhHj43C5wok3YDH4/vZFZRkB7rNnEfC39WS2Q3K78y525wFNTPf5f+/fN9YI1YyDvjuzV5rQtsfn1Ez1ka3PkeGxOZ6IODxDJqCLpF7vdb9Z3s/ufLr6jf/55zbW3LodwwVVg7Lmao+p3eGcqDFDGuuKnlBZAPSbnkYtTX+mZl2y57Gq85F3tDv7m7/yzpjXHoVA3YUObsHz80W3IUK1E8yRqggxTMzD4If2230ys7RDxWrLu9o9GdSWNwNRC2yMIg+HkTVT3BOZER49XLBMdljemLFMjw8VwZ8OdBti4lWdt7c7dzaSc5yILtztsTMT1GFGn/tysM23nF3xbOsnh/eQGKkxhWGEalljCvWZ+LDE+9t97uqEfb08rdYwZGhheLzG2SJzKS77OIAVgPDjf9jHt6c+0mjinS/v13iz9RV3vsPdmbNG1E+nD6s83jBrBEnlBiTojuJogGJNtzxtsIoD2CFuXYipzhGWHhWqCBSqd7l7GMrnuHzH6910FO+XYwgcDxoFRJNk2GUcpQ6I/GhLmqisuBS6uSFpfAz3Yb9Yatyed7r781ZYfr3+3FfXs1MykSbVcg4GiOKX19SZ9xFRwhG+UZGiROjsXhePVu12fCZTJ3CJ4Z3uXnyxz28RutHa5yCKG6jgfTBPuA9jHL7YdlAa2trNEr7BLANd3qNYcWZqnkvlDe8+F5Q/9k8jCFk17ObrIf0O/5U/iDnqcqA70mURr8FUN5pmQEzDcxuWvOPd1+KrbO4fd0vXK5OTtYEy5C2TA5L4ok6Y31WHR9ZR9lQr6IjwruSd775W6NVa2zz1fir2k1GWnT573Eu3mfMjIikYZkM4MDCnTWbmLrpK/Hs0KD5C8rZ3n0tnw0j76WuU8P1YBIjsvcESbnOQMY+gGC/sd/gG+hKKtDijJHhrcSj/GHa/FZ8oGLXeLx1IW+cgU8pqD0PzMzU3oG5lQ/ZaDPDMYq+aAPSEmHN+JiVIp0haHTvPt77732z5ed2K7NHs9FtCIk4BdNkKLRLvOKlFcw+UiovM4OB5sGgepyML+a4TEu/I29/dFtjJulojJR4Tg71ybApEdca0TSnaumNJyCWH2pjENASlQS/NIXMWtiPV9CHsvuftev08/lemYIcUnHSu6XEMvaBq41tqf/m0siLj7xeXsnBmhxY5z+nCwX4Iu4euTPaE4EQorgogisHrBtsAMdX+Huje7nlx3hMpKovdf+YftDQqytChXfEh7D5nyC8rzNTICINmpK5Ni0ngcAMzpmiYDwOMtmUTiCjvx2S2dIeSguP/QHZ3xYIeGhTt1CsCOIiEuVw8pGjVznDJppuojl30i9RvXccXzmXGj2b3H3XM38c/PZseyeOdplXhFekzZMZ2fUGuIBsKCcgQg4Ikqt4PDTkQiWQtMUBFAEhUH8vuvoAvnvGMCEP4/vMmZA2PnkmAJsQsHeFAIk43F00OS3sa/1TDJTPss2698T+i3V22L3PsIeFAHmWWi1FUh29TqpniVOt5hGA/q40Yubt4yXDEQomvldUNhfuuSvjHzPBysYhBMSmRrpuIUHJhQk5uw5V4EwpMp1NvklGkc03WYeC0KETcZ409HkEcwnEaE3EdNnIcfCb1jjWNfZyhhGH48AvsJ4WL+mYTM5i+yFNyM6PhbkuMGYREv48VihVyHXb9RjoE0HvoOuaO7fxxUYnQj1wB0DOZUagcEXfVkJ/nBgV+vl5yMfFaJs0myb9BjyNSsY9FbwZNq21wEFOEJ8Pk/vO1fSa6bOPZFCMc7grz9YXf8rBBPaK3qUJEfJG1A8nuytO1jg8CvWGEY1Z4o1gb3uEjILmNm5YfMXH3GtvyETX+j4jAXkkaA7FDQIdPzLZOcUJsqLQFxboX/MZ95f7MqPku/6IAGXer6xchZyiqcG2Tw4oSVcO0Q0vqOlmEcpsyBw2pwzcifb6t2th64vASkXGXzY9U7aFvkqJEOWSkEU0oL0FrnOfr432tJ5OtPUG1T0cg5yqNTNFAqKFxl80fxGGPFzIiASv+sEPaGMmewBjUEZNFtVCwzaG3PVSe5l+AIRNeFCzu2+H/7Cp2pbOjRUjNFFMX8ZEGl0D4uNWi4ykocIgBkGF+HAIHRNjAqioi4y7vjPtlTPTMXwl7aQD7gu9yVk+VdBwmVMnljIx4++8hq0qOtmjkwT1+RW4N0LhPQuahKrjGVIMy2hW3lgO8lqoLLBHAaTvRIgaPLNFx5ChJ8hTcsBdO383ouHspeqwelcvfEOELFMF0a+jWZJzZYWqZQlj9FnUeMq37zGWfbwRbvkDKOR0OKzAUNO5y8O+H24nczTdDZniPDwMUgIJDV1sEJn7xWMscorAcT3niXE+kcQS0NUMjkkoiNu43cbvQGGagTd6ycWgkkPbSb0Fi0iiYKTpXlKyTCKKHsWssGuM4dhzIaZqIjXvg2w1xqK8sqkQKhJUqWoGxcXTK4gi12ecTaa8+jmMYItoS41KhA4pbAWS2MyLk3n/lS0c4Cq4KcdLYTv4c3OPQZWJx+B9dSytYPUmGUKbKpg+Oy/g0iGuMDw+WRMjdCftaM30PxVSEW8Y6IeUpcGDoTFyDExFIC0coBCNDjx8XXBMWW53qAz2LgJA7G/zPcBcq5mjyfMo/dYTJMBQ3mkxItV2HHpsltIs49LLZK4w6TscoK/1x8FCEkPvP90Y3XVDu468z/HBkAdUMZLNwt3AqNiHOLQM/EYqMbxAWcgW1Rd5PFOnuX08+iNwt7wFWBWYdpDb3F5inFIe4vlXFLkUO3zVjzvJJWXGJOhyBSxV4O8z1FPBmVgZA7p+Ov5oh0XYD5DazDBODdJHHK3O3U5k2REDOWh7ZQSw6fDLBl4P4hixhuzJpGLmv9Ok/12dnFEMDomZm9pikmMevpvEAvZSq1rPziRSaXHMokc0TwRInpAVh5B7os8LBX4+z8rYaZxxQViQ7bndIOnucpgFahg7nBRTv9mUP1epZ+zzFYkXJvfvxUmkdewGhR3FtEE5gGUdAz8DbBFDQypm3jgUlFMru4RG5VIXGaThK7uZnNNDVq3igkGgQVnnSqodKgLGNEPnkAH3YgM0ABowQ5RsDpa4C8wuMrXP8JeioiBC5//ltLZOuePmXgZauU9FcpsvPvYH5yWt8P65HuRjLI62+zmNH28fZZ4odgbjp6AswlNzd74PbIkojkpXSKKF8h79BOJxhZFhDeSWAvb3D5jw2NtUDppI4eRSg5L7+5bTUdm0e7FZh2BgmZdVY/+WE7DLuqWZm3YvOEoQ0WcIIlI8bckcO2SkgZcHI/f63KJb0uWUR6gtorxgCE5ytH3wRr3kiWHlcdGk/SZO0UU+RYuFrCTjCdUAwGdEouf//Si1AhNmg7ZFRuMR+5qeQAaAdwKrG5O5pUnNAa8Ecb9Y2b6B8Rejwcffv5ii5h69Dhm55nhpJ3o/FYpTL1AWgmLIAG4t3qK8ocYnXxF06Fe0Dtv9kvv/LJZTcg/D4OB1FEtaC+mvh3RNhPLlOg3QniC0jov2Qjw3adeA/2GAIohAxCwSGlTsJ+pkOHU6K0EyY5osnN6tVyv56/OJNAOP9Kvi1wZx55EIcz0F2IYWAkvvDRypWSXUuGExX4QjQt4o5ptXHEaXK4z5RYV1C7cs6aLTigJYW8Lwcrv/R9cHuLsl1cfKzRlB5hgWzp/tpPDUF2sWA4tApdUKqSRX+TTogKnATAH44OLk7d36DCknABBAqTWQQz1QgQeq3EImJiwWdYSahYYXVOJmPCa6LqAvdEojcVT+xjjtNZoCcsYRHnvdK7bf2GreoKKsKDtgn5emh3lGmCdDzkDJPGid3PFAb/Bbwj1MCf2pdZqkSUBwWXgGpLWaUEjFG+0PmcDzclQBH2FDsA+UcILmHrzrHY6DKev0bBOYPD6lGy0Nw60gIAeP8HXWq0vZo5rbFGsYXSDtNb+QnSu7hPyLzvfMcaBTM2oF6rLx2CQaaYSljdEeodTvY2uqwUYvPtFlqNo0wxoWSu/8rQgNHO9WjggPFdxIG3socz0BCkQY1umhJ1oHI/lta72+zuU9tESX3+5++GF3dZeON4RZCnaoHjExonNAkjSXSyOtbbjmATzeZJBoWDR202FweApL78uWpYAitcpVDELbG9a7R9zukHUYYLTBBrysZM7cj0rgs1lgo1EXNwwmS+3P65ZvqICNr2C+AXNaOP04VKUZtyPItDaBCa2hawRB761AYFwgNmPsZRZDcn8OPBuIoKsjgxJOUP9x8f2TEHH5pcKqZXyCi2eduB3r9o1Kg1SSC0/OkCBEld/O5E6gWQmJ1s8jYY4HW5KGgNvD9RZpUY+3vwYBZfyHIM+koswIT86IJ6xCDjzuvo/v0laJA06ySyQbx7adCMiTg4oCWrHkUBFHcAAw8Zs1e1fEhrXkE0UDh/hoYuT/o0/OBjuEg97O4QpJ5B8QMB2u4oo/SPDGuW4Z3fnTbzgoUmpQCeZMIdAzBYuR+p09f9lD88wtshQ9yqJEpJnSslPMpqdjN/n61ba2dIiF+IoGkABIBlxnhcWdVOnY9rvmGIYoJgyI98CQrWXxRfWGzDi3jICiEzX2N3Fgp89vN2GmbsTN0uhJG7la4vt78WCwjaJc8uu+EUg7rMkghSWwuHuP0+4fLvRC0swGQZXSKb5yFmAFyf+7sfhkWMMId2oT4bFT06oNHcBJhNmNZ4dgZrb1ZOFoetT1gjgje0l51XkfExz25Q90Xc0it+06TRIXW1fHOGfK4RQxx2dNtriJ8cyns0pG11RrpikqJIlyA3J8uvXvsBRnhre1fOT2hASX6pqQf5xrRQaPAjJmaCvRIxI85yzm0mnXYKSWHxj0pwsjPavDyPJkuhnWPvoKptc/U9bt8HISJ2y1ag/TVNA6kOmIWEhbSWk0xPEBA4y7en+7Tb3oQPoAj9t+tzyxTpIkdIZ9pEVbOohduiU53ry0Vdw2hDhAgz99R4XF/Llx+Ov+OVrAv3zmzaX2m4cHVUcIP+dEs+U7Yx0qioIrQHrW3QJTXDR2cb3X4uBvxqRw5j5I1q1w2CLsuEwtNSVNQMAZ4l+lziBHy8eAjYEeK3DclFBt3tp1sbmNUO+KqVwSSpcbAdb4ns6h1mxhKtLTEQqgYuMP5RggqzoFXsQYHx/05pvL5HySE1MM6T9QLUUoxv5Rm4OLcKHkl9lvjEAib4QmNwyNqkwjk8uM7LO5cekr1LytEk045FrgejisDNO0G2yPXcEMVzVjdaWEgF5p+JmrETExrlwOEIAkb95UE+WntFZTua82BrGaS6C5uOI6HwKMzADyxqDQTVeqUgUIOyVivuQBABGN8SVzcWbTi+WjiH7EAB35nAKMGup7f4dQVE6QhErT0bSeowYYcX6D4DVExZm3wjn+8cMYf1u78CaZHxkeSIil45UfK3e2eUG8kDbJGM7cVHhlrwU3q84RUQOcXIHaeIjI+ot3Tsgbd44jjvRE0Sksd1EhDvHUEP7nF1H32sz52Ou4/UWAJX9cwEuQF5KSwdFpORCCr5KPanWVWGtGdgg8bevpjyXVDslUNnA/DnQoE2oRFQuKJx2/9es1eAUWd+aB251ZhQl3QkSPbMGRCIbVR05huHlcaC62eRAQ8yoymNW0RTZtFryPwnOa6MH9Iu/N+hZGVgrFO6fcbLFQMgtqHO2MMExdtMOI8penvNgQ1kIf4tBoOgFT0Qe3+7I/l0++DKIjLczbIN4MgrE9g9bqlDsi8G8mke4qmdN3Mr50dzcClH+dbCvsD2v3of3b7ZRzsY/wRMxriY36nlzDfVgswAhnCYDtsSITFClQM1Kw1BvFyTmnCh7J7OkZj+x+cGj7Kji60BplH5QypyMurm06L3JxRmfET0Wv/mVW3PZDnsYbrg9n9aI+6agYZuPj748JQugCkYc+RvXhLjKrSKTAeEiCFdV1FOd3vh1jaUTFO6uPZ3ZNSfvjncFtE0encKTkeU2SWsbhvKL54q0BTvpx8Ti1dAw1jVXKBa56NjOg+jt0Fn851+17mLainZ5viWtCEOleMm9X30Mddnx+59DpVNDZ7JjAlsQHC66PYXeHTJFyTEDDsci4KjA4Gm/ki8gMLEH8cAI19miOaUDWciVwEg9oedUDAYxMuYGDkg9j9e5ZShnz+um4PqZiL1oUkJWXtqlDHJzacvb8wGbkCU/j4Auefwb95hKV5xT+c7Q2St78793VM8mK+z2mks8fKOne2NtQqxRtHTuHsICa4macwO7QASsGcqINdIqT3v3tm0At/A67o6BD2mVbfCoYVAc/XfiLkfHN8rxcO7SdByZqHA6HYXgsUrnS65BP2vndP65L3p5dL4JvF5xtXJnIOMU5DKuStoQ59dsATxnO+RbuizcMTcpgkzqzV3vjuXCbK1992KMc5EaQ7Ko2M49wTsJALU9zDbDFpe/be9XF78rg+Oe4kanJF9J53V665yUcaP84L7vcNeXIJhe4tGIgJWv5jbZSoiER6FyriakY5YRv2d7y7IAuV0T8vu8UYaKk0e0YDJIZmiMqsuvDFQHqGc5+uWA5JAWgdQMxEgsmgUomN/m53l+QfUeGFqWaIFQ8Z0r/Db5DtM6WPYRwvFOKIqbL4QjcoQYF7EAb+drA6XfwI3+Pu6rVGZ1iDEeTq0hU4GHuciUHR1EmRacJiw44+IgA2QerjHCcOfFymK5L9VndX95ZL5g1hteUCIgDBHLwKiBOTJvQJXwTCg64VTcq4koFWfBAr2bA/K84nFQO/zd0PstVbLk/ww2bAWDaGICruS5Qm3DEcBDZyM+2I1hmlALKEAiOA6Tnf9yKl5/3tfiiOSuvPX8+PDV8fTJK7VCZaNqXFT0z547T10hzRrbfkj1XwHDimUYtJnJC3trtCd0vl9Yf5P2OfFR07o5s1Poxa1028bQ179kADrFZAtP9gb6SyIwYRZWxnqICqBkHmbeyuKVfcyVpDP/9+/mH1+HNU7v8q2qebw40v0IIQGEKJGwH8AvcDJTujYPFfR1BukLyb3TX5O6qkv9g7D3WyQHxRpWVIVeTqAXZ06Ik1CG5TYho7ooYOl8j3VEdQmnOwv4vdVWEj1dMf/v5O/6hOboXnGsZRQyDbyxz+Xwe+2Af8OE9IOupywuEhObDNAnhyy2fiFgkvvSuR72B3lfgkrCnn4W6047HzdQMUiyI4mufKTtUzyOEmp+F4SnkqZoeDS61FIyWjwF0GPQ337Hd+d1Rbf/jz8S/jpUDOqoP+/VzeUiM6hCvUaqbhL02rMTXXZLp9U7SamG4MlyN+6qhVNcuFcIQpiW/X4fx+AX5NeNfTKdS67fGL//mxOkun0s4M07L5EH7NH6vw2FY3mnp/CRBWUDggohgAADCGAJ0BKugBiAA+CQKBQIFmAAAQljaJLsWP/evrr7yi95IzsLxfJF/2VI9gDe9A/k2qd8QY6lh2+t9N/1LcuP1fYJiMX2v6T+M3b3zv9d/bfkx+Rn0Ocj+C3kPvH+7P+c/NK5S/Dy9+dr9B/gvyE+hv/b9af55/3fuC/pz/jv7B+7n9s+kHqs84v7oevB6XP8Z6hH9o/ynW0f0z/S+wj+zvrWf+v92fic/s/+2/c34DP2L///sAf//1AOi/9c+ADsaf1P4GnCn+Ht64N1GgnpjzX+f/yvRF9M+wT+q//L7AHoHfqOOffdUrKzVBhoFjf+JrTNIbKavxIA43AGpRqNz94rvyITk0o7pDGdWKgSfGnuMbT2yi7ALm4hyj6CcOnqm+n+fcJzmlIX9LduCbKqsU70TXwY3VVr0DFnyXcrzU/mHGg5O9KxgeBQidY8s/wX6gwOv4tUAPB8UFY38s/ahNxIMAbSmfoMUSx7t22EEj1+nJW7W36fP95EmUdMpkp3MTnc8vK/FrxQyHosWJTsvFYL+aHJU7JPsURW6LHIoqFllL+X5eFH0c1Ou+dkkOAUNUYQdDOTOWSm8ox3d7KJRwfMq2gEoo1LtS6tp+6zT/DKeqNJc2lNngkj0YRY484IxStFHED0Wz85S7YcIGM5ujhLXWdKPSO9Z6fZg2+ACpQeNvZ8/BRPUgOo6nklsaa3T8bJR8sC1Bh4OJ9I7mTlCz9Si1sNw7YB0T5rMvo6pDOR7xBIob/J0Bk/WGqwiUUvSIxTVR6g9I2kFpZyMB7h31vzWJOeBT3Lqew9hkH7bTdyUX9oXvzKE1S3WEjn7/iqwuVhztoPLzOPmnNerBqi+/sBGkTd/eRE5haqeHZOF4ybepTNf166A0arLq7d5qnpp5YXS9BCHyCsI0qG5xv4M2wKD3+maQE/x9Cdk+bUUVhpnvxHvDQ2wUccLKtOgDDtYX94D75aC+scPRaQGIUdXT9gL3vlhEAM4U27J4y1CfTIBqegwfuawnGNwgU3hNT69pVnz9gLuP0eqFQRc8DLwg3K/8Jn4YoLJ1lCaMy38fuYM2PTBp6vgHz/HtLKUD5xknyudwUb2Tqjnq5x2wL8PWRt65WlWXOJVLJkVFM3mv4Y+Jf5uaHwCGTf2/HrWszu2Ak4XD+xIo+g5TymY5uVfyfoFW439EWi22Q+QeY4zSh0T8OCbyXLh3nvr05tqxBMSLicoK3AgUSqDSksUZEe5dk3wR+0sUjXrh2erGdfuRwcGndYZxAnno4UWkNujHNUIU1WlT1nHfS7oB5qtLosyS2rNAIHkrSKilUP+MjaFPgWrwGg5fvVDWrWHHU8j37w3L9edYPoZqs5gJ3VREhecIWw59tAKLU2IuHpO7ZM8ydy2/ixnvTazHkX+HrCcadQ1YJcznZQDQDmtXpUlb0XBlDr7T9S/GDjR4AP7yZyAN///VgzJQHDWO7JErTE6Q/8CVSeWGd1zi72rvaZweKvqG52uuIv/9lVLpodKLbPcHXy86eQPaxQvGFy7n79F8J19siKJBMyFeMWwCk1osPBOI2uIu/0ExgOZAf9W332Lz2lYrHy9osPBOI7tdLZMzfb4RIgFpmExg5YeWn2/kUjSmPn2gZJwrXsevSwM6M4acUqOt2NFT6VwXXWLTC/zlWgCkmrg8ENPmBdISa5IRf9qwwc/v7+p7GDfRuWnwUW01Ey2TtAKd6HPgaNTND7wz05JMYG5FO7jrJI3360LRBoQisvpNEmktubHAth8V+QZ2WHqNA/EEmPZ3s2GzECfkO4vF3yFZZsCOP7y5QN+sH6VVrBXw6jpT6+Ou8IuVPS70ncDlsVE1eizPy11GQsswbduvja3hUe502hsaRRfW6eiOi3jvc99GEULqUTGu1kO+SpGHbmGypsVOQRX/MWqXFNz0e5dCRQvx7iY0DaC41xQOchtLl0t9IZMNNUNM4uhev47e4eJ983TdZ46veF6igpbAOx+B+OPipJUMRuHVAWOmo+yM0OHpdu7rFF8+6PfPlba/sfAjG/PMMWR8pafMsGcLbEfwxR+I4eFefK3rnowrEztg5/opz6sgCnTk3wdhjQcWRyZ5wDThXfXkLW35kjwP8XazddeGgtmSli1NJGpuiNjL//tS2Gb7vvbFKxjd5r8Efb2wFS/8X1i/ycBAIovjZaDO5rejgWIe8M/zwvvkRCRpvXQ26djqnZ3gbVe5pd6SzZwE+MtG7EqjrkvtDpWWNwPx2pI90+IwwphAABe//6iX/c1yZu7yAkGhNE1SoElwtyedmjmMsYC90jLx1jKEH//qJhEYR+Anbn92bXoKoC9POJ1A0jXjBWCRN3AGUuyQp461MBAfArnmbWdvCGvYWnWdycn61UYXYlyu3GuPxrd2pOFoF0kp+3tBOteItlFykyHZN0IHG1qaqyhprA7WnnQjYfhwe/K5FQsjeGxl0IiopkLbH6zvlC1O7oNIQNtLYuW/9y4W3LLoEp8qPtkUEnFmHX9Q71XVJqiuAEGnJ05arcEWpQJ+B9XO1vNkg61BD25ad6DU7V5XKrNEFurlwj7SBRAxV0ddpukTklX+VHeaaL2IBWdVBxEFoPerNNDWalYqO5kWpcRiLh71ClcjXwVqDePqPCSppvPjqN0rFqh+jMR5jrJcA3BI9av0RVeiOISKeesvvovvN7VzyxVOPnZuai7uhQ9ARrOFjEmYEUIA5Ck668QMT+h10WZxO5MOQcIoSUkVLe60jYgHb+dIVdDrG7lXaZdbrgXRYR1zxNy+qRr+hTVxeIBfmZJceN6sppr0OhaIjVtNalIr7euJFAHtZRKc/05i2Zyuwd6ohqW/zjFlNVAyS72/mHeo3sFqDO68T3XRouaKIoigOvekhgawA12lE+vyV8zYrzeoshDs2PA/XINrlBzCBW1Dd+4Yy/nUSjsfYAshLy1V/HjF6/0jXqwcYS1ztA/CQXivW9bZpN0JUOmBpb8UfU2g73GSp7TndPBHlP36XYM/fwawslzjMExtd9kGwelcXR/4Lj1MYtcil7QlG5IzQjMGgQQ3sb7R3QRMffX5cov5HJ9jXnfx2BX8Wwa8sIYezPyGQoqa3f8RI7JHk0mHSyqLksQg1AB2//0DbqDX20Yi6lYerVNFW/TSDwKwzYAmSGji6qmaoLzY/lHc7xZlo/0UahT3OTCWW1JuCWCiRuHmzlKtvcxxjf5k7HzojsFMz5MG2w3GHa+QiNjB9ssLhgMnxcSP+R2KbFmDADKD5yAI5LhAUNE0OL2WjaQ/jz2BwC/cIbb4iNnEv2/xrSlZAt+xgwNnoUuecP2nrYI2qPIEMs4zUca+YhLnMGv6mRGVNv95oribYJW84iuKWiuI2pjSPDBu4b4fKrkqB11/w9YBF9wE0DrAsIDi6Qb3a+e2p+T4dh9fRyj2DG07p8ZSy2PP9lxReMJhrurEwpgUMd+kxE9tUH6w2MXFM9aaxw0sUc88WHo9J32IroFH9pl0zlXEBtdtdobPVhJlilkLyRIEJ2PeJiUs4T03Pbx3T5L2aJ3nENQFD8+5ZmmoItfvh/KD7+74j1PiKMfpGvETStnoqG9OFN7yDP+uzDc9QV1qChSo9CQFabEZy1nqDBXr9q8hdIO+nfioC1JnRywRApGoL0INympsaeUKa8K+Aeq/etDYmdge/sAWALCUDee4xoxQnZPHqhQ9G+0d2eb/ZKOsq06z8FgmuDLWLckr3RPoSxWbNbzu8IUMn5g5lkrWKQjlsvzpsJp5nfmxwATK0gM1HVodoOVt//CC1VHAkEjpRC/HXPw9PvSu/g9PeZ/hP9AM+I3qepTNa3Fw5h3mkeE8ctflAx+rYRohuXGLj9wyPC7lWGtHTD+mZhrXP7EKOCnhSeX2JXD1ckY2+qbF+UNniELgAjxBpe+d0nSlPclyQ1vf02W22OWe6tgE4fpzZLpFH19VCl6MAw5jVG0Yfrfxdt/4PJ6fciOdJFUKNWiPVFxQqGHl44hfESLyV0KAvwVh3wHQgH753B5VYT0r5fjpZswNubx2tD8aCcT3BwoCktAjXzgBluKeV9KVtD5cIZCTU5qniHgU1IJGEfseEfSnBiNAKi1GkNXqb025Djdhg54SX/ZiDy9qUTN3K5AAHhmivTTjfObrVrF/lTUJOdXfPUDONVE8RCavJ3VEVV7V/PuVmgfjfwTfpX2uL02YCcaQvTt8Js+6z6F6bhJXSG8vbIh6q+/GBJFUjp/T4CfhW45bL9ET2WNf3SDBwslbjtlYu8Y1d0rsC4Sr4Ms1qReyaJ6+hYhZrGc+rDDLZ8itVMMEEXqTlGVgtqLlZNwrXZfzSpHbksZYeamBldwy3aFYlgoe6agXUIGXoHs/WfnmRmqjhMSU1LrRX7Ur1lpYpmhUbaXxZQ+tjCpao5xE30OSwgo8ItFsTt3h1eN8O2hI16IFcey81Mqjaa4JJZpEYmFe6hKObPaF4+2ogGHMJt9mQIbHEfpKihu2ekNLoExJtq3TByI84fzLVmGV7nO+Ub9AqCwiCtnbBLZSYRHh1MOiEmqUT/qN94PjnCdBPbInn3Qe/G5hhhqtqdLFyBjMSyWoCoDiEZTeurhc2vRD9yOBhCe+eL1K3rKpQZoN79+/w5/qK6WyN8nK/xHyousGN/RuH7tP+H8h6h0WymgzNS2TeIYwwBma/iLQ5+K52/Tv/+ESwqKjPJZQXCxgVWbYvK7ttdrsD3WSajikrvZ4TORd/gnxtFGm8iv4w/CxIgJ8iJsIVr4PNSnXTQI5Jx7T5y2dOyCsdj8nH6QK9ZqI6X4vQB2lSc3yOuJ9vuOPcgtEY3npHAJtqotqH6UVBAk/f0u7tz04wQ7UsJ/jGi0dwO8Thrw1zn0GeGn4Yonv92g9xSj+5WHsnwLjiTHG0RbgIbPZExOpmZbPfP+JlRmLBL6rZRpr4kpYTCgtlmt1JIp3bFHSTkvKNbEYjFxNCV6pnbM9Vd4J5NRT4MGXRyr7Uh8ASGnQvQlVoal8esOq4gJ/BRdaIjLIZDr3cJFFi03+mXkDC7rk0foA78kwWplSi2Bj5c2zv64KWAhYRiYffzJF3s0Gv7nGwchgy+0uLS42RCJ/rQ8HSsyHph7GBF8F2Cu1UtCbfCsPzbD5AG2xHTM4o5/ZeuXvoGgCZKe4DeXvxsURC9I7e7ykXJtCpWvlRf9JyKk9oYcF0YKnlDctspM8zjCv/FV7PkeospbI1Ja14j0ezgpuzohbjhiTF7c7v4+Fe3SYyb0EF/a6PIIk6I+D/Beb6mIhzUvVV/mnfjatzoc4W17kdNZek8QD1fdtX7i80RwbPn4NMCJresfSz3x1qpypg4LR0CgjLk8LQVrxXj1tzWhuGJ+6pQuTiJ4X3JeTjoU0VYuo55ZnLKnirh1CEvzkmoQ6VkoNAMeZrjPC7na07UHkadYWPDibMyt+OQ5VKs4SjvRqT4pu3Z89kSJBjPM4e06IsFmSqr1tdygMTLn82/KssPGApDHZEZKXzJkbQCnRiK8+17uBmmvRAzDQP+WrMjNi87v6tU6pwbRjSzjbKowMMd1AthO83+uCZ7SQcq8lUzaCb8pgJfxTngJno0WJr+lUjVEp9BHAqJ1DKp3cmZjr4/OoLbkkFt8YW1jLzCJdk6KuB4/2hLTCK4dTzpiLvxyFxskuySJKxftyF5wpA0JxN/+ClYCcisFeOoYu/tsgaVBe33i4vc3OxY7rakkVqdxqfza6eik7Ik5bTgx5hVC+8sBQIEyfVWlSGUq/txNTH7CBPdqgB0GUIzeJEQDEd314WANa1jQ5OwPXx0P5GASXo40M9HdK9QmJTe1+F3oXaQ8rxnUcXcQuNH+QyxdR0xt9fn3tReRpUg1zRk0UQN6aGr/iyW2sZKI2+QcA0jxav2Wu2G38T96nALwknFHwv6p7wx5zT8mjdpOff1AcZp9RsbiGEh5aT96KOVk6numlJmNeBJJ4KCjWi1g9YJKlJlstu8loc7oRv1xVd52+JsliVl5rUAue8Yysuy8oywiTfPtN6QbzbnQ3UGf1s5+Anq5bWGsaPxfVgGDjh8NTf0vvDuvos/vvzz9lKDoDVL9/zKqxfyvg8Suli1JHOKENdR1TQwyAL1426NY5Xtvc+L6XhHgxaL3vm2227BzEXWGM7vmi0e2MTma6SKn/+g59MLDbgobZC5QfwuOzKkLMcdldE1XBd4qYgf3itU0UmiQhxjX9M92YKOpPWQJf47frjeaCsd9Ck9BiSwVJGChTnIuF35WM5a14R+RXTbXOZdMsPNOwpOtI4p/th2PG0q/aEAoUKPfauCJxLBol/KU9lFn7jX6rnnNj6vQycRXiJVMatMWso3AFyE+XDPlZMmXxNOjABHwwsPMY0A4PrZn3BwBrWu5ytpA6zZEyacL5NLkivpuC3WT2uZvy48J7HGXC2NHSWbEWNxDutXEJIqUSD5YtyAy2tpNXK8YJldVLPqSUNQVQb+ryBJd/BT4+BbZfcvp6jZyJLueG9hHYte9C4pNQiM+AqoPTTzq3i4++9ar+ZTEwTvtp0omx2JhQCbVw9A2V0X4qEqXSBUewag0BBvIPGyb2xn9m1ryFDiUWPBQ4X76rFnmQGPuJR3Rm2tdlaJXlsOq23MP8oxZrU+OxiOJhTvVkynDerx5PuLnWG+8i1JYMPKjRPXZwZYsUPAKO8JrdptcLZ57M7nEmw/zKmKyhdeOjFC9WZ9QHCmYnXoB6BPq45Kwr8QmQJDZdbV355yi2in3RFIlpOVI1phHqv3aRqRSspZgDX6WcsMQgSKtkhZuAvyU5E1r9sCOnXe3n5jm3DQjcI64f6Jbaua4BKzmCnTGMiPaA1GgVtYQ+Se/ayJ2df3KZVFLsabDAkbqZyROEN3KHoAHOJobNVXYzkML+BqHKtaiFycwpkbntr3m/ocfs3jIXaTE1ficzPVB/85+6ICzmJzNnO3SWnCkxdINqfx8sz+8jxESCECbmN+0jnQDbi3+qg2NZp9HUlHxaVkmdl87DlE/yX0w6d5/G2v705ZZ+D85C9Z8GOSYTNO7+3PAVVHerlJ064ZT/nns1XE6H0p6zPAiGiht81bxpelObALTxFfES5//2Es+Ba/WU6aarmpAQPwksJoaFWG4iiKfqjt41Rv8aMw+NsH8Sbm/42pjCnttQd34yxVtD/T2xK4wqqnErqzLWBybKJqB77YX3JyRiVv5EHtXYMbKmkSAeO5zzsnfMS0FpQGEQCj1uSeAnujYZprjQNqNUAW8b5Q1dyFdT6q3wsoTgUV1bbkZg4V2hMmxmpAepAGLXbyoiVMN3k/3w0Jri7AFKFUwF9VNTX0kSlMvb1f7akoPC9aZyBEl+SLntnihC9vfBhNDJny2Qj7cCaI7EkK8IVwkACWYuKaGIW2Q15qZJuMnh4zgBCQm7KBMwWbbIJamIxgPtbzxIl5Ae7BW+n7txDNBZV43MIjgieXPYU7uTE17HknT7vxOeLO9fAQa7LQZSMCW387r0ei3R4IkzZJ5UrsPvlKq0fhJ8T29rGzlKS4n4MwuiruiTphOI/aATXDPq/dP/OLX6DU1ddyKQQ3jRxQe/Et1y/QnEMsolK/JoiQ0vYJio7SqosjFnBZIyQP39OG89r4f+Fnq8eXHfbTwVb5E0KXwf3WpPeKN3khkv0PRJJZmN7dsxkxGHLPmL70YgZweduYDTlE050bJsjQ3Tm8GfZvwPDew5sF8eYUBw3WjTeQqnxwgInrsUhtZYn0SZyfJ9///1fKxw9/8J1/J4X/0KEvAbVYsCV93mOlxsJ/+eY5CCUKygaAAAAAAA7YNi3HNYm68tdNCZKFjl2Gi8z9vaHjzOfbK5A0XLtfbQUTHoMcHfx0X+hZYIDKsG7ftQW/BAAQKh+jt9Tg//s6ZspKVp+BQOd+6aqGBkPAlViEZEaXLPLcRqsGNRwaDX+dTxP8dQ/0M+gtWLSf+Lh/F0C3c5FZ4CqFHe8va7ViehM4ENJOsXSkeBAtKBqwM1373DUjaeVZbgEJd5dMUfD1F7+xKN1bMJRaxnWQIDR6XHcCEOrdJcRsODH9UWSAMQIflMzTDD7MYsmzX+NxzlK6a4uHXiQNAmGoko23f+XQaxN2JaMM7YPNqm5Bq2PjAhmm/HW94ap41ZlBo6YCyvUd19/5DQawyUmIczRBdcQA19yxjvSMwR4WP3GTVWAnYmT/EKRw5EHnovBEXEhGhI43usyHHOQxJhOzjYZAQ2YyFVajfwN+2+gL0o14wMk8OQgCAl5J17ETpAnlSObY9MzP9W2gDrS9sAT7uB2yvsDfYslLmyPOdT0+nuK/jZk3fbZA8pc67mAHovryD/rsA1WFz6Wzo947pY9at/nv2VMf/xt///8wP52PpbzXZFkqu+6Yb0Qbu6o8HRXu9sU62+bAAAAAAAAA==" />
        <feGaussianBlur in="SourceGraphic" stdDeviation="0.04" result="blur" />
        <feDisplacementMap id="disp" in="blur" in2="map" scale="0.5" xChannelSelector="R" yChannelSelector="G">
        </feDisplacementMap>
      </filter>
      <filter id="toggler" primitiveUnits="objectBoundingBox">
        <feImage result="map" width="100%" height="100%" x="0" y="0" href="data:image/webp;base64,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" />
        <feGaussianBlur in="SourceGraphic" stdDeviation="0.01" result="blur" />
        <feDisplacementMap id="disp" in="blur" in2="map" scale="0.5" xChannelSelector="R" yChannelSelector="G">
        </feDisplacementMap>
      </filter>
    </svg>
  </div>
</fieldset>
<article class="article">
  <h1>Liquid glass</h1>
  <p>
    Lorem ipsum dolor sit amet consectetur adipisicing elit. Qui maxime optio quam debitis autem, maiores odio tenetur dicta aperiam aliquam, iusto nisi ipsum tempore dolore doloremque facere non culpa sint sequi ducimus corporis veritatis cumque corrupti sed. Ipsa dolor quod alias dicta dolores. Ducimus pariatur nostrum quo, impedit <a href="">facilis voluptatibus</a>! Non doloremque, facere neque dolorem animi earum odio placeat quae voluptatem nisi nihil deleniti voluptatibus harum magnam adipisci tenetur. Ab aliquid reprehenderit accusamus id quae velit quasi nostrum possimus ipsa vel. Voluptate nisi impedit placeat nemo neque asperiores iure saepe, eius eligendi odit quisquam delectus veniam autem, sequi laborum atque
  </p>
  <figure>
    <img src="https://images.unsplash.com/photo-1706720094773-d91e070e4b90?q=80&w=2515&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Photo by Neeqolah Creative Works on Unsplash">
    <figcaption>Photo by Neeqolah Creative Works on <a href="https://unsplash.com/photos/a-close-up-of-a-green-object-on-a-yellow-background-A2kyFEwh3zo">Unsplash</a></figcaption>
  </figure>
  <p>
    Sit amet consectetur adipisicing elit. Quibusdam illum in voluptates omnis reprehenderit inventore perferendis dolores, architecto doloribus. Quam error qui nam quis! Dolorum, dolore saepe ipsam quae aliquam tenetur dolores dolor repellendus facere a quasi soluta voluptate provident earum cum. Nostrum consequuntur corporis quibusdam tempora amet, animi inventore dicta voluptas nisi placeat ut illum explicabo! Consequatur, accusamus. Labore quo perspiciatis quos est molestiae illo possimus deserunt sed amet maxime, dolorum praesentium nihil doloribus similique cumque eveniet qui vero eius magni reprehenderit aspernatur, et, quod voluptatum? Voluptatem, eveniet vel sunt nesciunt recusandae alias maiores optio labore, dolorum aut possimus iusto quia. Non quas excepturi ea, maxime mollitia fugit, animi alias ad repudiandae dolores facere molestiae id, nemo temporibus explicabo.
  </p>
  <blockquote>
    Et aliquam libero deserunt maxime! Perspiciatis neque deserunt sequi deleniti!
  </blockquote>
  <p>
    Recusandae doloribus, ullam inventore esse culpa cupiditate dignissimos qui ducimus possimus ipsum reprehenderit, suscipit debitis nihil sit. Animi eligendi sed molestiae. Repellat, est ut eos voluptates tempora quisquam corporis mollitia, excepturi commodi cum dolore asperiores eaque debitis fuga quidem! A laborum ab reiciendis saepe rerum iste. Nisi quos earum ex nulla neque <a href="">tempora explicabo reiciendis</a> vitae sapiente accusamus tempore consequatur nemo, placeat magni quasi corrupti nobis alias, rerum ipsam fuga quisquam tenetur repellendus! Sit laborum aut odit pariatur quos cupiditate iure maiores. Quis nemo dicta numquam autem debitis blanditiis, odio explicabo maxime officiis magni soluta dolor suscipit, quo nisi eaque deleniti dolorem aperiam deserunt. Animi dolor expedita consectetur deserunt, quae id ab.
  </p>
  <p>
    Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi aperiam enim eos repellendus molestiae, ad ab maxime et, aut nam culpa eveniet. Eaque consequuntur laudantium ex dolore voluptas rem cumque, quos voluptatibus aut maxime. Animi enim odio ea porro molestiae laudantium ipsam nulla quia distinctio quod incidunt pariatur accusamus possimus laborum veritatis maxime, asperiores quasi!
  </p>
  <h2>
    Doloremque nisi eius quis
  </h2>
  <p>Magnam quo voluptate vitae voluptatem expedita vel illum ut. Tempore, sed? Sunt distinctio minus dolore, consequuntur eos qui eveniet error rerum tempora, autem et quaerat, ea repellendus unde iure. Fuga ad tempore cupiditate animi iste, eius nam beatae, aliquid quae id iusto perspiciatis. Quos quis cum itaque facilis libero sequi quaerat ipsam hic blanditiis doloribus modi sapiente tempora ipsum fugiat similique cupiditate beatae provident eos natus doloremque alias, sed praesentium.
  </p>
  <figure>
    <img src="https://images.unsplash.com/photo-1734606901283-489b25f7aa9b?q=80&w=2360&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Photo by Neeqolah Creative Works on Unsplash">
    <figcaption>Photo by Irene Demetri on <a href="https://unsplash.com/photos/a-close-up-of-a-green-object-on-a-yellow-background-A2kyFEwh3zo">Unsplash</a></figcaption>
  </figure>
  <p>
    Quod iste recusandae sed labore corporis ea provident debitis hic maxime placeat alias rem cumque animi explicabo laboriosam, dicta molestias? Corporis quibusdam, aliquam asperiores quo officia reiciendis nemo aspernatur similique voluptatibus in tempora? Laborum temporibus ipsa at exercitationem ullam labore tempore neque. Porro earum distinctio rem perferendis amet nihil maiores ab id cumque quae nesciunt, velit fugiat in aspernatur unde. Ut dolor eaque, vero quo sunt sed exercitationem cum accusamus, beatae, quasi rerum. Id eius dolorem nihil porro, voluptatem exercitationem tenetur impedit laudantium odit aliquam vel, facilis eos quasi voluptatibus optio. Ipsum voluptatum labore consequatur dolorum veritatis qui placeat magnam aut expedita. Error facere tempora ratione sapiente illo laudantium, ut nihil. Mollitia vitae, eveniet blanditiis corrupti at molestias aliquam ipsa esse optio temporibus dolorem, fugiat obcaecati culpa fuga numquam sapiente libero. Impedit maiores doloremque iusto autem temporibus! Quam cupiditate officiis possimus, ratione quasi nobis rem quidem quibusdam? Ut earum possimus expedita culpa nobis suscipit, non minima at dignissimos aut optio eum? Cumque?
  </p>
  <p>
    Lorem ipsum dolor sit amet consectetur adipisicing elit. Excepturi, aut commodi aliquid, blanditiis doloribus suscipit perferendis hic assumenda dolorum magnam enim. Error minima magnam modi aliquam quia at obcaecati sunt quas velit numquam, voluptatem inventore blanditiis necessitatibus harum possimus nostrum vel expedita quam dignissimos tenetur maxime provident aspernatur nam iste. At veritatis nemo quisquam, facilis numquam nostrum veniam perferendis repudiandae cum ipsa eos eius est doloremque excepturi delectus quidem repellat suscipit et tenetur vitae porro nisi iure? Ipsa facere at blanditiis nobis temporibus labore architecto nulla deleniti quod hic minus consequuntur, unde, quasi eius consequatur est! Laborum neque illo ab.
  </p>
  <blockquote>
    Consequatur itaque sit amet consectetur adipisicing elit.
  </blockquote>
  <p>
    Ab eveniet recusandae incidunt id cumque porro? Dignissimos nostrum iure possimus mollitia sed provident esse optio odio consequatur! Recusandae dicta ab atque? Eveniet molestias autem accusantium vero, tenetur placeat ratione voluptates nesciunt, adipisci ullam cum quae eius eaque quibusdam nihil dolorem! Ab eveniet recusandae incidunt id cumque porro? Sit tenetur quasi odit, facere dolore quae voluptatibus fugiat, quibusdam vitae labore dignissimos. Unde, maiores cumque dolore quas repellat vel adipisci harum incidunt consequatur ad!
  </p>
  <p>
    Ipsum explicabo quod laborum non provident, consectetur sint unde hic dolor quisquam ullam nulla veniam! Sapiente impedit sint expedita officiis fugit est quia. Neque, ea aspernatur? Quod voluptas ab aliquid saepe id, eos suscipit, autem quae reiciendis omnis veritatis officiis culpa doloremque? Reiciendis ea modi quod ducimus culpa omnis aliquid non minus ipsam necessitatibus distinctio, aspernatur iure. Nihil cupiditate amet officiis eligendi nostrum harum labore porro optio obcaecati reiciendis blanditiis quae, perspiciatis voluptatibus suscipit enim.
  </p>
  <p class="box">
    Perspiciatis sapiente eum velit inventore illum accusamus eos at esse mollitia debitis quae rem odit, ipsam nam. Voluptas beatae, velit voluptatum dolor obcaecati a nobis consequuntur quis id eaque! Sapiente nostrum rerum esse quo laborum excepturi explicabo, perspiciatis dicta corporis atque?
  </p>
  <h2>Quod voluptas aliquid id saepe</h2>
  <p>
    Et minima quas amet! Debitis commodi consectetur laborum fugit voluptatum qui distinctio, natus odit obcaecati. Voluptate suscipit consectetur, aspernatur ratione impedit minus facilis voluptatum tempora nesciunt pariatur ipsa provident qui distinctio ad quasi magnam exercitationem itaque. Recusandae maiores dolor saepe commodi a eligendi aut adipisci neque dolorem tempora perspiciatis odio asperiores quos iusto animi quia tempore, molestiae deleniti vel. Earum, atque? Ipsa reiciendis cum laborum fugiat doloremque asperiores, voluptatem exercitationem in sequi!
  </p>
  <p>
    Dolor sit, amet consectetur adipisicing elit. Earum, amet ea quaerat, repellendus ipsum sunt blanditiis ab tempora doloribus quas voluptas magni ut perferendis unde dolor tenetur repudiandae, porro dolore et nisi eius. Eos, esse minus molestiae maiores praesentium dolorum dolor eaque repellendus laudantium provident adipisci alias inventore ratione distinctio libero facilis voluptas hic illum odio laboriosam quos minima officia natus. Maxime eveniet repellendus non iste exercitationem aliquam velit, corrupti ut, id provident reprehenderit accusamus nemo reiciendis fugiat saepe quas quasi ad tempora! Qui earum eligendi architecto delectus animi similique laborum dolorum, recusandae beatae voluptatibus! Maiores ad libero quibusdam temporibus, quis possimus perferendis corporis modi, culpa voluptatum maxime. Earum, rem.
  </p>
  <p>
    Eos dolore dicta delectus dolorum porro fuga modi, perferendis natus ratione repellendus sit harum sint! Numquam quas facere quod doloribus enim omnis earum maxime perferendis ipsam magnam quos culpa, vitae, tenetur quisquam, unde quis maiores. Eos quos earum illo animi exercitationem nemo, at asperiores soluta similique aliquid cum quam quo corrupti accusantium iure praesentium. Omnis aliquam harum reiciendis necessitatibus temporibus expedita voluptate, dignissimos ratione dolorem perferendis ut aliquid magni nam nisi deserunt eum nulla excepturi quos sunt accusamus earum cupiditate minus. Expedita, corporis suscipit mollitia qui repellendus harum cumque id quis architecto quidem, aliquid vero dolores necessitatibus animi eos optio? Perspiciatis!
  </p>
  <p>
    Maxime in officiis fugiat corporis cum doloremque atque, at architecto dicta quod inventore corrupti facilis repellendus pariatur possimus error! At aperiam aliquam fugiat eveniet sapiente soluta hic ab ipsam, corrupti temporibus tempore? Velit, sed quos mollitia deleniti fugiat tempora fuga reprehenderit neque tempore doloremque ducimus quasi? Sit facere porro esse vero accusantium praesentium odit. Quisquam sint nobis dolorum labore distinctio ducimus animi alias quod necessitatibus rem non error consectetur voluptate tempora libero facilis ratione voluptates expedita, unde a blanditiis odio. Nostrum sapiente dolor enim?
  </p>
  Made with love by <a href="//codepen.io/DedaloD">Den Dionigi | UX Designer</a>
</article> 

---

## Assistant (Build · minimax-m2.5-free · 28.2s)

_Thinking:_

The user wants me to adapt their step switcher code to the title bar step navigation in the MANTAX project. Let me first look at the current titlebar implementation to understand what needs to be changed.

Looking at the example code, they have a radio button based switcher with:
- A fieldset with class "switcher"
- Radio inputs with name="theme" and c-option attributes
- JavaScript that tracks the previous value using c-previous attribute

I need to:
1. First check the current index.html to see the titlebar structure
2. Then check the current app.js to see how step navigation works
3. Adapt the example to work with the step navigation in the titlebar

Let me read the relevant files first.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/electron-ui/index.html",
  "limit": 150
}
```

**Output:**
```
<path>/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/electron-ui/index.html</path>
<type>file</type>
<content>1: <!DOCTYPE html>
2: <html lang="en">
3: 
4: <head>
5:     <meta charset="UTF-8">
6:     <meta name="viewport" content="width=device-width, initial-scale=1.0">
7:     <title>Face Recognition</title>
8:     <link rel="stylesheet" href="/styles/design-system.css">
9: </head>
10: 
11: <body>
12:     <!-- Sidebar -->
13:     <div class="sidebar" id="sidebar">
14:         <div class="sidebar-content">
15:             <div class="sidebar-step active" id="sidebar-step-1" onclick="jumpToStep(1)">
16:                 <img src="/styles/sf_icons/person.fill.svg" class="step-icon" alt="Step 1">
17:                 <span>Choose Photo</span>
18:             </div>
19:         </div>
20:     </div>
21: 
22:     <!-- Titlebar -->
23:     <div class="titlebar" id="titlebar">
24:         <div class="titlebar-left">
25:             <!-- Custom Traffic Lights -->
26:             <div class="traffic-lights">
27:                 <div class="light close" onclick="closeWindow()" title="Close"></div>
28:                 <div class="light minimize" onclick="minimizeWindow()" title="Minimize"></div>
29:                 <div class="light maximize" onclick="maximizeWindow()" title="Maximize"></div>
30:             </div>
31:             <!-- Sidebar Toggle -->
32:             <img src="/styles/sf_icons/sidebar.left.svg" class="sidebar-toggle-icon" onclick="toggleSidebar()" alt="Toggle Sidebar">
33:         </div>
34:         <div class="titlebar-center">
35:             <!-- Step Icons -->
36:             <img src="/styles/sf_icons/person.fill.svg" class="step-nav-icon" onclick="jumpToStep(1)" alt="Step 1">
37:             <img src="/styles/sf_icons/face.dashed.fill.svg" class="step-nav-icon" onclick="jumpToStep(2)" alt="Step 2">
38:             <img src="/styles/sf_icons/photo.badge.plus.svg" class="step-nav-icon" onclick="jumpToStep(3)" alt="Step 3">
39:             <img src="/styles/sf_icons/list.number.svg" class="step-nav-icon" onclick="jumpToStep(4)" alt="Step 4">
40:         </div>
41:         <div class="titlebar-right">
42:             <!-- 3-dot Menu -->
43:             <img src="/styles/sf_icons/ellipsis.svg" class="menu-icon" alt="Menu">
44:         </div>
45:     </div>
46:                     </svg>
47:                 </div>
48:                 <div class="switcher__option">
49:                     <input type="radio" name="theme" value="dark" class="switcher__input" c-option="3">
50:                     <svg class="switcher__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
51:                         <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
52:                         <circle cx="12" cy="12" r="3"></circle>
53:                     </svg>
54:                 </div>
55:             </div>
56:         </div>
57:     </div>
58: 
59:     <div class="container">
60:         <!-- Title moved to titlebar -->
61: 
62:         <!-- Input Methods Row -->
63:         <div class="steps-row">
64:             <!-- Step 1 -->
65:             <div class="step" id="step1">
66:                 <div class="step-checkmark">✓</div>
67:                 <div class="step-header">
68:                     <div>
69:                         <span class="step-number">Step 1</span>
70:                         <h2>Choose Your Photo</h2>
71:                         <p>Select a photo containing faces you want to analyze.</p>
72:                     </div>
73:                     <div class="button-group">
74:                         <button class="btn" onclick="clearAllCache()">Clear Cache</button>
75:                         <button class="btn btn-primary" onclick="selectImage()">Choose Photo</button>
76:                         <input type="file" id="imageInput" accept="image/*" onchange="handleImageSelect(event)">
77:                     </div>
78:                 </div>
79:                 <div class="preview-container hidden" id="previewContainer">
80:                     <div class="preview-box">
81:                         <img id="selectedImage" src="" alt="Selected">
82:                         <p class="preview-label">Selected Image</p>
83:                     </div>
84:                 </div>
85:             </div>
86: 
87:             <!-- Webcam -->
88:             <div class="step" id="webcamStep">
89:                 <div class="step-checkmark">✓</div>
90:                 <div class="step-header">
91:                     <div>
92:                         <span class="step-number webcam">Webcam</span>
93:                         <span class="step-number optional">Optional</span>
94:                         <h2>Capture from Webcam</h2>
95:                         <p>Use your webcam to capture a photo for analysis.</p>
96:                     </div>
97:                     <div class="button-group">
98:                         <button class="btn btn-primary" onclick="startWebcam()" id="startWebcamBtn">Start
99:                             Webcam</button>
100:                         <button class="btn" onclick="captureWebcam()" id="captureWebcamBtn" disabled>Capture</button>
101:                         <button class="btn" onclick="stopWebcam()" id="stopWebcamBtn" disabled>Stop</button>
102:                     </div>
103:                     <p id="webcamStatus" class="status">Click "Start Webcam" to begin</p>
104:                 </div>
105:                 <div id="webcamContainer" class="hidden text-center">
106:                     <video id="webcamVideo" autoplay playsinline class="webcam-video"></video>
107:                     <canvas id="webcamCanvas" class="hidden"></canvas>
108:                 </div>
109:             </div>
110:         </div>
111: 
112:         <!-- Step 2 & 3 -->
113:         <div class="steps-row">
114:             <div class="step" id="step2">
115:                 <div class="step-checkmark">✓</div>
116:                 <div class="step-header">
117:                     <div>
118:                         <span class="step-number">Step 2</span>
119:                         <h2>Find Faces</h2>
120:                         <p>The AI will automatically detect all faces in your photo.</p>
121:                         <p id="detectStatus" class="status">Waiting for image...</p>
122:                     </div>
123:                     <button class="btn btn-primary" onclick="detectFaces()" id="detectBtn" disabled>Find Faces</button>
124:                 </div>
125:                 <div id="facesContainer" class="hidden">
126:                     <div class="gallery" id="facesGallery"></div>
127:                 </div>
128:             </div>
129: 
130:             <div class="step" id="step3">
131:                 <div class="step-checkmark">✓</div>
132:                 <div class="step-header">
133:                     <div>
134:                         <span class="step-number">Step 3</span>
135:                         <h2>Create Signature</h2>
136:                         <p>Convert each face into a unique mathematical signature.</p>
137:                         <p id="extractStatus" class="status">Waiting for detection...</p>
138:                     </div>
139:                     <button class="btn btn-primary" onclick="extractFeatures()" id="extractBtn" disabled>Create
140:                         Signature</button>
141:                 </div>
142:             </div>
143:         </div>
144: 
145:         <!-- Step 4 + Comparison Result (side by side) -->
146:         <div class="steps-row">
147:             <!-- Step 4 -->
148:             <div class="step" id="step4">
149:                 <div class="step-checkmark">✓</div>
150:                 <div class="step-header">

(File has more lines. Use 'offset' parameter to read beyond line 150)
</content>
```

---

## User

---

## Assistant (Compaction · minimax-m2.5-free · 26.6s)

_Thinking:_

Let me summarize the conversation so far based on what happened:

1. User asked to read master and system prompts
2. I read master_promt.md and system_promt.md 
3. User asked to read all other .md files
4. I read all the .md files in the project (about 20+ files)
5. User asked to make a small change to the step switcher in the title bar
6. User provided example code for a switcher component
7. I started reading index.html to understand the current implementation
8. User then asked "What did we do so far?" and asked for a summary

Let me compile this into the requested format.

## Goal

The user wants to improve the step switcher in the title bar of the MANTAX facial recognition application. They've provided example code showing a radio-button-based switcher component with theme selection (light/dark/dim) that uses custom attributes (`c-option`) to track previous and current selections.

## Instructions

- Read all project documentation (master prompt, system prompts, and .md files) to understand the project context
- The step switcher needs to be adapted from the example code provided - it should be a radio-button-based component with `c-option` attributes
- Make a "very small and easy change" - don't break anything
- Follow the project's strict workflow rules:
  - Run tests after code changes: `python test_e2e_pipeline.py`, `python test_edge_cases.py`, `python test_frontend_integration.py`
  - Verify syntax checks pass
  - Run HTML-JS cross-check before committing

## Discoveries

- **Project**: MANTAX - NGO Facial Image Analysis System (v0.5.2)
- **Stack**: Flask backend (port 3000) + Electron frontend + Vanilla JS + SCSS
- **ML**: ArcFace (512-dim ONNX) + FaceNet (128-dim PyTorch)
- **Design**: macOS Tahoe Liquid Glass
- **Critical Rules**:
  - Always run tests after code changes
  - Use project venv (`source venv/bin/activate`)
  - Use `./start.sh` to restart (clears cache)
  - HTML-JS cross-check mandatory before commits

The example code shows a fieldset with radio inputs using `c-option` attributes and JavaScript that tracks the previous selection via `c-previous` attribute.

## Accomplished

- ✅ Read master_promt.md and system_promt.md
- ✅ Read all ~20+ .md documentation files in the project
- ❌ Started reading index.html to find current step switcher implementation (partially complete)
- ❌ Change to step switcher NOT YET IMPLEMENTED

## Relevant files / directories

- **Project Root**: `/Users/modernamusmenet/Desktop/MANTAX/`
- **App Directory**: `/Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo/`
- **Frontend HTML**: `electron-ui/index.html` - Currently being examined for the step switcher implementation
- **Frontend JS**: `electron-ui/renderer/app.js` - Contains step navigation functions like `jumpToStep(n)`
- **Documentation read**:
  - `master_promt.md` - Master prompt
  - `system_promt.md` - System prompt  
  - `CONTEXT.md` - Critical edit rules
  - `ARCHITECTURE.md` - System design
  - `README.md`, `START.md`, `DEVELOPMENT_LOG.md`, `WORKFLOW_PROMPT.md`, `SYSTEM_PROMPT.md`, `SYSTEM_PROMPT_COMPLETE.md`, `PROJECT_STRUCTURE.md`, `PROJECT_ARCHITECTURE.md`, `FRONTEND_SUMMARY.md`, `PHASE1_COMPLETE.md`, `IMAGE_STORAGE.md`, `USAGE.md`, `INSTALLATION.md`, `ETHICAL_COMPLIANCE.md`, `reference_images/README.md`, `ARCHITECTURE_REVIEW.md`, `rule_set.md`, `STYLES.md`, `FACE_ATTRIBUTES_PLAN.md`

## Next Steps

1. Complete reading `electron-ui/index.html` to find the current titlebar center section with step navigation icons
2. Adapt the example switcher code to work as a step switcher (replace theme switcher with step switcher)
3. Update the JavaScript to track step changes instead of theme changes
4. Run all tests to verify nothing broke
5. Say "FINISHED" only after all tests pass

---

## User

---

## Assistant (Build · minimax-m2.5-free · 23.7s)

_Thinking:_

The user is asking "What did we do so far?" - this is a meta-question about our session history, not a request to continue with a task.

Looking at the context, I can see this is a conversation where there's a "system reminder" tag showing what appears to be previous work context. This contains:
- A goal description
- Instructions
- Discoveries about a project
- Accomplishments status
- Relevant files
- Next steps

This appears to be context from a previous conversation or task. The user is now asking what has been done so far.

Based on the context provided in the system reminder, I should summarize what has been accomplished:

1. ✅ Read master_promt.md and system_promt.md
2. ✅ Read all ~20+ .md documentation files in the project
3. ❌ Started reading index.html to find current step switcher implementation (partially complete)
4. ❌ Change to step switcher NOT YET IMPLEMENTED

So the answer is: We read the documentation files but haven't made any code changes yet. The task of improving the step switcher has not been completed.

Based on the context, here's what we've done so far:

1. ✅ Read documentation files (master_promt.md, system_promt.md, and ~20+ other .md files)
2. ❌ **Not yet implemented** - We started reading `electron-ui/index.html` to find the current step switcher, but didn't complete this or make any code changes yet

The task of improving the step switcher in the title bar is still pending. We need to:

1. Complete reading `electron-ui/index.html` to find the current step switcher
2. Implement the radio-button-based switcher with `c-option` attributes
3. Run the test suite to verify nothing broke

Would you like me to continue with this task?

---

