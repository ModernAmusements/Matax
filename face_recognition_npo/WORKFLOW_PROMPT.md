# Comprehensive System Prompt - READ BEFORE EVERY TASK

You are working on the **NGO Facial Image Analysis System** - a face recognition application with Flask API + Electron UI.

---

## ⚠️ MANDATORY WORKFLOW - NEVER DEVIATE FROM THIS

### For EVERY code change, you MUST follow this exact sequence:

1. **Read relevant files first** - Understand the existing code before making changes
2. **Make your code changes**
3. **Check syntax** - Run `python -m py_compile <file>` for Python files
4. **Test locally** - Run the relevant test(s)
5. **Run ALL tests** (non-negotiable):
   ```bash
   python test_e2e_pipeline.py
   python test_edge_cases.py
   python test_frontend_integration.py
   ```
6. **Verify startup works** - Run `./start.sh` and confirm it works
7. **Say "FINISHED"** - ONLY after ALL tests pass

**NEVER say "done", "finished", or "ready to commit" until step 6 is complete.**

---

## PROJECT STRUCTURE - DETAILED VIEW

```
face_recognition_npo/
├── ROOT FILES
│   ├── api_server.py           # Flask API - main entry point (25+ endpoints)
│   ├── start.sh                # Startup script (kills servers, compiles SCSS, starts app)
│   ├── config_template.py      # Configuration template
│   ├── setup.py                # Setup script
│   └── visualize_biometric.py # Standalone visualization script
│
├── SRC/                        # Core ML modules
│   ├── detection/
│   │   ├── __init__.py        # FaceDetector class (OpenCV DNN + MediaPipe)
│   │   └── preprocessing.py    # ImagePreprocessor
│   │
│   ├── embedding/
│   │   ├── __init__.py        # FaceNetEmbeddingExtractor, SimilarityComparator
│   │   └── arcface_extractor.py # ArcFaceEmbeddingExtractor
│   │
│   └── reference/
│       └── __init__.py         # ReferenceImageManager, HumanReviewInterface
│
├── ELECTRON-UI/                # Frontend
│   ├── index.html             # Main UI HTML
│   ├── main.js               # Electron main process
│   ├── preload.js            # Electron preload script
│   ├── package.json          # NPM config (has sass)
│   │
│   ├── renderer/
│   │   └── app.js            # Frontend JavaScript (calls API)
│   │
│   └── styles/
│       ├── design-system.scss # SCSS source
│       └── design-system.css  # Compiled CSS
│
├── GUI/                        # Alternative GUI (not main)
│   ├── facial_analysis_gui.py
│   └── user_friendly_gui.py
│
├── UTILS/
│   └── webcam.py             # Webcam utilities
│
├── TESTS/                     # Unit tests
│   ├── test_detection.py
│   ├── test_embedding.py
│   ├── test_comparison.py
│   ├── test_reference.py
│   └── test_review.py
│
├── TEST FILES (root)          # Integration tests
│   ├── test_e2e_pipeline.py
│   ├── test_edge_cases.py
│   ├── test_frontend_integration.py
│   ├── test_eyewear.py
│   └── test_api_endpoints.py
│
├── REFERENCE_IMAGES/
│   ├── embeddings.json        # Stored references
│   └── README.md
│
├── TEST_IMAGES/               # Test images
│   ├── test_subject.jpg
│   └── reference_subject.jpg
│
├── _EXAMPLES/                 # Example scripts
│   ├── basic_usage.py
│   ├── webcam_demo.py
│   └── reference_images/
│
└── DOCUMENTATION (*.md)
    ├── README.md               # Main docs
    ├── CONTEXT.md             # Critical rules
    ├── ARCHITECTURE.md        # System design
    ├── DEVELOPMENT_LOG.md     # History
    ├── WORKFLOW_PROMPT.md     # This file
    └── ... (15+ more .md files)
```

---

## FILE INTERACTIONS - HOW THINGS CONNECT

### Backend Flow (api_server.py)

```
api_server.py (Flask)
    │
    ├── imports → src.detection.FaceDetector
    │            src.embedding.FaceNetEmbeddingExtractor
    │            src.embedding.ArcFaceEmbeddingExtractor
    │            src.embedding.SimilarityComparator
    │            src.reference.ReferenceImageManager
    │
    └── defines endpoints → /api/health, /api/detect, /api/extract,
                            /api/add-reference, /api/references, /api/compare, etc.
```

### Frontend Flow (electron-ui/)

```
index.html
    │
    ├── loads → app.js (JavaScript)
    │        design-system.css (compiled SCSS)
    │
    └── app.js calls → http://localhost:3000/api/* (Flask endpoints)
```

### Test Files → Source Files

```
test_e2e_pipeline.py
    │
    ├── imports → src.detection.FaceDetector
    │            src.embedding.FaceNetEmbeddingExtractor
    │            src.embedding.SimilarityComparator
    │            src.reference.ReferenceImageManager
    │
    └── Tests → Full pipeline (detect → extract → compare)

test_edge_cases.py
    │
    └── Tests → Edge cases (None inputs, empty lists, boundaries)

test_frontend_integration.py
    │
    └── Tests → API endpoints via HTTP requests

gui/facial_analysis_gui.py
    │
    └── Alternative GUI that uses same src modules
```

### Import Dependency Graph

```
api_server.py
    ├── src.detection (FaceDetector)
    ├── src.embedding (FaceNetEmbeddingExtractor, SimilarityComparator)
    └── src.reference (ReferenceImageManager)

test_e2e_pipeline.py
    ├── src.detection
    ├── src.embedding
    └── src.reference

gui/facial_analysis_gui.py
    ├── src.detection
    ├── src.embedding
    └── src.reference

tests/test_*.py
    └── (each imports relevant src modules)
```

---

## CRITICAL RULES - NEVER FORGET

### Rule 1: Always Use Project Virtual Environment
```bash
# WRONG - uses system Python
python api_server.py

# CORRECT - uses project venv
source venv/bin/activate
python api_server.py

# Or use start.sh which does this automatically
./start.sh
```

### Rule 2: Always Check start.sh Works
- start.sh MUST use `SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"` to resolve paths correctly
- start.sh MUST compile SCSS before starting
- NEVER use `$(dirname "$0")` alone - it returns "." when script is run with "./"

### Rule 3: SCSS Compilation
- SCSS source: `electron-ui/styles/design-system.scss`
- Compiled CSS: `electron-ui/styles/design-system.css`
- Always compile after changing SCSS: `npm run scss`
- start.sh should auto-compile SCSS

### Rule 4: Check for Duplicate Code
Before submitting, always check:
```bash
# Check for duplicate function definitions
grep -n "def " <file>

# Check for duplicate selectors in CSS
grep -n "\.class" <file>
```

### Rule 5: Count Parentheses and Brackets
Before any edit submission:
- Count opening and closing: `(`, `)`, `[`, `]`, `{`, `}`
- Ensure each `return` statement has a complete expression

---

## TEST COMMANDS

```bash
# Activate venv first
source venv/bin/activate

# Run all tests (REQUIRED after any code change)
python test_e2e_pipeline.py
python test_edge_cases.py
python test_frontend_integration.py

# Run specific test
python -m pytest tests/
python test_eyewear.py
```

---

## COMMON MISTAKES TO AVOID

| Mistake | Solution |
|---------|----------|
| Using system Python instead of venv | Always `source venv/bin/activate` first |
| Forgetting to compile SCSS | Run `npm run scss` or use `./start.sh` |
| start.sh path issues | Use `SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"` |
| Saying "done" before testing | ALWAYS run tests first |
| Breaking CSS with SCSS nesting | Verify compiled CSS matches original |
| Blocking UI with async/await | Use `.catch()` for fire-and-forget calls |
| Missing HTML-JS cross-check | Verify all onclick handlers exist in JS |
| Wrong import paths | Use `from src.detection import ...` not `from .src...` |

---

## CURRENT CONFIGURATION

- **Backend**: Flask on port 3000
- **Frontend**: Electron desktop app
- **ML Models**: ArcFace (512-dim) + FaceNet (128-dim)
- **Weights**: ArcFace 60%, FaceNet 20%, Landmarks 15%, Activation 1%, Quality 1%
- **Testing**: 3 test suites (E2E, Edge Cases, Frontend)
- **Persistence**: JSON file (`reference_images/embeddings.json`)

---

## API ENDPOINTS (from api_server.py)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/embedding-info` | Model info |
| POST | `/api/detect` | Face detection |
| POST | `/api/extract` | Embedding extraction |
| POST | `/api/add-reference` | Add reference |
| GET | `/api/references` | List references |
| DELETE | `/api/references/<id>` | Remove reference |
| POST | `/api/compare` | Compare embeddings |
| POST | `/api/clear` | Clear session |
| GET | `/api/visualizations/<type>` | Get visualization |
| POST | `/api/eyewear` | Eyewear detection |

---

## VERIFICATION CHECKLIST (Run before saying "FINISHED")

- [ ] Python syntax check: `python -m py_compile <file>`
- [ ] E2E tests pass: `python test_e2e_pipeline.py`
- [ ] Edge case tests pass: `python test_edge_cases.py`
- [ ] Frontend tests pass: `python test_frontend_integration.py`
- [ ] start.sh works: `./start.sh`
- [ ] No inline styles remaining in HTML/JS
- [ ] All CSS classes accounted for
- [ ] MD files updated if needed
- [ ] Understand file interactions for any changes

---

**REMEMBER**: After EVERY code change, run the tests, verify start.sh works, then say "FINISHED".