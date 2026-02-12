# Complete System Prompt: NGO Facial Image Analysis System

## Role and Context

You are working on the **NGO Facial Image Analysis System** - a complete, production-ready facial recognition system designed for ethical, consent-based documentation verification workflows. This is **version 0.4.1** and the system is **fully functional**.

## Complete System Understanding Checklist

BEFORE making any changes, you must have read and understood:

### ✅ Documentation Files (All Must Be Read)
1. **README.md** - Main documentation, features overview, ArcFace integration
2. **CONTEXT.md** - ⚠️ **CRITICAL**: Strict edit rules, developer workflow, common mistakes
3. **ARCHITECTURE.md** - Complete system design, API endpoints, visualization types
4. **PROJECT_STRUCTURE.md** - File organization, roadmap, critical lessons learned
5. **DEVELOPMENT_LOG.md** - Session-by-session development history
6. **ETHICAL_COMPLIANCE.md** - Privacy guidelines, ArcFace discrimination analysis
7. **INSTALLATION.md** - Setup and dependencies
8. **USAGE.md** - Quick start instructions
9. **IMAGE_STORAGE.md** - Embedding storage format
10. **SYSTEM_PROMPT.md** - This file (you are reading it)

### ✅ Code Files (Core Architecture)
1. **Python Backend**:
   - `api_server.py` - Flask API server (11 REST endpoints)
   - `src/detection/__init__.py` - FaceDetector class (OpenCV DNN + MediaPipe)
   - `src/detection/preprocessing.py` - ImagePreprocessor (CLAHE, histogram equalization)
   - `src/embedding/__init__.py` - FaceNetEmbeddingExtractor (128-dim, PyTorch)
   - `src/embedding/arcface_extractor.py` - ArcFaceEmbeddingExtractor (512-dim, ONNX)

2. **Frontend**:
   - `electron-ui/main.js` - Electron main process
   - `electron-ui/renderer/app.js` - Frontend JavaScript
   - `electron-ui/index.html` - UI with scrollable tabs (20 visualization types)

3. **Test Files**:
   - `test_frontend_integration.py` - Rich visual frontend tests
   - `test_e2e_pipeline.py` - End-to-end tests
   - `test_eyewear.py` - Eyewear detection tests

### ✅ Key Features (v0.4.1)
- **ArcFace**: 512-dim embeddings (default, recommended)
- **Pose Detection**: yaw, pitch, roll estimation
- **Pose-Aware Matching**: Adjusts similarity based on pose difference
- **Multi-Reference**: Store multiple poses per person
- **Image Preprocessing**: Auto CLAHE, histogram equalization
- **Eyewear Detection**: Sunglasses/glasses detection
- **20 Visualization Tabs**: Scrollable in UI

### ✅ API Endpoints (11 total)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/detect` | Face detection + preprocessing |
| POST | `/api/extract` | Embedding + pose extraction |
| POST | `/api/add-reference` | Add reference with pose |
| GET | `/api/references` | List references |
| DELETE | `/api/references/<id>` | Remove reference |
| POST | `/api/compare` | Pose-aware similarity |
| GET | `/api/visualizations/<type>` | 16 viz types |
| POST | `/api/clear` | Clear session |
| GET | `/api/status` | Debug state |
| GET | `/api/eyewear` | Eyewear detection |

### ✅ Visualization Types (16 total)
Detection, Extraction, Preprocessing, Landmarks, 3D Mesh, Alignment, Attention, Activations, Features, Multi-Scale, Confidence, Eyewear, Embedding, Similarity, Robustness, Biometric, Tests (9 individual test views)

### ✅ Thresholds (ArcFace)
- **≥70%**: Very High - Same person likely
- **45-70%**: High - Possibly same person  
- **20-45%**: Moderate - Different angles
- **<20%**: Insufficient - Different people

## Development Workflow

**Before any edit:**
```bash
python -m py_compile <file>
```

**HTML-JS Cross-Check (MANDATORY):**
```bash
grep -E 'onclick=|onchange=' electron-ui/index.html
```

2. **Frontend**:
   - `electron-ui/main.js` - Electron main process
   - `electron-ui/renderer/app.js` - Frontend JavaScript (25+ functions)
   - `electron-ui/index.html` - Ultra minimal UI with MANTAX branding

3. **Key Models Files**:
   - `deploy.prototxt.txt` - OpenCV DNN config
   - `res10_300x300_ssd_iter_140000.caffemodel` - Face detection weights
   - `arcface_model.onnx` - ArcFace model (117MB, 512-dim)

## Critical System Status

### ✅ Fully Functional (v0.3.0)
- **ArcFace Integration**: 512-dimensional embeddings, excellent discrimination
- **API Server**: 11 endpoints fully operational
- **Electron App**: Production-ready with MANTAX branding
- **Testing**: All tests passing (6/6 E2E, 30/30 unit, 11/11 edge cases)
- **Reference Management**: Persistent JSON storage with real embeddings
- **Visualizations**: 14 AI visualization types available

### 🎯 Key Achievement: False Positive Prevention
```
BEFORE (FaceNet): 65-70% similarity for different people (FALSE POSITIVES!)
AFTER  (ArcFace): <30% similarity for different people (CORRECT!)
```

This is **critical for NGO use** - prevents wrongful identifications.

## Development Workflow Rules (From CONTEXT.md)

### ⚠️ MANDATORY Rules (No Exceptions!)

**Rule 1: Syntax Check**
```bash
# BEFORE any edit
python -m py_compile <file>
```

**Rule 2: No Duplicate Code**
```bash
# Check for duplicates
grep -n "def " <file>
grep -rn "return {}" src/ --include="*.py"
```

**Rule 3: Function Preservation (JavaScript)**
```bash
# BEFORE any JavaScript edit
CRITICAL_FUNCS="selectImage handleImageSelect detectFaces extractFeatures compareFaces clearAllCache removeReference showReferenceVisualizations updateReferenceList"
for func in $CRITICAL_FUNCS; do
  if ! grep -q "function $func" electron-ui/renderer/app.js; then
    echo "MISSING: $func"
    exit 1
  fi
done
```

**Rule 4: HTML-JS Cross-Check (MANDATORY Pre-commit)**
```bash
# Extract HTML handlers and verify in JS
HTML_FUNCS=$(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u)
for func in $HTML_FUNCS; do
    if ! grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js; then
        echo "✗ MISSING: $func"
        exit 1
    fi
done
```

**Rule 5: Atomic Edits**
- Make ONE edit per function
- Verify each edit individually
- Never attempt multiple complex edits in one operation

**Rule 6: Read Context First**
- Read at least 50 lines around edit location
- Understand full context before making changes
- Identify all affected code

**Rule 7: Fire-and-Forget for Non-Critical APIs**
```javascript
// WRONG (blocks UI)
await fetch(`${API_BASE}/clear`, { method: 'POST' });

// RIGHT (fire-and-forget)
fetch(`${API_BASE}/clear`, { method: 'POST' }).catch(err => {
    console.log('Clear failed:', err.message);
});
```

## System Architecture

### Data Flow
```
User Upload → Electron UI → HTTP API → ML Pipeline → JSON Storage → Results
```

### Core Components
1. **Face Detection**: OpenCV DNN (primary) + Haar Cascade (fallback)
2. **Embedding Extraction**: ArcFace (512-dim, default) or FaceNet (128-dim, legacy)
3. **Similarity Comparison**: Cosine similarity with confidence bands
4. **Reference Storage**: JSON persistence with metadata tracking

### API Endpoints (11 total)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/health` | Health check |
| GET | `/api/embedding-info` | Model/dimension info |
| POST | `/api/detect` | Face detection |
| POST | `/api/extract` | Embedding extraction |
| POST | `/api/add-reference` | Add reference |
| GET | `/api/references` | List references |
| DELETE | `/api/references/<id>` | Remove reference |
| POST | `/api/compare` | Compare embeddings |
| GET | `/api/visualizations/<type>` | Get visualization |
| POST | `/api/clear` | Clear session |
| GET | `/api/status` | Debug state |

### 14 Visualization Types
1. **Detection**: Bounding boxes with confidence
2. **Extraction**: Face ROI preview
3. **Landmarks**: 15 facial keypoints
4. **3D Mesh**: 478-point wireframe
5. **Alignment**: Pitch/yaw/roll orientation
6. **Saliency**: Attention heatmap
7. **Activations**: CNN layer activations (placeholder for ArcFace)
8. **Features**: Feature map grid
9. **Multi-Scale**: Face at 5 scales
10. **Confidence**: Quality metrics dashboard
11. **Embedding**: 512-dim or 128-dim bar chart
12. **Similarity**: Similarity matrix
13. **Robustness**: Noise robustness test
14. **Biometric**: Biometric capture overview

## Critical Lessons Learned (From Development Log)

### Lesson 1: ArcFace vs FaceNet Discrimination
- FaceNet: 65-70% for different people (false positives!)
- ArcFace: <30% for different people (correct!)
- **Rule**: Always use ArcFace for NGO use cases

### Lesson 2: Dynamic Array Sizes
```python
# WRONG (causes broadcast error)
output = np.zeros((150, 300, 3), dtype=np.uint8)

# RIGHT (dynamic sizing)
output_size = max(150, n * cell_size)
output = np.zeros((output_size, output_size, 3), dtype=np.uint8)
```

### Lesson 3: Real Embeddings Only
```python
# WRONG (random embeddings)
embedding = np.random.rand(512)

# RIGHT (extract real embeddings)
embedding = extractor.extract_embedding(face_roi)
```

### Lesson 4: Clear Python Cache
```bash
# Always run after editing source
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### Lesson 5: Reference Persistence
- Call `save_references()` after add/remove operations
- Load references on API server startup
- Store real embeddings, not random values

### Lesson 6: UI Blocking with Async/Await
- Never await non-essential API calls in event handlers
- Use fire-and-forget pattern for operations like cache clear
- Blocking UI makes app appear frozen

## Common Bugs and Fixes

### Bug 1: Missing JavaScript Functions
**Symptom**: Buttons do nothing when clicked
**Cause**: HTML onclick handlers reference non-existent functions
**Fix**: Always run HTML-JS cross-check before committing

### Bug 2: Random Embeddings
**Symptom**: Similarity scores are meaningless
**Cause**: Using `np.random.rand()` instead of real extraction
**Fix**: Always call `extractor.extract_embedding(face_roi)`

### Bug 3: Broadcast Errors in Visualizations
**Symptom**: "could not broadcast input array" errors
**Cause**: Hardcoded array sizes for dynamic data
**Fix**: Use dynamic sizing based on input count

### Bug 4: References Not Persisting
**Symptom**: References disappear after server restart
**Cause**: Only stored in memory, never saved to JSON
**Fix**: Call `save_references()` after mutations

### Bug 5: Python Cache Issues
**Symptom**: Code changes don't take effect
**Cause**: Old `.pyc` files with stale bytecode
**Fix**: Clear cache and restart server

## Quick Start Commands

```bash
# Start system
cd face_recognition_npo
./start.sh

# Run tests
python test_e2e_pipeline.py
python test_edge_cases.py

# Clear cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Verify API
curl http://localhost:3000/api/health
curl http://localhost:3000/api/embedding-info
```

## Usage Workflow (Correct Steps)

```
Step 1: Choose Photo     → Upload image file
Step 2: Find Faces       → Click "Find Faces" button
Step 3: Create Signature → Click "Create Signature" (CRITICAL!)
Step 4: Add Reference    → Upload reference image
Step 5: Compare          → Click "Compare" button
```

**Common Mistakes:**
- Skipping "Create Signature" - no embedding extracted
- Clicking "Compare" before adding references
- Refreshing page (loses server state)

## ArcFace Confidence Thresholds

| Similarity | Confidence | Action |
|------------|------------|---------|
| ≥70% | Very High | Likely same person |
| 45-70% | High | Possibly same person |
| 30-45% | Moderate | Human review required |
| <30% | Insufficient | Likely different people |

## Ethical Guidelines

1. **Consent-Based**: All images require documented consent
2. **Human Oversight**: No automated decisions
3. **Uncertainty Handling**: Confidence bands instead of binary decisions
4. **Privacy Protection**: Non-reversible embeddings only
5. **Audit Trail**: Complete logging of all operations

## Pre-Commit Verification (Mandatory)

Run this before EVERY commit:

```bash
#!/bin/bash
cd /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo

echo "========================================="
echo "MANDATORY PRE-COMMIT VERIFICATION"
echo "========================================="

PASS=true

# 1. Python syntax
python -m py_compile api_server.py || PASS=false

# 2. Tests pass
python test_e2e_pipeline.py 2>&1 | grep -q "ALL TESTS PASSED" || PASS=false

# 3. HTML-JS cross-check
MISSING=""
for func in $(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u); do
    if ! grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js; then
        MISSING="$MISSING $func"
    fi
done
[ -n "$MISSING" ] && PASS=false

if [ "$PASS" = true ]; then
    echo "✓ ALL VERIFICATIONS PASSED"
    exit 0
else
    echo "✗ VERIFICATION FAILED - FIX BEFORE COMMITTING"
    exit 1
fi
```

## Current Status

- **Version**: 0.3.0
- **Status**: ✅ Fully Functional
- **All Tests Passing**: E2E (6/6), Unit (30/30), Edge Cases (11/11)
- **Default Model**: ArcFace (512-dim, ONNX)
- **UI**: Electron with ultra minimal design + MANTAX branding
- **API**: 11 endpoints fully operational
- **Storage**: Persistent JSON with real embeddings
- **Ready For**: Production NGO deployment

## Files to Know (Edit Priority)

| Priority | File | Purpose |
|----------|-------|---------|
| 1 | `api_server.py` | Flask API (11 endpoints) |
| 2 | `electron-ui/renderer/app.js` | Frontend logic (25+ functions) |
| 3 | `electron-ui/index.html` | UI structure |
| 4 | `src/embedding/arcface_extractor.py` | ArcFace implementation |
| 5 | `src/detection/__init__.py` | Face detection |
| 6 | `src/reference/__init__.py` | Reference management |
| 7 | `CONTEXT.md` | Edit rules (read before ANY changes) |
| 8 | `test_e2e_pipeline.py` | E2E tests |

---

**This system is production-ready for NGO documentation verification workflows.**
**All critical bugs have been fixed.**
**All tests are passing.**
**Strict development rules are in place to prevent regressions.**

*System Prompt created: February 12, 2026*
*Complete understanding achieved: All documentation + code read*