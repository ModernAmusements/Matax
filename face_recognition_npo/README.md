# NGO Facial Image Analysis System

**Version**: 0.1.0
**Last Updated**: February 11, 2026
**Status**: ✅ Fully Functional

A Python-based facial image analysis system with Electron desktop UI for ethical, consent-based NGO use in documentation verification and investigative work.

---

## Quick Start

### Option 1: Interactive Menu (Recommended)
```bash
cd face_recognition_npo
./start.sh
```

### Option 2: Electron Desktop App
```bash
cd face_recognition_npo/electron-ui
npm install
npm start
```

### Option 3: Flask API Server
```bash
cd face_recognition_npo
source venv/bin/activate
python api_server.py
# Open http://localhost:3000 in browser
```

---

## Features

- ✅ **Face Detection**: OpenCV DNN with Caffe model
- ✅ **Embedding Extraction**: 128-dimensional vectors (ResNet18 backbone)
- ✅ **Similarity Comparison**: Cosine similarity with confidence bands
- ✅ **Reference Management**: Store references with real embeddings
- ✅ **Persistent Storage**: References saved to `reference_images/embeddings.json`
- ✅ **14 AI Visualizations**: Detection, landmarks, mesh, activations, etc.
- ✅ **Electron Desktop UI**: Ultra minimal design
- ✅ **Flask API Server**: 11 REST endpoints
- ✅ **End-to-End Tests**: All 6/6 passing
- ✅ **Unit Tests**: All 30/30 passing

---

## Usage Workflow

```
Step 1: Choose Photo     → Upload image
Step 2: Find Faces       → Click "Find Faces"
Step 3: Create Signature → Click "Create Signature" (EXTRACTS EMBEDDING)
Step 4: Add Reference    → Upload reference image
Step 5: Compare          → Click "Compare"
```

**For Visualizations**: After Step 3, click the visualization tabs (Embedding, Activations, Features, etc.)

---

## Expected Results

| Scenario | Similarity Score |
|----------|-----------------|
| Same image | ~100% |
| Same person, different photo | 85-99% |
| Different person | <70% |

**Confidence Bands**:
- 🟢 **Very High**: ≥99% - Likely same person
- 🟢 **High**: 95-98% - Possibly same person
- 🟡 **Moderate**: 85-94% - Human review recommended
- 🟡 **Low**: 70-84% - Human review required
- 🔴 **Insufficient**: <70% - Likely different people

---

## Architecture

```
User → Electron UI → Flask API → ML Pipeline → Results
                     │
                     ├── FaceDetector (OpenCV DNN)
                     ├── EmbeddingExtractor (ResNet18)
                     ├── SimilarityComparator
                     └── ReferenceManager
```

---

## API Endpoints (11 Total)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/detect` | Detect faces |
| POST | `/api/extract` | Extract embedding |
| POST | `/api/add-reference` | Add reference |
| GET | `/api/references` | List references |
| DELETE | `/api/references/<id>` | Remove reference |
| POST | `/api/compare` | Compare embeddings |
| GET | `/api/visualizations/<type>` | Get query visualization |
| GET | `/api/visualizations/<type>/reference/<id>` | Get ref visualization |
| POST | `/api/clear` | Clear session |
| GET | `/api/status` | Debug state |

---

## 14 Visualization Types

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
| `multiscale` | FaceDetector | Multi-scale detection |
| `confidence` | FaceDetector | Quality metrics |
| `embedding` | EmbeddingExtractor | 128-dim bar chart |
| `similarity` | EmbeddingExtractor | Similarity comparison |
| `robustness` | EmbeddingExtractor | Noise robustness test |
| `biometric` | FaceDetector | Biometric overview |

---

## Testing

```bash
# End-to-end pipeline test (uses test_subject.jpg and reference_subject.jpg)
python test_e2e_pipeline.py

# Unit tests
python -m pytest tests/

# Clear cache before testing
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
```

**Test Results**:
```
E2E Tests: 6/6 PASSED
Unit Tests: 30/30 PASSED
```

---

## Reference Storage

References are stored in `reference_images/embeddings.json`:

```json
{
  "metadata": [
    {"id": 0, "name": "subject.jpg", "path": "...", "added_at": "..."}
  ],
  "embeddings": [
    {"id": 0, "embedding": [0.1, 0.5, ...]}  // 128-dim vector
  ]
}
```

**Note**: Only embeddings (128 floats) are stored, not images. The JSON references original image paths.

---

## Documentation

| File | Description |
|------|-------------|
| README.md | This file |
| PROJECT_STRUCTURE.md | Complete architecture, lessons learned, roadmap |
| ARCHITECTURE.md | Detailed system design |
| DEVELOPMENT_LOG.md | Development history |
| CONTEXT.md | Critical rules for code edits |
| ETHICAL_COMPLIANCE.md | Ethical guidelines |

---

## Critical Rules for Code Edits

### Rule 1: Syntax Check
```bash
python -m py_compile <file>
```

### Rule 2: Check for Duplicate Code
```bash
grep -n "def " <file>
```

### Rule 3: Read Before Edit
Read at least 50 lines around the edit location.

### Rule 4: Function Preservation (JS)
```bash
grep -n "function " electron-ui/renderer/app.js | wc -l
```

### Rule 5: HTML-JS Cross-Check (CRITICAL!)
```bash
# Verify all HTML onclick/onchange handlers exist in app.js
for func in $(grep -E 'onclick=|onchange=' electron-ui/index.html | grep -oE '[a-zA-Z_]+(?=\()' | sort -u); do
    grep -qE "^function $func|^async function $func" electron-ui/renderer/app.js || echo "MISSING: $func"
done
```

### Rule 6: Fire-and-Forget
```javascript
// WRONG - blocks UI
await fetch(`${API_BASE}/clear`, { method: 'POST' });

// RIGHT - non-blocking
fetch(`${API_BASE}/clear`, { method: 'POST' }).catch(err => console.log(err));
```

### Rule 7: Restart After API Changes
```bash
./start.sh  # Clears cache, restarts API + Electron
```

---

## Lessons Learned (Don't Repeat!)

### 1. Dynamic Array Sizes
```python
# WRONG: Hardcoded size
output = np.zeros((150, 300, 3))

# RIGHT: Dynamic size
output_size = max(150, n * cell_size)
output = np.zeros((output_size, output_size, 3))
```

### 2. Real Embeddings
```python
# WRONG
embedding = np.random.rand(128)

# RIGHT
embedding = extractor.extract_embedding(face_roi)
```

### 3. Clear Cache After Editing
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### 4. Restart Server After Changes
Old Python processes don't load new code. Always restart with `./start.sh`.

---

## Ethical Guidelines

1. **Consent-Based**: All images must have lawful basis
2. **Human Oversight**: No automated decisions
3. **Uncertainty Handling**: Use confidence bands
4. **Privacy Protection**: Non-reversible embeddings
5. **Documentation**: Maintain audit trails

---

## File Structure

```
face_recognition_npo/
├── api_server.py           # Flask API (11 endpoints)
├── start.sh               # Startup script (clears cache, starts servers)
├── test_e2e_pipeline.py   # End-to-end tests
├── reference_images/      # Persistent storage
│   ├── embeddings.json    # Stored references
│   └── README.md
├── src/
│   ├── detection/         # Face detection (OpenCV DNN)
│   ├── embedding/         # 128-dim extraction (ResNet18)
│   └── reference/         # Reference management
├── electron-ui/           # Desktop UI
│   ├── index.html
│   ├── renderer/app.js
│   └── package.json
├── tests/                 # Unit tests (30 tests)
└── gui/                   # Tkinter fallback GUI
```

---

## License

MIT License
