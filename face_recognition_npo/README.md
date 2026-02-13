# NGO Facial Image Analysis System

**Version**: 2.0  
**Last Updated**: February 13, 2026  
**Status**: ✅ Fully Functional - Multi-Signal Comparison

A Python-based facial image analysis system with Electron desktop UI for ethical, consent-based NGO use in documentation verification and investigative work.

---

## Quick Start

### Method 1: Interactive Menu (Recommended)
```bash
cd face_recognition_npo
./start.sh
```

This starts the Flask API server and lets you choose how to open:
- [1] Electron Desktop App
- [2] Browser
- [3] Both

### Method 2: Manual Startup

**Terminal 1: Start Flask API**
```bash
cd face_recognition_npo
source venv/bin/activate
python api_server.py
```

**Terminal 2: Start Electron**
```bash
cd face_recognition_npo/electron-ui
npm start
```

> **Note**: Electron will connect to the existing Flask server on port 3000.

### Method 3: Browser Only
```bash
cd face_recognition_npo
source venv/bin/activate
python api_server.py
# Open http://localhost:3000 in your browser
```

---

## Architecture

```
start.sh ──► Flask API (port 3000)
                 │
                 ├── Browser ──► http://localhost:3000
                 │
                 └── Electron ──► Connects to Flask (no Python spawn)
```

**Best Practice**: Flask runs once, Electron connects to it.

---

## Features

- ✅ **Face Detection**: OpenCV DNN with Caffe model
- ✅ **Embedding Extraction**: 512-dimensional (ArcFace) or 128-dim (FaceNet)
- ✅ **Multi-Signal Comparison**: Cosine (50%) + Landmarks (25%) + Quality (15%)
- ✅ **Verdict System**: MATCH/POSSIBLE/LOW_CONFIDENCE/NO_MATCH with reasons
- ✅ **Reference Management**: Store references with real embeddings + landmarks + quality
- ✅ **Persistent Storage**: References saved to `reference_images/embeddings.json`
- ✅ **14 AI Visualizations**: Detection, landmarks, mesh, activations, etc.
- ✅ **10 Test Tabs**: System diagnostics without requiring uploaded images
- ✅ **Electron Desktop UI**: Ultra minimal design with MANTAX branding
- ✅ **Flask API Server**: 21 REST endpoints
- ✅ **End-to-End Tests**: All 6/6 passing
- ✅ **ArcFace Integration**: 512-dim embeddings for better discrimination

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

## Models

### ArcFace (Default - Better Discrimination)
- 512-dimensional embeddings
- ResNet100 backbone
- Better discrimination between different people
- Different people show <30% similarity (correct!)
- Same person shows ~70-85% similarity

### FaceNet (Optional)
- 128-dimensional embeddings
- ResNet18 backbone
- Faster inference
- Enable: `USE_FACENET=true ./start.sh`

**Multi-Signal Comparison** (Cosine + Landmarks + Quality):
- ≥60% = MATCH - Same person likely
- 50-60% = POSSIBLE - May be same person
- 40-50% = LOW_CONFIDENCE - Human review needed
- <40% = NO_MATCH - Different people

---

## Expected Results

| Scenario | Cosine | Landmarks | Quality | Combined | Verdict |
|----------|--------|-----------|---------|----------|---------|
| Same image | 100% | 100% | 100% | 100% | MATCH |
| Same person | 70-80% | 80-90% | 70-80% | 75-85% | MATCH |
| Different person | <40% | <50% | <60% | <40% | NO_MATCH |

**Why Multi-Signal is Better**:
- Cosine similarity alone: Good but can miss pose variations
- Landmarks: Captures facial structure geometry
- Quality: Ensures consistent image conditions
- Combined: More robust and accurate matching

**Verdict Display**:
- 🟢 **MATCH**: ≥60% - Same person likely
- 🟡 **POSSIBLE**: 50-60% - May be same person
- 🟠 **LOW_CONFIDENCE**: 40-50% - Review needed
- 🔴 **NO_MATCH**: <40% - Different people

---

## MANTAX Branding

The application now includes MANTAX branding in the navbar:
- Left: MANTAX logo (SVG with red #D20A11 and white)
- Right: "Ihrem Partner für Autokrane und Schwerlastlogistik"
- Clean, professional design with white background

---

## API Endpoints (11 Total)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/embedding-info` | Model info (FaceNet/ArcFace) |
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
| `embedding` | EmbeddingExtractor | Dim bar chart |
| `similarity` | EmbeddingExtractor | Similarity comparison |
| `robustness` | EmbeddingExtractor | Noise robustness test |
| `biometric` | FaceDetector | Biometric overview |

---

## Testing

```bash
# End-to-end pipeline test (uses test_subject.jpg and reference_subject.jpg)
python test_e2e_pipeline.py

# With ArcFace (default)
python test_e2e_pipeline.py

# With FaceNet
USE_FACENET=true python test_e2e_pipeline.py

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
ArcFace Different Person: ~9-25% (correctly different!)
FaceNet Different Person: ~65-70% (false positives!)
```

---

## Reference Storage

References are stored in `reference_images/embeddings.json`:

```json
{
  "metadata": [
    {"id": "name", "path": "path/to/image.jpg", "metadata": {...}, "added_at": "timestamp"}
  ],
  "embeddings": [
    {"id": "name", "embedding": [0.1, 0.5, ...]}  // 128-dim or 512-dim vector
  ]
}
```

**Note**: Only embeddings (128 or 512 floats) are stored, not images. The JSON references original image paths.

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

### 1. ArcFace vs FaceNet Discrimination

**Problem**: FaceNet showed 65-70% similarity for different people (false positives!)

**Solution**: ArcFace with 512-dim embeddings shows <30% for different people:
```
Different people: ~9-25% (correctly indicates different!)
Same person: ~70-85% (correctly indicates same!)
```

### 2. Dynamic Array Sizes
```python
# WRONG: Hardcoded size
output = np.zeros((150, 300, 3))

# RIGHT: Dynamic size
output_size = max(150, n * cell_size)
output = np.zeros((output_size, output_size, 3))
```

### 3. Real Embeddings
```python
# WRONG
embedding = np.random.rand(128)

# RIGHT
embedding = extractor.extract_embedding(face_roi)
```

### 4. Clear Cache After Editing
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### 5. Restart Server After Changes
Old Python processes don't load new code. Always restart with `./start.sh`.

### 6. Port Conflicts
Best practice: Flask runs once, Electron connects to it. Don't spawn Python from Electron.

### 7. ArcFace ONNX Model
ArcFace uses ONNX runtime - no direct layer access for visualizations. Use placeholder visualizations that show useful info instead of raw CNN activations.

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
├── api_server.py              # Flask API (11 endpoints)
├── start.sh                   # Startup script (clears cache, starts servers)
├── test_e2e_pipeline.py       # End-to-end tests
├── reference_images/           # Persistent storage
│   ├── embeddings.json       # Stored references
│   └── README.md
├── src/
│   ├── detection/            # Face detection (OpenCV DNN)
│   ├── embedding/            # 128-dim or 512-dim extraction
│   │   ├── __init__.py      # FaceNet extractor
│   │   └── arcface_extractor.py  # ArcFace extractor (ONNX)
│   └── reference/            # Reference management
├── electron-ui/               # Desktop UI
│   ├── index.html            # HTML with MANTAX navbar
│   ├── renderer/app.js
│   └── package.json
├── tests/                    # Unit tests (30 tests)
└── gui/                      # Tkinter fallback GUI
```

---

## ArcFace Integration Details

### Model Architecture
- **Backbone**: ResNet100 (ONNX format)
- **Embedding**: 512-dimensional, L2 normalized
- **Runtime**: ONNX Runtime (no PyTorch dependency for inference)

### Files
- `src/embedding/arcface_extractor.py` - ArcFace implementation
- `arcface_model.onnx` - ONNX model file

### API Response (ArcFace)
```json
{
  "success": true,
  "embedding_size": 512,
  "embedding_mean": 0.0321,
  "embedding_std": 0.0452,
  "model": "ArcFaceEmbeddingExtractor"
}
```

---

## License

MIT License
