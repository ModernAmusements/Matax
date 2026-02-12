# System Prompt: Understanding the Face Recognition NPO Application

You are working on the Face Recognition NPO application - an ethical, consent-based facial recognition system for NGO documentation verification.

## Quick Start (5 Minutes)

1. **Start the system:** `./start.sh`
2. **Choose option:** [1] Electron Desktop App / [2] Browser / [3] Both
3. **Upload an image:** Click "Choose Photo"
4. **Find faces:** Click "Find Faces"
5. **Create signature:** Click "Create Signature"
6. **Add reference:** Upload a reference image
7. **Compare:** Click "Compare"

---

## Documentation Hierarchy (Read in Order)

### Level 1: Must Read (Start Here)

| File | Purpose | Time |
|------|---------|------|
| `README.md` | Quick overview, features, ArcFace integration | 2 min |
| `CONTEXT.md` | Code review findings, AI workflow rules, common mistakes | 5 min |
| `DEVELOPMENT_LOG.md` | Session-by-session development history | 5 min |

### Level 2: Architecture Understanding

| File | Purpose | Time |
|------|---------|------|
| `ARCHITECTURE.md` | Complete system architecture, ArcFace, API reference | 10 min |
| `PROJECT_STRUCTURE.md` | File structure, directory layout, critical lessons | 5 min |

### Level 3: Technical Details

| File | Purpose | Time |
|------|---------|------|
| `ETHICAL_COMPLIANCE.md` | Privacy, consent, ArcFace discrimination | 5 min |
| `IMAGE_STORAGE.md` | Embedding storage (not raw images!) | 3 min |
| `USAGE.md` | Quick usage instructions | 2 min |
| `INSTALLATION.md` | Setup and installation | 3 min |

---

## ArcFace vs FaceNet

**ArcFace (Default - Recommended)**
- 512-dimensional embeddings
- ONNX Runtime
- **Excellent discrimination**: Different people show <30% similarity
- Prevents false positives

**FaceNet (Legacy - Not Recommended)**
- 128-dimensional embeddings
- PyTorch
- **Poor discrimination**: Different people show ~65-70% similarity
- Causes false positives!

---

## Code Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ENTRY POINTS                             │
├─────────────────────────────────────────────────────────────┤
│  ./start.sh         → Interactive menu (clears cache)     │
│  python api_server.py → Flask API Server (:3000)           │
│  npm start          → Electron Desktop App (connects to Flask)│
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 ELECTRON DESKTOP APP                        │
├─────────────────────────────────────────────────────────────┤
│  main.js          → Connects to existing Flask server     │
│  renderer/app.js  → Frontend UI (fetch API calls)         │
│  index.html       → Ultra minimal UI with MANTAX navbar   │
└─────────────────────────────────────────────────────────────┘
                               │
                          HTTP :3000
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  FLASK API SERVER                           │
├─────────────────────────────────────────────────────────────┤
│  Endpoints:                                                 │
│  • GET  /api/health              → Status check           │
│  • GET  /api/embedding-info     → Model/dimension info   │
│  • POST /api/detect             → Face detection         │
│  • POST /api/extract            → Embedding extraction   │
│  • POST /api/add-reference      → Add reference image    │
│  • GET  /api/references         → List references        │
│  • DELETE /api/references/<id>  → Remove reference      │
│  • POST /api/compare            → Compare embeddings     │
│  • GET  /api/visualizations/<type> → Get visualization   │
│  • POST /api/clear             → Clear session         │
│  • GET  /api/status            → Debug state           │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   CORE ML PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐                                      │
│  │  FaceDetector    │  OpenCV DNN (Caffe model)          │
│  │  Input: Image    │  Output: Bounding boxes (x,y,w,h)  │
│  └────────┬─────────┘                                      │
│           │                                                 │
│           ▼                                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │           Embedding Extractor                       │    │
│  │  ┌─────────────────────┐  ┌─────────────────────┐ │    │
│  │  │  ArcFace (Default)  │  │  FaceNet (Option)  │ │    │
│  │  │  ONNX / ResNet100  │  │  PyTorch / ResNet18│ │    │
│  │  │  512-dimensional   │  │  128-dimensional   │ │    │
│  │  │  <30% for diff!  │  │  ~70% for diff    │ │    │
│  │  └─────────────────────┘  └─────────────────────┘ │    │
│  └─────────────────────────────┬──────────────────────┘    │
│                                │                             │
│                                ▼                             │
│  ┌──────────────────┐                                      │
│  │ Similarity       │  Cosine similarity → Confidence     │
│  │ Comparator      │  ArcFace: ≥70%, 45-70%, 30-45%, <30%│
│  └──────────────────┘                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ArcFace Confidence Thresholds

| Similarity | Confidence | Interpretation |
|------------|------------|----------------|
| ≥70% | Very High | Likely same person |
| 45-70% | High | Possibly same person |
| 30-45% | Moderate | Human review recommended |
| <30% | Insufficient | Likely different people |

---

## Source Code Structure

```
src/
├── detection/
│   └── __init__.py          # FaceDetector class
│                             # detect_faces, estimate_landmarks
│                             # visualize_* methods (10 viz types)
│
├── embedding/
│   ├── __init__.py          # FaceNetEmbeddingExtractor, SimilarityComparator
│   └── arcface_extractor.py # ArcFaceEmbeddingExtractor (512-dim, ONNX)
│                             # extract_embedding, get_activations
│                             # visualize_* methods (placeholder for ONNX)
│
└── reference/
    └── __init__.py          # ReferenceImageManager class
                              # HumanReviewInterface class

api_server.py                 # Flask API (11 endpoints)
utils/webcam.py              # WebcamCapture class
gui/*.py                     # Tkinter GUIs
```

---

## Critical Classes and Methods

### ArcFaceEmbeddingExtractor (`src/embedding/arcface_extractor.py`)

**Core Methods:**
- `extract_embedding(face_image)` → np.ndarray (512,)
- `preprocess(face_image)` → np.ndarray (112, 112)
- `get_activations(face_image)` → Dict[str, np.ndarray] (placeholder)
- `get_embedding_info()` → Dict with model info

**Visualization Methods:**
- `visualize_embedding()` → Bar chart of 512 values
- `visualize_similarity_matrix()` → Similarity grid
- `visualize_similarity_result()` → Similarity bar
- `test_robustness()` → Noise robustness test

**Note:** ArcFace ONNX doesn't expose internal layers, so activations visualization uses placeholder.

### FaceDetector (`src/detection/__init__.py`)

**Detection Methods:**
- `detect_faces(image)` → List[Tuple[int, int, int, int]]
- `detect_faces_with_confidence(image)` → List[Tuple[Tuple, float]]
- `estimate_landmarks(face_image, face_box)` → Dict with landmarks
- `compute_alignment(face_image, landmarks)` → Dict with pitch, yaw, roll
- `compute_quality_metrics(face_image, face_box)` → Dict metrics

**Visualization Methods (10):**
- `visualize_detection()` → Bounding boxes
- `visualize_extraction()` → Face ROI
- `visualize_landmarks()` → 15 keypoints
- `visualize_3d_mesh()` → 478-point mesh
- `visualize_alignment()` → Orientation indicator
- `visualize_saliency()` → Attention heatmap
- `visualize_biometric_capture()` → Biometric overview
- `visualize_multiscale()` → Multi-scale detection
- `visualize_quality()` → Quality metrics
- `visualize_confidence_levels()` → Confidence bands

### SimilarityComparator (`src/embedding/__init__.py`)

- `cosine_similarity(emb1, emb2)` → float
- `compare_embeddings(query, refs, ids)` → List[Tuple[str, float]]
- `get_confidence_band(similarity)` → str

---

## API Endpoints (11 total)

| Method | Endpoint | Returns |
|--------|----------|---------|
| GET | `/api/health` | Status |
| GET | `/api/embedding-info` | Model, dimension |
| POST | `/api/detect` | Faces + thumbnails + viz |
| POST | `/api/extract` | 512-dim or 128-dim embedding + viz |
| POST | `/api/add-reference` | Reference + embedding |
| GET | `/api/references` | List of references |
| DELETE | `/api/references/<id>` | Remove reference |
| POST | `/api/compare` | Similarity results + best match |
| GET | `/api/visualizations/<type>` | Query face viz |
| GET | `/api/visualizations/<type>/reference/<id>` | Ref viz |
| POST | `/api/clear` | Clear session |
| GET | `/api/status` | Debug state |

---

## 14 Visualization Types

| Type | Source | Description |
|------|--------|-------------|
| `detection` | FaceDetector | Bounding boxes with confidence |
| `extraction` | FaceDetector | Face ROI extraction |
| `landmarks` | FaceDetector | 15 facial keypoints |
| `mesh3d` | FaceDetector | 478-point 3D mesh |
| `alignment` | FaceDetector | Pitch/yaw/roll orientation |
| `saliency` | FaceDetector | Attention visualization |
| `activations` | EmbeddingExtractor | CNN activations (placeholder for ArcFace) |
| `features` | EmbeddingExtractor | Feature map grid |
| `multiscale` | FaceDetector | Multi-scale detection |
| `confidence` | FaceDetector | Quality metrics overlay |
| `embedding` | EmbeddingExtractor | 512-dim or 128-dim bar chart |
| `similarity` | EmbeddingExtractor | Similarity result bar |
| `robustness` | EmbeddingExtractor | Noise robustness test |
| `biometric` | FaceDetector | Biometric capture overview |

---

## Testing

### E2E Tests
```bash
python test_e2e_pipeline.py
# Tests: Detection → Embedding → Reference → Similarity → Full Pipeline
```

### API Tests
```bash
python test_api_endpoints.py
```

### Edge Case Tests
```bash
python test_edge_cases.py
# 11 tests covering boundary conditions
```

---

## Common Issues and Fixes

### Issue: Different people show high similarity
**Cause:** Using FaceNet (shows ~65-70% for different people)
**Fix:** Use ArcFace (shows <30% for different people)

### Issue: "No data available" for visualizations
**Cause:** No face detected or embedding not extracted
**Fix:** Run detection then extraction before viewing visualizations

### Issue: Visualizations show placeholder
**Cause:** ArcFace ONNX doesn't expose internal layers
**Fix:** Use placeholder visualizations that show useful info

### Issue: "Module has no attribute" errors
**Cause:** Python cache holding old code
**Fix:**
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
./start.sh
```

---

## Developer Workflow

### Before Making Changes
1. Read `CONTEXT.md` for rules
2. Read relevant source files
3. Check for duplicate code: `grep -n "def " <file>`
4. Understand architecture from `ARCHITECTURE.md`

### During Changes
1. Follow the Developer Mindset Checklist
2. Run syntax check: `python -m py_compile <file>`
3. Test imports: `python -c "import module; print('OK')"`

### After Changes
1. Run E2E tests: `python test_e2e_pipeline.py`
2. Run API tests: `python test_api_endpoints.py`
3. Update documentation

---

## Key Files for Understanding Specific Features

| To understand... | Read this file |
|-----------------|----------------|
| Overall system | `ARCHITECTURE.md` |
| ArcFace integration | `DEVELOPMENT_LOG.md` (Feb 12 section) |
| Code review history | `DEVELOPMENT_LOG.md` |
| Common mistakes | `CONTEXT.md` (Common Mistakes section) |
| API endpoints | `ARCHITECTURE.md` (API Reference section) |
| Visualization types | `ARCHITECTURE.md` (14 AI Visualizations) |
| ArcFace discrimination | `ETHICAL_COMPLIANCE.md` |
| Face detection | `src/detection/__init__.py` |
| ArcFace extractor | `src/embedding/arcface_extractor.py` |
| Flask API | `api_server.py` |
| Frontend UI | `electron-ui/renderer/app.js` |

---

## Performance Notes

- Face detection: ~100ms per image
- ArcFace embedding: ~50ms (ONNX)
- FaceNet embedding: ~100ms (PyTorch)
- Similarity comparison: <1ms
- All visualizations: ~50-200ms

---

## MANTAX Branding

The UI includes MANTAX branding:
- **Navbar**: White background, fixed at top
- **Logo**: SVG with red (#D20A11) and white
- **Tagline**: "Ihrem Partner für Autokrane und Schwerlastlogistik"

---

## Ethical Considerations

This system is designed for ethical NGO use:

- **Consent-based:** All images require documented consent
- **Human oversight:** No automated decisions
- **Confidence bands:** Shows uncertainty instead of binary decisions
- **Non-reversible:** 512-dim embeddings cannot reconstruct faces
- **ArcFace discrimination:** <30% for different people prevents false positives
- **Audit trail:** Complete logging of all operations

---

*System prompt updated: February 12, 2026*
*Includes ArcFace integration, 512-dim embeddings, ONNX model, and MANTAX branding*
