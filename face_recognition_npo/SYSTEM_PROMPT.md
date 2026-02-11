# System Prompt: Understanding the Face Recognition NPO Application

You are working on the Face Recognition NPO application - an ethical, consent-based facial recognition system for NGO documentation verification.

## Quick Start (5 Minutes)

1. **Start the system:** `./start.sh`
2. **Open browser:** http://localhost:3000
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
| `README.md` | Quick overview, features, quick start | 2 min |
| `CONTEXT.md` | Code review findings, AI workflow rules, common mistakes | 5 min |
| `DEVELOPMENT_LOG.md` | Session-by-session development history | 5 min |

### Level 2: Architecture Understanding

| File | Purpose | Time |
|------|---------|------|
| `ARCHITECTURE.md` | Complete system architecture, data flow, API reference | 10 min |
| `PROJECT_STRUCTURE.md` | File structure, directory layout, critical lessons learned | 5 min |

### Level 3: Technical Details

| File | Purpose | Time |
|------|---------|------|
| `ETHICAL_COMPLIANCE.md` | Privacy, consent, GDPR compliance | 5 min |
| `IMAGE_STORAGE.md` | How images are stored and processed | 3 min |
| `USAGE.md` | Detailed usage instructions | 5 min |
| `INSTALLATION.md` | Setup and installation guide | 3 min |

---

## Code Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ENTRY POINTS                             │
├─────────────────────────────────────────────────────────────┤
│  ./start.sh         → Interactive menu                       │
│  npm start          → Electron Desktop App                  │
│  python api_server.py → Flask API Server (:3000)            │
│  python gui/*.py    → Tkinter Standalone GUIs               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 ELECTRON DESKTOP APP                        │
├─────────────────────────────────────────────────────────────┤
│  main.js          → Spawns Flask server                     │
│  renderer/app.js  → Frontend UI (fetch API calls)           │
│  index.html       → Ultra minimal UI (black on white)       │
└─────────────────────────────────────────────────────────────┘
                              │
                         HTTP :3000
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  FLASK API SERVER                           │
├─────────────────────────────────────────────────────────────┤
│  Endpoints:                                                  │
│  • POST /api/detect      → Face detection                   │
│  • POST /api/extract     → Embedding extraction             │
│  • POST /api/add-reference → Add reference image            │
│  • POST /api/compare     → Compare embeddings               │
│  • GET  /api/visualizations/<type> → Get visualization      │
│  • GET  /api/visualizations/<type>/reference/<id> → Ref viz │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   CORE ML PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐                                       │
│  │  FaceDetector    │  OpenCV DNN (Caffe model)            │
│  │  Input: Image    │  Output: Bounding boxes (x,y,w,h)    │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │ FaceNetEmbedding │  ResNet18 → 128-dim vector           │
│  │  Input: Face ROI │  Output: L2-normalized embedding     │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │ Similarity       │  Cosine similarity (0-1)             │
│  │  Input: 2 emb.   │  Output: Score + confidence band     │
│  └──────────────────┘                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Source Code Structure

```
src/
├── detection/
│   └── __init__.py          # FaceDetector class (15 methods)
│                             # detect_faces, estimate_landmarks,
│                             # visualize_* methods (10 viz types)
│
├── embedding/
│   └── __init__.py          # FaceNetEmbeddingExtractor (12 methods)
│                             # extract_embedding, get_activations,
│                             # visualize_* methods (4 viz types)
│                             # SimilarityComparator class
│
└── reference/
    └── __init__.py          # ReferenceImageManager class
                              # HumanReviewInterface class

api_server.py                 # Flask API (9 endpoints)
utils/webcam.py               # WebcamCapture class
gui/*.py                      # Tkinter GUIs
```

---

## Critical Classes and Methods

### FaceDetector (`src/detection/__init__.py`)

**Detection Methods:**
- `detect_faces(image)` → List[Tuple[int, int, int, int]]
- `detect_faces_with_confidence(image)` → List[Tuple[Tuple, float]]
- `detect_eyes(face_image)` → List[Tuple[int, int, int, int]]
- `estimate_landmarks(face_image, face_box)` → Dict[str, Tuple[int, int]]
- `compute_alignment(face_image, landmarks)` → Dict[str, float]
- `compute_quality_metrics(face_image, face_box)` → Dict[str, float]

**Visualization Methods (10):**
- `visualize_detection()` → Bounding boxes
- `visualize_extraction()` → Face ROI
- `visualize_landmarks()` → 15 keypoints + regions
- `visualize_3d_mesh()` → 478-point mesh
- `visualize_alignment()` → Orientation indicator
- `visualize_saliency()` → Attention heatmap
- `visualize_biometric_capture()` → Biometric overview
- `visualize_multiscale()` → Multi-scale detection
- `visualize_quality()` → Quality metrics
- `visualize_confidence_levels()` → Confidence bands

### FaceNetEmbeddingExtractor (`src/embedding/__init__.py`)

**Core Methods:**
- `extract_embedding(face_image)` → np.ndarray (128,)
- `preprocess(face_image)` → torch.Tensor
- `get_activations(face_image)` → Dict[str, np.ndarray] (11 layers!)
- `extract_embeddings(face_images)` → List[np.ndarray]

**Embedding Visualization Methods (3):**
- `visualize_embedding()` → Bar chart of 128 values
- `visualize_similarity_matrix()` → Similarity grid
- `visualize_similarity_result()` → Similarity bar

**NN Visualization Methods (2):**
- `visualize_activations()` → CNN layer activations grid
- `visualize_feature_maps()` → Feature map visualization
- `test_robustness()` → Noise robustness test

### SimilarityComparator (`src/embedding/__init__.py`)

- `cosine_similarity(emb1, emb2)` → float
- `compare_embeddings(query, refs, ids)` → List[Tuple[str, float]]
- `get_confidence_band(similarity)` → str

**Confidence Bands:**
- High (>0.8): High confidence match
- Moderate (0.6-0.8): Moderate confidence
- Low (0.4-0.6): Low confidence
- Insufficient (<0.4): Not confident

---

## API Endpoints (14 total)

| Method | Endpoint | Returns |
|--------|----------|---------|
| GET | `/api/health` | Status |
| POST | `/api/detect` | Faces + thumbnails + viz |
| POST | `/api/extract` | 128-dim embedding + all viz |
| POST | `/api/add-reference` | Reference + embedding |
| GET | `/api/references` | List of references |
| POST | `/api/compare` | Similarity results + best match |
| GET | `/api/visualizations/<type>` | Query face viz |
| GET | `/api/visualizations/<type>/reference/<id>` | Reference face viz |
| POST | `/api/clear` | Clear session |
| GET | `/api/status` | Debug state |
| GET | `/api/quality` | Quality metrics |

---

## Visualization Types (14 total)

| Type | Source | Description |
|------|--------|-------------|
| `detection` | FaceDetector | Bounding boxes with confidence |
| `extraction` | FaceDetector | Face ROI extraction |
| `landmarks` | FaceDetector | 15 facial keypoints |
| `mesh3d` | FaceDetector | 478-point 3D mesh |
| `alignment` | FaceDetector | Pitch/yaw/roll orientation |
| `saliency` | FaceDetector | Attention visualization |
| `activations` | EmbeddingExtractor | CNN layer activations |
| `features` | EmbeddingExtractor | Feature map grid |
| `multiscale` | FaceDetector | Multi-scale detection |
| `confidence` | FaceDetector | Quality metrics overlay |
| `embedding` | EmbeddingExtractor | 128-dim bar chart |
| `similarity` | EmbeddingExtractor | Similarity result bar |
| `robustness` | EmbeddingExtractor | Noise robustness test |
| `biometric` | FaceDetector | Biometric capture overview |

---

## Testing

### E2E Tests
```bash
python test_e2e_pipeline.py
# Tests: Detection → Embedding → Reference Manager → Similarity → Full Pipeline
```

### Edge Case Tests
```bash
python test_edge_cases.py
# 11 tests covering boundary conditions
```

### Unit Tests
```bash
python -m pytest tests/
```

---

## Common Issues and Fixes

### Issue: "No data available" for visualizations
**Cause:** No face detected or embedding not extracted
**Fix:** Run detection then extraction before viewing visualizations

### Issue: High similarity between different people
**Cause:** All test images are of the same person (Kanye West)
**Fix:** Need test images of different people

### Issue: Visualizations show "Not available"
**Cause:** No face loaded in current session
**Fix:** Upload image, detect faces, create signature

### Issue: "Module has no attribute" errors
**Cause:** Python cache holding old code
**Fix:**
```bash
find . -type d -name "__pycache__" -exec rm -rf {}
find . -name "*.pyc" -delete
```

---

## Developer Workflow

### Before Making Changes
1. Read `CONTEXT.md` for rules
2. Read relevant source files
3. Check for duplicate code: `grep -n "def " <file>`
4. Understand the architecture from `ARCHITECTURE.md`

### During Changes
1. Follow the Developer Mindset Checklist
2. Run syntax check: `python -m py_compile <file>`
3. Test imports: `python -c "import module; print('OK')"`

### After Changes
1. Run E2E tests: `python test_e2e_pipeline.py`
2. Run edge cases: `python test_edge_cases.py`
3. Update documentation in `CONTEXT.md`

---

## Key Files for Understanding Specific Features

| To understand... | Read this file |
|-----------------|----------------|
| Overall system | `ARCHITECTURE.md` |
| Code review history | `DEVELOPMENT_LOG.md` |
| Common mistakes | `CONTEXT.md` (Common Mistakes section) |
| API endpoints | `ARCHITECTURE.md` (API Reference section) |
| Visualization types | `ARCHITECTURE.md` (14 AI Visualizations section) |
| Face detection | `src/detection/__init__.py` |
| Embedding extraction | `src/embedding/__init__.py` |
| Flask API | `api_server.py` |
| Frontend UI | `electron-ui/renderer/app.js` |

---

## Performance Notes

- Face detection: ~100ms per image
- Embedding extraction: ~50ms
- Similarity comparison: <1ms
- All visualizations: ~50-200ms

---

## Ethical Considerations

This system is designed for ethical NGO use:
- **Consent-based:** All images require documented consent
- **Human oversight:** No automated decisions
- **Confidence bands:** Shows uncertainty instead of binary decisions
- **Non-reversible:** Embeddings cannot reconstruct faces
- **Audit trail:** Complete logging of all operations

---

*System prompt created: February 11, 2026*
*Use this document to understand the application architecture and development workflow*
