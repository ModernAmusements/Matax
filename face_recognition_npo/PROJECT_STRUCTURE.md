# NGO Facial Image Analysis System - Project Structure

**Version**: 0.3.0
**Last Updated**: February 12, 2026
**Status**: ✅ Fully Functional - ArcFace Enabled

---

## Executive Summary

A complete, working facial recognition system for ethical NGO use. The system uses ArcFace 512-dimensional embeddings for excellent discrimination between different people.

**Key Achievement**: ArcFace shows <30% similarity for different people (correct!), vs FaceNet's 65-70% (false positives!).

## Quick Start

```bash
cd face_recognition_npo
./start.sh
# Select option 1 (Flask API + Browser) or 2 (Electron Desktop)
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENTRY POINTS                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  ./start.sh           → Interactive menu                                │
│  python api_server.py → Flask API Server :3000                          │
│  npm start           → Electron Desktop App (connects to Flask)         │
│  python gui/*.py      → Tkinter Standalone GUIs                       │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ELECTRON DESKTOP APP                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  main.js              → Connects to existing Flask server              │
│  renderer/app.js      → Frontend JavaScript (HTTP API calls)           │
│  index.html           → Ultra minimal UI with MANTAX navbar             │
│                                                                          │
│  Flow: User → UI → fetch() → Flask API → ML Models → Results          │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                           HTTP :3000 (REST API)
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     FLASK API SERVER (BACKEND)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  Endpoints:                                                              │
│  • GET  /api/health           → Status check                           │
│  • GET  /api/embedding-info   → Model info (ArcFace/FaceNet)           │
│  • POST /api/detect           → Face detection                         │
│  • POST /api/extract          → Embedding extraction                   │
│  • POST /api/add-reference    → Add reference                          │
│  • GET  /api/references      → List references                        │
│  • DELETE /api/references/<id> → Remove reference                      │
│  • POST /api/compare          → Similarity comparison                   │
│  • GET  /api/visualizations/<type> → Get visualization                 │
│  • POST /api/clear           → Clear session                           │
│  • GET  /api/status          → Debug server state                      │
│                                                                          │
│  In-Memory Session:                                                      │
│  • current_image, current_faces, current_embedding                       │
│  • references (in-memory list)                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CORE ML PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────┐                                                   │
│   │   FaceDetector   │  OpenCV DNN (Caffe model)                        │
│   │ Input: Image     │  Output: Bounding boxes (x, y, w, h)              │
│   └────────┬─────────┘                                                    │
│            │                                                              │
│            ▼                                                              │
│   ┌─────────────────────────────────────────────────────┐                │
│   │           Embedding Extractor                         │                │
│   │  ┌─────────────────────┐  ┌─────────────────────┐   │                │
│   │  │  ArcFace (Default)  │  │  FaceNet (Option)  │   │                │
│   │  │  ONNX / ResNet100   │  │  PyTorch / ResNet18│   │                │
│   │  │  512-dimensional    │  │  128-dimensional   │   │                │
│   │  │  <30% for diff!    │  │  ~70% for diff    │   │                │
│   │  └─────────────────────┘  └─────────────────────┘   │                │
│   │                                                      │                │
│   │ Input: Face ROI                                       │                │
│   │ Output: 512-dim or 128-dim embedding                 │                │
│   └────────────────────────────┬──────────────────────────┘                │
│                                │                                            │
│                                ▼                                            │
│   ┌──────────────────┐                                                   │
│   │ Similarity       │  Cosine similarity → Confidence bands              │
│   │ Comparator       │  ArcFace: ≥70%, 45-70%, 30-45%, <30%             │
│   └──────────────────┘                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     REFERENCE STORAGE                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  ReferenceImageManager                                                   │
│  • Stores references in reference_images/embeddings.json                 │
│  • Metadata: id, path, consent, timestamp                              │
│  • Embeddings: 512-dim (ArcFace) or 128-dim (FaceNet)                  │
│  • Persistence: Auto-saves to JSON on add/remove                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
face_recognition_npo/
├── start.sh                        # Interactive start script
├── api_server.py                   # Flask API server (BACKEND)
├── test_e2e_pipeline.py            # End-to-end test script
├── test_api_endpoints.py           # API verification script
├── src/
│   ├── detection/__init__.py       # FaceDetector (OpenCV DNN)
│   ├── embedding/
│   │   ├── __init__.py            # FaceNetEmbeddingExtractor, SimilarityComparator
│   │   └── arcface_extractor.py   # ArcFaceEmbeddingExtractor (ONNX)
│   └── reference/__init__.py       # ReferenceImageManager, HumanReviewInterface
├── gui/
│   ├── facial_analysis_gui.py       # Apple-styled Tkinter GUI
│   └── user_friendly_gui.py        # Step-by-step wizard GUI
├── electron-ui/                     # Electron Desktop Application
│   ├── package.json                # NPM dependencies
│   ├── main.js                     # Electron main process
│   ├── preload.js                  # Context bridge
│   ├── index.html                  # Ultra minimal UI with MANTAX navbar
│   ├── renderer/
│   │   └── app.js                 # Frontend JavaScript
│   └── styles/                     # CSS files
├── tests/                           # Unit tests
├── utils/webcam.py                  # WebcamCapture class
├── examples/                        # Usage examples
├── test_images/                     # 25+ test images
├── reference_images/                # Reference storage
│   ├── embeddings.json             # Stored references
│   └── README.md
│
├── deploy.prototxt.txt             # OpenCV DNN config
├── res10_300x300_ssd_iter_140000.caffemodel  # Face detection model
├── arcface_model.onnx              # ArcFace ONNX model (117MB)
│
└── *.md                            # Documentation
```

---

## API Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| GET | `/api/health` | Health check | ✅ |
| GET | `/api/embedding-info` | Model/dimension info | ✅ |
| POST | `/api/detect` | Detect faces in uploaded image | ✅ |
| POST | `/api/extract` | Extract embedding from detected face | ✅ |
| POST | `/api/add-reference` | Add reference image for comparison | ✅ |
| GET | `/api/references` | List all references | ✅ |
| DELETE | `/api/references/<id>` | Remove reference | ✅ |
| POST | `/api/compare` | Compare query embedding with references | ✅ |
| GET | `/api/visualizations/<type>` | Get specific AI visualization | ✅ |
| POST | `/api/clear` | Clear all session data | ✅ |
| GET | `/api/status` | Debug server state | ✅ |

---

## 14 AI Visualizations

| Visualization | Description | API Endpoint |
|--------------|-------------|--------------|
| Detection | Bounding boxes with confidence scores | `/api/detect` |
| Extraction | 160x160 face crop preview | `/api/detect` |
| Landmarks | Facial feature points | `/api/visualizations/landmarks` |
| 3D Mesh | 3D wireframe overlay | `/api/visualizations/mesh3d` |
| Alignment | Face alignment visualization | `/api/visualizations/alignment` |
| Attention | Saliency/attention heatmap | `/api/visualizations/saliency` |
| Activations | NN layer activations (placeholder for ArcFace) | `/api/visualizations/activations` |
| Features | Feature map visualizations | `/api/visualizations/features` |
| Multi-Scale | Face at 5 scales | `/api/visualizations/multiscale` |
| Confidence | Quality metrics dashboard | `/api/visualizations/confidence` |
| Embedding | 512-dim or 128-dim vector viz | `/api/extract` |
| Similarity | Similarity matrix | `/api/compare` |
| Robustness | Noise robustness test | `/api/visualizations/robustness` |
| Biometric | Biometric capture quality | `/api/detect` |

---

## Test Results (February 12, 2026)

```bash
$ python test_e2e_pipeline.py

============================================================
END-TO-END FACE RECOGNITION PIPELINE TEST
============================================================

[TEST 1] Face Detection Pipeline         ✅ PASS
[TEST 2] Embedding Extraction Pipeline   ✅ PASS
[TEST 3] Reference Manager (REAL emb)   ✅ PASS
[TEST 4] Same Image Similarity          ✅ 100%
[TEST 5] Different Images Similarity    ✅ ArcFace: ~9-25% (correct!)
[TEST 6] Full Reference Comparison      ✅ PASS

ALL TESTS PASSED - Pipeline is working correctly!
```

---

## ArcFace vs FaceNet Comparison

| Metric | ArcFace (Default) | FaceNet (Optional) |
|--------|-------------------|-------------------|
| **Dimension** | 512 | 128 |
| **Backbone** | ResNet100 (ONNX) | ResNet18 (PyTorch) |
| **Discrimination** | Excellent | Poor |
| **Same Person** | ~70-85% | ~85-99% |
| **Different Person** | <30% | ~65-70% |
| **False Positive Risk** | Low | High |
| **Inference Speed** | Fast (ONNX) | Slower (PyTorch) |

**Why ArcFace is Default**:
- FaceNet showed 65-70% similarity for different people (false positives!)
- ArcFace shows <30% for different people (correctly indicates different!)
- Much safer for NGO use cases

---

## MANTAX Branding

The Electron UI includes MANTAX branding:
- **Navbar**: White background with subtle border
- **Logo**: SVG with red (#D20A11) and white colors
- **Tagline**: "Ihrem Partner für Autokrane und Schwerlastlogistik" (right side)
- **Position**: Fixed at top, visible on all pages

---

## CRITICAL LESSONS LEARNED (Don't Repeat!)

### Lesson 1: ArcFace vs FaceNet Discrimination

**Problem Encountered**:
FaceNet 128-dim embeddings showed 65-70% similarity for different people - this caused false positives!

**The Solution**:
ArcFace 512-dim embeddings show <30% for different people:
```
Different people: ~9-25% (correctly indicates different!)
Same person: ~70-85% (correctly indicates same!)
```

**Rule**: Use ArcFace for better discrimination in NGO use cases.

---

### Lesson 2: Dynamic Array Sizes for Visualizations

**Problem Encountered**:
```
could not broadcast input array from shape (650,650,3) into shape (150,300,3)
```

**Root Cause**: Hardcoded array size `(150, 300)` in `visualize_similarity_matrix()`.

**The Fix**:
```python
# BEFORE (BROKEN):
output = np.zeros((150, 300, 3), dtype=np.uint8)
# ...
output[:n * cell_size, :n * cell_size] = matrix_colored

# AFTER (FIXED):
output_size = max(150, n * cell_size)
output = np.zeros((output_size, output_size, 3), dtype=np.uint8)
output.fill(245)
resized = cv2.resize(matrix_colored, (n * cell_size, n * cell_size))
output[:resized.shape[0], :resized.shape[1]] = resized
```

**Rule**: Always use dynamic array sizes when output depends on input count.

---

### Lesson 3: Reference Embeddings Must Be REAL

**Problem Encountered**: `ReferenceImageManager` was using random embeddings.

**Root Cause**:
```python
# BEFORE (BROKEN):
embedding = np.random.rand(128)  # FAKE!
```

**The Fix**:
```python
# AFTER (FIXED):
if self.embedding_extractor is not None:
    detector_to_use = self.detector
    if faces:
        x, y, w, h = faces[0]
        face_roi = image_array[y:y+h, x:x+w]
        embedding = self.embedding_extractor.extract_embedding(face_roi)
```

**Rule**: Never use random/placeholder values for embeddings.

---

### Lesson 4: Clear Python Cache After Editing

**Problem Encountered**: Changes to source files weren't picked up.

**Root Cause**: Python's `.pyc` bytecode cache.

**The Fix**:
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
```

**Rule**: Clear Python cache after editing source files.

---

### Lesson 5: Verify API with Test Script

**Rule**: Always test API directly before debugging frontend:
```python
r = requests.post('http://localhost:3000/api/detect', json={'image': base64_image})
r = requests.post('http://localhost:3000/api/extract', json={'face_id': 0})
r = requests.post('http://localhost:3000/api/add-reference', json={...})
r = requests.post('http://localhost:3000/api/compare', json={})
r = requests.get('http://localhost:3000/api/embedding-info')
```

---

### Lesson 6: ArcFace ONNX Model Limitations

**Problem**: ArcFace ONNX doesn't expose internal layers like PyTorch.

**Solution**: Use placeholder visualizations that show useful info:
- For activations: Show embedding channel groups instead of raw CNN
- For feature maps: Show pre-computed visualizations
- Always document this limitation

---

## Workflow (How to Use)

### Step 1: Choose Photo
- Upload an image file (JPG, PNG, BMP)

### Step 2: Find Faces
- Click "Find Faces" button
- AI detects faces and shows thumbnails

### Step 3: Create Signature
- Click "Create Signature" button
- Extracts 512-dim (ArcFace) or 128-dim (FaceNet) embedding

### Step 4: Add Reference
- Upload a reference image
- System extracts embedding and stores it

### Step 5: Compare
- Click "Compare" button
- Shows similarity scores with confidence bands

---

## Expected Similarity Scores (ArcFace)

| Scenario | Similarity | Confidence |
|----------|------------|------------|
| Same image | ~100% | Very High |
| Same person | ~70-85% | High |
| Different person | <30% | Insufficient |

**Note**: Similarity varies based on:
- Image quality
- Lighting conditions
- Face angle
- Expression differences

---

## ArcFace Thresholds

| Threshold | Confidence | Interpretation |
|-----------|------------|----------------|
| ≥70% | Very High | Likely same person |
| 45-70% | High | Possibly same person |
| 30-45% | Moderate | Human review recommended |
| <30% | Insufficient | Likely different people |

---

## Roadmap

### Completed ✅ (v0.3.0)

**Core Pipeline**
- [x] Face detection with OpenCV DNN
- [x] 512-dim embedding extraction (ArcFace ONNX)
- [x] 128-dim embedding extraction (FaceNet PyTorch) - legacy
- [x] Cosine similarity comparison
- [x] Confidence bands (ArcFace: ≥70%, 45-70%, 30-45%, <30%)
- [x] Auto-model selection (ArcFace if available)

**User Interfaces**
- [x] Electron desktop app (connects to Flask)
- [x] Flask API server with 11 endpoints
- [x] Tkinter GUI
- [x] Ultra minimal UI (black on white, no icons)
- [x] Sticky terminal footer
- [x] MANTAX navbar branding

**Reference Management**
- [x] Reference storage in JSON format
- [x] Metadata tracking (consent, source, timestamp)
- [x] Real embeddings (NOT random!)
- [x] Persistence across restarts

**Visualizations**
- [x] 14 AI visualization types
- [x] Dynamic array sizes (fixed broadcast bug!)
- [x] ArcFace placeholder visualizations

**Testing & Tools**
- [x] End-to-end test script
- [x] Interactive start script
- [x] API verification script

### In Progress
- [ ] GPU acceleration for ONNX Runtime

### Future Enhancements
- [ ] Batch processing API
- [ ] Cloud storage integration
- [ ] Mobile app
- [ ] WebSocket for real-time updates

---

## Dependencies

```
opencv-python>=4.8.0       # Face detection (DNN module)
numpy>=1.24.0              # Array operations
onnxruntime>=1.15.0        # ONNX Runtime for ArcFace
torch>=2.0.0              # PyTorch models (FaceNet, optional)
torchvision>=0.15.0        # Pre-trained models (ResNet18)
pillow>=10.0.0             # Image processing
flask                      # HTTP server
flask-cors                 # CORS support
```

---

## Ethical Design

This system is built with ethical principles for NGO use:

1. **Consent-Based**: All images require documented consent
2. **Human Oversight**: No automated decisions - human review required
3. **Uncertainty Handling**: Confidence bands instead of binary decisions
4. **Privacy Protection**: Non-reversible embeddings only
5. **Documentation**: Complete audit trail of all operations

**Key Ethical Features**:
- ArcFace <30% threshold for "different people" prevents false positives
- Human review required for all moderate confidence results
- Complete audit trail of all comparisons

---

## Documentation Index

| File | Description |
|------|-------------|
| README.md | Main documentation, quick start, features |
| PROJECT_STRUCTURE.md | This file - complete structure and roadmap |
| DEVELOPMENT_LOG.md | Development history and session notes |
| ARCHITECTURE.md | Detailed system architecture |
| ETHICAL_COMPLIANCE.md | Ethical guidelines |
| CONTEXT.md | Critical rules for code edits |

---

## Quick Reference Commands

```bash
# Start the system
cd face_recognition_npo
./start.sh

# Run tests
python test_e2e_pipeline.py

# Verify API endpoints
python test_api_endpoints.py

# Clear Python cache (after editing source)
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Run Flask directly
source venv/bin/activate
python api_server.py

# Run Electron
cd electron-ui
npm start

# Check which model is active
curl http://localhost:3000/api/embedding-info
```

---

## Model Information

| Model | File | Dimension | Runtime | Discrimination |
|-------|------|-----------|---------|----------------|
| ArcFace | `arcface_model.onnx` | 512 | ONNX | Excellent (<30% for diff) |
| FaceNet | torchvision ResNet18 | 128 | PyTorch | Poor (~70% for diff) |

---

*Project structure documentation updated: February 12, 2026*
*Includes ArcFace integration, 512-dim embeddings, MANTAX branding, and critical lessons learned*
