# NGO Facial Image Analysis System - Project Structure

**Version**: 0.1.0  
**Last Updated**: February 11, 2026  
**Status**: ✅ Fully Functional

---

## Executive Summary

A complete, working facial recognition system for ethical NGO use. The pipeline correctly extracts, stores, and compares real 128-dimensional face embeddings with cosine similarity.

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
│  npm start            → Electron Desktop App (spawns Flask automatically)│
│  python api_server.py → Flask API Server :3000                          │
│  python gui/*.py      → Tkinter Standalone GUIs                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ELECTRON DESKTOP APP                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  main.js              → Spawns Python Flask server                      │
│  renderer/app.js      → Frontend JavaScript (HTTP API calls)            │
│  index.html           → Ultra minimal UI (black on white)               │
│                                                                          │
│  Flow: User → UI → fetch() → Flask API → ML Models → Results            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
│                          HTTP :3000 (REST API)
│
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     FLASK API SERVER (BACKEND)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  Endpoints:                                                              │
│  • GET  /api/health           → Status check                            │
│  • POST /api/detect           → Face detection                          │
│  • POST /api/extract          → Embedding extraction                    │
│  • POST /api/add-reference    → Add reference                           │
│  • GET  /api/references       → List references                         │
│  • POST /api/compare          → Similarity comparison                   │
│  • GET  /api/visualizations/<type> → Get visualization                  │
│  • POST /api/clear            → Clear session                           │
│  • GET  /api/status           → Debug server state                      │
│                                                                          │
│  In-Memory Session:                                                      │
│  • current_image, current_faces, current_embedding                      │
│  • references (in-memory list)                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CORE ML PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────┐                                                  │
│   │   FaceDetector   │  OpenCV DNN (Caffe model)                       │
│   │ Input: Image     │  Output: Bounding boxes (x, y, w, h)           │
│   └────────┬─────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────┐                                                   │
│   │ FaceNetEmbedding │  ResNet18 backbone → 128-dim vector             │
│   │ Input: Face ROI  │  L2 normalized (norm = 1.0)                     │
│   └────────┬─────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────┐                                                   │
│   │ Similarity       │  Cosine similarity → Confidence bands           │
│   │ Comparator       │  High (>0.8), Moderate (0.6-0.8), Low (0.4-0.6) │
│   └──────────────────┘                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     REFERENCE STORAGE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  ReferenceImageManager                                                   │
│  • Stores references in reference_images/embeddings.json                │
│  • Metadata: id, path, consent, timestamp                               │
│  • Embeddings: 128-dim vectors (REAL, not random!)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
face_recognition_npo/
├── start.sh                        # Interactive start script
├── api_server.py                   # Flask API server (BACKEND)
├── test_e2e_pipeline.py            # End-to-end test script
├── src/
│   ├── detection/__init__.py       # FaceDetector (OpenCV DNN)
│   ├── embedding/__init__.py       # FaceNetEmbeddingExtractor, SimilarityComparator
│   └── reference/__init__.py       # ReferenceImageManager, HumanReviewInterface
├── gui/
│   ├── facial_analysis_gui.py      # Apple-styled Tkinter GUI
│   └── user_friendly_gui.py        # Step-by-step wizard GUI
├── electron-ui/                    # Electron Desktop Application
│   ├── package.json                # NPM dependencies
│   ├── main.js                     # Electron main process
│   ├── preload.js                  # Context bridge
│   ├── index.html                  # Ultra minimal UI
│   ├── renderer/
│   │   └── app.js                  # Frontend JavaScript
│   └── styles/                     # CSS files
├── tests/                          # Unit tests
├── utils/webcam.py                 # WebcamCapture class
├── examples/                       # Usage examples
├── test_images/                    # 25+ test images
├── reference_images/               # Reference storage (embeddings.json)
│
├── deploy.prototxt.txt             # OpenCV DNN config
├── res10_300x300_ssd_iter_140000.caffemodel  # Face detection model
│
└── *.md                            # Documentation
```

---

## API Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| GET | `/api/health` | Health check | ✅ |
| POST | `/api/detect` | Detect faces in uploaded image | ✅ |
| POST | `/api/extract` | Extract embedding from detected face | ✅ |
| POST | `/api/add-reference` | Add reference image for comparison | ✅ |
| GET | `/api/references` | List all references | ✅ |
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
| Activations | NN layer activations | `/api/visualizations/activations` |
| Features | Feature map visualizations | `/api/visualizations/features` |
| Multi-Scale | Face at 5 scales | `/api/visualizations/multiscale` |
| Confidence | Quality metrics dashboard | `/api/visualizations/confidence` |
| Embedding | 128-dim vector viz | `/api/extract` |
| Similarity | Similarity matrix | `/api/compare` |
| Robustness | Noise robustness test | `/api/visualizations/robustness` |
| Biometric | Biometric capture quality | `/api/detect` |

---

## Test Results (February 11, 2026)

```bash
$ python test_e2e_pipeline.py

============================================================
END-TO-END FACE RECOGNITION PIPELINE TEST
============================================================

[TEST 1] Face Detection Pipeline         ✅ PASS
[TEST 2] Embedding Extraction Pipeline   ✅ PASS
[TEST 3] Reference Manager (REAL emb)    ✅ PASS
[TEST 4] Same Image Similarity           ✅ 100%
[TEST 5] Different Images Similarity     ✅ 98.72%
[TEST 6] Full Reference Comparison       ✅ PASS

ALL TESTS PASSED - Pipeline is working correctly!
```

---

## CRITICAL LESSONS LEARNED (Don't Repeat!)

### Lesson 1: Dynamic Array Sizes for Visualizations

**Problem Encountered**:
```
could not broadcast input array from shape (650,650,3) into shape (150,300,3)
```

**Root Cause**: Hardcoded array size `(150, 300)` in `visualize_similarity_matrix()`. When the number of references exceeded what fit in 150x300, the array assignment failed.

**The Fix**:
```python
# BEFORE (BROKEN):
output = np.zeros((150, 300, 3), dtype=np.uint8)
# ...
output[:n * cell_size, :n * cell_size] = matrix_colored  # Fails when n * cell_size > 300

# AFTER (FIXED):
output_size = max(150, n * cell_size)  # Dynamic based on n
output = np.zeros((output_size, output_size, 3), dtype=np.uint8)
output.fill(245)
resized = cv2.resize(matrix_colored, (n * cell_size, n * cell_size))
output[:resized.shape[0], :resized.shape[1]] = resized  # Safe assignment
```

**Rule**: Always use dynamic array sizes when the output depends on input count.

---

### Lesson 2: Reference Embeddings Must Be REAL

**Problem Encountered**: `ReferenceImageManager` was using random embeddings.

**Root Cause**:
```python
# BEFORE (BROKEN):
embedding = np.random.rand(128)  # FAKE! Random values
```

**The Fix**:
```python
# AFTER (FIXED):
if self.embedding_extractor is not None:
    detector_to_use = self.detector
    if detector_to_use is None and hasattr(self.embedding_extractor, 'detector'):
        detector_to_use = self.embedding_extractor.detector

    if detector_to_use is not None:
        faces = detector_to_use.detect_faces(image_array)
        if faces:
            x, y, w, h = faces[0]
            face_roi = image_array[y:y+h, x:x+w]
            embedding = self.embedding_extractor.extract_embedding(face_roi)
```

**Rule**: Never use random/placeholder values for embeddings. Extract real embeddings from actual images.

---

### Lesson 3: Clear Python Cache After Editing

**Problem Encountered**: Changes to `src/embedding/__init__.py` weren't picked up by the running Flask server.

**Root Cause**: Python's `.pyc` bytecode cache was holding old code.

**The Fix**:
```bash
# Always clear cache after editing source files:
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
```

**Rule**: Clear Python cache (`__pycache__`, `.pyc`) after editing source files before restarting the server.

---

### Lesson 4: Verify API with Test Script

**Rule**: Always test the API directly with a script before assuming frontend works:
```python
# Test API endpoints directly
r = requests.post('http://localhost:3000/api/detect', json={'image': base64_image})
r = requests.post('http://localhost:3000/api/extract', json={'face_id': 0})
r = requests.post('http://localhost:3000/api/add-reference', json={...})
r = requests.post('http://localhost:3000/api/compare', json={})
```

---

## Workflow (How to Use)

### Step 1: Choose Photo
- Upload an image file (JPG, PNG, BMP)

### Step 2: Find Faces
- Click "Find Faces" button
- AI detects faces and shows thumbnails

### Step 3: Create Signature
- Click "Create Signature" button
- Extracts 128-dim embedding from detected face

### Step 4: Add Reference
- Upload a reference image
- System extracts embedding and stores it

### Step 5: Compare
- Click "Compare" button
- Shows similarity scores with confidence bands

---

## Expected Similarity Scores

| Scenario | Expected Similarity |
|----------|-------------------|
| Same image | ~100% |
| Same person, different photo | 70-99% |
| Different person | 20-60% |

**Note**: Similarity varies based on:
- Image quality
- Lighting conditions
- Face angle
- Expression differences

---

## Roadmap

### Completed ✅ (v0.1.0)

**Core Pipeline**
- [x] Face detection with OpenCV DNN
- [x] 128-dim embedding extraction (ResNet18 backbone)
- [x] Cosine similarity comparison
- [x] Confidence bands (High >0.8, Moderate 0.6-0.8, Low 0.4-0.6, Insufficient <0.4)

**User Interfaces**
- [x] Electron desktop app (spawns Flask automatically)
- [x] Flask API server with 9 endpoints
- [x] Tkinter GUI
- [x] Ultra minimal UI (black on white, no icons)
- [x] Sticky terminal footer

**Reference Management**
- [x] Reference storage in JSON format
- [x] Metadata tracking (consent, source, timestamp)
- [x] Real embeddings (NOT random!)

**Visualizations**
- [x] 14 AI visualization types
- [x] Dynamic array sizes (fixed broadcast bug!)

**Testing & Tools**
- [x] End-to-end test script
- [x] Interactive start script

### In Progress

- [ ] **facenet_model.pb Integration**: Use existing TensorFlow FaceNet model

### Future Enhancements

- [ ] Advanced embeddings (ArcFace, CosFace)
- [ ] GPU acceleration with CUDA
- [ ] Batch processing API
- [ ] Cloud storage integration
- [ ] Mobile app
- [ ] WebSocket for real-time updates

---

## Dependencies

```
opencv-python>=4.8.0    # Face detection (DNN module)
numpy>=1.24.0           # Array operations
torch>=2.0.0            # PyTorch models
torchvision>=0.15.0     # Pre-trained models (ResNet18)
pillow>=10.0.0          # Image processing
flask                   # HTTP server
flask-cors              # CORS support
```

---

## Ethical Design

This system is built with ethical principles for NGO use:

1. **Consent-Based**: All images require documented consent
2. **Human Oversight**: No automated decisions - human review required
3. **Uncertainty Handling**: Confidence bands instead of binary decisions
4. **Privacy Protection**: Non-reversible embeddings only
5. **Documentation**: Complete audit trail of all operations

---

## Documentation Index

| File | Description |
|------|-------------|
| `README.md` | Main documentation, quick start, features |
| `PROJECT_STRUCTURE.md` | This file - complete structure and roadmap |
| `DEVELOPMENT_LOG.md` | Development history and session notes |
| `ARCHITECTURE.md` | Detailed system architecture |
| `ETHICAL_COMPLIANCE.md` | Ethical guidelines |
| `INSTALLATION.md` | Setup instructions |
| `USAGE.md` | Usage guide |
| `IMAGE_STORAGE.md` | Data storage |

---

## Quick Reference Commands

```bash
# Start the system
cd face_recognition_npo
./start.sh

# Run tests
python test_e2e_pipeline.py

# Clear Python cache (after editing source)
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Run Flask directly
source venv/bin/activate
python api_server.py

# Run Electron
cd electron-ui
npm start
```

---

*Project structure documentation updated: February 11, 2026*
*Document includes critical lessons learned to prevent recurring bugs*
