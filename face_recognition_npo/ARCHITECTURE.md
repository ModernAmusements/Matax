# NGO Facial Image Analysis System - Architecture

**Version**: 0.1.0  
**Last Updated**: February 11, 2026  
**Status**: ✅ Fully Functional

---

## System Overview

This document describes the complete architecture of the NGO Facial Image Analysis System, an ethical, consent-based facial recognition system for NGO documentation verification.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SYSTEM ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        ENTRY POINTS                               │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │   ./start.sh          → Interactive menu                          │   │
│  │   npm start           → Electron Desktop App (spawns Flask)       │   │
│  │   python api_server.py → Flask API Server :3000                  │   │
│  │   python gui/*.py     → Tkinter Standalone GUIs                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    ELECTRON DESKTOP APP                           │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │   main.js              → Spawns Python Flask server              │   │
│  │   renderer/app.js      → Frontend JavaScript (HTTP API calls)    │   │
│  │   index.html           → Ultra minimal UI (black on white)        │   │
│  │   preload.js           → Context bridge                          │   │
│  │                                                                  │   │
│  │   Flow: User → UI → fetch() → Flask API → ML Models → Results   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                          HTTP :3000 (REST API)                          │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     FLASK API SERVER (BACKEND)                    │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │   Endpoints:                                                      │   │
│  │   • GET  /api/health                      → Status check         │   │
│  │   • POST /api/detect                      → Face detection       │   │
│  │   • POST /api/extract                     → Embedding extraction │   │
│  │   • POST /api/add-reference               → Add reference       │   │
│  │   • GET  /api/references                  → List references     │   │
│  │   • POST /api/compare                     → Similarity compare  │   │
│  │   • GET  /api/visualizations/<type>       → Get visualization   │   │
│  │   • POST /api/clear                       → Clear session       │   │
│  │   • GET  /api/status                      → Debug server state  │   │
│  │                                                                  │   │
│  │   In-Memory Session:                                              │   │
│  │   • current_image, current_faces, current_embedding              │   │
│  │   • references (in-memory list)                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      CORE ML PIPELINE                             │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │                                                                  │   │
│  │   ┌──────────────────┐                                           │   │
│  │   │   FaceDetector   │                                           │   │
│  │   │  (OpenCV DNN)    │                                           │   │
│  │   │                  │                                           │   │
│  │   │ Input: Image     │                                           │   │
│  │   │ Output: BBoxes   │                                           │   │
│  │   └────────┬─────────┘                                           │   │
│  │            │                                                     │   │
│  │            ▼                                                     │   │
│  │   ┌──────────────────┐                                           │   │
│  │   │ FaceNetEmbedding │                                           │   │
│  │   │   Extractor      │                                           │   │
│  │   │ (ResNet18)       │                                           │   │
│  │   │                  │                                           │   │
│  │   │ Input: Face ROI  │                                           │   │
│  │   │ Output: 128-dim  │                                           │   │
│  │   └────────┬─────────┘                                           │   │
│  │            │                                                     │   │
│  │            ▼                                                     │   │
│  │   ┌──────────────────┐                                           │   │
│  │   │ Similarity       │                                           │   │
│  │   │ Comparator       │                                           │   │
│  │   │                  │                                           │   │
│  │   │ Input: 2 emb.    │                                           │   │
│  │   │ Output: Score    │                                           │   │
│  │   └──────────────────┘                                           │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     REFERENCE STORAGE                             │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │   ReferenceImageManager                                          │   │
│  │   • Stores references in reference_images/embeddings.json        │   │
│  │   • Metadata: id, path, consent, timestamp                       │   │
│  │   • Embeddings: 128-dim vectors (REAL, not random!)              │   │
│  │                                                                  │   │
│  │   HumanReviewInterface                                           │   │
│  │   • Side-by-side comparison display                              │   │
│  │   • Review history tracking                                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Image     │ →  │   Detect    │ →  │   Extract   │ →  │  Compare    │
│   Upload    │    │   Faces     │    │   Embedding │    │  References │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Step 1: Image Upload
- Frontend reads image as Base64
- POST /api/detect with Base64 image

### Step 2: Face Detection
- OpenCV DNN with Caffe model (`res10_300x300_ssd_iter_140000.caffemodel`)
- Returns bounding boxes (x, y, w, h) for each face

### Step 3: Embedding Extraction
- ResNet18 backbone (torchvision pretrained)
- Custom head: FC(512→512) → BatchNorm → ReLU → Dropout → FC(512→128) → BatchNorm
- L2 normalization (norm = 1.0)
- Returns 128-dimensional numpy array

### Step 4: Comparison
- Cosine similarity: `dot(a, b) / (|a| * |b|)`
- Confidence bands:
  - High (>0.8): High confidence match
  - Moderate (0.6-0.8): Moderate confidence
  - Low (0.4-0.6): Low confidence
  - Insufficient (<0.4): Not confident

---

## Core Components

### 1. FaceDetector (`src/detection/__init__.py`)

**Purpose**: Detect faces in images using OpenCV DNN

**Model**: Caffe-based SSD detector
- Config: `deploy.prototxt.txt`
- Weights: `res10_300x300_ssd_iter_140000.caffemodel`

**Detection Methods**:
- `detect_faces(image)` → List[Tuple[int, int, int, int]]
- `detect_faces_with_confidence(image)` → List[Tuple[Tuple, float]]
- `detect_eyes(face_image)` → List[Tuple[int, int, int, int]]
- `estimate_landmarks(face_image, face_box)` → Dict with 15 keypoints + 468 full landmarks
- `compute_alignment(face_image, landmarks)` → Dict with pitch, yaw, roll
- `compute_quality_metrics(face_image, face_box)` → Dict[brightness, contrast, sharpness, eye_detection, centering, overall]

**Visualization Methods**:
- `visualize_detection(image, faces)` → Bounding boxes with confidence
- `visualize_extraction(image, faces)` → Face ROI extraction
- `visualize_landmarks(face_image, landmarks)` → 15 keypoints + facial regions
- `visualize_3d_mesh(face_image)` → 478-point mesh (MediaPipe or fallback)
- `visualize_alignment(face_image, landmarks, alignment)` → Orientation indicator
- `visualize_saliency(face_image)` → Attention/gradient visualization
- `visualize_biometric_capture(image, faces)` → Biometric capture overview
- `visualize_multiscale(face_image)` → Multi-scale detection
- `visualize_quality(face_image, face_box)` → Quality metrics overlay
- `visualize_confidence_levels(face_image, similarity)` → Confidence bands

### 2. FaceNetEmbeddingExtractor (`src/embedding/__init__.py`)

**Purpose**: Extract 128-dimensional face embeddings

**Architecture**:
- Backbone: torchvision ResNet18 (pretrained on ImageNet)
- Custom head: FC(512→512) → BatchNorm → ReLU → Dropout → FC(512→128) → BatchNorm
- L2 normalization

**Core Methods**:
- `extract_embedding(face_image)` → np.ndarray (128,)
- `preprocess(face_image)` → torch.Tensor
- `extract_embeddings(face_images)` → List[np.ndarray]
- `get_activations(face_image)` → Dict[str, np.ndarray] (11 layers!)

**Embedding Visualization Methods**:
- `visualize_embedding(embedding)` → (np.ndarray, Dict) - Bar chart of 128 values
- `visualize_similarity_matrix(query, references, ids)` → (np.ndarray, Dict)
- `visualize_similarity_result(query, ref, similarity)` → np.ndarray

**Neural Network Visualization Methods**:
- `visualize_activations(face_image, max_channels)` → CNN layer activations grid
- `visualize_feature_maps(face_image)` → Feature map visualization
- `test_robustness(face_image)` → (np.ndarray, Dict) - Noise robustness test

### 3. SimilarityComparator (`src/embedding/__init__.py`)

**Purpose**: Compare embeddings and return similarity scores

**Methods**:
- `cosine_similarity(embedding1, embedding2)` → float
- `compare_embeddings(query, references, ids)` → List[Tuple[str, float]]
- `get_confidence_band(similarity)` → str

### 4. ReferenceImageManager (`src/reference/__init__.py`)

**Purpose**: Manage reference images and their embeddings

**Features**:
- Stores references in `reference_images/embeddings.json`
- Extracts REAL embeddings (not random!)
- Metadata: id, path, consent info, timestamp

**Methods**:
- `__init__(reference_dir, embedding_extractor, detector)`
- `add_reference_image(image_path, reference_id, metadata)` → (bool, np.ndarray)
- `get_reference_embeddings()` → (List[np.ndarray], List[str])
- `list_references()` → List[dict]
- `remove_reference(reference_id)` → bool

### 5. HumanReviewInterface (`src/reference/__init__.py`)

**Purpose**: Human-in-the-loop review workflow

**Features**:
- Side-by-side comparison display
- Confidence-based decision making
- Review history tracking

---

## Model Files

| File | Size | Purpose |
|------|------|---------|
| `deploy.prototxt.txt` | 28KB | OpenCV DNN config |
| `res10_300x300_ssd_iter_140000.caffemodel` | 10MB | Face detection weights |
| `facenet_model.pb` | 298KB | TensorFlow FaceNet (NOT USED) |
| torchvision ResNet18 | ~44MB | PyTorch embedding backbone |

---

## API Reference

### POST /api/detect

**Request**:
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response**:
```json
{
  "success": true,
  "count": 2,
  "faces": [
    {"id": 0, "bbox": [x, y, w, h], "thumbnail": "base64..."}
  ],
  "visualizations": {
    "detection": "base64...",
    "extraction": "base64...",
    "biometric": "base64..."
  }
}
```

### POST /api/extract

**Request**:
```json
{
  "face_id": 0
}
```

**Response**:
```json
{
  "success": true,
  "embedding_size": 128,
  "embedding_mean": 0.0083,
  "embedding_std": 0.0880,
  "visualizations": {...},
  "visualization_data": {...}
}
```

### POST /api/add-reference

**Request**:
```json
{
  "image": "base64_encoded_image_string",
  "name": "reference_name"
}
```

**Response**:
```json
{
  "success": true,
  "reference": {
    "id": 0,
    "name": "reference_name",
    "embedding": [...128 values...],
    "thumbnail": "base64..."
  },
  "count": 1
}
```

### POST /api/compare

**Response**:
```json
{
  "success": true,
  "results": [
    {
      "id": 0,
      "name": "reference_name",
      "similarity": 0.8989,
      "confidence": "High confidence",
      "thumbnail": "base64..."
    }
  ],
  "best_match": {...},
  "similarity_viz": "base64...",
  "similarity_data": {...}
}
```

---

## CRITICAL IMPLEMENTATION NOTES

### Dynamic Array Sizes (Fixed Bug!)

When visualization output size depends on input count, use dynamic allocation:

```python
# WRONG - Hardcoded size causes broadcast error with many references
output = np.zeros((150, 300, 3), dtype=np.uint8)
output[:n * cell_size, :n * cell_size] = matrix_colored  # FAILS!

# RIGHT - Dynamic sizing based on input count
output_size = max(150, n * cell_size)
output = np.zeros((output_size, output_size, 3), dtype=np.uint8)
output.fill(245)
resized = cv2.resize(matrix_colored, (n * cell_size, n * cell_size))
output[:resized.shape[0], :resized.shape[1]] = resized  # Safe!
```

### Real Embeddings (Fixed Bug!)

Never use random values for embeddings:

```python
# WRONG - Random embeddings cause incorrect comparisons
embedding = np.random.rand(128)

# RIGHT - Extract real embeddings from actual images
if self.embedding_extractor is not None:
    faces = self.detector.detect_faces(image_array)
    if faces:
        x, y, w, h = faces[0]
        face_roi = image_array[y:y+h, x:x+w]
        embedding = self.embedding_extractor.extract_embedding(face_roi)
```

---

## Testing

### End-to-End Test
```bash
python test_e2e_pipeline.py
```

Tests:
1. Face Detection Pipeline
2. Embedding Extraction Pipeline
3. Reference Manager with Real Embeddings
4. Same Image Similarity (~100%)
5. Different Images Similarity (~98%)
6. Full Reference Comparison Pipeline

### Unit Tests
```bash
python -m unittest discover tests/
```

---

## Performance

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| Face Detection | O(n) | n = image pixels |
| Embedding Extraction | O(1) | Fixed network size |
| Similarity Comparison | O(m) | m = number of references |
| Embedding Storage | O(k) | k = number of stored refs |

---

## Roadmap

### Completed ✅ (v0.1.0)
- [x] Face detection with OpenCV DNN
- [x] 128-dim embedding extraction (ResNet18)
- [x] Cosine similarity comparison
- [x] Confidence bands (High/Moderate/Low/Insufficient)
- [x] Electron desktop UI
- [x] Flask API server
- [x] Tkinter GUI
- [x] Reference management with JSON storage
- [x] 14 AI visualizations
- [x] Fixed reference embeddings (was random, now real)
- [x] Fixed visualization array size (was hardcoded, now dynamic)
- [x] End-to-end test script
- [x] Interactive start menu

### In Progress
- [ ] facenet_model.pb integration (use TensorFlow FaceNet model)

### Future Enhancements
- [ ] FaceNet model integration (use existing facenet_model.pb)
- [ ] GPU acceleration with CUDA
- [ ] Advanced embedding architectures (ArcFace, CosFace)
- [ ] Batch processing API
- [ ] Cloud storage integration
- [ ] Mobile application support
- [ ] Advanced reporting features
- [ ] Model quantization for faster inference
- [ ] Distributed processing
- [ ] WebSocket support for real-time updates

---

## Complete API Reference

### All Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/detect` | Face detection |
| POST | `/api/extract` | Embedding extraction |
| POST | `/api/add-reference` | Add reference image |
| GET | `/api/references` | List references |
| POST | `/api/compare` | Compare embeddings |
| GET | `/api/visualizations/<type>` | Get visualization |
| POST | `/api/clear` | Clear session |
| GET | `/api/status` | Debug server state |

### Visualization Types (14 total)

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
| `embedding` | EmbeddingExtractor | 128-dim embedding bar chart |
| `similarity` | EmbeddingExtractor | Similarity result bar |
| `robustness` | EmbeddingExtractor | Noise robustness test |
| `biometric` | FaceDetector | Biometric capture overview |

---

## Edge Case Handling

The system handles various edge cases gracefully:

- **Empty/black images**: Returns 0 faces detected
- **Very small images (1x1)**: Extracts embedding (upsamples)
- **None inputs**: Returns placeholder/error image
- **NaN/Inf values**: Handled with safe defaults
- **Zero embeddings**: Returns 0.0 similarity
- **Empty reference list**: Returns empty results
- **Many references (50+)**: Dynamic array sizing
- **Long Unicode names**: Truncated in display
- **Zero-sized face boxes**: Returns error in quality metrics

Run `python test_edge_cases.py` to verify all edge cases.

---

## Testing

### End-to-End Tests
```bash
python test_e2e_pipeline.py
# Tests: Detection → Embedding → Reference Manager → Similarity → Full Pipeline
```

### Edge Case Tests
```bash
python test_edge_cases.py
# 11 edge case tests covering boundary conditions
```

### Unit Tests
```bash
python -m pytest tests/
```

---

## Ethical Design

This system is built with ethical principles:

1. **Consent-Based**: All images require documented consent
2. **Human Oversight**: No automated decisions - human review required
3. **Uncertainty Handling**: Confidence bands instead of binary decisions
4. **Privacy Protection**: Non-reversible embeddings only
5. **Documentation**: Complete audit trail of all operations

---

## Support

- **Documentation**: See all .md files in project root
- **Architecture**: PROJECT_STRUCTURE.md
- **Development Log**: DEVELOPMENT_LOG.md
- **Testing**: test_e2e_pipeline.py
- **Code Review**: CONTEXT.md

---

*Architecture documentation updated: February 11, 2026*
*Includes all 14 visualization types, edge case handling, and complete API reference*
