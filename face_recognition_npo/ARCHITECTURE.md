# NGO Facial Image Analysis System - Architecture

**Version**: 0.3.0  
**Last Updated**: February 12, 2026  
**Status**: ✅ Fully Functional - ArcFace Enabled

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
│  │   python api_server.py → Flask API Server :3000                    │   │
│  │   npm start           → Electron Desktop App (connects to Flask)  │   │
│  │   python gui/*.py     → Tkinter Standalone GUIs                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    ELECTRON DESKTOP APP                           │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │   main.js              → Connects to existing Flask server      │   │
│  │   renderer/app.js      → Frontend JavaScript (HTTP API calls)   │   │
│  │   index.html           → Ultra minimal UI with MANTAX navbar      │   │
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
│  │   • GET  /api/embedding-info              → Model info          │   │
│  │   • POST /api/detect                      → Face detection       │   │
│  │   • POST /api/extract                     → Embedding extraction │   │
│  │   • POST /api/add-reference               → Add reference       │   │
│  │   • GET  /api/references                  → List references     │   │
│  │   • DELETE /api/references/<id>           → Remove reference    │   │
│  │   • POST /api/compare                     → Similarity compare  │   │
│  │   • GET  /api/visualizations/<type>       → Get visualization  │   │
│  │   • POST /api/clear                       → Clear session       │   │
│  │   • GET  /api/status                      → Debug server state │   │
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
│  │   ┌─────────────────────────────────────────────────────┐        │   │
│  │   │           Embedding Extractor                        │        │   │
│  │   │  ┌─────────────────────┐  ┌─────────────────────┐   │        │   │
│  │   │  │  ArcFace (Default)  │  │  FaceNet (Option)  │   │        │   │
│  │   │  │  ONNX / ResNet100   │  │  PyTorch / ResNet18│   │        │   │
│  │   │  │  512-dimensional    │  │  128-dimensional   │   │        │   │
│  │   │  └─────────────────────┘  └─────────────────────┘   │        │   │
│  │   │                                                      │        │   │
│  │   │ Input: Face ROI                                      │        │   │
│  │   │ Output: 512-dim or 128-dim embedding                 │        │   │
│  │   └────────────────────────────┬──────────────────────────┘        │   │
│  │                                │                                     │   │
│  │                                ▼                                     │   │
│  │   ┌──────────────────┐                                           │   │
│  │   │ Similarity       │                                           │   │
│  │   │ Comparator       │                                           │   │
│  │   │                  │                                           │   │
│  │   │ Input: 2 emb.   │                                           │   │
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
│  │   • Metadata: id, path, consent, timestamp                     │   │
│  │   • Embeddings: 512-dim (ArcFace) or 128-dim (FaceNet)        │   │
│  │   • Persistence: Saved to JSON on add/remove                     │   │
│  │                                                                  │   │
│  │   HumanReviewInterface                                           │   │
│  │   • Side-by-side comparison display                             │   │
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

### Step 3: Embedding Extraction (ArcFace vs FaceNet)

#### ArcFace (Default - Recommended)
- **Model**: ONNX format ResNet100
- **Dimension**: 512-dimensional
- **L2 Normalized**: Yes
- **Discrimination**: Excellent - different people show <30% similarity
- **Thresholds**:
  - ≥70% = Very High - Likely same person
  - 45-70% = High - Possibly same person
  - 30-45% = Moderate - Human review recommended
  - <30% = Insufficient - Likely different people

#### FaceNet (Optional - Legacy)
- **Model**: PyTorch ResNet18
- **Dimension**: 128-dimensional
- **L2 Normalized**: Yes
- **Discrimination**: Poor - different people show ~65-70% similarity
- **Note**: Use `USE_FACENET=true` to enable

### Step 4: Comparison
- Cosine similarity: `dot(a, b) / (|a| * |b|)`
- Confidence bands (ArcFace):
  - Very High (>0.7): High confidence match
  - High (0.45-0.7): Moderate confidence
  - Moderate (0.3-0.45): Low confidence, human review required
  - Insufficient (<0.3): Likely different people

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

### 2. Embedding Extractor (`src/embedding/`)

#### ArcFaceEmbeddingExtractor (`arcface_extractor.py`)

**Purpose**: Extract 512-dimensional face embeddings using ONNX Runtime

**Architecture**:
- Backbone: ResNet100 (ONNX format)
- Embedding: 512-dimensional
- L2 normalized

**Core Methods**:
- `extract_embedding(face_image)` → np.ndarray (512,)
- `preprocess(face_image)` → np.ndarray (112, 112)
- `get_activations(face_image)` → Dict[str, np.ndarray] (placeholder for ONNX)
- `get_embedding_info()` → Dict with model info

**Visualization Methods**:
- `visualize_embedding(embedding)` → (np.ndarray, Dict) - Bar chart of 512 values
- `visualize_similarity_matrix(query, references, ids)` → (np.ndarray, Dict)
- `visualize_similarity_result(query, ref, similarity)` → np.ndarray
- `test_robustness(face_image)` → (np.ndarray, Dict) - Noise robustness test

**Note**: ArcFace ONNX model doesn't expose internal layers, so activations visualization uses placeholder that shows useful info.

#### FaceNetEmbeddingExtractor (`__init__.py`)

**Purpose**: Extract 128-dimensional face embeddings (legacy)

**Architecture**:
- Backbone: torchvision ResNet18 (pretrained on ImageNet)
- Custom head: FC(512→512) → BatchNorm → ReLU → Dropout → FC(512→128) → BatchNorm
- L2 normalization

**Core Methods**:
- `extract_embedding(face_image)` → np.ndarray (128,)
- `preprocess(face_image)` → torch.Tensor
- `extract_embeddings(face_images)` → List[np.ndarray]
- `get_activations(face_image)` → Dict[str, np.ndarray] (11 layers!)

**Visualization Methods**:
- `visualize_embedding(embedding)` → (np.ndarray, Dict) - Bar chart of 128 values
- `visualize_similarity_matrix(query, references, ids)` → (np.ndarray, Dict)
- `visualize_similarity_result(query, ref, similarity)` → np.ndarray
- `visualize_activations(face_image, max_channels)` → CNN layer activations grid
- `visualize_feature_maps(face_image)` → Feature map visualization
- `test_robustness(face_image)` → (np.ndarray, Dict) - Noise robustness test

### 3. SimilarityComparator (`src/embedding/__init__.py`)

**Purpose**: Compare embeddings and return similarity scores

**Methods**:
- `cosine_similarity(embedding1, embedding2)` → float
- `compare_embeddings(query, references, ids)` → List[Tuple[str, float]]
- `get_confidence_band(similarity, model='arcface')` → str

### 4. ReferenceImageManager (`src/reference/__init__.py`)

**Purpose**: Manage reference images and their embeddings

**Features**:
- Stores references in `reference_images/embeddings.json`
- Extracts REAL embeddings (not random!)
- Metadata: id, path, consent info, timestamp
- Auto-saves on add/remove

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
| `arcface_model.onnx` | ~117MB | ArcFace embedding extractor (ONNX) |
| torchvision ResNet18 | ~44MB | PyTorch embedding backbone (FaceNet) |

---

## API Reference

### GET /api/embedding-info

**Response**:
```json
{
  "model": "ArcFaceEmbeddingExtractor",
  "dimension": 512,
  "discrimination": "Excellent - different people show <30% similarity"
}
```

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
  "embedding_size": 512,
  "model": "ArcFaceEmbeddingExtractor",
  "embedding_mean": 0.0321,
  "embedding_std": 0.0452,
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
    "embedding": [...512 values...],
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
      "similarity": 0.75,
      "confidence": "High confidence",
      "verdict": "Likely same person",
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

### ArcFace ONNX Model (No Layer Access)

ArcFace uses ONNX Runtime which doesn't expose internal layer activations like PyTorch. This means:

**For ArcFace**:
- `get_activations()` returns placeholder with model info
- Visualizations show useful info instead of raw CNN activations
- `visualize_activations()` shows embedding channel groups

**For FaceNet** (if enabled):
- Full layer activations available (11 layers)
- Raw CNN feature maps accessible

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
embedding = np.random.rand(512)

# RIGHT - Extract real embeddings from actual images
if self.embedding_extractor is not None:
    faces = self.detector.detect_faces(image_array)
    if faces:
        x, y, w, h = faces[0]
        face_roi = image_array[y:y+h, x:x+w]
        embedding = self.embedding_extractor.extract_embedding(face_roi)
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
FaceNet showed 65-70% similarity for different people - this caused false positives! ArcFace correctly shows <30% for different people, making it much safer for NGO use cases.

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
5. Different Images Similarity (~9-25% with ArcFace)
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
| ArcFace Embedding | O(1) | Fixed network size (ONNX) |
| FaceNet Embedding | O(1) | Fixed network size (PyTorch) |
| Similarity Comparison | O(m) | m = number of references |
| Embedding Storage | O(k) | k = number of stored refs |

---

## Complete API Reference

### All Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/embedding-info` | Model info (ArcFace/FaceNet) |
| POST | `/api/detect` | Face detection |
| POST | `/api/extract` | Embedding extraction |
| POST | `/api/add-reference` | Add reference image |
| GET | `/api/references` | List references |
| DELETE | `/api/references/<id>` | Remove reference |
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
| `activations` | EmbeddingExtractor | CNN activations (placeholder for ArcFace) |
| `features` | EmbeddingExtractor | Feature map grid |
| `multiscale` | FaceDetector | Multi-scale detection |
| `confidence` | FaceDetector | Quality metrics overlay |
| `embedding` | EmbeddingExtractor | 512-dim or 128-dim bar chart |
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

## MANTAX Branding

The Electron UI includes MANTAX branding:
- **Navbar**: White background with subtle border
- **Logo**: SVG with red (#D20A11) and white colors
- **Tagline**: "Ihrem Partner für Autokrane und Schwerlastlogistik" (right side)
- **Compact Design**: 16px padding, clean typography

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

*Architecture documentation updated: February 12, 2026*
*Includes ArcFace integration, ONNX model, 512-dim embeddings, and MANTAX branding*
