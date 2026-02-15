# NGO Facial Image Analysis System - Architecture

**Version**: 0.5.0  
**Last Updated**: February 15, 2026  
**Status**: ✅ Fully Functional - Enhanced Features (LBP, Asymmetry, Multi-Pose, 3D Normalization)

---

## ⚠️ CRITICAL WORKFLOW RULE

After EVERY code change:
1. Run: `python test_e2e_pipeline.py`
2. Run: `python test_edge_cases.py`
3. Run: `python test_frontend_integration.py`
4. Say "FINISHED" ONLY after ALL tests pass

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
- `compute_lbp_descriptor(face_image)` → np.ndarray (256-dim histogram) - NEW
- `compute_facial_asymmetry(landmarks)` → Dict with asymmetry features - NEW
- `normalize_face_with_mesh(face_image, mesh_landmarks)` → np.ndarray - NEW (3D mesh-based alignment)

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
- `compute_pose_weight(pose1, pose2)` → float - NEW (pose-aware matching)
- `lbp_similarity(lbp1, lbp2)` → float - NEW (texture matching)
- `asymmetry_similarity(asym1, asym2)` → float - NEW (uniqueness analysis)
- `compute_multi_pose_score(query_emb, pose_embeddings)` → Tuple[float, Dict] - NEW (multi-pose matching)

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
| GET | `/api/diagnostics` | System diagnostics (MediaPipe, models) |
| POST | `/api/detect` | Face detection |
| POST | `/api/extract` | Embedding extraction (both models) |
| POST | `/api/add-reference` | Add reference image |
| POST | `/api/add-reference-pose/<id>` | Add pose variant to reference - NEW |
| GET | `/api/references` | List references |
| DELETE | `/api/references/<id>` | Remove reference |
| POST | `/api/compare` | Compare embeddings (enhanced) |
| GET | `/api/visualizations/<type>` | Get visualization |
| GET | `/api/visualizations/<type>/reference/<id>` | Get ref visualization |
| GET | `/api/quality` | Quality metrics |
| GET | `/api/eyewear` | Eyewear detection |
| POST | `/api/clear` | Clear session |
| GET | `/api/status` | Debug server state |

### Enhanced Comparison (Version 0.5.0)

The `/api/compare` endpoint now returns with all new features:

| Field | Description |
|-------|-------------|
| `arcface_similarity` | ArcFace cosine similarity (0-1) |
| `facenet_similarity` | FaceNet cosine similarity (0-1) |
| `landmark_similarity` | Landmark geometry similarity (0-1) |
| `activation_similarity` | Neural network activation similarity (0-1) |
| `pose_weight` | NEW - Pose-aware weight adjustment (0.95-1.0) |
| `lbp_similarity` | NEW - LBP texture similarity (lighting-invariant) |
| `asymmetry_similarity` | NEW - Facial asymmetry similarity (uniqueness) |
| `normalized_similarity` | NEW - 3D mesh-normalized embedding similarity |
| `multi_pose_score` | NEW - Best match across multiple pose variants |
| `multi_pose_used` | NEW - Whether multiple poses were available |
| `final_score` | Weighted combination (see below) |
| `status` | "match", "possible", or "no_match" |
| `match_label` | "Full Match", "Possible Match", "No Match" |
| `reasons` | List of match reasoning strings |

#### New Weights (v0.5.0)

| Factor | Weight | Purpose |
|--------|--------|---------|
| ArcFace | 40% | Primary embedding (reduced from 60%) |
| 3D Normalized | 15% | Handle extreme angles (NEW) |
| Multi-Pose Best | 10% | Best pose variant match (NEW) |
| LBP Texture | 8% | Lighting-invariant matching (NEW) |
| Asymmetry | 7% | Uniqueness analysis (NEW) |
| Landmark Geometry | 8% | Geometric consistency |
| FaceNet | 10% | Secondary embedding |
| Activation | 1% | Neural patterns |
| Quality | 1% | Reliability factor |

**Pose Weight Modifier**: 0.95-1.0x applied to all scores based on pose similarity

### Visualization Types (17 + 9 Test = 26 total)

| Type | Source | Description |
|------|--------|-------------|
| `detection` | FaceDetector | Bounding boxes with confidence |
| `extraction` | FaceDetector | Face ROI extraction |
| `preprocessing` | FaceDetector | CLAHE enhancement |
| `landmarks` | FaceDetector | **468 MediaPipe landmarks** |
| `mesh3d` | FaceDetector | 478-point 3D mesh |
| `alignment` | FaceDetector | Pitch/yaw/roll orientation |
| `saliency` | FaceDetector | Attention visualization |
| `activations` | EmbeddingExtractor | CNN activations (FaceNet) |
| `features` | EmbeddingExtractor | Feature map grid |
| `multiscale` | FaceDetector | Multi-scale detection |
| `confidence` | FaceDetector | Quality metrics overlay |
| `embedding` | EmbeddingExtractor | 512-dim or 128-dim bar chart |
| `similarity` | EmbeddingExtractor | Similarity result bar |
| `robustness` | EmbeddingExtractor | Noise robustness test |
| `biometric` | FaceDetector | Biometric capture overview |
| `eyewear` | FaceDetector | Eyewear detection visualization |
| `asymmetry` | FaceDetector | NEW - Uniqueness analysis |
| `texture` | FaceDetector | NEW - LBP texture visualization |
| `normalized` | FaceDetector | NEW - 3D normalized face |

### Test Tabs (9)

| Tab | Description |
|-----|-------------|
| API Health | Health check status |
| Detection | Face detection info |
| Extraction | Embedding extraction info |
| References | Reference management |
| Multi-Match | Multi-reference matching |
| Pose | Pose detection status |
| Eyewear | Eyewear detection |
| Viz Types | Available visualizations |
| Session | Session management |

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

## New Features (v0.5.0 - February 15, 2026)

### 1. 3D Mesh Normalization
**Purpose**: Handle extreme angles using 468-point MediaPipe mesh

Method: `normalize_face_with_mesh(face_image, mesh_landmarks)`
- Uses MediaPipe's 468-point facial mesh
- Computes eye centers from multiple mesh points (more accurate than 2-point)
- Rotates face to align eyes horizontally
- Extracts embedding from aligned face for better matching

**Weight in comparison**: 15%

### 2. LBP Texture Features
**Purpose**: Lighting-invariant matching

Method: `compute_lbp_descriptor(face_image)`
- Extracts Local Binary Pattern histogram (256 bins)
- Compares texture patterns between faces
- Robust to lighting variations

Method: `lbp_similarity(lbp1, lbp2)`
- Compares LBP histograms using intersection
- Returns 0-1 similarity score

**Weight in comparison**: 8%

### 3. Facial Asymmetry Analysis
**Purpose**: Uniqueness analysis

Method: `compute_facial_asymmetry(landmarks)`
- Computes distances between bilateral facial points
- Measures left-right asymmetry
- Each face has unique asymmetry pattern

Method: `asymmetry_similarity(asym1, asym2)`
- Compares asymmetry features between two faces
- Returns 0-1 similarity score

**Weight in comparison**: 7%

### 4. Pose-Aware Weight
**Purpose**: Adjust similarity based on pose difference

Method: `compute_pose_weight(pose1, pose2)`
- Computes pose difference (yaw + pitch)
- If poses are similar (< 15°): weight = 1.0
- If poses differ (15-30°): weight = 0.98
- If poses differ greatly (> 30°): weight = 0.95

**Modifier**: 0.95-1.0x applied to final score

### 5. Multi-Pose Enrollment
**Purpose**: Store and match against multiple poses of same person

New endpoint: `POST /api/add-reference-pose/<ref_id>`
- Adds additional pose variant to existing reference
- Automatically categorizes pose (frontal, left, right, up, down)
- Stores in reference's `poses` dictionary

Method: `compute_multi_pose_score(query_emb, pose_embeddings)`
- Compares query against all pose variants
- Returns best match score

**Weight in comparison**: 10%

---

## Storage Format

Reference storagejson
{
  "id": 0,
  " now includes:

```name": "John Doe",
  "embedding": {"arcface": [...], "facenet": [...]},
  "lbp_histogram": [...],
  "asymmetry": {"left_eye_right_eye_dist": 10.5, ...},
  "normalized_embedding": [...],
  "poses": {
    "frontal": {"embedding": {...}, "yaw": 0, "pitch": 0},
    "left": {"embedding": {...}, "yaw": -25, "pitch": 0},
    "right": {"embedding": {...}, "yaw": 25, "pitch": 0}
  },
  "pose": {"yaw": 0, "pitch": 0, "roll": 0},
  "pose_category": "frontal"
}
```

---

## Testing New Features

Run `python test_edge_cases.py` to verify:

- `test_lbp_descriptor()` - LBP extraction and similarity
- `test_asymmetry_features()` - Asymmetry computation
- `test_pose_weight()` - Pose weight calculation
- `test_mesh_normalization()` - 3D mesh normalization
- `test_multi_pose_score()` - Multi-pose comparison

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

*Architecture documentation updated: February 15, 2026*
*Version 0.5.0 - Includes LBP texture, facial asymmetry, 3D mesh normalization, multi-pose enrollment, pose-aware matching*
