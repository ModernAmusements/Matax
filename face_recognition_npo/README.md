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

### Option 4: Tkinter GUI
```bash
cd face_recognition_npo
source venv/bin/activate
python gui/facial_analysis_gui.py
```

---

## Features

- ✅ **Face Detection**: OpenCV DNN with Caffe model
- ✅ **Embedding Extraction**: 128-dimensional vectors (ResNet18 backbone)
- ✅ **Similarity Comparison**: Cosine similarity with confidence bands
- ✅ **Reference Management**: Store references with real embeddings (NOT random!)
- ✅ **14 AI Visualizations**: Detection, landmarks, mesh, activations, etc.
- ✅ **Electron Desktop UI**: Ultra minimal design
- ✅ **Flask API Server**: 9 REST endpoints
- ✅ **End-to-End Tests**: All passing

---

## Usage Workflow

```
Step 1: Choose Photo     → Upload image
Step 2: Find Faces       → Click "Find Faces"
Step 3: Create Signature → Click "Create Signature" (EXTRACTS EMBEDDING)
Step 4: Add Reference    → Upload reference image
Step 5: Compare          → Click "Compare"
```

---

## Expected Results

| Scenario | Similarity Score |
|----------|-----------------|
| Same image | ~100% |
| Same person, different photo | 70-99% |
| Different person | 20-60% |

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

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/detect` | Detect faces |
| POST | `/api/extract` | Extract embedding |
| POST | `/api/add-reference` | Add reference |
| POST | `/api/compare` | Compare embeddings |
| GET | `/api/visualizations/<type>` | Get visualization |

---

## Testing

```bash
# End-to-end pipeline test
python test_e2e_pipeline.py

# Unit tests
python -m unittest discover tests/
```

---

## Documentation

- **PROJECT_STRUCTURE.md**: Complete architecture, lessons learned, roadmap
- **ARCHITECTURE.md**: Detailed system design
- **DEVELOPMENT_LOG.md**: Development history
- **ETHICAL_COMPLIANCE.md**: Ethical guidelines

---

## Lessons Learned (Don't Repeat!)

### 1. Dynamic Array Sizes
Always use dynamic array sizes when output depends on input count:
```python
# WRONG: Hardcoded size
output = np.zeros((150, 300, 3), dtype=np.uint8)

# RIGHT: Dynamic size
output_size = max(150, n * cell_size)
output = np.zeros((output_size, output_size, 3), dtype=np.uint8)
```

### 2. Real Embeddings
Never use random values for embeddings:
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

---

## Ethical Guidelines

1. **Consent-Based**: All images must have lawful basis
2. **Human Oversight**: No automated decisions
3. **Uncertainty Handling**: Use confidence bands
4. **Privacy Protection**: Non-reversible embeddings
5. **Documentation**: Maintain audit trails

---

## License

MIT License
