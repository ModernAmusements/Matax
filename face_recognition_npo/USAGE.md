# NGO Facial Image Analysis System - Quick Start Guide

## Quick Start

### Start the System

```bash
cd face_recognition_npo
./start.sh
```

Choose:
- [1] Electron Desktop App
- [2] Browser
- [3] Both

### Manual Start

```bash
# Terminal 1: Start Flask API
cd face_recognition_npo
source venv/bin/activate
python api_server.py

# Terminal 2: Start Electron (or use browser)
cd electron-ui
npm start

# Or open http://localhost:3000 in browser
```

---

## Usage Workflow

```
Step 1: Choose Photo     → Upload image
Step 2: Find Faces       → Click "Find Faces"
Step 3: Create Signature → Click "Create Signature" (CRITICAL!)
Step 4: Add Reference    → Upload reference image
Step 5: Compare          → Click "Compare"
```

---

## API Usage

### Health Check
```bash
curl http://localhost:3000/api/health
```

### Embedding Info
```bash
curl http://localhost:3000/api/embedding-info
# Returns: {"model": "ArcFaceEmbeddingExtractor", "dimension": 512}
```

### Detect Faces
```bash
curl -X POST http://localhost:3000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image"}'
```

### Extract Embedding
```bash
curl -X POST http://localhost:3000/api/extract \
  -H "Content-Type: application/json" \
  -d '{"face_id": 0}'
```

### Add Reference
```bash
curl -X POST http://localhost:3000/api/add-reference \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image", "name": "reference_name"}'
```

### List References
```bash
curl http://localhost:3000/api/references
```

### Compare
```bash
curl -X POST http://localhost:3000/api/compare
```

### Clear Session
```bash
curl -X POST http://localhost:3000/api/clear
```

---

## Confidence Thresholds (ArcFace)

| Similarity | Confidence | Interpretation |
|------------|------------|----------------|
| ≥70% | Very High | Likely same person |
| 45-70% | High | Possibly same person |
| 30-45% | Moderate | Human review recommended |
| <30% | Insufficient | Likely different people |

---

## Testing

```bash
# E2E tests
python test_e2e_pipeline.py

# API tests
python test_api_endpoints.py
```

---

## Documentation

See README.md for complete documentation.
