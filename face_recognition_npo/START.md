# Quick Start - NGO Facial Image Analysis System

**Version**: 0.4.1  
**Last Updated**: February 15, 2026

---

## ⚠️ CRITICAL WORKFLOW RULE

After EVERY code change:
1. Run: `python test_e2e_pipeline.py`
2. Run: `python test_edge_cases.py`
3. Run: `python test_frontend_integration.py`
4. Say "FINISHED" ONLY after ALL tests pass

---

## Start the Application

```bash
cd /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo
./start.sh
```

Choose:
- `[1]` Electron Desktop App
- `[2]` Browser
- `[3]` Both

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

## SCSS Styling

The project uses SCSS for maintainable styling. 

**Files:**
- `electron-ui/styles/design-system.scss` - SCSS source
- `electron-ui/styles/design-system.css` - Compiled CSS

**Compile SCSS:**
```bash
cd electron-ui
npm run scss
```

**Note:** `./start.sh` automatically compiles SCSS before starting.

---

## Manual Start (Alternative)

```bash
# Terminal 1: Start Flask API
cd /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo
source venv/bin/activate
python api_server.py

# Terminal 2: Start Electron (or use browser)
cd electron-ui
npm start

# Or open http://localhost:3000 in browser
```

---

## Important Notes

### Always Use Project Virtual Environment
```bash
# CORRECT
source venv/bin/activate
python api_server.py

# WRONG - uses system Python
python api_server.py
```

### Clear Cache After Edits
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### Run Tests
```bash
# E2E tests
python test_e2e_pipeline.py

# API tests
python test_api_endpoints.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/embedding-info` | Model info |
| POST | `/api/detect` | Face detection |
| POST | `/api/extract` | Embedding extraction |
| POST | `/api/add-reference` | Add reference |
| GET | `/api/references` | List references |
| DELETE | `/api/references/<id>` | Remove reference |
| POST | `/api/compare` | Compare embeddings |
| POST | `/api/clear` | Clear session |

---

## Confidence Thresholds (ArcFace)

| Similarity | Confidence |
|------------|------------|
| ≥70% | Very High - Likely same person |
| 45-70% | High - Possibly same person |
| 30-45% | Moderate - Human review recommended |
| <30% | Insufficient - Likely different people |

---

## Comparison Score Breakdown

The comparison uses 5 factors with weighted scoring:

| Factor | Weight | Description |
|--------|--------|-------------|
| ArcFace | 60% | Primary embedding (512-dim) |
| FaceNet | 20% | Secondary embedding (128-dim) |
| Landmarks | 15% | Geometric consistency |
| Activation | 5% | Neural activation similarity |
| Quality | 5% | Image quality factor |

**Note**: No pose penalty - same person with different poses will match correctly.

---

## Documentation

| File | Description |
|------|-------------|
| `README.md` | Main documentation |
| `CONTEXT.md` | Critical rules for code edits |
| `ARCHITECTURE.md` | Complete system design |
| `DEVELOPMENT_LOG.md` | Development history |
| `STYLES.md` | CSS classes reference |

---

*Created: February 15, 2026*
