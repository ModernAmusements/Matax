# NGO Facial Image Analysis System - Development Log

**Last Updated**: February 10, 2026  
**Project**: Face Recognition GUI for NGO Use

---

## Session Summary: February 10, 2026

### Goals
- Fix visualization tabs to show real data instead of mock data
- Fix reference image handling
- Fix comparison workflow
- Ensure all 14 visualization tabs work correctly

---

## Issues Encountered & Solutions

### Issue 1: 3D Mesh Visualization - IndexError

**Error**:
```
IndexError: index 6 is out of bounds for axis 1 with size 6
```

**Location**: `src/detection/__init__.py`, `visualize_3d_mesh()`

**Root Cause**: Grid array dimensions didn't match loop bounds. The code was using inconsistent sizes for meshgrid and loop ranges.

**Fix**:
```python
# BEFORE (WRONG)
grid_x, grid_y = np.meshgrid(
    np.linspace(0.25, 0.75, 6),
    np.linspace(0.25, 0.75, 8)
)
for i in range(6):
    for j in range(8):  # Index out of bounds!

# AFTER (CORRECT)
rows, cols = 6, 6
grid_x, grid_y = np.meshgrid(
    np.linspace(0.25, 0.75, cols),
    np.linspace(0.25, 0.75, rows)
)
for i in range(rows):
    for j in range(cols):
```

**Lesson**: Always match loop bounds with array dimensions.

---

### Issue 2: Embedding Extraction - PyTorch Hook Error

**Error**:
```
TypeError: hook1() missing 1 required positional argument: 'output'
```

**Location**: `src/embedding/__init__.py`, `_register_hooks()`

**Root Cause**: Used `register_forward_pre_hook` with wrong signature. Pre-hook receives `(module, input)`, but code defined `(module, input, output)`.

**Fix**:
```python
# BEFORE (WRONG)
def hook1(module, input, output):
    self.activation_maps['conv1'] = input[0]
self.model[0].register_forward_pre_hook(hook1)

# AFTER (CORRECT)
def hook_fn(module, input, output):
    self.activation_maps[module] = input[0]
handle = module.register_forward_hook(hook_fn)  # Use forward_hook, not pre_hook
```

**Alternative**:
```python
# Simpler approach: Don't use hooks, compute activations separately
activations = {}
def hook_fn(module, input, output):
    activations[module] = input[0]
```

**Lesson**: PyTorch hook signatures vary by hook type. Pre-hook = `(module, input)`, Forward-hook = `(module, input, output)`.

---

### Issue 3: Similarity Visualization - "No Data Available"

**Problem**: Similarity tab showed "No Data Available" even when references were added.

**Root Cause**: 
1. `_on_extract()` only showed similarity if references existed
2. `_on_add_ref()` didn't trigger similarity update
3. Similarity matrix visualization required references

**Fix**:
```python
# In _on_extract(): Show self-comparison when no references
ref_embs = [r.get("embedding") for r in self.ref_gallery_4.ref_data.values() 
            if r.get("embedding") is not None]
if ref_embs:
    self.viz_panel.set_visualization("similarity",
        self.extractor.visualize_similarity_matrix(embedding, ref_embs, ref_names))
elif self.current_embedding is not None:
    # Show self-comparison (100% match)
    self.viz_panel.set_visualization("similarity",
        self.extractor.visualize_similarity_matrix(embedding, 
                                                   [self.current_embedding], 
                                                   ["Current"]))
```

**In _on_add_ref()**: Auto-compare when adding reference:
```python
if self.current_embedding is not None:
    sim = self.comparator.cosine_similarity(self.current_embedding, embedding)
    conf = self.comparator.get_confidence_band(sim)
    self.compare_4.set_comparison(self.current_face_image, face_img, sim, conf)
    self.status_4.config(text=f"Added: {name} ({int(sim*100)}% match)")
```

**Lesson**: UI feedback should update immediately. Don't wait for user to click "Compare".

---

### Issue 4: Python Bytecode Caching

**Problem**: Despite code fixes, GUI still showed old errors.

**Root Cause**: Python `.pyc` files in `__pycache__` directories cached old code.

**Solution**:
```bash
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
```

**Lesson**: Always clear bytecode cache after making changes. Add to development workflow.

---

## Architecture Changes

### Detection Module (`src/detection/__init__.py`)

**Added Methods**:
- `detect_faces_with_confidence()` - Returns confidence scores with faces
- `detect_eyes()` - Eye detection using Haar cascade
- `estimate_landmarks()` - Estimates 68-point facial landmarks
- `compute_alignment()` - Calculates yaw, pitch, roll
- `compute_saliency()` - Saliency/attention map using Sobel
- `compute_quality_metrics()` - Quality: brightness, contrast, sharpness
- `visualize_3d_mesh()` - 3D wireframe overlay
- `visualize_alignment()` - Pose visualization
- `visualize_saliency()` - Attention heatmap
- `visualize_multiscale()` - Multi-resolution analysis

### Embedding Module (`src/embedding/__init__.py`)

**Added Methods**:
- `_register_hooks()` - PyTorch forward hooks for activations
- `get_activations()` - Extract layer activations
- `visualize_activations()` - Show neural network patterns
- `visualize_feature_maps()` - Feature importance heatmap
- `visualize_embedding()` - 128-dim signature as bar chart
- `visualize_similarity_matrix()` - Query vs reference comparison
- `test_robustness()` - Noise tolerance testing

**Modified Methods**:
- `extract_embedding()` - Added hooks for activation extraction

### GUI (`gui/user_friendly_gui.py`)

**Added Features**:
- 14 visualization tabs (all now using real data)
- Reference gallery with thumbnails
- Real-time comparison when adding references
- Quality metrics display
- Step-by-step workflow indicator

---

## All 14 Visualizations (Now Working)

| Tab | Visualization | Data Source |
|-----|--------------|-------------|
| 1 | **Detection** | Bounding boxes with confidence from DNN |
| 2 | **Extraction** | 160x160 processed face crop |
| 3 | **Landmarks** | Estimated 10-point facial keypoints |
| 4 | **3D Mesh** | Depth-mapped wireframe overlay |
| 5 | **Alignment** | Yaw/pitch computed from eye positions |
| 6 | **Attention** | Sobel edge detection + center bias |
| 7 | **Activations** | PyTorch hooks on Conv layers |
| 8 | **Features** | Mean activation maps |
| 9 | **Multi-Scale** | Face at 0.5x, 0.75x, 1.0x, 1.25x, 1.5x |
| 10 | **Confidence** | Viridis heatmap on face regions |
| 11 | **Embedding** | 128-dim vector as sorted bar chart |
| 12 | **Similarity** | Cosine similarity with references |
| 13 | **Robustness** | Noise injection at σ=0.01-0.2 |
| 14 | **Biometric** | Quality metrics + bounding box |

---

## Workflow

```
┌─────────────────┐
│ 1. Choose Photo │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Find Faces   │  ← Detects faces, generates visualizations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Create       │  ← Extracts 128-dim embedding
│    Signature     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Compare      │  ← Add references, compare embeddings
└─────────────────┘
```

---

## Test Results

```
============================================================
FACE ANALYSIS WORKFLOW TEST
============================================================

[STEP 1] Loading images...
  ✓ Loaded query image: (562, 1000, 3)
  ✓ Loaded reference image: (306, 460, 3)

[STEP 2] Detecting faces...
  ✓ Detected 1 face(s)
  ✓ Face region: 246x328

[STEP 3] Generating visualizations...
  ✓ Detection: (562, 1000, 3)
  ✓ Extraction: (562, 1000, 3)
  ✓ Landmarks: (328, 246, 3)
  ✓ 3D Mesh: (328, 246, 3)
  ✓ Alignment: (328, 246, 3)
  ✓ Saliency: (328, 246, 3)
  ✓ Activations: (200, 320, 3)
  ✓ Features: (224, 224, 3)
  ✓ Multi-Scale: (306, 685, 3)
  ✓ Biometric: (562, 1000, 3)

[STEP 4] Extracting embedding...
  ✓ 128-dim embedding: 128 values

[STEP 5] Embedding visualization...
  ✓ Embedding viz: (120, 256, 3)

[STEP 6] Testing robustness...
  ✓ Robustness viz: (100, 200, 3)
    Avg similarity: 99.95%

[STEP 7] Processing reference image...
  ✓ Reference embedding extracted

[STEP 8] Comparing embeddings...
  ✓ Similarity: 0.9961 (99.61%)
  ✓ Confidence: High confidence

[STEP 9] Similarity matrix...
  ✓ Similarity matrix: (150, 300, 3)

[STEP 10] Quality metrics...
  ✓ Quality metrics calculated
============================================================
WORKFLOW COMPLETE - ALL TESTS PASSED!
============================================================
```

---

## Commands

### Start GUI
```bash
cd /Users/modernamusmenet/Desktop/MANTAX/face_recognition_npo
source venv/bin/activate
python3 gui/user_friendly_gui.py
```

### Clear Cache (After Code Changes)
```bash
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### Run Tests
```bash
python3 test_workflow.py
```

---

## Lessons Learned

### 1. PyTorch Hooks Are Tricky
- `register_forward_pre_hook(module, input)` - Before forward pass
- `register_forward_hook(module, input, output)` - After forward pass
- Pre-hook doesn't have access to output

### 2. Array Bounds Must Match
- When using `meshgrid`, store dimensions first
- Use variables in loops, not hardcoded values

### 3. Bytecode Caching
- Python caches `.pyc` files in `__pycache__`
- Always clear cache after code changes
- Consider using `python -B` to disable caching during development

### 4. UI Feedback
- Update visuals immediately, don't wait for user actions
- Show self-comparison when no references exist
- Provide confidence bands, not just numbers

### 5. Testing
- Test with real images, not just synthetic data
- Run full workflow tests, not just unit tests
- Clear cache before testing

---

## Future Work

### High Priority
- [ ] Add more reference images
- [ ] Batch processing for multiple images
- [ ] Export results to PDF/JSON
- [ ] Improve eye detection accuracy

### Medium Priority
- [ ] Real 3D mesh using dlib or MediaPipe
- [ ] Landmark visualization for all 68 points
- [ ] ROC curve for threshold tuning
- [ ] Save/load embeddings

### Low Priority
- [ ] Video processing mode
- [ ] Web interface
- [ ] Mobile app

---

## File Changes Summary

| File | Changes |
|------|---------|
| `src/detection/__init__.py` | Added 10 visualization methods |
| `src/embedding/__init__.py` | Added 5 visualization methods, fixed hooks |
| `gui/user_friendly_gui.py` | 14 tabs, real-time comparison, reference gallery |
| `test_workflow.py` | Full workflow test script |

---

## Questions for Next Session

1. Should we add video processing capability?
2. What export formats are needed (PDF, JSON, CSV)?
3. Should we add batch processing for multiple images?
4. Do we need a web interface?

---

*Document created: February 10, 2026*
*To be updated in future sessions*
