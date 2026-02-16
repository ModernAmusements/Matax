# Face Attributes Detection - Implementation Plan

**Created**: February 16, 2026  
**Status**: Ready for Implementation

---

## Overview

Add facial expression detection, age estimation, and gender detection to the face recognition system.

---

## What MediaPipe Provides (Built-in)

| Feature | Available | Implementation |
|---------|-----------|----------------|
| **Face Expressions** | ✅ Yes | 52 blendshapes → simple expressions |
| **Face Landmarks** | ✅ Yes | 468 points - already implemented |
| **Pose (yaw/pitch/roll)** | ✅ Yes | already implemented |
| **Age** | ❌ No | Requires external model |
| **Gender** | ❌ No | Requires external model |

---

## Implementation Options

### Option A: Expressions Only (Safe)
- Enable MediaPipe blendshapes
- Map 52 blendshapes → simple expressions
- Display in UI

### Option B: Expressions + Age/Gender (Full)
- Everything in Option A
- Download Caffe models for age/gender
- Add detection using OpenCV DNN

**Recommendation**: Option B - All features visible to user

---

## Complete Plan - Option B

### Phase 1: Enable MediaPipe Blendshapes

#### 1.1 Edit `src/detection/__init__.py`

**Line 45**: Enable blendshapes
```python
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=True,  # CHANGE: False → True
    output_facial_transformation_matrixes=True
)
```

#### 1.2 Add expression parsing method

Add new method to FaceDetector class:
```python
def parse_expressions(self, blendshapes) -> Dict[str, any]:
    """
    Parse 52 blendshapes into simple expressions.
    
    Returns:
        {
            'expressions': {
                'smile': bool,
                'eye_blink_left': bool,
                'eye_blink_right': bool,
                'mouth_open': bool,
                'eyebrow_raise': bool,
                'surprise': bool,
                'anger': bool,
                'cheek_squint': bool,
            },
            'scores': {
                'smile': 0.0-1.0,
                'eye_blink_left': 0.0-1.0,
                ...
            }
        }
    """
```

**Expression Mapping**:
| Expression | Condition |
|------------|-----------|
| smile | mouthSmile > 0.5 |
| eye_blink_left | eyeBlinkLeft > 0.5 |
| eye_blink_right | eyeBlinkRight > 0.5 |
| mouth_open | jawOpen > 0.3 OR mouthStretch > 0.3 |
| eyebrow_raise | browInnerUp > 0.4 OR browOuterUpLeft > 0.4 |
| surprise | browOuterUpLeft + browOuterUpRight + jawOpen > 1.0 |
| anger | browDownLeft + browDownRight + jawOpen > 1.2 |
| cheek_squint | cheekSquintLeft + cheekSquintRight > 0.5 |

---

### Phase 2: Add Age/Gender Models

#### 2.1 Download Caffe Models

Required files:
```
Models to download:
- deploy_age.prototxt
- age_net.caffemodel
- deploy_gender.prototxt  
- gender_net.caffemodel
```

Source: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector

#### 2.2 Add age/gender methods to FaceDetector

Add to `src/detection/__init__.py`:

```python
def __init__(self, ...):
    # Existing code...
    
    # Age/Gender models (lazy load)
    self._age_model = None
    self._gender_model = None
    self._age_net = None
    self._gender_net = None
    
    # Age ranges for classification
    self._age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', 
                      '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

def _load_age_model(self):
    """Lazy load age detection model."""
    if self._age_net is None:
        try:
            self._age_net = cv2.dnn.readNetFromCaffe(
                "deploy_age.prototxt",
                "age_net.caffemodel"
            )
        except:
            print("Age model not available")

def _load_gender_model(self):
    """Lazy load gender detection model."""
    if self._gender_net is None:
        try:
            self._gender_net = cv2.dnn.readNetFromCaffe(
                "deploy_gender.prototxt",
                "gender_net.caffemodel"
            )
        except:
            print("Gender model not available")

def estimate_age(self, face_image: np.ndarray) -> Dict[str, any]:
    """
    Estimate age from face image using Caffe model.
    
    Returns:
        {
            'age': int,  # Estimated age
            'age_range': str,  # e.g., "(25, 32)"
            'confidence': float  # 0.0-1.0
        }
    """
    # Preprocess
    blob = cv2.dnn.blobFromImage(
        cv2.resize(face_image, (227, 227)),
        1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746)
    )
    
    self._age_net.setInput(blob)
    predictions = self._age_net.forward()
    
    age_index = np.argmax(predictions[0])
    age_range = self._age_list[age_index]
    confidence = float(predictions[0][age_index])
    
    # Extract midpoint of range
    age = int(age_range.strip('()').split(',')[0]) + 5
    
    return {
        'age': age,
        'age_range': age_range,
        'confidence': confidence
    }

def estimate_gender(self, face_image: np.ndarray) -> Dict[str, any]:
    """
    Estimate gender from face image using Caffe model.
    
    Returns:
        {
            'gender': 'male' | 'female',
            'confidence': float  # 0.0-1.0
        }
    """
    # Preprocess
    blob = cv2.dnn.blobFromImage(
        cv2.resize(face_image, (227, 227)),
        1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746)
    )
    
    self._gender_net.setInput(blob)
    predictions = self._gender_net.forward()
    
    gender_index = np.argmax(predictions[0])
    gender = 'male' if gender_index == 0 else 'female'
    confidence = float(predictions[0][gender_index])
    
    return {
        'gender': gender,
        'confidence': confidence
    }
```

---

### Phase 3: API Endpoints

#### 3.1 Add face-attributes endpoint

In `api_server.py`:

```python
@app.route('/api/face-attributes', methods=['GET'])
def get_face_attributes():
    """Get facial attributes: expressions, age, gender for current face."""
    global current_image, current_faces
    
    try:
        if not current_faces:
            return jsonify({'success': False, 'error': 'No faces detected'})
        
        face_box = current_faces[0]
        x, y, w, h = face_box
        face_roi = current_image[y:y+h, x:x+w]
        
        attributes = {}
        
        # Expressions from blendshapes (if available)
        try:
            expressions = detector.parse_expressions(face_blendshapes)
            attributes['expressions'] = expressions
        except:
            attributes['expressions'] = None
        
        # Age estimation
        try:
            age = detector.estimate_age(face_roi)
            attributes['age'] = age
        except:
            attributes['age'] = None
        
        # Gender estimation
        try:
            gender = detector.estimate_gender(face_roi)
            attributes['gender'] = gender
        except:
            attributes['gender'] = None
        
        return jsonify({
            'success': True,
            'attributes': attributes
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
```

#### 3.2 Add visualization endpoint

```python
@app.route('/api/visualizations/attributes', methods=['GET'])
def get_attributes_visualization():
    """Get visualization of face attributes."""
    # Returns bar chart of expression scores
```

#### 3.3 Update /api/detect response

Update `detect_faces()` to include `expressions` in each face object.

---

### Phase 4: Frontend

#### 4.1 HTML - Face Gallery

Display in faces gallery:
```
[Face Image]
😊 Smile: Yes    👁️ Blink: No    😮 Surprise: No
👤 Age: 28-32    Gender: Male (95%)
```

#### 4.2 JavaScript

- Update `detectFaces()` to parse and display attributes
- Add attributes visualization tab

---

### Phase 5: Testing

Run all tests to ensure nothing breaks:

```bash
python test_e2e_pipeline.py
python test_edge_cases.py
python test_frontend_integration.py
```

---

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Breaking existing tests | LOW | Wrap age/gender in try/except, fallback if models missing |
| Model download | MEDIUM | Provide download URLs or skip if not available |
| Performance | LOW | Models are lightweight Caffe |
| Breaking UI | LOW | Add new content, don't modify existing |

---

## Files to Modify

1. `src/dection/__init__.py` - Enable blendshapes, add expression/age/gender methods
2. `api_server.py` - Add endpoints
3. `electron-ui/index.html` - Display attributes
4. `electron-ui/renderer/app.js` - Fetch and display

---

## Test Checklist

- [ ] E2E tests pass
- [ ] Edge case tests pass  
- [ ] Frontend integration tests pass
- [ ] Age estimation works (if model available)
- [ ] Gender estimation works (if model available)
- [ ] Expressions display in UI

---

*Plan created: February 16, 2026*
