# NGO Facial Image Analysis System - Installation Guide

## Quick Installation

### Prerequisites
- Python 3.8 or higher
- macOS/Linux (currently)

### Dependencies

**Required:**
- `opencv-python>=4.8.0` - Face detection (DNN module)
- `numpy>=1.24.0` - Array operations
- `onnxruntime>=1.15.0` - ONNX Runtime for ArcFace
- `pillow>=10.0.0` - Image processing
- `flask` - HTTP server
- `flask-cors` - CORS support

**Optional:**
- `torch>=2.0.0` - PyTorch (for FaceNet fallback)
- `torchvision>=0.15.0` - Pre-trained models (FaceNet)
- `pytest` - Testing

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install opencv-python numpy onnxruntime pillow flask flask-cors

# For FaceNet fallback (optional)
pip install torch torchvision
```

### Download Models

**Face Detection (Required):**
```bash
# deploy.prototxt.txt
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt.txt

# res10_300x300_ssd_iter_140000.caffemodel
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

**ArcFace Model (Included):**
- `arcface_model.onnx` should be in the project root
- If missing, download from: https://github.com/nerox8664/onnxsoftmax/blob/master/arcface_w600k_r100.onnx

### Directory Setup

```bash
mkdir -p reference_images test_images
```

---

## Development Setup

### Code Formatting
```bash
pip install black flake8
black src/ tests/
flake8 src/ tests/
```

### Testing
```bash
# Run E2E tests
python test_e2e_pipeline.py

# Run API tests
python test_api_endpoints.py

# Run with pytest
pip install pytest
pytest tests/
```

---

## Troubleshooting

### Module Not Found
```bash
pip install -e .
```

### OpenCV Errors
```bash
pip install opencv-python-headless opencv-contrib-python
```

### ONNX Runtime
```bash
pip install onnxruntime
```

### Permission Issues
```bash
pip install --user face-recognition-npo
```

---

## Verification

```bash
# Test imports
python -c "from src.detection import FaceDetector; from src.embedding import ArcFaceEmbeddingExtractor; print('Imports OK')"

# Test API
curl http://localhost:3000/api/health

# Check model
curl http://localhost:3000/api/embedding-info
```

The system is ready for use!
