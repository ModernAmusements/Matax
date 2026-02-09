# NGO Facial Image Analysis System - Installation Guide

## Prerequisites

### Required Dependencies
- Python 3.7 or higher
- OpenCV (for face detection)
- NumPy (for numerical operations)
- PyTorch (for embedding extraction)
- Pillow (for image processing)

### Optional Dependencies
- matplotlib (for visualization)
- pytest (for testing)
- sphinx (for documentation)

## Installation Methods

### Method 1: pip Installation (Recommended)

1. **Install from PyPI:**
   ```bash
   pip install face-recognition-npo
   ```

2. **Install with extras:**
   ```bash
   pip install "face-recognition-npo[dev]"
   pip install "face-recognition-npo[docs]"
   ```

### Method 2: From Source

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd face_recognition_npo
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

### Method 3: Development Installation

1. **Clone and install in development mode:**
   ```bash
   git clone <repository-url>
   cd face_recognition_npo
   pip install -e .[dev]
   ```

## Post-Installation Setup

### 1. Download Required Models

#### Face Detection Model
Download the pre-trained Caffe model:
```bash
# Download deploy.prototxt.txt
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt.txt

# Download res10_300x300_ssd_iter_140000.caffemodel
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

#### Embedding Model
Download a pre-trained FaceNet model or use the included simplified version.

### 2. Create Configuration File

Copy the configuration template:
```bash
cp config_template.py config.py
```

Edit `config.py` to match your environment:
```python
# Set your specific paths
FACE_DETECTION_MODEL = "/path/to/deploy.prototxt.txt"
FACE_DETECTION_WEIGHTS = "/path/to/res10_300x300_ssd_iter_140000.caffemodel"
REFERENCE_IMAGE_DIR = "/path/to/reference_images"
```

### 3. Create Required Directories

```bash
mkdir -p reference_images logs temp models data test_images docs examples
```

## Quick Start

### Basic Usage

1. **Add reference images:**
   ```bash
   python main.py reference add reference1.jpg id1 --metadata '{"source": "document", "consent": "yes"}'
   python main.py reference add reference2.jpg id2 --metadata '{"source": "photo", "consent": "yes"}'
   ```

2. **List reference images:**
   ```bash
   python main.py reference list
   ```

3. **Detect faces in image:**
   ```bash
   python main.py detect test_image.jpg
   ```

4. **Verify image for documentation:**
   ```bash
   python main.py verify verification_image.jpg
   ```

5. **Run webcam demo:**
   ```bash
   python main.py webcam
   ```

### Example Workflow

```bash
# 1. Add reference images
python main.py reference add john_doe.jpg john_doe --metadata '{"name": "John Doe", "document_id": "123456"}'
python main.py reference add jane_smith.jpg jane_smith --metadata '{"name": "Jane Smith", "document_id": "789012"}'

# 2. Detect faces in new image
python main.py detect new_arrival.jpg

# 3. Verify new image against references
python main.py verify new_arrival.jpg

# 4. Run live demo
python main.py webcam
```

## Testing

### Run All Tests
```bash
python main.py test
```

### Run Specific Tests
```bash
# Run only detection tests
python -m unittest tests.test_detection

# Run only embedding tests
python -m unittest tests.test_embedding
```

### Test Coverage
```bash
# Install coverage tools
pip install pytest pytest-cov

# Run with coverage
pytest --cov=src tests/
```

## Development Setup

### Code Formatting
```bash
# Install formatting tools
pip install black flake8

# Format code
black src/ tests/ examples/

# Check code style
flake8 src/ tests/ examples/
```

### Documentation
```bash
# Install documentation tools
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs
sphinx-build -b html . _build/html
```

## Troubleshooting

### Common Issues

#### Module Not Found
```bash
# If you get "ModuleNotFoundError", make sure the package is installed
pip install -e .
```

#### OpenCV Errors
```bash
# Install OpenCV with extra modules
pip install opencv-python-headless opencv-contrib-python
```

#### PyTorch Installation
```bash
# Install PyTorch for your system
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Permission Issues
```bash
# If you get permission errors, use --user flag
pip install --user face-recognition-npo
```

### Environment Setup

#### Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Install package in virtual environment
pip install -e .
```

#### Anaconda Environment
```bash
# Create conda environment
conda create -n face-recognition-npo python=3.9

# Activate environment
conda activate face-recognition-npo

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Verification

### Check Installation
```bash
# Test basic functionality
python -c "from main import FaceRecognitionApp; app = FaceRecognitionApp(); print('Installation successful')"
```

### Verify Dependencies
```bash
# Check installed packages
pip list | grep -E "(opencv|numpy|torch|pillow|matplotlib)"

# Check versions
python -c "import cv2, numpy, torch, PIL; print(f'OpenCV: {cv2.__version__}, NumPy: {numpy.__version__}, PyTorch: {torch.__version__}, Pillow: {PIL.__version__}')"
```

## Uninstallation

### Remove Package
```bash
# Remove package
pip uninstall face-recognition-npo

# Or remove from source
cd face_recognition_npo
pip uninstall -e .
```

### Clean Up
```bash
# Remove downloaded models
rm deploy.prototxt.txt res10_300_300_ssd_iter_140000.caffemodel

# Remove directories
rm -rf reference_images logs temp models data test_images
```

The system is now ready for use in NGO documentation verification workflows!