# NGO Facial Image Analysis System

A Python-based facial image analysis system with Electron desktop UI for ethical, consent-based NGO use in documentation verification and investigative work.

## Quick Start

### Desktop UI (Recommended)
```bash
cd face_recognition_npo/electron-ui
npm install
npm start
```

### Python API Server
```bash
cd face_recognition_npo
source venv/bin/activate
python api_server.py
# Open http://localhost:3000 in browser
```

## Features

- **Electron Desktop UI**: Ultra minimal design - black text on white, no icons
- **Sticky Terminal Footer**: Always-visible logs, click to expand/collapse
- **Face Detection**: Detect human faces in images using OpenCV's DNN module
- **Embedding Extraction**: Convert faces into non-reversible 128-dimensional embeddings
- **Similarity Comparison**: Compare embeddings using cosine similarity with confidence bands
- **Reference Management**: Manage reference images with metadata and consent information
- **14 AI Visualizations**: See how the AI analyzes faces

## UI Design

**Ultra Minimal:**
- Sans serif font (system stack)
- Black text on white background
- White buttons with black borders
- Step indicators (Step 1, 2, 3, 4)

**Terminal Footer:**
- Fixed at bottom, always visible
- Shows live processing logs
- Compact (5 lines) or expanded (click to toggle)
- Black background, green monospace text

## Architecture

```
face_recognition_npo/
├── api_server.py           # Flask API server
├── src/
│   ├── detection/          # Face detection module
│   └── embedding/          # Embedding extraction & comparison
├── electron-ui/           # Electron desktop application
│   ├── main.js             # Electron main process
│   ├── renderer/app.js     # Frontend JavaScript
│   └── index.html          # Ultra minimal UI
├── tests/                   # Unit tests
└── examples/              # Usage examples

## Core Components

### 1. Face Detection
- Uses OpenCV's DNN module with pre-trained Caffe model
- Detects multiple faces per image
- Outputs bounding boxes with confidence scores

### 2. Embedding Extraction
- Converts detected faces into 128-dimensional embeddings
- Embeddings are non-reversible and comparable via distance metrics
- Includes visualizations: activations, feature maps, robustness testing

### 3. Similarity Comparison
- Compares embeddings using cosine similarity
- Confidence bands: High/Moderate/Low/Insufficient
- Never makes binary match/no-match decisions
- Visual similarity matrix for multi-reference comparison

### 4. AI Visualizations (14 types)
| Visualization | Description |
|--------------|-------------|
| Detection | Bounding boxes with confidence scores |
| Extraction | 160x160 face crop preview |
| Landmarks | Facial feature points (eyes, nose, mouth) |
| 3D Mesh | 3D wireframe overlay on face |
| Alignment | Face alignment visualization |
| Attention | Saliency/attention heatmap |
| Activations | Neural network layer activations |
| Features | Feature map visualizations |
| Multi-Scale | Face at 5 different scales |
| Confidence | Quality metrics dashboard (brightness, sharpness, centering) |
| Embedding | 128-dim vector visualization |
| Similarity | Similarity gauge with threshold |
| Robustness | Noise robustness test results |
| Biometric | Biometric capture quality view |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/detect` | Detect faces in uploaded image |
| POST | `/api/extract` | Extract embedding from detected face |
| POST | `/api/add-reference` | Add reference image for comparison |
| GET | `/api/references` | List all references |
| POST | `/api/compare` | Compare query embedding with references |
| GET | `/api/visualizations/<type>` | Get specific AI visualization |
| POST | `/api/clear` | Clear all session data |

## Usage Examples

### Desktop UI
The Electron app provides a complete workflow:
1. Choose Photo → Upload image
2. Find Faces → AI detects all faces
3. Create Signature → Extract 128-dim embedding
4. Add Reference → Upload reference images
5. Compare → See similarity scores with confidence bands

### Python API

```python
from src.detection import FaceDetector
from src.embedding import FaceNetEmbeddingExtractor, SimilarityComparator

# Initialize
detector = FaceDetector()
extractor = FaceNetEmbeddingExtractor()
comparator = SimilarityComparator()

# Detect faces
image = cv2.imread("test.jpg")
faces = detector.detect_faces(image)

# Extract embedding
x, y, w, h = faces[0]
face_image = image[y:y+h, x:x+w]
embedding = extractor.extract_embedding(face_image)

# Compare
results = comparator.compare_embeddings(embedding, [ref_embedding], ["ref1"])
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd face_recognition_npo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required model files:
- Download `deploy.prototxt.txt` and `res10_300x300_ssd_iter_140000.caffemodel` for face detection
- Place them in the project directory

## Ethical Guidelines

This system is designed for ethical NGO use with the following principles:

1. **Consent-Based**: All images must have lawful basis for use
2. **Human Oversight**: No automated identification - human review required
3. **Uncertainty Handling**: Never claim certainty - use confidence bands
4. **Privacy Protection**: Embeddings are non-reversible
5. **Documentation**: Maintain clear audit trails

## Testing

Run all tests:
```bash
python -m unittest discover tests/
```

Run specific test:
```bash
python -m unittest tests.test_detection
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For questions or support, please contact the NGO technical team.