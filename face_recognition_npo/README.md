# NGO Facial Image Analysis System

A Python-based facial image analysis system designed for ethical, consent-based NGO use in documentation verification and investigative work.

## Features

- **Face Detection**: Detect human faces in images using OpenCV's DNN module
- **Embedding Extraction**: Convert faces into non-reversible numerical embeddings using FaceNet architecture
- **Similarity Comparison**: Compare embeddings using cosine similarity with confidence bands
- **Reference Management**: Manage reference images with metadata and consent information
- **Human Review Interface**: Ensure human oversight at every decision point
- **Webcam Support**: Real-time face detection and comparison for testing

## Architecture

```
face_recognition_npo/
├── src/
│   ├── detection/     # Face detection module
│   ├── embedding/     # Embedding extraction module
│   ├── comparison/    # Similarity comparison module
│   ├── reference/     # Reference image management
│   └── __init__.py
├── tests/             # Unit tests
├── utils/             # Utility functions (webcam)
├── examples/          # Usage examples
├── requirements.txt
└── README.md
```

## Core Components

### 1. Face Detection
- Uses OpenCV's DNN module with pre-trained Caffe model
- Detects multiple faces per image
- Handles variations in lighting, pose, and occlusion
- Outputs bounding boxes with confidence scores

### 2. Embedding Extraction
- Converts detected faces into 128-dimensional embeddings
- Uses FaceNet architecture (simplified for this implementation)
- Embeddings are non-reversible and comparable via distance metrics
- Normalized for consistent comparison

### 3. Similarity Comparison
- Compares embeddings using cosine similarity
- Provides ranked similarity scores
- Includes confidence bands (High/Moderate/Low/Insufficient)
- Never makes binary match/no-match decisions

### 4. Reference Management
- Manages reference images with unique IDs
- Stores metadata including consent information
- Supports manual review and verification
- Ensures all operations are consent-based

### 5. Human Review Interface
- Side-by-side image comparison
- Visual similarity scores with explanations
- Confidence indicators
- Review history tracking

## Usage Examples

### Basic Face Detection

```python
from src.detection import FaceDetector
import cv2

# Initialize detector
detector = FaceDetector()

# Load image
image = cv2.imread("test_image.jpg")

# Detect faces
faces = detector.detect_faces(image)
print(f"Detected {len(faces)} faces")

# Draw detections
result_image = detector.draw_detections(image, faces)
cv2.imshow("Face Detection", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Embedding Extraction and Comparison

```python
from src.embedding import FaceNetEmbeddingExtractor, SimilarityComparator

# Initialize components
extractor = FaceNetEmbeddingExtractor()
comparator = SimilarityComparator()

# Extract embedding from face image
face_image = cv2.imread("face.jpg")
embedding = extractor.extract_embedding(face_image)

# Compare with reference embedding
reference_embedding = np.random.rand(128)  # Load from reference
results = comparator.compare_embeddings(
    embedding, [reference_embedding], ["reference_id"]
)

for ref_id, similarity in results:
    confidence = comparator.get_confidence_band(similarity)
    print(f"{ref_id}: {similarity:.2f} ({confidence})")
```

### Webcam Demo

```python
from utils.webcam import WebcamCapture

# Initialize webcam
webcam = WebcamCapture()
webcam.initialize()

# Process live video (30 seconds)
reference_embeddings = [np.random.rand(128)]  # Load your references
reference_ids = ["reference_1"]
webcam.process_live_video(reference_embeddings, reference_ids, duration=30)
webcam.close()
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