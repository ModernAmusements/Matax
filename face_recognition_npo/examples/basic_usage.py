from src.detection import FaceDetector
from src.embedding import FaceNetEmbeddingExtractor, SimilarityComparator
from src.reference import ReferenceImageManager, HumanReviewInterface
from utils.webcam import WebcamCapture

# Initialize components
detector = FaceDetector()
extractor = FaceNetEmbeddingExtractor()
comparator = SimilarityComparator()
manager = ReferenceImageManager()
review_interface = HumanReviewInterface()

# Test initialization
print("FaceDetector initialized")
print("FaceNetEmbeddingExtractor initialized")
print("SimilarityComparator initialized")
print("ReferenceImageManager initialized")
print("HumanReviewInterface initialized")

# Test webcam (optional)
try:
    webcam = WebcamCapture()
    webcam.initialize()
    print("WebcamCapture initialized")
    webcam.close()
except Exception as e:
    print(f"Webcam test skipped: {e}")

print("All modules loaded successfully!")