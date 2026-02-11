import unittest
import numpy as np
import cv2
import os
from src.detection import FaceDetector
from src.embedding import FaceNetEmbeddingExtractor, SimilarityComparator
from src.reference import ReferenceImageManager, HumanReviewInterface

class TestFaceDetection(unittest.TestCase):
    """
    Test face detection functionality.
    """
    
    def setUp(self):
        self.detector = FaceDetector()
        
    def test_detector_initialization(self):
        """Test that detector initializes properly."""
        self.assertIsNotNone(self.detector.net)
        self.assertEqual(self.detector.confidence_threshold, 0.5)
    
    def test_detect_faces_empty_image(self):
        """Test detection on empty image."""
        empty_image = np.zeros((200, 200, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(empty_image)
        self.assertEqual(len(faces), 0)
    
    def test_visualize_detection(self):
        """Test that visualize_detection doesn't crash."""
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        faces = [(50, 50, 100, 100)]  # One face
        result_image = self.detector.visualize_detection(test_image, faces)
        self.assertIsNotNone(result_image)
        self.assertEqual(result_image.shape, (200, 200, 3))

class TestEmbeddingExtraction(unittest.TestCase):
    """
    Test face embedding extraction functionality.
    """
    
    def setUp(self):
        self.extractor = FaceNetEmbeddingExtractor()
        
    def test_extractor_initialization(self):
        """Test that extractor initializes properly."""
        self.assertEqual(self.extractor.embedding_size, 128)
        self.assertIsNotNone(self.extractor.model)
    
    def test_extract_embedding_valid_face(self):
        """Test embedding extraction on a valid face image."""
        # Create a simple face-like image
        face_image = np.ones((160, 160, 3), dtype=np.uint8) * 128
        embedding = self.extractor.extract_embedding(face_image)
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 128)
    
    def test_extract_embedding_invalid_image(self):
        """Test embedding extraction on very small image."""
        invalid_image = np.zeros((10, 10, 3), dtype=np.uint8)
        embedding = self.extractor.extract_embedding(invalid_image)
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 128)

class TestSimilarityComparison(unittest.TestCase):
    """
    Test similarity comparison functionality.
    """
    
    def setUp(self):
        self.comparator = SimilarityComparator()
        
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Identical vectors should have similarity 1.0
        vec1 = np.ones(128)
        vec2 = np.ones(128)
        similarity = self.comparator.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0, places=2)
        
        # Opposite vectors should have similarity -1.0
        vec1 = np.ones(128)
        vec2 = -np.ones(128)
        similarity = self.comparator.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, -1.0, places=2)
    
    def test_compare_embeddings(self):
        """Test comparison of embeddings."""
        # Create two similar embeddings
        embedding1 = np.random.rand(128)
        embedding2 = embedding1 + np.random.normal(0, 0.1, 128)
        
        results = self.comparator.compare_embeddings(
            embedding1, [embedding2], ["test_id"]
        )
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0][1], 0.5)  # Should be reasonably similar
    
    def test_get_confidence_band(self):
        """Test confidence band determination."""
        self.assertEqual(self.comparator.get_confidence_band(0.9), "High confidence")
        self.assertEqual(self.comparator.get_confidence_band(0.7), "Moderate confidence")
        self.assertEqual(self.comparator.get_confidence_band(0.5), "Low confidence")
        self.assertEqual(self.comparator.get_confidence_band(0.3), "Insufficient confidence")

class TestReferenceManagement(unittest.TestCase):
    """
    Test reference image management functionality.
    """
    
    def setUp(self):
        self.manager = ReferenceImageManager(reference_dir="test_references")
        
    def tearDown(self):
        # Clean up test directory
        if os.path.exists("test_references"):
            for filename in os.listdir("test_references"):
                file_path = os.path.join("test_references", filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            os.rmdir("test_references")
    
    def test_reference_manager_initialization(self):
        """Test that reference manager initializes properly."""
        self.assertIsNotNone(self.manager.reference_dir)
        self.assertTrue(os.path.exists(self.manager.reference_dir))
    
    def test_add_reference_image(self):
        """Test adding a reference image."""
        # Create a test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv2.imwrite("test_image.jpg", test_image)
        
        success = self.manager.add_reference_image(
            "test_image.jpg", "test_id", {"source": "test"}
        )
        
        self.assertTrue(success)
        self.assertEqual(len(self.manager.list_references()), 1)
        
        # Clean up
        os.remove("test_image.jpg")
    
    def test_get_reference_metadata(self):
        """Test getting reference metadata."""
        # Add test reference
        self.manager.reference_data["metadata"].append({
            "id": "test_id",
            "path": "test_path",
            "metadata": {"source": "test"},
            "added_at": "2024-01-01"
        })
        
        metadata = self.manager.get_reference_metadata("test_id")
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["id"], "test_id")
    
    def test_remove_reference(self):
        """Test removing a reference."""
        # Add test reference
        self.manager.reference_data["metadata"].append({
            "id": "test_id",
            "path": "test_path",
            "metadata": {"source": "test"},
            "added_at": "2024-01-01"
        })
        
        success = self.manager.remove_reference("test_id")
        self.assertTrue(success)
        self.assertEqual(len(self.manager.list_references()), 0)

class TestHumanReviewInterface(unittest.TestCase):
    """
    Test human review interface functionality.
    """
    
    def setUp(self):
        self.review_interface = HumanReviewInterface()
        
    def test_review_interface_initialization(self):
        """Test that review interface initializes properly."""
        self.assertIsNotNone(self.review_interface.review_history)
        self.assertEqual(len(self.review_interface.review_history), 0)
    
    def test_get_confidence_text(self):
        """Test confidence text generation."""
        self.assertEqual(self.review_interface._get_confidence_text(0.9), "HIGH CONFIDENCE")
        self.assertEqual(self.review_interface._get_confidence_text(0.7), "MODERATE CONFIDENCE")
        self.assertEqual(self.review_interface._get_confidence_text(0.5), "LOW CONFIDENCE")
        self.assertEqual(self.review_interface._get_confidence_text(0.3), "INSUFFICIENT CONFIDENCE")

if __name__ == "__main__":
    unittest.main()