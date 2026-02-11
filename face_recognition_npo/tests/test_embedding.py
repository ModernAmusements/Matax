import unittest
import numpy as np
from src.embedding import FaceNetEmbeddingExtractor, SimilarityComparator

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

if __name__ == "__main__":
    unittest.main()