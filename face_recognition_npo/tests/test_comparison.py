import unittest
from src.comparison import SimilarityComparator

class TestSimilarityComparison(unittest.TestCase):
    """
    Test similarity comparison functionality.
    """
    
    def setUp(self):
        self.comparator = SimilarityComparator()
        
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Identical vectors should have similarity 1.0
        vec1 = [1.0] * 128
        vec2 = [1.0] * 128
        similarity = self.comparator.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0, places=2)
        
        # Opposite vectors should have similarity -1.0
        vec1 = [1.0] * 128
        vec2 = [-1.0] * 128
        similarity = self.comparator.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, -1.0, places=2)
    
    def test_compare_embeddings(self):
        """Test comparison of embeddings."""
        # Create two similar embeddings
        embedding1 = [0.5] * 128
        embedding2 = [0.5] * 128
        
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