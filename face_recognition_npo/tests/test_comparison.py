import unittest
import numpy as np
from src.embedding import SimilarityComparator

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
        embedding1 = np.ones(128) * 0.5
        embedding2 = np.ones(128) * 0.5
        
        results = self.comparator.compare_embeddings(
            embedding1, [embedding2], ["test_id"]
        )
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0]['similarity'], 0.5)  # Should be reasonably similar
    
    def test_get_confidence_band(self):
        """Test confidence band determination."""
        self.assertEqual(self.comparator.get_confidence_band(0.98), "Very High")
        self.assertEqual(self.comparator.get_confidence_band(0.90), "High")
        self.assertEqual(self.comparator.get_confidence_band(0.75), "Moderate")
        self.assertEqual(self.comparator.get_confidence_band(0.55), "Low")
        self.assertEqual(self.comparator.get_confidence_band(0.30), "Insufficient")

    def test_get_verdict(self):
        """Test verdict determination."""
        self.assertEqual(self.comparator.get_verdict(0.98), "Likely same person")
        self.assertEqual(self.comparator.get_verdict(0.90), "Possibly same person")
        self.assertEqual(self.comparator.get_verdict(0.60), "Uncertain - human review required")
        self.assertEqual(self.comparator.get_verdict(0.30), "Likely different people")

    def test_euclidean_distance(self):
        """Test euclidean distance calculation."""
        emb1 = np.ones(128)
        emb2 = np.zeros(128)
        distance = self.comparator.euclidean_distance(emb1, emb2)
        self.assertAlmostEqual(distance, np.sqrt(128), places=5)

    def test_get_distance_verdict(self):
        """Test distance verdict determination."""
        self.assertEqual(self.comparator.get_distance_verdict(0.20), "Very close (same person likely)")
        self.assertEqual(self.comparator.get_distance_verdict(0.50), "Moderate distance")
        self.assertEqual(self.comparator.get_distance_verdict(0.80), "Large distance (different people likely)")

    def test_compare_embeddings_with_metrics(self):
        """Test compare_embeddings returns both similarity and distance."""
        emb1 = np.random.rand(128)
        emb2 = emb1.copy()
        emb3 = np.random.rand(128)
        results = self.comparator.compare_embeddings(emb1, [emb2, emb3], ["same", "different"])
        self.assertEqual(len(results), 2)
        self.assertIn('similarity', results[0])
        self.assertIn('euclidean_distance', results[0])
        self.assertIn('confidence', results[0])
        self.assertIn('verdict', results[0])

if __name__ == "__main__":
    unittest.main()