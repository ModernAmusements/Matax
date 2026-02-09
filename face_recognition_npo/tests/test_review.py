import unittest
from src.reference import HumanReviewInterface

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