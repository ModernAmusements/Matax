import unittest
from src.reference import ReferenceImageManager

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
        test_image = [[128, 128, 128]] * 100  # Simple grayscale image
        
        success = self.manager.add_reference_image(
            "test_image.jpg", "test_id", {"source": "test"}
        )
        
        self.assertTrue(success)
        self.assertEqual(len(self.manager.list_references()), 1)
    
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

if __name__ == "__main__":
    unittest.main()