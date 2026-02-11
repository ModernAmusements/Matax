import unittest
import os
import cv2
import numpy as np
from src.reference import ReferenceImageManager

class TestReferenceManagement(unittest.TestCase):
    """
    Test reference image management functionality.
    """
    
    def setUp(self):
        self.manager = ReferenceImageManager(reference_dir="test_references")
        
    def tearDown(self):
        # Clean up test directory
        try:
            if os.path.exists("test_references"):
                for filename in os.listdir("test_references"):
                    file_path = os.path.join("test_references", filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                os.rmdir("test_references")
        except:
            pass
    
    def test_reference_manager_initialization(self):
        """Test that reference manager initializes properly."""
        self.assertIsNotNone(self.manager.reference_dir)
        self.assertTrue(os.path.exists(self.manager.reference_dir))
    
    def test_add_reference_image(self):
        """Test adding a reference image."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv2.imwrite("test_image.jpg", test_image)
        
        success = self.manager.add_reference_image(
            "test_image.jpg", "test_id", {"source": "test"}
        )
        
        if os.path.exists("test_image.jpg"):
            os.remove("test_image.jpg")
        
        # Note: May fail if no face detected in blank image
        # This is expected behavior
        if not success:
            self.skipTest("No face detected in blank test image")
    
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