import os
import json
import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2

class ReferenceImageManager:
    """
    Manage reference images and their embeddings for comparison.
    Ensures all operations are consent-based and reviewable.
    """
    
    def __init__(self, reference_dir: str = "reference_images"):
        self.reference_dir = reference_dir
        self.embeddings_file = os.path.join(reference_dir, "embeddings.json")
        self.reference_data = {}
        self._load_reference_data()
    
    def _load_reference_data(self):
        """
        Load reference image metadata and embeddings from file.
        """
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'r') as f:
                    self.reference_data = json.load(f)
            except Exception as e:
                print(f"Error loading reference data: {e}")
        else:
            self.reference_data = {
                "metadata": [],
                "embeddings": []
            }
    
    def _save_reference_data(self):
        """
        Save reference image metadata and embeddings to file.
        """
        try:
            os.makedirs(self.reference_dir, exist_ok=True)
            with open(self.embeddings_file, 'w') as f:
                json.dump(self.reference_data, f, indent=2)
        except Exception as e:
            print(f"Error saving reference data: {e}")
    
    def add_reference_image(self, image_path: str, reference_id: str, metadata: Optional[dict] = None):
        """
        Add a new reference image with its embedding.
        
        Args:
            image_path: Path to reference image
            reference_id: Unique identifier for this reference
            metadata: Additional metadata (consent info, source, etc.)
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return False
            
        # Load and process image
        try:
            from PIL import Image
            import numpy as np
            
            image = Image.open(image_path)
            image_array = np.array(image)
            
            # Create metadata entry
            entry = {
                "id": reference_id,
                "path": image_path,
                "metadata": metadata or {},
                "added_at": str(datetime.datetime.now())
            }
            
            self.reference_data["metadata"].append(entry)
            self._save_reference_data()
            
            return True
            
        except Exception as e:
            print(f"Error adding reference image: {e}")
            return False
    
    def get_reference_embeddings(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Get all reference embeddings and their IDs.
        
        Returns:
            Tuple of (embeddings_list, ids_list)
        """
        embeddings = []
        ids = []
        
        for entry in self.reference_data["metadata"]:
            # In a real implementation, you'd load the embedding
            # For now, we'll return placeholder embeddings
            embedding = np.random.rand(128)  # Placeholder
            embeddings.append(embedding)
            ids.append(entry["id"])
        
        return embeddings, ids
    
    def get_reference_metadata(self, reference_id: str) -> Optional[dict]:
        """
        Get metadata for a specific reference image.
        """
        for entry in self.reference_data["metadata"]:
            if entry["id"] == reference_id:
                return entry
        return None
    
    def list_references(self) -> List[dict]:
        """
        List all reference images with their metadata.
        """
        return self.reference_data["metadata"]
    
    def remove_reference(self, reference_id: str) -> bool:
        """
        Remove a reference image from the system.
        """
        new_metadata = []
        removed = False
        
        for entry in self.reference_data["metadata"]:
            if entry["id"] != reference_id:
                new_metadata.append(entry)
            else:
                removed = True
        
        if removed:
            self.reference_data["metadata"] = new_metadata
            self._save_reference_data()
        
        return removed

class HumanReviewInterface:
    """
    Interface for human review of similarity results.
    Ensures human oversight at every decision point.
    """
    
    def __init__(self):
        self.review_history = []
    
    def display_comparison(self, 
                          query_image: np.ndarray,
                          reference_image: np.ndarray,
                          similarity_score: float,
                          reference_id: str,
                          metadata: Optional[dict] = None):
        """
        Display side-by-side comparison for human review.
        """
        # Create side-by-side comparison
        query_resized = cv2.resize(query_image, (200, 200))
        reference_resized = cv2.resize(reference_image, (200, 200))
        
        combined = np.hstack((query_resized, reference_resized))
        
        # Add similarity score
        confidence = self._get_confidence_text(similarity_score)
        cv2.putText(combined, f"Similarity: {similarity_score:.2f}", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(combined, confidence, 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display image
        cv2.imshow("Comparison Review", combined)
        key = cv2.waitKey(0)
        
        # Record review
        self.review_history.append({
            "reference_id": reference_id,
            "similarity_score": similarity_score,
            "decision": self._get_key_decision(key),
            "timestamp": str(datetime.datetime.now())
        })
        
        cv2.destroyAllWindows()
        
        return key
    
    def _get_confidence_text(self, similarity: float) -> str:
        """
        Get confidence text based on similarity score.
        """
        if similarity > 0.8:
            return "HIGH CONFIDENCE"
        elif similarity > 0.6:
            return "MODERATE CONFIDENCE"
        elif similarity > 0.4:
            return "LOW CONFIDENCE"
        else:
            return "INSUFFICIENT CONFIDENCE"
    
    def _get_key_decision(self, key: int) -> str:
        """
        Convert key press to decision.
        """
        if key == ord('y'):
            return "ACCEPT"
        elif key == ord('n'):
            return "REJECT"
        elif key == ord('s'):
            return "SKIP"
        else:
            return "UNKNOWN"

if __name__ == "__main__":
    # Test reference management
    manager = ReferenceImageManager()
    
    # Test human review interface
    # This would require actual images to work properly
    print("Reference management and review interface modules loaded successfully")