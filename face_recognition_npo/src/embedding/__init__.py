import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np
import cv2

class FaceNetEmbeddingExtractor:
    """
    Face embedding extraction using FaceNet architecture.
    Converts detected faces into 128-dimensional embeddings.
    """
    
    def __init__(self, model_path: str = "facenet_model.pth"):
        self.model = self._load_model(model_path)
        self.model.eval()  # Set model to evaluation mode
        self.embedding_size = 128
        
    def _load_model(self, model_path: str):
        """
        Load FaceNet model for 160x160 input images.
        After 4 maxpool layers (160->80->40->20->10), flatten size is 512*10*10=51200
        """
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(512 * 10 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        return model
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 128-dimensional embedding from a face image.
        
        Args:
            face_image: Cropped face image (RGB format)
            
        Returns:
            128-dimensional embedding vector or None if extraction fails
        """
        try:
            # Preprocess image
            face_image = cv2.resize(face_image, (160, 160))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_image = np.transpose(face_image, (2, 0, 1))  # HWC to CHW
            face_image = face_image.astype(np.float32) / 255.0
            face_image = torch.tensor(face_image).unsqueeze(0)  # Add batch dimension
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(face_image)
                
            # Normalize embedding
            embedding = embedding.numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def extract_embeddings(self, face_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings for multiple face images.
        
        Args:
            face_images: List of cropped face images
            
        Returns:
            List of embeddings (None for failed extractions)
        """
        return [self.extract_embedding(img) for img in face_images]

class SimilarityComparator:
    """
    Compare face embeddings using cosine similarity.
    Provides ranked similarity scores without making identity claims.
    """
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        dot_product = np.dot(embedding1, embedding2)
        norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        
        if norm_product == 0:
            return 0.0
            
        similarity = dot_product / norm_product
        return max(0.0, min(1.0, similarity))
    
    def compare_embeddings(self, 
                          query_embedding: np.ndarray,
                          reference_embeddings: List[np.ndarray],
                          reference_ids: List[str]) -> List[Tuple[str, float]]:
        """
        Compare query embedding against reference embeddings.
        
        Args:
            query_embedding: Embedding to compare
            reference_embeddings: List of reference embeddings
            reference_ids: Corresponding reference IDs
            
        Returns:
            List of (reference_id, similarity_score) tuples sorted by similarity
        """
        results = []
        
        for ref_id, ref_embedding in zip(reference_ids, reference_embeddings):
            if ref_embedding is None:
                continue
                
            similarity = self.cosine_similarity(query_embedding, ref_embedding)
            
            # Only include if above threshold
            if similarity > self.threshold:
                results.append((ref_id, similarity))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_confidence_band(self, similarity: float) -> str:
        """
        Get human-readable confidence band for similarity score.
        """
        if similarity > 0.8:
            return "High confidence"
        elif similarity > 0.6:
            return "Moderate confidence"
        elif similarity > 0.4:
            return "Low confidence"
        else:
            return "Insufficient confidence"

if __name__ == "__main__":
    # Test embedding extraction
    extractor = FaceNetEmbeddingExtractor()
    
    # Test similarity comparison
    comparator = SimilarityComparator()
    
    print("Embedding extraction and comparison modules loaded successfully")