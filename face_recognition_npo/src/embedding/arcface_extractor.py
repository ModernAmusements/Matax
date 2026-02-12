import numpy as np
import cv2
from typing import Optional, Dict, Tuple, List
from insightface.app import FaceAnalysis


class ArcFaceEmbeddingExtractor:
    """ArcFace-based embedding extractor using InsightFace FaceAnalysis app.
    
    Model options:
        - 'buffalo_l': ResNet100, 512-dim, most accurate (default)
        - 'buffalo_m': ResNet50, 512-dim, faster
    
    Uses FaceAnalysis app which provides:
        - Face detection
        - Face recognition (512-dim embedding)
        - Face alignment
        - Gender/Age estimation
    """
    
    MODEL_NAME = 'buffalo_l'
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.embedding_dim = 512
        self.detector = None  # Uses built-in detection
        
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract 512-dim embedding from face image.
        
        Args:
            face_image: Face crop in BGR format (any size)
                        The API passes cropped face ROIs
            
        Returns:
            512-dim embedding vector or None on error
        """
        try:
            h, w = face_image.shape[:2]
            
            # Always use recognition model forward pass for pre-cropped faces
            # This avoids re-detection which may fail on cropped images
            
            # Resize to 112x112 (ArcFace input size)
            face_resized = cv2.resize(face_image, (112, 112))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_input = np.transpose(face_rgb, (2, 0, 1))  # HWC to CHW
            face_input = np.expand_dims(face_input, axis=0).astype(np.float32)
            
            # Use recognition model forward pass
            rec_model = self.app.models['recognition']
            embedding = rec_model.forward(face_input)
            return embedding.flatten()
            
        except Exception as e:
            print(f"ArcFace extraction error: {e}")
            return None
            
    def extract_embeddings(self, face_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Extract embeddings from multiple face images."""
        return [self.extract_embedding(img) for img in face_images]
        
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image using InsightFace detector.
        
        Args:
            image: Full image in BGR format
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        try:
            faces = self.app.get(image)
            bboxes = []
            for face in faces:
                bbox = face['bbox']  # [x1, y1, x2, y2]
                x, y, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                w, h = x2 - x, y2 - y
                bboxes.append((x, y, w, h))
            return bboxes
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
        
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        if norm_product == 0:
            return 0.0
        return float(dot_product / norm_product)
        
    def euclidean_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate Euclidean distance between two embeddings."""
        return float(np.linalg.norm(embedding1 - embedding2))
        
    def compare_embeddings(self, query_embedding: np.ndarray, 
                          reference_embeddings: List[np.ndarray], 
                          reference_ids: List[str]) -> List[Dict]:
        """Compare query embedding against references."""
        results = []
        for ref_id, ref_embedding in zip(reference_ids, reference_embeddings):
            if ref_embedding is None:
                continue
            similarity = self.cosine_similarity(query_embedding, ref_embedding)
            distance = self.euclidean_distance(query_embedding, ref_embedding)
            confidence = self.get_confidence_band(similarity)
            verdict = self.get_verdict(similarity)
            distance_verdict = self.get_distance_verdict(distance)
            results.append({
                'id': ref_id,
                'similarity': similarity,
                'euclidean_distance': distance,
                'confidence': confidence,
                'verdict': verdict,
                'distance_verdict': distance_verdict
            })
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results
        
    def get_confidence_band(self, similarity: float, 
                           threshold_high: float = 0.70,
                           threshold_moderate: float = 0.45,
                           threshold_low: float = 0.30) -> str:
        """ArcFace uses lower thresholds since embeddings are more discriminative.
        
        Expected scores:
        - Same person: ~70-80%
        - Different people: ~20-30%
        """
        if similarity >= threshold_high:
            return "Very High"
        elif similarity >= threshold_moderate:
            return "High"
        elif similarity >= threshold_low:
            return "Moderate"
        elif similarity >= 0.20:
            return "Low"
        else:
            return "Insufficient"
            
    def get_verdict(self, similarity: float,
                   threshold_high: float = 0.70,
                   threshold_moderate: float = 0.45) -> str:
        """Return human-readable verdict.
        
        ArcFace is more discriminative, so lower thresholds are appropriate.
        """
        if similarity >= threshold_high:
            return "Likely same person"
        elif similarity >= threshold_moderate:
            return "Possibly same person"
        elif similarity >= 0.20:
            return "Uncertain - human review required"
        else:
            return "Likely different people"
            
    def get_distance_verdict(self, distance: float,
                            threshold_low: float = 0.8,
                            threshold_moderate: float = 1.2) -> str:
        """Return verdict based on Euclidean distance."""
        if distance <= threshold_low:
            return "MATCH (same person likely)"
        elif distance <= threshold_moderate:
            return "POSSIBLE (human review recommended)"
        else:
            return "NO MATCH (different people)"
            
    def visualize_embedding(self, embedding: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Visualize 512-dim embedding as heatmap.
        
        Args:
            embedding: 512-dim embedding vector
            
        Returns:
            Tuple of (visualization_image, data_dict)
        """
        dim = len(embedding)
        
        output = np.zeros((200, 600, 3), dtype=np.uint8)
        output.fill(245)
        
        cv2.putText(output, f"ArcFace {dim}-Dim Embedding", (20, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        cv2.putText(output, f"Mean: {np.mean(embedding):.4f}  Std: {np.std(embedding):.4f}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        
        if dim == 512:
            embedding_2d = embedding.reshape((16, 32))
            embedding_norm = (embedding_2d - embedding_2d.min()) / (embedding_2d.max() - embedding_2d.min() + 1e-8)
            viz = (embedding_norm * 255).astype(np.uint8)
            viz_color = cv2.applyColorMap(viz, cv2.COLORMAP_VIRIDIS)
            viz_resized = cv2.resize(viz_color, (560, 120))
            output[60:180, 20:580] = viz_resized
            
        data = {
            'mean': float(np.mean(embedding)),
            'std': float(np.std(embedding)),
            'norm': float(np.linalg.norm(embedding)),
            'min': float(np.min(embedding)),
            'max': float(np.max(embedding)),
            'dim': dim
        }
        
        return output, data
        
    def visualize_similarity_matrix(self, query_embedding: np.ndarray,
                                   reference_embeddings: List[np.ndarray],
                                   reference_ids: List[str]) -> Tuple[np.ndarray, Dict]:
        """Visualize similarity scores as a matrix."""
        n = len(reference_ids)
        height = max(150, n * 50 + 80)
        output = np.zeros((height, 500, 3), dtype=np.uint8)
        output.fill(245)
        
        cv2.putText(output, "Similarity Comparison", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        bar_width = 280
        bar_height = 20
        
        data = {'comparisons': [], 'best_score': 0.0}
        
        for i, (ref_id, ref_emb) in enumerate(zip(reference_ids, reference_embeddings)):
            if ref_emb is None:
                continue
            similarity = self.cosine_similarity(query_embedding, ref_emb)
            data['comparisons'].append({'id': ref_id, 'similarity': similarity})
            if similarity > data['best_score']:
                data['best_score'] = similarity
                
        for i, (ref_id, ref_emb) in enumerate(zip(reference_ids, reference_embeddings)):
            if ref_emb is None:
                continue
            similarity = self.cosine_similarity(query_embedding, ref_emb)
            y = 60 + i * 45
            
            cv2.putText(output, ref_id[:18], (10, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
            
            bar_len = int(similarity * bar_width)
            
            if similarity > 0.70:
                color = (0, 200, 0)
            elif similarity > 0.45:
                color = (0, 165, 255)
            elif similarity > 0.30:
                color = (0, 100, 255)
            else:
                color = (0, 0, 150)
                
            cv2.rectangle(output, (100, y), (100 + bar_len, y + bar_height), color, -1)
            cv2.rectangle(output, (100, y), (100 + bar_width, y + bar_height), (200, 200, 200), 1)
            
            cv2.putText(output, f"{similarity:.2f}", (390, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
        return output, data
        
    def visualize_similarity_result(self, query_embedding: np.ndarray,
                                   reference_embedding: np.ndarray = None,
                                   similarity: float = 0.75) -> np.ndarray:
        """Visualize a single similarity result."""
        output = np.zeros((100, 300, 3), dtype=np.uint8)
        output.fill(245)
        
        cv2.putText(output, "Similarity Result", (20, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        bar_width = 200
        bar_height = 25
        start_x = 50
        start_y = 50
        
        if similarity > 0.70:
            color = (0, 200, 0)
        elif similarity > 0.45:
            color = (0, 165, 255)
        elif similarity > 0.30:
            color = (0, 100, 255)
        else:
            color = (0, 0, 150)
            
        bar_len = int(similarity * bar_width)
        cv2.rectangle(output, (start_x, start_y), (start_x + bar_len, start_y + bar_height), color, -1)
        cv2.rectangle(output, (start_x, start_y), (start_x + bar_width, start_y + bar_height), (200, 200, 200), 1)
        
        if similarity > 0.70:
            confidence = "High confidence"
        elif similarity > 0.45:
            confidence = "Moderate confidence"
        elif similarity > 0.30:
            confidence = "Low confidence"
        else:
            confidence = "Insufficient confidence"
            
        cv2.putText(output, f"{similarity:.2f} - {confidence}", (start_x, start_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        
        return output


if __name__ == "__main__":
    extractor = ArcFaceEmbeddingExtractor()
    print(f"ArcFace model loaded: {extractor.model_name}")
    print(f"Embedding dimension: {extractor.embedding_dim}")
    print("ArcFace extractor module loaded successfully")
