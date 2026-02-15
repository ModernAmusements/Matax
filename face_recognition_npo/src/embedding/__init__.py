import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as torchvision_models
import numpy as np
from typing import List, Tuple, Optional, Dict


class ImprovedEmbeddingExtractor(nn.Module):
    def __init__(self, embedding_size=128, backbone='resnet18', pretrained=True):
        super(ImprovedEmbeddingExtractor, self).__init__()
        
        if backbone == 'resnet18':
            base_model = torchvision_models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            feat_dim = 512
        elif backbone == 'mobilenet_v3_large':
            base_model = torchvision_models.mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
            feat_dim = 960
        elif backbone == 'efficientnet_b0':
            base_model = torchvision_models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            feat_dim = 1280
        else:
            base_model = torchvision_models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            feat_dim = 512
        
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        
        self.embedding_size = embedding_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.eval()
        
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            embedding = self.embedding(features)
            embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


class FaceNetEmbeddingExtractor:
    def __init__(self, embedding_size=128, backbone='resnet18'):
        self.embedding_size = embedding_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ImprovedEmbeddingExtractor(embedding_size, backbone).to(self.device)
        self.model.eval()
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
    def preprocess(self, face_image: np.ndarray) -> torch.Tensor:
        face_image = cv2.resize(face_image, (224, 224))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = face_image.astype(np.float32) / 255.0
        face_image = (face_image - self.mean) / self.std
        face_image = torch.from_numpy(face_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        return face_image

    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            input_tensor = self.preprocess(face_image)
            with torch.no_grad():
                embedding = self.model(input_tensor)
                embedding = embedding.cpu().numpy().flatten()
            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

    def extract_embeddings(self, face_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        return [self.extract_embedding(img) for img in face_images]

    def get_activations(self, face_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract neural network activations from multiple layers.
        Returns activation maps from different stages of the network.
        
        Args:
            face_image: Input face image (BGR format, any size)
            
        Returns:
            Dictionary mapping layer names to activation arrays
        """
        activations = {}
        
        if face_image is None or face_image.size == 0:
            return activations
        
        try:
            face_tensor = self.preprocess(face_image)
            backbone = self.model.backbone
            x = face_tensor
            
            layer_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'gap', 'fc']
            layer_index = 0
            
            with torch.no_grad():
                for i, child in enumerate(backbone.children()):
                    x = child(x)
                    
                    if layer_index < len(layer_names):
                        act = x.detach().cpu().numpy()[0]
                        activations[layer_names[layer_index]] = act
                        layer_index += 1
                    
                    if layer_index >= len(layer_names):
                        break
                
                if layer_index < len(layer_names):
                    remaining = len(layer_names) - layer_index
                    final_act = x.detach().cpu().numpy()[0]
                    for j in range(remaining):
                        activations[layer_names[layer_index + j]] = final_act
                
                activations['embedding'] = self.extract_embedding(face_image)
                
        except Exception as e:
            print(f"Error extracting activations: {e}")
        
        return activations

    def visualize_embedding(self, embedding: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Visualize the embedding as a heatmap/bar chart.
        
        Args:
            embedding: 128-dimensional face embedding
            
        Returns:
            Tuple of (visualization_image, data_dict)
        """
        output = np.zeros((200, 400, 3), dtype=np.uint8)
        output.fill(245)
        
        data = {}
        
        if embedding is None:
            cv2.putText(output, "No embedding available", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 0, 0), 2)
            return output, data
        
        data['mean'] = float(np.mean(embedding))
        data['std'] = float(np.std(embedding))
        data['norm'] = float(np.linalg.norm(embedding))
        data['min'] = float(np.min(embedding))
        data['max'] = float(np.max(embedding))
        
        cv2.putText(output, "128-Dim Face Embedding", (20, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        cv2.putText(output, f"Mean: {data['mean']:.4f}  Std: {data['std']:.4f}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        
        bar_height = 12
        bar_width = 280
        start_x = 50
        start_y = 80
        
        num_values = len(embedding)
        step = max(1, num_values // 16)
        
        for i in range(0, num_values, step):
            value = embedding[i]
            normalized = (value - embedding.min()) / (embedding.max() - embedding.min() + 1e-8)
            bar_len = int(normalized * bar_width)
            
            y = start_y + (i // step) * (bar_height + 2)
            if y > 180:
                break
                
            color_val = int(normalized * 255)
            color = (255 - color_val, color_val, 100)
            
            cv2.rectangle(output, (start_x, y), (start_x + bar_len, y + bar_height), color, -1)
        
        return output, data

    def visualize_similarity_matrix(self, query_embedding: np.ndarray, reference_embeddings: List[np.ndarray], reference_ids: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Visualize similarity scores between query and references as a matrix.
        
        Args:
            query_embedding: The query embedding to compare
            reference_embeddings: List of reference embeddings
            reference_ids: List of reference identifiers
            
        Returns:
            Tuple of (visualization_image, data_dict)
        """
        n = len(reference_embeddings)
        if n == 0:
            output = np.zeros((100, 200, 3), dtype=np.uint8)
            output.fill(245)
            cv2.putText(output, "No references", (30, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 0, 0), 2)
            return output, {}
        
        output_size = max(150, n * 50)
        output = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        output.fill(245)
        
        data = {'similarities': [], 'best_match': None, 'best_score': 0}
        
        similarities = []
        for i, (ref_emb, ref_id) in enumerate(zip(reference_embeddings, reference_ids)):
            if ref_emb is None:
                similarity = 0.0
            else:
                dot_prod = np.dot(query_embedding, ref_emb)
                norm_prod = np.linalg.norm(query_embedding) * np.linalg.norm(ref_emb) + 1e-8
                similarity = float(dot_prod / norm_prod)
            similarities.append((ref_id, similarity))
            
            if similarity > data['best_score']:
                data['best_score'] = similarity
                data['best_match'] = ref_id
        
        data['similarities'] = similarities
        
        cv2.putText(output, "Similarity Matrix", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        bar_height = 30
        bar_width = output_size - 120
        start_x = 110
        start_y = 50
        
        for i, (ref_id, similarity) in enumerate(similarities):
            y = start_y + i * (bar_height + 5)
            
            cv2.putText(output, ref_id[:12], (10, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
            
            bar_len = int(similarity * bar_width)
            
            if similarity > 0.95:
                color = (0, 200, 0)
            elif similarity > 0.85:
                color = (0, 165, 255)
            elif similarity > 0.70:
                color = (0, 100, 255)
            else:
                color = (0, 0, 150)
            
            cv2.rectangle(output, (start_x, y), (start_x + bar_len, y + bar_height), color, -1)
            cv2.rectangle(output, (start_x, y), (start_x + bar_width, y + bar_height), (200, 200, 200), 1)
            
            cv2.putText(output, f"{similarity:.2f}", (start_x + bar_width + 10, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return output, data

    def visualize_similarity_result(self, query_embedding: np.ndarray, reference_embedding: np.ndarray = None, similarity: float = 0.75) -> np.ndarray:
        """
        Visualize a single similarity result.
        
        Args:
            query_embedding: The query embedding
            reference_embedding: The reference embedding (optional)
            similarity: Similarity score
            
        Returns:
            Visualization image
        """
        output = np.zeros((100, 300, 3), dtype=np.uint8)
        output.fill(245)
        
        cv2.putText(output, "Similarity Result", (20, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        bar_width = 200
        bar_height = 25
        start_x = 50
        start_y = 50
        
        if similarity > 0.95:
            color = (0, 200, 0)
        elif similarity > 0.85:
            color = (0, 165, 255)
        elif similarity > 0.70:
            color = (0, 100, 255)
        else:
            color = (0, 0, 150)
        
        bar_len = int(similarity * bar_width)
        cv2.rectangle(output, (start_x, start_y), (start_x + bar_len, start_y + bar_height), color, -1)
        cv2.rectangle(output, (start_x, start_y), (start_x + bar_width, start_y + bar_height), (200, 200, 200), 1)
        
        if similarity > 0.95:
            confidence = "High confidence"
        elif similarity > 0.85:
            confidence = "Moderate confidence"
        elif similarity > 0.70:
            confidence = "Low confidence"
        else:
            confidence = "Insufficient confidence"
        
        cv2.putText(output, f"{similarity:.2f} - {confidence}", (start_x, start_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        
        return output

    def visualize_activations(self, face_image: np.ndarray, max_channels: int = 8) -> np.ndarray:
        """
        Visualize neural network activations for the first convolutional layer.
        Shows the intermediate activations of the network when processing the face.
        """
        try:
            import torch

            if face_image is None or face_image.size == 0:
                raise ValueError("Face image is None or empty")

            # Get backbone from model
            backbone = self.model.backbone

            # Preprocess face image
            face_tensor = self.preprocess(face_image)

            # Get first layer activations using the actual model structure
            x = face_tensor

            with torch.no_grad():
                # Go through first few layers to get activations
                activations = None
                for i, child in enumerate(backbone.children()):
                    x = child(x)
                    if i == 3:  # After maxpool
                        activations = x.detach().cpu().numpy()[0]  # Shape: (channels, H, W)
                        break

                if activations is None:
                    activations = x.detach().cpu().numpy()[0]

            num_channels = min(activations.shape[0], max_channels)

            # Create grid visualization
            grid_size = int(np.ceil(np.sqrt(num_channels)))
            cell_size = 32
            output = np.zeros((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8)
            output.fill(245)

            for i in range(num_channels):
                row = i // grid_size
                col = i % grid_size
                channel = activations[i]
                channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
                channel = (channel * 255).astype(np.uint8)
                channel = cv2.resize(channel, (cell_size, cell_size))
                # Apply colormap and convert RGB to BGR for OpenCV
                channel_colored = cv2.applyColorMap(channel, cv2.COLORMAP_VIRIDIS)
                channel_bgr = cv2.cvtColor(channel_colored, cv2.COLOR_RGB2BGR)

                y_start = row * cell_size
                x_start = col * cell_size
                output[y_start:y_start+cell_size, x_start:x_start+cell_size] = channel_bgr

            # Add title
            cv2.putText(output, f"CNN Activations ({num_channels} channels)", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            return output

        except Exception as e:
            print(f"Error visualizing activations: {e}")
            output = np.zeros((200, 320, 3), dtype=np.uint8)
            output.fill(245)
            cv2.putText(output, f"Activation Error: {str(e)[:30]}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(output, "See terminal for details", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            return output

    def visualize_feature_maps(self, face_image: np.ndarray) -> np.ndarray:
        """
        Visualize the feature maps from a convolutional layer with spatial resolution.
        Shows the learned features that represent the face.
        """
        try:
            import torch

            if face_image is None or face_image.size == 0:
                raise ValueError("Face image is None or empty")

            # Get embedding features
            face_tensor = self.preprocess(face_image)

            with torch.no_grad():
                backbone = self.model.backbone
                x = face_tensor

                # Go through backbone layers until we get spatial features
                # Layer 7 has 7x7 spatial resolution (512 channels)
                activations = None
                for i, child in enumerate(backbone.children()):
                    x = child(x)
                    if i == 7:  # Use layer before global pooling
                        activations = x.detach().cpu().numpy()[0]  # Shape: (512, 7, 7)
                        break

                if activations is None:
                    activations = x.detach().cpu().numpy()[0]

            # Create feature map visualization
            num_features = min(16, activations.shape[0])
            grid_size = int(np.ceil(np.sqrt(num_features)))
            cell_size = 32
            output = np.zeros((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8)
            output.fill(245)

            for i in range(num_features):
                row = i // grid_size
                col = i % grid_size
                feature = activations[i]
                feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
                feature = (feature * 255).astype(np.uint8)
                feature = cv2.resize(feature, (cell_size, cell_size))
                # Apply colormap and convert RGB to BGR for OpenCV
                feature_colored = cv2.applyColorMap(feature, cv2.COLORMAP_PLASMA)
                feature_bgr = cv2.cvtColor(feature_colored, cv2.COLOR_RGB2BGR)

                y_start = row * cell_size
                x_start = col * cell_size
                output[y_start:y_start+cell_size, x_start:x_start+cell_size] = feature_bgr

            # Add title
            cv2.putText(output, f"Feature Maps ({num_features} channels)", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            return output

        except Exception as e:
            print(f"Error visualizing feature maps: {e}")
            output = np.zeros((224, 224, 3), dtype=np.uint8)
            output.fill(245)
            cv2.putText(output, f"Feature Error: {str(e)[:30]}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(output, "See terminal for details", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            return output

    def test_robustness(self, face_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        original_embedding = self.extract_embedding(face_image)
        robustness_results = {}

        if original_embedding is None:
            return np.zeros((100, 200, 3), dtype=np.uint8), {}

        noise_levels = [0.01, 0.05, 0.1, 0.15, 0.2]
        similarities = []

        face_image_float = face_image.astype(np.float32)

        for noise_sigma in noise_levels:
            noise = np.random.randn(*face_image.shape) * noise_sigma * 255
            noisy_image = np.clip(face_image_float + noise, 0, 255).astype(np.uint8)
            noisy_embedding = self.extract_embedding(noisy_image)

            if noisy_embedding is not None:
                sim = float(np.dot(original_embedding, noisy_embedding) / (np.linalg.norm(original_embedding) * np.linalg.norm(noisy_embedding) + 1e-8))
                similarities.append(sim)
                robustness_results[f'noise_{noise_sigma}'] = sim
            else:
                similarities.append(0.0)

        output = np.zeros((100, 200, 3), dtype=np.uint8)
        output.fill(245)

        bar_width = 30
        gap = 5
        start_x = 10
        start_y = 85

        for i, sim in enumerate(similarities):
            x = start_x + i * (bar_width + gap)
            height = int(sim * 70)
            y = start_y - height

            color = (0, int(sim * 255), int((1 - sim) * 255))
            cv2.rectangle(output, (x, y), (x + bar_width, start_y), color, -1)

        avg_sim = float(np.mean(similarities))
        robustness_results['avg_similarity'] = avg_sim
        robustness_results['robustness_score'] = avg_sim

        return output, robustness_results


class SimilarityComparator:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        dot_product = np.dot(embedding1, embedding2)
        norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        if norm_product == 0:
            return 0.0
        similarity = dot_product / norm_product
        return float(similarity)

    def compare_embeddings(self, query_embedding: np.ndarray, reference_embeddings: List[np.ndarray], reference_ids: List[str]) -> List[Dict]:
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

    def get_confidence_band(self, similarity: float, threshold_high: float = 0.99, threshold_moderate: float = 0.95, threshold_low: float = 0.85) -> str:
        if similarity >= threshold_high:
            return "Very High"
        elif similarity >= threshold_moderate:
            return "High"
        elif similarity >= threshold_low:
            return "Moderate"
        elif similarity >= 0.70:
            return "Low"
        else:
            return "Insufficient"

    def get_verdict(self, similarity: float, threshold_high: float = 0.99, threshold_moderate: float = 0.95) -> str:
        if similarity >= threshold_high:
            return "Likely same person"
        elif similarity >= threshold_moderate:
            return "Possibly same person"
        elif similarity >= 0.70:
            return "Uncertain - human review required"
        else:
            return "Likely different people"

    def euclidean_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        distance = np.linalg.norm(embedding1 - embedding2)
        return float(distance)

    def get_distance_verdict(self, distance: float, threshold_low: float = 0.15, threshold_moderate: float = 0.35) -> str:
        if distance <= threshold_low:
            return "MATCH (same person likely)"
        elif distance <= threshold_moderate:
            return "POSSIBLE (human review recommended)"
        else:
            return "NO MATCH (different people)"
    
    def landmark_similarity(self, landmarks1: Dict[str, float], landmarks2: Dict[str, float]) -> float:
        """
        Compare landmark geometric features between two faces.
        Returns similarity score 0-1 (1 = identical geometry).
        """
        if not landmarks1 or not landmarks2:
            return 0.5
        
        common_keys = set(landmarks1.keys()) & set(landmarks2.keys())
        if not common_keys:
            return 0.5
        
        total_diff = 0.0
        for key in common_keys:
            v1 = landmarks1.get(key, 0)
            v2 = landmarks2.get(key, 0)
            diff = abs(v1 - v2)
            total_diff += diff
        
        avg_diff = total_diff / len(common_keys)
        similarity = max(0, 1 - avg_diff * 5)
        return float(similarity)
    
    def quality_score(self, quality: Dict) -> float:
        """
        Calculate overall quality score from quality metrics.
        Returns 0-1 (1 = best quality).
        """
        if not quality:
            return 0.5
        
        if 'overall' in quality:
            return float(quality['overall'])
        
        brightness = quality.get('brightness', 0.5)
        contrast = quality.get('contrast', 0.5)
        sharpness = quality.get('sharpness', 0.5)
        
        quality_score = (brightness * 0.3 + contrast * 0.3 + sharpness * 0.4)
        return float(quality_score)
    
    def activation_similarity(self, activations1: Dict[str, np.ndarray], activations2: Dict[str, np.ndarray]) -> float:
        """
        Calculate similarity between neural network activations.
        Compares intermediate layer outputs from FaceNet.
        Returns 0-1 (1 = identical activations).
        """
        if not activations1 or not activations2:
            return 0.7  # Default middle value if activations unavailable
        
        similarities = []
        
        # Compare each layer's activations
        for layer_name in activations1.keys():
            if layer_name in activations2:
                act1 = activations1[layer_name]
                act2 = activations2[layer_name]
                
                if act1 is None or act2 is None:
                    continue
                
                # Flatten and compute cosine similarity
                try:
                    act1_flat = act1.flatten()
                    act2_flat = act2.flatten()
                    
                    # Ensure same shape
                    if act1_flat.shape != act2_flat.shape:
                        continue
                    
                    # Cosine similarity
                    dot_product = np.dot(act1_flat, act2_flat)
                    norm1 = np.linalg.norm(act1_flat)
                    norm2 = np.linalg.norm(act2_flat)
                    
                    if norm1 > 0 and norm2 > 0:
                        sim = dot_product / (norm1 * norm2)
                        similarities.append(float(sim))
                except Exception:
                    continue
        
        if similarities:
            return np.mean(similarities)
        
        return 0.7  # Default if comparison fails
    
    def compute_match_score(self, embedding_sim: float, landmark_sim: float = 0.5, 
                           quality: Optional[Dict] = None, use_landmarks: bool = True) -> Dict:
        """
        Compute final match score with multiple factors.
        
        Weights (configurable):
        - Embedding: 75% (main face recognition signal)
        - Landmark geometry: 20% (robustness)
        - Quality adjustment: 5% (handle bad images)
        """
        EMBEDDING_WEIGHT = 0.75
        LANDMARK_WEIGHT = 0.20
        QUALITY_WEIGHT = 0.05
        
        reasons = []
        
        embedding_score = embedding_sim
        
        if embedding_score >= 0.70:
            reasons.append("High embedding similarity")
        elif embedding_score >= 0.45:
            reasons.append("Moderate embedding similarity")
        else:
            reasons.append("Low embedding similarity")
        
        final_score = embedding_score * EMBEDDING_WEIGHT
        
        if use_landmarks and landmark_sim > 0:
            landmark_contribution = landmark_sim * LANDMARK_WEIGHT
            final_score += landmark_contribution
            
            if landmark_sim >= 0.80:
                reasons.append("Similar facial proportions")
            elif landmark_sim >= 0.60:
                reasons.append("Some facial features match")
            else:
                reasons.append("Different facial proportions")
        
        quality_factor = self.quality_score(quality) if quality else 0.7
        quality_contribution = quality_factor * QUALITY_WEIGHT
        final_score += quality_contribution
        
        if quality:
            if quality_factor >= 0.7:
                reasons.append("Good image quality")
            elif quality_factor >= 0.5:
                reasons.append("Moderate image quality")
            else:
                reasons.append("Low image quality - may affect accuracy")
        
        final_score = min(1.0, max(0.0, final_score))
        
        return {
            'score': final_score,
            'embedding_similarity': embedding_score,
            'landmark_similarity': landmark_sim if use_landmarks else None,
            'quality_factor': quality_factor if quality else 0.7,
            'reasons': reasons
        }
    
    def get_match_verdict(self, score: float) -> Tuple[str, str, str]:
        """
        Get verdict based on combined score.
        Returns (status, label, description).
        """
        if score >= 0.70:
            return ('match', 'Full Match', 'High confidence - likely the same person')
        elif score >= 0.45:
            return ('possible', 'Possible Match', 'Medium confidence - human review recommended')
        else:
            return ('no_match', 'No Match', 'Low confidence - likely different people')
    
    def compute_dual_match_score(self, arcface_sim: Optional[float], facenet_sim: Optional[float],
                                  landmark_sim: float = 0.5, quality: Optional[Dict] = None,
                                  activation_sim: Optional[float] = None) -> Dict:
        """
        Compute final match score using BOTH ArcFace and FaceNet embeddings.
        
        Weights:
        - ArcFace: 60% (best discrimination)
        - FaceNet: 20% (additional signal)
        - Landmark geometry: 15% (geometric consistency)
        - Activation: 1% (neural activation similarity)
        - Quality: 1% (reliability factor)
        """
        ARCFACE_WEIGHT = 0.60
        FACENET_WEIGHT = 0.20
        LANDMARK_WEIGHT = 0.15
        ACTIVATION_WEIGHT = 0.01
        QUALITY_WEIGHT = 0.01
        
        reasons = []
        
        # Calculate embedding contributions
        arcface_contribution = 0.0
        if arcface_sim is not None:
            arcface_contribution = arcface_sim * ARCFACE_WEIGHT
            if arcface_sim >= 0.70:
                reasons.append(f"ArcFace: High similarity ({arcface_sim:.0%})")
            elif arcface_sim >= 0.45:
                reasons.append(f"ArcFace: Moderate similarity ({arcface_sim:.0%})")
            else:
                reasons.append(f"ArcFace: Low similarity ({arcface_sim:.0%})")
        else:
            reasons.append("ArcFace: Not available")
        
        facenet_contribution = 0.0
        if facenet_sim is not None:
            facenet_contribution = facenet_sim * FACENET_WEIGHT
            if facenet_sim >= 0.70:
                reasons.append(f"FaceNet: High similarity ({facenet_sim:.0%})")
            elif facenet_sim >= 0.45:
                reasons.append(f"FaceNet: Moderate similarity ({facenet_sim:.0%})")
            else:
                reasons.append(f"FaceNet: Low similarity ({facenet_sim:.0%})")
        else:
            reasons.append("FaceNet: Not available")
        
        # Landmark contribution
        landmark_contribution = landmark_sim * LANDMARK_WEIGHT
        if landmark_sim >= 0.80:
            reasons.append("Similar facial proportions")
        elif landmark_sim >= 0.60:
            reasons.append("Some facial features match")
        else:
            reasons.append("Different facial proportions")
        
        # Activation contribution (neural network internal representations)
        activation_contribution = 0.0
        if activation_sim is not None:
            activation_contribution = activation_sim * ACTIVATION_WEIGHT
            if activation_sim >= 0.70:
                reasons.append("Similar neural activation patterns")
            elif activation_sim >= 0.50:
                reasons.append("Moderate neural activation similarity")
            else:
                reasons.append("Different neural activation patterns")
        
        # Quality contribution
        quality_factor = self.quality_score(quality) if quality else 0.7
        quality_contribution = quality_factor * QUALITY_WEIGHT
        
        if quality_factor >= 0.7:
            reasons.append("Good image quality")
        elif quality_factor >= 0.5:
            reasons.append("Moderate image quality")
        else:
            reasons.append("Low image quality - may affect accuracy")
        
        # Calculate final score
        final_score = arcface_contribution + facenet_contribution + landmark_contribution + activation_contribution + quality_contribution
        final_score = min(1.0, max(0.0, final_score))
        
        return {
            'score': final_score,
            'arcface_similarity': arcface_sim,
            'facenet_similarity': facenet_sim,
            'landmark_similarity': landmark_sim,
            'activation_similarity': activation_sim,
            'quality_factor': quality_factor,
            'reasons': reasons
        }

    def compute_pose_weight(self, pose1: Optional[Dict], pose2: Optional[Dict]) -> float:
        """Calculate weight modifier based on pose similarity.
        ArcFace is robust to pose, so adjustment is minimal."""
        if not pose1 or not pose2:
            return 1.0
        
        yaw1, pitch1 = pose1.get('yaw', 0), pose1.get('pitch', 0)
        yaw2, pitch2 = pose2.get('yaw', 0), pose2.get('pitch', 0)
        
        pose_diff = np.sqrt((yaw1 - yaw2)**2 + (pitch1 - pitch2)**2)
        
        if pose_diff < 15:
            return 1.0
        elif pose_diff < 30:
            return 0.98
        else:
            return 0.95

    def lbp_similarity(self, lbp1: Optional[np.ndarray], lbp2: Optional[np.ndarray]) -> float:
        """Compare LBP histograms using intersection."""
        if lbp1 is None or lbp2 is None:
            return 0.5
        
        intersection = np.sum(np.minimum(lbp1, lbp2))
        return float(intersection)

    def asymmetry_similarity(self, asym1: Optional[Dict], asym2: Optional[Dict]) -> float:
        """Compare asymmetry features between two faces."""
        if not asym1 or not asym2:
            return 0.5
        
        common = set(asym1.keys()) & set(asym2.keys())
        if not common:
            return 0.5
        
        diffs = []
        for k in common:
            v1, v2 = asym1[k], asym2[k]
            if v1 > 0:
                diffs.append(1.0 - min(abs(v1 - v2) / v1, 1.0))
        
        return float(np.mean(diffs)) if diffs else 0.5

    def compute_multi_pose_score(self, query_emb: np.ndarray, 
                                pose_embeddings: List[Dict]) -> Tuple[float, Optional[Dict]]:
        """Compare query against multiple pose variants.
        Returns (best_score, best_pose_data)."""
        if not pose_embeddings:
            return 0.5, None
        
        best_score = 0
        best_pose = None
        
        for pose_data in pose_embeddings:
            emb = pose_data.get('embedding')
            if emb is not None:
                if isinstance(emb, dict):
                    emb = emb.get('arcface') or emb.get('facenet')
                if emb is not None:
                    sim = self.cosine_similarity(query_emb, np.array(emb))
                    if sim > best_score:
                        best_score = sim
                        best_pose = pose_data
        
        return float(best_score), best_pose

    def visualize_comparison_metrics(self, query_embedding: np.ndarray, reference_embeddings: List[np.ndarray], 
                                      reference_ids: List[str], similarities: List[float],
                                      distances: List[float]) -> Tuple[np.ndarray, Dict]:
        """
        Visualize both cosine similarity and euclidean distance for all comparisons.
        
        Args:
            query_embedding: The query embedding
            reference_embeddings: List of reference embeddings
            reference_ids: List of reference IDs
            similarities: List of cosine similarity scores
            distances: List of euclidean distances
            
        Returns:
            Tuple of (visualization_image, data_dict)
        """
        n = len(reference_ids)
        height = max(150, n * 50 + 80)
        output = np.zeros((height, 500, 3), dtype=np.uint8)
        output.fill(245)
        
        cv2.putText(output, "Comparison Metrics", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        cv2.putText(output, "Cosine Similarity", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1)
        cv2.putText(output, "Euclidean Distance", (250, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1)
        
        data = {'comparisons': []}
        
        bar_width_sim = 150
        bar_width_dist = 100
        bar_height = 20
        start_x_sim = 20
        start_x_dist = 250
        start_y = 70
        
        for i, (ref_id, sim, dist) in enumerate(zip(reference_ids, similarities, distances)):
            y = start_y + i * 45
            
            cv2.putText(output, ref_id[:15], (20, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            sim_bar_len = int(sim * bar_width_sim)
            if sim >= 0.95:
                sim_color = (0, 180, 0)
            elif sim >= 0.85:
                sim_color = (0, 200, 100)
            elif sim >= 0.70:
                sim_color = (0, 165, 255)
            else:
                sim_color = (0, 50, 200)
            
            cv2.rectangle(output, (start_x_sim, y), (start_x_sim + sim_bar_len, y + bar_height), sim_color, -1)
            cv2.rectangle(output, (start_x_sim, y), (start_x_sim + bar_width_sim, y + bar_height), (200, 200, 200), 1)
            cv2.putText(output, f"{sim:.3f}", (start_x_sim + bar_width_sim + 5, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            
            dist_normalized = min(dist * bar_width_dist, bar_width_dist)
            if dist <= 0.35:
                dist_color = (0, 180, 0)
            elif dist <= 0.60:
                dist_color = (0, 165, 255)
            else:
                dist_color = (0, 50, 200)
            
            cv2.rectangle(output, (start_x_dist, y), (start_x_dist + int(dist_normalized), y + bar_height), dist_color, -1)
            cv2.rectangle(output, (start_x_dist, y), (start_x_dist + bar_width_dist, y + bar_height), (200, 200, 200), 1)
            cv2.putText(output, f"{dist:.3f}", (start_x_dist + bar_width_dist + 5, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            
            confidence = self.get_confidence_band(sim)
            data['comparisons'].append({
                'id': ref_id,
                'similarity': float(sim),
                'euclidean_distance': float(dist),
                'confidence': confidence,
                'verdict': self.get_verdict(sim),
                'distance_verdict': self.get_distance_verdict(dist)
            })
        
        cv2.line(output, (230, 60), (230, height - 10), (200, 200, 200), 1)
        
        return output, data


# ArcFace Integration
try:
    from .arcface_extractor import ArcFaceEmbeddingExtractor
    ARCFACE_AVAILABLE = True
except ImportError:
    ARCFACE_AVAILABLE = False
    print("Warning: ArcFace unavailable, using FaceNetEmbeddingExtractor")


def get_embedding_extractor(use_arcface: bool = False) -> 'FaceNetEmbeddingExtractor':
    """Get embedding extractor instance.
    
    Args:
        use_arcface: If True, try to return ArcFace extractor
        
    Returns:
        Embedding extractor instance
    """
    if use_arcface and ARCFACE_AVAILABLE:
        return ArcFaceEmbeddingExtractor()
    return FaceNetEmbeddingExtractor()


if __name__ == "__main__":
    extractor = FaceNetEmbeddingExtractor()
    comparator = SimilarityComparator()
    print("Embedding modules loaded successfully")
    if ARCFACE_AVAILABLE:
        print("ArcFace available: use get_embedding_extractor(use_arcface=True)")
