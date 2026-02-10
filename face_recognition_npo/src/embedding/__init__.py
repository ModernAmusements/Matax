import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2


class FaceNetEmbeddingExtractor:
    def __init__(self, model_path: str = "facenet_model.pth"):
        self.model = self._load_model(model_path)
        self.model.eval()
        self.embedding_size = 128
        self.activation_maps = {}

        self._register_hooks()

    def _load_model(self, model_path: str):
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

    def _register_hooks(self):
        self.activation_maps = {}
        self.hooks = []
        
        def hook_fn(module, input, output):
            self.activation_maps[module] = input[0]
        
        modules_to_hook = [self.model[0], self.model[4], self.model[8], self.model[12]]
        for module in modules_to_hook:
            try:
                handle = module.register_forward_hook(hook_fn)
                self.hooks.append(handle)
            except:
                pass

    def preprocess(self, face_image: np.ndarray) -> torch.Tensor:
        face_image = cv2.resize(face_image, (160, 160))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = np.transpose(face_image, (2, 0, 1))
        face_image = face_image.astype(np.float32) / 255.0
        face_image = torch.tensor(face_image).unsqueeze(0)
        return face_image

    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            face_image = self.preprocess(face_image)
            with torch.no_grad():
                embedding = self.model(face_image)
            embedding = embedding.numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

    def extract_embeddings(self, face_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        return [self.extract_embedding(img) for img in face_images]

    def get_activations(self, face_image: np.ndarray) -> Dict[str, np.ndarray]:
        face_image = self.preprocess(face_image)
        with torch.no_grad():
            _ = self.model(face_image)
        activations = {}
        for name, act in self.activation_maps.items():
            act_np = act[0].cpu().numpy()
            activations[name] = act_np
        return activations

    def visualize_activations(self, face_image: np.ndarray, max_channels: int = 16) -> np.ndarray:
        activations = self.get_activations(face_image)

        if not activations:
            return np.zeros((160, 160, 3), dtype=np.uint8)

        output = np.zeros((200, 320, 3), dtype=np.uint8)
        output.fill(245)

        if not activations:
            cv2.putText(output, "No activations", (100, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            return output

        cell_size = 70
        max_per_row = 4
        row = 0
        col = 0

        for layer_name, act in list(activations.items()):
            for i in range(min(max_channels, act.shape[0])):
                act_channel = act[i]
                if act_channel.ndim == 3:
                    act_channel = act_channel.mean(axis=0)
                act_channel = (act_channel - act_channel.min()) / (act_channel.max() - act_channel.min() + 1e-8)
                act_channel = (act_channel * 255).astype(np.uint8)
                act_channel = cv2.resize(act_channel, (cell_size, cell_size))

                color = (act_channel.mean(), 100, 255 - act_channel.mean())
                color_uint8 = (int(color[0]), int(color[1]), int(color[2]))

                canvas = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                canvas[:] = color_uint8

                x = 10 + col * (cell_size + 5)
                y = 10 + row * (cell_size + 5)

                if y + cell_size < output.shape[0] and x + cell_size < output.shape[1]:
                    output[y:y+cell_size, x:x+cell_size] = canvas
                    col += 1
                    if col >= max_per_row:
                        col = 0
                        row += 1
                        if row >= 2:
                            break
            if row >= 2:
                break

        cv2.putText(output, "Neural Network Activations", (10, 195),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

        return output

    def visualize_feature_maps(self, face_image: np.ndarray) -> np.ndarray:
        activations = self.get_activations(face_image)

        output = np.zeros((224, 224, 3), dtype=np.uint8)
        output.fill(245)

        if not activations:
            cv2.putText(output, "No Feature Maps", (60, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            return output

        combined = None
        for layer_name, act in activations.items():
            act_mean = act[0].mean(axis=0)
            act_mean = (act_mean - act_mean.min()) / (act_mean.max() - act_mean.min() + 1e-8)
            act_mean = cv2.resize(act_mean, (224, 224))
            if combined is None:
                combined = act_mean
            else:
                combined = (combined + act_mean) / 2

        if combined is not None:
            combined = (combined * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_VIRIDIS)
            output = heatmap.copy()

        cv2.putText(output, "Feature Importance", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return output

    def visualize_embedding(self, embedding: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if embedding is None:
            return np.zeros((100, 256, 3), dtype=np.uint8), {}

        output = np.zeros((120, 256, 3), dtype=np.uint8)
        output.fill(30)

        bar_width = 2
        gap = 1
        start_x = 10
        start_y = 100

        sorted_indices = np.argsort(embedding)[::-1]
        sorted_embedding = embedding[sorted_indices]
        
        data = {}
        max_val = float(sorted_embedding[0]) if len(sorted_embedding) > 0 else 0
        for i, val in enumerate(sorted_embedding[:128]):
            if i >= 128:
                break
            x = start_x + i * (bar_width + gap)
            height = int(val * 80)
            y = start_y - height

            intensity = int(val * 255)
            color = (int(intensity * 0.2), intensity, 255 - intensity)

            cv2.rectangle(output, (x, y), (x + bar_width, start_y), color, -1)
            
            idx = int(sorted_indices[i])
            data[f"dim_{idx}"] = float(val)

        return output, data

    def visualize_similarity_matrix(self, query_embedding: np.ndarray, reference_embeddings: List[np.ndarray], reference_names: List[str]) -> Tuple[np.ndarray, Dict]:
        output = np.zeros((150, 300, 3), dtype=np.uint8)
        output.fill(245)

        data = {}
        
        if query_embedding is None or not reference_embeddings:
            return output, data

        all_embs = [query_embedding] + reference_embeddings
        n = len(all_embs)

        cell_size = 50
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                sim = np.dot(all_embs[i], all_embs[j]) / (np.linalg.norm(all_embs[i]) * np.linalg.norm(all_embs[j]) + 1e-8)
                matrix[i, j] = sim
                key = f"{['Query', *reference_names][i]}_{['Query', *reference_names][j]}"
                data[key] = float(sim)

        matrix = (matrix * 255).astype(np.uint8)

        colormap = cv2.COLORMAP_VIRIDIS
        matrix_colored = cv2.applyColorMap(matrix, colormap)
        matrix_colored = cv2.resize(matrix_colored, (n * cell_size, n * cell_size))

        output[:n * cell_size, :n * cell_size] = matrix_colored

        return output, data

    def visualize_similarity_result(self, query_embedding: np.ndarray, ref_embedding: np.ndarray, 
                                   similarity: float, query_name: str = "Query", 
                                   ref_name: str = "Reference") -> Tuple[np.ndarray, Dict]:
        """Visualize a single similarity comparison result."""
        output = np.zeros((200, 400, 3), dtype=np.uint8)
        output.fill(245)
        
        center_x = 200
        gauge_y = 80
        gauge_radius = 60
        
        cv2.circle(output, (center_x, gauge_y), gauge_radius, (200, 200, 200), 20)
        
        angle = np.pi - (similarity * np.pi)
        end_x = int(center_x + gauge_radius * np.cos(angle))
        end_y = int(gauge_y - gauge_radius * np.sin(angle))
        cv2.line(output, (center_x, gauge_y), (end_x, end_y), (76, 175, 80), 4)
        
        from src.embedding import SimilarityComparator
        comp = SimilarityComparator()
        band = comp.get_confidence_band(similarity)
        
        data = {
            'similarity': float(similarity),
            'confidence': band,
            'query': query_name,
            'reference': ref_name
        }
        
        return output, data

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
        return max(0.0, min(1.0, similarity))

    def compare_embeddings(self, query_embedding: np.ndarray, reference_embeddings: List[np.ndarray], reference_ids: List[str]) -> List[Tuple[str, float]]:
        results = []
        for ref_id, ref_embedding in zip(reference_ids, reference_embeddings):
            if ref_embedding is None:
                continue
            similarity = self.cosine_similarity(query_embedding, ref_embedding)
            if similarity > self.threshold:
                results.append((ref_id, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_confidence_band(self, similarity: float) -> str:
        if similarity > 0.8:
            return "High confidence"
        elif similarity > 0.6:
            return "Moderate confidence"
        elif similarity > 0.4:
            return "Low confidence"
        else:
            return "Insufficient confidence"


if __name__ == "__main__":
    extractor = FaceNetEmbeddingExtractor()
    comparator = SimilarityComparator()
    print("Embedding modules loaded successfully")
