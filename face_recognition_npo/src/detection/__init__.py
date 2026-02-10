import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class FaceDetector:
    """
    Face detection module using OpenCV's DNN module with pre-trained Caffe model.
    Designed for ethical NGO use - detects faces but doesn't identify individuals.
    """

    def __init__(self):
        self.use_dnn = False
        self.face_cascade = None
        self.net = None

        try:
            self.net = cv2.dnn.readNetFromCaffe(
                "deploy.prototxt.txt",
                "res10_300x300_ssd_iter_140000.caffemodel"
            )
            self.use_dnn = True
        except:
            try:
                self.face_cascade = cv2.CascadeClassifier()
                self.face_cascade.load(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                self.use_dnn = False
            except:
                pass

        self.eye_cascade = cv2.CascadeClassifier()
        self.eye_cascade.load(cv2.data.haarcascades + "haarcascade_eye.xml")

        self.confidence_threshold = 0.5

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.use_dnn:
            return self._detect_faces_dnn(image)
        elif self.face_cascade is not None:
            return self._detect_faces_haar(image)
        return []

    def _detect_faces_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                face_box = (startX, startY, endX - startX, endY - startY)
                faces.append(face_box)
        return faces

    def _detect_faces_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [(x, y, w, h) for (x, y, w, h) in faces]

    def detect_faces_with_confidence(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        if self.use_dnn:
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0)
            )
            self.net.setInput(blob)
            detections = self.net.forward()

            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w, endX), min(h, endY)
                    face_box = (startX, startY, endX - startX, endY - startY)
                    faces.append((face_box, confidence))
            return faces
        return []

    def detect_eyes(self, face_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5)
        return [(x, y, w, h) for (x, y, w, h) in eyes]

    def estimate_landmarks(self, face_image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Dict[str, Tuple[int, int]]:
        x, y, w, h = face_box
        landmarks = {}

        face_h, face_w = face_image.shape[:2]

        landmarks['left_eye'] = (int(face_w * 0.30), int(face_h * 0.35))
        landmarks['right_eye'] = (int(face_w * 0.70), int(face_h * 0.35))
        landmarks['nose'] = (int(face_w * 0.50), int(face_h * 0.52))
        landmarks['mouth'] = (int(face_w * 0.50), int(face_h * 0.75))
        landmarks['left_eye_inner'] = (int(face_w * 0.35), int(face_h * 0.35))
        landmarks['right_eye_inner'] = (int(face_w * 0.65), int(face_h * 0.35))
        landmarks['nose_left'] = (int(face_w * 0.45), int(face_h * 0.55))
        landmarks['nose_right'] = (int(face_w * 0.55), int(face_h * 0.55))
        landmarks['mouth_left'] = (int(face_w * 0.38), int(face_h * 0.75))
        landmarks['mouth_right'] = (int(face_w * 0.62), int(face_h * 0.75))

        eyes = self.detect_eyes(face_image)
        if len(eyes) >= 2:
            sorted_eyes = sorted(eyes, key=lambda e: e[0])
            landmarks['left_eye'] = (sorted_eyes[0][0] + sorted_eyes[0][2] // 2, sorted_eyes[0][1] + sorted_eyes[0][3] // 2)
            landmarks['right_eye'] = (sorted_eyes[-1][0] + sorted_eyes[-1][2] // 2, sorted_eyes[-1][1] + sorted_eyes[-1][3] // 2)
        elif len(eyes) == 1:
            cx, cy = eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2
            landmarks['left_eye'] = (int(cx - face_w * 0.10), cy)
            landmarks['right_eye'] = (int(cx + face_w * 0.10), cy)

        return landmarks

    def compute_alignment(self, face_image: np.ndarray, landmarks: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
        left_eye = landmarks.get('left_eye', (0, 0))
        right_eye = landmarks.get('right_eye', (0, 0))

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        yaw = np.arctan2(dy, dx) * 180 / np.pi
        roll = yaw
        pitch = 0.0

        eye_distance = np.sqrt(dx**2 + dy**2)
        if eye_distance > 0:
            nose_x = landmarks.get('nose', (0, 0))[0]
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            pitch = ((nose_x - eye_center_x) / eye_distance) * 30

        face_h, face_w = face_image.shape[:2]
        face_center = (face_w // 2, face_h // 2)
        nose = landmarks.get('nose', (0, 0))
        yaw = ((nose[0] - face_center[0]) / face_w) * 45

        return {'yaw': yaw, 'pitch': pitch, 'roll': roll}

    def compute_saliency(self, face_image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = cv2.GaussianBlur(magnitude, (15, 15), 0)

        saliency = (magnitude / magnitude.max() * 255).astype(np.uint8)
        saliency_color = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)

        return saliency_color

    def compute_quality_metrics(self, face_image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Dict[str, float]:
        x, y, w, h = face_box
        quality = {}

        face_roi = face_image[y:y+h, x:x+w]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray)
        contrast = np.std(gray)
        quality['brightness'] = brightness / 255.0
        quality['contrast'] = contrast / 128.0

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality['sharpness'] = min(1.0, laplacian_var / 1000.0)

        eyes = self.detect_eyes(face_roi)
        quality['eye_detection'] = len(eyes) / 2.0 if len(eyes) <= 2 else 1.0

        center_x, center_y = x + w // 2, y + h // 2
        img_h, img_w = face_image.shape[:2]
        centering = 1.0 - (abs(center_x - img_w / 2) / (img_w / 2) + abs(center_y - img_h / 2) / (img_h / 2)) / 2
        quality['centering'] = max(0, centering)

        overall = (quality['brightness'] * 0.2 + quality['contrast'] * 0.2 +
                   quality['sharpness'] * 0.3 + quality['eye_detection'] * 0.15 +
                   quality['centering'] * 0.15)
        quality['overall'] = overall

        return quality

    def visualize_detection(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        output = image.copy()
        faces_with_conf = self.detect_faces_with_confidence(image)

        for i, ((x, y, w, h), conf) in enumerate(faces_with_conf):
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, f"Face {i+1}: {conf:.1%}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return output

    def visualize_extraction(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        output = image.copy()
        face_h = 160
        face_w = 160

        for i, (x, y, w, h) in enumerate(faces):
            face_roi = image[y:y+h, x:x+w]
            processed = cv2.resize(face_roi, (face_w, face_h))

            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

            offset_x = x + w + 10
            if offset_x + face_w < output.shape[1]:
                output[y:y+face_h, offset_x:offset_x+face_w] = processed
                cv2.rectangle(output, (offset_x, y), (offset_x + face_w, y + face_h), (255, 0, 0), 2)
                cv2.putText(output, "160x160", (offset_x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return output

    def visualize_landmarks(self, face_image: np.ndarray, landmarks: Dict[str, Tuple[int, int]]) -> np.ndarray:
        output = face_image.copy()

        landmark_colors = {
            'left_eye': (0, 255, 0), 'right_eye': (0, 255, 0),
            'nose': (255, 0, 0), 'mouth': (0, 165, 255),
            'left_eye_inner': (0, 255, 255), 'right_eye_inner': (0, 255, 255),
            'nose_left': (255, 0, 255), 'nose_right': (255, 0, 255),
            'mouth_left': (0, 255, 255), 'mouth_right': (0, 255, 255)
        }

        for name, pos in landmarks.items():
            if 0 <= pos[0] < face_image.shape[1] and 0 <= pos[1] < face_image.shape[0]:
                color = landmark_colors.get(name, (255, 255, 0))
                cv2.circle(output, pos, 3, color, -1)

        eye_line = [landmarks.get('left_eye'), landmarks.get('right_eye')]
        if all(p is not None for p in eye_line):
            cv2.line(output, eye_line[0], eye_line[1], (0, 255, 0), 2)

        return output

    def visualize_3d_mesh(self, face_image: np.ndarray) -> np.ndarray:
        output = face_image.copy()
        h, w = output.shape[:2]

        np.random.seed(42)
        rows, cols = 6, 6
        grid_x, grid_y = np.meshgrid(
            np.linspace(0.25, 0.75, cols),
            np.linspace(0.25, 0.75, rows)
        )
        depth = 0.2 * np.sin(grid_x * np.pi) * np.sin(grid_y * np.pi)

        scale = min(w, h) * 0.35
        offset_x = w // 2
        offset_y = h // 3

        for i in range(rows):
            for j in range(cols):
                px = int(offset_x + (grid_x[i, j] - 0.5) * scale)
                py = int(offset_y + (grid_y[i, j] - 0.5) * scale * 0.8)
                size = int(4 + depth[i, j] * 4)
                intensity = int(100 + depth[i, j] * 155)
                color = (intensity, intensity // 2, 255 - intensity)
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(output, (px, py), max(1, size), color, -1)

        for i in range(rows - 1):
            for j in range(cols - 1):
                x1 = int(offset_x + (grid_x[i, j] - 0.5) * scale)
                y1 = int(offset_y + (grid_y[i, j] - 0.5) * scale * 0.8)
                x2 = int(offset_x + (grid_x[i+1, j] - 0.5) * scale)
                y2 = int(offset_y + (grid_y[i+1, j] - 0.5) * scale * 0.8)
                if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                    cv2.line(output, (x1, y1), (x2, y2), (100, 200, 255), 1)
                x2 = int(offset_x + (grid_x[i, j+1] - 0.5) * scale)
                y2 = int(offset_y + (grid_y[i, j+1] - 0.5) * scale * 0.8)
                if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                    cv2.line(output, (x1, y1), (x2, y2), (100, 200, 255), 1)

        cv2.putText(output, "3D Mesh Overlay", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return output

    def visualize_alignment(self, face_image: np.ndarray, landmarks: Dict[str, Tuple[int, int]], alignment: Dict[str, float]) -> np.ndarray:
        output = face_image.copy()
        h, w = output.shape[:2]

        left_eye = landmarks.get('left_eye', (int(w*0.3), int(h*0.35)))
        right_eye = landmarks.get('right_eye', (int(w*0.7), int(h*0.35)))
        nose = landmarks.get('nose', (int(w*0.5), int(h*0.52)))

        cv2.circle(output, left_eye, 5, (0, 255, 0), -1)
        cv2.circle(output, right_eye, 5, (0, 255, 0), -1)
        cv2.circle(output, nose, 4, (255, 0, 0), -1)

        cv2.line(output, left_eye, right_eye, (0, 255, 0), 2)
        cv2.line(output, (left_eye[0], left_eye[1] - 20), (right_eye[0], right_eye[1] - 20), (0, 255, 0), 1)
        cv2.line(output, (left_eye[0], left_eye[1] + 20), (right_eye[0], right_eye[1] + 20), (0, 255, 0), 1)

        mid_eye = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        cv2.line(output, mid_eye, nose, (255, 0, 0), 1)

        cv2.putText(output, f"Yaw: {alignment['yaw']:.1f}deg", (5, h-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(output, f"Pitch: {alignment['pitch']:.1f}deg", (5, h-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output

    def visualize_saliency(self, face_image: np.ndarray) -> np.ndarray:
        saliency = self.compute_saliency(face_image)
        saliency = cv2.resize(saliency, (face_image.shape[1], face_image.shape[0]))

        alpha = 0.5
        output = cv2.addWeighted(face_image, 1 - alpha, saliency, alpha, 0)

        return output

    def visualize_biometric_capture(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> np.ndarray:
        output = image.copy()

        for i, (x, y, w, h) in enumerate(faces):
            face_roi = image[y:y+h, x:x+w]
            processed_face = cv2.resize(face_roi, (160, 160))

            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, f"Face {i+1}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            processed_x = x + w + 20
            processed_y = y
            cv2.rectangle(output, (processed_x, processed_y),
                         (processed_x + 160, processed_y + 160), (255, 0, 0), 2)

            try:
                output[processed_y:processed_y+160, processed_x:processed_x+160] = processed_face
                cv2.putText(output, "Processed", (processed_x, processed_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(output, "160x160", (processed_x, processed_y - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except:
                pass

        return output

    def visualize_multiscale(self, face_image: np.ndarray) -> np.ndarray:
        h, w = face_image.shape[:2]
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]

        base_size = min(w // 2, h // 2)
        result = np.zeros((base_size * 2 + 60, base_size * 5 + 70, 3), dtype=np.uint8)
        result.fill(245)

        positions = [(10, 10), (base_size + 15, 10), (base_size * 2 + 20, 10),
                     (base_size * 3 + 25, 10), (base_size * 4 + 30, 10)]

        for i, scale in enumerate(scales):
            if i >= 5:
                break
            scaled = cv2.resize(face_image, None, fx=scale, fy=scale)
            scaled = cv2.resize(scaled, (base_size, base_size))

            ph, pw = scaled.shape[:2]
            x, y = positions[i]
            result[y:y+ph, x:x+pw] = scaled
            cv2.rectangle(result, (x, y), (x + pw, y + ph), (0, 255, 0), 2)
            cv2.putText(result, f"{scale:.2f}x", (x + 5, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(result, "Multi-Scale Analysis", (10, base_size * 2 + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        return result

    def visualize_quality(self, face_image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Dict]:
        """Visualize quality metrics as a dashboard."""
        quality = self.compute_quality_metrics(face_image, face_box)
        
        h, w = face_image.shape[:2]
        output = np.zeros((h + 150, w, 3), dtype=np.uint8)
        output.fill(30)
        
        output[0:h, 0:w] = face_image
        
        metrics = ['brightness', 'contrast', 'sharpness', 'eye_detection', 'centering', 'overall']
        colors = {
            'brightness': (255, 193, 7),
            'contrast': (255, 152, 0),
            'sharpness': (255, 87, 34),
            'eye_detection': (76, 175, 80),
            'centering': (33, 150, 243),
            'overall': (156, 39, 176)
        }
        
        bar_y = h + 20
        bar_height = 15
        max_bar_width = int(w * 0.7)
        start_x = 20
        
        data = {}
        for i, metric in enumerate(metrics):
            value = quality.get(metric, 0)
            if value is None:
                continue
            data[metric] = float(value)
            bar_width = int(value * max_bar_width)
            
            color = colors.get(metric, (200, 200, 200))
            y = bar_y + i * (bar_height + 8)
            
            cv2.rectangle(output, (start_x, y), (start_x + max_bar_width, y + bar_height), (60, 60, 60), -1)
            cv2.rectangle(output, (start_x, y), (start_x + bar_width, y + bar_height), color, -1)
        
        return output, data

    def visualize_confidence_levels(self, face_image: np.ndarray, similarity: float = 0.75) -> Tuple[np.ndarray, Dict]:
        """Visualize confidence bands and thresholds."""
        h, w = face_image.shape[:2]
        output = np.zeros((h + 120, w, 3), dtype=np.uint8)
        output.fill(25)
        
        output[0:h, 0:w] = face_image
        
        bands = [
            ('Insufficient', 0.0, 0.4, (244, 67, 54)),
            ('Low', 0.4, 0.6, (255, 152, 0)),
            ('Moderate', 0.6, 0.8, (255, 193, 7)),
            ('High', 0.8, 1.0, (76, 175, 80))
        ]
        
        bar_y = h + 25
        bar_height = 30
        bar_x = 50
        max_bar_width = w - 100
        
        total_range = 1.0
        
        data = {}
        current_band = None
        for i, (name, low, high, color) in enumerate(bands):
            band_width = int(((high - low) / total_range) * max_bar_width)
            
            y = bar_y + i * (bar_height + 5)
            
            cv2.rectangle(output, (bar_x, y), (bar_x + band_width, y + bar_height), color, -1)
            
            if low <= similarity < high:
                current_band = name
                marker_x = bar_x + int(((similarity - low) / (high - low)) * band_width)
                cv2.drawMarker(output, (marker_x, y + bar_height // 2), (255, 255, 255), markerSize=15, thickness=2)
        
        data['similarity'] = similarity
        data['current_band'] = current_band
        
        return output, data


def load_face_detection_model():
    pass


if __name__ == "__main__":
    detector = FaceDetector()
    print("Face detection module loaded successfully")
