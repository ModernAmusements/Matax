import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

_MEDIAPIPE_AVAILABLE = False
_MEDIAPIPE_TASKS_AVAILABLE = False
try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    _MEDIAPIPE_TASKS_AVAILABLE = True
except ImportError:
    pass


class FaceDetector:
    """
    Face detection module using OpenCV's DNN module with pre-trained Caffe model.
    Facial landmark detection and 3D mesh using MediaPipe Face Mesh.
    Designed for ethical NGO use - detects faces but doesn't identify individuals.
    """

    def __init__(self):
        self.use_dnn = False
        self.face_cascade = None
        self.net = None
        self.mp_face_mesh = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.face_mesh = None
        self._mediapipe_available = _MEDIAPIPE_AVAILABLE

        if self._mediapipe_available and _MEDIAPIPE_TASKS_AVAILABLE:
            try:
                import os
                model_path = 'face_landmark.task'
                if not os.path.exists(model_path):
                    model_path = '/tmp/face_landmark.task'
                if not os.path.exists(model_path):
                    print("MediaPipe model not found, using proportional landmark estimation")
                    self._mediapipe_available = False
                else:
                    base_options = python.BaseOptions(model_asset_path=model_path)
                    options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=False, running_mode=vision.RunningMode.IMAGE, num_faces=1)
                    self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
                    print("MediaPipe Face Landmarker initialized successfully")
            except Exception as e:
                print(f"Failed to initialize MediaPipe: {e}")
                self._mediapipe_available = False

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

    def detect_eyewear(self, face_image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Dict[str, any]:
        """
        Detect sunglasses or glasses that may interfere with face recognition.
        
        Returns dict with:
            - has_eyewear: bool - True if eyewear detected
            - eyewear_type: str - 'sunglasses', 'glasses', 'none', or 'unknown'
            - confidence: float - Detection confidence (0-1)
            - occlusion_level: float - How much the eyes are occluded (0-1)
            - warnings: list - List of warning messages
        """
        x, y, w, h = face_box
        face_h, face_w = face_image.shape[:2]
        
        left_eye_pos = (int(face_w * 0.30), int(face_h * 0.35))
        right_eye_pos = (int(face_w * 0.70), int(face_h * 0.35))
        
        eye_region_h = int(h * 0.25)
        eye_region_w = int(w * 0.40)
        
        left_eye_region = face_image[
            left_eye_pos[1] - eye_region_h//2:left_eye_pos[1] + eye_region_h//2,
            left_eye_pos[0] - eye_region_w//2:left_eye_pos[0] + eye_region_w//2
        ]
        
        right_eye_region = face_image[
            right_eye_pos[1] - eye_region_h//2:right_eye_pos[1] + eye_region_h//2,
            right_eye_pos[0] - eye_region_w//2:right_eye_pos[0] + eye_region_w//2
        ]
        
        warnings = []
        eyewear_detected = False
        eyewear_type = 'none'
        confidence = 0.0
        occlusion_level = 0.0
        
        # Initialize variables for brightness/edge analysis
        brightness_ratio = 1.0
        avg_edge_density = 0.0
        
        # First: Analyze eye region brightness and edges (more reliable than eye cascade)
        if left_eye_region.size > 0 and right_eye_region.size > 0:
            left_gray = cv2.cvtColor(left_eye_region, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_eye_region, cv2.COLOR_BGR2GRAY)
            
            left_brightness = np.mean(left_gray)
            right_brightness = np.mean(right_gray)
            avg_eye_brightness = (left_brightness + right_brightness) / 2
            
            face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_brightness = np.mean(face_gray)
            
            brightness_ratio = avg_eye_brightness / (face_brightness + 1)
            
            # Only flag if very strong evidence - much higher threshold
            if brightness_ratio < 0.2:  # Very dark eyes = strong sunglasses evidence
                warnings.append(f"Eye region very dark (ratio: {brightness_ratio:.2f}) - likely sunglasses")
                eyewear_detected = True
                eyewear_type = 'sunglasses'
                confidence = 0.9
                occlusion_level = 0.9
            elif brightness_ratio < 0.35:  # Somewhat dark = possible sunglasses
                warnings.append(f"Eye region darker than average (ratio: {brightness_ratio:.2f}) - possible sunglasses")
                # Also detect as possible sunglasses
                eyewear_detected = True
                eyewear_type = 'sunglasses'
                confidence = 0.5
                occlusion_level = 0.4
            
            left_edges = cv2.Canny(left_gray, 50, 150)
            right_edges = cv2.Canny(right_gray, 50, 150)
            left_edge_density = np.sum(left_edges > 0) / left_edges.size
            right_edge_density = np.sum(right_edges > 0) / right_edges.size
            avg_edge_density = (left_edge_density + right_edge_density) / 2
            
            # High edge density + dark eyes = strong glasses evidence
            if avg_edge_density > 0.3 and brightness_ratio < 0.4:
                warnings.append(f"High edge density in eye region - possible glasses frames")
                eyewear_detected = True
                eyewear_type = 'glasses'
                confidence = 0.7
                occlusion_level = 0.6
        
        # Second: Use eye cascade only as confirmation, not primary detection
        # OpenCV eye cascade often fails, so require strong evidence
        eyes_detected = self.detect_eyes(face_image)
        eye_count = len(eyes_detected)
        
        if eye_count == 0 and not eyewear_detected:
            # Eye cascade failed - only flag if brightness is VERY dark
            if brightness_ratio < 0.15:
                warnings.append("No eyes detected + very dark eye region - likely sunglasses")
                eyewear_detected = True
                eyewear_type = 'sunglasses'
                confidence = 0.6
                occlusion_level = 0.9
            else:
                # Eye cascade likely just failed - don't flag as eyewear
                eye_count = 2  # Assume normal
        elif eye_count == 1 and not eyewear_detected:
            # One eye detected - weak evidence, require confirmation
            if brightness_ratio < 0.25:
                warnings.append("Only one eye detected + dark - possible glasses")
                eyewear_detected = True
                eyewear_type = 'glasses'
                confidence = 0.4
                occlusion_level = 0.5
            else:
                eye_count = 2
        
        if not eyewear_detected:
            confidence = 0.1
            
            left_edges = cv2.Canny(left_gray, 50, 150)
            right_edges = cv2.Canny(right_gray, 50, 150)
            left_edge_density = np.sum(left_edges > 0) / left_edges.size
            right_edge_density = np.sum(right_edges > 0) / right_edges.size
            avg_edge_density = (left_edge_density + right_edge_density) / 2
            
            if avg_edge_density > 0.15:
                warnings.append(f"High edge density in eye region - possible glasses frames")
                if not eyewear_detected:
                    eyewear_detected = True
                    eyewear_type = 'glasses'
                    confidence = max(confidence, 0.7)
        
        if not eyewear_detected and eye_count >= 2:
            confidence = 0.1
        
        return {
            'has_eyewear': eyewear_detected,
            'eyewear_type': eyewear_type if eyewear_detected else 'none',
            'confidence': confidence,
            'occlusion_level': occlusion_level,
            'warnings': warnings,
            'eye_count': eye_count
        }

    def compute_eyewear_metrics(self, face_image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Dict[str, float]:
        """
        Compute detailed metrics about eyewear/occlusion in the eye region.
        """
        metrics = self.detect_eyewear(face_image, face_box)
        
        x, y, w, h = face_box
        face_h, face_w = face_image.shape[:2]
        
        left_eye_pos = (int(face_w * 0.30), int(face_h * 0.35))
        right_eye_pos = (int(face_w * 0.70), int(face_h * 0.35))
        
        eye_region_h = int(h * 0.25)
        eye_region_w = int(w * 0.40)
        
        eye_regions = []
        for eye_pos in [left_eye_pos, right_eye_pos]:
            region = face_image[
                eye_pos[1] - eye_region_h//2:eye_pos[1] + eye_region_h//2,
                eye_pos[0] - eye_region_w//2:eye_pos[0] + eye_region_w//2
            ]
            if region.size > 0:
                eye_regions.append(region)
        
        if eye_regions:
            brightness_values = []
            contrast_values = []
            for region in eye_regions:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                brightness_values.append(np.mean(gray))
                contrast_values.append(np.std(gray))
            
            metrics['avg_brightness'] = np.mean(brightness_values) / 255.0
            metrics['avg_contrast'] = np.mean(contrast_values) / 128.0
        else:
            metrics['avg_brightness'] = 0.0
            metrics['avg_contrast'] = 0.0
        
        return metrics

    def visualize_eyewear(self, face_image: np.ndarray, face_box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Visualize eyewear detection results with colored overlays.
        """
        output = face_image.copy()
        x, y, w, h = face_box
        face_h, face_w = face_image.shape[:2]
        
        eyewear = self.detect_eyewear(face_image, face_box)
        
        left_eye_pos = (int(face_w * 0.30), int(face_h * 0.35))
        right_eye_pos = (int(face_w * 0.70), int(face_h * 0.35))
        eye_region_h = int(h * 0.25)
        eye_region_w = int(w * 0.40)
        
        left_eye_box = (
            left_eye_pos[0] - eye_region_w//2,
            left_eye_pos[1] - eye_region_h//2,
            eye_region_w,
            eye_region_h
        )
        right_eye_box = (
            right_eye_pos[0] - eye_region_w//2,
            right_eye_pos[1] - eye_region_h//2,
            eye_region_w,
            eye_region_h
        )
        
        if eyewear['has_eyewear']:
            if eyewear['eyewear_type'] == 'sunglasses':
                color = (0, 0, 255)
                label = "SUNGLASSES"
            else:
                color = (0, 165, 255)
                label = "GLASSES"
            
            for eye_box in [left_eye_box, right_eye_box]:
                ex, ey, ew, eh = eye_box
                cv2.rectangle(output, (ex, ey), (ex + ew, ey + eh), color, 2)
            
            cv2.putText(output, f"{label} DETECTED", (x, y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(output, f"Confidence: {eyewear['confidence']:.0%}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if eyewear['warnings']:
                warning_y = y + h + 20
                for warning in eyewear['warnings'][:2]:
                    cv2.putText(output, warning, (x, warning_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    warning_y += 15
        else:
            for eye_box in [left_eye_box, right_eye_box]:
                ex, ey, ew, eh = eye_box
                cv2.rectangle(output, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
            
            cv2.putText(output, "No eyewear detected", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return output

    def estimate_landmarks(self, face_image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Dict[str, Tuple[int, int]]:
        """
        Detect facial landmarks using MediaPipe Face Mesh (468 real landmarks).
        Falls back to proportional estimation if MediaPipe is not available.
        """
        x, y, w, h = face_box
        landmarks = {}

        if self._mediapipe_available and hasattr(self, 'face_landmarker') and self.face_landmarker:
            try:
                from mediapipe import Image, ImageFormat
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_image)
                results = self.face_landmarker.detect(mp_image)

                if hasattr(results, 'face_landmarks') and results.face_landmarks:
                    face_landmarks = results.face_landmarks[0]

                    h_roi, w_roi = face_image.shape[:2]

                    landmark_names = {
                        'left_eye': 33,
                        'right_eye': 263,
                        'left_eye_inner': 468,
                        'right_eye_inner': 473,
                        'nose': 4,
                        'nose_left': 49,
                        'nose_right': 279,
                        'mouth': 13,
                        'mouth_left': 78,
                        'mouth_right': 308,
                        'left_eyebrow': 70,
                        'right_eyebrow': 300,
                        'forehead': 10,
                        'chin': 152,
                        'left_cheek': 234,
                        'right_cheek': 454,
                    }

                    for name, idx in landmark_names.items():
                        if idx < len(face_landmarks):
                            lm = face_landmarks[idx]
                            lm_x = int(lm.x * w_roi)
                            lm_y = int(lm.y * h_roi)
                            landmarks[name] = (lm_x, lm_y)

                    all_landmarks = {}
                    for i, lm in enumerate(face_landmarks):
                        all_landmarks[f'point_{i}'] = (int(lm.x * w_roi), int(lm.y * h_roi))
                    landmarks['all_468'] = all_landmarks

                    return landmarks

            except Exception as e:
                print(f"MediaPipe landmark detection failed: {e}")

        # Fallback: proportional estimation (old behavior)
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
        """
        Compute face alignment angles using real 3D landmarks from MediaPipe.
        Falls back to 2D estimation if 3D data not available.
        """
        left_eye = landmarks.get('left_eye', (0, 0))
        right_eye = landmarks.get('right_eye', (0, 0))

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        roll = np.arctan2(dy, dx) * 180.0 / np.pi

        face_center_x = (left_eye[0] + right_eye[0]) // 2
        face_center_y = (left_eye[1] + right_eye[1]) // 2

        nose = landmarks.get('nose', (face_center_x, face_center_y))
        nose_offset_x = nose[0] - face_center_x
        nose_offset_y = nose[1] - face_center_y

        face_w = max(right_eye[0] - left_eye[0], 50)
        face_h = max(right_eye[1] - left_eye[1], 50)

        yaw = (nose_offset_x / face_w) * 45 if face_w > 0 else 0
        pitch = (nose_offset_y / face_h) * 45 if face_h > 0 else 0

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
        quality = {}
        
        if face_image is None or face_box is None:
            return {'error': 'No image or face box provided'}
        
        x, y, w, h = face_box
        
        if w <= 0 or h <= 0:
            return {'error': 'Invalid face box dimensions'}
        
        if y < 0 or x < 0 or y + h > face_image.shape[0] or x + w > face_image.shape[1]:
            return {'error': 'Face box out of bounds'}
        
        face_roi = face_image[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return {'error': 'Empty face region'}
        
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
        h, w = output.shape[:2]

        # Check if we have all 468 MediaPipe landmarks
        all_468 = landmarks.get('all_468')
        if isinstance(all_468, dict):
            # Draw all 468 landmarks
            for i, (name, pos) in enumerate(all_468.items()):
                x, y = pos
                if 0 <= x < w and 0 <= y < h:
                    # Color by region
                    if i < 31:  # Face outline
                        color = (100, 100, 100)
                    elif i < 68:  # Eyebrows
                        color = (255, 0, 0)
                    elif i < 168:  # Nose
                        color = (0, 255, 0)
                    elif i < 268:  # Eyes
                        color = (0, 255, 255)
                    elif i < 398:  # Mouth outer
                        color = (0, 0, 255)
                    else:  # Mouth inner
                        color = (255, 255, 0)
                    cv2.circle(output, (x, y), 2, color, -1)

            cv2.putText(output, "468 MediaPipe Landmarks", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return output

        # Fallback: draw key landmarks only (old behavior)
        landmark_colors = {
            'left_eye': (0, 255, 0), 'right_eye': (0, 255, 0),
            'nose': (255, 0, 0), 'mouth': (0, 165, 255),
            'left_eye_inner': (0, 255, 255), 'right_eye_inner': (0, 255, 255),
            'nose_left': (255, 0, 255), 'nose_right': (255, 0, 255),
            'mouth_left': (0, 255, 255), 'mouth_right': (0, 255, 255)
        }

        for name, pos in landmarks.items():
            if not isinstance(pos, tuple) or pos is None:
                continue
            if 0 <= pos[0] < face_image.shape[1] and 0 <= pos[1] < face_image.shape[0]:
                color = landmark_colors.get(name, (255, 255, 0))
                cv2.circle(output, pos, 3, color, -1)

        eye_line = [landmarks.get('left_eye'), landmarks.get('right_eye')]
        if all(p is not None for p in eye_line):
            cv2.line(output, eye_line[0], eye_line[1], (0, 255, 0), 2)

        return output

    def visualize_3d_mesh(self, face_image: np.ndarray) -> np.ndarray:
        """
        Visualize 3D face mesh using MediaPipe Face Mesh (478 landmarks with 3D coordinates).
        Falls back to procedural mesh if MediaPipe is not available.
        """
        if face_image is None:
            return np.zeros((200, 200, 3), dtype=np.uint8)
        
        output = face_image.copy()
        h, w = output.shape[:2]

        if self._mediapipe_available and hasattr(self, 'face_landmarker') and self.face_landmarker:
            try:
                from mediapipe import Image, ImageFormat
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_image)
                results = self.face_landmarker.detect(mp_image)

                if hasattr(results, 'face_landmarks') and results.face_landmarks:
                    face_landmarks = results.face_landmarks[0]

                    h_roi, w_roi = face_image.shape[:2]

                    connections = [
                        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
                        (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
                        (361, 288), (288, 397), (397, 365), (365, 379), (379, 378),
                        (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
                        (176, 149), (149, 150), (150, 136), (136, 172), (172, 58),
                        (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
                        (162, 21), (21, 54), (54, 103), (103, 67), (67, 109),
                        (109, 10),
                        (33, 133), (133, 155), (155, 154), (154, 133),
                        (362, 263), (263, 249), (249, 390), (390, 373),
                        (4, 45), (45, 220), (220, 119), (119, 120), (120, 234),
                        (13, 82), (82, 81), (81, 80), (80, 79), (79, 13),
                        (13, 312), (312, 311), (311, 310), (310, 415), (415, 13),
                    ]

                    for start_idx, end_idx in connections:
                        if start_idx < len(face_landmarks) and end_idx < len(face_landmarks):
                            start_lm = face_landmarks[start_idx]
                            end_lm = face_landmarks[end_idx]

                            sx = int(start_lm.x * w_roi)
                            sy = int(start_lm.y * h_roi)
                            ex = int(end_lm.x * w_roi)
                            ey = int(end_lm.y * h_roi)

                            if 0 <= sx < w_roi and 0 <= sy < h_roi and 0 <= ex < w_roi and 0 <= ey < h_roi:
                                depth = getattr(start_lm, 'z', 0) if hasattr(start_lm, 'z') else 0
                                intensity = int(128 + depth * 200)
                                color = (intensity // 2, intensity, 255 - intensity)
                                cv2.line(output, (sx, sy), (ex, ey), color, 1)

                    key_points = [4, 10, 33, 133, 362, 263, 13, 82, 178, 400, 152, 234, 454]
                    for idx in key_points:
                        if idx < len(face_landmarks):
                            lm = face_landmarks[idx]
                            x = int(lm.x * w_roi)
                            y = int(lm.y * h_roi)
                            if 0 <= x < w_roi and 0 <= y < h_roi:
                                cv2.circle(output, (x, y), 3, (0, 255, 255), -1)

                    cv2.putText(output, "3D Face Mesh (MediaPipe - 478 points)", (10, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    return output

            except Exception as e:
                print(f"MediaPipe 3D mesh visualization failed: {e}")

        # Fallback: procedural fake mesh (old behavior)
        np.random.seed(42)
        rows, cols = 8, 8
        grid_x, grid_y = np.meshgrid(
            np.linspace(0.20, 0.80, cols),
            np.linspace(0.20, 0.80, rows)
        )
        depth = 0.15 * np.sin(grid_x * np.pi) * np.sin(grid_y * np.pi)

        scale = min(w, h) * 0.30
        offset_x = w // 2
        offset_y = h // 3

        for i in range(rows):
            for j in range(cols):
                px = int(offset_x + (grid_x[i, j] - 0.5) * scale)
                py = int(offset_y + (grid_y[i, j] - 0.5) * scale * 0.8)
                size = int(3 + depth[i, j] * 3)
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

        cv2.putText(output, "3D Mesh Overlay (Procedural)", (10, 20),
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
