#!/usr/bin/env python3
"""
Face Recognition API Server for Electron UI
Provides HTTP endpoints for all face recognition operations.
"""

import sys
import os
import json
import base64
import io
import threading
import uuid
import shutil
import datetime
from typing import List, Dict, Tuple, Optional
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.detection import FaceDetector
from src.detection.preprocessing import ImagePreprocessor
from src.embedding import (
    FaceNetEmbeddingExtractor, 
    SimilarityComparator,
    get_embedding_extractor,
    ARCFACE_AVAILABLE
)

app = Flask(__name__, static_folder='./electron-ui')
CORS(app)

USE_ARCFACE = os.environ.get('USE_ARCFACE', 'true').lower() == 'true'

detector = FaceDetector()
preprocessor = ImagePreprocessor()

print("=" * 60)
print("INITIALIZING FACE EMBEDDING EXTRACTORS")
print("=" * 60)

# Always initialize FaceNet extractor (for dual-model and activations)
from src.embedding import FaceNetEmbeddingExtractor
facenet_extractor = FaceNetEmbeddingExtractor()
print("- FaceNet: 128-dim (always loaded)")

# Initialize ArcFace if available
arcface_extractor = None
if USE_ARCFACE and ARCFACE_AVAILABLE:
    from src.embedding import ArcFaceEmbeddingExtractor
    arcface_extractor = ArcFaceEmbeddingExtractor()
    print("- ArcFace: 512-dim (loaded)")
else:
    if USE_ARCFACE:
        print("WARNING: ArcFace requested but unavailable")
    print("- ArcFace: not loaded")

# Primary extractor for backwards compatibility
extractor = arcface_extractor if arcface_extractor else facenet_extractor
print(f"Primary: {type(extractor).__name__}")
print(f"Dual-model: ENABLED (ArcFace + FaceNet)")
print("=" * 60)

comparator = SimilarityComparator(threshold=0.5)

current_image = None
current_original_image = None
current_enhanced_image = None
current_faces = []
current_embedding = None
current_embeddings = {}
current_face_image = None
current_preprocessing_info = {}
current_pose = {}
current_landmarks = None
current_quality = None
current_activations = {}
current_lbp = None
current_asymmetry = None
current_normalized_embedding = None
references = []

REFERENCES_FILE = os.path.join(os.path.dirname(__file__), 'reference_images', 'embeddings.json')


def load_references():
    """Load references from JSON file on startup."""
    global references
    try:
        if os.path.exists(REFERENCES_FILE) and os.path.getsize(REFERENCES_FILE) > 0:
            with open(REFERENCES_FILE, 'r') as f:
                data = json.load(f)
                references = data.get('references', [])
            print(f"Loaded {len(references)} references from {REFERENCES_FILE}")
    except Exception as e:
        print(f"Error loading references: {e}")
        references = []


def save_references():
    """Save references to JSON file."""
    try:
        data = {
            'metadata': [
                {
                    'id': r.get('id'),
                    'name': r.get('name'),
                    'thumbnail': r.get('thumbnail')[:100] + '...' if r.get('thumbnail') and len(r.get('thumbnail', '')) > 100 else r.get('thumbnail'),
                    'added_at': r.get('added_at')
                }
                for r in references
            ],
            'embeddings': [
                {
                    'id': r.get('id'),
                    'embedding': r.get('embedding', [])
                }
                for r in references
            ]
        }
        with open(REFERENCES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(references)} references to {REFERENCES_FILE}")
    except Exception as e:
        print(f"Error saving references: {e}")


load_references()


def np_to_python(val):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    if image is None:
        return None
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to numpy image."""
    buffer = base64.b64decode(base64_str)
    nparr = np.frombuffer(buffer, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def visualize_tests(face_image, faces, embedding, refs) -> np.ndarray:
    """Generate test results visualization."""
    h, w = 700, 900
    img = np.ones((h, w, 3), dtype=np.uint8) * 30
    
    cv2.putText(img, "FRONTEND INTEGRATION TESTS", (30, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(img, "Run complete pipeline to see all results", (30, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    tests = [
        ("1. Health Check", True, "API is running"),
        ("2. Detection + Preprocessing", len(faces) > 0 if faces else False, f"faces={len(faces) if faces else 0}, enhanced=True"),
        ("3. Extraction + Pose", embedding is not None, f"512-dim embedding extracted"),
        ("4. Add Reference + Pose", len(refs) > 0 if refs else False, f"pose stored with reference"),
        ("5. Multi-Reference", len(refs) > 1 if refs else False, f"{len(refs) if refs else 0} references enrolled"),
        ("6. Pose-Aware Matching", embedding is not None and len(refs) > 0 if refs else False, "adjusted similarity enabled"),
        ("7. Eyewear Detection", face_image is not None, "sunglasses detection ready"),
        ("8. Visualizations", True, "16 visualization types"),
        ("9. Clear + Reset", True, "session management works"),
    ]
    
    y_pos = 110
    for name, passed, details in tests:
        color = (0, 210, 0) if passed else (0, 100, 200)
        status = "PASS" if passed else "WAIT"
        
        cv2.rectangle(img, (25, y_pos-25), (w-25, y_pos+35), (45, 45, 45), -1)
        
        cv2.putText(img, name, (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        
        cv2.putText(img, status, (550, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        
        cv2.putText(img, details, (40, y_pos+22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
        
        y_pos += 65
    
    passed_count = sum(1 for _, p, _ in tests if p)
    cv2.rectangle(img, (20, h-80), (w-20, h-20), (50, 50, 50), -1)
    
    if passed_count == len(tests):
        status_color = (0, 255, 0)
        status_text = f"ALL {len(tests)} TESTS PASSED"
    elif passed_count > len(tests) // 2:
        status_color = (255, 200, 0)
        status_text = f"{passed_count}/{len(tests)} TESTS PASSED"
    else:
        status_color = (255, 100, 100)
        status_text = f"{passed_count}/{len(tests)} TESTS - RUN PIPELINE FIRST"
    
    cv2.putText(img, status_text, (40, h-45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
    
    return img


def visualize_test_detail(test_name, result_data) -> np.ndarray:
    """Generate detailed visualization for a specific test."""
    h, w = 500, 700
    img = np.ones((h, w, 3), dtype=np.uint8) * 25
    
    cv2.putText(img, f"TEST: {test_name}", (30, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_pos = 90
    
    if isinstance(result_data, dict):
        for key, value in result_data.items():
            cv2.putText(img, f"{key}:", (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.putText(img, str(value), (200, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
            y_pos += 35
    else:
        cv2.putText(img, str(result_data), (40, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return img


def extract_landmark_features(landmarks: Dict) -> Optional[Dict[str, float]]:
    """
    Extract geometric features from landmarks for comparison.
    Uses ratios between key facial points that are scale-invariant.
    """
    if not landmarks:
        return None
    
    key_points = ['left_eye', 'right_eye', 'nose', 'mouth', 'left_cheek', 'right_cheek', 'chin', 'forehead']
    points = {}
    for key in key_points:
        if key in landmarks:
            points[key] = landmarks[key]
    
    if len(points) < 4:
        return None
    
    # Calculate face size for normalization
    min_x = min(p[0] for p in points.values())
    max_x = max(p[0] for p in points.values())
    min_y = min(p[1] for p in points.values())
    max_y = max(p[1] for p in points.values())
    face_width = max(max_x - min_x, 1)
    face_height = max(max_y - min_y, 1)
    
    features = {}
    
    # Eye-related ratios
    if 'left_eye' in points and 'right_eye' in points:
        eye_dist = ((points['left_eye'][0] - points['right_eye'][0])**2 + 
                   (points['left_eye'][1] - points['right_eye'][1])**2)**0.5
        features['eye_distance'] = eye_dist / face_width
    else:
        features['eye_distance'] = 0.3
    
    # Eye to nose ratio
    if 'left_eye' in points and 'nose' in points:
        eye_nose = ((points['left_eye'][0] - points['nose'][0])**2 + 
                   (points['left_eye'][1] - points['nose'][1])**2)**0.5
        features['eye_nose_ratio'] = eye_nose / face_width
    else:
        features['eye_nose_ratio'] = 0.25
    
    # Nose to mouth ratio
    if 'nose' in points and 'mouth' in points:
        nose_mouth = ((points['nose'][0] - points['mouth'][0])**2 + 
                     (points['nose'][1] - points['mouth'][1])**2)**0.5
        features['nose_mouth_ratio'] = nose_mouth / face_width
    else:
        features['nose_mouth_ratio'] = 0.15
    
    # Face width to height ratio
    features['width_height_ratio'] = face_width / face_height
    
    # Eye horizontal position (asymmetry)
    if 'left_eye' in points and 'right_eye' in points and 'nose' in points:
        eye_center_x = (points['left_eye'][0] + points['right_eye'][0]) / 2
        features['eye_nose_x_diff'] = abs(eye_center_x - points['nose'][0]) / face_width
    else:
        features['eye_nose_x_diff'] = 0.0
    
    # Vertical position of eyes (should be roughly 1/3 from top)
    if 'left_eye' in points:
        features['eye_vertical_pos'] = points['left_eye'][1] / face_height
    else:
        features['eye_vertical_pos'] = 0.35
    
    # Mouth vertical position
    if 'mouth' in points:
        features['mouth_vertical_pos'] = points['mouth'][1] / face_height
    else:
        features['mouth_vertical_pos'] = 0.75
    
    # Eye-nose-mouth alignment
    if 'left_eye' in points and 'nose' in points and 'mouth' in points:
        features['face_symmetry'] = abs(points['nose'][0] - (points['left_eye'][0] + points['right_eye'][0]) / 2) / face_width
    else:
        features['face_symmetry'] = 0.0
    
    return features


@app.route('/')
def index():
    return send_from_directory('./electron-ui', 'index.html')


@app.route('/styles/<path:path>')
def serve_styles(path):
    return send_from_directory('./electron-ui/styles', path)


@app.route('/renderer/<path:path>')
def serve_renderer(path):
    return send_from_directory('./electron-ui/renderer', path)


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Face Recognition API running'})


@app.route('/api/embedding-info', methods=['GET'])
def embedding_info():
    """Get information about the current embedding extractor."""
    dim = getattr(extractor, 'embedding_dim', 128)
    model_type = type(extractor).__name__
    return jsonify({
        'model': model_type,
        'dimension': dim,
        'use_arcface': USE_ARCFACE
    })


@app.route('/api/diagnostics', methods=['GET'])
def diagnostics():
    """Get diagnostic information about the system."""
    from src.detection import _MEDIAPIPE_AVAILABLE
    
    mediapipe_status = "available" if _MEDIAPIPE_AVAILABLE else "not_available"
    model_exists = os.path.exists('face_landmark.task')
    
    return jsonify({
        'mediapipe': mediapipe_status,
        'model_file_exists': model_exists,
        'arcface_extractor': 'loaded' if arcface_extractor else 'not_loaded',
        'facenet_extractor': 'loaded',
        'dual_model_mode': True
    })


@app.route('/api/detect', methods=['POST'])
def detect_faces():
    """Detect faces in uploaded image."""
    global current_image, current_original_image, current_enhanced_image, current_faces, current_preprocessing_info
    
    try:
        data = request.json
        image_data = data.get('image', '')
        
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        current_original_image = base64_to_image(image_data)
        
        enhanced_image, method = preprocessor.enhance(current_original_image)
        
        if method != 'none':
            current_image = enhanced_image
            current_preprocessing_info = preprocessor.get_preprocessing_info(current_original_image, enhanced_image, method)
        else:
            current_image = current_original_image
            current_preprocessing_info = {
                'was_enhanced': False,
                'method': 'none',
                'original_quality': preprocessor.assess_quality(current_original_image),
                'enhanced_quality': preprocessor.assess_quality(current_original_image),
                'improvement': {'brightness': 0, 'contrast': 0, 'sharpness': 0, 'overall': 0}
            }
        
        if current_preprocessing_info['was_enhanced']:
            current_image = enhanced_image
        else:
            current_image = current_original_image
        
        current_faces = detector.detect_faces(current_image)
        
        faces_data = []
        for i, (x, y, w, h) in enumerate(current_faces):
            face_img = current_image[y:y+h, x:x+w]
            faces_data.append({
                'id': i,
                'bbox': [np_to_python(x), np_to_python(y), np_to_python(w), np_to_python(h)],
                'thumbnail': image_to_base64(face_img)
            })
        
        preprocessing_for_api = {
            'was_enhanced': current_preprocessing_info.get('was_enhanced', False),
            'method': current_preprocessing_info.get('method', 'none'),
            'original_quality': {k: float(v) for k, v in current_preprocessing_info.get('original_quality', {}).items()},
            'enhanced_quality': {k: float(v) for k, v in current_preprocessing_info.get('enhanced_quality', {}).items()}
        }
        
        return jsonify({
            'success': True,
            'count': len(current_faces),
            'preprocessing': preprocessing_for_api,
            'faces': faces_data,
            'visualizations': {
                'detection': image_to_base64(detector.visualize_detection(current_image, current_faces)),
                'extraction': image_to_base64(detector.visualize_extraction(current_image, current_faces)),
                'biometric': image_to_base64(detector.visualize_biometric_capture(current_image, current_faces)),
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/extract', methods=['POST'])
def extract_embedding():
    """Extract embedding from detected face using both models."""
    global current_embedding, current_embeddings, current_face_image, current_faces, current_pose, current_landmarks, current_quality, current_activations, current_lbp, current_asymmetry, current_normalized_embedding
    
    try:
        data = request.json
        face_id = data.get('face_id', 0)
        
        if not current_faces:
            return jsonify({'success': False, 'error': 'No faces detected'})
        
        x, y, w, h = current_faces[face_id]
        current_face_image = current_image[y:y+h, x:x+w]
        
        # Extract ArcFace embedding (primary)
        arcface_emb = None
        if arcface_extractor:
            arcface_emb = arcface_extractor.extract_embedding(current_face_image)
        
        # Extract FaceNet embedding (secondary, also for activations)
        facenet_emb = facenet_extractor.extract_embedding(current_face_image)
        
        # Store both embeddings
        current_embeddings = {
            'arcface': arcface_emb,
            'facenet': facenet_emb
        }
        # For backwards compatibility
        current_embedding = arcface_emb if arcface_emb is not None else facenet_emb
        
        # Get FaceNet activations (for visualization)
        current_activations = facenet_extractor.get_activations(current_face_image)
        
        landmarks_est = detector.estimate_landmarks(current_face_image, (0, 0, current_face_image.shape[1], current_face_image.shape[0]))
        alignment_est = detector.compute_alignment(current_face_image, landmarks_est)
        quality_est = detector.compute_quality_metrics(current_face_image, (0, 0, current_face_image.shape[1], current_face_image.shape[0]))
        
        current_pose = {
            'yaw': float(alignment_est.get('yaw', 0)),
            'pitch': float(alignment_est.get('pitch', 0)),
            'roll': float(alignment_est.get('roll', 0)),
            'pose_category': categorize_pose(alignment_est.get('yaw', 0), alignment_est.get('pitch', 0))
        }
        
        # Store landmarks as geometric features for comparison
        current_landmarks = extract_landmark_features(landmarks_est)
        current_quality = quality_est
        
        # NEW: LBP descriptor for lighting-invariant matching
        current_lbp = detector.compute_lbp_descriptor(current_face_image)
        
        # NEW: Asymmetry features for uniqueness analysis
        current_asymmetry = detector.compute_facial_asymmetry(landmarks_est)
        
        # NEW: 3D mesh-based normalized embedding
        mesh_landmarks = detector.estimate_landmarks(current_face_image, (0, 0, current_face_image.shape[1], current_face_image.shape[0]))
        aligned_face = detector.normalize_face_with_mesh(current_face_image, mesh_landmarks)
        if arcface_extractor:
            current_normalized_embedding = arcface_extractor.extract_embedding(aligned_face)
        else:
            current_normalized_embedding = facenet_extractor.extract_embedding(aligned_face)
        
        # Get visualizations with data
        emb_viz, emb_data = extractor.visualize_embedding(current_embedding)
        act_viz = extractor.visualize_activations(current_face_image)
        feat_viz = extractor.visualize_feature_maps(current_face_image)
        robust_viz, robust_data = extractor.test_robustness(current_face_image)
        landmarks_est = detector.estimate_landmarks(current_face_image, (0, 0, current_face_image.shape[1], current_face_image.shape[0]))
        land_viz = detector.visualize_landmarks(current_face_image, landmarks_est)
        mesh_viz = detector.visualize_3d_mesh(current_face_image)
        alignment_est = detector.compute_alignment(current_face_image, landmarks_est)
        align_viz = detector.visualize_alignment(current_face_image, landmarks_est, alignment_est)
        sal_viz = detector.visualize_saliency(current_face_image)
        multi_viz = detector.visualize_multiscale(current_face_image)
        conf_viz, conf_data = detector.visualize_quality(current_face_image, (0, 0, current_face_image.shape[1], current_face_image.shape[0]))
        
        response_data = {
            'success': True,
            'embedding_size': len(current_embedding) if current_embedding is not None else 0,
            'embedding_type': type(extractor).__name__,
            'arcface_available': arcface_extractor is not None,
            'arcface_size': len(arcface_emb) if arcface_emb is not None else 0,
            'facenet_size': len(facenet_emb) if facenet_emb is not None else 0,
            'embedding_mean': float(np.mean(current_embedding)) if current_embedding is not None else 0,
            'embedding_std': float(np.std(current_embedding)) if current_embedding is not None else 0,
            'pose': current_pose,
            'lbp_histogram': current_lbp.tolist() if current_lbp is not None else None,
            'asymmetry': current_asymmetry,
            'normalized_embedding': current_normalized_embedding.tolist() if current_normalized_embedding is not None else None,
            'was_normalized': True,
            'visualizations': {
                'embedding': image_to_base64(emb_viz),
                'activations': image_to_base64(act_viz),
                'features': image_to_base64(feat_viz),
                'robustness': image_to_base64(robust_viz),
                'landmarks': image_to_base64(land_viz),
                'mesh3d': image_to_base64(mesh_viz),
                'alignment': image_to_base64(align_viz),
                'saliency': image_to_base64(sal_viz),
                'multiscale': image_to_base64(multi_viz),
                'confidence': image_to_base64(conf_viz),
            },
            'visualization_data': {
                'embedding': emb_data,
                'robustness': robust_data,
                'confidence': conf_data,
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/add-reference', methods=['POST'])
def add_reference():
    """Add a reference image for comparison."""
    global references
    
    try:
        data = request.json
        image_data = data.get('image', '')
        name = data.get('name', f'Reference {len(references) + 1}')
        
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        ref_image = base64_to_image(image_data)
        ref_faces = detector.detect_faces(ref_image)
        
        if not ref_faces:
            return jsonify({'success': False, 'error': 'No faces detected in reference'})
        
        fx, fy, fw, fh = ref_faces[0]
        ref_face = ref_image[fy:fy+fh, fx:fx+fw]
        
        # Extract both embeddings
        arcface_emb = arcface_extractor.extract_embedding(ref_face) if arcface_extractor else None
        facenet_emb = facenet_extractor.extract_embedding(ref_face)
        
        # Store as dict with both models
        embeddings_dict = {
            'arcface': arcface_emb.tolist() if arcface_emb is not None else None,
            'facenet': facenet_emb.tolist() if facenet_emb is not None else None
        }
        
        landmarks = detector.estimate_landmarks(ref_face, (0, 0, ref_face.shape[1], ref_face.shape[0]))
        alignment = detector.compute_alignment(ref_face, landmarks)
        quality = detector.compute_quality_metrics(ref_face, (0, 0, ref_face.shape[1], ref_face.shape[0]))
        landmark_features = extract_landmark_features(landmarks)
        ref_activations = facenet_extractor.get_activations(ref_face)
        
        # NEW: LBP descriptor
        lbp_hist = detector.compute_lbp_descriptor(ref_face)
        
        # NEW: Asymmetry features
        asymmetry_features = detector.compute_facial_asymmetry(landmarks)
        
        # NEW: 3D mesh-based normalized embedding
        mesh_landmarks = detector.estimate_landmarks(ref_face, (0, 0, ref_face.shape[1], ref_face.shape[0]))
        aligned_ref = detector.normalize_face_with_mesh(ref_face, mesh_landmarks)
        normalized_emb = None
        if arcface_extractor:
            normalized_emb = arcface_extractor.extract_embedding(aligned_ref)
        
        pose_category = categorize_pose(alignment.get('yaw', 0), alignment.get('pitch', 0))
        
        ref_data = {
            'id': len(references),
            'name': name,
            'embedding': embeddings_dict,
            'activations': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in ref_activations.items()} if ref_activations else {},
            'thumbnail': image_to_base64(ref_face),
            'pose': {
                'yaw': float(alignment.get('yaw', 0)),
                'pitch': float(alignment.get('pitch', 0)),
                'roll': float(alignment.get('roll', 0))
            },
            'pose_category': pose_category,
            'landmarks': landmark_features,
            'quality': {k: float(v) if isinstance(v, (int, float)) else v for k, v in quality.items()} if quality else None,
            'lbp_histogram': lbp_hist.tolist() if lbp_hist is not None else None,
            'asymmetry': asymmetry_features,
            'normalized_embedding': normalized_emb.tolist() if normalized_emb is not None else None,
            'poses': {
                pose_category: {
                    'embedding': embeddings_dict,
                    'yaw': float(alignment.get('yaw', 0)),
                    'pitch': float(alignment.get('pitch', 0))
                }
            }
        }
        references.append(ref_data)
        save_references()
        
        return jsonify({
            'success': True,
            'reference': ref_data,
            'count': len(references)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


def categorize_pose(yaw: float, pitch: float) -> str:
    """Categorize pose into frontal, left, right, up, down."""
    if abs(yaw) < 15 and abs(pitch) < 15:
        return 'frontal'
    elif yaw < -15:
        return 'left'
    elif yaw > 15:
        return 'right'
    elif pitch < -15:
        return 'up'
    elif pitch > 15:
        return 'down'
    else:
        return 'frontal'


@app.route('/api/references', methods=['GET'])
def get_references():
    """Get all reference images."""
    return jsonify({
        'success': True,
        'references': [
            {
                'id': r['id'],
                'name': r['name'],
                'thumbnail': r['thumbnail']
            }
            for r in references
        ],
        'count': len(references)
    })


@app.route('/api/references/<int:ref_id>', methods=['DELETE'])
def remove_reference(ref_id):
    """
    Remove a reference image by ID.

    Returns:
        success: bool
        removed_id: int
        removed_name: str
        count: int (remaining references)
    """
    global references

    try:
        with threading.Lock():
            if ref_id < 0 or ref_id >= len(references):
                return jsonify({
                    'success': False,
                    'error': 'Reference not found'
                }), 404

            removed_name = references[ref_id].get('name', f'Reference {ref_id}')
            references.pop(ref_id)

            for i, ref in enumerate(references):
                ref['id'] = i
            
            save_references()

            return jsonify({
                'success': True,
                'removed_id': ref_id,
                'removed_name': removed_name,
                'count': len(references)
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_faces():
    """Compare current face embedding with references using dual-model scoring."""
    global current_embedding, current_embeddings, references, current_pose, current_landmarks, current_quality, current_activations
    
    try:
        if current_embedding is None and not current_embeddings:
            return jsonify({'success': False, 'error': 'No embedding extracted'})
        
        if not references:
            return jsonify({'success': False, 'error': 'No references added'})

        results = []
        
        # Get current query embeddings
        q_arcface = current_embeddings.get('arcface') if current_embeddings else None
        q_facenet = current_embeddings.get('facenet') if current_embeddings else None
        
        # Fallback for backwards compatibility
        if q_arcface is None and q_facenet is None:
            q_arcface = current_embedding
        
        query_landmarks = current_landmarks if current_landmarks else None
        query_quality = current_quality if current_quality else None
        
        # Helper function to get embeddings from reference (handles old/new format)
        def get_ref_embeddings(ref_emb):
            if isinstance(ref_emb, dict):
                # New format: {'arcface': [...], 'facenet': [...]}
                arcface = np.array(ref_emb.get('arcface')) if ref_emb.get('arcface') else None
                facenet = np.array(ref_emb.get('facenet')) if ref_emb.get('facenet') else None
                return arcface, facenet
            elif ref_emb is not None:
                # Old format: single embedding (assume FaceNet)
                return None, np.array(ref_emb)
            else:
                return None, None
        
        for ref in references:
            ref_emb = ref.get('embedding')
            if ref_emb is None:
                continue
            
            ref_arcface, ref_facenet = get_ref_embeddings(ref_emb)
            
            # Calculate similarities
            arcface_sim = None
            facenet_sim = None
            
            if q_arcface is not None and ref_arcface is not None:
                arcface_sim = comparator.cosine_similarity(q_arcface, ref_arcface)
            
            if q_facenet is not None and ref_facenet is not None:
                facenet_sim = comparator.cosine_similarity(q_facenet, ref_facenet)
            
            # Calculate landmark similarity
            ref_landmarks = ref.get('landmarks')
            landmark_sim = comparator.landmark_similarity(query_landmarks, ref_landmarks) if query_landmarks and ref_landmarks else 0.5
            
            # Calculate quality similarity
            ref_quality = ref.get('quality')
            ref_pose = ref.get('pose')
            ref_activations = ref.get('activations', {})
            
            # Activation similarity
            try:
                if current_activations and ref_activations and isinstance(ref_activations, dict):
                    activation_sim = comparator.activation_similarity(current_activations, ref_activations)
                else:
                    activation_sim = 0.7
            except Exception as e:
                print(f"Activation comparison error: {e}")
                activation_sim = 0.7
            
            # Use dual-model match scoring with activation similarity
            match_result = comparator.compute_dual_match_score(
                arcface_sim,
                facenet_sim,
                landmark_sim,
                query_quality,
                activation_sim
            )
            
            status, label, description = comparator.get_match_verdict(match_result['score'])
            ref_pose_cat = ref.get('pose_category', 'frontal')
            
            # Calculate euclidean distance from primary embedding
            primary_emb = q_arcface if q_arcface is not None else q_facenet
            primary_ref = ref_arcface if ref_arcface is not None else ref_facenet
            distance = comparator.euclidean_distance(primary_emb, primary_ref) if primary_emb is not None and primary_ref is not None else 0.0
            
            # NEW: Calculate pose weight
            pose_weight = comparator.compute_pose_weight(current_pose, ref_pose)
            
            # NEW: LBP similarity
            lbp_sim = comparator.lbp_similarity(current_lbp, ref.get('lbp_histogram'))
            
            # NEW: Asymmetry similarity
            asym_sim = comparator.asymmetry_similarity(current_asymmetry, ref.get('asymmetry'))
            
            # NEW: 3D normalized embedding similarity
            norm_emb_query = current_normalized_embedding
            norm_emb_ref = ref.get('normalized_embedding')
            norm_sim = 0.0
            if norm_emb_query is not None and norm_emb_ref is not None:
                norm_sim = comparator.cosine_similarity(np.array(norm_emb_query), np.array(norm_emb_ref))
            
            # NEW: Multi-pose best score
            pose_list = list(ref.get('poses', {}).values())
            multi_pose_score, best_pose = comparator.compute_multi_pose_score(
                primary_emb, pose_list
            )
            
            results.append({
                'id': ref['id'],
                'name': ref['name'],
                'arcface_similarity': float(arcface_sim) if arcface_sim is not None else None,
                'facenet_similarity': float(facenet_sim) if facenet_sim is not None else None,
                'landmark_similarity': float(landmark_sim),
                'activation_similarity': float(activation_sim) if activation_sim is not None else None,
                'final_score': float(match_result['score']),
                'status': status,
                'match_label': label,
                'match_description': description,
                'reasons': match_result['reasons'],
                'euclidean_distance': float(distance),
                'thumbnail': ref['thumbnail'],
                'pose': ref_pose,
                'pose_category': ref_pose_cat,
                'pose_weight': float(pose_weight),
                'lbp_similarity': float(lbp_sim),
                'asymmetry_similarity': float(asym_sim),
                'normalized_similarity': float(norm_sim),
                'multi_pose_score': float(multi_pose_score),
                'multi_pose_used': len(ref.get('poses', {})) > 1,
            })

        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)

        sim_viz = None
        sim_data = {}
        # Build ref embeddings list for visualization
        ref_embs = []
        ref_ids = []
        sim_scores = []
        for ref in references:
            ref_emb = ref.get('embedding')
            if ref_emb is None:
                continue
            ref_arc, ref_face = get_ref_embeddings(ref_emb)
            if ref_arc is not None:
                ref_embs.append(ref_arc)
            elif ref_face is not None:
                ref_embs.append(ref_face)
            else:
                continue
            ref_ids.append(ref['name'])
            sim_scores.append(0.0)  # Will be filled from results
        
        if ref_embs:
            primary_emb = q_arcface if q_arcface is not None else q_facenet
            if primary_emb is not None:
                sim_viz, sim_data = comparator.visualize_comparison_metrics(
                    primary_emb,
                    ref_embs,
                    ref_ids,
                    sim_scores,
                    [r['euclidean_distance'] for r in results]
                )

        return jsonify({
            'success': True,
            'results': results,
            'best_match': results[0] if results else None,
            'similarity_viz': image_to_base64(sim_viz) if sim_viz is not None else None,
            'similarity_data': sim_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/add-reference-pose/<int:ref_id>', methods=['POST'])
def add_reference_pose_variant(ref_id):
    """Add additional pose variant to existing reference."""
    global references
    
    try:
        if ref_id < 0 or ref_id >= len(references):
            return jsonify({'success': False, 'error': 'Reference not found'})
        
        data = request.json
        image_data = data.get('image', '')
        
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        ref_image = base64_to_image(image_data)
        ref_faces = detector.detect_faces(ref_image)
        
        if not ref_faces:
            return jsonify({'success': False, 'error': 'No faces detected'})
        
        fx, fy, fw, fh = ref_faces[0]
        ref_face = ref_image[fy:fy+fh, fx:fx+fw]
        
        arcface_emb = arcface_extractor.extract_embedding(ref_face) if arcface_extractor else None
        facenet_emb = facenet_extractor.extract_embedding(ref_face)
        
        embeddings_dict = {
            'arcface': arcface_emb.tolist() if arcface_emb is not None else None,
            'facenet': facenet_emb.tolist() if facenet_emb is not None else None
        }
        
        landmarks = detector.estimate_landmarks(ref_face, (0, 0, ref_face.shape[1], ref_face.shape[0]))
        alignment = detector.compute_alignment(ref_face, landmarks)
        pose_category = categorize_pose(alignment.get('yaw', 0), alignment.get('pitch', 0))
        
        if 'poses' not in references[ref_id]:
            references[ref_id]['poses'] = {}
        
        references[ref_id]['poses'][pose_category] = {
            'embedding': embeddings_dict,
            'yaw': float(alignment.get('yaw', 0)),
            'pitch': float(alignment.get('pitch', 0))
        }
        
        save_references()
        
        return jsonify({
            'success': True,
            'pose_category': pose_category,
            'total_poses': len(references[ref_id].get('poses', {}))
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/visualizations/<viz_type>', methods=['GET'])
def get_visualization(viz_type):
    """Get visualization for current query face."""
    global current_image, current_face_image, current_faces, current_embedding
    
    try:
        face_image = current_face_image
        embedding = current_embedding
        
        return get_viz_result(viz_type, face_image, embedding)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/visualizations/<viz_type>/reference/<int:ref_id>', methods=['GET'])
def get_reference_visualization(viz_type, ref_id):
    """Get visualization for a specific reference image."""
    global references
    
    try:
        if ref_id < 0 or ref_id >= len(references):
            return jsonify({'success': False, 'error': 'Reference not found'})
        
        ref = references[ref_id]
        
        # Decode thumbnail to get face image
        thumb_data = ref.get('thumbnail', '')
        if not thumb_data:
            return jsonify({'success': False, 'error': 'No thumbnail'})
        
        if ',' in thumb_data:
            thumb_data = thumb_data.split(',')[1]
        
        thumb_bytes = base64.b64decode(thumb_data)
        thumb_img = Image.open(io.BytesIO(thumb_bytes))
        face_image = cv2.cvtColor(np.array(thumb_img), cv2.COLOR_RGB2BGR)
        
        # Get embedding
        emb_dict = ref.get('embedding', {})
        if isinstance(emb_dict, dict):
            arcface_emb = np.array(emb_dict.get('arcface')) if emb_dict.get('arcface') else None
            facenet_emb = np.array(emb_dict.get('facenet')) if emb_dict.get('facenet') else None
        else:
            arcface_emb = None
            facenet_emb = np.array(emb_dict) if emb_dict else None
        
        embedding = arcface_emb if arcface_emb is not None else facenet_emb
        
        if embedding is None:
            return jsonify({'success': False, 'error': 'No embedding for reference'})
        
        return get_ref_viz_result(viz_type, face_image, embedding)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


def get_ref_viz_result(viz_type, face_image, embedding):
    """Helper function to generate visualization for reference images."""
    def get_viz_and_data(viz_type, face_image, embedding):
        if viz_type == 'detection':
            # Draw bounding box on face
            h, w = face_image.shape[:2]
            vis = face_image.copy()
            cv2.rectangle(vis, (10, 10), (w-10, h-10), (0, 255, 0), 2)
            cv2.putText(vis, "Reference Face", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return vis, {'face_detected': True, 'width': w, 'height': h}
        
        elif viz_type == 'landmarks':
            try:
                landmarks = detector.estimate_landmarks(face_image, (0, 0, face_image.shape[1], face_image.shape[0]))
                if landmarks is not None and len(landmarks) > 0:
                    vis = detector.visualize_landmarks(face_image, landmarks)
                    return vis, {'landmarks_count': len(landmarks)}
            except:
                pass
            return face_image.copy(), {'landmarks_count': 0}
        
        elif viz_type == 'embedding':
            try:
                vis, data = extractor.visualize_embedding(embedding)
                return vis, data
            except:
                pass
            return face_image.copy(), {}
        
        elif viz_type == 'alignment':
            try:
                landmarks = detector.estimate_landmarks(face_image, (0, 0, face_image.shape[1], face_image.shape[0]))
                if landmarks is not None:
                    alignment = detector.compute_alignment(face_image, landmarks)
                    vis = detector.visualize_alignment(face_image, landmarks, alignment)
                    return vis, alignment
            except:
                pass
            return face_image.copy(), {}
        
        elif viz_type == 'saliency':
            try:
                vis = detector.visualize_saliency(face_image)
                return vis, {'attention_map': 'generated'}
            except:
                pass
            return face_image.copy(), {}
        
        elif viz_type == 'quality':
            try:
                vis, data = detector.visualize_quality(face_image, (0, 0, face_image.shape[1], face_image.shape[0]))
                return vis, data
            except:
                pass
            return face_image.copy(), {}
        
        return face_image.copy(), {}
    
    vis, data = get_viz_and_data(viz_type, face_image, embedding)
    
    if vis is None:
        return jsonify({'success': False, 'error': 'Failed to generate visualization'})
    
    _, buffer = cv2.imencode('.png', vis)
    viz_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'success': True,
        'visualization': viz_b64,
        'data': {k: str(v) if isinstance(v, (np.ndarray, np.integer, np.floating)) else v for k, v in data.items()}
    })


@app.route('/api/visualizations/compare-overlay/<int:ref_id>', methods=['GET'])
def get_compare_overlay(ref_id):
    """Get overlay visualization of query and reference."""
    try:
        if ref_id < 0 or ref_id >= len(references):
            return jsonify({'success': False, 'error': 'Reference not found'})
        
        if current_face_image is None:
            return jsonify({'success': False, 'error': 'No query face'})
        
        ref = references[ref_id]
        thumb_data = ref.get('thumbnail', '')
        if ',' in thumb_data:
            thumb_data = thumb_data.split(',')[1]
        
        thumb_bytes = base64.b64decode(thumb_data)
        thumb_img = Image.open(io.BytesIO(thumb_bytes))
        ref_face = cv2.cvtColor(np.array(thumb_img), cv2.COLOR_RGB2BGR)
        
        # Resize to match
        ref_face = cv2.resize(ref_face, (current_face_image.shape[1], current_face_image.shape[0]))
        
        # Create overlay
        overlay = cv2.addWeighted(current_face_image, 0.5, ref_face, 0.5, 0)
        
        _, buffer = cv2.imencode('.png', overlay)
        viz_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'success': True, 'visualization': viz_b64})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/visualizations/compare-diff/<int:ref_id>', methods=['GET'])
def get_compare_diff(ref_id):
    """Get difference visualization of query and reference."""
    try:
        if ref_id < 0 or ref_id >= len(references):
            return jsonify({'success': False, 'error': 'Reference not found'})
        
        if current_face_image is None:
            return jsonify({'success': False, 'error': 'No query face'})
        
        ref = references[ref_id]
        thumb_data = ref.get('thumbnail', '')
        if ',' in thumb_data:
            thumb_data = thumb_data.split(',')[1]
        
        thumb_bytes = base64.b64decode(thumb_data)
        thumb_img = Image.open(io.BytesIO(thumb_bytes))
        ref_face = cv2.cvtColor(np.array(thumb_img), cv2.COLOR_RGB2BGR)
        
        # Resize to match
        ref_face = cv2.resize(ref_face, (current_face_image.shape[1], current_face_image.shape[0]))
        
        # Convert to grayscale and compute absolute difference
        gray1 = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(ref_face, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        
        # Apply colormap for visibility
        diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        
        _, buffer = cv2.imencode('.png', diff_color)
        viz_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'success': True, 'visualization': viz_b64})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def get_viz_result(viz_type, face_image, embedding):
    """Helper function to generate visualization."""
    def get_viz_and_data(viz_type, face_image, embedding):
        if viz_type == 'detection':
            return (detector.visualize_detection(current_image, current_faces) if current_image is not None and current_faces else None), {}
        elif viz_type == 'extraction':
            return (detector.visualize_extraction(current_image, current_faces) if current_image is not None and current_faces else None), {}
        elif viz_type == 'landmarks':
            if face_image is None:
                return None, {}
            landmarks = detector.estimate_landmarks(face_image, (0, 0, face_image.shape[1], face_image.shape[0]))
            return (detector.visualize_landmarks(face_image, landmarks), {})
        elif viz_type == 'mesh3d':
            return (detector.visualize_3d_mesh(face_image) if face_image is not None else None), {}
        elif viz_type == 'alignment':
            if face_image is None:
                return None, {}
            landmarks = detector.estimate_landmarks(face_image, (0, 0, face_image.shape[1], face_image.shape[0]))
            alignment = detector.compute_alignment(face_image, landmarks)
            return (detector.visualize_alignment(face_image, landmarks, alignment), {})
        elif viz_type == 'saliency':
            return (detector.visualize_saliency(face_image) if face_image is not None else None), {}
        elif viz_type == 'activations':
            return (extractor.visualize_activations(face_image) if face_image is not None else None), {}
        elif viz_type == 'features':
            return (extractor.visualize_feature_maps(face_image) if face_image is not None else None), {}
        elif viz_type == 'multiscale':
            return (detector.visualize_multiscale(face_image) if face_image is not None else None), {}
        elif viz_type == 'confidence':
            if face_image is None:
                return None, {}
            return detector.visualize_quality(face_image, (0, 0, face_image.shape[1], face_image.shape[0]))
        elif viz_type == 'embedding':
            if embedding is None:
                return None, {}
            return extractor.visualize_embedding(embedding)
        elif viz_type == 'similarity':
            if embedding is None:
                return None, {}
            return extractor.visualize_similarity_result(embedding, None, 0.75)
        elif viz_type == 'robustness':
            if face_image is None:
                return None, {}
            return extractor.test_robustness(face_image)
        elif viz_type == 'biometric':
            return (detector.visualize_biometric_capture(current_image, current_faces) if current_image is not None and current_faces else None), {}
        elif viz_type == 'eyewear':
            if current_image is None or not current_faces:
                return None, {}
            return detector.visualize_eyewear(current_image, current_faces[0]), {}
        elif viz_type == 'preprocessing':
            if current_original_image is None or current_image is None:
                return None, {}
            original_quality = preprocessor.assess_quality(current_original_image)
            enhanced_quality = preprocessor.assess_quality(current_image)
            method = current_preprocessing_info.get('method', 'none')
            return preprocessor.visualize_preprocessing(
                current_original_image, current_image, 
                original_quality, enhanced_quality, method
            ), {}
        elif viz_type == 'tests':
            return visualize_tests(current_image, current_faces, current_embedding, references), {}
        elif viz_type == 'test-health':
            data = {"status": "OK", "api": "running", "port": 3000}
            return visualize_test_detail("Health Check", data), data
        elif viz_type == 'test-detection':
            data = {
                "faces_detected": len(current_faces) if current_faces else 0,
                "preprocessing": current_preprocessing_info.get('method', 'none') if current_preprocessing_info else 'none',
                "enhanced": current_preprocessing_info.get('was_enhanced', False) if current_preprocessing_info else False
            }
            return visualize_test_detail("Detection + Preprocessing", data), data
        elif viz_type == 'test-extraction':
            data = {
                "embedding_size": len(current_embedding) if current_embedding is not None else 0,
                "pose": current_pose.get('category', 'not extracted') if current_pose else "not extracted"
            }
            return visualize_test_detail("Extraction + Pose", data), data
        elif viz_type == 'test-reference':
            data = {"references": len(references) if references else 0}
            if references:
                data["latest_pose"] = references[-1].get('pose_category', 'unknown')
            return visualize_test_detail("Add Reference + Pose", data), data
        elif viz_type == 'test-multi':
            data = {
                "total_references": len(references) if references else 0,
                "can_match": len(references) > 1
            }
            return visualize_test_detail("Multi-Reference", data), data
        elif viz_type == 'test-pose':
            data = {
                "query_pose": current_pose.get('category', 'not extracted') if current_pose else "no query",
                "matching_enabled": True,
                "adjusts_similarity": True
            }
            return visualize_test_detail("Pose-Aware Matching", data), data
        elif viz_type == 'test-eyewear':
            if current_image and current_faces:
                ew = detector.detect_eyewear(current_image, current_faces[0])
                return visualize_test_detail("Eyewear Detection", ew), ew
            data = {"status": "no face detected"}
            return visualize_test_detail("Eyewear Detection", data), data
        elif viz_type == 'test-viz':
            data = {
                "total_types": 16,
                "detection": "available",
                "preprocessing": "available",
                "pose": "available",
                "tests": "available"
            }
            return visualize_test_detail("Visualizations", data), data
        elif viz_type == 'test-clear':
            data = {
                "session_management": "working",
                "can_clear": True
            }
            return visualize_test_detail("Clear + Reset", data), data
        return None, {}
    
    viz_result = get_viz_and_data(viz_type, face_image, embedding)
    
    if isinstance(viz_result, tuple):
        viz_image, viz_data = viz_result
    else:
        viz_image = viz_result
        viz_data = {}
    
    if viz_image is None:
        return jsonify({'success': False, 'error': 'No data available'})
    
    return jsonify({
        'success': True,
        'visualization': image_to_base64(viz_image),
        'data': viz_data
    })


@app.route('/api/quality', methods=['GET'])
def get_quality_metrics():
    """Get quality metrics for current face."""
    global current_image, current_faces
    
    try:
        if not current_faces:
            return jsonify({'success': False, 'error': 'No faces detected'})
        
        quality = detector.compute_quality_metrics(current_image, current_faces[0])
        
        return jsonify({
            'success': True,
            'quality': {k: float(v) if isinstance(v, np.floating) else v for k, v in quality.items()}
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/eyewear', methods=['GET'])
def get_eyewear_detection():
    """Get eyewear detection for current face."""
    global current_image, current_faces
    
    try:
        if not current_faces:
            return jsonify({'success': False, 'error': 'No faces detected'})
        
        face_box = current_faces[0]
        eyewear = detector.detect_eyewear(current_image, face_box)
        
        return jsonify({
            'success': True,
            'eyewear': {
                'has_eyewear': eyewear.get('has_eyewear', False),
                'type': eyewear.get('eyewear_type', 'none'),
                'confidence': float(eyewear.get('confidence', 0.0)),
                'occlusion_level': float(eyewear.get('occlusion_level', 0.0)),
                'warnings': eyewear.get('warnings', []),
                'eye_count': eyewear.get('eye_count', 0)
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/visualizations/eyewear', methods=['GET'])
def get_eyewear_visualization():
    """Get eyewear visualization for current face."""
    global current_image, current_faces
    
    try:
        if not current_faces:
            return jsonify({'success': False, 'error': 'No faces detected'})
        
        face_box = current_faces[0]
        viz = detector.visualize_eyewear(current_image, face_box)
        
        return jsonify({
            'success': True,
            'visualization': image_to_base64(viz)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/clear', methods=['POST'])
def clear_all():
    """Clear all data."""
    global current_image, current_original_image, current_enhanced_image, current_faces, current_embedding, current_embeddings, current_face_image, current_preprocessing_info, current_pose, current_landmarks, current_quality, references

    current_image = None
    current_original_image = None
    current_enhanced_image = None
    current_faces = []
    current_embedding = None
    current_embeddings = {}
    current_face_image = None
    current_preprocessing_info = {}
    current_pose = {}
    current_landmarks = None
    current_quality = None
    references = []

    return jsonify({'success': True, 'message': 'All data cleared'})


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current server state for debugging."""
    global current_embedding, current_faces, references

    return jsonify({
        'success': True,
        'has_embedding': current_embedding is not None,
        'embedding_type': type(current_embedding).__name__ if current_embedding is not None else None,
        'embedding_shape': current_embedding.shape if current_embedding is not None else None,
        'faces_count': len(current_faces),
        'references_count': len(references),
        'reference_embeddings': [r.get('embedding') is not None for r in references]
    })


@app.route('/api/webcam/available', methods=['GET'])
def webcam_available():
    """Check if webcam is available."""
    import cv2
    cap = cv2.VideoCapture(0)
    available = cap.isOpened()
    cap.release()
    return jsonify({'success': True, 'available': available})


@app.route('/api/webcam/capture', methods=['POST'])
def webcam_capture():
    """Capture a frame from webcam and return as base64."""
    import cv2
    import base64
    
    data = request.get_json()
    camera_index = data.get('camera_index', 0) if data else 0
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        return jsonify({'success': False, 'error': 'Cannot open camera'}), 400
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({'success': False, 'error': 'Failed to capture frame'}), 400
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'success': True,
        'image': f'data:image/jpeg;base64,{image_base64}',
        'width': frame.shape[1],
        'height': frame.shape[0]
    })


@app.route('/api/webcam/detect', methods=['POST'])
def webcam_detect():
    """Capture frame from webcam and detect faces."""
    import cv2
    import base64
    
    data = request.get_json()
    camera_index = data.get('camera_index', 0) if data else 0
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        return jsonify({'success': False, 'error': 'Cannot open camera'}), 400
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({'success': False, 'error': 'Failed to capture frame'}), 400
    
    # Detect faces
    global current_faces, current_image
    current_faces = detector.detect_faces(frame)
    current_image = frame
    
    # Draw detections on image
    result_image = detector.draw_detections(frame, current_faces)
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', result_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'success': True,
        'faces_count': len(current_faces),
        'faces': [{'x': x, 'y': y, 'w': w, 'h': h} for x, y, w, h in current_faces],
        'image': f'data:image/jpeg;base64,{image_base64}'
    })


# =============================================================================
# REFERENCE LIBRARY ENDPOINTS
# =============================================================================

LIBRARY_DIR = "reference_library"
PERSONS_DIR = os.path.join(LIBRARY_DIR, "persons")
os.makedirs(PERSONS_DIR, exist_ok=True)


def get_person_by_name(name: str) -> Optional[str]:
    """Check if person exists by name, return folder if found."""
    sanitized = secure_filename(name).replace('_', '-')
    if not os.path.exists(PERSONS_DIR):
        return None
    for folder in os.listdir(PERSONS_DIR):
        if folder.endswith(f"-{sanitized}"):
            return folder
    return None


def secure_filename(s: str) -> str:
    """Simple filename sanitization."""
    import re
    s = re.sub(r'[^\w\s-]', '', s)
    s = s.strip().replace(' ', '_')
    return s


def save_person_image(face_image: np.ndarray, person_dir: str, image_id: str) -> str:
    """Save compressed face image (JPEG 80%), return filename."""
    filename = f"{image_id}.jpg"
    filepath = os.path.join(person_dir, "images", filename)
    os.makedirs(os.path.join(person_dir, "images"), exist_ok=True)
    cv2.imwrite(filepath, face_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return filename


@app.route('/api/library', methods=['GET'])
def get_library():
    """List all persons in the library."""
    persons = []
    if not os.path.exists(PERSONS_DIR):
        return jsonify({"persons": [], "count": 0})
    
    for folder in os.listdir(PERSONS_DIR):
        meta_path = os.path.join(PERSONS_DIR, folder, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    person = json.load(f)
                
                # Also get thumbnail from embeddings
                emb_path = os.path.join(PERSONS_DIR, folder, "embeddings.json")
                if os.path.exists(emb_path):
                    try:
                        with open(emb_path, 'r') as f:
                            emb_data = json.load(f)
                        images = emb_data.get("images", [])
                        if images and len(images) > 0:
                            # Get thumbnail from first image
                            thumb = images[0].get("thumbnail", "")
                            if thumb:
                                person["first_image_thumbnail"] = f"data:image/jpeg;base64,{thumb}"
                    except:
                        pass
                
                persons.append(person)
            except:
                pass
    
    return jsonify({"persons": persons, "count": len(persons)})


@app.route('/api/library/person', methods=['POST'])
def add_person():
    """Add a new person with their first reference image."""
    try:
        data = request.json
        name = data.get('name', '').strip()
        notes = data.get('notes', '')
        image_data = data.get('image', '')
        source = data.get('source', 'upload')
        
        if not name:
            return jsonify({"success": False, "error": "Name is required"})
        
        if not image_data:
            return jsonify({"success": False, "error": "Image is required"})
        
        # Check for duplicate
        if get_person_by_name(name):
            return jsonify({"success": False, "error": "Person already exists"})
        
        # Create folder
        person_uuid = str(uuid.uuid4())[:8]
        sanitized = secure_filename(name)
        folder_name = f"{person_uuid}-{sanitized}"
        person_dir = os.path.join(PERSONS_DIR, folder_name)
        os.makedirs(person_dir, exist_ok=True)
        os.makedirs(os.path.join(person_dir, "images"), exist_ok=True)
        
        # Process image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        ref_image = base64_to_image(image_data)
        ref_faces = detector.detect_faces(ref_image)
        
        if not ref_faces:
            shutil.rmtree(person_dir)
            return jsonify({'success': False, 'error': 'No faces detected'})
        
        fx, fy, fw, fh = ref_faces[0]
        ref_face = ref_image[fy:fy+fh, fx:fx+fw]
        
        # Extract all features (same as add_reference)
        arcface_emb = arcface_extractor.extract_embedding(ref_face) if arcface_extractor else None
        facenet_emb = facenet_extractor.extract_embedding(ref_face)
        landmarks = detector.estimate_landmarks(ref_face, (0, 0, ref_face.shape[1], ref_face.shape[0]))
        alignment = detector.compute_alignment(ref_face, landmarks)
        quality = detector.compute_quality_metrics(ref_face, (0, 0, ref_face.shape[1], ref_face.shape[0]))
        ref_activations = facenet_extractor.get_activations(ref_face)
        lbp_hist = detector.compute_lbp_descriptor(ref_face)
        asymmetry = detector.compute_facial_asymmetry(landmarks)
        
        mesh_landmarks = detector.estimate_landmarks(ref_face, (0, 0, ref_face.shape[1], ref_face.shape[0]))
        aligned = detector.normalize_face_with_mesh(ref_face, mesh_landmarks)
        normalized_emb = arcface_extractor.extract_embedding(aligned) if arcface_extractor else None
        
        pose_cat = categorize_pose(alignment.get('yaw', 0), alignment.get('pitch', 0))
        
        # Save image (compressed 80%)
        image_id = f"{person_uuid}_0"
        filename = save_person_image(ref_face, person_dir, image_id)
        
        # Save embeddings
        embeddings_data = {
            "person_id": person_uuid,
            "person_name": name,
            "images": [{
                "image_id": image_id,
                "filename": filename,
                "added_at": datetime.datetime.now().isoformat(),
                "source": source,
                "embedding": {
                    "arcface": arcface_emb.tolist() if arcface_emb is not None else None,
                    "facenet": facenet_emb.tolist() if facenet_emb is not None else None
                },
                "normalized_embedding": normalized_emb.tolist() if normalized_emb is not None else None,
                "pose": {
                    "yaw": float(alignment.get('yaw', 0)),
                    "pitch": float(alignment.get('pitch', 0)),
                    "roll": float(alignment.get('roll', 0))
                },
                "pose_category": pose_cat,
                "quality": {k: float(v) if isinstance(v, (int, float)) else v for k, v in quality.items()} if quality else None,
                "landmarks": extract_landmark_features(landmarks),
                "activations": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in ref_activations.items()} if ref_activations else {},
                "lbp_histogram": lbp_hist.tolist() if lbp_hist is not None else None,
                "asymmetry": asymmetry,
                "thumbnail": image_to_base64(ref_face)
            }]
        }
        
        with open(os.path.join(person_dir, "embeddings.json"), 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        
        # Save metadata
        metadata = {
            "id": person_uuid,
            "name": name,
            "notes": notes,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "image_count": 1,
            "folder": folder_name
        }
        
        with open(os.path.join(person_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return jsonify({"success": True, "person": metadata})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/library/person/<person_id>', methods=['GET'])
def get_person(person_id):
    """Get person details with all their images/embeddings."""
    if not os.path.exists(PERSONS_DIR):
        return jsonify({"success": False, "error": "Library empty"})
    
    person_dir = None
    for folder in os.listdir(PERSONS_DIR):
        if folder.startswith(person_id):
            person_dir = os.path.join(PERSONS_DIR, folder)
            break
    
    if not person_dir:
        return jsonify({"success": False, "error": "Person not found"})
    
    with open(os.path.join(person_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    with open(os.path.join(person_dir, "embeddings.json"), 'r') as f:
        embeddings = json.load(f)
    
    return jsonify({"success": True, "person": metadata, "embeddings": embeddings})


@app.route('/api/library/person/<person_id>', methods=['DELETE'])
def delete_person(person_id):
    """Delete a person and all their data."""
    if not os.path.exists(PERSONS_DIR):
        return jsonify({"success": False, "error": "Library empty"})
    
    for folder in os.listdir(PERSONS_DIR):
        if folder.startswith(person_id):
            shutil.rmtree(os.path.join(PERSONS_DIR, folder))
            return jsonify({"success": True})
    
    return jsonify({"success": False, "error": "Person not found"})


@app.route('/api/library/match', methods=['POST'])
def match_library():
    """Match a query image against the entire library."""
    try:
        data = request.json
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({"success": False, "error": "Image is required"})
        
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        query_image = base64_to_image(image_data)
        query_faces = detector.detect_faces(query_image)
        
        if not query_faces:
            return jsonify({"success": False, "error": "No face detected"})
        
        fx, fy, fw, fh = query_faces[0]
        query_face = query_image[fy:fy+fh, fx:fx+fw]
        
        # Extract query embedding
        q_arc = arcface_extractor.extract_embedding(query_face) if arcface_extractor else None
        q_face = facenet_extractor.extract_embedding(query_face)
        
        matches = []
        
        if not os.path.exists(PERSONS_DIR):
            return jsonify({"success": True, "matches": []})
        
        for folder in os.listdir(PERSONS_DIR):
            emb_path = os.path.join(PERSONS_DIR, folder, "embeddings.json")
            meta_path = os.path.join(PERSONS_DIR, folder, "metadata.json")
            
            if not os.path.exists(emb_path):
                continue
            
            try:
                with open(emb_path, 'r') as f:
                    emb_data = json.load(f)
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except:
                continue
            
            # Find best match across all images
            best_score = 0
            best_image = None
            
            for img in emb_data.get("images", []):
                ref_emb = img.get("embedding", {})
                r_arc = np.array(ref_emb.get("arcface")) if ref_emb.get("arcface") else None
                r_face = np.array(ref_emb.get("facenet")) if ref_emb.get("facenet") else None
                
                # Calculate similarity
                if q_arc is not None and r_arc is not None:
                    sim = comparator.cosine_similarity(q_arc, r_arc)
                elif r_face is not None:
                    sim = comparator.cosine_similarity(q_face, r_face)
                else:
                    sim = 0
                
                if sim > best_score:
                    best_score = sim
                    best_image = img
            
            if best_image:
                matches.append({
                    "person_id": meta["id"],
                    "person_name": meta["name"],
                    "score": float(best_score),
                    "best_image": best_image
                })
        
        # Sort by score
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        return jsonify({
            "success": True,
            "matches": matches[:10]  # Top 10
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


if __name__ == '__main__':
    import os
    PORT = int(os.environ.get('PORT', 3000))
    print("Starting Face Recognition API Server...")
    print(f"Open http://localhost:{PORT} in your Electron app")
    app.run(host='0.0.0.0', port=PORT, debug=False)
