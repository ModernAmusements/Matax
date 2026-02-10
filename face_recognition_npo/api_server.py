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
from typing import List, Dict, Tuple, Optional
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.detection import FaceDetector
from src.embedding import FaceNetEmbeddingExtractor, SimilarityComparator

app = Flask(__name__, static_folder='./electron-ui')
CORS(app)

detector = FaceDetector()
extractor = FaceNetEmbeddingExtractor()
comparator = SimilarityComparator(threshold=0.5)

current_image = None
current_faces = []
current_embedding = None
current_face_image = None
references = []


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


@app.route('/api/detect', methods=['POST'])
def detect_faces():
    """Detect faces in uploaded image."""
    global current_image, current_faces
    
    try:
        data = request.json
        image_data = data.get('image', '')
        
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        current_image = base64_to_image(image_data)
        current_faces = detector.detect_faces(current_image)
        
        faces_data = []
        for i, (x, y, w, h) in enumerate(current_faces):
            face_img = current_image[y:y+h, x:x+w]
            faces_data.append({
                'id': i,
                'bbox': [np_to_python(x), np_to_python(y), np_to_python(w), np_to_python(h)],
                'thumbnail': image_to_base64(face_img)
            })
        
        return jsonify({
            'success': True,
            'count': len(current_faces),
            'faces': faces_data,
            'visualizations': {
                'detection': image_to_base64(detector.visualize_detection(current_image, current_faces)),
                'extraction': image_to_base64(detector.visualize_extraction(current_image, current_faces)),
                'biometric': image_to_base64(detector.visualize_biometric_capture(current_image, current_faces)),
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/extract', methods=['POST'])
def extract_embedding():
    """Extract embedding from detected face."""
    global current_embedding, current_face_image, current_faces
    
    try:
        data = request.json
        face_id = data.get('face_id', 0)
        
        if not current_faces:
            return jsonify({'success': False, 'error': 'No faces detected'})
        
        x, y, w, h = current_faces[face_id]
        current_face_image = current_image[y:y+h, x:x+w]
        current_embedding = extractor.extract_embedding(current_face_image)
        
        return jsonify({
            'success': True,
            'embedding_size': len(current_embedding) if current_embedding is not None else 0,
            'embedding_mean': float(current_embedding.mean()) if current_embedding is not None else 0,
            'embedding_std': float(current_embedding.std()) if current_embedding is not None else 0,
            'visualizations': {
                'embedding': image_to_base64(extractor.visualize_embedding(current_embedding)),
                'activations': image_to_base64(extractor.visualize_activations(current_face_image)),
                'features': image_to_base64(extractor.visualize_feature_maps(current_face_image)),
                'robustness': image_to_base64(extractor.test_robustness(current_face_image)[0]),
                'landmarks': image_to_base64(detector.visualize_landmarks(current_face_image, detector.estimate_landmarks(current_face_image, (0, 0, current_face_image.shape[1], current_face_image.shape[0])))),
                'mesh3d': image_to_base64(detector.visualize_3d_mesh(current_face_image)),
                'alignment': image_to_base64(detector.visualize_alignment(current_face_image, detector.estimate_landmarks(current_face_image, (0, 0, current_face_image.shape[1], current_face_image.shape[0])), detector.compute_alignment(current_face_image, detector.estimate_landmarks(current_face_image, (0, 0, current_face_image.shape[1], current_face_image.shape[0]))))),
                'saliency': image_to_base64(detector.visualize_saliency(current_face_image)),
                'multiscale': image_to_base64(detector.visualize_multiscale(current_face_image)),
                'confidence': image_to_base64(detector.visualize_quality(current_face_image, (0, 0, current_face_image.shape[1], current_face_image.shape[0]))),
            }
        })
        
    except Exception as e:
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
        ref_embedding = extractor.extract_embedding(ref_face)
        
        ref_data = {
            'id': len(references),
            'name': name,
            'embedding': ref_embedding.tolist() if ref_embedding is not None else None,
            'thumbnail': image_to_base64(ref_face),
        }
        references.append(ref_data)
        
        return jsonify({
            'success': True,
            'reference': ref_data,
            'count': len(references)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


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


@app.route('/api/compare', methods=['POST'])
def compare_faces():
    """Compare current face embedding with references."""
    global current_embedding, references
    
    try:
        if current_embedding is None:
            return jsonify({'success': False, 'error': 'No embedding extracted'})
        
        if not references:
            return jsonify({'success': False, 'error': 'No references added'})

        results = []
        ref_embeddings = []
        ref_names = []

        for ref in references:
            if ref['embedding'] is None:
                continue

            ref_emb = np.array(ref['embedding'])
            ref_embeddings.append(ref_emb)
            ref_names.append(ref['name'])

            similarity = comparator.cosine_similarity(current_embedding, ref_emb)
            confidence = comparator.get_confidence_band(similarity)
            
            results.append({
                'id': ref['id'],
                'name': ref['name'],
                'similarity': float(similarity),
                'confidence': confidence,
                'thumbnail': ref['thumbnail']
            })

        results.sort(key=lambda x: x['similarity'], reverse=True)

        # Generate similarity visualization
        sim_viz = None
        if ref_embeddings:
            sim_viz = extractor.visualize_similarity_matrix(
                current_embedding,
                ref_embeddings,
                ref_names
            )

        return jsonify({
            'success': True,
            'results': results,
            'best_match': results[0] if results else None,
            'similarity_viz': image_to_base64(sim_viz) if sim_viz is not None else None
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/visualizations/<viz_type>', methods=['GET'])
def get_visualization(viz_type):
    """Get specific visualization."""
    global current_image, current_face_image, current_faces, current_embedding
    
    try:
        viz_methods = {
            'detection': lambda: detector.visualize_detection(current_image, current_faces) if current_image is not None and current_faces else None,
            'extraction': lambda: detector.visualize_extraction(current_image, current_faces) if current_image is not None and current_faces else None,
            'landmarks': lambda: detector.visualize_landmarks(current_face_image, detector.estimate_landmarks(current_face_image, (0, 0, current_face_image.shape[1], current_face_image.shape[0]))) if current_face_image is not None else None,
            'mesh3d': lambda: detector.visualize_3d_mesh(current_face_image) if current_face_image is not None else None,
            'alignment': lambda: detector.visualize_alignment(current_face_image, detector.estimate_landmarks(current_face_image, (0, 0, current_face_image.shape[1], current_face_image.shape[0])), detector.compute_alignment(current_face_image, detector.estimate_landmarks(current_face_image, (0, 0, current_face_image.shape[1], current_face_image.shape[0])))) if current_face_image is not None else None,
            'saliency': lambda: detector.visualize_saliency(current_face_image) if current_face_image is not None else None,
            'activations': lambda: extractor.visualize_activations(current_face_image) if current_face_image is not None else None,
            'features': lambda: extractor.visualize_feature_maps(current_face_image) if current_face_image is not None else None,
            'multiscale': lambda: detector.visualize_multiscale(current_face_image) if current_face_image is not None else None,
            'confidence': lambda: detector.visualize_quality(current_face_image, (0, 0, current_face_image.shape[1], current_face_image.shape[0])) if current_face_image is not None else None,
            'embedding': lambda: extractor.visualize_embedding(current_embedding) if current_embedding is not None else None,
            'similarity': lambda: extractor.visualize_similarity_result(current_embedding, None, 0.75) if current_embedding is not None else None,
            'robustness': lambda: extractor.test_robustness(current_face_image)[0] if current_face_image is not None else None,
            'biometric': lambda: detector.visualize_biometric_capture(current_image, current_faces) if current_image is not None and current_faces else None,
        }
        
        viz_fn = viz_methods.get(viz_type)
        if viz_fn is None:
            return jsonify({'success': False, 'error': f'Unknown visualization: {viz_type}'})
        
        result = viz_fn()
        if result is None:
            return jsonify({'success': False, 'error': 'No data available'})
        
        return jsonify({
            'success': True,
            'visualization': image_to_base64(result)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


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
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/clear', methods=['POST'])
def clear_all():
    """Clear all data."""
    global current_image, current_faces, current_embedding, current_face_image, references
    
    current_image = None
    current_faces = []
    current_embedding = None
    current_face_image = None
    references = []
    
    return jsonify({'success': True, 'message': 'All data cleared'})


if __name__ == '__main__':
    print("Starting Face Recognition API Server...")
    print("Open http://localhost:3000 in your Electron app")
    app.run(host='0.0.0.0', port=3000, debug=False)
