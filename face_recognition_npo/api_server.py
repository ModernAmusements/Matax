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
        import traceback
        traceback.print_exc()
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
            'embedding_mean': float(np.mean(current_embedding)) if current_embedding is not None else 0,
            'embedding_std': float(np.std(current_embedding)) if current_embedding is not None else 0,
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
        ref_embedding = extractor.extract_embedding(ref_face)
        
        ref_data = {
            'id': len(references),
            'name': name,
            'embedding': ref_embedding.tolist() if ref_embedding is not None else None,
            'thumbnail': image_to_base64(ref_face),
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

            distance = comparator.euclidean_distance(current_embedding, ref_emb)
            similarity = comparator.cosine_similarity(current_embedding, ref_emb)
            distance_verdict = comparator.get_distance_verdict(distance)
            
            results.append({
                'id': ref['id'],
                'name': ref['name'],
                'similarity': float(similarity),
                'euclidean_distance': float(distance),
                'distance_verdict': distance_verdict,
                'thumbnail': ref['thumbnail']
            })

        results.sort(key=lambda x: x['similarity'], reverse=True)

        sim_viz = None
        sim_data = {}
        if ref_embeddings:
            similarities = [r['similarity'] for r in results]
            distances = [r['euclidean_distance'] for r in results]
            sim_viz, sim_data = extractor.model.visualize_comparison_metrics(
                current_embedding,
                ref_embeddings,
                ref_names,
                similarities,
                distances
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
        if ref.get('embedding') is None:
            return jsonify({'success': False, 'error': 'No embedding for reference'})
        
        # Reconstruct face from thumbnail
        import base64
        from PIL import Image
        import io
        
        thumb_data = ref.get('thumbnail', '')
        if not thumb_data:
            return jsonify({'success': False, 'error': 'No thumbnail'})
        
        # Decode base64 thumbnail
        if ',' in thumb_data:
            thumb_data = thumb_data.split(',')[1]
        
        thumb_bytes = base64.b64decode(thumb_data)
        thumb_img = Image.open(io.BytesIO(thumb_bytes))
        face_image = cv2.cvtColor(np.array(thumb_img), cv2.COLOR_RGB2BGR)
        
        embedding = np.array(ref['embedding'])
        
        return get_viz_result(viz_type, face_image, embedding)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
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


if __name__ == '__main__':
    print("Starting Face Recognition API Server...")
    print("Open http://localhost:3000 in your Electron app")
    app.run(host='0.0.0.0', port=3000, debug=False)
