#!/usr/bin/env python3
"""
Eyewear Detection Test
Tests the sunglasses/glasses detection feature.
"""

import sys
import os
import json
import base64
import numpy as np

API_BASE = "http://localhost:3000"

def image_to_base64(image_path):
    """Convert image file to base64."""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return None
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def test_eyewear_api():
    """Test the eyewear detection API endpoint."""
    import requests
    
    test_images = [
        'test_images/test_subject.jpg',
        '_examples/reference_images/kanye_west_ref.jpg'
    ]
    
    img_b64 = None
    for path in test_images:
        if os.path.exists(path):
            img_b64 = image_to_base64(path)
            print(f"Using test image: {path}")
            break
    
    if not img_b64:
        print("ERROR: No test images found")
        return False
    
    print("\n1. Testing /api/detect endpoint...")
    detect_response = requests.post(
        f"{API_BASE}/api/detect",
        json={"image": img_b64},
        timeout=30
    )
    
    if detect_response.status_code != 200:
        print(f"ERROR: /api/detect failed with status {detect_response.status_code}")
        return False
    
    detect_data = detect_response.json()
    if not detect_data.get('success'):
        print(f"ERROR: /api/detect failed: {detect_data.get('error')}")
        return False
    
    print(f"   ✓ Found {detect_data.get('count')} face(s)")
    
    print("\n2. Testing /api/eyewear endpoint...")
    eyewear_response = requests.get(f"{API_BASE}/api/eyewear", timeout=10)
    
    if eyewear_response.status_code != 200:
        print(f"ERROR: /api/eyewear failed with status {eyewear_response.status_code}")
        return False
    
    eyewear_data = eyewear_response.json()
    
    if not eyewear_data.get('success'):
        print(f"ERROR: /api/eyewear failed: {eyewear_data.get('error')}")
        return False
    
    eyewear = eyewear_data.get('eyewear', {})
    print(f"   ✓ Has eyewear: {eyewear.get('has_eyewear')}")
    print(f"   ✓ Type: {eyewear.get('type')}")
    print(f"   ✓ Confidence: {eyewear.get('confidence')}")
    print(f"   ✓ Occlusion level: {eyewear.get('occlusion_level')}")
    print(f"   ✓ Warnings: {eyewear.get('warnings')}")
    print(f"   ✓ Eye count: {eyewear.get('eye_count')}")
    
    print("\n3. Testing /api/visualizations/eyewear endpoint...")
    viz_response = requests.get(f"{API_BASE}/api/visualizations/eyewear", timeout=10)
    
    if viz_response.status_code != 200:
        print(f"ERROR: /api/visualizations/eyewear failed with status {viz_response.status_code}")
        return False
    
    viz_data = viz_response.json()
    
    if not viz_data.get('success'):
        print(f"ERROR: /api/visualizations/eyewear failed: {viz_data.get('error')}")
        return False
    
    if not viz_data.get('visualization'):
        print("ERROR: No visualization returned")
        return False
    
    print(f"   ✓ Visualization received ({len(viz_data.get('visualization'))} chars)")
    
    return True

def test_eyewear_backend():
    """Test eyewear detection directly on backend."""
    import cv2
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.detection import FaceDetector
    
    detector = FaceDetector()
    
    test_images = [
        'test_images/test_subject.jpg',
        '_examples/reference_images/kanye_west_ref.jpg'
    ]
    
    img = None
    for path in test_images:
        if os.path.exists(path):
            img = cv2.imread(path)
            print(f"Using test image: {path}")
            break
    
    if img is None:
        print("ERROR: No test images found")
        return False
    
    faces = detector.detect_faces(img)
    if not faces:
        print("ERROR: No faces detected")
        return False
    
    print(f"\n✓ Detected {len(faces)} face(s)")
    
    eyewear = detector.detect_eyewear(img, faces[0])
    print("\n=== Eyewear Detection Results ===")
    print(f"Has eyewear: {eyewear.get('has_eyewear')}")
    print(f"Type: {eyewear.get('eyewear_type')}")
    print(f"Confidence: {eyewear.get('confidence')}")
    print(f"Occlusion level: {eyewear.get('occlusion_level')}")
    print(f"Warnings: {eyewear.get('warnings')}")
    print(f"Eye count: {eyewear.get('eye_count')}")
    
    viz = detector.visualize_eyewear(img, faces[0])
    print(f"\n✓ Visualization generated: {viz.shape}")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("EYEWEAR DETECTION TEST")
    print("="*60)
    
    print("\n=== Part 1: Backend Unit Test ===")
    try:
        backend_passed = test_eyewear_backend()
        print(f"\nBackend test: {'✓ PASS' if backend_passed else '✗ FAIL'}")
    except Exception as e:
        print(f"\n✗ Backend test failed: {e}")
        backend_passed = False
    
    print("\n=== Part 2: API Endpoint Test ===")
    print("NOTE: Make sure API server is running on localhost:3000")
    print("Start with: cd face_recognition_npo && python api_server.py")
    
    try:
        api_passed = test_eyewear_api()
        print(f"\nAPI test: {'✓ PASS' if api_passed else '✗ FAIL'}")
    except Exception as e:
        print(f"\n✗ API test failed: {e}")
        print("   (Is the API server running?)")
        api_passed = False
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Backend: {'✓ PASS' if backend_passed else '✗ FAIL'}")
    print(f"API:     {'✓ PASS' if api_passed else '✗ FAIL'}")
    
    if backend_passed and api_passed:
        print("\n✓ ALL EYEWEAR TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED")
        sys.exit(1)
