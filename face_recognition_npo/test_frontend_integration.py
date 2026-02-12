#!/usr/bin/env python3
"""
Frontend Integration Tests - Rich Visual Version
"""

import sys
import os
import json
import base64
import numpy as np
import requests
import time
from datetime import datetime

API_BASE = "http://localhost:3000"
TEST_IMAGE = 'test_images/test_subject.jpg'
TEST_IMAGE_2 = 'test_images/reference_subject.jpg'

# Colors
C = {
    'reset': '\033[0m',
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'magenta': '\033[95m',
    'cyan': '\033[96m',
    'white': '\033[97m',
    'gray': '\033[90m',
    'bold': '\033[1m',
    'dim': '\033[2m',
}

# Spinner frames
SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

def spin_frames(frame):
    return f"{C['cyan']}{frame}{C['reset']}"

def image_to_base64(image_path):
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return None
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def wait_for_api(timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{API_BASE}/api/health", timeout=2)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False

def print_header():
    print(f"""
{C['bold']}{C['cyan']}══════════════════════════════════════════════════════════════════════{C['reset']}
{C['bold']}{C['white']}                    FRONTEND INTEGRATION TESTS                     {C['reset']}
{C['bold']}{C['cyan']}══════════════════════════════════════════════════════════════════════{C['reset']}
""")

def print_step(num, total, name, status, details=""):
    arrow = f"{C['cyan']}➜{C['reset']}"
    if status == "running":
        frame = SPINNER_FRAMES[int(time.time() * 10) % len(SPINNER_FRAMES)]
        print(f"  {spin_frames(frame)} {C['white']}[{C['cyan']}{num}/{C['reset']}{C['white']}] {name}...{C['reset']}", end='\r')
    elif status == "pass":
        print(f"  {C['green']}✓{C['reset']} {C['white']}[{num}/{total}] {name}{C['reset']}")
        if details:
            print(f"     {C['gray']}{details}{C['reset']}")
    elif status == "fail":
        print(f"  {C['red']}✗{C['reset']} {C['white']}[{num}/{total}] {name}{C['reset']}")
        if details:
            print(f"     {C['red']}{details}{C['reset']}")
    elif status == "skip":
        print(f"  {C['yellow']}◦{C['reset']} {C['white']}[{num}/{total}] {name}{C['reset']}")
    sys.stdout.flush()

def print_result(name, passed, details=""):
    if passed:
        print(f"     {C['green']}✓ PASS{C['reset']} {C['gray']}{details}{C['reset']}")
    else:
        print(f"     {C['red']}✗ FAIL{C['reset']} {C['red']}{details}{C['reset']}")

def print_summary(passed, failed, total_time):
    print(f"""
{C['bold']}{C['cyan']}══════════════════════════════════════════════════════════════════════{C['reset']}
{C['bold']}{C['white']}                         TEST RESULTS SUMMARY                        {C['reset']}
{C['bold']}{C['cyan']}══════════════════════════════════════════════════════════════════════{C['reset']}

  {C['green']}Passed:{C['reset']}  {C['bold']}{passed}{C['reset']}
  {C['red']}Failed:{C['reset']}  {C['bold']}{failed}{C['reset']}
  {C['gray']}Total:{C['reset']}   {C['bold']}{passed + failed}{C['reset']}
  {C['gray']}Time:{C['reset']}    {C['bold']}{total_time:.2f}s{C['reset']}

{C['bold']}{C['cyan']}══════════════════════════════════════════════════════════════════════{C['reset']}""")
    
    if failed == 0:
        print(f"  {C['green']}{C['bold']}ALL TESTS PASSED!{C['reset']}")
    else:
        print(f"  {C['red']}{C['bold']}{failed} TEST(S) FAILED{C['reset']}")
    
    print(f"{C['bold']}{C['cyan']}══════════════════════════════════════════════════════════════════════{C['reset']}")

# =============================================================================
# TESTS
# =============================================================================

def test_health(step_num, total):
    print_step(step_num, total, "Health Check", "running")
    time.sleep(0.3)
    
    resp = requests.get(f"{API_BASE}/api/health")
    passed = resp.status_code == 200 and resp.json().get('status') == 'ok'
    
    print_step(step_num, total, "Health Check", "pass" if passed else "fail", 
              f"HTTP {resp.status_code}")
    return passed

def test_detection_with_preprocessing(step_num, total):
    print_step(step_num, total, "Detection with Preprocessing", "running")
    time.sleep(0.3)
    
    img_b64 = image_to_base64(TEST_IMAGE)
    if not img_b64:
        print_step(step_num, total, "Detection with Preprocessing", "fail", "Image not found")
        return False
    
    resp = requests.post(f"{API_BASE}/api/detect", json={'image': img_b64}, timeout=30)
    data = resp.json()
    
    passed = (resp.status_code == 200 and 
              data.get('success') and 
              data.get('count', 0) > 0 and
              'preprocessing' in data)
    
    if passed:
        prep = data['preprocessing']
        details = f"faces={data['count']}, enhanced={prep['was_enhanced']}, method={prep['method']}"
        print_step(step_num, total, "Detection with Preprocessing", "pass", details)
    else:
        print_step(step_num, total, "Detection with Preprocessing", "fail", str(data.get('error', 'Failed')))
    
    return passed

def test_extraction_with_pose(step_num, total):
    print_step(step_num, total, "Extraction with Pose", "running")
    time.sleep(0.3)
    
    img_b64 = image_to_base64(TEST_IMAGE)
    
    requests.post(f"{API_BASE}/api/clear")
    requests.post(f"{API_BASE}/api/detect", json={'image': img_b64}, timeout=30)
    
    resp = requests.post(f"{API_BASE}/api/extract", json={}, timeout=30)
    data = resp.json()
    
    passed = (resp.status_code == 200 and 
              data.get('success') and 
              'pose' in data)
    
    if passed:
        pose = data['pose']
        details = f"yaw={pose['yaw']:.1f}, pitch={pose['pitch']:.1f}, cat={pose['pose_category']}"
        print_step(step_num, total, "Extraction with Pose", "pass", details)
    else:
        print_step(step_num, total, "Extraction with Pose", "fail", str(data.get('error', 'Failed')))
    
    return passed

def test_add_reference_with_pose(step_num, total):
    print_step(step_num, total, "Add Reference with Pose", "running")
    time.sleep(0.3)
    
    img_b64 = image_to_base64(TEST_IMAGE)
    requests.post(f"{API_BASE}/api/clear")
    
    resp = requests.post(f"{API_BASE}/api/add-reference", 
                         json={'image': img_b64, 'name': 'Test Person'}, 
                         timeout=30)
    data = resp.json()
    
    passed = (resp.status_code == 200 and 
              data.get('success') and 
              'pose' in data.get('reference', {}))
    
    if passed:
        ref = data['reference']
        pose = ref['pose']
        details = f"yaw={pose['yaw']:.1f}, category={ref['pose_category']}"
        print_step(step_num, total, "Add Reference with Pose", "pass", details)
    else:
        print_step(step_num, total, "Add Reference with Pose", "fail", str(data.get('error', 'Failed')))
    
    return passed

def test_multi_reference_enrollment(step_num, total):
    print_step(step_num, total, "Multi-Reference Enrollment", "running")
    time.sleep(0.3)
    
    img1 = image_to_base64(TEST_IMAGE)
    img2 = image_to_base64(TEST_IMAGE_2)
    
    requests.post(f"{API_BASE}/api/clear")
    
    resp1 = requests.post(f"{API_BASE}/api/add-reference", 
                          json={'image': img1, 'name': 'John Doe'}, 
                          timeout=30)
    resp2 = requests.post(f"{API_BASE}/api/add-reference", 
                          json={'image': img2, 'name': 'John Doe'}, 
                          timeout=30)
    
    refs_resp = requests.get(f"{API_BASE}/api/references")
    refs_data = refs_resp.json()
    
    passed = (resp1.status_code == 200 and 
              resp2.status_code == 200 and
              len(refs_data.get('references', [])) == 2)
    
    if passed:
        details = f"2 references with name 'John Doe'"
        print_step(step_num, total, "Multi-Reference Enrollment", "pass", details)
    else:
        print_step(step_num, total, "Multi-Reference Enrollment", "fail", "Failed to add multiple refs")
    
    return passed

def test_pose_aware_matching(step_num, total):
    print_step(step_num, total, "Pose-Aware Matching", "running")
    time.sleep(0.3)
    
    img_b64 = image_to_base64(TEST_IMAGE)
    
    requests.post(f"{API_BASE}/api/clear")
    requests.post(f"{API_BASE}/api/detect", json={'image': img_b64}, timeout=30)
    requests.post(f"{API_BASE}/api/extract", json={}, timeout=30)
    requests.post(f"{API_BASE}/api/add-reference", 
                  json={'image': img_b64, 'name': 'Same Person'}, 
                  timeout=30)
    
    resp = requests.post(f"{API_BASE}/api/compare", json={}, timeout=30)
    data = resp.json()
    
    passed = (resp.status_code == 200 and 
              data.get('success') and
              'query_pose' in data and
              'results' in data and
              len(data['results']) > 0)
    
    if passed:
        result = data['results'][0]
        details = f"raw={result['similarity']:.2f}, adjusted={result['adjusted_similarity']:.2f}, pose_match={result['pose_match']}"
        print_step(step_num, total, "Pose-Aware Matching", "pass", details)
    else:
        print_step(step_num, total, "Pose-Aware Matching", "fail", str(data.get('error', 'Failed')))
    
    return passed

def test_eyewear_detection(step_num, total):
    print_step(step_num, total, "Eyewear Detection", "running")
    time.sleep(0.3)
    
    img_b64 = image_to_base64(TEST_IMAGE)
    
    requests.post(f"{API_BASE}/api/clear")
    requests.post(f"{API_BASE}/api/detect", json={'image': img_b64}, timeout=30)
    
    resp = requests.get(f"{API_BASE}/api/eyewear")
    data = resp.json()
    
    passed = (resp.status_code == 200 and 
              data.get('success') and
              'eyewear' in data)
    
    if passed:
        ew = data['eyewear']
        details = f"type={ew['type']}, confidence={ew['confidence']:.2f}"
        print_step(step_num, total, "Eyewear Detection", "pass", details)
    else:
        print_step(step_num, total, "Eyewear Detection", "fail", str(data.get('error', 'Failed')))
    
    return passed

def test_visualization_endpoints(step_num, total):
    print_step(step_num, total, "Visualization Endpoints", "running")
    time.sleep(0.3)
    
    img_b64 = image_to_base64(TEST_IMAGE)
    
    requests.post(f"{API_BASE}/api/clear")
    requests.post(f"{API_BASE}/api/detect", json={'image': img_b64}, timeout=30)
    requests.post(f"{API_BASE}/api/extract", json={}, timeout=30)
    
    viz_types = ['detection', 'extraction', 'preprocessing', 'eyewear', 'embedding', 'similarity']
    results = []
    
    for vt in viz_types:
        resp = requests.get(f"{API_BASE}/api/visualizations/{vt}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            results.append(data.get('success') and bool(data.get('visualization')))
    
    passed = len(results) == len(viz_types) and all(results)
    
    if passed:
        details = f"{sum(results)}/{len(viz_types)} endpoints working"
        print_step(step_num, total, "Visualization Endpoints", "pass", details)
    else:
        print_step(step_num, total, "Visualization Endpoints", "fail", f"{sum(results)}/{len(viz_types)} passed")
    
    return passed

def test_clear_endpoint(step_num, total):
    print_step(step_num, total, "Clear Endpoint", "running")
    time.sleep(0.3)
    
    img_b64 = image_to_base64(TEST_IMAGE)
    
    requests.post(f"{API_BASE}/api/clear")
    requests.post(f"{API_BASE}/api/detect", json={'image': img_b64}, timeout=30)
    requests.post(f"{API_BASE}/api/add-reference", 
                  json={'image': img_b64, 'name': 'Test'}, 
                  timeout=30)
    
    resp = requests.post(f"{API_BASE}/api/clear")
    data = resp.json()
    
    refs_resp = requests.get(f"{API_BASE}/api/references")
    refs_data = refs_resp.json()
    
    passed = (resp.status_code == 200 and 
              data.get('success') and
              len(refs_data.get('references', [])) == 0)
    
    if passed:
        details = "All data cleared successfully"
        print_step(step_num, total, "Clear Endpoint", "pass", details)
    else:
        print_step(step_num, total, "Clear Endpoint", "fail", "Clear failed")
    
    return passed

# =============================================================================
# MAIN
# =============================================================================

def main():
    print_header()
    
    print(f"  {C['gray']}Connecting to API...{C['reset']}")
    if not wait_for_api():
        print(f"\n  {C['red']}ERROR: API not available{C['reset']}")
        print(f"  {C['gray']}Start with: python api_server.py{C['reset']}\n")
        return False
    
    print(f"  {C['green']}API connected{C['reset']} - {C['cyan']}http://localhost:3000{C['reset']}\n")
    
    tests = [
        ("Health Check", test_health),
        ("Detection with Preprocessing", test_detection_with_preprocessing),
        ("Extraction with Pose", test_extraction_with_pose),
        ("Add Reference with Pose", test_add_reference_with_pose),
        ("Multi-Reference Enrollment", test_multi_reference_enrollment),
        ("Pose-Aware Matching", test_pose_aware_matching),
        ("Eyewear Detection", test_eyewear_detection),
        ("Visualization Endpoints", test_visualization_endpoints),
        ("Clear Endpoint", test_clear_endpoint),
    ]
    
    total = len(tests)
    passed = 0
    failed = 0
    
    start_time = time.time()
    
    for i, (name, test_func) in enumerate(tests, 1):
        try:
            if test_func(i, total):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print_step(i, total, name, "fail", str(e))
            failed += 1
        time.sleep(0.2)
    
    total_time = time.time() - start_time
    
    print_summary(passed, failed, total_time)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
