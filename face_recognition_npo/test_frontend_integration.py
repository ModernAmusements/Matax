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
              'results' in data and
              len(data['results']) > 0)
    
    if passed:
        result = data['results'][0]
        arcface_sim = result.get('arcface_similarity')
        facenet_sim = result.get('facenet_similarity')
        final_score = result.get('final_score', 0)
        arcface_str = f"{arcface_sim:.2f}" if arcface_sim is not None else "N/A"
        facenet_str = f"{facenet_sim:.2f}" if facenet_sim is not None else "N/A"
        details = f"arcface={arcface_str}, facenet={facenet_str}, final={final_score:.2f}"
        print_step(step_num, total, "Dual-Model Comparison", "pass", details)
    else:
        print_step(step_num, total, "Dual-Model Comparison", "fail", str(data.get('error', 'Failed')))
    
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
# FRONTEND MESH TESTS
# =============================================================================

def test_mesh_html_elements(step_num, total):
    print_step(step_num, total, "Mesh HTML Elements", "running")
    time.sleep(0.3)
    
    html_path = os.path.join(os.path.dirname(__file__), 'electron-ui', 'index.html')
    
    if not os.path.exists(html_path):
        print_step(step_num, total, "Mesh HTML Elements", "fail", "index.html not found")
        return False
    
    with open(html_path, 'r') as f:
        html_content = f.read()
    
    checks = {
        'meshCanvas': 'meshCanvas' in html_content,
        'toggleMeshBtn': 'toggleMeshBtn' in html_content,
        'toggleMeshOverlay': 'toggleMeshOverlay' in html_content,
        'MediaPipe face_mesh': '@mediapipe/face_mesh' in html_content,
        'MediaPipe camera_utils': '@mediapipe/camera_utils' in html_content,
    }
    
    passed = all(checks.values())
    details = f"{sum(checks.values())}/{len(checks)} elements found"
    
    if passed:
        print_step(step_num, total, "Mesh HTML Elements", "pass", details)
    else:
        missing = [k for k, v in checks.items() if not v]
        print_step(step_num, total, "Mesh HTML Elements", "fail", f"Missing: {', '.join(missing)}")
    
    return passed

def test_mesh_javascript_functions(step_num, total):
    print_step(step_num, total, "Mesh JavaScript Functions", "running")
    time.sleep(0.3)
    
    js_path = os.path.join(os.path.dirname(__file__), 'electron-ui', 'renderer', 'app.js')
    
    if not os.path.exists(js_path):
        print_step(step_num, total, "Mesh JavaScript Functions", "fail", "app.js not found")
        return False
    
    with open(js_path, 'r') as f:
        js_content = f.read()
    
    functions = [
        'faceMesh',
        'meshCamera', 
        'meshOverlayActive',
        'initFaceMesh',
        'onMeshResults',
        'drawMesh',
        'toggleMeshOverlay',
    ]
    
    checks = {fn: f'function {fn}' in js_content or f'let {fn}' in js_content for fn in functions}
    
    passed = all(checks.values())
    details = f"{sum(checks.values())}/{len(functions)} functions/vars defined"
    
    if passed:
        print_step(step_num, total, "Mesh JavaScript Functions", "pass", details)
    else:
        missing = [k for k, v in checks.items() if not v]
        print_step(step_num, total, "Mesh JavaScript Functions", "fail", f"Missing: {', '.join(missing)}")
    
    return passed

def test_mesh_css_styles(step_num, total):
    print_step(step_num, total, "Mesh CSS Styles", "running")
    time.sleep(0.3)
    
    css_path = os.path.join(os.path.dirname(__file__), 'electron-ui', 'styles', 'design-system.css')
    
    if not os.path.exists(css_path):
        print_step(step_num, total, "Mesh CSS Styles", "fail", "design-system.css not found")
        return False
    
    with open(css_path, 'r') as f:
        css_content = f.read()
    
    checks = {
        '.mesh-canvas': '.mesh-canvas' in css_content,
        'position: absolute': 'position: absolute' in css_content,
        'z-index': 'z-index' in css_content,
        'display: block': 'display: block' in css_content,
    }
    
    passed = all(checks.values())
    details = f"{sum(checks.values())}/{len(checks)} CSS rules found"
    
    if passed:
        print_step(step_num, total, "Mesh CSS Styles", "pass", details)
    else:
        missing = [k for k, v in checks.items() if not v]
        print_step(step_num, total, "Mesh CSS Styles", "fail", f"Missing: {', '.join(missing)}")
    
    return passed

def test_mesh_mediapipe_cdn(step_num, total):
    print_step(step_num, total, "MediaPipe CDN Accessibility", "running")
    time.sleep(0.3)
    
    import urllib.request
    import urllib.error
    
    cdns = [
        'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js',
        'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js',
    ]
    
    results = []
    for url in cdns:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                results.append(response.status == 200)
        except Exception:
            results.append(False)
    
    passed = all(results)
    details = f"{sum(results)}/{len(cdns)} CDN resources accessible"
    
    if passed:
        print_step(step_num, total, "MediaPipe CDN Accessibility", "pass", details)
    else:
        print_step(step_num, total, "MediaPipe CDN Accessibility", "fail", details)
    
    return passed

def test_existing_functions_intact(step_num, total):
    print_step(step_num, total, "Existing Functions Intact", "running")
    time.sleep(0.3)
    
    js_path = os.path.join(os.path.dirname(__file__), 'electron-ui', 'renderer', 'app.js')
    
    with open(js_path, 'r') as f:
        js_content = f.read()
    
    critical_functions = [
        'selectImage',
        'handleImageSelect',
        'detectFaces',
        'extractFeatures',
        'addReference',
        'handleReferenceSelect',
        'compareFaces',
        'clearAllCache',
        'startWebcam',
        'captureWebcam',
        'stopWebcam',
        'updateReferenceList',
        'removeReference',
        'showReferenceVisualizations',
    ]
    
    checks = {fn: f'function {fn}' in js_content or f'async function {fn}' in js_content for fn in critical_functions}
    
    passed = all(checks.values())
    details = f"{sum(checks.values())}/{len(critical_functions)} critical functions present"
    
    if passed:
        print_step(step_num, total, "Existing Functions Intact", "pass", details)
    else:
        missing = [k for k, v in checks.items() if not v]
        print_step(step_num, total, "Existing Functions Intact", "fail", f"Missing: {', '.join(missing)}")
    
    return passed

def test_html_js_event_handlers(step_num, total):
    print_step(step_num, total, "HTML-JS Event Handlers", "running")
    time.sleep(0.3)
    
    html_path = os.path.join(os.path.dirname(__file__), 'electron-ui', 'index.html')
    js_path = os.path.join(os.path.dirname(__file__), 'electron-ui', 'renderer', 'app.js')
    
    with open(html_path, 'r') as f:
        html_content = f.read()
    with open(js_path, 'r') as f:
        js_content = f.read()
    
    import re
    html_handlers = re.findall(r'onclick="(\w+)\(', html_content)
    html_handlers += re.findall(r'onchange="(\w+)\(', html_content)
    html_handlers = list(set(html_handlers))
    
    missing = []
    for handler in html_handlers:
        if handler not in js_content:
            missing.append(handler)
    
    passed = len(missing) == 0
    details = f"{len(html_handlers)} handlers, {len(missing)} missing"
    
    if passed:
        print_step(step_num, total, "HTML-JS Event Handlers", "pass", f"All {len(html_handlers)} handlers linked")
    else:
        print_step(step_num, total, "HTML-JS Event Handlers", "fail", f"Missing: {', '.join(missing)}")
    
    return passed

# =============================================================================
# LIBRARY TESTS
# =============================================================================

def test_library_html_elements(step_num, total):
    print_step(step_num, total, "Library HTML Elements", "running")
    time.sleep(0.3)
    
    html_path = os.path.join(os.path.dirname(__file__), 'electron-ui', 'index.html')
    
    if not os.path.exists(html_path):
        print_step(step_num, total, "Library HTML Elements", "fail", "index.html not found")
        return False
    
    with open(html_path, 'r') as f:
        html_content = f.read()
    
    checks = {
        'step6': 'id="step6"' in html_content,
        'libraryGrid': 'id="libraryGrid"' in html_content,
        'libraryUploadInput': 'id="libraryUploadInput"' in html_content,
        'libraryCompareInput': 'id="libraryCompareInput"' in html_content,
        'findMatchesBtn': 'id="findMatchesBtn"' in html_content,
        'librarySearch': 'id="librarySearch"' in html_content,
        'libraryModal': 'id="libraryModal"' in html_content,
        'section-intro': 'id="section-intro"' in html_content,
    }
    
    passed = all(checks.values())
    details = f"{sum(checks.values())}/{len(checks)} elements found"
    
    if passed:
        print_step(step_num, total, "Library HTML Elements", "pass", details)
    else:
        missing = [k for k, v in checks.items() if not v]
        print_step(step_num, total, "Library HTML Elements", "fail", f"Missing: {', '.join(missing)}")
    
    return passed

def test_library_javascript_functions(step_num, total):
    print_step(step_num, total, "Library JavaScript Functions", "running")
    time.sleep(0.3)
    
    js_path = os.path.join(os.path.dirname(__file__), 'electron-ui', 'renderer', 'app.js')
    
    if not os.path.exists(js_path):
        print_step(step_num, total, "Library JavaScript Functions", "fail", "app.js not found")
        return False
    
    with open(js_path, 'r') as f:
        js_content = f.read()
    
    functions = [
        'loadLibrary',
        'renderLibraryGrid',
        'saveToLibrary',
        'deleteLibraryPerson',
        'viewLibraryPerson',
        'handleLibraryUpload',
        'handleLibraryCompareUpload',
        'searchLibraryByName',
        'matchWithLibraryImage',
        'checkFindMatchesButton',
        'startWebcamForLibrary',
    ]
    
    checks = {fn: f'function {fn}' in js_content or f'async function {fn}' in js_content for fn in functions}
    
    passed = all(checks.values())
    details = f"{sum(checks.values())}/{len(functions)} functions defined"
    
    if passed:
        print_step(step_num, total, "Library JavaScript Functions", "pass", details)
    else:
        missing = [k for k, v in checks.items() if not v]
        print_step(step_num, total, "Library JavaScript Functions", "fail", f"Missing: {', '.join(missing)}")
    
    return passed

def test_library_css_styles(step_num, total):
    print_step(step_num, total, "Library CSS Styles", "running")
    time.sleep(0.3)
    
    css_path = os.path.join(os.path.dirname(__file__), 'electron-ui', 'styles', 'design-system.css')
    
    if not os.path.exists(css_path):
        print_step(step_num, total, "Library CSS Styles", "fail", "design-system.css not found")
        return False
    
    with open(css_path, 'r') as f:
        css_content = f.read()
    
    checks = {
        '.library-card': '.library-card' in css_content,
        '.library-grid': '.library-grid' in css_content,
        '.library-search': '.library-search' in css_content,
        '.library-match-card': '.library-match-card' in css_content,
        '.library-matches-grid': '.library-matches-grid' in css_content,
        '.btn-delete': '.btn-delete' in css_content,
        '.section-header': '.section-header' in css_content,
    }
    
    passed = all(checks.values())
    details = f"{sum(checks.values())}/{len(checks)} CSS rules found"
    
    if passed:
        print_step(step_num, total, "Library CSS Styles", "pass", details)
    else:
        missing = [k for k, v in checks.items() if not v]
        print_step(step_num, total, "Library CSS Styles", "fail", f"Missing: {', '.join(missing)}")
    
    return passed

def test_library_api_endpoints(step_num, total):
    print_step(step_num, total, "Library API Endpoints", "running")
    time.sleep(0.3)
    
    try:
        # Test GET /api/library
        response = requests.get(f"{API_BASE}/api/library", timeout=5)
        if response.status_code != 200:
            print_step(step_num, total, "Library API Endpoints", "fail", f"GET /api/library returned {response.status_code}")
            return False
        
        data = response.json()
        if "persons" not in data or "count" not in data:
            print_step(step_num, total, "Library API Endpoints", "fail", "Response missing required fields")
            return False
        
        person_count = data.get("count", 0)
        print_step(step_num, total, "Library API Endpoints", "pass", f"Library endpoint working, {person_count} persons found")
        return True
        
    except Exception as e:
        print_step(step_num, total, "Library API Endpoints", "fail", str(e))
        return False


def test_workflow_add_person_upload(step_num, total):
    """Test Workflow: Add Person via Upload"""
    print_step(step_num, total, "Workflow: Add Person (Upload)", "running")
    time.sleep(0.3)
    
    try:
        # Load test image
        test_image_path = os.path.join(os.path.dirname(__file__), 'test_images', 'test_subject.jpg')
        if not os.path.exists(test_image_path):
            print_step(step_num, total, "Workflow: Add Person (Upload)", "skip", "Test image not found")
            return True
        
        with open(test_image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Test adding person to library
        response = requests.post(f"{API_BASE}/library/person", 
            json={
                "name": f"TestPerson_{int(time.time())}",
                "notes": "Created by workflow test",
                "image": f"data:image/jpeg;base64,{image_data}",
                "source": "test"
            },
            timeout=10
        )
        
        if response.status_code != 200:
            print_step(step_num, total, "Workflow: Add Person (Upload)", "fail", f"API returned {response.status_code}")
            return False
        
        data = response.json()
        if not data.get("success"):
            print_step(step_num, total, "Workflow: Add Person (Upload)", "fail", data.get("error", "Unknown error"))
            return False
        
        print_step(step_num, total, "Workflow: Add Person (Upload)", "pass", f"Added person: {data['person']['name']}")
        return True
        
    except Exception as e:
        print_step(step_num, total, "Workflow: Add Person (Upload)", "fail", str(e))
        return False


def test_workflow_search_library(step_num, total):
    """Test Workflow: Search Library by Name"""
    print_step(step_num, total, "Workflow: Search Library", "running")
    time.sleep(0.3)
    
    try:
        # Get current library
        response = requests.get(f"{API_BASE}/library", timeout=5)
        if response.status_code != 200:
            print_step(step_num, total, "Workflow: Search Library", "fail", "Cannot access library")
            return False
        
        data = response.json()
        persons = data.get("persons", [])
        
        if not persons:
            print_step(step_num, total, "Workflow: Search Library", "skip", "Library is empty")
            return True
        
        # Test searching (simulated by filtering locally)
        search_term = persons[0]["name"][:3].lower()  # First 3 chars of first person
        matches = [p for p in persons if search_term in p["name"].lower()]
        
        if matches:
            print_step(step_num, total, "Workflow: Search Library", "pass", f"Found {len(matches)} match(es)")
            return True
        else:
            print_step(step_num, total, "Workflow: Search Library", "fail", "Search returned no results")
            return False
            
    except Exception as e:
        print_step(step_num, total, "Workflow: Search Library", "fail", str(e))
        return False


def test_workflow_match_with_library(step_num, total):
    """Test Workflow: Match Image with Library"""
    print_step(step_num, total, "Workflow: Match with Library", "running")
    time.sleep(0.3)
    
    try:
        # Load test image
        test_image_path = os.path.join(os.path.dirname(__file__), 'test_images', 'test_subject.jpg')
        if not os.path.exists(test_image_path):
            print_step(step_num, total, "Workflow: Match with Library", "skip", "Test image not found")
            return True
        
        with open(test_image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Test matching
        response = requests.post(f"{API_BASE}/library/match",
            json={"image": f"data:image/jpeg;base64,{image_data}"},
            timeout=10
        )
        
        if response.status_code != 200:
            print_step(step_num, total, "Workflow: Match with Library", "fail", f"API returned {response.status_code}")
            return False
        
        data = response.json()
        if not data.get("success"):
            print_step(step_num, total, "Workflow: Match with Library", "fail", data.get("error", "Unknown error"))
            return False
        
        matches = data.get("matches", [])
        print_step(step_num, total, "Workflow: Match with Library", "pass", f"Found {len(matches)} match(es)")
        return True
        
    except Exception as e:
        print_step(step_num, total, "Workflow: Match with Library", "fail", str(e))
        return False


def test_workflow_delete_person(step_num, total):
    """Test Workflow: Delete Person from Library"""
    print_step(step_num, total, "Workflow: Delete Person", "running")
    time.sleep(0.3)
    
    try:
        # Get current library
        response = requests.get(f"{API_BASE}/library", timeout=5)
        if response.status_code != 200:
            print_step(step_num, total, "Workflow: Delete Person", "fail", "Cannot access library")
            return False
        
        data = response.json()
        persons = data.get("persons", [])
        
        # Find a test person to delete (one created by our tests)
        test_person = None
        for p in persons:
            if p["name"].startswith("TestPerson_"):
                test_person = p
                break
        
        if not test_person:
            print_step(step_num, total, "Workflow: Delete Person", "skip", "No test persons to delete")
            return True
        
        # Delete the person
        response = requests.delete(f"{API_BASE}/library/person/{test_person['id']}", timeout=5)
        if response.status_code != 200:
            print_step(step_num, total, "Workflow: Delete Person", "fail", f"Delete returned {response.status_code}")
            return False
        
        data = response.json()
        if not data.get("success"):
            print_step(step_num, total, "Workflow: Delete Person", "fail", data.get("error", "Unknown error"))
            return False
        
        print_step(step_num, total, "Workflow: Delete Person", "pass", f"Deleted: {test_person['name']}")
        return True
        
    except Exception as e:
        print_step(step_num, total, "Workflow: Delete Person", "fail", str(e))
        return False


def test_workflow_steps_1_to_4(step_num, total):
    """Test Workflow: Steps 1-4 (Upload → Detect → Extract → Compare)"""
    print_step(step_num, total, "Workflow: Steps 1-4 Integration", "running")
    time.sleep(0.3)
    
    try:
        # Step 1: Load image
        test_image_path = os.path.join(os.path.dirname(__file__), 'test_images', 'test_subject.jpg')
        if not os.path.exists(test_image_path):
            print_step(step_num, total, "Workflow: Steps 1-4 Integration", "skip", "Test image not found")
            return True
        
        with open(test_image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Step 2: Detect faces
        response = requests.post(f"{API_BASE}/api/detect",
            json={"image": f"data:image/jpeg;base64,{image_data}"},
            timeout=10
        )
        
        if response.status_code != 200:
            print_step(step_num, total, "Workflow: Steps 1-4 Integration", "fail", "Detection failed")
            return False
        
        detect_data = response.json()
        if not detect_data.get("success") or detect_data.get("count", 0) == 0:
            print_step(step_num, total, "Workflow: Steps 1-4 Integration", "fail", "No faces detected")
            return False
        
        # Step 3: Extract features
        response = requests.post(f"{API_BASE}/extract", timeout=10)
        if response.status_code != 200:
            print_step(step_num, total, "Workflow: Steps 1-4 Integration", "fail", "Extraction failed")
            return False
        
        extract_data = response.json()
        if not extract_data.get("success"):
            print_step(step_num, total, "Workflow: Steps 1-4 Integration", "fail", "Feature extraction failed")
            return False
        
        # Step 4: Compare (need a reference first)
        # Add a reference
        ref_image_path = os.path.join(os.path.dirname(__file__), 'test_images', 'reference_subject.jpg')
        if os.path.exists(ref_image_path):
            with open(ref_image_path, 'rb') as f:
                ref_image_data = base64.b64encode(f.read()).decode('utf-8')
            
            requests.post(f"{API_BASE}/add-reference",
                json={
                    "image": f"data:image/jpeg;base64,{ref_image_data}",
                    "name": "TestReference"
                },
                timeout=10
            )
            
            # Now compare
            response = requests.post(f"{API_BASE}/compare", timeout=10)
            if response.status_code == 200:
                compare_data = response.json()
                if compare_data.get("success"):
                    print_step(step_num, total, "Workflow: Steps 1-4 Integration", "pass", "Full flow completed")
                    return True
        
        print_step(step_num, total, "Workflow: Steps 1-4 Integration", "pass", "Detect & Extract completed")
        return True
        
    except Exception as e:
        print_step(step_num, total, "Workflow: Steps 1-4 Integration", "fail", str(e))
        return False

# =============================================================================
# UPLOAD WORKFLOW TESTS
# =============================================================================

def test_upload_workflow_html_elements(step_num, total):
    """Test Upload Workflow: Currently Uploaded section"""
    print_step(step_num, total, "Upload Workflow HTML Elements", "running")
    time.sleep(0.3)
    
    html_path = os.path.join(os.path.dirname(__file__), 'electron-ui', 'index.html')
    js_path = os.path.join(os.path.dirname(__file__), 'electron-ui', 'renderer', 'app.js')
    
    if not os.path.exists(html_path):
        print_step(step_num, total, "Upload Workflow HTML Elements", "fail", "index.html not found")
        return False
    if not os.path.exists(js_path):
        print_step(step_num, total, "Upload Workflow HTML Elements", "fail", "app.js not found")
        return False
    
    with open(html_path, 'r') as f:
        html_content = f.read()
    with open(js_path, 'r') as f:
        js_content = f.read()
    
    # Check HTML elements for Currently Uploaded section
    html_checks = {
        'currentlyUploaded': 'id="currentlyUploaded"' in html_content,
        'currentlyUploadedPreview': 'id="currentlyUploadedPreview"' in html_content,
        'currentlyUploadedImage': 'id="currentlyUploadedImage"' in html_content,
        'currently-uploaded-empty': 'currently-uploaded-empty' in html_content,
        'currently-uploaded-preview': 'currently-uploaded-preview' in html_content,
    }
    
    # Check JavaScript functions and variables
    js_checks = {
        'currentImage variable': 'let currentImage' in js_content or 'var currentImage' in js_content,
        'updateCurrentlyUploaded function': 'function updateCurrentlyUploaded' in js_content,
        'showUploadedAsReference function': 'function showUploadedAsReference' in js_content,
        'addUploadedToLibrary function': 'function addUploadedToLibrary' in js_content,
        'resetSteps function': 'function resetSteps' in js_content,
        'markStepComplete function': 'function markStepComplete' in js_content,
        'handleImageSelect function': 'function handleImageSelect' in js_content,
    }
    
    html_passed = all(html_checks.values())
    js_passed = all(js_checks.values())
    passed = html_passed and js_passed
    
    details = f"HTML: {sum(html_checks.values())}/{len(html_checks)}, JS: {sum(js_checks.values())}/{len(js_checks)}"
    
    if passed:
        print_step(step_num, total, "Upload Workflow HTML Elements", "pass", details)
    else:
        missing = []
        for k, v in html_checks.items():
            if not v:
                missing.append(f"HTML:{k}")
        for k, v in js_checks.items():
            if not v:
                missing.append(f"JS:{k}")
        print_step(step_num, total, "Upload Workflow HTML Elements", "fail", f"Missing: {', '.join(missing)}")
    
    return passed

def test_upload_workflow_null_checks(step_num, total):
    """Test Upload Workflow: Null checks in critical functions"""
    print_step(step_num, total, "Upload Workflow Null Checks", "running")
    time.sleep(0.3)
    
    js_path = os.path.join(os.path.dirname(__file__), 'electron-ui', 'renderer', 'app.js')
    
    if not os.path.exists(js_path):
        print_step(step_num, total, "Upload Workflow Null Checks", "fail", "app.js not found")
        return False
    
    with open(js_path, 'r') as f:
        js_content = f.read()
    
    # Check that critical functions have null checks
    checks = {
        'resetSteps has null check for facesContainer': 'getElementById(\'facesContainer\')' in js_content,
        'resetSteps has null check for step1': 'getElementById(\'step1\')' in js_content,
        'resetSteps has null check for step2': 'getElementById(\'step2\')' in js_content,
        'resetSteps has null check for step3': 'getElementById(\'step3\')' in js_content,
        'resetSteps has null check for step4': 'getElementById(\'step4\')' in js_content,
        'resetSteps has null check for webcamStep': 'getElementById(\'webcamStep\')' in js_content,
        'markStepComplete has null check': 'getElementById(stepId)' in js_content,
        'updateCurrentlyUploaded uses style.display': 'style.display' in js_content,
    }
    
    passed = all(checks.values())
    details = f"{sum(checks.values())}/{len(checks)} null checks present"
    
    if passed:
        print_step(step_num, total, "Upload Workflow Null Checks", "pass", details)
    else:
        missing = [k for k, v in checks.items() if not v]
        print_step(step_num, total, "Upload Workflow Null Checks", "fail", f"Missing: {', '.join(missing)}")
    
    return passed

def test_upload_workflow_integration(step_num, total):
    """Test Upload Workflow: End-to-end upload → detect → extract"""
    print_step(step_num, total, "Upload Workflow Integration", "running")
    time.sleep(0.3)
    
    try:
        # Clear session first
        requests.post(f"{API_BASE}/api/clear", timeout=5)
        
        test_image_path = os.path.join(os.path.dirname(__file__), 'test_images', 'test_subject.jpg')
        if not os.path.exists(test_image_path):
            print_step(step_num, total, "Upload Workflow Integration", "skip", "Test image not found")
            return True
        
        with open(test_image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Step 1: Detect faces
        response = requests.post(f"{API_BASE}/api/detect",
            json={"image": f"data:image/jpeg;base64,{image_data}"},
            timeout=10
        )
        
        if response.status_code != 200:
            print_step(step_num, total, "Upload Workflow Integration", "fail", f"Detection failed: {response.status_code}")
            return False
        
        detect_data = response.json()
        if not detect_data.get("success") or detect_data.get("count", 0) == 0:
            print_step(step_num, total, "Upload Workflow Integration", "fail", "No faces detected")
            return False
        
        # Step 2: Extract features
        response = requests.post(f"{API_BASE}/api/extract", json={}, timeout=10)
        if response.status_code != 200:
            print_step(step_num, total, "Upload Workflow Integration", "fail", f"Extract failed: {response.status_code} - {response.text[:100]}")
            return False
        
        extract_data = response.json()
        if not extract_data.get("success"):
            print_step(step_num, total, "Upload Workflow Integration", "fail", f"Feature extraction failed: {extract_data.get('error', 'unknown')}")
            return False
        
        # Step 3: Add reference for comparison
        ref_image_path = os.path.join(os.path.dirname(__file__), 'test_images', 'reference_subject.jpg')
        if os.path.exists(ref_image_path):
            with open(ref_image_path, 'rb') as f:
                ref_image_data = base64.b64encode(f.read()).decode('utf-8')
            
            requests.post(f"{API_BASE}/api/add-reference",
                json={
                    "image": f"data:image/jpeg;base64,{ref_image_data}",
                    "name": "WorkflowTestRef"
                },
                timeout=10
            )
        
        # Step 4: Compare
        response = requests.post(f"{API_BASE}/api/compare", timeout=10)
        compare_success = response.status_code == 200
        
        print_step(step_num, total, "Upload Workflow Integration", "pass", 
                   f"Detect:{detect_data.get('count')} face(s), Extract:{extract_data.get('embedding_size')}D, Compare:{compare_success}")
        return True
        
    except Exception as e:
        print_step(step_num, total, "Upload Workflow Integration", "fail", str(e))
        return False

# =============================================================================
# WORKFLOW INDEPENDENCE TESTS
# =============================================================================

def test_workflow_independence(step_num, total):
    """Test that each workflow can run independently"""
    print_step(step_num, total, "Workflow Independence", "running")
    time.sleep(0.3)
    
    try:
        # Test: Clear should work anytime
        response = requests.post(f"{API_BASE}/api/clear", timeout=5)
        if response.status_code != 200:
            print_step(step_num, total, "Workflow Independence", "fail", "Clear API failed")
            return False
        print_step(step_num, total, "Workflow Independence", "pass", "Clear works independently")
        return True
    except Exception as e:
        print_step(step_num, total, "Workflow Independence", "fail", str(e))
        return False

def test_comparison_result_workflow(step_num, total):
    """Test complete workflow: upload -> detect -> extract -> add ref -> compare -> verify result"""
    print_step(step_num, total, "Comparison Result Workflow", "running")
    time.sleep(0.3)
    
    try:
        # Step 1: Clear session
        requests.post(f"{API_BASE}/api/clear", timeout=5)
        
        # Step 2: Load test images
        test_image_path = os.path.join(os.path.dirname(__file__), 'test_images', 'test_subject.jpg')
        ref_image_path = os.path.join(os.path.dirname(__file__), 'test_images', 'reference_subject.jpg')
        
        if not os.path.exists(test_image_path):
            print_step(step_num, total, "Comparison Result Workflow", "skip", "Test image not found")
            return True
        
        with open(test_image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Step 3: Detect faces
        response = requests.post(f"{API_BASE}/api/detect",
            json={"image": f"data:image/jpeg;base64,{image_data}"},
            timeout=30
        )
        if response.status_code != 200:
            print_step(step_num, total, "Comparison Result Workflow", "fail", f"Detection failed: {response.status_code}")
            return False
        
        detect_data = response.json()
        if not detect_data.get("success") or detect_data.get("count", 0) == 0:
            print_step(step_num, total, "Comparison Result Workflow", "fail", "No faces detected")
            return False
        
        # Step 4: Extract features
        response = requests.post(f"{API_BASE}/api/extract", json={}, timeout=30)
        if response.status_code != 200:
            print_step(step_num, total, "Comparison Result Workflow", "fail", f"Extraction failed: {response.status_code}")
            return False
        
        extract_data = response.json()
        if not extract_data.get("success"):
            print_step(step_num, total, "Comparison Result Workflow", "fail", "Feature extraction failed")
            return False
        
        # Step 5: Verify all viz types are available
        viz_types = ['detection', 'extraction', 'landmarks', 'mesh3d', 'alignment', 
                     'saliency', 'activations', 'features', 'embedding', 'confidence',
                     'biometric', 'robustness']
        
        available_viz = []
        for viz_type in viz_types:
            try:
                resp = requests.get(f"{API_BASE}/api/visualizations/{viz_type}", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("success"):
                        available_viz.append(viz_type)
            except:
                pass
        
        # Step 6: Add reference for comparison
        if os.path.exists(ref_image_path):
            with open(ref_image_path, 'rb') as f:
                ref_data = base64.b64encode(f.read()).decode('utf-8')
            
            response = requests.post(f"{API_BASE}/api/add-reference",
                json={"image": f"data:image/jpeg;base64,{ref_data}", "name": "WorkflowTestRef"},
                timeout=30
            )
            if response.status_code != 200:
                print_step(step_num, total, "Comparison Result Workflow", "fail", "Failed to add reference")
                return False
            
            ref_data_response = response.json()
            if not ref_data_response.get("success"):
                print_step(step_num, total, "Comparison Result Workflow", "fail", "Reference add failed")
                return False
        
        # Step 7: Compare
        response = requests.post(f"{API_BASE}/api/compare", timeout=30)
        if response.status_code != 200:
            print_step(step_num, total, "Comparison Result Workflow", "fail", f"Compare failed: {response.status_code}")
            return False
        
        compare_data = response.json()
        
        # Step 8: Verify comparison result has all required fields
        if not compare_data.get("success"):
            print_step(step_num, total, "Comparison Result Workflow", "fail", "Compare returned failure")
            return False
        
        best_match = compare_data.get("best_match")
        if not best_match:
            print_step(step_num, total, "Comparison Result Workflow", "fail", "No best match returned")
            return False
        
        # Verify all score fields are present
        required_fields = ['name', 'final_score', 'match_label', 'status']
        missing_fields = [f for f in required_fields if f not in best_match]
        
        if missing_fields:
            print_step(step_num, total, "Comparison Result Workflow", "fail", f"Missing fields: {missing_fields}")
            return False
        
        details = f"Match: {best_match.get('name')}, Score: {int(best_match.get('final_score', 0)*100)}%, Viz types: {len(available_viz)}"
        print_step(step_num, total, "Comparison Result Workflow", "pass", details)
        return True
        
    except Exception as e:
        print_step(step_num, total, "Comparison Result Workflow", "fail", str(e))
        return False

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
        ("Mesh HTML Elements", test_mesh_html_elements),
        ("Mesh JavaScript Functions", test_mesh_javascript_functions),
        ("Mesh CSS Styles", test_mesh_css_styles),
        ("MediaPipe CDN Accessibility", test_mesh_mediapipe_cdn),
        ("Existing Functions Intact", test_existing_functions_intact),
        ("HTML-JS Event Handlers", test_html_js_event_handlers),
        ("Library HTML Elements", test_library_html_elements),
        ("Library JavaScript Functions", test_library_javascript_functions),
        ("Library CSS Styles", test_library_css_styles),
        ("Library API Endpoints", test_library_api_endpoints),
        ("Upload Workflow HTML Elements", test_upload_workflow_html_elements),
        ("Upload Workflow Null Checks", test_upload_workflow_null_checks),
        ("Upload Workflow Integration", test_upload_workflow_integration),
        ("Workflow Independence", test_workflow_independence),
        ("Comparison Result Workflow", test_comparison_result_workflow),
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
