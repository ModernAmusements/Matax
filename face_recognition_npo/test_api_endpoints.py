#!/usr/bin/env python3
"""
API Endpoint Verification Script
Tests all API endpoints for both FaceNet and ArcFace models.
"""

import sys
import os
import json
import requests
import time

API_BASE = "http://localhost:3000"

def print_header(text):
    print(f"\n{'='*60}")
    print(f"{text}")
    print(f"{'='*60}")

def print_result(name, passed, error=None):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {name}")
    if error:
        print(f"   Error: {error}")

def test_endpoint(name, method, path, data=None, expected_keys=None):
    """Test a single API endpoint."""
    url = f"{API_BASE}{path}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            return False, f"Unknown method: {method}"

        if response.status_code == 200:
            try:
                result = response.json()
                if expected_keys:
                    for key in expected_keys:
                        if key not in result:
                            return False, f"Missing key: {key}"
                return True, result
            except:
                return True, response.text[:100]
        else:
            return False, f"Status {response.status_code}"
    except Exception as e:
        return False, str(e)

def test_all_endpoints(use_arcface=False):
    """Test all API endpoints."""
    model_name = "ArcFace" if use_arcface else "FaceNet"
    dim = "512-dim" if use_arcface else "128-dim"
    
    print_header(f"TESTING {model_name} ({dim})")
    
    # Set environment
    env = os.environ.copy()
    if use_arcface:
        env["USE_ARCFACE"] = "true"
    
    results = []
    
    # Test 1: Health check
    passed, result = test_endpoint("GET /api/health", "GET", "/api/health")
    results.append(("GET /api/health", passed))
    
    # Test 2: Embedding info
    passed, result = test_endpoint("GET /api/embedding-info", "GET", "/api/embedding-info")
    if passed:
        if use_arcface:
            passed = result.get("model") == "ArcFaceEmbeddingExtractor"
        else:
            passed = result.get("model") == "FaceNetEmbeddingExtractor"
    results.append(("GET /api/embedding-info", passed))
    
    # Test 3: Status
    passed, result = test_endpoint("GET /api/status", "GET", "/api/status")
    results.append(("GET /api/status", passed))
    
    # Test 4: References (empty initially)
    passed, result = test_endpoint("GET /api/references", "GET", "/api/references")
    results.append(("GET /api/references", passed))
    
    # Test 5: Visualizations (need active session)
    viz_types = [
        "detection", "extraction", "landmarks", "mesh3d",
        "alignment", "saliency", "activations", "features",
        "multiscale", "confidence", "embedding", "similarity",
        "robustness", "biometric"
    ]
    
    print(f"\n  Testing {len(viz_types)} visualization endpoints...")
    for viz_type in viz_types:
        passed, result = test_endpoint(
            f"GET /api/visualizations/{viz_type}",
            "GET",
            f"/api/visualizations/{viz_type}"
        )
        results.append((f"GET /api/visualizations/{viz_type}", passed))
    
    # Summary
    print_header(f"RESULTS SUMMARY - {model_name}")
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    for name, passed in results:
        print_result(name, passed)
    
    print(f"\n  {passed_count}/{total_count} tests passed")
    
    return passed_count == total_count

def test_full_workflow(use_arcface=False):
    """Test the complete user workflow."""
    model_name = "ArcFace" if use_arcface else "FaceNet"
    
    print_header(f"WORKFLOW TEST - {model_name}")
    
    # This test requires actual image files and would need to be run manually
    # since it requires user interaction or pre-loaded images
    
    print("  Note: Full workflow test requires:")
    print("  - Image files in test_images/")
    print("  - Active face detection session")
    print("  - Reference images to compare")
    print("\n  Skipping interactive workflow tests.")
    print("  Run 'python test_e2e_pipeline.py' for full workflow tests.")
    
    return True

def main():
    """Main verification function."""
    print("\n" + "="*60)
    print("API ENDPOINT VERIFICATION")
    print("="*60)
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code != 200:
            print("✗ Server is not responding correctly")
            print("  Start the server first: ./start.sh")
            sys.exit(1)
    except:
        print("✗ Server is not running")
        print("  Start the server first: ./start.sh")
        sys.exit(1)
    
    all_passed = True
    
    # Test FaceNet (default)
    if not test_all_endpoints(use_arcface=False):
        all_passed = False
    
    # Test ArcFace
    if not test_all_endpoints(use_arcface=True):
        all_passed = False
    
    # Print final summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL API ENDPOINTS WORKING")
        print("="*60)
        print("\nBoth FaceNet and ArcFace are fully functional.")
    else:
        print("✗ SOME ENDPOINTS FAILED")
        print("="*60)
        print("\nCheck the output above for failed tests.")
        sys.exit(1)

if __name__ == "__main__":
    main()
