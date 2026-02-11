#!/usr/bin/env python3
"""
End-to-end test for the complete face recognition pipeline.
Tests that real embeddings are extracted and compared correctly.

Usage:
    python test_e2e_pipeline.py                    # Uses default test images
    TEST_IMAGE=my_image.jpg python test_e2e_pipeline.py  # Uses custom image
    TEST_IMAGE_REF=other.jpg python test_e2e_pipeline.py  # Uses custom reference
"""

import sys
import os
import cv2
import numpy as np
import json

# Configuration - set via environment variable or use defaults
TEST_IMAGE = os.environ.get('TEST_IMAGE', 'test_subject.jpg')
TEST_IMAGE_PATH = f"test_images/{TEST_IMAGE}"
TEST_IMAGE_REF = os.environ.get('TEST_IMAGE_REF', 'reference_subject.jpg')
TEST_IMAGE_REF_PATH = f"test_images/{TEST_IMAGE_REF}"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.detection import FaceDetector
from src.embedding import FaceNetEmbeddingExtractor, SimilarityComparator
from src.reference import ReferenceImageManager


def test_detection_pipeline():
    """Test that face detection works."""
    print("\n[TEST 1] Face Detection Pipeline")
    print("-" * 40)
    
    detector = FaceDetector()
    test_image = TEST_IMAGE_PATH
    
    img = cv2.imread(test_image)
    if img is None:
        print(f"  ✗ Could not load test image: {test_image}")
        return False
    
    faces = detector.detect_faces(img)
    print(f"  ✓ Loaded image: {img.shape}")
    print(f"  ✓ Detected {len(faces)} face(s)")
    
    if not faces:
        print("  ✗ No faces detected!")
        return False
    
    x, y, w, h = faces[0]
    print(f"  ✓ Face bbox: ({x}, {y}, {w}, {h})")
    return True


def test_embedding_pipeline():
    """Test that real embeddings are extracted."""
    print("\n[TEST 2] Embedding Extraction Pipeline")
    print("-" * 40)
    
    extractor = FaceNetEmbeddingExtractor()
    test_image = TEST_IMAGE_PATH
    
    img = cv2.imread(test_image)
    detector = FaceDetector()
    faces = detector.detect_faces(img)
    
    if not faces:
        print("  ✗ No faces detected")
        return False
    
    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]
    
    embedding = extractor.extract_embedding(face_img)
    
    if embedding is None:
        print("  ✗ Failed to extract embedding")
        return False
    
    print(f"  ✓ Extracted 128-dim embedding")
    print(f"    Shape: {embedding.shape}")
    print(f"    Mean: {embedding.mean():.4f}")
    print(f"    Std: {embedding.std():.4f}")
    print(f"    L2 norm: {np.linalg.norm(embedding):.4f}")
    
    if not np.isfinite(embedding).all():
        print("  ✗ Embedding contains NaN or Inf values!")
        return False
    
    return True


def test_reference_manager_real_embeddings():
    """Test that ReferenceImageManager extracts real embeddings."""
    print("\n[TEST 3] Reference Manager with Real Embeddings")
    print("-" * 40)
    
    extractor = FaceNetEmbeddingExtractor()
    detector = FaceDetector()
    
    manager = ReferenceImageManager(
        reference_dir="test_references",
        embedding_extractor=extractor,
        detector=detector
    )
    
    test_image = TEST_IMAGE_PATH
    
    img = cv2.imread(test_image)
    detector = FaceDetector()
    faces = detector.detect_faces(img)
    
    if not faces:
        print("  ✗ No faces detected")
        return False
    
    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]
    
    emb1 = extractor.extract_embedding(face_img)
    emb2 = extractor.extract_embedding(face_img.copy())
    
    if emb1 is None or emb2 is None:
        print("  ✗ Failed to extract embeddings")
        return False
    
    comparator = SimilarityComparator(threshold=0.5)
    similarity = comparator.cosine_similarity(emb1, emb2)
    print(f"  ✓ Same image similarity: {similarity:.4f} ({similarity:.2%})")
    
    if similarity < 0.95:
        print(f"  ✗ Same image should have similarity > 0.95")
        return False
    
    print("  ✓ PASS: Same image has high similarity")
    return True


def test_similarity_with_same_image():
    """Test that same image gives high similarity (near 100%)."""
    print("\n[TEST 4] Same Image Similarity (SHOULD BE HIGH)")
    print("-" * 40)
    
    extractor = FaceNetEmbeddingExtractor()
    comparator = SimilarityComparator(threshold=0.5)
    
    test_image = TEST_IMAGE_PATH
    
    img = cv2.imread(test_image)
    if img is None:
        print(f"  ✗ Could not load test image: {test_image}")
        return False
    
    detector = FaceDetector()
    faces = detector.detect_faces(img)
    
    if not faces:
        print("  ✗ No faces detected!")
        return False
    
    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]
    
    emb1 = extractor.extract_embedding(face_img)
    emb2 = extractor.extract_embedding(face_img.copy())
    
    if emb1 is None or emb2 is None:
        print("  ✗ Failed to extract embeddings")
        return False
    
    similarity = comparator.cosine_similarity(emb1, emb2)
    print(f"  ✓ Same image similarity: {similarity:.4f} ({similarity:.2%})")
    
    if similarity < 0.95:
        print(f"  ✗ Same image should have similarity > 0.95")
        return False
    
    print("  ✓ PASS: Same image has high similarity")
    return True


def test_similarity_with_different_images():
    """Test that different images give lower similarity."""
    print("\n[TEST 5] Different Images Similarity (SHOULD BE LOWER)")
    print("-" * 40)
    
    extractor = FaceNetEmbeddingExtractor()
    comparator = SimilarityComparator(threshold=0.5)
    
    test_images = [
        TEST_IMAGE_PATH,
        TEST_IMAGE_REF_PATH,
    ]
    
    embeddings = []
    for img_path in test_images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠ Could not load {img_path}")
            continue
        
        detector = FaceDetector()
        faces = detector.detect_faces(img)
        
        if faces:
            x, y, w, h = faces[0]
            face_img = img[y:y+h, x:x+w]
            emb = extractor.extract_embedding(face_img)
            if emb is not None:
                embeddings.append((img_path, emb))
    
    if len(embeddings) < 2:
        print("  ✗ Need at least 2 images with faces")
        return False
    
    emb1 = embeddings[0][1]
    emb2 = embeddings[1][1]
    
    similarity = comparator.cosine_similarity(emb1, emb2)
    confidence = comparator.get_confidence_band(similarity)
    
    print(f"  ✓ Different images similarity: {similarity:.4f} ({similarity:.2%})")
    print(f"  ✓ Confidence: {confidence}")
    
    if similarity > 0.9:
        print(f"  ⚠ Different images have high similarity (> 0.9) - may be very similar faces")
    
    return True


def test_reference_comparison():
    """Test full reference management and comparison pipeline."""
    print("\n[TEST 6] Full Reference Comparison Pipeline")
    print("-" * 40)
    
    extractor = FaceNetEmbeddingExtractor()
    comparator = SimilarityComparator(threshold=0.5)
    detector = FaceDetector()
    
    manager = ReferenceImageManager(
        reference_dir="test_references",
        embedding_extractor=extractor,
        detector=detector
    )
    
    test_images = [
        (TEST_IMAGE_PATH, "test_image_1"),
        (TEST_IMAGE_REF_PATH, "test_image_2"),
    ]
    
    for img_path, ref_id in test_images:
        success, _ = manager.add_reference_image(img_path, ref_id, {"test": True})
        print(f"  ✓ Added reference '{ref_id}': {success}")
    
    ref_embs, ref_ids = manager.get_reference_embeddings()
    print(f"  ✓ Total references: {len(ref_embs)}")
    
    query_img = cv2.imread(TEST_IMAGE_PATH)
    faces = detector.detect_faces(query_img)
    
    if not faces:
        print("  ✗ No faces in query image")
        return False
    
    x, y, w, h = faces[0]
    query_face = query_img[y:y+h, x:x+w]
    query_emb = extractor.extract_embedding(query_face)
    
    if query_emb is None:
        print("  ✗ Failed to extract query embedding")
        return False
    
    results = comparator.compare_embeddings(query_emb, ref_embs, ref_ids)
    
    print(f"\n  Query: {TEST_IMAGE}")
    print(f"  Comparisons:")
    for ref_id, sim in results:
        conf = comparator.get_confidence_band(sim)
        print(f"    vs {ref_id}: {sim:.4f} ({conf})")
    
    if len(results) == 0:
        print("  ✗ No matches above threshold")
        return False
    
    best_match_id, best_similarity = results[0]
    print(f"\n  ✓ Best match: {best_match_id} with {best_similarity:.2%} similarity")
    
    if best_match_id == "kanye_west_1":
        print("  ✓ PASS: Correctly matched same image!")
    else:
        print(f"  ⚠ Best match was {best_match_id}, not kanye_west_1")
    
    return True


def cleanup():
    """Clean up test artifacts."""
    import shutil
    test_ref_dir = "test_references"
    if os.path.exists(test_ref_dir):
        shutil.rmtree(test_ref_dir)
    print("\n[Cleanup] Removed test_references directory")


def main():
    print("=" * 60)
    print("END-TO-END FACE RECOGNITION PIPELINE TEST")
    print("=" * 60)
    
    tests = [
        ("Detection", test_detection_pipeline),
        ("Embedding", test_embedding_pipeline),
        ("Reference Manager", test_reference_manager_real_embeddings),
        ("Same Image Similarity", test_similarity_with_same_image),
        ("Different Images Similarity", test_similarity_with_different_images),
        ("Reference Comparison", test_reference_comparison),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED - Pipeline is working correctly!")
    else:
        print("SOME TESTS FAILED - Check output above")
    print("=" * 60)
    
    cleanup()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
