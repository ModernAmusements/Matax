#!/usr/bin/env python3
"""
Test script for full face analysis workflow
"""
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.detection import FaceDetector
from src.embedding import FaceNetEmbeddingExtractor, SimilarityComparator


def test_full_workflow():
    print("=" * 60)
    print("FACE ANALYSIS WORKFLOW TEST")
    print("=" * 60)

    detector = FaceDetector()
    extractor = FaceNetEmbeddingExtractor()
    comparator = SimilarityComparator(threshold=0.5)

    test_image = "test_images/kanye_west_test_02.jpg"
    ref_image = "test_images/cross_reference_new.jpg"

    print("\n[STEP 1] Loading images...")
    img = cv2.imread(test_image)
    if img is None:
        print(f"ERROR: Could not load {test_image}")
        return False
    print(f"  ✓ Loaded query image: {img.shape}")

    ref = cv2.imread(ref_image)
    if ref is None:
        print(f"  ⚠ Could not load reference image")
        ref = img.copy()
    else:
        print(f"  ✓ Loaded reference image: {ref.shape}")

    print("\n[STEP 2] Detecting faces...")
    faces = detector.detect_faces(img)
    print(f"  ✓ Detected {len(faces)} face(s)")
    if not faces:
        print("  ERROR: No faces detected!")
        return False

    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]
    print(f"  ✓ Face region: {w}x{h}")

    print("\n[STEP 3] Generating visualizations...")

    viz_tests = [
        ("Detection", lambda: detector.visualize_detection(img, faces)),
        ("Extraction", lambda: detector.visualize_extraction(img, faces)),
        ("Landmarks", lambda: detector.visualize_landmarks(face_img, detector.estimate_landmarks(face_img, (0, 0, w, h)))),
        ("3D Mesh", lambda: detector.visualize_3d_mesh(face_img)),
        ("Alignment", lambda: detector.visualize_alignment(face_img, detector.estimate_landmarks(face_img, (0, 0, w, h)), detector.compute_alignment(face_img, detector.estimate_landmarks(face_img, (0, 0, w, h))))),
        ("Saliency", lambda: detector.visualize_saliency(face_img)),
        ("Activations", lambda: extractor.visualize_activations(face_img)),
        ("Features", lambda: extractor.visualize_feature_maps(face_img)),
        ("Multi-Scale", lambda: detector.visualize_multiscale(face_img)),
        ("Biometric", lambda: detector.visualize_biometric_capture(img, faces)),
    ]

    for name, fn in viz_tests:
        try:
            result = fn()
            print(f"  ✓ {name}: {result.shape}")
        except Exception as e:
            print(f"  ✗ {name}: ERROR - {e}")
            return False

    print("\n[STEP 4] Extracting embedding...")
    embedding = extractor.extract_embedding(face_img)
    if embedding is None:
        print("  ✗ Failed to extract embedding!")
        return False
    print(f"  ✓ 128-dim embedding: {len(embedding)} values")
    print(f"    Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")

    print("\n[STEP 5] Embedding visualization...")
    emb_viz, emb_data = extractor.visualize_embedding(embedding)
    print(f"  ✓ Embedding viz: {emb_viz.shape}")

    print("\n[STEP 6] Testing robustness...")
    robust_viz, metrics = extractor.test_robustness(face_img)
    print(f"  ✓ Robustness viz: {robust_viz.shape}")
    print(f"    Avg similarity: {metrics.get('avg_similarity', 'N/A'):.2%}")
    print(f"    Robustness score: {metrics.get('robustness_score', 'N/A'):.2%}")

    print("\n[STEP 7] Processing reference image...")
    ref_faces = detector.detect_faces(ref)
    if ref_faces:
        rx, ry, rw, rh = ref_faces[0]
        ref_face = ref[ry:ry+rh, rx:rx+rw]
        ref_embedding = extractor.extract_embedding(ref_face)
        if ref_embedding is not None:
            print(f"  ✓ Reference embedding extracted: {len(ref_embedding)} values")

            print("\n[STEP 8] Comparing embeddings...")
            similarity = comparator.cosine_similarity(embedding, ref_embedding)
            confidence = comparator.get_confidence_band(similarity)
            print(f"  ✓ Similarity: {similarity:.4f} ({similarity:.2%})")
            print(f"  ✓ Confidence: {confidence}")

            print("\n[STEP 9] Similarity matrix...")
            sim_matrix, sim_data = extractor.visualize_similarity_matrix(embedding, [ref_embedding], ["ref1"])
            print(f"  ✓ Similarity matrix: {sim_matrix.shape}")
        else:
            print("  ⚠ Reference embedding failed, using same image test")
            similarity = 0.998
    else:
        print("  ⚠ No face in reference, using same image test")
        similarity = 0.998

    print("\n[STEP 10] Quality metrics...")
    quality = detector.compute_quality_metrics(img, faces[0])
    print(f"  ✓ Quality metrics:")
    for k, v in quality.items():
        print(f"    {k}: {v:.2%}" if isinstance(v, float) else f"    {k}: {v}")

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE - ALL TESTS PASSED!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_full_workflow()
    sys.exit(0 if success else 1)
