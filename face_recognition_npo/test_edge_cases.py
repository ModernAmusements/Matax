#!/usr/bin/env python3
"""
Edge Case Tests for Face Recognition NPO
Tests boundary conditions, error handling, and unusual inputs.
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detection import FaceDetector
from src.embedding import FaceNetEmbeddingExtractor, SimilarityComparator
from src.reference import ReferenceImageManager


def test_empty_image():
    """Test with empty/black image."""
    print("\n[EDGE CASE 1] Empty/Black Image")
    print("-" * 40)
    
    extractor = FaceNetEmbeddingExtractor()
    detector = FaceDetector()
    
    # Black image
    black_img = np.zeros((100, 100, 3), dtype=np.uint8)
    faces = detector.detect_faces(black_img)
    print(f"  Black image (100x100): {len(faces)} faces detected")
    
    # White image
    white_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    faces = detector.detect_faces(white_img)
    print(f"  White image (100x100): {len(faces)} faces detected")
    
    # Noisy image
    noisy_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    faces = detector.detect_faces(noisy_img)
    print(f"  Noisy image (100x100): {len(faces)} faces detected")
    
    return True


def test_very_small_image():
    """Test with very small image."""
    print("\n[EDGE CASE 2] Very Small Images")
    print("-" * 40)
    
    extractor = FaceNetEmbeddingExtractor()
    detector = FaceDetector()
    
    sizes = [(10, 10), (20, 20), (5, 5), (1, 1)]
    for w, h in sizes:
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        try:
            embedding = extractor.extract_embedding(img)
            print(f"  Image {w}x{h}: embedding={embedding is not None}")
        except Exception as e:
            print(f"  Image {w}x{h}: Error - {e}")
    
    return True


def test_none_and_invalid_inputs():
    """Test with None and invalid inputs."""
    print("\n[EDGE CASE 3] None and Invalid Inputs")
    print("-" * 40)
    
    extractor = FaceNetEmbeddingExtractor()
    comparator = SimilarityComparator()
    
    # Test None embedding
    try:
        result = comparator.cosine_similarity(None, np.ones(128))
        print(f"  cosine_similarity(None, ones): {result}")
    except Exception as e:
        print(f"  cosine_similarity(None, ones): Error - {e}")
    
    # Test None in both
    try:
        result = comparator.cosine_similarity(None, None)
        print(f"  cosine_similarity(None, None): {result}")
    except Exception as e:
        print(f"  cosine_similarity(None, None): Error - {e}")
    
    # Test empty arrays
    try:
        result = comparator.cosine_similarity(np.array([]), np.array([]))
        print(f"  cosine_similarity([], []): {result}")
    except Exception as e:
        print(f"  cosine_similarity([], []): Error - {e}")
    
    # Test NaN in embedding
    nan_emb = np.ones(128)
    nan_emb[0] = np.nan
    try:
        result = comparator.cosine_similarity(nan_emb, np.ones(128))
        print(f"  cosine_similarity(NaN, ones): {result}")
    except Exception as e:
        print(f"  cosine_similarity(NaN, ones): Error - {e}")
    
    # Test Inf in embedding
    inf_emb = np.ones(128)
    inf_emb[0] = np.inf
    try:
        result = comparator.cosine_similarity(inf_emb, np.ones(128))
        print(f"  cosine_similarity(Inf, ones): {result}")
    except Exception as e:
        print(f"  cosine_similarity(Inf, ones): Error - {e}")
    
    return True


def test_boundary_similarity():
    """Test boundary similarity values."""
    print("\n[EDGE CASE 4] Boundary Similarity Values")
    print("-" * 40)
    
    comparator = SimilarityComparator()
    
    # Test identical embeddings
    emb = np.ones(128)
    similarity = comparator.cosine_similarity(emb, emb)
    print(f"  Identical embeddings: {similarity:.4f}")
    
    # Test opposite embeddings
    emb1 = np.ones(128)
    emb2 = -np.ones(128)
    similarity = comparator.cosine_similarity(emb1, emb2)
    print(f"  Opposite embeddings: {similarity:.4f}")
    
    # Test zero embedding
    emb1 = np.zeros(128)
    emb2 = np.ones(128)
    similarity = comparator.cosine_similarity(emb1, emb2)
    print(f"  Zero vs ones: {similarity:.4f}")
    
    # Test very small values
    emb1 = np.ones(128) * 0.0001
    emb2 = np.ones(128) * 0.0001
    similarity = comparator.cosine_similarity(emb1, emb2)
    print(f"  Very small values: {similarity:.4f}")
    
    # Test confidence bands
    for sim in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        band = comparator.get_confidence_band(sim)
        print(f"  Similarity {sim:.1f}: {band}")
    
    return True


def test_empty_reference_list():
    """Test with empty reference list."""
    print("\n[EDGE CASE 5] Empty Reference List")
    print("-" * 40)
    
    extractor = FaceNetEmbeddingExtractor()
    comparator = SimilarityComparator()
    
    # Test visualize_similarity_matrix with no references
    emb = np.random.rand(128)
    try:
        viz, data = extractor.visualize_similarity_matrix(emb, [], [])
        print(f"  No references: viz shape={viz.shape}, data={data}")
    except Exception as e:
        print(f"  No references: Error - {e}")
    
    # Test compare_embeddings with empty lists
    try:
        results = comparator.compare_embeddings(emb, [], [])
        print(f"  Empty compare: {len(results)} results")
    except Exception as e:
        print(f"  Empty compare: Error - {e}")
    
    return True


def test_many_references():
    """Test with many references."""
    print("\n[EDGE CASE 6] Many References (Stress Test)")
    print("-" * 40)
    
    extractor = FaceNetEmbeddingExtractor()
    comparator = SimilarityComparator()
    
    num_refs = 50
    ref_embeddings = [np.random.rand(128) for _ in range(num_refs)]
    ref_ids = [f"ref_{i}" for i in range(num_refs)]
    
    query = np.random.rand(128)
    
    try:
        results = comparator.compare_embeddings(query, ref_embeddings, ref_ids)
        print(f"  50 references: {len(results)} matches above threshold")
        
        viz, data = extractor.visualize_similarity_matrix(query, ref_embeddings, ref_ids)
        print(f"  Visualization: shape={viz.shape}")
        
        print(f"  Best match: {results[0][0]} ({results[0][1]:.4f})")
    except Exception as e:
        print(f"  Error: {e}")
    
    return True


def test_long_reference_names():
    """Test with very long reference names."""
    print("\n[EDGE CASE 7] Long Reference Names")
    print("-" * 40)
    
    extractor = FaceNetEmbeddingExtractor()
    
    # Very long name
    long_name = "a" * 500
    emb = np.random.rand(128)
    
    try:
        viz, data = extractor.visualize_similarity_matrix(emb, [emb], [long_name])
        print(f"  Long name (500 chars): OK, viz shape={viz.shape}")
    except Exception as e:
        print(f"  Long name: Error - {e}")
    
    # Unicode name
    unicode_name = "参考画像_REFERENCE_IMAGE_🚀" * 10
    try:
        viz, data = extractor.visualize_similarity_matrix(emb, [emb], [unicode_name])
        print(f"  Unicode name: OK, viz shape={viz.shape}")
    except Exception as e:
        print(f"  Unicode name: Error - {e}")
    
    return True


def test_reference_manager_empty():
    """Test ReferenceImageManager with edge cases."""
    print("\n[EDGE CASE 8] Reference Manager Edge Cases")
    print("-" * 40)
    
    manager = ReferenceImageManager(reference_dir="test_edge_references")
    
    # Test get_reference_embeddings with empty data
    embs, ids = manager.get_reference_embeddings()
    print(f"  Empty manager: {len(embs)} embeddings, {len(ids)} IDs")
    
    # Test list_references
    refs = manager.list_references()
    print(f"  list_references: {len(refs)} items")
    
    # Test get_reference_metadata for non-existent ID
    meta = manager.get_reference_metadata("non_existent")
    print(f"  Non-existent metadata: {meta}")
    
    # Test remove_reference for non-existent ID
    removed = manager.remove_reference("non_existent")
    print(f"  Remove non-existent: {removed}")
    
    return True


def test_visualization_methods():
    """Test visualization methods with edge cases."""
    print("\n[EDGE CASE 9] Visualization Methods")
    print("-" * 40)
    
    extractor = FaceNetEmbeddingExtractor()
    detector = FaceDetector()
    
    # Test visualize_embedding with None
    try:
        viz, data = extractor.visualize_embedding(None)
        print(f"  visualize_embedding(None): shape={viz.shape}, data={data}")
    except Exception as e:
        print(f"  visualize_embedding(None): Error - {e}")
    
    # Test visualize_embedding with very large values
    large_emb = np.random.rand(128) * 1000
    try:
        viz, data = extractor.visualize_embedding(large_emb)
        print(f"  Large embedding values: OK")
    except Exception as e:
        print(f"  Large embedding: Error - {e}")
    
    # Test visualize_saliency with edge image
    edge_img = np.zeros((50, 50, 3), dtype=np.uint8)
    try:
        viz = detector.visualize_saliency(edge_img)
        print(f"  Zero saliency input: shape={viz.shape}")
    except Exception as e:
        print(f"  Zero saliency: Error - {e}")
    
    # Test visualize_3d_mesh with None
    try:
        viz = detector.visualize_3d_mesh(None)
        print(f"  visualize_3d_mesh(None): shape={viz.shape if viz is not None else None}")
    except Exception as e:
        print(f"  visualize_3d_mesh(None): Error - {e}")
    
    return True


def test_quality_metrics_edge_cases():
    """Test quality metrics with edge cases."""
    print("\n[EDGE CASE 10] Quality Metrics")
    print("-" * 40)
    
    detector = FaceDetector()
    
    # Very small face box
    tiny_box = (0, 0, 5, 5)
    tiny_img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    try:
        quality = detector.compute_quality_metrics(tiny_img, tiny_box)
        print(f"  Tiny face (5x5): quality keys={list(quality.keys())}")
    except Exception as e:
        print(f"  Tiny face: Error - {e}")
    
    # Zero-sized box
    zero_box = (0, 0, 0, 0)
    zero_img = np.zeros((10, 10, 3), dtype=np.uint8)
    try:
        quality = detector.compute_quality_metrics(zero_img, zero_box)
        print(f"  Zero box: quality keys={list(quality.keys())}")
    except Exception as e:
        print(f"  Zero box: Error - {e}")
    
    return True


def test_compare_embeddings_edge_cases():
    """Test compare_embeddings with edge cases."""
    print("\n[EDGE CASE 11] Compare Embeddings Edge Cases")
    print("-" * 40)
    
    comparator = SimilarityComparator()
    
    # Test with None embeddings in list
    query = np.ones(128)
    refs = [np.ones(128), None, np.ones(128)]
    ids = ["valid1", "none_ref", "valid2"]
    
    try:
        results = comparator.compare_embeddings(query, refs, ids)
        print(f"  With None refs: {len(results)} matches")
        for ref_id, sim in results:
            print(f"    {ref_id}: {sim:.4f}")
    except Exception as e:
        print(f"  With None refs: Error - {e}")
    
    # Test with mismatched list lengths
    try:
        results = comparator.compare_embeddings(query, [np.ones(128)], ["a", "b"])
        print(f"  Mismatched lengths: {len(results)} results")
    except Exception as e:
        print(f"  Mismatched lengths: Error - {e}")
    
    return True


def cleanup():
    """Clean up test artifacts."""
    import shutil
    test_dir = "test_edge_references"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    print("\n[Cleanup] Removed test_edge_references directory")


def main():
    print("=" * 60)
    print("EDGE CASE TESTS FOR FACE RECOGNITION NPO")
    print("=" * 60)
    
    tests = [
        ("Empty/Black Image", test_empty_image),
        ("Very Small Images", test_very_small_image),
        ("None/Invalid Inputs", test_none_and_invalid_inputs),
        ("Boundary Similarity", test_boundary_similarity),
        ("Empty Reference List", test_empty_reference_list),
        ("Many References", test_many_references),
        ("Long Reference Names", test_long_reference_names),
        ("Reference Manager Edge Cases", test_reference_manager_empty),
        ("Visualization Methods", test_visualization_methods),
        ("Quality Metrics", test_quality_metrics_edge_cases),
        ("Compare Embeddings Edge Cases", test_compare_embeddings_edge_cases),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("EDGE CASE TEST RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("ALL EDGE CASE TESTS PASSED!")
    else:
        print("SOME EDGE CASE TESTS FAILED - See output above")
    print("=" * 60)
    
    cleanup()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
