import numpy as np
import pytest

from recommender.model.tags import compute_tag_vector


def test_empty_top_tags_returns_zero():
    emb = np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32)
    result = compute_tag_vector([], emb, dim=4)
    np.testing.assert_array_equal(result, np.zeros(4, dtype=np.float32))


def test_empty_embeddings_returns_zero():
    emb = np.zeros((0, 4), dtype=np.float32)
    result = compute_tag_vector([(0, 1.0)], emb, dim=4)
    np.testing.assert_array_equal(result, np.zeros(4, dtype=np.float32))


def test_out_of_bounds_tag_id_skipped():
    emb = np.ones((3, 4), dtype=np.float32)  # vocab_size=3
    # tag_id=5 exceeds vocab_size; only tag_id=1 should contribute
    result = compute_tag_vector([(1, 1.0), (5, 999.0)], emb, dim=4)
    expected = emb[1] / np.linalg.norm(emb[1])
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_output_is_normalized():
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((10, 8)).astype(np.float32)
    result = compute_tag_vector([(0, 0.5), (2, 1.0), (5, 0.8)], emb, dim=8)
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)


def test_weighted_sum_then_normalized():
    """Uses identity matrix so the weighted sum is easy to verify analytically."""
    emb = np.eye(4, dtype=np.float32)
    # tag 0 (weight=1) + tag 1 (weight=1) → sum=[1,1,0,0] → normalized=[1/√2, 1/√2, 0, 0]
    result = compute_tag_vector([(0, 1.0), (1, 1.0)], emb, dim=4)
    expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_weight_scaling_affects_direction():
    """Higher weight on one tag should pull the result toward that tag's embedding."""
    emb = np.eye(4, dtype=np.float32)
    # tag 0 with large weight should dominate
    result = compute_tag_vector([(0, 10.0), (1, 1.0)], emb, dim=4)
    # result[0] should be larger than result[1]
    assert result[0] > result[1]
