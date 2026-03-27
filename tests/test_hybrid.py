import numpy as np
import pytest

from recommender.model.hybrid import (
    _l2_normalize_rows,
    compute_hybrid_vectors,
    cosine_score,
)


def test_normalize_rows_unit_norm():
    matrix = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    result = _l2_normalize_rows(matrix)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)


def test_normalize_rows_zero_vector_no_nan():
    matrix = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    result = _l2_normalize_rows(matrix)
    assert not np.any(np.isnan(result))
    np.testing.assert_array_equal(result[0], [0.0, 0.0])


def test_compute_hybrid_vectors_output_dtype():
    rng = np.random.default_rng(0)
    cf = rng.standard_normal((10, 8)).astype(np.float32)
    tag = rng.standard_normal((10, 8)).astype(np.float32)
    result = compute_hybrid_vectors(cf, tag, w_cf=1.0, w_tag=0.3)
    assert result.dtype == np.float16


def test_compute_hybrid_vectors_rows_are_normalized():
    rng = np.random.default_rng(1)
    cf = rng.standard_normal((20, 16)).astype(np.float32)
    tag = rng.standard_normal((20, 16)).astype(np.float32)
    result = compute_hybrid_vectors(cf, tag, w_cf=1.0, w_tag=0.3)
    norms = np.linalg.norm(result.astype(np.float32), axis=1)
    np.testing.assert_allclose(norms, np.ones(20), atol=0.01)  # float16 precision


def test_compute_hybrid_vectors_zero_cf_uses_tag_signal():
    """All-zero CF should still produce nonzero output from the tag component."""
    rng = np.random.default_rng(2)
    cf = np.zeros((5, 8), dtype=np.float32)
    tag = rng.standard_normal((5, 8)).astype(np.float32)
    result = compute_hybrid_vectors(cf, tag, w_cf=1.0, w_tag=0.3)
    norms = np.linalg.norm(result.astype(np.float32), axis=1)
    assert np.all(norms > 0.5)


def test_cosine_score_identical_vectors():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_score(v, v) == pytest.approx(1.0, abs=1e-5)


def test_cosine_score_orthogonal_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_score(a, b) == pytest.approx(0.0, abs=1e-5)


def test_cosine_score_opposite_vectors():
    v = np.array([1.0, 0.0], dtype=np.float32)
    assert cosine_score(v, -v) == pytest.approx(-1.0, abs=1e-5)


def test_cosine_score_zero_vector_returns_zero():
    v = np.array([1.0, 0.0], dtype=np.float32)
    z = np.zeros(2, dtype=np.float32)
    assert cosine_score(v, z) == 0.0
    assert cosine_score(z, z) == 0.0
