"""Hybrid vector computation and per-component score helpers."""
import numpy as np


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    # float64 upcast is intentional: float32 norm computation squares each element,
    # which overflows to inf when embeddings reach extreme magnitudes under SGD.
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
    bad = (norms == 0) | ~np.isfinite(norms)
    safe = np.where(bad, 1.0, norms)
    result = (matrix / safe).astype(matrix.dtype)
    result[bad.squeeze(axis=1)] = 0.0
    return result


def compute_hybrid_vectors(
    post_cf: np.ndarray,    # float32 (N, D) — collaborative embeddings
    post_tag: np.ndarray,   # float32 (N, D) — tag vectors
    w_cf: float,
    w_tag: float,
) -> np.ndarray:
    """
    Blend and normalize to produce final serving vectors.
    Returns float16 (N, D).
    """
    cf_norm = _l2_normalize_rows(post_cf)
    tag_norm = _l2_normalize_rows(post_tag)
    blended = w_cf * cf_norm + w_tag * tag_norm
    hybrid = _l2_normalize_rows(blended)
    return hybrid.astype(np.float16)


def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_breakdown(
    query_cf: np.ndarray,
    cand_cf: np.ndarray,
    query_tag: np.ndarray,
    cand_tag: np.ndarray,
) -> dict[str, float]:
    return {
        "cf": round(cosine_score(query_cf, cand_cf), 4),
        "tag": round(cosine_score(query_tag, cand_tag), 4),
    }
