"""Thin hnswlib wrapper."""
from pathlib import Path

import numpy as np
import hnswlib


def build_index(
    vectors: np.ndarray,   # float32 or float16 (N, D)
    ids: np.ndarray,       # int64 (N,)
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 100,
) -> hnswlib.Index:
    n, dim = vectors.shape
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
    index.add_items(vectors.astype(np.float32), ids.astype(np.int64))
    index.set_ef(ef_search)
    return index


def load_index(path: Path, dim: int, ef_search: int = 100) -> hnswlib.Index:
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(str(path))
    index.set_ef(ef_search)
    return index


def query_index(
    index: hnswlib.Index,
    vector: np.ndarray,   # 1-D float32
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels, distances) arrays of shape (limit,)."""
    labels, distances = index.knn_query(vector.astype(np.float32).reshape(1, -1), k=limit)
    return labels[0], distances[0]
