"""Tag vocabulary and per-post tag feature computation."""
from __future__ import annotations

import numpy as np


class TagVocab:
    """Append-only mapping: tag_string -> tag_id (int)."""

    def __init__(self):
        self._str_to_id: dict[str, int] = {}
        self._id_to_str: dict[int, str] = {}

    def get_or_add(self, tag: str) -> int:
        if tag not in self._str_to_id:
            tid = len(self._str_to_id)
            self._str_to_id[tag] = tid
            self._id_to_str[tid] = tag
        return self._str_to_id[tag]

    def __len__(self) -> int:
        return len(self._str_to_id)

    def to_dict(self) -> dict[str, str]:
        """Serializable form: {tag_id_str: tag_string}."""
        return {str(k): v for k, v in self._id_to_str.items()}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "TagVocab":
        vocab = cls()
        for id_str, tag_str in data.items():
            tid = int(id_str)
            vocab._str_to_id[tag_str] = tid
            vocab._id_to_str[tid] = tag_str
        return vocab


def compute_post_top_tags(
    tag_string: str,
    vocab: TagVocab,
    n_top: int = 50,
) -> list[tuple[int, float]]:
    """
    Parse a space-separated tag string, assign weight 1.0 per tag,
    return top-N (tag_id, weight) sorted by tag_id (for O(N) intersection).
    """
    tags = tag_string.split() if tag_string else []
    seen: dict[int, float] = {}
    for t in tags:
        tid = vocab.get_or_add(t)
        seen[tid] = seen.get(tid, 0.0) + 1.0

    # sort by weight descending, take top N, then re-sort by tag_id for intersection
    top = sorted(seen.items(), key=lambda x: -x[1])[:n_top]
    return sorted(top, key=lambda x: x[0])


def compute_tag_vector(
    top_tags: list[tuple[int, float]],
    tag_embeddings: np.ndarray,   # shape (vocab_size, D)
    dim: int,
) -> np.ndarray:
    """
    Compute a dense tag vector as a weighted sum of tag embedding rows, L2-normalized.
    Returns float32 vector of shape (D,).
    """
    if not top_tags or len(tag_embeddings) == 0:
        return np.zeros(dim, dtype=np.float32)

    vec = np.zeros(dim, dtype=np.float32)
    vocab_size = len(tag_embeddings)
    for tag_id, weight in top_tags:
        if tag_id < vocab_size:
            vec += weight * tag_embeddings[tag_id]

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec
