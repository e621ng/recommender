"""Tag vocabulary and per-post tag feature computation."""
from __future__ import annotations

import math
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

import numpy as np


@dataclass
class TagMeta:
    category: int
    post_count: int


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
    n_top: int,
    n_posts: int,
    tag_metadata: dict[str, TagMeta],
    category_multipliers: dict[int, float],
    excluded_tags: AbstractSet[str] = frozenset(),
) -> list[tuple[int, float]]:
    """
    Parse a space-separated tag string, compute a weight per tag from its
    category multiplier and IDF score, return top-N (tag_id, weight) sorted
    by tag_id (for O(N) intersection during explainability).

    Tags with a category multiplier of 0.0 (invalid) are excluded entirely.
    Unknown tags (not in tag_metadata) fall back to category 0, post_count 1.
    """
    tags = tag_string.split() if tag_string else []
    seen: dict[int, float] = {}
    for t in tags:
        if t in excluded_tags:
            continue
        meta = tag_metadata.get(t, TagMeta(category=0, post_count=1))
        multiplier = category_multipliers.get(meta.category, 1.0)
        if multiplier == 0.0:
            continue
        idf = math.log((n_posts + 1) / (meta.post_count + 1)) + 1.0
        weight = multiplier * idf
        tid = vocab.get_or_add(t)
        # accumulate in case of duplicate tags in the string (shouldn't happen, but safe)
        seen[tid] = seen.get(tid, 0.0) + weight

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
