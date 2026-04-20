"""
Incremental collaborative embedding tables (user + post).

Update rule per event (user_id, post_id, action ∈ {+1, -1}):
  U[user] += lr * action * P[post] - lr * reg * U[user]
  P[post] += lr * action * U[user] - lr * reg * P[post]
  (using old copies for symmetry)
"""
from __future__ import annotations

import numpy as np


class EmbeddingTable:
    """Float32 embedding table backed by a growable dict of numpy arrays."""

    def __init__(self, dim: int, rng: np.random.Generator | None = None):
        self.dim = dim
        self._rng = rng or np.random.default_rng()
        # id -> row index in _matrix
        self._id_to_idx: dict[int, int] = {}
        self._rows: list[np.ndarray] = []

    def get_or_init(self, entity_id: int) -> np.ndarray:
        if entity_id not in self._id_to_idx:
            idx = len(self._rows)
            vec = self._rng.standard_normal(self.dim).astype(np.float32) * 0.01
            self._rows.append(vec)
            self._id_to_idx[entity_id] = idx
        return self._rows[self._id_to_idx[entity_id]]

    def __contains__(self, entity_id: int) -> bool:
        return entity_id in self._id_to_idx

    def ids(self) -> list[int]:
        return list(self._id_to_idx.keys())

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (id_array int64, matrix float32 (N, D))."""
        ids = np.array(self.ids(), dtype=np.int64)
        if len(ids) == 0:
            return ids, np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.stack([self._rows[self._id_to_idx[i]] for i in ids.tolist()], axis=0)
        return ids, matrix

    @classmethod
    def from_arrays(cls, ids: np.ndarray, matrix: np.ndarray) -> "EmbeddingTable":
        dim = matrix.shape[1] if matrix.ndim == 2 and len(matrix) > 0 else 64
        table = cls(dim=dim)
        m = matrix.astype(np.float32)
        for i, entity_id in enumerate(ids.tolist()):
            table._id_to_idx[entity_id] = len(table._rows)
            table._rows.append(m[i].copy())
        return table


def _clip_to_max_norm(vec: np.ndarray, max_norm: float) -> None:
    norm = float(np.linalg.norm(vec.astype(np.float64)))
    if np.isfinite(norm) and norm > max_norm > 0:
        vec *= max_norm / norm


def apply_event_batch(
    user_table: EmbeddingTable,
    post_table: EmbeddingTable,
    fav_count: dict[int, int],
    events: list[tuple[int, int, int]],   # (user_id, post_id, action)
    lr: float,
    reg: float,
    max_norm: float = 10.0,
) -> None:
    """Apply a batch of favorite/unfavorite events in-place."""
    for user_id, post_id, action in events:
        u = user_table.get_or_init(user_id).copy()
        p = post_table.get_or_init(post_id).copy()

        new_u = u + lr * action * p - lr * reg * u
        new_p = p + lr * action * u - lr * reg * p

        _clip_to_max_norm(new_u, max_norm)
        _clip_to_max_norm(new_p, max_norm)

        user_table.get_or_init(user_id)[:] = new_u
        post_table.get_or_init(post_id)[:] = new_p

        # update fav count (clamp >= 0)
        fav_count[post_id] = max(0, fav_count.get(post_id, 0) + action)
