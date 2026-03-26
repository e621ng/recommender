"""Similarity query engine backed by a loaded ModelBundle."""
from __future__ import annotations

import numpy as np

from recommender.api.models import (
    Explanation, FavCounts, ScoreBreakdown, SimilarResponse, SimilarResult,
)
from recommender.model.ann import query_index
from recommender.store.reader import ModelBundle


class SimilarityEngine:
    def __init__(self, bundle: ModelBundle, m_shared_tags: int = 6):
        self._b = bundle
        self._m = m_shared_tags

    def query(
        self,
        post_id: int,
        limit: int,
        explain: bool,
        include_scores: bool,
    ) -> SimilarResponse:
        b = self._b
        idx = b.post_index.get(post_id)
        if idx is None:
            return SimilarResponse(post_id=post_id, model_version=b.version, results=[])

        q_vec = b.post_vectors[idx].astype(np.float32)
        # Request limit+1 to exclude the query post itself
        labels, distances = query_index(b.ann, q_vec, limit=min(limit + 1, b.ann.get_current_count()))

        results: list[SimilarResult] = []
        for label, dist in zip(labels.tolist(), distances.tolist()):
            if int(label) == post_id:
                continue
            if len(results) >= limit:
                break

            score = round(1.0 - float(dist), 4) if include_scores else 0.0
            explanation = None

            if explain:
                cand_idx = b.post_index.get(int(label))
                if cand_idx is not None:
                    explanation = self._explain(idx, cand_idx, post_id, int(label))

            results.append(SimilarResult(
                post_id=int(label),
                score=score,
                explanation=explanation,
            ))

        return SimilarResponse(post_id=post_id, model_version=b.version, results=results)

    def _explain(
        self, query_idx: int, cand_idx: int, query_post_id: int, cand_post_id: int,
    ) -> Explanation:
        b = self._b
        a_tags = b.get_top_tags(query_idx)
        c_tags = b.get_top_tags(cand_idx)

        shared = _intersect_top(a_tags, c_tags, self._m)
        shared_names = [b.tag_name(tid) for tid, _ in shared]

        # Score breakdown: cosine of raw component vectors
        q_vec = b.post_vectors[query_idx].astype(np.float32)
        c_vec = b.post_vectors[cand_idx].astype(np.float32)
        # For now, both components are encoded in the single hybrid vector;
        # report the hybrid score as "cf" and 0.0 tag (components not stored separately in serving)
        combined = float(np.dot(q_vec, c_vec) / (
            (np.linalg.norm(q_vec) or 1.0) * (np.linalg.norm(c_vec) or 1.0)
        ))
        breakdown = ScoreBreakdown(cf=round(combined, 4), tag=0.0)

        fav_counts = FavCounts(
            query=int(b.fav_count[query_idx]),
            candidate=int(b.fav_count[cand_idx]),
        )

        return Explanation(
            shared_tags=shared_names,
            score_breakdown=breakdown,
            fav_counts=fav_counts,
        )


def _intersect_top(
    a: list[tuple[int, float]],
    b: list[tuple[int, float]],
    m: int,
) -> list[tuple[int, float]]:
    """O(N) merge of two tag_id-sorted lists; return top-m by min(wA, wB)."""
    shared: list[tuple[int, float]] = []
    ia, ib = 0, 0
    while ia < len(a) and ib < len(b):
        if a[ia][0] == b[ib][0]:
            shared.append((a[ia][0], min(a[ia][1], b[ib][1])))
            ia += 1
            ib += 1
        elif a[ia][0] < b[ib][0]:
            ia += 1
        else:
            ib += 1
    return sorted(shared, key=lambda x: -x[1])[:m]
