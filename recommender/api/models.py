"""Pydantic request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class FavCounts(BaseModel):
    query: int
    candidate: int


class Explanation(BaseModel):
    shared_tags: list[str]
    fav_counts: FavCounts | None = None


class SimilarResult(BaseModel):
    post_id: int
    score: float
    explanation: Explanation | None = None


class SimilarResponse(BaseModel):
    post_id: int
    model_version: str
    results: list[SimilarResult]
