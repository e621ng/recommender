import math

import pytest

from recommender.model.tags import TagMeta, TagVocab, compute_post_top_tags

# Mirrors the defaults in config.py
CAT_MULTS = {
    0: 1.0,   # general
    1: 3.0,   # artist
    3: 1.5,   # copyright
    4: 2.5,   # character
    5: 2.0,   # species
    6: 0.0,   # invalid — excluded
    7: 0.5,   # meta
    8: 1.2,   # lore
}


def _idf(n_posts, post_count):
    return math.log((n_posts + 1) / (post_count + 1)) + 1.0


def test_output_sorted_by_tag_id_not_weight():
    """Output must be sorted by tag_id ascending, not by weight descending."""
    vocab = TagVocab()
    # Assign IDs in a known order: "low_weight" gets 0, "high_weight" gets 1
    vocab.get_or_add("low_weight")
    vocab.get_or_add("high_weight")
    tag_meta = {
        "low_weight": TagMeta(category=7, post_count=1),   # mult=0.5
        "high_weight": TagMeta(category=1, post_count=1),  # mult=3.0
    }
    result = compute_post_top_tags(
        "low_weight high_weight", vocab,
        n_top=10, n_posts=100,
        tag_metadata=tag_meta,
        category_multipliers=CAT_MULTS,
    )
    assert len(result) == 2
    ids = [tid for tid, _ in result]
    assert ids == sorted(ids), "output must be sorted by tag_id, not by weight"
    assert ids[0] == 0  # low_weight (id=0) first despite lower weight


def test_invalid_category_excluded():
    vocab = TagVocab()
    tag_meta = {
        "good": TagMeta(category=0, post_count=1),
        "bad": TagMeta(category=6, post_count=1),  # mult=0.0 — excluded
    }
    result = compute_post_top_tags(
        "good bad", vocab,
        n_top=10, n_posts=100,
        tag_metadata=tag_meta,
        category_multipliers=CAT_MULTS,
    )
    assert len(result) == 1
    assert "bad" not in vocab._str_to_id   # never added to vocab
    assert result[0][0] == vocab._str_to_id["good"]


def test_idf_weight_formula():
    vocab = TagVocab()
    n_posts, post_count = 1000, 10
    tag_meta = {"mytag": TagMeta(category=0, post_count=post_count)}
    result = compute_post_top_tags(
        "mytag", vocab,
        n_top=10, n_posts=n_posts,
        tag_metadata=tag_meta,
        category_multipliers=CAT_MULTS,
    )
    assert len(result) == 1
    expected_weight = 1.0 * _idf(n_posts, post_count)  # category 0 mult=1.0
    assert result[0][1] == pytest.approx(expected_weight, rel=1e-6)


def test_category_multiplier_scales_weight():
    vocab = TagVocab()
    n_posts = 100
    tag_meta = {
        "artist_tag": TagMeta(category=1, post_count=10),   # mult=3.0
        "general_tag": TagMeta(category=0, post_count=10),  # mult=1.0
    }
    result = compute_post_top_tags(
        "artist_tag general_tag", vocab,
        n_top=10, n_posts=n_posts,
        tag_metadata=tag_meta,
        category_multipliers=CAT_MULTS,
    )
    by_id = {tid: w for tid, w in result}
    idf = _idf(n_posts, 10)
    assert by_id[vocab._str_to_id["artist_tag"]] == pytest.approx(3.0 * idf, rel=1e-6)
    assert by_id[vocab._str_to_id["general_tag"]] == pytest.approx(1.0 * idf, rel=1e-6)


def test_unknown_tag_fallback():
    """Tags absent from tag_metadata fall back to category=0, post_count=1."""
    vocab = TagVocab()
    result = compute_post_top_tags(
        "mystery_tag", vocab,
        n_top=10, n_posts=100,
        tag_metadata={},
        category_multipliers=CAT_MULTS,
    )
    assert len(result) == 1
    expected = 1.0 * _idf(100, 1)  # category 0 mult=1.0, post_count=1
    assert result[0][1] == pytest.approx(expected, rel=1e-6)


def test_empty_tag_string():
    vocab = TagVocab()
    result = compute_post_top_tags(
        "", vocab,
        n_top=10, n_posts=100,
        tag_metadata={},
        category_multipliers=CAT_MULTS,
    )
    assert result == []


def test_duplicate_tags_accumulate_weight():
    """A tag appearing more than once in tag_string has its weight summed."""
    vocab = TagVocab()
    tag_meta = {"mytag": TagMeta(category=0, post_count=10)}
    single = compute_post_top_tags(
        "mytag", vocab,
        n_top=10, n_posts=100,
        tag_metadata=tag_meta,
        category_multipliers=CAT_MULTS,
    )
    double = compute_post_top_tags(
        "mytag mytag", vocab,
        n_top=10, n_posts=100,
        tag_metadata=tag_meta,
        category_multipliers=CAT_MULTS,
    )
    assert len(double) == 1
    assert double[0][1] == pytest.approx(single[0][1] * 2, rel=1e-6)


def test_n_top_limits_output():
    vocab = TagVocab()
    tags = [f"tag_{i}" for i in range(20)]
    tag_meta = {t: TagMeta(category=0, post_count=i + 1) for i, t in enumerate(tags)}
    result = compute_post_top_tags(
        " ".join(tags), vocab,
        n_top=5, n_posts=1000,
        tag_metadata=tag_meta,
        category_multipliers=CAT_MULTS,
    )
    assert len(result) == 5
    ids = [tid for tid, _ in result]
    assert ids == sorted(ids), "truncated output must still be sorted by tag_id"
