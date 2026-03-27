import tempfile
from pathlib import Path

import numpy as np
import pytest

from recommender.store.top_tags import encode, decode_post


def _roundtrip(posts):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        encode(posts, d / "offsets", d / "payload")
        offsets = np.fromfile(str(d / "offsets"), dtype=np.uint64)
        payload = memoryview((d / "payload").read_bytes())
        # decode all posts while the temp dir is still alive
        return [decode_post(i, offsets, payload) for i in range(len(posts))]


def test_basic_roundtrip():
    posts = [
        [(1, 0.5), (3, 1.0), (7, 0.25)],
        [(2, 0.75)],
        [(4, 2.0), (5, 0.5)],
    ]
    results = _roundtrip(posts)
    for result, expected in zip(results, posts):
        assert len(result) == len(expected)
        for (atid, aw), (etid, ew) in zip(result, expected):
            assert atid == etid
            assert aw == pytest.approx(ew, rel=1e-5)


def test_empty_posts_roundtrip():
    posts = [[], [(1, 1.0)], []]
    results = _roundtrip(posts)
    assert results[0] == []
    assert results[1][0][0] == 1
    assert results[1][0][1] == pytest.approx(1.0)
    assert results[2] == []


def test_single_post_single_tag():
    results = _roundtrip([[(42, 0.5)]])
    assert results[0] == [(42, pytest.approx(0.5))]


def test_offsets_file_length():
    posts = [[(1, 1.0)], [(2, 1.0), (3, 1.0)], []]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        encode(posts, d / "offsets", d / "payload")
        offsets = np.fromfile(str(d / "offsets"), dtype=np.uint64)
        assert len(offsets) == len(posts) + 1
        assert int(offsets[0]) == 0


def test_offsets_are_monotone():
    # varying tag counts: 0, 1, 2, 3, 4 tags per post
    posts = [[(i, 1.0) for i in range(j)] for j in range(5)]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        encode(posts, d / "offsets", d / "payload")
        offsets = np.fromfile(str(d / "offsets"), dtype=np.uint64)
        for i in range(len(offsets) - 1):
            assert int(offsets[i]) <= int(offsets[i + 1])


def test_all_empty_posts():
    results = _roundtrip([[], [], []])
    assert all(r == [] for r in results)
