import tempfile
from pathlib import Path

import numpy as np
import orjson
import pytest

from recommender.store.post_top_tags_store import PostTopTagsStore, _MIGRATE_LIMIT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tags(post_id: int, n: int = 3) -> list[tuple[int, float]]:
    """Deterministic tags for a post_id so tests can verify exact values."""
    return [(post_id * 100 + i, float(post_id + i) * 0.1) for i in range(n)]


def _assert_tags_equal(actual, expected):
    assert len(actual) == len(expected)
    for (atid, aw), (etid, ew) in zip(actual, expected):
        assert atid == etid
        assert aw == pytest.approx(ew, rel=1e-5)


# ---------------------------------------------------------------------------
# Empty store
# ---------------------------------------------------------------------------

def test_empty_store_keys_and_get():
    store = PostTopTagsStore._empty()
    assert store.keys() == set()
    assert store.get(1, []) == []
    assert store.get(999) is None


def test_load_missing_returns_empty():
    with tempfile.TemporaryDirectory() as d:
        store = PostTopTagsStore.load(Path(d))
    assert store.keys() == set()
    assert store.get(1, None) is None


# ---------------------------------------------------------------------------
# Delta-only (before save)
# ---------------------------------------------------------------------------

def test_setitem_getitem_delta():
    store = PostTopTagsStore._empty()
    tags = _make_tags(10)
    store[10] = tags
    _assert_tags_equal(store[10], tags)
    assert store.get(99, []) == []
    assert 10 in store.keys()
    assert 99 not in store.keys()


def test_getitem_missing_raises():
    store = PostTopTagsStore._empty()
    with pytest.raises(KeyError):
        _ = store[42]


def test_delta_priority_over_base():
    """Delta entry for an existing base key overrides base without needing save."""
    with tempfile.TemporaryDirectory() as d:
        tdir = Path(d)
        # Write base via from_dict
        base = {5: [(1, 0.9), (2, 0.8)]}
        PostTopTagsStore.from_dict(base).save(tdir)

        store = PostTopTagsStore.load(tdir)
        # Override key 5 in delta
        new_tags = [(7, 0.1)]
        store[5] = new_tags
        _assert_tags_equal(store[5], new_tags)


# ---------------------------------------------------------------------------
# Save and reload roundtrip
# ---------------------------------------------------------------------------

def test_save_and_reload():
    with tempfile.TemporaryDirectory() as d:
        tdir = Path(d)
        store = PostTopTagsStore._empty()
        posts = {1: _make_tags(1), 42: _make_tags(42), 100: _make_tags(100)}
        for pid, tags in posts.items():
            store[pid] = tags
        store.save(tdir)

        reloaded = PostTopTagsStore.load(tdir)
        assert reloaded.keys() == set(posts)
        for pid, tags in posts.items():
            _assert_tags_equal(reloaded.get(pid, []), tags)


def test_from_dict_roundtrip():
    """Backfill path: from_dict → save → load → verify."""
    with tempfile.TemporaryDirectory() as d:
        tdir = Path(d)
        d_in = {i: _make_tags(i, n=5) for i in range(1, 201)}
        PostTopTagsStore.from_dict(d_in).save(tdir)

        store = PostTopTagsStore.load(tdir)
        assert store.keys() == set(d_in)
        for pid, tags in d_in.items():
            _assert_tags_equal(store[pid], tags)


def test_posts_with_empty_tag_list():
    """Posts that have no qualifying tags store an empty list."""
    with tempfile.TemporaryDirectory() as d:
        tdir = Path(d)
        store = PostTopTagsStore._empty()
        store[1] = []
        store[2] = [(10, 0.5)]
        store.save(tdir)

        reloaded = PostTopTagsStore.load(tdir)
        assert reloaded.get(1, None) == []
        _assert_tags_equal(reloaded.get(2, []), [(10, 0.5)])


# ---------------------------------------------------------------------------
# Merge: delta into base
# ---------------------------------------------------------------------------

def test_save_merges_delta_into_base():
    """Two-step: write base, reload, update + insert, save, reload — verify."""
    with tempfile.TemporaryDirectory() as d:
        tdir = Path(d)

        # Step 1: populate base with posts 1..10
        base = {i: _make_tags(i) for i in range(1, 11)}
        PostTopTagsStore.from_dict(base).save(tdir)

        # Step 2: load, update post 5, insert post 99
        store = PostTopTagsStore.load(tdir)
        updated_tags = [(999, 9.9)]
        store[5] = updated_tags
        store[99] = _make_tags(99)
        store.save(tdir)

        # Step 3: reload and verify
        reloaded = PostTopTagsStore.load(tdir)
        expected_keys = set(range(1, 11)) | {99}
        assert reloaded.keys() == expected_keys

        # Unchanged base entries survive
        for pid in range(1, 11):
            if pid == 5:
                continue
            _assert_tags_equal(reloaded[pid], base[pid])

        # Updated entry reflects delta
        _assert_tags_equal(reloaded[5], updated_tags)
        # New entry present
        _assert_tags_equal(reloaded[99], _make_tags(99))


def test_keys_union():
    """keys() returns union of base IDs and delta keys."""
    with tempfile.TemporaryDirectory() as d:
        tdir = Path(d)
        PostTopTagsStore.from_dict({1: _make_tags(1), 2: _make_tags(2)}).save(tdir)

        store = PostTopTagsStore.load(tdir)
        store[3] = _make_tags(3)  # delta only
        store[2] = _make_tags(2)  # in both

        assert store.keys() == {1, 2, 3}


# ---------------------------------------------------------------------------
# In-place update after save (used by _build_tag_matrix)
# ---------------------------------------------------------------------------

def test_save_updates_self_inplace():
    """After save(), the same instance can serve reads via lazy reload."""
    with tempfile.TemporaryDirectory() as d:
        tdir = Path(d)
        store = PostTopTagsStore._empty()
        tags = _make_tags(7, n=4)
        store[7] = tags
        store.save(tdir)

        # Without creating a new instance, reads should still work.
        _assert_tags_equal(store.get(7, []), tags)
        assert 7 in store.keys()


def test_save_then_set_and_read():
    """After save(), new delta writes are visible without another save."""
    with tempfile.TemporaryDirectory() as d:
        tdir = Path(d)
        store = PostTopTagsStore._empty()
        store[1] = _make_tags(1)
        store.save(tdir)

        # Add a new post to the delta
        store[2] = _make_tags(2)
        _assert_tags_equal(store.get(2, []), _make_tags(2))
        # Old post still readable from base (via lazy reload)
        _assert_tags_equal(store.get(1, []), _make_tags(1))


# ---------------------------------------------------------------------------
# Legacy JSON migration
# ---------------------------------------------------------------------------

def test_auto_migrate_legacy_json():
    """Small JSON in tdir → load() migrates to binary, data intact."""
    with tempfile.TemporaryDirectory() as d:
        tdir = Path(d)
        original = {i: _make_tags(i) for i in range(1, 51)}
        json_bytes = orjson.dumps({str(k): v for k, v in original.items()})
        (tdir / "post_top_tags.pkl.json").write_bytes(json_bytes)

        store = PostTopTagsStore.load(tdir)

        # Binary files should now exist
        assert (tdir / "post_top_tags_post_ids.npy").exists()
        assert (tdir / "post_top_tags.offsets.u64").exists()
        assert (tdir / "post_top_tags.payload.bin").exists()

        assert store.keys() == set(original)
        for pid, tags in original.items():
            _assert_tags_equal(store[pid], tags)


def test_skip_large_legacy_json(tmp_path):
    """JSON exceeding _MIGRATE_LIMIT → load() returns empty store."""
    import os
    json_path = tmp_path / "post_top_tags.pkl.json"
    # Create a sparse file that reports the correct size without filling disk.
    json_path.write_bytes(b"{}")
    os.truncate(str(json_path), _MIGRATE_LIMIT + 1)
    assert json_path.stat().st_size == _MIGRATE_LIMIT + 1

    store = PostTopTagsStore.load(tmp_path)

    assert store.keys() == set()
    assert not (tmp_path / "post_top_tags_post_ids.npy").exists()


# ---------------------------------------------------------------------------
# Large-scale merge correctness
# ---------------------------------------------------------------------------

def test_merge_large():
    """50K base posts + 500 delta (mix of updates and inserts) → spot-check."""
    rng = np.random.default_rng(0)
    n_base = 50_000

    # Base: post IDs 0..49999, each with 5 tags
    base = {i: [(i * 10 + j, float(j) * 0.1) for j in range(5)] for i in range(n_base)}

    with tempfile.TemporaryDirectory() as d:
        tdir = Path(d)
        PostTopTagsStore.from_dict(base).save(tdir)

        store = PostTopTagsStore.load(tdir)

        # 250 updates to existing posts
        updated_ids = rng.choice(n_base, size=250, replace=False).tolist()
        update_map = {}
        for pid in updated_ids:
            new_tags = [(pid * 100 + 1, 9.9)]
            store[pid] = new_tags
            update_map[pid] = new_tags

        # 250 new post IDs beyond the base range
        new_ids = list(range(n_base, n_base + 250))
        new_map = {}
        for pid in new_ids:
            tags = [(pid, float(pid) * 0.01)]
            store[pid] = tags
            new_map[pid] = tags

        store.save(tdir)

        reloaded = PostTopTagsStore.load(tdir)
        expected_keys = set(range(n_base)) | set(new_ids)
        assert reloaded.keys() == expected_keys

        # Spot-check 200 random base IDs not in update_map
        unchanged = [i for i in rng.choice(n_base, 300, replace=False).tolist()
                     if i not in update_map][:200]
        for pid in unchanged:
            _assert_tags_equal(reloaded[pid], base[pid])

        # Updated entries reflect delta
        for pid, tags in update_map.items():
            _assert_tags_equal(reloaded[pid], tags)

        # New entries present
        for pid, tags in new_map.items():
            _assert_tags_equal(reloaded[pid], tags)
