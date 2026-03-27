import numpy as np
import pytest

from recommender.model.embeddings import EmbeddingTable, apply_event_batch


def test_get_or_init_returns_same_object():
    table = EmbeddingTable(dim=4)
    v1 = table.get_or_init(42)
    v2 = table.get_or_init(42)
    assert v1 is v2


def test_contains():
    table = EmbeddingTable(dim=4)
    assert 1 not in table
    table.get_or_init(1)
    assert 1 in table


def test_empty_table_to_arrays():
    table = EmbeddingTable(dim=8)
    ids, matrix = table.to_arrays()
    assert ids.shape == (0,)
    assert matrix.shape == (0, 8)


def test_roundtrip_preserves_values():
    table = EmbeddingTable(dim=4, rng=np.random.default_rng(0))
    table.get_or_init(10)
    table.get_or_init(20)
    table.get_or_init(30)
    ids, matrix = table.to_arrays()
    restored = EmbeddingTable.from_arrays(ids, matrix)
    for entity_id in (10, 20, 30):
        np.testing.assert_array_equal(
            table.get_or_init(entity_id),
            restored.get_or_init(entity_id),
        )


def test_apply_event_favorite_direction():
    """action=+1 moves user toward post and post toward user."""
    user_table = EmbeddingTable(dim=2)
    post_table = EmbeddingTable(dim=2)
    user_table.get_or_init(1)[:] = [1.0, 0.0]
    post_table.get_or_init(100)[:] = [0.0, 1.0]

    apply_event_batch(user_table, post_table, {}, [(1, 100, 1)], lr=0.1, reg=0.0)

    # u_new = [1,0] + 0.1*[0,1] = [1.0, 0.1]
    # p_new = [0,1] + 0.1*[1,0] = [0.1, 1.0]
    np.testing.assert_allclose(user_table.get_or_init(1), [1.0, 0.1], atol=1e-6)
    np.testing.assert_allclose(post_table.get_or_init(100), [0.1, 1.0], atol=1e-6)


def test_apply_event_unfavorite_direction():
    """action=-1 moves user away from post and post away from user."""
    user_table = EmbeddingTable(dim=2)
    post_table = EmbeddingTable(dim=2)
    user_table.get_or_init(1)[:] = [1.0, 0.0]
    post_table.get_or_init(100)[:] = [0.0, 1.0]

    apply_event_batch(user_table, post_table, {100: 2}, [(1, 100, -1)], lr=0.1, reg=0.0)

    # u_new = [1,0] + 0.1*(-1)*[0,1] = [1.0, -0.1]
    # p_new = [0,1] + 0.1*(-1)*[1,0] = [-0.1, 1.0]
    np.testing.assert_allclose(user_table.get_or_init(1), [1.0, -0.1], atol=1e-6)
    np.testing.assert_allclose(post_table.get_or_init(100), [-0.1, 1.0], atol=1e-6)


def test_apply_event_uses_old_vectors():
    """Both u and p must be updated using their pre-update copies."""
    user_table = EmbeddingTable(dim=2)
    post_table = EmbeddingTable(dim=2)
    user_table.get_or_init(1)[:] = [1.0, 0.0]
    post_table.get_or_init(100)[:] = [0.0, 1.0]

    # lr=1.0 makes the difference obvious if new values were used instead of old
    apply_event_batch(user_table, post_table, {}, [(1, 100, 1)], lr=1.0, reg=0.0)

    # Correct (old copies): u_new=[1,1], p_new=[1,1]
    # Wrong (if p was updated first and u used p_new): u_new=[1,1], p_new=[1+1,1]=[2,1]
    np.testing.assert_allclose(user_table.get_or_init(1), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(post_table.get_or_init(100), [1.0, 1.0], atol=1e-6)


def test_fav_count_incremented_on_favorite():
    u, p, fc = EmbeddingTable(dim=2), EmbeddingTable(dim=2), {}
    apply_event_batch(u, p, fc, [(1, 100, 1)], lr=0.01, reg=0.001)
    assert fc[100] == 1


def test_fav_count_clamped_at_zero():
    u, p = EmbeddingTable(dim=2), EmbeddingTable(dim=2)
    fc = {100: 0}
    apply_event_batch(u, p, fc, [(1, 100, -1)], lr=0.01, reg=0.001)
    assert fc[100] == 0  # must not go negative


def test_regularization_shrinks_norm():
    """reg > 0 should reduce the norm of the user vector."""
    user_table = EmbeddingTable(dim=2)
    post_table = EmbeddingTable(dim=2)
    user_table.get_or_init(1)[:] = [10.0, 10.0]
    post_table.get_or_init(100)[:] = [0.0, 0.0]

    norm_before = np.linalg.norm(user_table.get_or_init(1).copy())
    apply_event_batch(user_table, post_table, {}, [(1, 100, 1)], lr=0.1, reg=0.5)
    assert np.linalg.norm(user_table.get_or_init(1)) < norm_before
