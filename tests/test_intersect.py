import pytest

from recommender.api.engine import _intersect_top


def test_disjoint_lists():
    a = [(1, 1.0), (3, 0.5)]
    b = [(2, 1.0), (4, 0.5)]
    assert _intersect_top(a, b, m=6) == []


def test_full_overlap_uses_min_weight():
    a = [(1, 0.8), (2, 1.2)]
    b = [(1, 0.6), (2, 1.5)]
    result = _intersect_top(a, b, m=6)
    assert len(result) == 2
    by_id = {tid: w for tid, w in result}
    assert by_id[1] == pytest.approx(0.6)   # min(0.8, 0.6)
    assert by_id[2] == pytest.approx(1.2)   # min(1.2, 1.5)


def test_partial_overlap_correct_ids():
    a = [(1, 1.0), (2, 0.5), (5, 0.8)]
    b = [(2, 0.7), (3, 1.0), (5, 0.9)]
    result = _intersect_top(a, b, m=6)
    assert {tid for tid, _ in result} == {2, 5}


def test_result_capped_at_m():
    a = [(i, 1.0) for i in range(10)]
    b = [(i, 1.0) for i in range(10)]
    assert len(_intersect_top(a, b, m=3)) == 3


def test_result_ordered_by_contribution_descending():
    a = [(1, 0.1), (2, 0.5), (3, 0.9)]
    b = [(1, 0.2), (2, 0.4), (3, 0.8)]
    result = _intersect_top(a, b, m=6)
    weights = [w for _, w in result]
    assert weights == sorted(weights, reverse=True)


def test_empty_a():
    assert _intersect_top([], [(1, 1.0)], m=6) == []


def test_empty_b():
    assert _intersect_top([(1, 1.0)], [], m=6) == []


def test_both_empty():
    assert _intersect_top([], [], m=6) == []


def test_single_element_overlap():
    result = _intersect_top([(7, 0.5)], [(7, 0.5)], m=6)
    assert len(result) == 1
    assert result[0][0] == 7
    assert result[0][1] == pytest.approx(0.5)


def test_single_element_no_overlap():
    assert _intersect_top([(1, 0.5)], [(2, 0.5)], m=6) == []


def test_m_zero_returns_empty():
    a = [(1, 1.0), (2, 0.5)]
    b = [(1, 1.0), (2, 0.5)]
    assert _intersect_top(a, b, m=0) == []
