import numpy as np
import scipy.sparse as sp

from rbl import filtering, normalize
from tests.toy import toy_graph


def dense_reference(graph, tau, k):
    visible = graph.visible(tau)
    adjacency = graph.adjacency(visible).toarray()
    augmented = adjacency + np.eye(len(adjacency))
    scale = 1 / np.sqrt(augmented.sum(1))
    operator = scale[:, None] * augmented * scale[None, :]
    structural = np.stack([np.log1p(adjacency.sum(1)), graph.ages(tau)], 1) * (k > 0)
    features = np.hstack([graph.column_slab(0, graph.feature_dim), structural])
    out = np.linalg.matrix_power(operator, k) @ features
    out[~visible] = 0
    return out


def test_aug_normalized_adjacency_matches_formula():
    adjacency = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32))
    augmented = adjacency.toarray() + np.eye(3)
    scale = 1 / np.sqrt(augmented.sum(1))
    expected = scale[:, None] * augmented * scale[None, :]
    np.testing.assert_allclose(normalize.aug_normalized_adjacency(adjacency).toarray(), expected)


def test_absnorm_rows_have_unit_absolute_sum():
    features = np.array([[1.0, -3.0], [0.0, 0.0]], np.float32)
    out = normalize.absnorm_normalize(features)
    np.testing.assert_allclose(np.abs(out).sum(1), [1.0, 0.0])


def test_row_and_column_paths_match_dense_reference():
    graph = toy_graph()
    rows = np.arange(graph.num_nodes)
    for k in (0, 1, 2, 3):
        for tau in (0, 55, 200):
            np.testing.assert_allclose(filtering.filter_rows(graph, rows, tau, k),
                                       dense_reference(graph, tau, k)[rows], atol=1e-5)


def test_rows_are_computed_independently():
    graph = toy_graph()
    everything = filtering.filter_rows(graph, np.arange(graph.num_nodes), 55, 2)
    subset = np.array([1, 4, 9])
    np.testing.assert_array_equal(filtering.filter_rows(graph, subset, 55, 2), everything[subset])


def test_column_ranges_are_exact_slices():
    graph = toy_graph()
    rows = np.array([0, 3, 7, 12, 20])
    width = graph.feature_dim + 2
    for k in (1, 2, 3):
        everything = filtering.filter_rows(graph, rows, 55, k)
        for start, stop in ((0, width), (0, 4), (2, 9), (5, width), (width - 1, width)):
            np.testing.assert_array_equal(
                filtering.filter_rows(graph, rows, 55, k, columns=(start, stop)),
                everything[:, start:stop])


def test_future_rows_are_invisible():
    graph = toy_graph()
    later_order = graph.offsets["orders"] + 11
    assert not graph.visible(50)[later_order]
    assert graph.adjacency(graph.visible(50))[later_order].nnz == 0
    np.testing.assert_array_equal(filtering.filter_rows(graph, [later_order], 50, 2), 0)


def test_unfiltered_features_are_the_visible_rows_with_zero_degree_and_age():
    graph = toy_graph()
    rows = np.arange(graph.num_nodes)
    out = filtering.filter_rows(graph, rows, 55, 0)
    expected = graph.column_slab(0, graph.feature_dim) * graph.visible(55)[:, None]
    np.testing.assert_array_equal(out[:, :-2], expected)
    np.testing.assert_array_equal(out[:, -2:], 0)


def test_reach_estimate_is_exact_when_every_row_is_sampled():
    graph = toy_graph()
    operator = normalize.aug_normalized_adjacency(graph.adjacency(graph.visible(200)))
    rows = np.arange(graph.num_nodes)
    assert filtering._reach_nnz(operator, rows, 2) == (operator @ operator).nnz
    assert filtering._reach_nnz(operator, rows, 1) == operator.nnz


def test_both_propagation_paths_give_the_same_features():
    graph = toy_graph()
    rows = np.array([0, 3, 7, 12, 20])
    width = graph.feature_dim + 2
    adjacency = graph.adjacency(graph.visible(200))
    operator = normalize.aug_normalized_adjacency(adjacency)
    extra = filtering._structural_columns(graph, adjacency, 200, 2)
    for k in (1, 2):
        cost = filtering._row_costs(operator, rows, k)
        by_rows = filtering._propagate_rows(graph, operator, extra, rows, k, cost, 0, width)
        by_columns = filtering._propagate_columns(graph, operator, extra, rows, k, 0, width)
        np.testing.assert_allclose(by_rows, by_columns, atol=1e-6)


def test_dense_reach_selects_full_propagation(monkeypatch):
    graph = toy_graph()
    rows = np.arange(graph.num_nodes)
    expected = filtering.filter_rows(graph, rows, 200, 2)
    called = []
    monkeypatch.setattr(filtering, "_reach_nnz", lambda operator, rows, k: float("inf"))
    monkeypatch.setattr(filtering, "_propagate_rows", lambda *args: called.append("rows"))
    np.testing.assert_allclose(filtering.filter_rows(graph, rows, 200, 2), expected, atol=1e-6)
    assert not called
