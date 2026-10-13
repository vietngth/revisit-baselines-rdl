import numpy as np
import pytest

from rbl import label_graphs, labels, projection
from tests.toy import Task, day, toy_database


def seconds(offset):
    return labels.seconds(np.array([day(offset)]))[0]


def build(variant, split="test", history_splits=("train", "val"), **options):
    task = Task()
    rows = labels.split_rows(task)
    history = labels.label_history(task, rows, history_splits)
    links = projection.build(toy_database(), "users")
    return label_graphs.build(variant, rows[split], history, links, **options), rows, task


def subgraph(graph, row):
    """Edges {(node, node): weight}, labels and positions of one prediction row, with nodes
    numbered from its seed."""
    block = graph.blocks[0]
    nodes = np.flatnonzero(block.owners == row)
    matrix = block.adjacency[nodes][:, nodes].tocoo()
    edges = {(int(u), int(v)): float(w) for u, v, w in zip(matrix.row, matrix.col, matrix.data)
             if u < v}
    return edges, block.labels[nodes], block.positions[nodes]


def labels_of(task, user, splits=("train", "val")):
    values = []
    for split in splits:
        frame = task.get_table(split).df
        values += frame[frame["user"] == user]["label"].tolist()
    return values


def test_every_row_has_an_unlabelled_seed_holding_its_own_prediction():
    for variant in label_graphs.VARIANTS:
        graph, rows, _ = build(variant)
        block = graph.blocks[0]
        assert np.isnan(block.labels[block.seeds]).all()
        np.testing.assert_array_equal(block.positions[block.seeds], rows["test"].positions)
        np.testing.assert_array_equal(block.owners[block.seeds], np.arange(4))
        assert (np.diff(block.owners) >= 0).all()


def test_self_star_weights_decay_from_the_most_recent_label():
    graph, _, task = build("self")
    edges, values, _ = subgraph(graph, 0)
    history = labels_of(task, 0)
    assert values[1:].tolist() == history[::-1]
    weights = [edges[0, node] for node in range(1, len(values))]
    np.testing.assert_allclose(weights, 0.7 ** np.arange(len(history)), rtol=1e-6)


def test_self_chain_links_labels_in_reverse_time_order():
    graph, _, task = build("self-chain")
    edges, values, _ = subgraph(graph, 0)
    history = labels_of(task, 0)
    assert values[1:].tolist() == history[::-1]
    assert edges == {(node, node + 1): 1.0 for node in range(len(history))}


def test_label_nodes_hold_the_prediction_of_the_row_that_was_labelled():
    graph, rows, _ = build("self-chain")
    _, _, positions = subgraph(graph, 2)
    stacked = {name: np.concatenate([getattr(rows[split], name) for split in labels.SPLITS])
               for name in ("entities", "times")}
    assert set(stacked["entities"][positions]) == {2}
    assert stacked["times"][positions].tolist() == [seconds(d) for d in (36, 28, 21, 14, 7)]


def test_cross_attaches_the_latest_label_of_each_neighbour_with_its_link_weight():
    graph, _, task = build("cross")
    edges, values, _ = subgraph(graph, 0)
    assert values[1:].tolist() == [labels_of(task, user)[-1] for user in (1, 2, 3)]
    assert edges == {(0, 1): 2.0, (0, 2): 0.5, (0, 3): 1.5}


def test_cross_self_adds_own_labels_after_the_neighbours():
    graph, _, task = build("cross+self")
    edges, values, _ = subgraph(graph, 0)
    own = labels_of(task, 0)
    assert values[4:].tolist() == own
    assert all(edges[0, node] == 1.0 for node in range(4, 4 + len(own)))


def test_cross_hist_hangs_three_labels_off_an_unlabelled_relay():
    graph, _, task = build("cross+hist")
    edges, values, _ = subgraph(graph, 0)
    assert np.isnan(values[1]) and values[2:5].tolist() == labels_of(task, 1)[-3:]
    assert edges[0, 1] == 2.0 and all(edges[1, leaf] == 1.0 for leaf in (2, 3, 4))
    assert (0, 2) not in edges


def test_all_adds_one_relay_per_own_label():
    graph, _, task = build("all")
    _, values, _ = subgraph(graph, 0)
    own = labels_of(task, 0)
    relays = np.flatnonzero(np.isnan(values))[1:]
    assert len(relays) == 3 + len(own)
    assert values[relays[-1] + 1:].tolist() == own[-3:]


def test_chain_hist_3_keeps_three_labels_per_chain():
    full, _, task = build("chain+hist")
    capped, _, _ = build("chain+hist-3")
    assert len(subgraph(full, 0)[1]) == 1 + sum(len(labels_of(task, user)) for user in range(4))
    edges, values, _ = subgraph(capped, 0)
    assert len(values) == 1 + 3 * 4
    assert values[1:4].tolist() == labels_of(task, 0)[::-1][:3]
    assert edges[0, 4] == 2.0 and edges[4, 5] == 1.0


def test_chain_cross_keeps_the_own_chain_and_one_label_per_neighbour():
    graph, _, task = build("chain+cross")
    edges, values, _ = subgraph(graph, 0)
    own = len(labels_of(task, 0))
    assert len(values) == 1 + own + 3
    assert [edges[0, own + offset] for offset in (1, 2, 3)] == [2.0, 0.5, 1.5]


def test_labels_whose_window_is_still_open_are_not_in_the_graph():
    graph, _, task = build("chain+hist", split="val", history_splits=("train",))
    frame = task.get_table("train").df
    closed = frame[frame["time"] < day(21)]
    _, values, _ = subgraph(graph, 0)
    assert len(values) - 1 == len(closed)
    assert sorted(values[1:].tolist()) == sorted(closed["label"].tolist())


def test_variants_without_neighbours_need_no_projection():
    task = Task()
    rows = labels.split_rows(task)
    history = labels.label_history(task, rows, ("train",))
    for variant in label_graphs.VARIANTS:
        if label_graphs.uses_neighbours(variant):
            with pytest.raises(AttributeError):
                label_graphs.build(variant, rows["val"], history)
        else:
            label_graphs.build(variant, rows["val"], history)


def test_rows_are_split_into_blocks():
    whole, _, _ = build("chain+hist")
    split, _, _ = build("chain+hist", block_nodes=1)
    assert len(whole.blocks) == 1 and len(split.blocks) == 4
    assert split.num_nodes == whole.num_nodes
    np.testing.assert_array_equal(np.concatenate([block.labels for block in split.blocks]),
                                  whole.blocks[0].labels)
