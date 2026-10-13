import numpy as np
import scipy.sparse as sp

from rbl import filtering, snapshots
from tests.toy import toy_graph, toy_rows


def serial_reference(graph, rows, k):
    out = {}
    for split in snapshots.SPLITS:
        matrix = np.zeros((len(rows.nodes[split]), graph.feature_dim + 2), np.float32)
        for index, (node, tau) in enumerate(zip(rows.nodes[split], rows.seed_times[split])):
            if node != snapshots.MISSING:
                matrix[index] = filtering.filter_rows(graph, [node], tau, k)[0]
        out[split] = matrix
    return out


def test_plan_covers_every_snapshot_column_once(monkeypatch):
    monkeypatch.setattr(snapshots, "BUILD_COST_PER_EDGE", 0.0)
    graph, rows = toy_graph(), toy_rows()
    width = graph.feature_dim + 2
    for num_shards in (1, 2, 3, 8):
        units = [u for shard in snapshots.plan(graph, rows, 2, num_shards) for u in shard]
        for tau in rows.taus():
            ranges = sorted(u.columns(width) for u in units if u.tau == tau)
            assert ranges[0][0] == 0 and ranges[-1][1] == width
            assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))


def test_many_shards_split_snapshots(monkeypatch):
    monkeypatch.setattr(snapshots, "BUILD_COST_PER_EDGE", 0.0)
    graph, rows = toy_graph(), toy_rows()
    units = [u for shard in snapshots.plan(graph, rows, 2, 8) for u in shard]
    assert any(u.parts > 1 for u in units)


def test_sharded_run_equals_serial_filtering(tmp_path, monkeypatch):
    graph, rows = toy_graph(), toy_rows()
    expected = serial_reference(graph, rows, 2)
    single = None
    for build_cost, num_shards in ((200.0, 1), (200.0, 3), (0.0, 3), (0.0, 8)):
        monkeypatch.setattr(snapshots, "BUILD_COST_PER_EDGE", build_cost)
        out_dir = tmp_path / f"shards{num_shards}_build{build_cost}"
        for shard in range(num_shards):
            snapshots.run_shard(graph, rows, out_dir, 2, shard, num_shards)
        assembled = snapshots.assemble(out_dir)
        single = single or assembled
        for split in snapshots.SPLITS:
            np.testing.assert_allclose(assembled[split], expected[split], atol=1e-6)
            np.testing.assert_array_equal(assembled[split], single[split])


def test_assemble_rejects_unfinished_work(tmp_path):
    graph, rows = toy_graph(), toy_rows()
    snapshots.run_shard(graph, rows, tmp_path, 2, 0, 3)
    try:
        snapshots.assemble(tmp_path)
    except RuntimeError as error:
        assert "unfinished" in str(error)
    else:
        raise AssertionError("assemble accepted missing rows")


def test_missing_entities_get_zero_rows(tmp_path):
    graph, rows = toy_graph(), toy_rows()
    snapshots.run_shard(graph, rows, tmp_path, 2, 0, 1)
    np.testing.assert_array_equal(snapshots.assemble(tmp_path)["train"][5], 0)


def test_all_time_row_costs_match_the_filter_on_the_full_graph():
    graph = toy_graph()
    operator = sp.csr_matrix(graph.adjacency(graph.visible(10**9)) + sp.eye(graph.num_nodes))
    rows = np.arange(graph.num_nodes)
    for k in (1, 2):
        expected = filtering._row_costs(operator, rows, k)
        np.testing.assert_allclose(snapshots._all_time_row_costs(graph, k), expected)
