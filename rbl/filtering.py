"""Fixed low-pass filtering of one temporal snapshot, read out at selected nodes."""
import numpy as np

from rbl import normalize

NNZ_PER_SLICE = 1.5e8
COLUMNS_PER_CHUNK = 256
REACH_SAMPLE = 1024
FULL_COST_PER_NNZ = 16.0


def filter_rows(graph, rows, tau, k, columns=None):
    """Return columns [start, stop) of S_tau^k [X | log1p(deg) | age] at the global node indices
    `rows`; `columns` defaults to all of them.

    Only nodes visible at `tau` take part, so no later row can reach a prediction. Rows that
    are themselves not visible at `tau` are returned as zeros. Without filtering (k = 0) the
    degree and age columns are kept but set to zero. Every output column is computed
    independently, so any column range gives exactly the corresponding slice of the full result.
    """
    rows = np.asarray(rows, np.int64)
    start, stop = columns or (0, graph.feature_dim + 2)
    visible = graph.visible(tau)
    adjacency = graph.adjacency(visible)
    operator = normalize.aug_normalized_adjacency(adjacency)
    extra = _structural_columns(graph, adjacency, tau, k)
    cost = _row_costs(operator, rows, k)
    full_work = k * operator.nnz
    if (cost is not None and cost.sum() <= full_work
            and _reach_nnz(operator, rows, k) <= FULL_COST_PER_NNZ * full_work):
        features = _propagate_rows(graph, operator, extra, rows, k, cost, start, stop)
    else:
        features = _propagate_columns(graph, operator, extra, rows, k, start, stop)
    features[~visible[rows]] = 0.0
    return features


def _structural_columns(graph, adjacency, tau, k):
    if k == 0:
        return np.zeros((graph.num_nodes, 2), np.float32)
    degree = np.asarray(adjacency.sum(1)).ravel()
    return np.stack([np.log1p(degree), graph.ages(tau)], 1).astype(np.float32)


def _row_costs(operator, rows, k):
    """Nonzeros touched per row by the row-restricted product; None when it does not apply."""
    if not 1 <= k <= 2:
        return None
    row_nnz = np.diff(operator.indptr).astype(np.float64)
    if k == 1:
        return row_nnz[rows]
    return np.asarray(operator[rows] @ row_nnz).ravel()


def _reach_nnz(operator, rows, k):
    """Estimated nonzeros of S[rows] S^(k-1), the matrix the row-restricted product multiplies the
    features by: counted on an evenly spaced sample of the rows and scaled to all of them. Full
    propagation multiplies them by k * nnz(S) nonzeros but writes to every node, which was measured
    to cost from 4 to more than 11 times as much per nonzero (FULL_COST_PER_NNZ).
    """
    if len(rows) > REACH_SAMPLE:
        sample = rows[np.linspace(0, len(rows) - 1, REACH_SAMPLE).astype(np.int64)]
    else:
        sample = rows
    reach = operator[sample, :]
    for _ in range(k - 1):
        reach = reach @ operator
    return reach.nnz * len(rows) / max(len(sample), 1)


def _propagate_rows(graph, operator, extra, rows, k, cost, start, stop):
    """Compute only the needed rows: (S^k)[rows] = S[rows] S^(k-1), in slices of bounded cost."""
    features = np.zeros((len(rows), stop - start), np.float32)
    blocks = _blocks(graph, extra, start, stop)
    for first, last in _slices(cost, NNZ_PER_SLICE):
        reach = operator[rows[first:last], :]
        for _ in range(k - 1):
            reach = reach @ operator
        for column, nodes, block in blocks:
            features[first:last, column:column + block.shape[1]] = reach[:, nodes] @ block
    return features


def _blocks(graph, extra, start, stop):
    """(output column, node range, features) of every feature block overlapping [start, stop)."""
    pieces = [(first, graph.offsets[t], graph.features[t])
              for t, (first, width) in graph.columns.items() if width]
    pieces.append((graph.feature_dim, 0, extra))
    blocks = []
    for first, offset, block in pieces:
        lo, hi = max(first, start), min(first + block.shape[1], stop)
        if lo < hi:
            blocks.append((lo - start, slice(offset, offset + block.shape[0]),
                           np.ascontiguousarray(block[:, lo - first:hi - first])))
    return blocks


def _propagate_columns(graph, operator, extra, rows, k, start, stop):
    """Propagate all nodes, a chunk of columns at a time, and read out the needed rows."""
    features = np.zeros((len(rows), stop - start), np.float32)
    for first in range(start, stop, COLUMNS_PER_CHUNK):
        last = min(first + COLUMNS_PER_CHUNK, stop)
        chunk = _columns(graph, extra, first, last)
        for _ in range(k):
            chunk = operator @ chunk
        features[:, first - start:last - start] = chunk[rows]
    return features


def _columns(graph, extra, start, stop):
    base = graph.feature_dim
    if start >= base:
        return extra[:, start - base:stop - base].copy()
    if stop <= base:
        return graph.column_slab(start, stop)
    return np.hstack([graph.column_slab(start, base), extra[:, :stop - base]])


def _slices(cost, budget):
    """Split consecutive rows into slices whose summed cost stays within `budget`."""
    bounds, total, start = [], 0.0, 0
    for index, value in enumerate(cost):
        if index > start and total + value > budget:
            bounds.append((start, index))
            start, total = index, 0.0
        total += value
    bounds.append((start, len(cost)))
    return bounds
