"""A relational database collapsed into one homogeneous graph with block-structured features."""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

SECONDS_PER_YEAR = 31_557_600.0
MAX_AGE_YEARS = 50.0
TIMELESS = np.iinfo(np.int64).min


@dataclass
class CollapsedGraph:
    """All rows of all tables as one node set; each table's features occupy their own columns.

    `features[t]` holds the normalized feature block of node type `t`, `times[t]` its row
    timestamps in seconds (None for static tables), and `src`/`dst` the primary-foreign key
    edges in global node indices.
    """

    node_types: list
    features: dict
    times: dict
    src: np.ndarray
    dst: np.ndarray
    offsets: dict = field(init=False)
    columns: dict = field(init=False)

    def __post_init__(self):
        self.offsets, self.columns = {}, {}
        node, column = 0, 0
        for node_type in self.node_types:
            rows, width = self.features[node_type].shape
            self.offsets[node_type] = node
            self.columns[node_type] = (column, width)
            node += rows
            column += width
        self.num_nodes = node
        self.feature_dim = column
        self.node_times = np.full(node, TIMELESS, np.int64)
        for node_type, stamps in self.times.items():
            if stamps is not None:
                start = self.offsets[node_type]
                self.node_times[start:start + len(stamps)] = stamps

    def visible(self, tau):
        """Nodes a prediction at time `tau` may use: static rows and rows stamped at or before
        it."""
        return self.node_times <= tau

    def adjacency(self, visible):
        """Symmetric adjacency restricted to edges between visible nodes."""
        keep = visible[self.src] & visible[self.dst]
        rows, cols = self.src[keep], self.dst[keep]
        adj = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                            shape=(self.num_nodes, self.num_nodes)).tocsr()
        return adj.maximum(adj.T)

    def ages(self, tau):
        """Age of every stamped node at `tau` in years, clipped to [0, 50]; static nodes get 0."""
        ages = np.zeros(self.num_nodes, np.float32)
        stamped = self.node_times != TIMELESS
        ages[stamped] = np.clip((tau - self.node_times[stamped]) / SECONDS_PER_YEAR,
                                0, MAX_AGE_YEARS)
        return ages

    def column_slab(self, start, stop):
        """Dense feature columns [start, stop) over all nodes, built from the per-type blocks."""
        slab = np.zeros((self.num_nodes, stop - start), np.float32)
        for node_type, (first, width) in self.columns.items():
            lo, hi = max(first, start), min(first + width, stop)
            if lo >= hi:
                continue
            rows = slice(self.offsets[node_type],
                         self.offsets[node_type] + self.features[node_type].shape[0])
            slab[rows, lo - start:hi - start] = self.features[node_type][:, lo - first:hi - first]
        return slab
