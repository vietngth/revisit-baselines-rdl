"""Graphs of historical labels, one small graph per prediction row.

The relational entity graph never links two labelled rows, so Correct and Smooth runs on a
graph built for the purpose. The row to predict is the unlabelled seed of its own subgraph; the
other nodes carry labels whose window closed before the seed time, taken from the seed's own
entity and from the entities it co-occurs with.

VARIANTS lists the constructions of the paper. Each is a sequence of (source, layout) parts,
added to a subgraph in that order: the source is the row's own entity or its co-occurring
entities, and the layout wires the visible labels of one such entity to the seed.
"""
import numpy as np
import scipy.sparse as sp

from rbl import cs

EWMA_DECAY = 0.3
RELAY_LABELS = 3
OWN, NEIGHBOURS = "own", "neighbours"


def chain(limit=None):
    """Return a layout that hangs the `limit` latest labels (all by default) in a path, most
    recent first; only the first edge carries the weight."""
    def layout(block, parent, weight, labels, positions):
        for label, position in zip(labels[::-1][:limit], positions[::-1][:limit]):
            parent = block.attach(parent, weight, label, position)
            weight = 1.0
    return layout


def star(block, seed, weight, labels, positions):
    """Attach every label to the seed."""
    for label, position in zip(labels, positions):
        block.attach(seed, weight, label, position)


def decayed_star(block, seed, weight, labels, positions):
    """Attach every label to the seed, most recent first, with exponentially decaying weights."""
    weights = weight * (1.0 - EWMA_DECAY) ** np.arange(len(labels))
    for decayed, label, position in zip(weights.tolist(), labels[::-1], positions[::-1]):
        block.attach(seed, decayed, label, position)


def relay(block, seed, weight, labels, positions):
    """Attach an unlabelled node that holds the base prediction of the latest label's row and
    carries the RELAY_LABELS latest labels."""
    node = block.attach(seed, weight, np.nan, positions[-1])
    star(block, node, 1.0, labels[-RELAY_LABELS:], positions[-RELAY_LABELS:])


def relay_per_label(block, seed, weight, labels, positions):
    """Attach one relay per label, each holding the base prediction of that label's row and
    carrying the RELAY_LABELS latest labels."""
    for position in positions:
        node = block.attach(seed, weight, np.nan, position)
        star(block, node, 1.0, labels[-RELAY_LABELS:], positions[-RELAY_LABELS:])


VARIANTS = {
    "self": ((OWN, decayed_star),),
    "cross": ((NEIGHBOURS, chain(1)),),
    "cross+self": ((NEIGHBOURS, chain(1)), (OWN, star)),
    "cross+hist": ((NEIGHBOURS, relay),),
    "all": ((NEIGHBOURS, relay), (OWN, relay_per_label)),
    "self-chain": ((OWN, chain()),),
    "chain+cross": ((OWN, chain()), (NEIGHBOURS, chain(1))),
    "chain+hist-3": ((OWN, chain(3)), (NEIGHBOURS, chain(3))),
    "chain+hist": ((OWN, chain()), (NEIGHBOURS, chain())),
}


def uses_neighbours(variant):
    return any(source == NEIGHBOURS for source, _ in VARIANTS[variant])


def build(variant, rows, history, projection=None, block_nodes=cs.ELEMENTS_PER_BLOCK):
    """Return the cs.LabelGraph of `rows` under one of VARIANTS.

    Args:
        variant: name of the construction.
        rows: labels.Rows to predict.
        history: labels.LabelHistory the rows may read.
        projection: projection.Projection, needed by the variants that use neighbours.
        block_nodes: nodes after which the subgraphs of the next rows start a new block.
    """
    blocks, builder = [], _BlockBuilder()
    for entity, time, position in zip(rows.entities.tolist(), rows.times.tolist(),
                                      rows.positions.tolist()):
        seed = builder.seed(position)
        for source, layout in VARIANTS[variant]:
            if source == OWN:
                layout(builder, seed, 1.0, *history.before(entity, time))
                continue
            neighbours, weights = projection.before(entity, time)
            for neighbour, weight in zip(neighbours.tolist(), weights.tolist()):
                labels, positions = history.before(neighbour, time)
                if labels:
                    layout(builder, seed, weight, labels, positions)
        if len(builder.labels) >= block_nodes:
            blocks.append(builder.finish())
            builder = _BlockBuilder()
    if builder.seeds:
        blocks.append(builder.finish())
    return cs.LabelGraph(blocks)


class _BlockBuilder:
    def __init__(self):
        self.labels, self.positions, self.owners, self.seeds = [], [], [], []
        self.sources, self.targets, self.weights = [], [], []

    def seed(self, position):
        self.seeds.append(len(self.labels))
        return self._node(np.nan, position)

    def attach(self, parent, weight, label, position):
        node = self._node(label, position)
        self.sources.append(parent)
        self.targets.append(node)
        self.weights.append(weight)
        return node

    def _node(self, label, position):
        self.labels.append(label)
        self.positions.append(position)
        self.owners.append(len(self.seeds) - 1)
        return len(self.labels) - 1

    def finish(self):
        size = len(self.labels)
        sources, targets = np.array(self.sources, np.int64), np.array(self.targets, np.int64)
        weights = np.array(self.weights, np.float32)
        adjacency = sp.csr_matrix(
            (np.concatenate([weights, weights]),
             (np.concatenate([sources, targets]), np.concatenate([targets, sources]))),
            shape=(size, size))
        adjacency.sort_indices()
        return cs.Block(adjacency, np.array(self.labels, np.float32),
                        np.array(self.positions, np.int32), np.array(self.owners, np.int32),
                        np.array(self.seeds, np.int32))
