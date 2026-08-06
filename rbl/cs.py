"""Correct-and-Smooth over a meta-graph on the target entities (Huang et al., 2021). Adapted from their code base.

The meta-graph is not given by the relational entity graph: labels exist only on the rows of
the task table. We build it per prediction row from labels observable strictly before the seed
time, in the variants listed in VARIANTS.
"""
import numpy as np


VARIANTS = {
    # name:            (own labels, cross-entity labels, neighbour history, chain own labels)
    "self":            dict(own=True,  cross=False, hist=0, chain=False),
    "cross":           dict(own=False, cross=True,  hist=0, chain=False),
    "cross+self":      dict(own=True,  cross=True,  hist=0, chain=False),
    "cross+hist":      dict(own=False, cross=True,  hist=3, chain=False),
    "all":             dict(own=True,  cross=True,  hist=3, chain=False),
    "self-chain":      dict(own=True,  cross=False, hist=0, chain=True),
    "chain+cross":     dict(own=True,  cross=True,  hist=0, chain=True),
    "chain+hist":      dict(own=True,  cross=True,  hist=3, chain=True),
}


def build_meta_graph(entity, seed_time, own_labels, related, variant, ewma=0.3, cap=0):
    """Return (nodes, edges, values, labels, seed_index) for one prediction row.

    own_labels: [(label, observed_time)] of the target entity, newest first.
    related:    [(weight, [(label, observed_time)])] per co-occurring entity, newest first.
    Only labels with observed_time < seed_time may be passed in.
    """
    cfg = VARIANTS[variant]
    # idx = 0: unlabeled seed
    values, labels, edges = [0.0], [np.nan], []         
    prev = 0
    if cfg["own"]:
        own = own_labels[:cap] if cap else own_labels
        for rank, (y, _t) in enumerate(own):
            idx = len(values)
            values.append(0.0)
            labels.append(y)
            if cfg["chain"]:
                 # temporal path: recency via hops
                edges.append((prev, idx, 1.0))           
                prev = idx
            else:
                # star: recency via edge weight
                edges.append((0, idx, (1 - ewma) ** rank))  
    if cfg["cross"] or cfg["hist"]:
        for w, hist in related:
            if not hist:
                continue
            take = hist[: cfg["hist"]] if cfg["hist"] else hist[:1]
            anchor = 0
            for k, (y, _t) in enumerate(take):
                idx = len(values)
                values.append(0.0)
                labels.append(y)
                edges.append((anchor, idx, w if k == 0 else 1.0))
                if cfg["hist"]:
                    anchor = idx
    return values, labels, edges, 0


def correct_and_smooth(z, y, edges, n, alpha1, alpha2, n_prop=50, clamp=(0.0, 1.0)):
    """Two parameter-free stages over one meta-graph; z base predictions, y labels (nan = none)."""
    S = _normalize(edges, n)
    labeled = ~np.isnan(y)
    e = np.zeros(n, np.float64)
    e[labeled] = y[labeled] - z[labeled]
    for _ in range(n_prop):
        e = (1 - alpha1) * e + alpha1 * (S @ e)
        e = np.clip(e, -1.0, 1.0)
    zc = z + e
    g = zc.copy()
    g[labeled] = y[labeled]
    for _ in range(n_prop):
        g = (1 - alpha2) * g + alpha2 * (S @ g)
        g[labeled] = y[labeled]
        g = np.clip(g, *clamp)
    return g


def _normalize(edges, n):
    A = np.zeros((n, n), np.float64)
    for u, v, w in edges:
        A[u, v] = w
        A[v, u] = w
    d = A.sum(1)
    d[d == 0] = 1.0
    return A / d[:, None]
