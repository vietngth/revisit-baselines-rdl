"""Correct and Smooth (Huang et al., 2021) on graphs of historical labels.

A port of the reference implementation, https://github.com/CUAI/CorrectAndSmooth
(outcome_correlation.py): Correct diffuses the residuals of the labelled nodes and adds them to
the base predictions, Smooth diffuses the corrected predictions with the labels put in place.
All seeds of a base model share one graph, so their predictions are propagated together.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property, partial

import numpy as np
import scipy.sparse as sp

STEPS = 50
ELEMENTS_PER_BLOCK = 131_072
NORMALIZATIONS = ("DA", "AD", "DAD")
SCALINGS = ("none", "fdiff", "autoscale")
SCALE_LIMIT = 1000.0
ISOLATED_DEGREE = 1e-12


@dataclass(frozen=True)
class Setting:
    """Hyperparameters of one run.

    `alpha1` and `norm1` drive Correct, `alpha2` and `norm2` Smooth; an alpha of zero turns its
    step off. `scaling` picks how the diffused residual is scaled: "autoscale" and "fdiff"
    (with the factor `scale`) as in the reference, or "none".
    """

    scaling: str = "none"
    alpha1: float = 0.0
    alpha2: float = 0.0
    norm1: str = "DA"
    norm2: str = "DA"
    scale: float = 1.0


@dataclass
class Block:
    """Subgraphs of consecutive prediction rows, stored as one graph.

    `adjacency` is symmetric with float32 weights. `labels[node]` is NaN for unlabelled nodes,
    `positions[node]` the row whose base prediction the node holds, `owners[node]` the
    prediction row (within the block) the node belongs to and `seeds[row]` the node of each
    prediction row. The nodes of a subgraph are contiguous and start with its seed.
    """

    adjacency: sp.csr_matrix
    labels: np.ndarray
    positions: np.ndarray
    owners: np.ndarray
    seeds: np.ndarray
    _operators: dict = field(default_factory=dict, repr=False)

    @cached_property
    def labelled(self):
        return ~np.isnan(self.labels)

    @cached_property
    def degree(self):
        return self.adjacency @ np.ones(len(self.labels), np.float32)

    @cached_property
    def members(self):
        """Indicator matrix (prediction rows x nodes) of the subgraphs."""
        nodes = len(self.labels)
        return sp.csr_matrix((np.ones(nodes, np.float32), (self.owners, np.arange(nodes))),
                             shape=(len(self.seeds), nodes))

    @cached_property
    def label_counts(self):
        """Labelled nodes per subgraph, at least one."""
        counts = np.bincount(self.owners[self.labelled], minlength=len(self.seeds))
        return np.maximum(counts, 1).astype(np.float32)

    def operator(self, norm):
        """Return the adjacency normalized as D^-1 A ("DA"), A D^-1 ("AD") or
        D^-1/2 A D^-1/2 ("DAD")."""
        if norm not in self._operators:
            with np.errstate(divide="ignore"):
                inverse = np.where(self.degree == 0, 0, 1 / self.degree).astype(np.float32)
                root = np.where(self.degree == 0, 0, 1 / np.sqrt(self.degree)).astype(np.float32)
            ones = np.ones_like(self.degree)
            left, right = {"DA": (inverse, ones), "AD": (ones, inverse), "DAD": (root, root)}[norm]
            matrix = self.adjacency
            rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
            data = left[rows] * matrix.data * right[matrix.indices]
            self._operators[norm] = sp.csr_matrix((data, matrix.indices, matrix.indptr),
                                                  shape=matrix.shape)
        return self._operators[norm]


@dataclass(frozen=True)
class LabelGraph:
    blocks: list

    @property
    def num_nodes(self):
        return sum(len(block.labels) for block in self.blocks)


def correct_and_smooth(graph, base, setting, bounds, binary, threads=None):
    """Return the refined predictions of the graph's rows, one column per seed.

    Args:
        graph: LabelGraph of the rows to predict.
        base: float32 base predictions, (rows of all splits) x (seeds).
        setting: the Setting to apply.
        bounds: (low, high) range the predictions are clipped to.
        binary: True for binary classification, whose residuals lie in [-1, 1].
        threads: blocks processed at a time; the available cores by default.
    """
    refine = partial(_refine, base=base, setting=setting, binary=binary,
                     bounds=(np.float32(bounds[0]), np.float32(bounds[1])))
    with ThreadPoolExecutor(threads or available_cores()) as pool:
        return np.concatenate(list(pool.map(refine, graph.blocks)))


def block_nodes(seeds):
    """Return the block size that keeps a thread's working arrays in cache when `seeds` seeds are
    propagated together."""
    return ELEMENTS_PER_BLOCK // seeds


def available_cores():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def diffuse(block, norm, start, alpha, post):
    """Return x after STEPS updates x <- post(alpha * S x + (1 - alpha) * start), S being the
    block's operator under `norm`; `post` changes its argument in place."""
    if alpha == 0:
        state = start.copy()
        post(state)
        return state
    operator = block.operator(norm)
    weight, restart = np.float32(alpha), np.float32(1.0 - alpha) * start
    state = start
    for _ in range(STEPS):
        state = operator @ state
        state *= weight
        state += restart
        post(state)
    return state


def _refine(block, base, setting, bounds, binary):
    start = base[block.positions]
    targets = block.labels[block.labelled][:, None]
    state = _correct(block, start, targets, setting, binary) if setting.alpha1 else start.copy()
    state[block.labelled] = targets
    smoothed = diffuse(block, setting.norm2, state, setting.alpha2,
                       lambda x: np.clip(x, *bounds, out=x))
    isolated = block.degree[block.seeds] < ISOLATED_DEGREE
    return np.where(isolated[:, None], start[block.seeds], smoothed[block.seeds])


def _correct(block, start, targets, setting, binary):
    residual = np.zeros_like(start)
    residual[block.labelled] = targets - start[block.labelled]
    if setting.scaling == "fdiff":
        pinned = residual[block.labelled]

        def pin(state):
            state[block.labelled] = pinned
        correction = diffuse(block, setting.norm1, residual, setting.alpha1, pin)
        correction *= np.float32(setting.scale)
    else:
        correction = diffuse(block, setting.norm1, residual, setting.alpha1,
                             _clip_residual if binary else _keep)
        if binary and setting.scaling == "autoscale":
            correction *= _autoscale(block, residual, correction)
    corrected = start + correction
    np.copyto(corrected, start, where=np.isnan(corrected))
    return corrected


def _clip_residual(state):
    np.clip(state, -1.0, 1.0, out=state)


def _keep(state):
    pass


def _autoscale(block, residual, diffused):
    """Return the factor that brings each node's diffused residual to the mean absolute residual
    of the labels in its subgraph; factors above SCALE_LIMIT are replaced by one."""
    mean = (block.members @ np.abs(residual)) / block.label_counts[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = mean[block.owners] / np.abs(diffused)
    scale[scale > SCALE_LIMIT] = 1.0
    scale[np.isnan(scale)] = 0.0
    return scale
