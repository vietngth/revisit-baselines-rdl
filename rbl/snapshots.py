"""Split a task's filtering into independent work units, compute them, and assemble the result.

A task row (entity, seed time) may only read the graph as visible at its seed time, so the
rows of one seed time form an independent snapshot. A work unit is one range of feature columns
of one snapshot; units can run in any order, in separate processes, and are skipped when their
output exists.
"""
import heapq
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rbl import filtering

SPLITS = ("train", "val", "test")
MISSING = -1
BUILD_COST_PER_EDGE = 200.0


@dataclass(frozen=True)
class TaskRows:
    """Graph node and seed time (seconds) of every row of every split; MISSING if no node."""

    nodes: dict
    seed_times: dict

    @classmethod
    def from_task(cls, task, database, graph):
        table = database.table_dict[task.entity_table]
        keys = table.df[table.pkey_col].to_numpy()
        position = pd.Series(np.arange(len(keys)), index=keys)
        position = position[~position.index.duplicated()]
        offset = graph.offsets[task.entity_table]
        nodes, seed_times = {}, {}
        for split in SPLITS:
            df = task.get_table(split).df
            local = position.reindex(df[task.entity_col].to_numpy()).to_numpy()
            nodes[split] = np.where(np.isnan(local), MISSING,
                                    offset + np.nan_to_num(local)).astype(np.int64)
            seed_times[split] = (df[task.time_col].to_numpy().astype("datetime64[s]")
                                 .astype(np.int64))
        return cls(nodes, seed_times)

    def taus(self):
        return np.unique(np.concatenate(list(self.seed_times.values())))

    def nodes_at(self, tau):
        """Sorted distinct graph nodes whose rows have seed time `tau`."""
        found = [self.nodes[s][(self.seed_times[s] == tau) & (self.nodes[s] != MISSING)]
                 for s in SPLITS]
        return np.unique(np.concatenate(found))

    def sizes(self):
        return {split: len(nodes) for split, nodes in self.nodes.items()}


@dataclass(frozen=True)
class Unit:
    tau: int
    part: int
    parts: int

    @property
    def name(self):
        return f"{self.tau}_{self.part}of{self.parts}"

    def columns(self, width):
        return width * self.part // self.parts, width * (self.part + 1) // self.parts


def plan(graph, task_rows, k, num_shards):
    """Assign work units to `num_shards` shards, balancing an estimated cost per unit.

    A snapshot costs one operator build plus the propagation of every feature column.
    Snapshots whose propagation exceeds a shard's fair share are split into column ranges,
    each repeating the build. Units go to shards longest first (LPT).
    """
    taus = task_rows.taus()
    build, propagation = _snapshot_costs(graph, task_rows, taus, k)
    width = graph.feature_dim + 2
    share = (build.sum() + propagation.sum()) / num_shards
    units, costs = [], []
    for tau, build_cost, work in zip(taus, build, propagation):
        parts = max(1, min(width, math.ceil(work / share))) if num_shards > 1 else 1
        for part in range(parts):
            units.append(Unit(int(tau), part, parts))
            costs.append(build_cost + work / parts)
    return _longest_first(units, costs, num_shards)


def _snapshot_costs(graph, task_rows, taus, k):
    """Estimated build and propagation cost of each snapshot, in multiply-adds.

    Propagation follows filter_rows: the cheaper of the row-restricted product and full
    propagation (k times the nonzeros), per feature column. Row-restricted costs are bounded
    on the all-time graph, so hub entities are priced by their real neighbourhoods. Building
    the operator was measured at about 200 multiply-adds per edge.
    """
    edge_times = np.sort(np.maximum(graph.node_times[graph.src], graph.node_times[graph.dst]))
    node_times = np.sort(graph.node_times)
    edges = np.searchsorted(edge_times, taus, side="right").astype(np.float64)
    nodes = np.searchsorted(node_times, taus, side="right").astype(np.float64)
    work = k * (edges + nodes)
    if 1 <= k <= 2:
        row_costs = _all_time_row_costs(graph, k)
        restricted = np.array([row_costs[task_rows.nodes_at(tau)].sum() for tau in taus])
        work = np.minimum(work, restricted)
    return BUILD_COST_PER_EDGE * edges, work * (graph.feature_dim + 2)


def _all_time_row_costs(graph, k):
    """Nonzeros the row-restricted product touches per node, with every edge visible."""
    degree = np.bincount(graph.src, minlength=graph.num_nodes) + 1.0
    if k == 1:
        return degree
    return np.bincount(graph.src, weights=degree[graph.dst], minlength=graph.num_nodes) + degree


def _longest_first(units, costs, num_shards):
    shards = [[] for _ in range(num_shards)]
    loads = [(0.0, shard) for shard in range(num_shards)]
    for index in sorted(range(len(units)), key=lambda i: (-costs[i], i)):
        load, shard = heapq.heappop(loads)
        shards[shard].append(units[index])
        heapq.heappush(loads, (load + costs[index], shard))
    return shards


def compute_unit(graph, task_rows, unit, k):
    """Filtered features of one unit: row indices i_<split>, features x_<split> of the unit's
    column range, and that range."""
    nodes = task_rows.nodes_at(unit.tau)
    columns = unit.columns(graph.feature_dim + 2)
    features = filtering.filter_rows(graph, nodes, unit.tau, k, columns)
    out = {"columns": np.array(columns)}
    for split in SPLITS:
        rows = np.flatnonzero(task_rows.seed_times[split] == unit.tau)
        if not len(rows):
            continue
        split_nodes = task_rows.nodes[split][rows]
        present = split_nodes != MISSING
        block = np.zeros((len(rows), features.shape[1]), np.float32)
        block[present] = features[np.searchsorted(nodes, split_nodes[present])]
        out[f"i_{split}"] = rows
        out[f"x_{split}"] = block
    return out


def run_shard(graph, task_rows, out_dir, k, shard, num_shards):
    """Compute this shard's units into `out_dir/slices`, skipping units already written."""
    out_dir = Path(out_dir)
    slices = out_dir / "slices"
    slices.mkdir(parents=True, exist_ok=True)
    _write_meta(out_dir, graph.feature_dim + 2, task_rows.sizes(), k, num_shards)
    timings = []
    for unit in plan(graph, task_rows, k, num_shards)[shard]:
        path = slices / f"{unit.name}.npz"
        if path.exists():
            continue
        start = time.perf_counter()
        _save_atomic(path, compute_unit(graph, task_rows, unit, k))
        timings.append({"unit": unit.name, "seconds": time.perf_counter() - start})
        print(f"[unit] {unit.name} {timings[-1]['seconds']:.1f}s", flush=True)
    with open(out_dir / f"timings_shard{shard}of{num_shards}.json", "w") as handle:
        json.dump(timings, handle, indent=1)


def assemble(out_dir):
    """Scatter all unit outputs into dense train/val/test matrices; fails if any cell is missing."""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    matrices = {s: np.zeros((n, meta["dim"]), np.float32) for s, n in meta["rows"].items()}
    filled = {s: np.zeros(n, np.int64) for s, n in meta["rows"].items()}
    for path in sorted((out_dir / "slices").glob("*.npz")):
        with np.load(path) as unit:
            start, stop = unit["columns"]
            for split in SPLITS:
                if f"i_{split}" in unit.files:
                    rows = unit[f"i_{split}"]
                    matrices[split][rows, start:stop] = unit[f"x_{split}"]
                    filled[split][rows] += stop - start
    missing = {s: int((f != meta["dim"]).sum()) for s, f in filled.items()
               if (f != meta["dim"]).any()}
    if missing:
        raise RuntimeError(f"{out_dir}: rows without features {missing}; some units are unfinished")
    return matrices


def _write_meta(out_dir, dim, rows, k, num_shards):
    path = out_dir / "meta.json"
    if not path.exists():
        _write_text_atomic(path, json.dumps({"dim": dim, "rows": rows, "k": k,
                                             "planned_shards": num_shards}, indent=1))


def _save_atomic(path, arrays):
    temporary = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _write_text_atomic(path, text):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)
