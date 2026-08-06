"""Collapsed-graph construction and temporal snapshot filtering.

The database graph (RelBench's pkey/fkey hetero graph) is collapsed into one union adjacency
over all rows. For every distinct prediction timestamp tau the filter recomputes visibility:
a node participates iff it is timeless or its timestamp is <= tau, edges exist only between
visible nodes, and degrees are recomputed after masking. The filtered features are
S^K [X | log1p(deg) | age] read out at the task's entity rows. This is the per-timestamp
masking, the part of the pipeline that prevents future information reaches a prediction.
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.base import TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.tasks import get_task

from .sgc import (absnorm_normalize, aug_normalized_adjacency, row_normalize,
                  rw_normalized_adjacency)
from .embedder import GloveTextEmbedding

# CPU nodes must be able to read caches written on GPU nodes: force map_location when no CUDA.
if not torch.cuda.is_available():
    _torch_load_orig = torch.load
    torch.load = lambda *a, **k: _torch_load_orig(*a, **{**k, "map_location": "cpu"})


def load_graph(ds_name):
    cache = os.environ.get("RBL_CACHE", os.path.expanduser("~/.cache/relbench_examples"))
    ds = get_dataset(ds_name, download=False)
    # PORT of RelGNN examples/gnn_node.py:55-66. We previously only had the read branch, so any
    # dataset whose stypes.json had not already been written by an example script died with FileNotFoundError
    _sp = Path(f"{cache}/{ds_name}/stypes.json")
    try:
        ct = json.load(open(_sp))
        for _t, cols in ct.items():
            for c, s in cols.items():
                cols[c] = stype(s)
    except FileNotFoundError:
        from relbench.modeling.utils import get_stype_proposal
        print(f"[stypes] {_sp} missing; proposing from the database", flush=True)
        ct = get_stype_proposal(ds.get_db())
        _sp.parent.mkdir(parents=True, exist_ok=True)
        with open(_sp, "w") as _f:
            json.dump(ct, _f, indent=2, default=str)
    dev = torch.device("cpu") if os.environ.get("RBL_EMBED_CPU") == "1" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    data, _ = make_pkey_fkey_graph(
        ds.get_db(), col_to_stype_dict=ct,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=dev), batch_size=256),
        cache_dir=f"{cache}/{ds_name}/materialized")
    return ds, data


def flatten_tf(tf):
    parts = []
    fd = tf.feat_dict
    if stype.numerical in fd:
        parts.append(torch.nan_to_num(fd[stype.numerical].float()).detach().cpu())
    if stype.embedding in fd:
        e = fd[stype.embedding]
        e = e.values if hasattr(e, "values") else e
        parts.append(torch.nan_to_num(e.float().reshape(len(tf), -1)).detach().cpu())
    if not parts:
        return np.zeros((len(tf), 0), np.float32)
    return torch.cat(parts, 1).numpy().astype(np.float32)




def build_union(data, ntypes, nnodes):
    """Union node index, per-type feature blocks, per-type node times (None = timeless).

    No union feature matrix is ever materialized (it was 60 GB on amazon); the block-diagonal
    layout is realized on demand by `slab`.
    """
    offs, n = {}, 0
    for nt in ntypes:
        offs[nt] = n
        n += nnodes[nt]
    blocks, dims, c0 = {}, {}, 0
    xnorm = os.environ.get("SGC_XNORM", "sgc")
    if xnorm == "l1":
        xnorm = "absnorm"
    for nt in ntypes:
        x = flatten_tf(data[nt].tf)
        # a node's union row is its own type's block padded with zeros elsewhere, so normalizing
        # the union rows is exactly normalizing each block's rows.
        if x.shape[1] and xnorm == "sgc":
            x = row_normalize(x)
        elif x.shape[1] and xnorm == "absnorm":
            x = absnorm_normalize(x)
        blocks[nt] = x
        try:
            data[nt].tf = None
        except Exception:
            pass
        dims[nt] = (c0, x.shape[1])
        c0 += x.shape[1]
    D = c0
    times = {}
    for nt in ntypes:
        t = getattr(data[nt], "time", None)
        times[nt] = None if t is None else t.numpy().astype("int64")
    return offs, D, times, n, blocks, dims


def slab(blocks, dims, offs, n, c0, c1):
    """Dense (n x (c1-c0)) column slab built straight from the per-type blocks, so we never
    materialize a sparse matrix holding dense text embeddings"""
    out = np.zeros((n, c1 - c0), np.float32)
    for nt, (o, w) in dims.items():
        if w == 0 or o >= c1 or o + w <= c0:
            continue
        a, b = max(o, c0), min(o + w, c1)
        out[offs[nt]:offs[nt] + blocks[nt].shape[0], a - c0:b - c0] = blocks[nt][:, a - o:b - o]
    return out


def visible_adj(data, offs, vis, n):
    """Symmetric adjacency over the union index, keeping only edges between visible nodes."""
    rows, cols = [], []
    for (src, _r, dst), ei in data.edge_index_dict.items():
        s, d = ei.numpy()
        gs, gd = offs[src] + s, offs[dst] + d
        ok = vis[gs] & vis[gd]
        rows.append(gs[ok]); cols.append(gd[ok])
    r = np.concatenate(rows); c = np.concatenate(cols)
    A = sp.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n)).tocsr()
    return A.maximum(A.T)


def visible_mask(ntypes, offs, times, n, nnodes, tau):
    """Nodes a predictor at time tau may see: timeless ones always, stamped ones iff time <= tau."""
    vis = np.zeros(n, bool)
    for nt in ntypes:
        lo = offs[nt]
        tt = times[nt]
        vis[lo:lo + nnodes[nt]] = (np.ones(nnodes[nt], bool) if tt is None else (tt <= tau))
    return vis

# (1-a)^rank, the recency weightings the model gets to pick
LABF_EWMA = (0.1, 0.3, 0.5, 0.7, 0.9)   
# continuous-time kernels exp(-dt/hl), in days
LABF_HL = (30.0, 180.0)                 
# 5 scalars (has_history, log1p count, log1p staleness, last label, uniform mean = Entity Mean) then one column per decay rate.
LABF_D = 5 + len(LABF_EWMA) + len(LABF_HL)
LABF_NAMES = (["has", "count", "stale", "last", "mean"]
              + [f"ewma{a}" for a in LABF_EWMA] + [f"hl{int(h)}" for h in LABF_HL])



def snapshot_feats(data, ntypes, offs, times, n, ent_pos, tau, K, norm, blocks, dims, nnodes):
    """Filtered features for the entity rows at cutoff tau (seconds)."""
    vis = visible_mask(ntypes, offs, times, n, nnodes, tau)
    A = visible_adj(data, offs, vis, n)
    deg = np.asarray(A.sum(1)).ravel()
    age = np.zeros(n, np.float32)
    for nt in ntypes:
        t_ = times[nt]
        if t_ is not None:
            lo = offs[nt]
            age[lo:lo + nnodes[nt]] = np.clip((tau - t_) / 31557600.0, 0, 50)
    Ah = rw_normalized_adjacency(A) if norm == "row" else aug_normalized_adjacency(A)
    use_rows, cost = 1 <= K <= 2, None   # K=0 must take the chunked path, which returns X
    if use_rows:
        rown = np.diff(Ah.indptr).astype(np.float64)
        cost = (np.asarray(Ah[ent_pos] @ rown).ravel() if K == 2
                else rown[ent_pos].astype(np.float64))
        if cost.sum() > K * Ah.nnz:
            print(f"  [path] restricted product would be {cost.sum():.3g} nnz vs "
                  f"{K * Ah.nnz:.3g} for full propagation; using column chunks", flush=True)
            use_rows = False
    if use_rows:
        # TODO: improve this slicing
        Dt = sum(w for (_o, w) in dims.values()) + 2
        F = np.zeros((len(ent_pos), Dt), np.float32)
        struct = np.stack([np.log1p(deg), age], 1).astype(np.float32)
        budget = float(os.environ.get("SGC_NNZ_BUDGET", "1.5e8"))
        cuts, acc, start = [], 0.0, 0
        for i, c in enumerate(cost):
            if i > start and acc + c > budget:
                cuts.append((start, i))
                start, acc = i, 0.0
            acc += c
        cuts.append((start, len(ent_pos)))
        print(f"  [rows] {len(ent_pos)} entity rows in {len(cuts)} slices "
              f"(max slice cost {budget:.3g} nnz)", flush=True)
        for r0, r1 in cuts:
            R = Ah[ent_pos[r0:r1], :]
            for _ in range(K - 1):
                R = R @ Ah                      
            for nt in ntypes:
                o, w = dims[nt]
                if w == 0:
                    continue
                lo = offs[nt]
                F[r0:r1, o:o + w] = R[:, lo:lo + blocks[nt].shape[0]] @ blocks[nt]
            F[r0:r1, -2:] = R @ struct
            del R
    else:
        Dbase = sum(w for (_o, w) in dims.values())
        Dt = Dbase + 2
        chunk = max(1, int(os.environ.get("SGC_CHUNK", "256")))
        F = np.zeros((len(ent_pos), Dt), np.float32)
        struct_d = np.stack([np.log1p(deg), age], 1).astype(np.float32)
        for c0 in range(0, Dt, chunk):
            c1 = min(c0 + chunk, Dt)
            if c0 >= Dbase:
                Xc = struct_d[:, c0 - Dbase:c1 - Dbase].copy()
            elif c1 <= Dbase:
                Xc = slab(blocks, dims, offs, n, c0, c1)
            else:
                Xc = np.hstack([slab(blocks, dims, offs, n, c0, Dbase),
                                struct_d[:, :c1 - Dbase]])
            for _ in range(K):
                Xc = Ah @ Xc
            F[:, c0:c1] = Xc[ent_pos]
            del Xc
    F[~vis[ent_pos]] = 0.0
    return F, int(vis.sum()), int(A.nnz)          # F has D+2 columns (degree, age appended)


