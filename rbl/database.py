"""Load a RelBench database and collapse it into a CollapsedGraph."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig

from rbl import normalize
from rbl.embedder import GloveTextEmbedding
from rbl.graph import CollapsedGraph

CACHE_DIR = Path(os.environ.get("RBL_CACHE", Path.home() / ".cache" / "relbench_examples"))
TOP_CATEGORIES = 6
SECONDS_PER_YEAR = 31_557_600.0
EPOCH = pd.Timestamp("1990-01-01")


def load(dataset_name):
    """Return the RelBench dataset and its collapsed graph with normalized feature blocks.

    A table contributes its numerical and text columns as encoded by RelBench, and its
    categorical and timestamp columns encoded without learning (see _categorical_and_time).
    """
    _load_tensors_on_cpu_without_gpu()
    dataset = get_dataset(dataset_name, download=False)
    database = dataset.get_db()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    column_types = _column_types(dataset_name, database)
    data, _ = make_pkey_fkey_graph(
        database,
        col_to_stype_dict=column_types,
        text_embedder_cfg=TextEmbedderConfig(text_embedder=GloveTextEmbedding(device=device),
                                             batch_size=256),
        cache_dir=str(CACHE_DIR / dataset_name / "materialized"))
    node_types = list(data.node_types)
    features, times = {}, {}
    for node_type in node_types:
        block = np.hstack([_flatten(data[node_type].tf),
                           _categorical_and_time(database.table_dict[node_type],
                                                 column_types[node_type], dataset.val_timestamp)])
        features[node_type] = normalize.absnorm_normalize(block) if block.shape[1] else block
        data[node_type].tf = None
        stamps = getattr(data[node_type], "time", None)
        times[node_type] = None if stamps is None else stamps.numpy().astype(np.int64)
    offsets = np.cumsum([0] + [len(features[t]) for t in node_types])
    offset_of = dict(zip(node_types, offsets))
    src, dst = [], []
    for (source, _, target), edge_index in data.edge_index_dict.items():
        s, d = edge_index.numpy()
        src.append(offset_of[source] + s)
        dst.append(offset_of[target] + d)
    graph = CollapsedGraph(node_types, features, times, np.concatenate(src), np.concatenate(dst))
    return dataset, graph


def _column_types(dataset_name, database):
    path = CACHE_DIR / dataset_name / "stypes.json"
    if path.exists():
        saved = json.loads(path.read_text())
        return {table: {column: stype(kind) for column, kind in columns.items()}
                for table, columns in saved.items()}
    proposal = get_stype_proposal(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({table: {column: kind.value for column, kind in columns.items()}
                                for table, columns in proposal.items()}, indent=2))
    return proposal


def _categorical_and_time(table, column_types, cutoff):
    """Categorical columns as one-hot of the most frequent values plus log frequency, counted on
    rows up to `cutoff` only; timestamp columns as years before the row's own time (years since
    1990 for tables without a time column)."""
    df = table.df
    keys = {table.pkey_col, table.time_col, *table.fkey_col_to_pkey_table}
    static = table.time_col is None
    visible = np.ones(len(df), bool) if static else (df[table.time_col] <= cutoff).to_numpy()
    reference = None if static else pd.to_datetime(df[table.time_col], errors="coerce")
    columns = []
    for column, kind in column_types.items():
        if column in keys or column not in df:
            continue
        if kind in (stype.categorical, stype.multicategorical):
            values = df[column].map(str)
            counts = values[visible].value_counts()
            for category in counts.index[:TOP_CATEGORIES]:
                columns.append((values == category).to_numpy(np.float32))
            columns.append(np.log1p(values.map(counts).fillna(0.0).to_numpy(np.float64)))
        elif kind == stype.timestamp:
            stamps = pd.to_datetime(df[column], errors="coerce")
            elapsed = (stamps - EPOCH if static else reference - stamps).dt.total_seconds()
            columns.append((elapsed / SECONDS_PER_YEAR).fillna(0.0).clip(-50, 50).to_numpy())
    if not columns:
        return np.zeros((len(df), 0), np.float32)
    return np.stack(columns, 1).astype(np.float32)


def _flatten(frame):
    """Numerical and embedding columns of a TensorFrame as one float32 matrix (NaN -> 0)."""
    parts = []
    if stype.numerical in frame.feat_dict:
        parts.append(frame.feat_dict[stype.numerical].float())
    if stype.embedding in frame.feat_dict:
        embedding = frame.feat_dict[stype.embedding]
        embedding = embedding.values if hasattr(embedding, "values") else embedding
        parts.append(embedding.float().reshape(len(frame), -1))
    if not parts:
        return np.zeros((len(frame), 0), np.float32)
    return torch.nan_to_num(torch.cat(parts, 1)).detach().cpu().numpy().astype(np.float32)


def _load_tensors_on_cpu_without_gpu():
    """Let CPU-only nodes read TensorFrame caches that were written on a GPU node."""
    if torch.cuda.is_available() or getattr(torch.load, "_rbl_cpu", False):
        return
    original = torch.load

    def cpu_load(*args, **kwargs):
        return original(*args, **{**kwargs, "map_location": "cpu"})

    cpu_load._rbl_cpu = True
    torch.load = cpu_load
