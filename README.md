# Revisiting Simple Baselines for Relational Deep Learning

Code for a simple relational baseline on RelBench. The pipeline has two steps:

1. **Precompute features.** The relational entity graph of a database is collapsed into a
   single homogeneous graph, the per-table features are stacked into one matrix, and a fixed
   low-pass graph filter propagates them over the graph. One feature set is computed per
   distinct seed time of a task, using only rows visible at that time.
2. **Train an MLP.** A per-row multilayer perceptron is trained on the precomputed features.

3. **Refine with label propagation.** Correct-and-Smooth propagates historical labels over a
   meta-graph on the target entities (`rbl/cs.py`), which the relational entity graph does not
   define; the constructions we evaluate are listed in `VARIANTS`.


## Requirements
All Python dependencies are declared in `pyproject.toml`.
You should have Python >= 3.10 and [uv](https://docs.astral.sh/uv/) to run this project smoothly. 

## Setup

```bash
uv sync
```

RelBench datasets are downloaded on first use into `~/.cache/relbench` (override with
`RELBENCH_CACHE_DIR`). Encoded tables and column-type assignments are cached under
`~/.cache/relbench_examples` (override with `RBL_CACHE`). You should start from a clean cache directory to avoid problems.

## Usage

### 1. Precompute features

```bash
uv run python -m rbl.cli precompute --dataset rel-f1 --task driver-top3 --k 2 --out features
```

Note: use `--k 0` for turning off filtering process.

### 2. Train the MLP

```bash
uv run python -m rbl.cli train --dataset rel-f1 --task driver-top3 --k 2 --seed 0
```

Running on GPU is the default support setting. In case you are using HPC servers, we provide some slurm examples in `scripts/` folder.

### 3. Refine with Correct-and-Smooth

```bash
uv run python -m rbl.cli cs --dataset rel-f1 --task driver-top3 --k 2 --seed 0 --variant self
```

C&S: run this once you have any base predictions in `preds/`. You can skip the filtering part to perform C&S directly on unfiltered bases. 


## Limitations

- **Propagation runs on CPU.** The sparse-dense products in `rbl/pipeline.py` use scipy and
  are the dominant cost of `precompute` on wide datasets. Both propagation paths (row slices
  for shallow `K`, column chunks for deep `K`) are slice-local and map directly onto
  `torch.sparse` on a GPU: the operator has a few million nonzeros and fits in device memory. Future implementation will be extended on GPU
- **Design matrices are materialized in memory.** `rbl/cli.py` allocates the train, validation
  and test matrices densely and holds them until the consolidated feature file is written, so
  precompute needs about `(rows) x (features + 2) x 4` bytes. This is small for most datasets
  but reaches roughly 180 GB on `rel-ratebeer`/`beer-churn` (2.5M rows, 17,765 columns), and it
  is also why resuming from checkpointed snapshots costs a full I/O pass. Memory mapping the
  blocks, storing them in half precision, or training directly from per-snapshot slices would
  each remove the limit.

## Code adaptation

- `rbl/sgc.py`: `aug_normalized_adjacency` and `row_normalize` are copied from
  [Tiiiger/SGC](https://github.com/Tiiiger/SGC) (only the returned sparse format differs);
  `rw_normalized_adjacency` is the row-stochastic variant from gear/gfnn. `absnorm` is our custom feature normalization. it divides by the sum of absolute values instead of the signed row sum, which is unstable for signed embedding features.
- `rbl/embedder.py`: the GloVe-300d text embedder used by the RelBench examples.