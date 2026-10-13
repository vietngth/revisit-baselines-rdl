# Revisiting Simple Baselines for Relational Deep Learning

Code for the paper "Revisiting Simple Baselines for Relational Deep Learning" (to appear at the Learning on Graphs conference, 2026). The pipeline
has three steps:

1. **Precompute features.** The relational entity graph of a database is collapsed into a
   single homogeneous graph, the per-table features are stacked into one matrix, and a fixed
   low-pass graph filter propagates them over the graph. One feature set is computed per
   distinct seed time of a task, using only rows visible at that time.
2. **Train an MLP.** A per-row multilayer perceptron is trained on the precomputed features.
3. **Refine with historical labels.** Correct and Smooth propagates the labels that are already
   known at prediction time over a small graph built for every prediction row.

## Requirements

Python >= 3.10 and [uv](https://docs.astral.sh/uv/). All Python dependencies are declared in
`pyproject.toml`.

## Setup

```bash
uv sync
```

RelBench datasets are read from `~/.cache/relbench` (override with `RELBENCH_CACHE_DIR`); download
them once with `relbench.datasets.get_dataset(name, download=True)` and
`relbench.tasks.get_task(name, task, download=True)`. Encoded tables and column-type assignments
are cached under `~/.cache/relbench_examples` (override with `RBL_CACHE`). Start from a clean cache
directory to avoid stale encodings.

## Usage

### 1. Precompute features

```bash
uv run python -m rbl.cli precompute --dataset rel-avito --task user-clicks --k 2
```

`--k` is the number of filter steps. With `--k 0` a row keeps the attributes of its own table and
the degree and age columns are zero.

Every seed time of a task is an independent snapshot, so the work can be split into shards that
run as separate processes and write disjoint files under `features/<dataset>_<task>_K<k>/`:

```bash
sbatch --array=0-31 scripts/precompute.sbatch rel-hm user-churn 2 32
```

Large snapshots are split further into ranges of feature columns (each column propagates
independently). 

### 2. Train the MLP

```bash
uv run python -m rbl.cli train --dataset rel-avito --task user-clicks --k 2 --seeds 0
```
### 3. Refine with Correct and Smooth
This is the post-processing step based on the C&S algorithm, with an additional step of constructing different graph variants for label propagation.

```bash
uv run python -m rbl.cli cs --dataset rel-avito --task user-clicks --k 2 --seeds 0
```
The co-occurrence graph between the entities of the
task's table is built once and stored in `projections/`. The command then builds the label graph
of the validation rows (labels of the training split) and tunes C&S on it with
Optuna. Each stage is searched under each residual scaling
(none, FDiff-scale, and Autoscale for classification), with `--trials` settings per search, scored
on the seeds of `--tune-seeds`. Every selected setting is then applied unchanged to all seeds on the
label graph of the test rows (labels of the training and validation splits) and written to
`cs_results.csv`. The search with the best validation score is the reported one; scores are
compared at four decimals, and ties go to the simpler scaling, then to the earlier stage.

`--graph` selects how historical labels are wired to a prediction row (see our paper for the more detailed descriptions):

| `--graph` | Own past labels | Labels of co-occurring entities |
|---|---|---|
| `self` | star with exponentially decaying weights | |
| `cross` | | latest label |
| `cross+self` (default) | star | latest label |
| `cross+hist` | | relay node with the three latest labels |
| `all` | one relay per label | relay node with the three latest labels |
| `self-chain` | chain, most recent first | |
| `chain+cross` | chain | latest label |
| `chain+hist-3` | chain of the three latest labels | chain of the three latest labels |
| `chain+hist` | chain | chain |

## Main experiments

To reproduce the experiments in our main Tables 1 and 2:

| Column | Commands |
|---|---|
| MLP (target attributes) | `precompute --k 0`, `train --k 0` |
| MLP (+relational attributes) | `precompute --k 2`, `train --k 2` |
| C&S (+historical labels) | `cs --k 2 --graph cross+self` |

Correct and Smooth uses `--trials 60 --tune-seeds 0,7,13` on the RelBench-v1 classification tasks
and the defaults (300 trials, all seeds) on the other tasks.

- RelBench-v1 classification: `rel-amazon` `user-churn`, `item-churn`; `rel-avito` `user-clicks`,
  `user-visits`; `rel-hm` `user-churn`; `rel-stack` `user-badge`, `user-engagement`; `rel-trial`
  `study-outcome`.
- RelBench-v1 regression: `rel-amazon` `user-ltv`, `item-ltv`; `rel-avito` `ad-ctr`; `rel-hm`
  `item-sales`; `rel-stack` `post-votes`; `rel-trial` `site-success`, `study-adverse`.
- RelBench-v2 classification: `rel-ratebeer` `user-churn`, `beer-churn`, `brewer-dormant`;
  `rel-arxiv` `paper-citation`.
- RelBench-v2 regression: `rel-ratebeer` `user-count`; `rel-arxiv` `author-publication`.

## Tests

```bash
uv run --with pytest python -m pytest tests
```

## Limitations

- **Propagation runs on CPU** with scipy sparse-dense products, for the filter and for Correct and
  Smooth. Sharding parallelizes the filter across snapshots and feature columns.
- **Training holds the design matrices in memory**, about `(rows) x (features + 2) x 4` bytes,
  which reaches roughly 180 GB on `rel-ratebeer`/`beer-churn`.
- **Label graphs are built in Python**, one prediction row at a time, and the co-occurrence graph
  of a large entity table is quite large. For example, customer rows in `rel-hm` have 3.2 billion links, taking
  half an hour and 154 GB to build and 48 GB on disk.

## Code adaptation

- `rbl/normalize.py`: `aug_normalized_adjacency` follows
  [Tiiiger/SGC](https://github.com/Tiiiger/SGC). `absnorm_normalize` divides by the sum of
  absolute values instead of SGC's signed row sum, which is unstable for signed embedding features.
- `rbl/embedder.py`: the GloVe-300d text embedder used by the RelBench examples.
- `rbl/cs.py`: a port of [CUAI/CorrectAndSmooth](https://github.com/CUAI/CorrectAndSmooth),
  `outcome_correlation.py`: the diffusion of `general_outcome_correlation` (lines 160-172), Correct
  with Autoscale and Smooth of `double_correlation_autoscale` (186-209), and the FDiff-scale Correct
  of `double_correlation_fixed` (211-235). The settings tried first in `rbl/tune.py` are the ones
  released in its `run_experiments.py` (lines 46-54, 63-72, 80-89 and 110-118).
- `rbl/atomic_routes.py`: copied unchanged from
  [snap-stanford/RelGNN](https://github.com/snap-stanford/RelGNN), `examples/atomic_routes.py`
  (MIT License, see `THIRD_PARTY_LICENSES`).
