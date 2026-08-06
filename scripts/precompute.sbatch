#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x-%j.out
# Precompute filtered features for one dataset/task (CPU-bound; GPU only speeds up the
# one-off text-embedding materialization on the first run per database).
#
# Measured wall times on 8-core nodes, K=2, AFTER the one-off materialization
# (filter + graph union; first run adds the embedding pass, unmeasured, GPU recommended):
#   rel-f1        ~0.1 h    (all tasks)
#   rel-avito     ~0.1 h
#   rel-trial     ~0.2 h
#   rel-amazon    ~0.2 h
#   rel-hm        ~0.4 h
#   rel-ratebeer  ~0.5 h    (user-churn / user-count; beer-churn and brewer-dormant
#                            exceed 24 h and resume from slice checkpoints automatically)
#   rel-stack     ~0.8 h
# Memory: 200G covers every task; rel-stack and rel-amazon are the largest.
#
# Usage: sbatch scripts/precompute.sbatch rel-f1 driver-top3 2
DS=$1; TK=$2; K=${3:-2}
FEATURES=${FEATURES_DIR:-features}   # point at a large filesystem; matrices reach ~60 GB
uv run python -m rbl.cli precompute --dataset "$DS" --task "$TK" --k "$K" --out "$FEATURES"
