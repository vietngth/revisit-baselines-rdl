#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%j.out
# Train the reference MLP on precomputed features (1 seed).
# Measured per-seed train times (CPU nodes; GPU is faster): most tasks 0.02-1.7 min,
# amazon/user-ltv 2.8 min. The 4 h wall covers the 3-value learning-rate search plus
# loading the largest design matrices (rel-stack ~57 GB).
# Usage: sbatch scripts/train.sbatch rel-f1 driver-top3 2 0
DS=$1; TK=$2; K=${3:-2}; SEED=${4:-0}
FEATURES=${FEATURES_DIR:-features}
uv run python -m rbl.cli train --dataset "$DS" --task "$TK" --k "$K" --seed "$SEED" \
  --features "$FEATURES" --out-csv results.csv
