#!/bin/bash
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x-%j.out
DATASET=$1; TASK=$2; GRAPH=${3:-cross+self}; TRIALS=${4:-300}; TUNE_SEEDS=${5:-0,7,13,21,42,77,123,555,1234,2026}
uv run python -m rbl.cli cs --dataset "$DATASET" --task "$TASK" --graph "$GRAPH" \
  --trials "$TRIALS" --tune-seeds "$TUNE_SEEDS"
