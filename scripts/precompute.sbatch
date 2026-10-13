#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x-%A_%a.out
DATASET=$1; TASK=$2; K=${3:-2}; SHARDS=${4:-1}
uv run python -m rbl.cli precompute --dataset "$DATASET" --task "$TASK" --k "$K" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$SHARDS"
