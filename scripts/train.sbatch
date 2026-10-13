#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%j.out
DATASET=$1; TASK=$2; K=${3:-2}; SEEDS=${4:-0,7,13,21,42,77,123,555,1234,2026}
uv run python -m rbl.cli train --dataset "$DATASET" --task "$TASK" --k "$K" --seeds "$SEEDS"
