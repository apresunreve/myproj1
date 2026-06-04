#!/bin/bash
#SBATCH -M notchpeak
#SBATCH -p notchpeak-gpu-guest
#SBATCH --account=owner-gpu-guest
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH -o /scratch/general/vast/j.yan/myproj1/logs/%x.%j.out
#SBATCH -e /scratch/general/vast/j.yan/myproj1/logs/%x.%j.err

# Usage:
#   sbatch --job-name=<name> \
#     --export=ALL,MODEL_KEY=<x3d-l|i3d-l|...>,[CLIP_MANIFEST_DIR=...],[MAX_EPOCH=N],[OUT_ROOT=...] \
#     scripts/sbatch_baboonland_clip_train.sbatch

set -euo pipefail

REPO=/scratch/general/vast/j.yan/myproj1

echo "host: $(hostname)"
echo "date: $(date)"
echo "model_key: ${MODEL_KEY:?MODEL_KEY must be set via --export}"
nvidia-smi || true

module load cuda/12.1.0 || true

bash "$REPO/scripts/run_baboonland_clip_video_train.sh" "$MODEL_KEY"

echo "done: $(date)"
