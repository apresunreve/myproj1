#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --exclude=grn025,grn026,grn027,grn028,grn029,grn030,grn031,grn032,grn066,grn067,grn080
#SBATCH -o /scratch/general/vast/j.yan/myproj1/logs/%x.%j.out
#SBATCH -e /scratch/general/vast/j.yan/myproj1/logs/%x.%j.err

# Usage: sbatch --job-name=<name> --export=ALL,MODEL_KEY=<x3d-xs|x3d-s|x3d-m|i3d-l> ...
set -euo pipefail

REPO=/scratch/general/vast/j.yan/myproj1
echo "host: $(hostname)"
echo "model_key: ${MODEL_KEY:?MODEL_KEY must be set}"
nvidia-smi || true

module load cuda/12.1.0 || true

bash "$REPO/scripts/run_kabr_clip_video_train.sh" "$MODEL_KEY"

echo "done: $(date)"
