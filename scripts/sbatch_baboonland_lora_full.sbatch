#!/bin/bash
#SBATCH -M notchpeak
#SBATCH -p notchpeak-gpu-guest
#SBATCH --account=owner-gpu-guest
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/general/vast/j.yan/myproj1/logs/%x.%j.out
#SBATCH -e /scratch/general/vast/j.yan/myproj1/logs/%x.%j.err

# Generic baboonland LoRA full-run sbatch wrapper.
# Usage:
#   sbatch --job-name=<name> --export=LAUNCH=<per-family-script>.sh \
#          scripts/sbatch_baboonland_lora_full.sbatch

set -euo pipefail

REPO=/scratch/general/vast/j.yan/myproj1
export KABR_ENABLE_FLASH_ATTN=0

if [ -z "${LAUNCH:-}" ]; then
    echo "LAUNCH env var must point to a per-family launch script"
    exit 2
fi

echo "host: $(hostname)"
echo "date: $(date)"
echo "launcher: $LAUNCH"
nvidia-smi || true

bash "$REPO/scripts/$LAUNCH"

echo "done: $(date)"
