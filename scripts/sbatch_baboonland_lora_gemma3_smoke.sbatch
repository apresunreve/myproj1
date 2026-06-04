#!/bin/bash
#SBATCH -M notchpeak
#SBATCH -p notchpeak-gpu-guest
#SBATCH --account=owner-gpu-guest
#SBATCH --job-name=bbl_gemma3_smoke
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -o /scratch/general/vast/j.yan/myproj1/logs/%x.%j.out
#SBATCH -e /scratch/general/vast/j.yan/myproj1/logs/%x.%j.err

set -euo pipefail

REPO=/scratch/general/vast/j.yan/myproj1
export KABR_ENABLE_FLASH_ATTN=0
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set via --export}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

echo "host: $(hostname)"
echo "date: $(date)"
nvidia-smi || true

bash "$REPO/scripts/run_gemma3_4b_baboonland_lora_smoke.sh"

echo "done: $(date)"
