#!/bin/bash
#SBATCH -M notchpeak
#SBATCH -p notchpeak-gpu-guest
#SBATCH --account=owner-gpu-guest
#SBATCH --job-name=bbl_lora_smoke
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/general/vast/j.yan/myproj1/logs/bbl_lora_smoke.%j.out
#SBATCH -e /scratch/general/vast/j.yan/myproj1/logs/bbl_lora_smoke.%j.err

set -euo pipefail

REPO=/scratch/general/vast/j.yan/myproj1
export KABR_ENABLE_FLASH_ATTN=0

echo "host: $(hostname)"
echo "date: $(date)"
nvidia-smi || true

bash "$REPO/scripts/run_vlm_baboonland_full.sh" \
    "$REPO/configs/llava_onevision_qwen2_0_5b_baboonland_behavior_lora_smoke.yaml" \
    llava_ov_0_5b_baboonland_smoke \
    llava-hf/llava-onevision-qwen2-0.5b-ov-hf

echo "done: $(date)"
