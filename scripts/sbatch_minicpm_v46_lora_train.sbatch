#!/bin/bash
#SBATCH -M granite
#SBATCH -p granite-gpu
#SBATCH --qos=granite-gpu-freecycle
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/general/vast/j.yan/myproj1/logs/%x.%j.out
#SBATCH -e /scratch/general/vast/j.yan/myproj1/logs/%x.%j.err

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: sbatch --job-name=<name> $0 <config_path>"
  exit 2
fi

REPO=/scratch/general/vast/j.yan/myproj1
CONFIG_PATH=$1
CACHE_ROOT=${CACHE_ROOT:-/scratch/general/nfs1/u1320844}
PYTHON_BIN="$REPO/.venv_minicpm_smoke/bin/python"

mkdir -p "$REPO/logs" \
  "$CACHE_ROOT/huggingface-cache" \
  "$CACHE_ROOT/xdg-cache" \
  "$CACHE_ROOT/uv-cache"

export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv-cache}"
export PYTHONPATH="$REPO/src"
export KABR_ENABLE_FLASH_ATTN="${KABR_ENABLE_FLASH_ATTN:-0}"

cd "$REPO"

echo "host: $(hostname)"
echo "date: $(date)"
echo "config: $CONFIG_PATH"
echo "python: $PYTHON_BIN"
nvidia-smi || true

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing MiniCPM Python environment: $PYTHON_BIN"
  echo "Create it with: python -m venv --system-site-packages .venv_minicpm_smoke"
  exit 1
fi

"$PYTHON_BIN" -m kabr_vlm.training.finetune \
  --config "$CONFIG_PATH" \
  --resume-from-checkpoint auto

echo "done: $(date)"
