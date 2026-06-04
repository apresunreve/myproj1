#!/usr/bin/env bash
#SBATCH --ntasks=1
set -euo pipefail

ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
JOB_FILE="${JOB_FILE:?JOB_FILE is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
DRY_RUN="${DRY_RUN:-false}"

if [[ ! -f "$JOB_FILE" ]]; then
  echo "Job file not found: $JOB_FILE" >&2
  exit 1
fi

JOB_LINE="$(tail -n +2 "$JOB_FILE" | sed '/^[[:space:]]*$/d' | sed -n "${TASK_ID}p")"
if [[ -z "$JOB_LINE" ]]; then
  echo "No job definition for array index $TASK_ID" >&2
  exit 1
fi

IFS=$'\t' read -r JOB_NAME LAUNCHER_SCRIPT RESOURCE_TIER SKIP_IF_METRICS_EXISTS <<< "$JOB_LINE"

if [[ -z "$JOB_NAME" || -z "$LAUNCHER_SCRIPT" || -z "$RESOURCE_TIER" || -z "$SKIP_IF_METRICS_EXISTS" ]]; then
  echo "Malformed job definition for array index $TASK_ID" >&2
  exit 1
fi

if [[ ! -f "$LAUNCHER_SCRIPT" ]]; then
  echo "Launcher script not found: $LAUNCHER_SCRIPT" >&2
  exit 1
fi

CACHE_ROOT="${CACHE_ROOT:-/scratch/general/nfs1/u1320844}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv-cache}"
HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface-cache}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg-cache}"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$ROOT_DIR/slurm/logs"

export UV_CACHE_DIR
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME

echo "job_name=$JOB_NAME"
echo "launcher_script=$LAUNCHER_SCRIPT"
echo "resource_tier=$RESOURCE_TIER"
echo "skip_if_metrics_exists=$SKIP_IF_METRICS_EXISTS"
echo "hostname=$(hostname)"
echo "cwd=$ROOT_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

cd "$ROOT_DIR"

if [[ -f "$SKIP_IF_METRICS_EXISTS" ]]; then
  echo "Skipping $JOB_NAME because metrics already exist at $SKIP_IF_METRICS_EXISTS"
  exit 0
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY_RUN=true"
  echo "Would run: bash $LAUNCHER_SCRIPT"
  exit 0
fi

echo "Starting $JOB_NAME"
bash "$LAUNCHER_SCRIPT"
