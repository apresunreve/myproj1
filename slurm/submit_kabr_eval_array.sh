#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_FILE="${1:-$ROOT_DIR/slurm/kabr_eval_jobs.tsv}"
PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-32G}"
TIME="${TIME:-04:00:00}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found on PATH. Run this script on a Slurm login node." >&2
  exit 1
fi

if [[ ! -f "$JOB_FILE" ]]; then
  echo "Job file not found: $JOB_FILE" >&2
  exit 1
fi

JOB_COUNT="$(tail -n +2 "$JOB_FILE" | sed '/^[[:space:]]*$/d' | wc -l)"
if [[ "$JOB_COUNT" -eq 0 ]]; then
  echo "No jobs found in $JOB_FILE" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/slurm/logs"

sbatch \
  --job-name=kabr-vlm-enum \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM" \
  --time="$TIME" \
  --array="1-$JOB_COUNT" \
  --export=ALL,ROOT_DIR="$ROOT_DIR",JOB_FILE="$JOB_FILE" \
  --output="$ROOT_DIR/slurm/logs/%x_%A_%a.out" \
  --error="$ROOT_DIR/slurm/logs/%x_%A_%a.err" \
  "$ROOT_DIR/slurm/run_kabr_eval_array.sh"
