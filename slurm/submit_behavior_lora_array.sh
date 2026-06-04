#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_FILE="${1:-}"

if [[ -z "$JOB_FILE" ]]; then
  echo "Usage: $0 <job_manifest.tsv>" >&2
  exit 1
fi

if [[ "$JOB_FILE" != /* ]]; then
  JOB_FILE="$ROOT_DIR/$JOB_FILE"
fi

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

RESOURCE_TIER="$(tail -n +2 "$JOB_FILE" | sed '/^[[:space:]]*$/d' | cut -f3 | sort -u)"
if [[ "$(printf '%s\n' "$RESOURCE_TIER" | sed '/^[[:space:]]*$/d' | wc -l)" -ne 1 ]]; then
  echo "Manifest must contain exactly one resource_tier: $JOB_FILE" >&2
  exit 1
fi

case "$RESOURCE_TIER" in
  small)
    PARTITION="${PARTITION:-granite-gpu}"
    QOS="${QOS:-granite-gpu-freecycle}"
    GRES="${GRES:-gpu:1}"
    CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
    MEM="${MEM:-32G}"
    TIME="${TIME:-08:00:00}"
    ;;
  medium)
    PARTITION="${PARTITION:-granite-gpu}"
    QOS="${QOS:-granite-gpu-freecycle}"
    GRES="${GRES:-gpu:1}"
    CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
    MEM="${MEM:-48G}"
    TIME="${TIME:-16:00:00}"
    ;;
  large)
    PARTITION="${PARTITION:-granite-gpu}"
    QOS="${QOS:-granite-gpu-freecycle}"
    GRES="${GRES:-gpu:1}"
    CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
    MEM="${MEM:-80G}"
    TIME="${TIME:-24:00:00}"
    ;;
  *)
    echo "Unsupported resource tier: $RESOURCE_TIER" >&2
    exit 1
    ;;
esac

mkdir -p "$ROOT_DIR/slurm/logs"

JOB_NAME="kabr-lora-${RESOURCE_TIER}"
sbatch \
  --job-name="$JOB_NAME" \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --gres="$GRES" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM" \
  --time="$TIME" \
  --array="1-$JOB_COUNT" \
  --export=ALL,ROOT_DIR="$ROOT_DIR",JOB_FILE="$JOB_FILE" \
  --output="$ROOT_DIR/slurm/logs/%x_%A_%a.out" \
  --error="$ROOT_DIR/slurm/logs/%x_%A_%a.err" \
  "$ROOT_DIR/slurm/run_behavior_lora_array.sh"
