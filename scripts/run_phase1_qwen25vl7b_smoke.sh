#!/usr/bin/env bash
set -euo pipefail

REPO=/scratch/general/vast/j.yan/myproj1
cd "$REPO"

export PYTHONPATH="$REPO/src"
export KABR_ENABLE_FLASH_ATTN="${KABR_ENABLE_FLASH_ATTN:-0}"

OUT_ROOT="${OUT_ROOT:-$REPO/artifacts/smoke/efficiency_phase1_qwen25vl7b}"
PREDICTIONS="$OUT_ROOT/behavior_val_predictions.jsonl"
METRICS="$OUT_ROOT/behavior_val_predictions.metrics.json"
CONFUSION="$OUT_ROOT/behavior_val_predictions.confusion.csv"

mkdir -p "$OUT_ROOT"
rm -f "$PREDICTIONS" "$METRICS" "$CONFUSION"

echo "host: $(hostname)"
echo "date: $(date)"
echo "predictions: $PREDICTIONS"
nvidia-smi || true

"$REPO/.venv/bin/python" -m kabr_vlm.cli eval-finetuned-hf-vlm \
  --annotation /scratch/general/nfs1/u1320844/KABR_files2/annotation/val.cleaned.csv \
  --image-root /scratch/general/nfs1/u1320844/KABR_files2/dataset/image \
  --classes /scratch/general/nfs1/u1320844/KABR_files2/annotation/classes.json \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter-path "$REPO/artifacts/runs/qwen25vl7b_behavior_lora_full/checkpoint" \
  --task behavior \
  --prompt-mode label \
  --max-samples "${MAX_SAMPLES:-1}" \
  --seed "${SEED:-42}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-8}" \
  --output "$PREDICTIONS"

"$REPO/.venv/bin/python" -m kabr_vlm.cli score-predictions \
  --predictions "$PREDICTIONS" \
  --metrics-output "$METRICS" \
  --confusion-output "$CONFUSION" \
  --task behavior

echo "done: $(date)"
