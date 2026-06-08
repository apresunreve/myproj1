#!/usr/bin/env bash
set -euo pipefail

REPO=/scratch/general/vast/j.yan/myproj1
cd "$REPO"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="$REPO/src"
export KABR_ENABLE_FLASH_ATTN="${KABR_ENABLE_FLASH_ATTN:-0}"

OUT_ROOT="${OUT_ROOT:-$REPO/artifacts/smoke/efficiency_phase1_llava_0_5b}"
PREDICTIONS="$OUT_ROOT/behavior_val_predictions.jsonl"
METRICS="$OUT_ROOT/behavior_val_predictions.metrics.json"
CONFUSION="$OUT_ROOT/behavior_val_predictions.confusion.csv"

mkdir -p "$OUT_ROOT"
rm -f "$PREDICTIONS" "$METRICS" "$CONFUSION"

echo "host: $(hostname)"
echo "date: $(date)"
echo "cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
echo "predictions: $PREDICTIONS"
nvidia-smi || true

"$REPO/.venv/bin/python" -m kabr_vlm.cli eval-finetuned-hf-vlm \
  --annotation /scratch/general/nfs1/u1320844/KABR_files2/annotation/val.cleaned.csv \
  --image-root /scratch/general/nfs1/u1320844/KABR_files2/dataset/image \
  --classes /scratch/general/nfs1/u1320844/KABR_files2/annotation/classes.json \
  --model-id llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
  --adapter-path "$REPO/artifacts/runs/llava_onevision_qwen2_0_5b_behavior_lora_full/checkpoint" \
  --task behavior \
  --prompt-mode label \
  --max-samples 1 \
  --seed 42 \
  --max-new-tokens 8 \
  --output "$PREDICTIONS"

"$REPO/.venv/bin/python" -m kabr_vlm.cli score-predictions \
  --predictions "$PREDICTIONS" \
  --metrics-output "$METRICS" \
  --confusion-output "$CONFUSION" \
  --task behavior

echo "done: $(date)"
