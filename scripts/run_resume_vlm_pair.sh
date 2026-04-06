#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.uvenv/bin/python}"
ANNOTATION_PATH="${ANNOTATION_PATH:-$ROOT_DIR/../KABR_files/annotation/val.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-$ROOT_DIR/../KABR_files/dataset/image}"
CLASSES_PATH="${CLASSES_PATH:-$ROOT_DIR/../KABR_files/annotation/classes.json}"
TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/vlm_suite}"
STATUS_FILE="${STATUS_FILE:-$ROOT_DIR/artifacts/full_eval/vlm_resume_pair_status.txt}"

mkdir -p "$LOG_DIR" "$(dirname "$STATUS_FILE")"

if [[ -z "$TOKEN" ]]; then
  echo "HF token missing. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN." >&2
  exit 1
fi

timestamp() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

mark() {
  local phase="$1"
  printf "%s\t%s\n" "$(timestamp)" "$phase" | tee "$STATUS_FILE"
}

run_eval() {
  local model_id="$1"
  local output_jsonl="$2"
  local metrics_json="$3"
  local confusion_csv="$4"
  local log_file="$5"

  HF_TOKEN="$TOKEN" HUGGINGFACE_HUB_TOKEN="$TOKEN" \
    "$PYTHON_BIN" -m kabr_vlm.cli eval-clip-hf-vlm \
      --annotation "$ANNOTATION_PATH" \
      --image-root "$IMAGE_ROOT" \
      --classes "$CLASSES_PATH" \
      --model-id "$model_id" \
      --task behavior \
      --prompt-mode plain \
      --max-samples 3761 \
      --max-frames-per-clip 8 \
      --max-new-tokens 16 \
      --resume \
      --output "$output_jsonl" >> "$log_file" 2>&1

  "$PYTHON_BIN" -m kabr_vlm.cli score-predictions \
    --predictions "$output_jsonl" \
    --metrics-output "$metrics_json" \
    --confusion-output "$confusion_csv" >> "$log_file" 2>&1
}

mark "gemma3_4b:start"
run_eval \
  "google/gemma-3-4b-it" \
  "$ROOT_DIR/artifacts/full_eval/gemma3_4b_behavior_clip_val_full.jsonl" \
  "$ROOT_DIR/artifacts/full_eval/gemma3_4b_behavior_clip_val_full.metrics.json" \
  "$ROOT_DIR/artifacts/full_eval/gemma3_4b_behavior_clip_val_full.confusion.csv" \
  "$LOG_DIR/gemma3_4b_behavior_clip_val_full.log"
mark "gemma3_4b:completed"

mark "idefics3_8b:resume"
run_eval \
  "HuggingFaceM4/Idefics3-8B-Llama3" \
  "$ROOT_DIR/artifacts/full_eval/idefics3_8b_behavior_clip_val_full.jsonl" \
  "$ROOT_DIR/artifacts/full_eval/idefics3_8b_behavior_clip_val_full.metrics.json" \
  "$ROOT_DIR/artifacts/full_eval/idefics3_8b_behavior_clip_val_full.confusion.csv" \
  "$LOG_DIR/idefics3_8b_behavior_clip_val_full.log"
mark "idefics3_8b:completed"

mark "suite:completed"
