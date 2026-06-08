#!/usr/bin/env bash
set -euo pipefail

REPO=/scratch/general/vast/j.yan/myproj1
cd "$REPO"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$REPO/src"
export KABR_ENABLE_FLASH_ATTN="${KABR_ENABLE_FLASH_ATTN:-0}"

MINICPM_PY="$REPO/.venv_minicpm_smoke/bin/python"
DEFAULT_PY="$REPO/.venv/bin/python"
OUT_ROOT="${OUT_ROOT:-$REPO/artifacts/latency_baselines/compact_100}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
STATUS_FILE="$OUT_ROOT/status.tsv"

mkdir -p "$OUT_ROOT"
printf "timestamp\tslug\tstatus\tdetail\n" > "$STATUS_FILE"

timestamp() { date +"%Y-%m-%dT%H:%M:%S%z"; }
mark() {
  printf "%s\t%s\t%s\t%s\n" "$(timestamp)" "$1" "$2" "$3" >> "$STATUS_FILE"
}

run_eval() {
  local slug="$1"
  local python_bin="$2"
  local target_dataset="$3"
  local val_jsonl="$4"
  local model_id="$5"
  local adapter_path="$6"
  local output_jsonl="$OUT_ROOT/${slug}.jsonl"
  local log_path="$OUT_ROOT/${slug}.log"

  mark "$slug" "running" "$output_jsonl"
  if "$python_bin" "$REPO/scripts/eval_cross_dataset_vlm.py" \
      --val_jsonl "$val_jsonl" \
      --target_dataset "$target_dataset" \
      --model_id "$model_id" \
      --adapter_path "$adapter_path" \
      --max_samples "$MAX_SAMPLES" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --output_jsonl "$output_jsonl" \
      --resume > "$log_path" 2>&1; then
    mark "$slug" "completed" "$output_jsonl.metrics.json"
  else
    mark "$slug" "failed" "$log_path"
    return 1
  fi
}

echo "host: $(hostname)"
echo "date: $(date)"
echo "cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
echo "out_root: $OUT_ROOT"
echo "max_samples: $MAX_SAMPLES"
nvidia-smi || true

run_eval \
  "minicpm_v46_kabr_lora_kabr_${MAX_SAMPLES}" \
  "$MINICPM_PY" \
  "kabr" \
  "$REPO/data/processed/kabr_behavior_val_cleaned.jsonl" \
  "openbmb/MiniCPM-V-4.6" \
  "$REPO/artifacts/runs/minicpm_v46_behavior_lora_full/checkpoint"

run_eval \
  "qwen25vl7b_kabr_lora_kabr_${MAX_SAMPLES}" \
  "$DEFAULT_PY" \
  "kabr" \
  "$REPO/data/processed/kabr_behavior_val_cleaned.jsonl" \
  "Qwen/Qwen2.5-VL-7B-Instruct" \
  "$REPO/artifacts/runs/qwen25vl7b_behavior_lora_full/checkpoint"

run_eval \
  "llava_ov_0_5b_kabr_lora_kabr_${MAX_SAMPLES}" \
  "$DEFAULT_PY" \
  "kabr" \
  "$REPO/data/processed/kabr_behavior_val_cleaned.jsonl" \
  "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" \
  "$REPO/artifacts/runs/llava_onevision_qwen2_0_5b_behavior_lora_full/checkpoint"

run_eval \
  "minicpm_v46_baboonland_lora_baboonland_${MAX_SAMPLES}" \
  "$MINICPM_PY" \
  "baboonland" \
  "$REPO/data/processed/baboonland_behavior_val.jsonl" \
  "openbmb/MiniCPM-V-4.6" \
  "$REPO/artifacts/runs/baboonland/minicpm_v46_behavior_lora_full/checkpoint"

run_eval \
  "llava_ov_0_5b_baboonland_lora_baboonland_${MAX_SAMPLES}" \
  "$DEFAULT_PY" \
  "baboonland" \
  "$REPO/data/processed/baboonland_behavior_val.jsonl" \
  "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" \
  "$REPO/artifacts/runs/baboonland/llava_onevision_qwen2_0_5b_behavior_lora/checkpoint"

echo "done: $(date)"
