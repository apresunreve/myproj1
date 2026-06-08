#!/usr/bin/env bash
set -euo pipefail

REPO=/scratch/general/vast/j.yan/myproj1
cd "$REPO"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$REPO/src"
export KABR_ENABLE_FLASH_ATTN="${KABR_ENABLE_FLASH_ATTN:-0}"

PYTHON_BIN="$REPO/.venv_minicpm_smoke/bin/python"
OUT_ROOT="${OUT_ROOT:-$REPO/artifacts/smoke/efficiency_phase1_minicpm_v46}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
MODEL_ID="openbmb/MiniCPM-V-4.6"
STATUS_FILE="$OUT_ROOT/status.tsv"

mkdir -p "$OUT_ROOT"
printf "timestamp\tslug\tstatus\tdetail\n" > "$STATUS_FILE"

timestamp() { date +"%Y-%m-%dT%H:%M:%S%z"; }
mark() {
  printf "%s\t%s\t%s\t%s\n" "$(timestamp)" "$1" "$2" "$3" >> "$STATUS_FILE"
}

run_eval() {
  local slug="$1"
  local target_dataset="$2"
  local val_jsonl="$3"
  local adapter_path="$4"
  local output_jsonl="$OUT_ROOT/${slug}.jsonl"
  local log_path="$OUT_ROOT/${slug}.log"

  rm -f "$output_jsonl" "$output_jsonl.metrics.json"
  mark "$slug" "running" "$output_jsonl"
  if "$PYTHON_BIN" "$REPO/scripts/eval_cross_dataset_vlm.py" \
      --val_jsonl "$val_jsonl" \
      --target_dataset "$target_dataset" \
      --model_id "$MODEL_ID" \
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
nvidia-smi || true

run_eval \
  "minicpm_v46_kabr_lora_on_kabr_1sample" \
  "kabr" \
  "$REPO/data/processed/kabr_behavior_val_cleaned.jsonl" \
  "$REPO/artifacts/runs/minicpm_v46_behavior_lora_full/checkpoint"

run_eval \
  "minicpm_v46_baboonland_lora_on_baboonland_1sample" \
  "baboonland" \
  "$REPO/data/processed/baboonland_behavior_val.jsonl" \
  "$REPO/artifacts/runs/baboonland/minicpm_v46_behavior_lora_full/checkpoint"

run_eval \
  "minicpm_v46_joint_lora_on_kabr_1sample" \
  "kabr" \
  "$REPO/data/processed/kabr_behavior_val_cleaned.jsonl" \
  "$REPO/artifacts/runs/joint/minicpm_v46_joint_kabr_bbl_lora_full/checkpoint"

run_eval \
  "minicpm_v46_joint_lora_on_baboonland_1sample" \
  "baboonland" \
  "$REPO/data/processed/baboonland_behavior_val.jsonl" \
  "$REPO/artifacts/runs/joint/minicpm_v46_joint_kabr_bbl_lora_full/checkpoint"

echo "done: $(date)"
