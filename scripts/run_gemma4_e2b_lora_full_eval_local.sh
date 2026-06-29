#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/scratch/general/vast/j.yan/myproj1}"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv_minicpm_smoke/bin/python}"
MODEL_ID="${MODEL_ID:-google/gemma-4-E2B-it}"
OUT_ROOT="${OUT_ROOT:-$REPO/artifacts/eval/gemma4_e2b_lora_full}"
LOG_ROOT="${LOG_ROOT:-$REPO/logs/gemma4_e2b_lora_full_eval}"
MAX_INPUT_FRAMES="${MAX_INPUT_FRAMES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

KABR_VAL_JSONL="${KABR_VAL_JSONL:-$REPO/data/processed/kabr_behavior_val_cleaned.jsonl}"
KABR_ADAPTER="${KABR_ADAPTER:-$REPO/artifacts/runs/gemma4_e2b_behavior_lora_full/checkpoint}"
KABR_OUTPUT="${KABR_OUTPUT:-$OUT_ROOT/kabr_behavior_val_predictions.jsonl}"

BABOONLAND_VAL_JSONL="${BABOONLAND_VAL_JSONL:-$REPO/data/processed/baboonland_behavior_val_from_joint.jsonl}"
BABOONLAND_ADAPTER="${BABOONLAND_ADAPTER:-$REPO/artifacts/runs/baboonland/gemma4_e2b_behavior_lora_full/checkpoint}"
BABOONLAND_OUTPUT="${BABOONLAND_OUTPUT:-$OUT_ROOT/baboonland_behavior_val_predictions.jsonl}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/general/nfs1/u1320844/huggingface-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/general/nfs1/u1320844/xdg-cache}"

cd "$REPO"

run_eval() {
  local name="$1"
  local target_dataset="$2"
  local val_jsonl="$3"
  local adapter_path="$4"
  local output_jsonl="$5"
  local log_path="$LOG_ROOT/${name}.log"

  echo "=== $(date -Is) starting $name ==="
  echo "val_jsonl=$val_jsonl"
  echo "adapter_path=$adapter_path"
  echo "output_jsonl=$output_jsonl"

  "$PYTHON_BIN" "$REPO/scripts/eval_cross_dataset_vlm.py" \
    --val_jsonl "$val_jsonl" \
    --target_dataset "$target_dataset" \
    --model_id "$MODEL_ID" \
    --adapter_path "$adapter_path" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_input_frames "$MAX_INPUT_FRAMES" \
    --frame_selection uniform \
    --output_jsonl "$output_jsonl" \
    --resume 2>&1 | tee "$log_path"

  echo "=== $(date -Is) finished $name ==="
}

run_eval "kabr" "kabr" "$KABR_VAL_JSONL" "$KABR_ADAPTER" "$KABR_OUTPUT"
run_eval "baboonland" "baboonland" "$BABOONLAND_VAL_JSONL" "$BABOONLAND_ADAPTER" "$BABOONLAND_OUTPUT"

echo "Done."
echo "KABR metrics: ${KABR_OUTPUT}.metrics.json"
echo "BaboonLand metrics: ${BABOONLAND_OUTPUT}.metrics.json"
