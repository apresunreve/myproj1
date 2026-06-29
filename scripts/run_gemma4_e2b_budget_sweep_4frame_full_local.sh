#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/scratch/general/vast/j.yan/myproj1}"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv_minicpm_smoke/bin/python}"
OUT_ROOT="${OUT_ROOT:-$REPO/artifacts/eval/gemma4_e2b_lora_budget_sweep_4frame_full}"
LOG_ROOT="${LOG_ROOT:-$REPO/logs/gemma4_e2b_lora_budget_sweep_4frame_full}"
MODEL_ID="${MODEL_ID:-google/gemma-4-E2B-it}"
SOFT_TOKEN_BUDGETS="${SOFT_TOKEN_BUDGETS:-70,140,280}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
MAX_INPUT_FRAMES="${MAX_INPUT_FRAMES:-4}"
MAX_ROWS="${MAX_ROWS:-1000000000}"

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/general/nfs1/u1320844/huggingface-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/general/nfs1/u1320844/xdg-cache}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
cd "$REPO"

run_one() {
  local dataset="$1"
  local val_jsonl="$2"
  local adapter_path="$3"
  local output_jsonl="$OUT_ROOT/${dataset}_4frames_full.jsonl"
  local summary_json="$OUT_ROOT/${dataset}_4frames_full.summary.json"
  local log_path="$LOG_ROOT/${dataset}_4frames_full.log"

  if [ -f "$summary_json" ]; then
    echo "Skipping $dataset because summary exists: $summary_json"
    return
  fi

  echo "=== $(date -Is) ${dataset} 4-frame full budget sweep ==="
  "$PYTHON_BIN" "$REPO/scripts/eval_gemma4_soft_token_budget.py" \
    --val-jsonl "$val_jsonl" \
    --target-dataset "$dataset" \
    --model-id "$MODEL_ID" \
    --adapter-path "$adapter_path" \
    --max-rows "$MAX_ROWS" \
    --max-input-frames "$MAX_INPUT_FRAMES" \
    --soft-token-budgets "$SOFT_TOKEN_BUDGETS" \
    --warmup-budget 70 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output-jsonl "$output_jsonl" \
    --summary-json "$summary_json" 2>&1 | tee "$log_path"
}

run_one \
  kabr \
  "$REPO/data/processed/kabr_behavior_val_cleaned.jsonl" \
  "$REPO/artifacts/runs/gemma4_e2b_behavior_lora_full/checkpoint"

run_one \
  baboonland \
  "$REPO/data/processed/baboonland_behavior_val_from_joint.jsonl" \
  "$REPO/artifacts/runs/baboonland/gemma4_e2b_behavior_lora_full/checkpoint"

echo "Done."
echo "Summaries: $OUT_ROOT/*.summary.json"
