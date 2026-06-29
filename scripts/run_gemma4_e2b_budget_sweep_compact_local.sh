#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/scratch/general/vast/j.yan/myproj1}"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv_minicpm_smoke/bin/python}"
OUT_ROOT="${OUT_ROOT:-$REPO/artifacts/smoke_eval/visual_tokens/gemma4_e2b_lora_budget_sweep_compact}"
LOG_ROOT="${LOG_ROOT:-$REPO/logs/gemma4_e2b_lora_budget_sweep_compact}"
MODEL_ID="${MODEL_ID:-google/gemma-4-E2B-it}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-5}"
SOFT_TOKEN_BUDGETS="${SOFT_TOKEN_BUDGETS:-70,140,280}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/general/nfs1/u1320844/huggingface-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/general/nfs1/u1320844/xdg-cache}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
cd "$REPO"

run_one() {
  local dataset="$1"
  local frames="$2"
  local val_jsonl="$3"
  local adapter_path="$4"
  local output_jsonl="$OUT_ROOT/${dataset}_${frames}frames.jsonl"
  local summary_json="$OUT_ROOT/${dataset}_${frames}frames.summary.json"
  local log_path="$LOG_ROOT/${dataset}_${frames}frames.log"

  echo "=== $(date -Is) ${dataset} frames=${frames} ==="
  "$PYTHON_BIN" "$REPO/scripts/eval_gemma4_soft_token_budget.py" \
    --val-jsonl "$val_jsonl" \
    --target-dataset "$dataset" \
    --model-id "$MODEL_ID" \
    --adapter-path "$adapter_path" \
    --samples-per-class "$SAMPLES_PER_CLASS" \
    --max-input-frames "$frames" \
    --soft-token-budgets "$SOFT_TOKEN_BUDGETS" \
    --warmup-budget 70 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output-jsonl "$output_jsonl" \
    --summary-json "$summary_json" 2>&1 | tee "$log_path"
}

for frames in 1 2 4; do
  run_one \
    kabr \
    "$frames" \
    "$REPO/data/processed/kabr_behavior_val_cleaned.jsonl" \
    "$REPO/artifacts/runs/gemma4_e2b_behavior_lora_full/checkpoint"
done

for frames in 1 2 4; do
  run_one \
    baboonland \
    "$frames" \
    "$REPO/data/processed/baboonland_behavior_val_from_joint.jsonl" \
    "$REPO/artifacts/runs/baboonland/gemma4_e2b_behavior_lora_full/checkpoint"
done

echo "Done."
echo "Summaries: $OUT_ROOT/*.summary.json"
