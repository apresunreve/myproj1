#!/usr/bin/env bash
# Wrapper for cross-dataset VLM eval jobs run via chpc_manager.
#
# Required keys:
#   --val_jsonl        absolute path to processed JSONL (KABR or baboonland)
#   --target_dataset   kabr | baboonland
#   --model_id         HF model id
#   --output_jsonl     absolute path; {output_jsonl}.metrics.json is the completion file
# Optional:
#   --adapter_path     LoRA adapter dir (omit for zero-shot)
#   --max_samples      integer
#   --max_new_tokens   integer (default 16)
#   --hf_token         HF token
set -euo pipefail

VAL_JSONL=""
TARGET_DATASET=""
MODEL_ID=""
OUTPUT_JSONL=""
ADAPTER_PATH=""
MAX_SAMPLES=""
MAX_NEW_TOKENS=""
HF_TOKEN_VAL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --val_jsonl) VAL_JSONL=$2; shift 2 ;;
        --target_dataset) TARGET_DATASET=$2; shift 2 ;;
        --model_id) MODEL_ID=$2; shift 2 ;;
        --output_jsonl) OUTPUT_JSONL=$2; shift 2 ;;
        --adapter_path) ADAPTER_PATH=$2; shift 2 ;;
        --max_samples) MAX_SAMPLES=$2; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS=$2; shift 2 ;;
        --hf_token) HF_TOKEN_VAL=$2; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

for req in VAL_JSONL TARGET_DATASET MODEL_ID OUTPUT_JSONL; do
    if [ -z "${!req}" ]; then
        echo "Missing required --${req,,}" >&2
        exit 2
    fi
done

REPO=/scratch/general/vast/j.yan/myproj1
export KABR_ENABLE_FLASH_ATTN=${KABR_ENABLE_FLASH_ATTN:-0}
CACHE_ROOT=${CACHE_ROOT:-/scratch/general/nfs1/u1320844}
export HF_HOME=${HF_HOME:-$CACHE_ROOT/huggingface-cache}
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
if [ -n "$HF_TOKEN_VAL" ]; then
    export HF_TOKEN="$HF_TOKEN_VAL"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VAL"
fi

echo "host: $(hostname)"
echo "date: $(date)"
echo "target: $TARGET_DATASET  model: $MODEL_ID  adapter: ${ADAPTER_PATH:-<none>}"
nvidia-smi || true

cd "$REPO"

CMD=("$REPO/.venv/bin/python" "$REPO/scripts/eval_cross_dataset_vlm.py"
     --model_id "$MODEL_ID"
     --val_jsonl "$VAL_JSONL"
     --target_dataset "$TARGET_DATASET"
     --output_jsonl "$OUTPUT_JSONL"
     --max_new_tokens "${MAX_NEW_TOKENS:-16}"
     --resume)
if [ -n "$ADAPTER_PATH" ]; then
    CMD+=(--adapter_path "$ADAPTER_PATH")
fi
if [ -n "$MAX_SAMPLES" ]; then
    CMD+=(--max_samples "$MAX_SAMPLES")
fi

"${CMD[@]}"
echo "done: $(date)"
