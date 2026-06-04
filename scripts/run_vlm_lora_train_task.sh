#!/usr/bin/env bash
# Wrapper for the chpc_manager task_executor (command_type: bash).
# Accepts --key value pairs and delegates to scripts/run_vlm_<dataset>_full.sh.
#
# Required keys:
#   --dataset        kabr | baboonland
#   --config_path    absolute path to LoRA yaml config
#   --job_name       unique run name (used for log prefixes)
#   --model_id       HF model id (e.g. google/gemma-3-4b-it)
#   --output_dir     absolute path; used only for completion-file templating
# Optional:
#   --hf_token       HF token (for gated repos like Gemma3)
#
# The task_executor writes the completion file {output_dir}/training_metrics.json
# when the training script finishes.
set -euo pipefail

DATASET=""
CONFIG_PATH=""
JOB_NAME=""
MODEL_ID=""
OUTPUT_DIR=""
HF_TOKEN_VAL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET=$2; shift 2 ;;
        --config_path) CONFIG_PATH=$2; shift 2 ;;
        --job_name) JOB_NAME=$2; shift 2 ;;
        --model_id) MODEL_ID=$2; shift 2 ;;
        --output_dir) OUTPUT_DIR=$2; shift 2 ;;
        --hf_token) HF_TOKEN_VAL=$2; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

for req in DATASET CONFIG_PATH JOB_NAME MODEL_ID OUTPUT_DIR; do
    if [ -z "${!req}" ]; then
        echo "Missing required --${req,,}" >&2
        exit 2
    fi
done

REPO=/scratch/general/vast/j.yan/myproj1
export KABR_ENABLE_FLASH_ATTN=${KABR_ENABLE_FLASH_ATTN:-0}
if [ -n "$HF_TOKEN_VAL" ]; then
    export HF_TOKEN="$HF_TOKEN_VAL"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VAL"
fi

case "$DATASET" in
    baboonland)
        exec bash "$REPO/scripts/run_vlm_baboonland_full.sh" "$CONFIG_PATH" "$JOB_NAME" "$MODEL_ID"
        ;;
    kabr)
        exec bash "$REPO/scripts/run_vlm_behavior_full.sh" "$CONFIG_PATH" "$JOB_NAME" "$MODEL_ID"
        ;;
    joint|config)
        # Use the train/val paths from the config as-is. No path overrides.
        exec bash "$REPO/scripts/run_vlm_config_train.sh" "$CONFIG_PATH" "$JOB_NAME" "$MODEL_ID"
        ;;
    *)
        echo "Unknown dataset: $DATASET" >&2
        exit 2
        ;;
esac
