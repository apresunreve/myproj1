#!/usr/bin/env bash
set -euo pipefail

export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-16}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-8}"

exec /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_smoke.sh \
  /scratch/general/vast/j.yan/myproj1/configs/llava_v1_6_mistral_7b_behavior_lora_smoke.yaml \
  llava_v1_6_mistral_7b_behavior_lora_smoke \
  llava-hf/llava-v1.6-mistral-7b-hf
