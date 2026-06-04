#!/usr/bin/env bash
set -euo pipefail

export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-16}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-8}"

exec /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_smoke.sh \
  /scratch/general/vast/j.yan/myproj1/configs/idefics3_8b_behavior_lora_smoke.yaml \
  idefics3_8b_behavior_lora_smoke \
  HuggingFaceM4/Idefics3-8B-Llama3
