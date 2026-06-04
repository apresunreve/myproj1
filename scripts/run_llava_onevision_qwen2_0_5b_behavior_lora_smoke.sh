#!/usr/bin/env bash
set -euo pipefail

export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-16}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-8}"

exec /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_smoke.sh \
  /scratch/general/vast/j.yan/myproj1/configs/llava_onevision_qwen2_0_5b_behavior_lora_smoke.yaml \
  llava_onevision_qwen2_0_5b_behavior_lora_smoke \
  llava-hf/llava-onevision-qwen2-0.5b-ov-hf
