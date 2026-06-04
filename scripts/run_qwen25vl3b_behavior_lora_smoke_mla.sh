#!/usr/bin/env bash
set -euo pipefail

export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-16}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-8}"

exec /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_smoke.sh \
  /scratch/general/vast/j.yan/myproj1/configs/qwen25vl3b_behavior_lora_smoke_mla.yaml \
  qwen25vl3b_behavior_lora_smoke_mla \
  Qwen/Qwen2.5-VL-3B-Instruct
