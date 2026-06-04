#!/usr/bin/env bash
set -euo pipefail

export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-16}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-8}"

exec /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_smoke.sh \
  /scratch/general/vast/j.yan/myproj1/configs/smolvlm_256m_behavior_lora_smoke.yaml \
  smolvlm_256m_behavior_lora_smoke \
  HuggingFaceTB/SmolVLM-256M-Instruct
