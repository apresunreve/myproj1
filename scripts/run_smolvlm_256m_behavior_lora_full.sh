#!/usr/bin/env bash
set -euo pipefail

exec bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full.sh \
  /scratch/general/vast/j.yan/myproj1/configs/smolvlm_256m_behavior_lora_full.yaml \
  smolvlm_256m_behavior_lora_full \
  HuggingFaceTB/SmolVLM-256M-Instruct
