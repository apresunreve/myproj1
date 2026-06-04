#!/usr/bin/env bash
set -euo pipefail
bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full_sft.sh \
  /scratch/general/vast/j.yan/myproj1/configs/smolvlm_256m_behavior_full_sft_full.yaml \
  smolvlm_256m_behavior_full_sft_full \
  HuggingFaceTB/SmolVLM-256M-Instruct
