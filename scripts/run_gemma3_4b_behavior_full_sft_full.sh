#!/usr/bin/env bash
set -euo pipefail
bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full_sft.sh \
  /scratch/general/vast/j.yan/myproj1/configs/gemma3_4b_behavior_full_sft_full.yaml \
  gemma3_4b_behavior_full_sft_full \
  google/gemma-3-4b-it
