#!/usr/bin/env bash
set -euo pipefail
bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_smoke_sft.sh \
  /scratch/general/vast/j.yan/myproj1/configs/gemma3_4b_behavior_full_sft_smoke.yaml \
  gemma3_4b_behavior_full_sft_smoke \
  google/gemma-3-4b-it
