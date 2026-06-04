#!/usr/bin/env bash
set -euo pipefail
bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full_sft.sh \
  /scratch/general/vast/j.yan/myproj1/configs/internvl2_5_2b_behavior_full_sft_full.yaml \
  internvl2_5_2b_behavior_full_sft_full \
  OpenGVLab/InternVL2_5-2B
