#!/usr/bin/env bash
set -euo pipefail

exec bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full.sh \
  /scratch/general/vast/j.yan/myproj1/configs/internvl2_5_2b_behavior_lora_full.yaml \
  internvl2_5_2b_behavior_lora_full \
  OpenGVLab/InternVL2_5-2B
