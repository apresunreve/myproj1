#!/usr/bin/env bash
set -euo pipefail
bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_smoke_sft.sh \
  /scratch/general/vast/j.yan/myproj1/configs/phi35_vision_behavior_full_sft_smoke.yaml \
  phi35_vision_behavior_full_sft_smoke \
  microsoft/Phi-3.5-vision-instruct
