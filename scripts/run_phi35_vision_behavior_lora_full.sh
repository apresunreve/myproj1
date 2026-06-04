#!/usr/bin/env bash
set -euo pipefail

exec bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full.sh \
  /scratch/general/vast/j.yan/myproj1/configs/phi35_vision_behavior_lora_full.yaml \
  phi35_vision_behavior_lora_full \
  microsoft/Phi-3.5-vision-instruct
