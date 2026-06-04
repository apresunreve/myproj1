#!/usr/bin/env bash
set -euo pipefail

exec bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full.sh \
  /scratch/general/vast/j.yan/myproj1/configs/gemma3_4b_behavior_lora_full.yaml \
  gemma3_4b_behavior_lora_full \
  google/gemma-3-4b-it
