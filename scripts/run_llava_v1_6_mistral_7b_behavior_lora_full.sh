#!/usr/bin/env bash
set -euo pipefail

exec bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full.sh \
  /scratch/general/vast/j.yan/myproj1/configs/llava_v1_6_mistral_7b_behavior_lora_full.yaml \
  llava_v1_6_mistral_7b_behavior_lora_full \
  llava-hf/llava-v1.6-mistral-7b-hf
