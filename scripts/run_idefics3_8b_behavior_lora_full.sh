#!/usr/bin/env bash
set -euo pipefail

exec bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full.sh \
  /scratch/general/vast/j.yan/myproj1/configs/idefics3_8b_behavior_lora_full.yaml \
  idefics3_8b_behavior_lora_full \
  HuggingFaceM4/Idefics3-8B-Llama3
