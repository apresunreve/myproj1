#!/usr/bin/env bash
set -euo pipefail
bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full_sft.sh \
  /scratch/general/vast/j.yan/myproj1/configs/llava_v1_6_mistral_7b_behavior_full_sft_full.yaml \
  llava_v1_6_mistral_7b_behavior_full_sft_full \
  llava-hf/llava-v1.6-mistral-7b-hf
