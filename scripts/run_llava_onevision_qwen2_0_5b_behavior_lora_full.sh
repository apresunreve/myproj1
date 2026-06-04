#!/usr/bin/env bash
set -euo pipefail

exec bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full.sh \
  /scratch/general/vast/j.yan/myproj1/configs/llava_onevision_qwen2_0_5b_behavior_lora_full.yaml \
  llava_onevision_qwen2_0_5b_behavior_lora_full \
  llava-hf/llava-onevision-qwen2-0.5b-ov-hf
