#!/usr/bin/env bash
set -euo pipefail

exec bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full.sh \
  /scratch/general/vast/j.yan/myproj1/configs/qwen25vl7b_behavior_lora_full_rerun1.yaml \
  qwen25vl7b_behavior_lora_full_rerun1 \
  Qwen/Qwen2.5-VL-7B-Instruct
