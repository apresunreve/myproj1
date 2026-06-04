#!/usr/bin/env bash
set -euo pipefail

exec bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full.sh \
  /scratch/general/vast/j.yan/myproj1/configs/qwen25vl3b_behavior_lora_full_mla.yaml \
  qwen25vl3b_behavior_lora_full_mla \
  Qwen/Qwen2.5-VL-3B-Instruct
