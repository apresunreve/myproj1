#!/usr/bin/env bash
set -euo pipefail
bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_smoke_sft.sh \
  /scratch/general/vast/j.yan/myproj1/configs/qwen25vl7b_behavior_full_sft_smoke.yaml \
  qwen25vl7b_behavior_full_sft_smoke \
  Qwen/Qwen2.5-VL-7B-Instruct
