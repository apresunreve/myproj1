#!/usr/bin/env bash
set -euo pipefail
bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_smoke_sft.sh \
  /scratch/general/vast/j.yan/myproj1/configs/llava_onevision_qwen2_0_5b_behavior_full_sft_smoke.yaml \
  llava_onevision_qwen2_0_5b_behavior_full_sft_smoke \
  llava-hf/llava-onevision-qwen2-0.5b-ov-hf
