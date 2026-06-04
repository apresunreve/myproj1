#!/usr/bin/env bash
set -euo pipefail

exec bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full.sh \
  /scratch/general/vast/j.yan/myproj1/configs/llava_onevision_1_5_4b_behavior_lora_full.yaml \
  llava_onevision_1_5_4b_behavior_lora_full \
  lmms-lab/LLaVA-OneVision-1.5-4B-Instruct
