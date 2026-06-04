#!/usr/bin/env bash
set -euo pipefail
bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full_sft.sh \
  /scratch/general/vast/j.yan/myproj1/configs/llava_onevision_1_5_4b_behavior_full_sft_full.yaml \
  llava_onevision_1_5_4b_behavior_full_sft_full \
  lmms-lab/LLaVA-OneVision-1.5-4B-Instruct
