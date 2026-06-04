#!/usr/bin/env bash
set -euo pipefail
bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_smoke_sft.sh \
  /scratch/general/vast/j.yan/myproj1/configs/idefics3_8b_behavior_full_sft_smoke.yaml \
  idefics3_8b_behavior_full_sft_smoke \
  HuggingFaceM4/Idefics3-8B-Llama3
