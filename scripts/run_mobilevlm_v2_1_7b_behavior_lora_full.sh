#!/usr/bin/env bash
set -euo pipefail

exec bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_full.sh \
  /scratch/general/vast/j.yan/myproj1/configs/mobilevlm_v2_1_7b_behavior_lora_full.yaml \
  mobilevlm_v2_1_7b_behavior_lora_full \
  mtgv/MobileVLM_V2-1.7B
