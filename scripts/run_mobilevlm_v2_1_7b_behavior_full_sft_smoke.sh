#!/usr/bin/env bash
set -euo pipefail
bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_behavior_smoke_sft.sh \
  /scratch/general/vast/j.yan/myproj1/configs/mobilevlm_v2_1_7b_behavior_full_sft_smoke.yaml \
  mobilevlm_v2_1_7b_behavior_full_sft_smoke \
  mtgv/MobileVLM_V2-1.7B
