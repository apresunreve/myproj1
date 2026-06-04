#!/usr/bin/env bash
set -euo pipefail

exec bash /scratch/general/vast/j.yan/myproj1/scripts/run_vlm_baboonland_full.sh \
  /scratch/general/vast/j.yan/myproj1/configs/gemma3_4b_baboonland_behavior_lora_smoke.yaml \
  gemma3_4b_baboonland_lora_smoke \
  google/gemma-3-4b-it
