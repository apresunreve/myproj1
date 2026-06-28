#!/usr/bin/env bash
set -euo pipefail

exec bash /scratch/general/vast/j.yan/myproj1/scripts/run_gemma4_e2b_config_lora.sh \
  /scratch/general/vast/j.yan/myproj1/configs/gemma4_e2b_baboonland_behavior_lora_full.yaml \
  gemma4_e2b_baboonland_lora_full \
  google/gemma-4-E2B-it
