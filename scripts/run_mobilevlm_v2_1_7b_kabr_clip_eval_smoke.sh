#!/usr/bin/env bash
set -euo pipefail

export MAX_SAMPLES="${MAX_SAMPLES:-16}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
export OUT_ROOT="${OUT_ROOT:-/scratch/general/nfs1/u1320844/manual_smoke/vlm/mobilevlm_v2_1_7b_clip_eval}"

exec /scratch/general/vast/j.yan/myproj1/scripts/run_mobilevlm_v2_1_7b_kabr_clip_eval.sh
