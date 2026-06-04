#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <checkpoint_path> [cfg_path]"
  exit 1
fi

SLOWFAST=/scratch/general/vast/j.yan/NAS/SlowFast
CHECKPOINT_PATH=$1
CFG=${2:-/scratch/general/vast/j.yan/NAS/SlowFast/configs/Charades/X3D_L_KABR.yaml}
ANN_DIR=/scratch/general/nfs1/u1320844/KABR_files2_cleaned_annotations
OUT_ROOT=${OUT_ROOT:-/scratch/general/nfs1/u1320844/manual_smoke/x3d_eval}
PY=/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Missing checkpoint: $CHECKPOINT_PATH"
  exit 1
fi

mkdir -p "$OUT_ROOT"

cd "$SLOWFAST"
export PYTHONPATH="$SLOWFAST"

echo "=== X3D KABR eval smoke ==="
echo "Config: $CFG"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output: $OUT_ROOT"

"$PY" tools/run_net.py \
  --cfg "$CFG" \
  --opts \
  DATA.PATH_TO_DATA_DIR "$ANN_DIR" \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  TEST.CHECKPOINT_FILE_PATH "$CHECKPOINT_PATH" \
  DATA_LOADER.NUM_WORKERS 0 \
  OUTPUT_DIR "$OUT_ROOT" \
  2>&1 | tee "$OUT_ROOT/eval.log"

echo "=== Done ==="
echo "Eval log: $OUT_ROOT/eval.log"
