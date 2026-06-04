#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <checkpoint_path> [cfg_path]"
  exit 1
fi

SLOWFAST_ROOT=${SLOWFAST_ROOT:-/scratch/general/vast/j.yan/SlowFast}
CHECKPOINT_PATH=$1
CFG_PATH=${2:-/scratch/general/vast/j.yan/myproj1/configs/x3d_clip_level.yaml}

CLIP_DATASET_ROOT=${CLIP_DATASET_ROOT:-/scratch/general/nfs1/u1320844/KABR_clip_dataset_cleaned_2026-04-07}
CLIP_MANIFEST_DIR=${CLIP_MANIFEST_DIR:-$CLIP_DATASET_ROOT/annotation/clip_manifests}
IMAGE_ROOT=${IMAGE_ROOT:-/scratch/general/nfs1/u1320844/KABR_files2/dataset/image}
CLASSES_PATH=${CLASSES_PATH:-/scratch/general/nfs1/u1320844/KABR_files2/annotation/classes.json}

OUT_ROOT=${OUT_ROOT:-/scratch/general/nfs1/u1320844/manual_smoke/x3d_clip_eval}
PY=${PYTHON_BIN:-/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python}
KABR_VLM_PY=${KABR_VLM_PY:-/uufs/chpc.utah.edu/common/home/u1320844/.local/bin/uv run --project /scratch/general/vast/j.yan/myproj1 --extra infer python}

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Missing checkpoint: $CHECKPOINT_PATH"
  exit 1
fi

if [ ! -d "$CLIP_MANIFEST_DIR" ]; then
  echo "Missing clip manifest directory: $CLIP_MANIFEST_DIR"
  exit 1
fi

if [ ! -f "$CLIP_MANIFEST_DIR/train.jsonl" ]; then
  echo "Missing train manifest: $CLIP_MANIFEST_DIR/train.jsonl"
  exit 1
fi

if [ ! -f "$CLIP_MANIFEST_DIR/val.jsonl" ]; then
  echo "Missing val manifest: $CLIP_MANIFEST_DIR/val.jsonl"
  exit 1
fi

if [ ! -f "$CLASSES_PATH" ]; then
  echo "Missing classes.json: $CLASSES_PATH"
  exit 1
fi

if [ ! -f "$CFG_PATH" ]; then
  echo "Missing clip-level SlowFast config: $CFG_PATH"
  echo "KABR_CLIP_EVAL_GUIDE.md expects a clip-level config."
  exit 1
fi

if [ ! -f "$SLOWFAST_ROOT/slowfast/datasets/kabr_clips.py" ]; then
  echo "Missing SlowFast clip dataset loader: $SLOWFAST_ROOT/slowfast/datasets/kabr_clips.py"
  echo "This SlowFast checkout does not currently match the clip-level guide."
  echo "Please switch to the branch/checkout that contains kabr_clips.py, then rerun."
  exit 1
fi

mkdir -p "$OUT_ROOT"
TMP_MANIFEST_DIR=$(mktemp -d)

cleanup() {
  rm -rf "$TMP_MANIFEST_DIR"
}
trap cleanup EXIT

ln -s "$CLIP_MANIFEST_DIR/train.jsonl" "$TMP_MANIFEST_DIR/train.jsonl"
ln -s "$CLIP_MANIFEST_DIR/val.jsonl" "$TMP_MANIFEST_DIR/val.jsonl"
ln -s "$CLASSES_PATH" "$TMP_MANIFEST_DIR/classes.json"

# SlowFast test mode usually looks for test.jsonl/csv. Mirror val -> test for clip eval.
if [ -f "$CLIP_MANIFEST_DIR/test.jsonl" ]; then
  ln -s "$CLIP_MANIFEST_DIR/test.jsonl" "$TMP_MANIFEST_DIR/test.jsonl"
else
  ln -s "$CLIP_MANIFEST_DIR/val.jsonl" "$TMP_MANIFEST_DIR/test.jsonl"
fi

cd "$SLOWFAST_ROOT"
export PYTHONPATH="$SLOWFAST_ROOT"

echo "=== X3D KABR clip-level eval ==="
echo "SlowFast root:  $SLOWFAST_ROOT"
echo "Config:         $CFG_PATH"
echo "Checkpoint:     $CHECKPOINT_PATH"
echo "Manifest dir:   $CLIP_MANIFEST_DIR"
echo "Image root:     $IMAGE_ROOT"
echo "Temp data dir:  $TMP_MANIFEST_DIR"
echo "Output dir:     $OUT_ROOT"

"$PY" tools/run_net.py \
  --cfg "$CFG_PATH" \
  --opts \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  TEST.CHECKPOINT_FILE_PATH "$CHECKPOINT_PATH" \
  DATA.PATH_TO_DATA_DIR "$TMP_MANIFEST_DIR" \
  DATA.PATH_PREFIX "$IMAGE_ROOT" \
  OUTPUT_DIR "$OUT_ROOT" \
  TEST.BATCH_SIZE 4 \
  DATA.NUM_FRAMES 16 \
  DATA.SAMPLING_RATE 5 \
  DATA.TRAIN_CROP_SIZE 300 \
  DATA.TEST_CROP_SIZE 300 \
  DATA.TRAIN_JITTER_SCALES "[300,300]" \
  DATA_LOADER.NUM_WORKERS 4 \
  LOG_MODEL_INFO False \
  2>&1 | tee "$OUT_ROOT/eval.log"

PREDICTIONS_PATH="$OUT_ROOT/clip_level/predictions.jsonl"
METRICS_PATH="$OUT_ROOT/clip_level/behavior.metrics.json"
CONFUSION_PATH="$OUT_ROOT/clip_level/behavior.confusion.csv"

if [ -f "$PREDICTIONS_PATH" ]; then
  echo "=== Scoring predictions ==="
  # shellcheck disable=SC2086
  $KABR_VLM_PY -m kabr_vlm.cli score-predictions \
    --predictions "$PREDICTIONS_PATH" \
    --metrics-output "$METRICS_PATH" \
    --confusion-output "$CONFUSION_PATH" \
    --task behavior \
    2>&1 | tee "$OUT_ROOT/score.log"
else
  echo "Predictions file not found at $PREDICTIONS_PATH; skipping scoring."
fi

echo "=== Done ==="
echo "Eval log:        $OUT_ROOT/eval.log"
echo "Predictions:     $PREDICTIONS_PATH"
echo "Metrics:         $METRICS_PATH"
echo "Confusion CSV:   $CONFUSION_PATH"
