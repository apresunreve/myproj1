#!/usr/bin/env bash
set -euo pipefail

REPO=/scratch/general/vast/j.yan/myproj1
MODEL_ID=${MODEL_ID:-OpenGVLab/InternVL2_5-4B}
ANNOTATION=${ANNOTATION:-/scratch/general/nfs1/u1320844/KABR_files2/annotation/val.cleaned.csv}
IMAGE_ROOT=${IMAGE_ROOT:-/scratch/general/nfs1/u1320844/KABR_files2/dataset/image}
CLASSES=${CLASSES:-/scratch/general/nfs1/u1320844/KABR_files2/annotation/classes.json}
OUT_ROOT=${OUT_ROOT:-/scratch/general/nfs1/u1320844/manual_eval/internvl2_5_4b_kabr_clip}
PREDICTIONS=${PREDICTIONS:-$OUT_ROOT/behavior_val_predictions.jsonl}
METRICS=${METRICS:-$OUT_ROOT/behavior_val_predictions.metrics.json}
CONFUSION=${CONFUSION:-$OUT_ROOT/behavior_val_predictions.confusion.csv}
TASK=${TASK:-behavior}
PROMPT_MODE=${PROMPT_MODE:-label}
MAX_SAMPLES=${MAX_SAMPLES:-3981}
MAX_FRAMES_PER_CLIP=${MAX_FRAMES_PER_CLIP:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8}
SEED=${SEED:-42}
RESUME=${RESUME:-false}

mkdir -p "$OUT_ROOT"
cd "$REPO"

if [ -x "$REPO/.uvenv/bin/python" ]; then
  RUN=( "$REPO/.uvenv/bin/python" -m kabr_vlm.cli )
else
  RUN=( /uufs/chpc.utah.edu/common/home/u1320844/.local/bin/uv run
    --project "$REPO"
    --extra infer
    python -m kabr_vlm.cli )
fi

EVAL_CMD=(
  "${RUN[@]}"
  eval-clip-hf-vlm
  --annotation "$ANNOTATION"
  --image-root "$IMAGE_ROOT"
  --classes "$CLASSES"
  --model-id "$MODEL_ID"
  --task "$TASK"
  --prompt-mode "$PROMPT_MODE"
  --max-samples "$MAX_SAMPLES"
  --max-frames-per-clip "$MAX_FRAMES_PER_CLIP"
  --seed "$SEED"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --output "$PREDICTIONS"
)

if [ "$RESUME" = "true" ]; then
  EVAL_CMD+=( --resume )
fi

SCORE_CMD=(
  "${RUN[@]}"
  score-predictions
  --predictions "$PREDICTIONS"
  --metrics-output "$METRICS"
  --confusion-output "$CONFUSION"
  --task "$TASK"
)

echo "=== Running InternVL2_5-4B KABR clip evaluation ==="
printf ' %q' "${EVAL_CMD[@]}"
printf '\n'
"${EVAL_CMD[@]}" 2>&1 | tee "$OUT_ROOT/eval.log"

echo "=== Scoring predictions ==="
printf ' %q' "${SCORE_CMD[@]}"
printf '\n'
"${SCORE_CMD[@]}" 2>&1 | tee "$OUT_ROOT/score.log"

echo "Done."
echo "Predictions: $PREDICTIONS"
echo "Metrics:     $METRICS"
echo "Confusion:   $CONFUSION"
