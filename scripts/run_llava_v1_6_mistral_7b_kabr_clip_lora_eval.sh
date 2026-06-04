#!/usr/bin/env bash
set -euo pipefail

REPO=/scratch/general/vast/j.yan/myproj1
MODEL_ID=${MODEL_ID:-llava-hf/llava-v1.6-mistral-7b-hf}
ADAPTER_PATH=${ADAPTER_PATH:-$REPO/artifacts/runs/llava_v1_6_mistral_7b_behavior_lora_full/checkpoint}
ANNOTATION=${ANNOTATION:-/scratch/general/nfs1/u1320844/KABR_files2/annotation/val.cleaned.csv}
IMAGE_ROOT=${IMAGE_ROOT:-/scratch/general/nfs1/u1320844/KABR_files2/dataset/image}
CLASSES=${CLASSES:-/scratch/general/nfs1/u1320844/KABR_files2/annotation/classes.json}
OUT_ROOT=${OUT_ROOT:-$REPO/artifacts/runs/llava_v1_6_mistral_7b_behavior_lora_full}
PREDICTIONS=${PREDICTIONS:-$OUT_ROOT/behavior_val_predictions.jsonl}
METRICS=${METRICS:-$OUT_ROOT/behavior_val_predictions.metrics.json}
CONFUSION=${CONFUSION:-$OUT_ROOT/behavior_val_predictions.confusion.csv}
TASK=${TASK:-behavior}
PROMPT_MODE=${PROMPT_MODE:-label}
MAX_SAMPLES=${MAX_SAMPLES:-3981}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8}
SEED=${SEED:-42}

mkdir -p "$OUT_ROOT"
cd "$REPO"

if [ ! -d "$ADAPTER_PATH" ]; then
  echo "Missing adapter directory: $ADAPTER_PATH"
  exit 1
fi

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
  eval-finetuned-hf-vlm
  --annotation "$ANNOTATION"
  --image-root "$IMAGE_ROOT"
  --classes "$CLASSES"
  --model-id "$MODEL_ID"
  --adapter-path "$ADAPTER_PATH"
  --task "$TASK"
  --prompt-mode "$PROMPT_MODE"
  --max-samples "$MAX_SAMPLES"
  --seed "$SEED"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --output "$PREDICTIONS"
)

SCORE_CMD=(
  "${RUN[@]}"
  score-predictions
  --predictions "$PREDICTIONS"
  --metrics-output "$METRICS"
  --confusion-output "$CONFUSION"
  --task "$TASK"
)

echo "=== Running LLaVA v1.6 Mistral 7B LoRA KABR clip eval ==="
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
