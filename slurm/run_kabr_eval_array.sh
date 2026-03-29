#!/usr/bin/env bash
#SBATCH --ntasks=1
set -euo pipefail

ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
JOB_FILE="${JOB_FILE:?JOB_FILE is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

if [[ ! -f "$JOB_FILE" ]]; then
  echo "Job file not found: $JOB_FILE" >&2
  exit 1
fi

JOB_LINE="$(tail -n +2 "$JOB_FILE" | sed -n "${TASK_ID}p")"
if [[ -z "$JOB_LINE" ]]; then
  echo "No job definition for array index $TASK_ID" >&2
  exit 1
fi

IFS=$'\t' read -r JOB_NAME COMMAND MODEL_ID TASK PROMPT_MODE MAX_SAMPLES MAX_FRAMES_PER_CLIP MAX_NEW_TOKENS OUTPUT_NAME <<< "$JOB_LINE"

ANNOTATION_PATH="${ANNOTATION_PATH:-$ROOT_DIR/../KABR_files/annotation/val.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-$ROOT_DIR/../KABR_files/dataset/image}"
CLASSES_PATH="${CLASSES_PATH:-$ROOT_DIR/../KABR_files/annotation/classes.json}"
SEED="${SEED:-42}"
HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
UV_PROJECT_ENV="${UV_PROJECT_ENV:-$ROOT_DIR/.uvenv}"
PYTHON_BIN="${PYTHON_BIN:-$UV_PROJECT_ENV/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/artifacts/slurm_eval}"

mkdir -p "$RESULTS_DIR" "$ROOT_DIR/slurm/logs" "$HF_HOME"

echo "job_name=$JOB_NAME"
echo "command=$COMMAND"
echo "model_id=$MODEL_ID"
echo "task=$TASK"
echo "prompt_mode=$PROMPT_MODE"
echo "max_samples=$MAX_SAMPLES"
echo "output=$RESULTS_DIR/$OUTPUT_NAME"
echo "hostname=$(hostname)"

if [[ ! -x "$PYTHON_BIN" || ! -x "$UV_BIN" ]]; then
  echo "Expected uv and a uv-managed environment at $UV_PROJECT_ENV." >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME"

"$UV_BIN" run --python "$PYTHON_BIN" python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
    print("device_name", torch.cuda.get_device_name(0))
PY

"$UV_BIN" run --python "$PYTHON_BIN" kabr-vlm "$COMMAND" \
  --annotation "$ANNOTATION_PATH" \
  --image-root "$IMAGE_ROOT" \
  --classes "$CLASSES_PATH" \
  --model-id "$MODEL_ID" \
  --task "$TASK" \
  --prompt-mode "$PROMPT_MODE" \
  --max-samples "$MAX_SAMPLES" \
  --seed "$SEED" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-frames-per-clip "$MAX_FRAMES_PER_CLIP" \
  --output "$RESULTS_DIR/$OUTPUT_NAME"
