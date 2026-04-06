#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.uvenv/bin/python}"
MODEL_LIST="${MODEL_LIST:-$ROOT_DIR/configs/vlm_eval_models.tsv}"
ANNOTATION_PATH="${ANNOTATION_PATH:-$ROOT_DIR/../KABR_files/annotation/val.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-$ROOT_DIR/../KABR_files/dataset/image}"
CLASSES_PATH="${CLASSES_PATH:-$ROOT_DIR/../KABR_files/annotation/classes.json}"
TASK="${TASK:-behavior}"
PROMPT_MODE="${PROMPT_MODE:-plain}"
MAX_SAMPLES="${MAX_SAMPLES:-3761}"
MAX_FRAMES_PER_CLIP="${MAX_FRAMES_PER_CLIP:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/vlm_suite}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/artifacts/full_eval}"
STATUS_FILE="${STATUS_FILE:-$OUT_DIR/vlm_eval_suite_status.tsv}"
CURRENT_FILE="${CURRENT_FILE:-$OUT_DIR/vlm_eval_suite_current.txt}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$HF_HOME"
export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTHONUNBUFFERED=1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python env not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$MODEL_LIST" ]]; then
  echo "Model list not found: $MODEL_LIST" >&2
  exit 1
fi

if [[ ! -f "$STATUS_FILE" ]]; then
  printf "timestamp\tslug\tstatus\tdetail\n" > "$STATUS_FILE"
fi

timestamp() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

mark_status() {
  local slug="$1"
  local status="$2"
  local detail="$3"
  printf "%s\t%s\t%s\t%s\n" "$(timestamp)" "$slug" "$status" "$detail" >> "$STATUS_FILE"
  printf "%s\t%s\t%s\n" "$(timestamp)" "$slug" "$status" > "$CURRENT_FILE"
}

run_model() {
  local slug="$1"
  local model_id="$2"
  local pred_file="$OUT_DIR/${slug}.jsonl"
  local metrics_file="$OUT_DIR/${slug}.metrics.json"
  local confusion_file="$OUT_DIR/${slug}.confusion.csv"
  local smoke_file="$OUT_DIR/${slug}.smoke.jsonl"
  local model_log="$LOG_DIR/${slug}.log"

  if [[ -f "$metrics_file" ]]; then
    mark_status "$slug" "skipped" "metrics already exist"
    return 0
  fi

  mark_status "$slug" "starting" "$model_id"
  rm -f "$smoke_file" "$pred_file" "$confusion_file"

  if {
    echo "[$(timestamp)] model_id=$model_id"
    echo "[$(timestamp)] smoke test"
    "$PYTHON_BIN" -m kabr_vlm.cli eval-clip-hf-vlm \
      --annotation "$ANNOTATION_PATH" \
      --image-root "$IMAGE_ROOT" \
      --classes "$CLASSES_PATH" \
      --model-id "$model_id" \
      --task "$TASK" \
      --prompt-mode "$PROMPT_MODE" \
      --max-samples 1 \
      --max-frames-per-clip "$MAX_FRAMES_PER_CLIP" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --output "$smoke_file"

    echo "[$(timestamp)] full evaluation"
    "$PYTHON_BIN" -m kabr_vlm.cli eval-clip-hf-vlm \
      --annotation "$ANNOTATION_PATH" \
      --image-root "$IMAGE_ROOT" \
      --classes "$CLASSES_PATH" \
      --model-id "$model_id" \
      --task "$TASK" \
      --prompt-mode "$PROMPT_MODE" \
      --max-samples "$MAX_SAMPLES" \
      --max-frames-per-clip "$MAX_FRAMES_PER_CLIP" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --output "$pred_file"

    echo "[$(timestamp)] scoring"
    "$PYTHON_BIN" -m kabr_vlm.cli score-predictions \
      --predictions "$pred_file" \
      --metrics-output "$metrics_file" \
      --confusion-output "$confusion_file"
  } >> "$model_log" 2>&1; then
    if [[ -f "$metrics_file" ]]; then
      mark_status "$slug" "completed" "$metrics_file"
      return 0
    fi

    mark_status "$slug" "failed" "$model_log"
    return 1
  fi

  mark_status "$slug" "failed" "$model_log"
  return 1
}

tail -n +2 "$MODEL_LIST" | while IFS=$'\t' read -r slug model_id; do
  [[ -z "$slug" ]] && continue
  if run_model "$slug" "$model_id"; then
    :
  else
    continue
  fi
done

mark_status "suite" "completed" "$STATUS_FILE"
