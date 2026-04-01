#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PID_TO_WATCH="${1:?usage: monitor_behavior_distill_retry.sh <pid>}"
STATUS_FILE="${STATUS_FILE:-$ROOT_DIR/artifacts/paper_suite/behavior_distill_retry_status.txt}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/logs/paper_suite/behavior_distill_mlp_train_retry.log}"
EVAL_LOG_FILE="${EVAL_LOG_FILE:-$ROOT_DIR/logs/paper_suite/behavior_distill_mlp_eval_retry.log}"
CLI_BIN="${CLI_BIN:-$ROOT_DIR/.uvenv/bin/kabr-vlm}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/qwen25vl3b_behavior_distill_mlp.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT_DIR/artifacts/paper/behavior/distill/qwen25vl3b/mlp/classifier.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/artifacts/paper/behavior/distill/qwen25vl3b/mlp/predictions.jsonl}"
METRICS_PATH="${METRICS_PATH:-$ROOT_DIR/artifacts/paper/behavior/distill/qwen25vl3b/mlp/predictions.metrics.json}"

mkdir -p "$(dirname "$STATUS_FILE")" "$(dirname "$LOG_FILE")"

while kill -0 "$PID_TO_WATCH" 2>/dev/null; do
  {
    echo "state: training"
    echo "pid: $PID_TO_WATCH"
    echo "updated_at: $(date --iso-8601=seconds)"
  } > "$STATUS_FILE"
  sleep 60
done

if [[ -f "$CHECKPOINT_PATH" ]]; then
  {
    echo "state: evaluating"
    echo "updated_at: $(date --iso-8601=seconds)"
  } > "$STATUS_FILE"
  if "$CLI_BIN" eval-distill \
    --config "$CONFIG_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --output "$OUTPUT_PATH" >"$EVAL_LOG_FILE" 2>&1; then
    {
      echo "state: completed"
      echo "updated_at: $(date --iso-8601=seconds)"
      echo "metrics_path: $METRICS_PATH"
    } > "$STATUS_FILE"
  else
    {
      echo "state: failed_during_eval"
      echo "updated_at: $(date --iso-8601=seconds)"
      echo "eval_log: $EVAL_LOG_FILE"
    } > "$STATUS_FILE"
  fi
else
  {
    echo "state: failed_during_train"
    echo "updated_at: $(date --iso-8601=seconds)"
    echo "train_log: $LOG_FILE"
  } > "$STATUS_FILE"
fi
