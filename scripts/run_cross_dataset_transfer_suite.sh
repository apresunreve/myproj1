#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI_BIN="${CLI_BIN:-$ROOT_DIR/.uvenv/bin/kabr-vlm}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.uvenv/bin/python}"
KABR_ROOT="${KABR_ROOT:-/home/jhyan/KABR_files}"
BABOON_ROOT="${BABOON_ROOT:-/home/jhyan/baboon_data/charades}"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/cross_dataset_transfer}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/artifacts/cross_dataset_transfer}"
STATUS_FILE="${STATUS_FILE:-$OUT_DIR/status.tsv}"
CURRENT_FILE="${CURRENT_FILE:-$OUT_DIR/current.txt}"

SMOKE_BABOON_PRED="${ROOT_DIR}/artifacts/transfer_smoke/qwen25vl3b_baboonland_lora_smoke_on_baboonland.jsonl"
SMOKE_KABR_PRED="${ROOT_DIR}/artifacts/transfer_smoke/qwen25vl3b_baboonland_lora_smoke_on_kabr.jsonl"
FULL_BABOON_PRED="${ROOT_DIR}/artifacts/transfer_eval/qwen25vl3b_baboonland_behavior_lora_on_baboonland.jsonl"
FULL_KABR_PRED="${ROOT_DIR}/artifacts/transfer_eval/qwen25vl3b_baboonland_behavior_lora_on_kabr.jsonl"

mkdir -p "$LOG_DIR" "$OUT_DIR"
printf "timestamp\tstage\tstatus\tdetail\n" > "$STATUS_FILE"

timestamp() { date +"%Y-%m-%dT%H:%M:%S%z"; }
mark() {
  printf "%s\t%s\t%s\t%s\n" "$(timestamp)" "$1" "$2" "$3" >> "$STATUS_FILE"
  printf "%s\n" "$1" > "$CURRENT_FILE"
}

wait_for_running_evals() {
  local pattern='run_baboonland_vlm_eval_suite|eval-clip-hf-vlm|eval-finetuned-hf-vlm|eval-frozen-features|eval-distill'
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    mark "waiting_for_eval_queue" "running" "$pattern"
    sleep 60
  done
  mark "waiting_for_eval_queue" "completed" "clear"
}

run_step() {
  local stage="$1"
  local command="$2"
  local log="$3"
  mark "$stage" "running" "$log"
  if bash -lc "$command" > "$log" 2>&1; then
    mark "$stage" "completed" "$log"
  else
    mark "$stage" "failed" "$log"
    return 1
  fi
}

wait_for_running_evals

run_step \
  "baboonland_lora_smoke_train" \
  "\"$PYTHON_BIN\" -m kabr_vlm.training.finetune --config \"$ROOT_DIR/configs/qwen25vl3b_baboonland_behavior_lora_smoke.yaml\"" \
  "$LOG_DIR/baboonland_lora_smoke_train.log"

run_step \
  "baboonland_lora_smoke_eval_in_domain" \
  "\"$CLI_BIN\" eval-finetuned-hf-vlm --dataset baboonland --annotation \"$BABOON_ROOT/annotation/val.csv\" --image-root \"$BABOON_ROOT/dataset/image\" --classes \"$BABOON_ROOT/annotation/classes.json\" --model-id Qwen/Qwen2.5-VL-3B-Instruct --adapter-path \"$ROOT_DIR/artifacts/smoke/baboonland/qwen25vl3b_behavior_lora/checkpoint\" --task behavior --prompt-mode label --max-samples 32 --output \"$SMOKE_BABOON_PRED\"" \
  "$LOG_DIR/baboonland_lora_smoke_eval_in_domain.log"

run_step \
  "baboonland_lora_smoke_eval_transfer" \
  "\"$CLI_BIN\" eval-finetuned-hf-vlm --dataset kabr --annotation \"$KABR_ROOT/annotation/val.csv\" --image-root \"$KABR_ROOT/dataset/image\" --classes \"$KABR_ROOT/annotation/classes.json\" --model-id Qwen/Qwen2.5-VL-3B-Instruct --adapter-path \"$ROOT_DIR/artifacts/smoke/baboonland/qwen25vl3b_behavior_lora/checkpoint\" --task behavior --prompt-mode label --max-samples 32 --output \"$SMOKE_KABR_PRED\"" \
  "$LOG_DIR/baboonland_lora_smoke_eval_transfer.log"

run_step \
  "baboonland_lora_full_train" \
  "\"$PYTHON_BIN\" -m kabr_vlm.training.finetune --config \"$ROOT_DIR/configs/qwen25vl3b_baboonland_behavior_lora_full.yaml\"" \
  "$LOG_DIR/baboonland_lora_full_train.log"

run_step \
  "baboonland_lora_full_eval_in_domain" \
  "\"$CLI_BIN\" eval-finetuned-hf-vlm --dataset baboonland --annotation \"$BABOON_ROOT/annotation/val.csv\" --image-root \"$BABOON_ROOT/dataset/image\" --classes \"$BABOON_ROOT/annotation/classes.json\" --model-id Qwen/Qwen2.5-VL-3B-Instruct --adapter-path \"$ROOT_DIR/artifacts/runs/baboonland/qwen25vl3b_behavior_lora/checkpoint\" --task behavior --prompt-mode label --max-samples 6198 --output \"$FULL_BABOON_PRED\"" \
  "$LOG_DIR/baboonland_lora_full_eval_in_domain.log"

run_step \
  "baboonland_lora_full_eval_transfer" \
  "\"$CLI_BIN\" eval-finetuned-hf-vlm --dataset kabr --annotation \"$KABR_ROOT/annotation/val.csv\" --image-root \"$KABR_ROOT/dataset/image\" --classes \"$KABR_ROOT/annotation/classes.json\" --model-id Qwen/Qwen2.5-VL-3B-Instruct --adapter-path \"$ROOT_DIR/artifacts/runs/baboonland/qwen25vl3b_behavior_lora/checkpoint\" --task behavior --prompt-mode label --max-samples 3761 --output \"$FULL_KABR_PRED\"" \
  "$LOG_DIR/baboonland_lora_full_eval_transfer.log"

run_step \
  "aggregate_transfer_tables" \
  "\"$CLI_BIN\" aggregate-paper-results --manifest \"$ROOT_DIR/configs/paper_runs_transfer.yaml\" --output-dir \"$ROOT_DIR/artifacts/paper_tables_transfer\"" \
  "$LOG_DIR/aggregate_transfer_tables.log"

mark "suite" "completed" "$STATUS_FILE"
