#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.uvenv/bin/python"
CLI_BIN="${ROOT_DIR}/.uvenv/bin/kabr-vlm"

DATA_ROOT="${ROOT_DIR}/../KABR_files"
ANNOTATION_DIR="${DATA_ROOT}/annotation"
IMAGE_ROOT="${DATA_ROOT}/dataset/image"
CLASSES_PATH="${ANNOTATION_DIR}/classes.json"

STATUS_DIR="${ROOT_DIR}/artifacts/qwen25vl3b_behavior_lora_16frames"
LOG_DIR="${ROOT_DIR}/logs/qwen25vl3b_behavior_lora_16frames"
CURRENT_FILE="${STATUS_DIR}/current.txt"
STATUS_FILE="${STATUS_DIR}/status.tsv"

SMOKE_OUTPUT_DIR="${ROOT_DIR}/artifacts/smoke/qwen25vl3b_behavior_lora_16frames"
FULL_OUTPUT_DIR="${ROOT_DIR}/artifacts/runs/qwen25vl3b_behavior_lora_full_16frames"
FULL_PREDICTIONS="${FULL_OUTPUT_DIR}/behavior_val_predictions.jsonl"

mkdir -p "${STATUS_DIR}" "${LOG_DIR}" "${SMOKE_OUTPUT_DIR}" "${FULL_OUTPUT_DIR}"

timestamp() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

set_current() {
  printf "%s\n" "$1" > "${CURRENT_FILE}"
}

append_status() {
  printf "%s\t%s\t%s\t%s\n" "$(timestamp)" "$1" "$2" "$3" >> "${STATUS_FILE}"
}

run_logged() {
  local slug="$1"
  local output_check="$2"
  local command="$3"
  local log_file="${LOG_DIR}/${slug}.log"

  if [[ -n "${output_check}" && -e "${output_check}" ]]; then
    append_status "${slug}" "skipped" "${output_check}"
    return 0
  fi

  set_current "${slug}"
  append_status "${slug}" "running" "${output_check:-$log_file}"
  if bash -lc "${command}" > "${log_file}" 2>&1; then
    append_status "${slug}" "completed" "${output_check:-$log_file}"
    return 0
  fi

  append_status "${slug}" "failed" "${log_file}"
  return 1
}

printf "timestamp\tslug\tstatus\tdetail\n" > "${STATUS_FILE}"

run_logged \
  "convert_train_16frames" \
  "${ROOT_DIR}/data/processed/kabr_behavior_train_16frames.jsonl" \
  "\"${CLI_BIN}\" convert-kabr-split --annotation \"${ANNOTATION_DIR}/train.csv\" --image-root \"${IMAGE_ROOT}\" --classes \"${CLASSES_PATH}\" --task action_recognition --max-frames-per-clip 16 --output-format plain --output \"${ROOT_DIR}/data/processed/kabr_behavior_train_16frames.jsonl\""

run_logged \
  "convert_val_16frames" \
  "${ROOT_DIR}/data/processed/kabr_behavior_val_16frames.jsonl" \
  "\"${CLI_BIN}\" convert-kabr-split --annotation \"${ANNOTATION_DIR}/val.csv\" --image-root \"${IMAGE_ROOT}\" --classes \"${CLASSES_PATH}\" --task action_recognition --max-frames-per-clip 16 --output-format plain --output \"${ROOT_DIR}/data/processed/kabr_behavior_val_16frames.jsonl\""

run_logged \
  "smoke_train_16frames" \
  "${SMOKE_OUTPUT_DIR}/checkpoint/training_metrics.json" \
  "\"${PYTHON_BIN}\" -m kabr_vlm.training.finetune --config \"${ROOT_DIR}/configs/qwen25vl3b_behavior_lora_16frames_smoke.yaml\""

run_logged \
  "smoke_eval_16frames" \
  "${SMOKE_OUTPUT_DIR}/behavior_val_predictions.metrics.json" \
  "\"${CLI_BIN}\" eval-finetuned-hf-vlm --annotation \"${ANNOTATION_DIR}/val.csv\" --image-root \"${IMAGE_ROOT}\" --classes \"${CLASSES_PATH}\" --model-id Qwen/Qwen2.5-VL-3B-Instruct --adapter-path \"${SMOKE_OUTPUT_DIR}/checkpoint\" --task behavior --prompt-mode label --max-samples 1 --max-frames-per-clip 16 --output \"${SMOKE_OUTPUT_DIR}/behavior_val_predictions.jsonl\""

run_logged \
  "full_train_16frames" \
  "${FULL_OUTPUT_DIR}/checkpoint/training_metrics.json" \
  "\"${PYTHON_BIN}\" -m kabr_vlm.training.finetune --config \"${ROOT_DIR}/configs/qwen25vl3b_behavior_lora_full_16frames.yaml\""

run_logged \
  "full_eval_16frames" \
  "${FULL_OUTPUT_DIR}/behavior_val_predictions.metrics.json" \
  "\"${CLI_BIN}\" eval-finetuned-hf-vlm --annotation \"${ANNOTATION_DIR}/val.csv\" --image-root \"${IMAGE_ROOT}\" --classes \"${CLASSES_PATH}\" --model-id Qwen/Qwen2.5-VL-3B-Instruct --adapter-path \"${FULL_OUTPUT_DIR}/checkpoint\" --task behavior --prompt-mode label --max-frames-per-clip 16 --output \"${FULL_PREDICTIONS}\""

set_current "completed"
append_status "suite" "completed" "${STATUS_FILE}"
