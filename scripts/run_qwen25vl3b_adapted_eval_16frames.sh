#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.uvenv/bin/python"
CLI_BIN="${ROOT_DIR}/.uvenv/bin/kabr-vlm"

ANNOTATION_PATH="${ROOT_DIR}/../KABR_files/annotation/val.csv"
IMAGE_ROOT="${ROOT_DIR}/../KABR_files/dataset/image"
CLASSES_PATH="${ROOT_DIR}/../KABR_files/annotation/classes.json"

STATUS_DIR="${ROOT_DIR}/artifacts/qwen25vl3b_eval_16frames"
LOG_DIR="${ROOT_DIR}/logs/qwen25vl3b_eval_16frames"
CURRENT_FILE="${STATUS_DIR}/current.txt"
STATUS_FILE="${STATUS_DIR}/status.tsv"

mkdir -p "${STATUS_DIR}" "${LOG_DIR}"

timestamp() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

set_current() {
  printf "%s\n" "$1" > "${CURRENT_FILE}"
}

append_status() {
  printf "%s\t%s\t%s\t%s\n" "$(timestamp)" "$1" "$2" "$3" >> "${STATUS_FILE}"
}

run_step() {
  local slug="$1"
  local output_check="$2"
  local command="$3"
  local log_file="${LOG_DIR}/${slug}.log"

  if [[ -f "${output_check}" ]]; then
    append_status "${slug}" "skipped" "${output_check}"
    return 0
  fi

  set_current "${slug}"
  append_status "${slug}" "running" "${output_check}"
  if bash -lc "${command}" > "${log_file}" 2>&1; then
    append_status "${slug}" "completed" "${output_check}"
  else
    append_status "${slug}" "failed" "${log_file}"
    return 1
  fi
}

printf "timestamp\tslug\tstatus\tdetail\n" > "${STATUS_FILE}"

run_step \
  "qwen25vl3b_lora_plain_8f_ckpt_eval16" \
  "${ROOT_DIR}/artifacts/runs/qwen25vl3b_behavior_lora_full/behavior_val_predictions_16frames.metrics.json" \
  "\"${CLI_BIN}\" eval-finetuned-hf-vlm --annotation \"${ANNOTATION_PATH}\" --image-root \"${IMAGE_ROOT}\" --classes \"${CLASSES_PATH}\" --model-id Qwen/Qwen2.5-VL-3B-Instruct --adapter-path \"${ROOT_DIR}/artifacts/runs/qwen25vl3b_behavior_lora_full/checkpoint\" --task behavior --prompt-mode label --max-frames-per-clip 16 --output \"${ROOT_DIR}/artifacts/runs/qwen25vl3b_behavior_lora_full/behavior_val_predictions_16frames.jsonl\""

run_step \
  "qwen25vl3b_lora_plain_4f_ckpt_eval16" \
  "${ROOT_DIR}/artifacts/paper/behavior/lora/qwen25vl3b/plain_4frames/behavior_val_predictions_16frames.metrics.json" \
  "\"${CLI_BIN}\" eval-finetuned-hf-vlm --annotation \"${ANNOTATION_PATH}\" --image-root \"${IMAGE_ROOT}\" --classes \"${CLASSES_PATH}\" --model-id Qwen/Qwen2.5-VL-3B-Instruct --adapter-path \"${ROOT_DIR}/artifacts/paper/behavior/lora/qwen25vl3b/plain_4frames/checkpoint\" --task behavior --prompt-mode label --max-frames-per-clip 16 --output \"${ROOT_DIR}/artifacts/paper/behavior/lora/qwen25vl3b/plain_4frames/behavior_val_predictions_16frames.jsonl\""

run_step \
  "qwen25vl3b_frozen_linear_eval16" \
  "${ROOT_DIR}/artifacts/paper/behavior/frozen/qwen25vl3b/linear_16frames_eval/predictions.metrics.json" \
  "\"${CLI_BIN}\" eval-frozen-features --config \"${ROOT_DIR}/configs/qwen25vl3b_behavior_frozen_linear_16frames_eval.yaml\" --checkpoint-path \"${ROOT_DIR}/artifacts/paper/behavior/frozen/qwen25vl3b/linear/classifier.pt\" --output \"${ROOT_DIR}/artifacts/paper/behavior/frozen/qwen25vl3b/linear_16frames_eval/predictions.jsonl\""

run_step \
  "qwen25vl3b_frozen_mlp_eval16" \
  "${ROOT_DIR}/artifacts/paper/behavior/frozen/qwen25vl3b/mlp_16frames_eval/predictions.metrics.json" \
  "\"${CLI_BIN}\" eval-frozen-features --config \"${ROOT_DIR}/configs/qwen25vl3b_behavior_frozen_mlp_16frames_eval.yaml\" --checkpoint-path \"${ROOT_DIR}/artifacts/paper/behavior/frozen/qwen25vl3b/mlp/classifier.pt\" --output \"${ROOT_DIR}/artifacts/paper/behavior/frozen/qwen25vl3b/mlp_16frames_eval/predictions.jsonl\""

run_step \
  "qwen25vl3b_distill_mlp_eval16" \
  "${ROOT_DIR}/artifacts/paper/behavior/distill/qwen25vl3b/mlp_16frames_eval/predictions.metrics.json" \
  "\"${CLI_BIN}\" eval-distill --config \"${ROOT_DIR}/configs/qwen25vl3b_behavior_distill_mlp_16frames_eval.yaml\" --checkpoint-path \"${ROOT_DIR}/artifacts/paper/behavior/distill/qwen25vl3b/mlp/classifier.pt\" --output \"${ROOT_DIR}/artifacts/paper/behavior/distill/qwen25vl3b/mlp_16frames_eval/predictions.jsonl\""

run_step \
  "qwen25vl3b_distill_mlp_alpha70_eval16" \
  "${ROOT_DIR}/artifacts/paper/behavior/distill/qwen25vl3b/mlp_alpha70_16frames_eval/predictions.metrics.json" \
  "\"${CLI_BIN}\" eval-distill --config \"${ROOT_DIR}/configs/qwen25vl3b_behavior_distill_mlp_alpha70_16frames_eval.yaml\" --checkpoint-path \"${ROOT_DIR}/artifacts/paper/behavior/distill/qwen25vl3b/mlp_alpha70/classifier.pt\" --output \"${ROOT_DIR}/artifacts/paper/behavior/distill/qwen25vl3b/mlp_alpha70_16frames_eval/predictions.jsonl\""

run_step \
  "qwen25vl3b_lora_structured_ckpt_eval16" \
  "${ROOT_DIR}/artifacts/paper/behavior/lora/qwen25vl3b/structured/behavior_val_predictions_16frames.metrics.json" \
  "\"${CLI_BIN}\" eval-finetuned-hf-vlm --annotation \"${ANNOTATION_PATH}\" --image-root \"${IMAGE_ROOT}\" --classes \"${CLASSES_PATH}\" --model-id Qwen/Qwen2.5-VL-3B-Instruct --adapter-path \"${ROOT_DIR}/artifacts/paper/behavior/lora/qwen25vl3b/structured/checkpoint\" --task behavior --prompt-mode structured --max-frames-per-clip 16 --output \"${ROOT_DIR}/artifacts/paper/behavior/lora/qwen25vl3b/structured/behavior_val_predictions_16frames.jsonl\""

set_current "completed"
append_status "suite" "completed" "${STATUS_FILE}"
