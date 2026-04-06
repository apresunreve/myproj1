#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/jhyan/myproj"
SLOWFAST_ROOT="/home/jhyan/SlowFast"
PYTHON_BIN="${ROOT_DIR}/.uvenv/bin/python"
CLI_BIN="${ROOT_DIR}/.uvenv/bin/kabr-vlm"

STATUS_DIR="${ROOT_DIR}/artifacts/baseline_expansion"
LOG_DIR="${ROOT_DIR}/logs/baseline_expansion"
CURRENT_TXT="${STATUS_DIR}/current.txt"
STATUS_TSV="${STATUS_DIR}/status.tsv"

X3D_OUTPUT_DIR="${SLOWFAST_ROOT}/outputs/x3d_l_kabr_clips"
X3D_METRICS="${ROOT_DIR}/artifacts/x3d_metrics/x3d_l_behavior_clip.metrics.json"
X3D_CONFUSION="${ROOT_DIR}/artifacts/x3d_metrics/x3d_l_behavior_clip.confusion.csv"

GEMMA_OUTPUT_DIR="${ROOT_DIR}/artifacts/runs/gemma3_4b_behavior_lora_full"
GEMMA_CHECKPOINT_DIR="${GEMMA_OUTPUT_DIR}/checkpoint"
GEMMA_PREDICTIONS="${GEMMA_OUTPUT_DIR}/behavior_val_predictions.jsonl"

mkdir -p "${STATUS_DIR}" "${LOG_DIR}" "${ROOT_DIR}/artifacts/x3d_metrics" "${GEMMA_OUTPUT_DIR}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

set_status() {
  printf '%s\n' "$1" > "${CURRENT_TXT}"
}

append_status() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "${STATUS_TSV}"
}

latest_checkpoint() {
  ls -1 "${X3D_OUTPUT_DIR}"/checkpoints/checkpoint_epoch_*.pyth 2>/dev/null | sort | tail -n1
}

cat > "${STATUS_TSV}" <<'EOF'
run	status	phase	output_path	notes
EOF

set_status "x3d_l:train"
append_status "x3d_l" "running" "train" "${X3D_OUTPUT_DIR}" ""
(
  cd "${SLOWFAST_ROOT}"
  "${PYTHON_BIN}" tools/run_net.py \
    --cfg configs/KABR/X3D_CLIP_LEVEL.yaml \
    --opts \
      TRAIN.ENABLE True \
      TEST.ENABLE False \
      DATA.PATH_TO_DATA_DIR /home/jhyan/KABR_files/annotation/clip_manifests \
      OUTPUT_DIR "${X3D_OUTPUT_DIR}" \
      DATA.NUM_FRAMES 16 \
      DATA.SAMPLING_RATE 5 \
      DATA.TRAIN_CROP_SIZE 300 \
      DATA.TEST_CROP_SIZE 300 \
      DATA.TRAIN_JITTER_SCALES "[300,300]" \
      TRAIN.BATCH_SIZE 4 \
      DATA_LOADER.NUM_WORKERS 0 \
      LOG_MODEL_INFO False \
      X3D.DEPTH_FACTOR 5.0 \
      X3D.WIDTH_FACTOR 2.0 \
      X3D.DIM_C1 12 \
      X3D.BOTTLENECK_FACTOR 2.25
) > "${LOG_DIR}/x3d_l_train.log" 2>&1

set_status "x3d_l:eval"
append_status "x3d_l" "running" "eval" "${X3D_OUTPUT_DIR}" ""
X3D_CHECKPOINT="$(latest_checkpoint)"
if [[ -z "${X3D_CHECKPOINT}" ]]; then
  echo "[$(timestamp)] no X3D-L checkpoint found" > "${LOG_DIR}/x3d_l_eval.log"
  append_status "x3d_l" "failed" "eval" "${X3D_OUTPUT_DIR}" "missing_checkpoint"
  set_status "suite:failed"
  exit 1
fi
(
  cd "${SLOWFAST_ROOT}"
  "${PYTHON_BIN}" tools/run_net.py \
    --cfg configs/KABR/X3D_CLIP_LEVEL.yaml \
    --opts \
      TRAIN.ENABLE False \
      TEST.ENABLE True \
      TEST.CHECKPOINT_FILE_PATH "${X3D_CHECKPOINT}" \
      DATA.PATH_TO_DATA_DIR /home/jhyan/KABR_files/annotation/clip_manifests \
      OUTPUT_DIR "${X3D_OUTPUT_DIR}" \
      DATA.NUM_FRAMES 16 \
      DATA.SAMPLING_RATE 5 \
      DATA.TRAIN_CROP_SIZE 300 \
      DATA.TEST_CROP_SIZE 300 \
      DATA.TRAIN_JITTER_SCALES "[300,300]" \
      TEST.BATCH_SIZE 4 \
      DATA_LOADER.NUM_WORKERS 0 \
      LOG_MODEL_INFO False \
      X3D.DEPTH_FACTOR 5.0 \
      X3D.WIDTH_FACTOR 2.0 \
      X3D.DIM_C1 12 \
      X3D.BOTTLENECK_FACTOR 2.25
) > "${LOG_DIR}/x3d_l_eval.log" 2>&1

set_status "x3d_l:score"
append_status "x3d_l" "running" "score" "${X3D_METRICS}" ""
"${CLI_BIN}" score-predictions \
  --predictions "${X3D_OUTPUT_DIR}/clip_level/predictions.jsonl" \
  --metrics-output "${X3D_METRICS}" \
  --confusion-output "${X3D_CONFUSION}" \
  --task behavior \
  > "${LOG_DIR}/x3d_l_score.log" 2>&1
append_status "x3d_l" "completed" "done" "${X3D_METRICS}" ""

if [[ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]]; then
  append_status "gemma3_4b_lora" "failed" "startup" "${GEMMA_OUTPUT_DIR}" "missing_hf_token"
  set_status "suite:failed"
  exit 1
fi

set_status "gemma3_4b_lora:train"
append_status "gemma3_4b_lora" "running" "train" "${GEMMA_CHECKPOINT_DIR}" ""
"${PYTHON_BIN}" -m kabr_vlm.training.finetune \
  --config "${ROOT_DIR}/configs/gemma3_4b_behavior_lora_full.yaml" \
  > "${LOG_DIR}/gemma3_4b_lora_train.log" 2>&1

set_status "gemma3_4b_lora:eval"
append_status "gemma3_4b_lora" "running" "eval" "${GEMMA_PREDICTIONS}" ""
"${CLI_BIN}" eval-finetuned-hf-vlm \
  --annotation /home/jhyan/KABR_files/annotation/val.csv \
  --image-root /home/jhyan/KABR_files/dataset/image \
  --classes /home/jhyan/KABR_files/annotation/classes.json \
  --model-id google/gemma-3-4b-it \
  --adapter-path "${GEMMA_CHECKPOINT_DIR}" \
  --task behavior \
  --prompt-mode label \
  --output "${GEMMA_PREDICTIONS}" \
  > "${LOG_DIR}/gemma3_4b_lora_eval.log" 2>&1
append_status "gemma3_4b_lora" "completed" "done" "${GEMMA_PREDICTIONS}" ""

set_status "suite:completed"
