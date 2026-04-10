#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/jhyan/myproj"
SLOWFAST_ROOT="/home/jhyan/SlowFast"
DATA_ROOT="/home/jhyan/baboon_data/charades"
PYTHON_BIN="${ROOT_DIR}/.uvenv/bin/python"
CLI_BIN="${ROOT_DIR}/.uvenv/bin/kabr-vlm"

STATUS_DIR="${ROOT_DIR}/artifacts/baboonland_smoke"
LOG_DIR="${ROOT_DIR}/logs/baboonland_smoke"
CURRENT_TXT="${STATUS_DIR}/current.txt"
STATUS_TSV="${STATUS_DIR}/status.tsv"

PROCESSED_TRAIN="${ROOT_DIR}/data/processed/baboonland_behavior_train.jsonl"
PROCESSED_VAL="${ROOT_DIR}/data/processed/baboonland_behavior_val.jsonl"
MANIFEST_DIR="${ROOT_DIR}/artifacts/baboonland_smoke/clip_manifests_smoke"
VLM_SMOKE_OUT="${ROOT_DIR}/artifacts/baboonland_smoke/qwen25vl3b_behavior_clip_smoke.jsonl"
LORA_SMOKE_DIR="${ROOT_DIR}/artifacts/smoke/baboonland/qwen25vl3b_behavior_lora"
LORA_SMOKE_PRED="${LORA_SMOKE_DIR}/behavior_val_predictions.jsonl"
X3D_OUTPUT_DIR="${SLOWFAST_ROOT}/outputs/x3d_xs_baboonland_smoke"
X3D_METRICS="${ROOT_DIR}/artifacts/baboonland_smoke/x3d_xs_behavior_smoke.metrics.json"
X3D_CONFUSION="${ROOT_DIR}/artifacts/baboonland_smoke/x3d_xs_behavior_smoke.confusion.csv"

mkdir -p "${STATUS_DIR}" "${LOG_DIR}" "${ROOT_DIR}/artifacts/baboonland_smoke"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

mark_status() {
  local slug="$1"
  local status="$2"
  local detail="$3"
  printf "%s\t%s\t%s\t%s\n" "$(timestamp)" "$slug" "$status" "$detail" >> "${STATUS_TSV}"
  printf "%s\n" "${slug}" > "${CURRENT_TXT}"
}

cat > "${STATUS_TSV}" <<'EOF'
timestamp	slug	status	detail
EOF

mark_status "convert_train" "running" "${PROCESSED_TRAIN}"
"${CLI_BIN}" convert-kabr-split \
  --dataset baboonland \
  --annotation "${DATA_ROOT}/annotation/train.csv" \
  --image-root "${DATA_ROOT}/dataset/image" \
  --classes "${DATA_ROOT}/annotation/classes.json" \
  --task action_recognition \
  --max-frames-per-clip 8 \
  --output-format plain \
  --output "${PROCESSED_TRAIN}" \
  > "${LOG_DIR}/convert_train.log" 2>&1
mark_status "convert_train" "completed" "${PROCESSED_TRAIN}"

mark_status "convert_val" "running" "${PROCESSED_VAL}"
"${CLI_BIN}" convert-kabr-split \
  --dataset baboonland \
  --annotation "${DATA_ROOT}/annotation/val.csv" \
  --image-root "${DATA_ROOT}/dataset/image" \
  --classes "${DATA_ROOT}/annotation/classes.json" \
  --task action_recognition \
  --max-frames-per-clip 8 \
  --output-format plain \
  --output "${PROCESSED_VAL}" \
  > "${LOG_DIR}/convert_val.log" 2>&1
mark_status "convert_val" "completed" "${PROCESSED_VAL}"

mark_status "prepare_manifests" "running" "${MANIFEST_DIR}"
"${CLI_BIN}" prepare-charades-clip-manifests \
  --dataset baboonland \
  --annotation-dir "${DATA_ROOT}/annotation" \
  --image-root "${DATA_ROOT}/dataset/image" \
  --classes "${DATA_ROOT}/annotation/classes.json" \
  --output-dir "${MANIFEST_DIR}" \
  --max-clips 128 \
  --copy-val-to-test \
  > "${LOG_DIR}/prepare_manifests.log" 2>&1
mark_status "prepare_manifests" "completed" "${MANIFEST_DIR}"

mark_status "vlm_eval_smoke" "running" "${VLM_SMOKE_OUT}"
"${CLI_BIN}" eval-clip-hf-vlm \
  --dataset baboonland \
  --annotation "${DATA_ROOT}/annotation/val.csv" \
  --image-root "${DATA_ROOT}/dataset/image" \
  --classes "${DATA_ROOT}/annotation/classes.json" \
  --model-id "Qwen/Qwen2.5-VL-3B-Instruct" \
  --task behavior \
  --prompt-mode label \
  --max-samples 2 \
  --max-frames-per-clip 8 \
  --max-new-tokens 8 \
  --output "${VLM_SMOKE_OUT}" \
  > "${LOG_DIR}/vlm_eval_smoke.log" 2>&1
"${CLI_BIN}" score-predictions \
  --predictions "${VLM_SMOKE_OUT}" \
  --metrics-output "${VLM_SMOKE_OUT%.jsonl}.metrics.json" \
  --confusion-output "${VLM_SMOKE_OUT%.jsonl}.confusion.csv" \
  > "${LOG_DIR}/vlm_eval_smoke_score.log" 2>&1
mark_status "vlm_eval_smoke" "completed" "${VLM_SMOKE_OUT%.jsonl}.metrics.json"

mark_status "lora_train_smoke" "running" "${LORA_SMOKE_DIR}/checkpoint"
"${PYTHON_BIN}" -m kabr_vlm.training.finetune \
  --config "${ROOT_DIR}/configs/qwen25vl3b_baboonland_behavior_lora_smoke.yaml" \
  > "${LOG_DIR}/lora_train_smoke.log" 2>&1
mark_status "lora_train_smoke" "completed" "${LORA_SMOKE_DIR}/checkpoint"

mark_status "lora_eval_smoke" "running" "${LORA_SMOKE_PRED}"
"${CLI_BIN}" eval-finetuned-hf-vlm \
  --dataset baboonland \
  --annotation "${DATA_ROOT}/annotation/val.csv" \
  --image-root "${DATA_ROOT}/dataset/image" \
  --classes "${DATA_ROOT}/annotation/classes.json" \
  --model-id "Qwen/Qwen2.5-VL-3B-Instruct" \
  --adapter-path "${LORA_SMOKE_DIR}/checkpoint" \
  --task behavior \
  --prompt-mode label \
  --max-samples 2 \
  --max-new-tokens 8 \
  --max-frames-per-clip 8 \
  --output "${LORA_SMOKE_PRED}" \
  > "${LOG_DIR}/lora_eval_smoke.log" 2>&1
mark_status "lora_eval_smoke" "completed" "${LORA_SMOKE_PRED%.jsonl}.metrics.json"

mark_status "x3d_train_smoke" "running" "${X3D_OUTPUT_DIR}"
(
  cd "${SLOWFAST_ROOT}"
  "${PYTHON_BIN}" tools/run_net.py \
    --cfg configs/KABR/X3D_CLIP_LEVEL.yaml \
    --opts \
      TRAIN.ENABLE True \
      TEST.ENABLE False \
      DATA.PATH_TO_DATA_DIR "${MANIFEST_DIR}" \
      OUTPUT_DIR "${X3D_OUTPUT_DIR}" \
      MODEL.NUM_CLASSES 13 \
      SOLVER.MAX_EPOCH 1 \
      TRAIN.CHECKPOINT_PERIOD 1 \
      TRAIN.BATCH_SIZE 2 \
      DATA.NUM_FRAMES 4 \
      DATA.SAMPLING_RATE 8 \
      DATA.TRAIN_CROP_SIZE 160 \
      DATA.TEST_CROP_SIZE 182 \
      DATA.TRAIN_JITTER_SCALES "[160,160]" \
      DATA_LOADER.NUM_WORKERS 0 \
      LOG_MODEL_INFO False \
      X3D.DEPTH_FACTOR 2.2 \
      X3D.WIDTH_FACTOR 2.0 \
      X3D.DIM_C1 12 \
      X3D.BOTTLENECK_FACTOR 2.25
) > "${LOG_DIR}/x3d_train_smoke.log" 2>&1
mark_status "x3d_train_smoke" "completed" "${X3D_OUTPUT_DIR}"

X3D_CHECKPOINT="$(ls -1 "${X3D_OUTPUT_DIR}"/checkpoints/checkpoint_epoch_*.pyth | sort | tail -n1)"
mark_status "x3d_eval_smoke" "running" "${X3D_CHECKPOINT}"
(
  cd "${SLOWFAST_ROOT}"
  "${PYTHON_BIN}" tools/run_net.py \
    --cfg configs/KABR/X3D_CLIP_LEVEL.yaml \
    --opts \
      TRAIN.ENABLE False \
      TEST.ENABLE True \
      TEST.CHECKPOINT_FILE_PATH "${X3D_CHECKPOINT}" \
      DATA.PATH_TO_DATA_DIR "${MANIFEST_DIR}" \
      OUTPUT_DIR "${X3D_OUTPUT_DIR}" \
      MODEL.NUM_CLASSES 13 \
      TEST.BATCH_SIZE 2 \
      DATA.NUM_FRAMES 4 \
      DATA.SAMPLING_RATE 8 \
      DATA.TRAIN_CROP_SIZE 160 \
      DATA.TEST_CROP_SIZE 182 \
      DATA.TRAIN_JITTER_SCALES "[160,160]" \
      DATA_LOADER.NUM_WORKERS 0 \
      LOG_MODEL_INFO False \
      X3D.DEPTH_FACTOR 2.2 \
      X3D.WIDTH_FACTOR 2.0 \
      X3D.DIM_C1 12 \
      X3D.BOTTLENECK_FACTOR 2.25
) > "${LOG_DIR}/x3d_eval_smoke.log" 2>&1
"${CLI_BIN}" score-predictions \
  --predictions "${X3D_OUTPUT_DIR}/clip_level/predictions.jsonl" \
  --metrics-output "${X3D_METRICS}" \
  --confusion-output "${X3D_CONFUSION}" \
  --task behavior \
  --classes-path "${DATA_ROOT}/annotation/classes.json" \
  > "${LOG_DIR}/x3d_score_smoke.log" 2>&1
mark_status "x3d_eval_smoke" "completed" "${X3D_METRICS}"

mark_status "suite" "completed" "${STATUS_TSV}"
