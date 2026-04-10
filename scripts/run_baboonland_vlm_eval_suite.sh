#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.uvenv/bin/python}"
CLI_BIN="${CLI_BIN:-$ROOT_DIR/.uvenv/bin/kabr-vlm}"
DATA_ROOT="${DATA_ROOT:-/home/jhyan/baboon_data/charades}"
MODEL_LIST="${MODEL_LIST:-$ROOT_DIR/configs/baboonland_vlm_eval_models.tsv}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/baboonland_vlm_eval}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/artifacts/baboonland_eval}"
STATUS_FILE="${STATUS_FILE:-$OUT_DIR/status.tsv}"
CURRENT_FILE="${CURRENT_FILE:-$OUT_DIR/current.txt}"

ANNOTATION_PATH="${DATA_ROOT}/annotation/val.csv"
IMAGE_ROOT="${DATA_ROOT}/dataset/image"
CLASSES_PATH="${DATA_ROOT}/annotation/classes.json"

mkdir -p "$LOG_DIR" "$OUT_DIR"
printf "timestamp\tslug\tstatus\tdetail\n" > "$STATUS_FILE"

timestamp() { date +"%Y-%m-%dT%H:%M:%S%z"; }
mark() {
  printf "%s\t%s\t%s\t%s\n" "$(timestamp)" "$1" "$2" "$3" >> "$STATUS_FILE"
  printf "%s\n" "$1" > "$CURRENT_FILE"
}

run_zero_shot() {
  local slug="$1"
  local model_id="$2"
  local pred="$OUT_DIR/${slug}.jsonl"
  local metrics="$OUT_DIR/${slug}.metrics.json"
  local confusion="$OUT_DIR/${slug}.confusion.csv"
  local log="$LOG_DIR/${slug}.log"

  mark "$slug" "running" "$pred"
  if {
    "$CLI_BIN" eval-clip-hf-vlm \
      --dataset baboonland \
      --annotation "$ANNOTATION_PATH" \
      --image-root "$IMAGE_ROOT" \
      --classes "$CLASSES_PATH" \
      --model-id "$model_id" \
      --task behavior \
      --prompt-mode label \
      --max-samples 6198 \
      --max-frames-per-clip 8 \
      --max-new-tokens 16 \
      --output "$pred"
    "$CLI_BIN" score-predictions \
      --predictions "$pred" \
      --metrics-output "$metrics" \
      --confusion-output "$confusion" \
      --classes-path "$CLASSES_PATH"
  } > "$log" 2>&1; then
    mark "$slug" "completed" "$metrics"
  else
    mark "$slug" "failed" "$log"
  fi
}

run_adapter_eval() {
  local slug="$1"
  local adapter_path="$2"
  local prompt_mode="$3"
  local pred="$OUT_DIR/${slug}.jsonl"
  local log="$LOG_DIR/${slug}.log"

  mark "$slug" "running" "$pred"
  if "$CLI_BIN" eval-finetuned-hf-vlm \
      --dataset baboonland \
      --annotation "$ANNOTATION_PATH" \
      --image-root "$IMAGE_ROOT" \
      --classes "$CLASSES_PATH" \
      --model-id "Qwen/Qwen2.5-VL-3B-Instruct" \
      --adapter-path "$adapter_path" \
      --task behavior \
      --prompt-mode "$prompt_mode" \
      --max-frames-per-clip 8 \
      --max-new-tokens 8 \
      --output "$pred" > "$log" 2>&1; then
    mark "$slug" "completed" "${pred%.jsonl}.metrics.json"
  else
    mark "$slug" "failed" "$log"
  fi
}

run_feature_eval() {
  local slug="$1"
  local cmd="$2"
  local config="$3"
  local checkpoint="$4"
  local pred="$5"
  local log="$LOG_DIR/${slug}.log"

  mark "$slug" "running" "$pred"
  if "$CLI_BIN" "$cmd" \
      --config "$config" \
      --checkpoint-path "$checkpoint" \
      --output "$pred" > "$log" 2>&1; then
    mark "$slug" "completed" "${pred%.jsonl}.metrics.json"
  else
    mark "$slug" "failed" "$log"
  fi
}

tail -n +2 "$MODEL_LIST" | while IFS=$'\t' read -r slug model_id; do
  [[ -z "$slug" ]] && continue
  run_zero_shot "$slug" "$model_id"
done

run_adapter_eval \
  "qwen25vl3b_baboonland_behavior_lora_plain" \
  "$ROOT_DIR/artifacts/runs/qwen25vl3b_behavior_lora_full/checkpoint" \
  "label"

run_adapter_eval \
  "qwen25vl3b_baboonland_behavior_lora_plain_4frames_ckpt" \
  "$ROOT_DIR/artifacts/paper/behavior/lora/qwen25vl3b/plain_4frames/checkpoint" \
  "label"

run_adapter_eval \
  "qwen25vl3b_baboonland_behavior_lora_structured_ckpt" \
  "$ROOT_DIR/artifacts/paper/behavior/lora/qwen25vl3b/structured/checkpoint" \
  "structured"

run_feature_eval \
  "qwen25vl3b_baboonland_behavior_frozen_linear" \
  "eval-frozen-features" \
  "$ROOT_DIR/configs/qwen25vl3b_baboonland_behavior_frozen_linear_eval.yaml" \
  "$ROOT_DIR/artifacts/paper/behavior/frozen/qwen25vl3b/linear/classifier.pt" \
  "$ROOT_DIR/artifacts/baboonland_eval/qwen25vl3b_baboonland_behavior_frozen_linear.jsonl"

run_feature_eval \
  "qwen25vl3b_baboonland_behavior_frozen_mlp" \
  "eval-frozen-features" \
  "$ROOT_DIR/configs/qwen25vl3b_baboonland_behavior_frozen_mlp_eval.yaml" \
  "$ROOT_DIR/artifacts/paper/behavior/frozen/qwen25vl3b/mlp/classifier.pt" \
  "$ROOT_DIR/artifacts/baboonland_eval/qwen25vl3b_baboonland_behavior_frozen_mlp.jsonl"

run_feature_eval \
  "qwen25vl3b_baboonland_behavior_distill_mlp" \
  "eval-distill" \
  "$ROOT_DIR/configs/qwen25vl3b_baboonland_behavior_distill_mlp_eval.yaml" \
  "$ROOT_DIR/artifacts/paper/behavior/distill/qwen25vl3b/mlp/classifier.pt" \
  "$ROOT_DIR/artifacts/baboonland_eval/qwen25vl3b_baboonland_behavior_distill_mlp.jsonl"

run_feature_eval \
  "qwen25vl3b_baboonland_behavior_distill_mlp_alpha70" \
  "eval-distill" \
  "$ROOT_DIR/configs/qwen25vl3b_baboonland_behavior_distill_mlp_alpha70_eval.yaml" \
  "$ROOT_DIR/artifacts/paper/behavior/distill/qwen25vl3b/mlp_alpha70/classifier.pt" \
  "$ROOT_DIR/artifacts/baboonland_eval/qwen25vl3b_baboonland_behavior_distill_mlp_alpha70.jsonl"

mark "suite" "completed" "$STATUS_FILE"
