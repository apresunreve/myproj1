#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.uvenv/bin/python}"
CLI_BIN="${CLI_BIN:-$ROOT_DIR/.uvenv/bin/kabr-vlm}"
STATUS_DIR="${STATUS_DIR:-$ROOT_DIR/artifacts/behavior_expansion_suite}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/behavior_expansion_suite}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/../KABR_files}"

mkdir -p "$STATUS_DIR" "$LOG_DIR"

CURRENT_FILE="$STATUS_DIR/current.txt"
STATUS_TSV="$STATUS_DIR/status.tsv"

if [[ ! -f "$STATUS_TSV" ]]; then
  printf "run_name\tstage\tstatus\tlog_path\toutput_dir\n" > "$STATUS_TSV"
fi

update_status() {
  local run_name="$1"
  local stage="$2"
  local status="$3"
  local log_path="$4"
  local output_dir="$5"
  python - "$STATUS_TSV" "$run_name" "$stage" "$status" "$log_path" "$output_dir" <<'PY'
from pathlib import Path
import sys

status_path = Path(sys.argv[1])
run_name, stage, status, log_path, output_dir = sys.argv[2:]
lines = status_path.read_text(encoding="utf-8").splitlines()
header = lines[0] if lines else "run_name\tstage\tstatus\tlog_path\toutput_dir"
rows = [line for line in lines[1:] if not line.startswith(f"{run_name}\t")]
rows.append("\t".join([run_name, stage, status, log_path, output_dir]))
status_path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
PY
}

run_step() {
  local run_name="$1"
  local stage="$2"
  local command="$3"
  local log_path="$4"
  local output_dir="$5"
  local expected_path="${6:-}"

  if [[ -n "$expected_path" && -e "$expected_path" ]]; then
    printf "%s: %s (skipped, found %s)\n" "$run_name" "$stage" "$expected_path" > "$CURRENT_FILE"
    update_status "$run_name" "$stage" "completed" "$log_path" "$output_dir"
    return
  fi

  printf "%s: %s\n" "$run_name" "$stage" > "$CURRENT_FILE"
  update_status "$run_name" "$stage" "running" "$log_path" "$output_dir"
  if bash -lc "$command" >"$log_path" 2>&1; then
    update_status "$run_name" "$stage" "completed" "$log_path" "$output_dir"
  else
    update_status "$run_name" "$stage" "failed" "$log_path" "$output_dir"
    printf "%s: failed at %s\n" "$run_name" "$stage" > "$CURRENT_FILE"
    exit 1
  fi
}

run_experiment() {
  local run_name="$1"
  local output_dir="$2"
  local train_cmd="$3"
  local eval_cmd="$4"
  local train_expected="$5"
  local eval_expected="$6"

  run_step "$run_name" "train" "$train_cmd" "$LOG_DIR/${run_name}_train.log" "$output_dir" "$train_expected"
  run_step "$run_name" "eval" "$eval_cmd" "$LOG_DIR/${run_name}_eval.log" "$output_dir" "$eval_expected"
}

run_step \
  "behavior_structured_data" \
  "convert" \
  "\"$CLI_BIN\" convert-kabr-split --annotation \"$DATA_ROOT/annotation/train.csv\" --image-root \"$DATA_ROOT/dataset/image\" --classes \"$DATA_ROOT/annotation/classes.json\" --task action_recognition --max-frames-per-clip 8 --output-format structured --output data/processed/kabr_behavior_train_structured.jsonl && \"$CLI_BIN\" convert-kabr-split --annotation \"$DATA_ROOT/annotation/val.csv\" --image-root \"$DATA_ROOT/dataset/image\" --classes \"$DATA_ROOT/annotation/classes.json\" --task action_recognition --max-frames-per-clip 8 --output-format structured --output data/processed/kabr_behavior_val_structured.jsonl" \
  "$LOG_DIR/behavior_structured_data_convert.log" \
  "$ROOT_DIR/data/processed" \
  "data/processed/kabr_behavior_val_structured.jsonl"

run_step \
  "behavior_4frame_data" \
  "convert" \
  "\"$CLI_BIN\" convert-kabr-split --annotation \"$DATA_ROOT/annotation/train.csv\" --image-root \"$DATA_ROOT/dataset/image\" --classes \"$DATA_ROOT/annotation/classes.json\" --task action_recognition --max-frames-per-clip 4 --output-format plain --output data/processed/kabr_behavior_train_4frames.jsonl && \"$CLI_BIN\" convert-kabr-split --annotation \"$DATA_ROOT/annotation/val.csv\" --image-root \"$DATA_ROOT/dataset/image\" --classes \"$DATA_ROOT/annotation/classes.json\" --task action_recognition --max-frames-per-clip 4 --output-format plain --output data/processed/kabr_behavior_val_4frames.jsonl" \
  "$LOG_DIR/behavior_4frame_data_convert.log" \
  "$ROOT_DIR/data/processed" \
  "data/processed/kabr_behavior_val_4frames.jsonl"

run_experiment \
  "behavior_frozen_linear" \
  "$ROOT_DIR/artifacts/paper/behavior/frozen/qwen25vl3b/linear" \
  "\"$CLI_BIN\" train-frozen-features --config configs/qwen25vl3b_behavior_frozen_linear.yaml" \
  "\"$CLI_BIN\" eval-frozen-features --config configs/qwen25vl3b_behavior_frozen_linear.yaml --checkpoint-path artifacts/paper/behavior/frozen/qwen25vl3b/linear/classifier.pt --output artifacts/paper/behavior/frozen/qwen25vl3b/linear/predictions.jsonl" \
  "artifacts/paper/behavior/frozen/qwen25vl3b/linear/training_metrics.json" \
  "artifacts/paper/behavior/frozen/qwen25vl3b/linear/predictions.metrics.json"

run_experiment \
  "behavior_lora_structured" \
  "$ROOT_DIR/artifacts/paper/behavior/lora/qwen25vl3b/structured" \
  "\"$PYTHON_BIN\" -m kabr_vlm.training.finetune --config configs/qwen25vl3b_behavior_lora_structured.yaml" \
  "\"$CLI_BIN\" eval-finetuned-hf-vlm --annotation \"$DATA_ROOT/annotation/val.csv\" --image-root \"$DATA_ROOT/dataset/image\" --classes \"$DATA_ROOT/annotation/classes.json\" --model-id Qwen/Qwen2.5-VL-3B-Instruct --adapter-path artifacts/paper/behavior/lora/qwen25vl3b/structured/checkpoint --task behavior --prompt-mode structured --output artifacts/paper/behavior/lora/qwen25vl3b/structured/behavior_val_predictions.jsonl" \
  "artifacts/paper/behavior/lora/qwen25vl3b/structured/checkpoint/training_metrics.json" \
  "artifacts/paper/behavior/lora/qwen25vl3b/structured/behavior_val_predictions.metrics.json"

run_experiment \
  "behavior_lora_llava_onevision_qwen2_0_5b_plain_8frames" \
  "$ROOT_DIR/artifacts/runs/llava_onevision_qwen2_0_5b_behavior_lora_full" \
  "bash \"$ROOT_DIR/scripts/run_llava_onevision_qwen2_0_5b_behavior_lora_full.sh\"" \
  "bash \"$ROOT_DIR/scripts/run_llava_onevision_qwen2_0_5b_kabr_clip_lora_eval.sh\"" \
  "artifacts/runs/llava_onevision_qwen2_0_5b_behavior_lora_full/checkpoint/training_metrics.json" \
  "artifacts/runs/llava_onevision_qwen2_0_5b_behavior_lora_full/behavior_val_predictions.metrics.json"

run_experiment \
  "behavior_lora_llava_onevision_qwen2_7b_plain_8frames" \
  "$ROOT_DIR/artifacts/runs/llava_onevision_qwen2_7b_behavior_lora_full" \
  "bash \"$ROOT_DIR/scripts/run_llava_onevision_qwen2_7b_behavior_lora_full.sh\"" \
  "bash \"$ROOT_DIR/scripts/run_llava_onevision_qwen2_7b_kabr_clip_lora_eval.sh\"" \
  "artifacts/runs/llava_onevision_qwen2_7b_behavior_lora_full/checkpoint/training_metrics.json" \
  "artifacts/runs/llava_onevision_qwen2_7b_behavior_lora_full/behavior_val_predictions.metrics.json"

run_experiment \
  "behavior_lora_llava_onevision_1_5_4b_plain_8frames" \
  "$ROOT_DIR/artifacts/runs/llava_onevision_1_5_4b_behavior_lora_full" \
  "bash \"$ROOT_DIR/scripts/run_llava_onevision_1_5_4b_behavior_lora_full.sh\"" \
  "bash \"$ROOT_DIR/scripts/run_llava_onevision_1_5_4b_kabr_clip_lora_eval.sh\"" \
  "artifacts/runs/llava_onevision_1_5_4b_behavior_lora_full/checkpoint/training_metrics.json" \
  "artifacts/runs/llava_onevision_1_5_4b_behavior_lora_full/behavior_val_predictions.metrics.json"

run_experiment \
  "behavior_lora_llava_v1_6_mistral_7b_plain_8frames" \
  "$ROOT_DIR/artifacts/runs/llava_v1_6_mistral_7b_behavior_lora_full" \
  "bash \"$ROOT_DIR/scripts/run_llava_v1_6_mistral_7b_behavior_lora_full.sh\"" \
  "bash \"$ROOT_DIR/scripts/run_llava_v1_6_mistral_7b_kabr_clip_lora_eval.sh\"" \
  "artifacts/runs/llava_v1_6_mistral_7b_behavior_lora_full/checkpoint/training_metrics.json" \
  "artifacts/runs/llava_v1_6_mistral_7b_behavior_lora_full/behavior_val_predictions.metrics.json"

run_experiment \
  "behavior_distill_alpha70" \
  "$ROOT_DIR/artifacts/paper/behavior/distill/qwen25vl3b/mlp_alpha70" \
  "mkdir -p artifacts/paper/behavior/distill/qwen25vl3b/mlp_alpha70/features && cp -n artifacts/paper/behavior/distill/qwen25vl3b/mlp/features/train.pt artifacts/paper/behavior/distill/qwen25vl3b/mlp_alpha70/features/train.pt && cp -n artifacts/paper/behavior/distill/qwen25vl3b/mlp/features/val.pt artifacts/paper/behavior/distill/qwen25vl3b/mlp_alpha70/features/val.pt && cp -n artifacts/paper/behavior/distill/qwen25vl3b/mlp/teacher_probs.pt artifacts/paper/behavior/distill/qwen25vl3b/mlp_alpha70/teacher_probs.pt && \"$CLI_BIN\" train-distill --config configs/qwen25vl3b_behavior_distill_mlp_alpha70.yaml" \
  "\"$CLI_BIN\" eval-distill --config configs/qwen25vl3b_behavior_distill_mlp_alpha70.yaml --checkpoint-path artifacts/paper/behavior/distill/qwen25vl3b/mlp_alpha70/classifier.pt --output artifacts/paper/behavior/distill/qwen25vl3b/mlp_alpha70/predictions.jsonl" \
  "artifacts/paper/behavior/distill/qwen25vl3b/mlp_alpha70/training_metrics.json" \
  "artifacts/paper/behavior/distill/qwen25vl3b/mlp_alpha70/predictions.metrics.json"

run_experiment \
  "behavior_lora_plain_4frames" \
  "$ROOT_DIR/artifacts/paper/behavior/lora/qwen25vl3b/plain_4frames" \
  "\"$PYTHON_BIN\" -m kabr_vlm.training.finetune --config configs/qwen25vl3b_behavior_lora_plain_4frames.yaml" \
  "\"$CLI_BIN\" eval-finetuned-hf-vlm --annotation \"$DATA_ROOT/annotation/val.csv\" --image-root \"$DATA_ROOT/dataset/image\" --classes \"$DATA_ROOT/annotation/classes.json\" --model-id Qwen/Qwen2.5-VL-3B-Instruct --adapter-path artifacts/paper/behavior/lora/qwen25vl3b/plain_4frames/checkpoint --task behavior --prompt-mode label --output artifacts/paper/behavior/lora/qwen25vl3b/plain_4frames/behavior_val_predictions.jsonl" \
  "artifacts/paper/behavior/lora/qwen25vl3b/plain_4frames/checkpoint/training_metrics.json" \
  "artifacts/paper/behavior/lora/qwen25vl3b/plain_4frames/behavior_val_predictions.metrics.json"

run_step \
  "behavior_reporting" \
  "aggregate" \
  "\"$CLI_BIN\" aggregate-paper-results --manifest configs/paper_runs_behavior.yaml --output-dir artifacts/paper_tables_behavior" \
  "$LOG_DIR/behavior_reporting_aggregate.log" \
  "$ROOT_DIR/artifacts/paper_tables_behavior" \
  "artifacts/paper_tables_behavior/main_table.md"

printf "suite: completed\n" > "$CURRENT_FILE"
