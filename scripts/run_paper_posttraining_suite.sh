#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.uvenv/bin/python}"
CLI_BIN="${CLI_BIN:-$ROOT_DIR/.uvenv/bin/kabr-vlm}"
STATUS_DIR="${STATUS_DIR:-$ROOT_DIR/artifacts/paper_suite}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/paper_suite}"

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
  local config_path="$2"
  local checkpoint_path="$3"
  local output_path="$4"
  local train_cmd="$5"
  local eval_cmd="$6"
  local output_dir
  output_dir="$(dirname "$output_path")"
  local checkpoint_metrics="$output_dir/training_metrics.json"
  local predictions_metrics="${output_path%.jsonl}.metrics.json"

  run_step "$run_name" "train" "$train_cmd" "$LOG_DIR/${run_name}_train.log" "$output_dir" "$checkpoint_metrics"
  run_step "$run_name" "eval" "$eval_cmd" "$LOG_DIR/${run_name}_eval.log" "$output_dir" "$predictions_metrics"
}

run_experiment \
  "species_frozen_mlp" \
  "configs/qwen25vl3b_species_frozen_mlp.yaml" \
  "artifacts/paper/species/frozen/qwen25vl3b/mlp/classifier.pt" \
  "artifacts/paper/species/frozen/qwen25vl3b/mlp/predictions.jsonl" \
  "\"$CLI_BIN\" train-frozen-features --config configs/qwen25vl3b_species_frozen_mlp.yaml" \
  "\"$CLI_BIN\" eval-frozen-features --config configs/qwen25vl3b_species_frozen_mlp.yaml --checkpoint-path artifacts/paper/species/frozen/qwen25vl3b/mlp/classifier.pt --output artifacts/paper/species/frozen/qwen25vl3b/mlp/predictions.jsonl"

run_experiment \
  "behavior_frozen_mlp" \
  "configs/qwen25vl3b_behavior_frozen_mlp.yaml" \
  "artifacts/paper/behavior/frozen/qwen25vl3b/mlp/classifier.pt" \
  "artifacts/paper/behavior/frozen/qwen25vl3b/mlp/predictions.jsonl" \
  "\"$CLI_BIN\" train-frozen-features --config configs/qwen25vl3b_behavior_frozen_mlp.yaml" \
  "\"$CLI_BIN\" eval-frozen-features --config configs/qwen25vl3b_behavior_frozen_mlp.yaml --checkpoint-path artifacts/paper/behavior/frozen/qwen25vl3b/mlp/classifier.pt --output artifacts/paper/behavior/frozen/qwen25vl3b/mlp/predictions.jsonl"

run_experiment \
  "species_distill_mlp" \
  "configs/qwen25vl3b_species_distill_mlp.yaml" \
  "artifacts/paper/species/distill/qwen25vl3b/mlp/classifier.pt" \
  "artifacts/paper/species/distill/qwen25vl3b/mlp/predictions.jsonl" \
  "\"$CLI_BIN\" train-distill --config configs/qwen25vl3b_species_distill_mlp.yaml" \
  "\"$CLI_BIN\" eval-distill --config configs/qwen25vl3b_species_distill_mlp.yaml --checkpoint-path artifacts/paper/species/distill/qwen25vl3b/mlp/classifier.pt --output artifacts/paper/species/distill/qwen25vl3b/mlp/predictions.jsonl"

run_experiment \
  "behavior_distill_mlp" \
  "configs/qwen25vl3b_behavior_distill_mlp.yaml" \
  "artifacts/paper/behavior/distill/qwen25vl3b/mlp/classifier.pt" \
  "artifacts/paper/behavior/distill/qwen25vl3b/mlp/predictions.jsonl" \
  "\"$CLI_BIN\" train-distill --config configs/qwen25vl3b_behavior_distill_mlp.yaml" \
  "\"$CLI_BIN\" eval-distill --config configs/qwen25vl3b_behavior_distill_mlp.yaml --checkpoint-path artifacts/paper/behavior/distill/qwen25vl3b/mlp/classifier.pt --output artifacts/paper/behavior/distill/qwen25vl3b/mlp/predictions.jsonl"

printf "suite: completed\n" > "$CURRENT_FILE"
