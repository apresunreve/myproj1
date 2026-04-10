#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_LIST="${MODEL_LIST:-$ROOT_DIR/configs/vlm_eval_models_16frames.tsv}"
export TASK="${TASK:-behavior}"
export PROMPT_MODE="${PROMPT_MODE:-plain}"
export MAX_SAMPLES="${MAX_SAMPLES:-3761}"
export MAX_FRAMES_PER_CLIP="${MAX_FRAMES_PER_CLIP:-16}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
export OUT_DIR="${OUT_DIR:-$ROOT_DIR/artifacts/full_eval_16frames}"
export LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/vlm_suite_16frames}"
export STATUS_FILE="${STATUS_FILE:-$OUT_DIR/vlm_eval_suite_16frames_status.tsv}"
export CURRENT_FILE="${CURRENT_FILE:-$OUT_DIR/vlm_eval_suite_16frames_current.txt}"

exec "$ROOT_DIR/scripts/run_vlm_eval_suite.sh"
