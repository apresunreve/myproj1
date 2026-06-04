#!/usr/bin/env bash
# Run inference latency benchmark for all evaluated VLMs on KABR and Baboonland val.
#
# Usage:
#   bash scripts/run_latency_benchmark.sh            # zero-shot only
#   INCLUDE_LORA=1 bash scripts/run_latency_benchmark.sh   # also bench LoRA adapters
#
# Outputs go to artifacts/latency/
# Each model × dataset produces:
#   artifacts/latency/<slug>_<dataset>_<mode>.jsonl           (per-clip predictions + latency_ms)
#   artifacts/latency/<slug>_<dataset>_<mode>.latency.json    (compact summary)
#
# After all runs, call:
#   python scripts/plot_pareto.py

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CLI_BIN="${CLI_BIN:-$ROOT_DIR/.uvenv/bin/kabr-vlm}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/artifacts/latency}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/latency_benchmark}"
N_CLIPS="${N_CLIPS:-100}"
WARMUP="${WARMUP:-5}"
FRAMES="${FRAMES:-8}"
INCLUDE_LORA="${INCLUDE_LORA:-0}"

# ── Data paths ─────────────────────────────────────────────────────────────
KABR_ROOT="${KABR_ROOT:-/scratch/general/nfs1/u1320844/KABR_files2}"
KABR_ANN="$KABR_ROOT/annotation/val.csv"
KABR_IMG="$KABR_ROOT/dataset/image"
KABR_CLS="$KABR_ROOT/annotation/classes.json"

BBL_ROOT="${BBL_ROOT:-/home/jhyan/baboon_data/charades}"
BBL_ANN="$BBL_ROOT/annotation/val.csv"
BBL_IMG="$BBL_ROOT/dataset/image"
BBL_CLS="$BBL_ROOT/annotation/classes.json"

# ── LoRA adapter paths ─────────────────────────────────────────────────────
# Only used when INCLUDE_LORA=1
QWEN3B_KABR_ADAPTER="${QWEN3B_KABR_ADAPTER:-$ROOT_DIR/artifacts/paper/behavior/lora/qwen25vl3b/plain/checkpoint}"
QWEN3B_JOINT_ADAPTER="${QWEN3B_JOINT_ADAPTER:-$ROOT_DIR/artifacts/paper/behavior/lora/joint_a1/checkpoint}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

# ── Model list ─────────────────────────────────────────────────────────────
declare -A MODELS
MODELS["qwen25vl_3b"]="Qwen/Qwen2.5-VL-3B-Instruct"
MODELS["qwen25vl_7b"]="Qwen/Qwen2.5-VL-7B-Instruct"
MODELS["llava_ov_0_5b"]="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
MODELS["llava_ov_7b"]="llava-hf/llava-onevision-qwen2-7b-ov-hf"
MODELS["gemma3_4b"]="google/gemma-3-4b-it"
MODELS["minicpm_v46"]="openbmb/MiniCPM-V-4_6"

run_bench() {
  local slug="$1"
  local model_id="$2"
  local dataset="$3"   # kabr or baboonland
  local mode="$4"      # zero-shot or lora
  local adapter="${5:-}"

  local ann img cls
  if [[ "$dataset" == "kabr" ]]; then
    ann="$KABR_ANN"; img="$KABR_IMG"; cls="$KABR_CLS"
  else
    ann="$BBL_ANN"; img="$BBL_IMG"; cls="$BBL_CLS"
  fi

  local out="$OUT_DIR/${slug}_${dataset}_${mode}.jsonl"
  local log="$LOG_DIR/${slug}_${dataset}_${mode}.log"

  if [[ -f "${out%.jsonl}.latency.json" ]]; then
    echo "SKIP  $slug / $dataset / $mode  (already done)"
    return
  fi

  echo "RUN   $slug / $dataset / $mode ..."
  local cmd=(
    "$CLI_BIN" bench-latency
    --annotation "$ann"
    --image-root  "$img"
    --classes     "$cls"
    --model-id    "$model_id"
    --dataset     "$dataset"
    --task        behavior
    --n-clips     "$N_CLIPS"
    --warmup-clips "$WARMUP"
    --max-frames-per-clip "$FRAMES"
    --output      "$out"
  )
  if [[ -n "$adapter" ]]; then
    cmd+=(--adapter-path "$adapter")
  fi

  if "${cmd[@]}" 2>&1 | tee "$log"; then
    echo "DONE  $slug / $dataset / $mode"
  else
    echo "FAIL  $slug / $dataset / $mode  (see $log)"
  fi
}

# ── Zero-shot benchmarks ───────────────────────────────────────────────────
echo "=== Zero-shot benchmarks ==="
for slug in "${!MODELS[@]}"; do
  model_id="${MODELS[$slug]}"
  run_bench "$slug" "$model_id" "kabr"       "zero-shot"
  run_bench "$slug" "$model_id" "baboonland" "zero-shot"
done

# ── LoRA benchmarks (opt-in) ───────────────────────────────────────────────
if [[ "$INCLUDE_LORA" == "1" ]]; then
  echo "=== LoRA benchmarks ==="
  run_bench "qwen25vl_3b" "${MODELS[qwen25vl_3b]}" "kabr"       "lora" "$QWEN3B_KABR_ADAPTER"
  run_bench "qwen25vl_3b" "${MODELS[qwen25vl_3b]}" "baboonland" "lora" "$QWEN3B_JOINT_ADAPTER"
fi

echo ""
echo "All done. Summaries in $OUT_DIR"
echo "Run: python scripts/plot_pareto.py --latency-dir $OUT_DIR"
