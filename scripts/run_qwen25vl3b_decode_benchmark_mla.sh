#!/usr/bin/env bash
set -euo pipefail

REPO=/scratch/general/vast/j.yan/myproj1
ANNOTATION=${ANNOTATION:-/scratch/general/nfs1/u1320844/KABR_files2/annotation/val.cleaned.csv}
IMAGE_ROOT=${IMAGE_ROOT:-/scratch/general/nfs1/u1320844/KABR_files2/dataset/image}
CLASSES=${CLASSES:-/scratch/general/nfs1/u1320844/KABR_files2/annotation/classes.json}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-VL-3B-Instruct}
OUT_ROOT=${OUT_ROOT:-/scratch/general/nfs1/u1320844/manual_eval/qwen25vl3b_kabr_clip_mla_benchmark}
OUTPUT_JSON=${OUTPUT_JSON:-$OUT_ROOT/decode_benchmark.json}
TASK=${TASK:-behavior}
PROMPT_MODE=${PROMPT_MODE:-label}
MAX_FRAMES_PER_CLIP=${MAX_FRAMES_PER_CLIP:-8}
CLIP_INDEX=${CLIP_INDEX:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
WARMUP_RUNS=${WARMUP_RUNS:-1}
BENCHMARK_RUNS=${BENCHMARK_RUNS:-5}
FORCE_EXACT_NEW_TOKENS=${FORCE_EXACT_NEW_TOKENS:-true}

mkdir -p "$OUT_ROOT"
cd "$REPO"

RUN=(
  /uufs/chpc.utah.edu/common/home/u1320844/.local/bin/uv run
  --project "$REPO"
  --extra infer
  --extra train
  python
)

CMD=(
  "${RUN[@]}"
  "$REPO/scripts/benchmark_qwen25vl3b_decode_mla.py"
  --annotation "$ANNOTATION"
  --image-root "$IMAGE_ROOT"
  --classes "$CLASSES"
  --model-id "$MODEL_ID"
  --task "$TASK"
  --prompt-mode "$PROMPT_MODE"
  --max-frames-per-clip "$MAX_FRAMES_PER_CLIP"
  --clip-index "$CLIP_INDEX"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --warmup-runs "$WARMUP_RUNS"
  --benchmark-runs "$BENCHMARK_RUNS"
  --output "$OUTPUT_JSON"
)

if [ "$FORCE_EXACT_NEW_TOKENS" = "true" ]; then
  CMD+=( --force-exact-new-tokens )
fi

echo "=== Running Qwen2.5-VL 3B baseline vs MLA decode benchmark ==="
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee "$OUT_ROOT/benchmark.log"

echo "Done."
echo "Benchmark JSON: $OUTPUT_JSON"
echo "Log:            $OUT_ROOT/benchmark.log"
