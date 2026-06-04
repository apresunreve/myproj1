#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --exclude=grn025,grn026,grn027,grn028,grn029,grn030,grn031,grn032,grn066,grn067,grn080
#SBATCH -o /scratch/general/vast/j.yan/myproj1/logs/%x.%j.out
#SBATCH -e /scratch/general/vast/j.yan/myproj1/logs/%x.%j.err

# Generic baboonland eval wrapper.
# Required env: MODEL_ID, OUT_JSONL
# Optional: ADAPTER_PATH, MAX_SAMPLES, MAX_NEW_TOKENS, HF_TOKEN

set -euo pipefail

REPO=/scratch/general/vast/j.yan/myproj1
export KABR_ENABLE_FLASH_ATTN=0

CACHE_ROOT=${CACHE_ROOT:-/scratch/general/nfs1/u1320844}
export HF_HOME=${HF_HOME:-$CACHE_ROOT/huggingface-cache}
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "host: $(hostname)"
echo "date: $(date)"
echo "model: $MODEL_ID"
echo "adapter: ${ADAPTER_PATH:-<none>}"
echo "out: $OUT_JSONL"
nvidia-smi || true

cd "$REPO"

CMD=("$REPO/.venv/bin/python" "$REPO/scripts/eval_baboonland_vlm.py"
     --model_id "$MODEL_ID"
     --output_jsonl "$OUT_JSONL"
     --max_new_tokens "${MAX_NEW_TOKENS:-16}"
     --resume)

if [ -n "${ADAPTER_PATH:-}" ]; then
    CMD+=(--adapter_path "$ADAPTER_PATH")
fi
if [ -n "${MAX_SAMPLES:-}" ]; then
    CMD+=(--max_samples "$MAX_SAMPLES")
fi

"${CMD[@]}"

echo "done: $(date)"
