#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <config_path> <job_name> <model_id>"
  exit 1
fi

REPO=/scratch/general/vast/j.yan/myproj1
CONFIG_PATH=$1
JOB_NAME=$2
MODEL_ID=$3

TRAIN_FILE=/scratch/general/vast/j.yan/myproj1/data/processed/kabr_behavior_train.jsonl
VAL_FILE=/scratch/general/vast/j.yan/myproj1/data/processed/kabr_behavior_val_cleaned.jsonl
ANNOTATION=/scratch/general/nfs1/u1320844/KABR_files2/annotation/val.cleaned.csv
IMAGE_ROOT=/scratch/general/nfs1/u1320844/KABR_files2/dataset/image
CLASSES=/scratch/general/nfs1/u1320844/KABR_files2/annotation/classes.json

CACHE_ROOT=${CACHE_ROOT:-/scratch/general/nfs1/u1320844}
UV_CACHE_DIR=${UV_CACHE_DIR:-$CACHE_ROOT/uv-cache}
HF_HOME=${HF_HOME:-$CACHE_ROOT/huggingface-cache}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$CACHE_ROOT/xdg-cache}
KABR_ENABLE_FLASH_ATTN=${KABR_ENABLE_FLASH_ATTN:-1}
PYTHON_BIN=""
CLI_PREFIX=()

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"

export UV_CACHE_DIR
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME

cd "$REPO"

TMP_CFG=$(mktemp /tmp/"$JOB_NAME".runtime.XXXXXX.yaml)

cleanup() {
  rm -f "$TMP_CFG"
}
trap cleanup EXIT

/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python - <<'PY' "$CONFIG_PATH" "$TMP_CFG" "$TRAIN_FILE" "$VAL_FILE" "$MODEL_ID"
import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
train_file = sys.argv[3]
val_file = sys.argv[4]
model_id = sys.argv[5]

cfg = yaml.safe_load(src.read_text())
cfg["data"]["train_file"] = train_file
cfg["data"]["val_file"] = val_file
cfg["model"]["name"] = model_id
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(dst)
PY

RUN_ROOT=$(/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python - <<'PY' "$TMP_CFG"
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(cfg["training"]["output_dir"]).parent)
PY
)
ATTENTION_EXPERIMENT=$(/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python - <<'PY' "$TMP_CFG"
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(cfg.get("model", {}).get("attention_experiment", "baseline") or "baseline")
PY
)
TRAIN_OUT="$RUN_ROOT/checkpoint"
EVAL_OUT="$RUN_ROOT/behavior_val_predictions.jsonl"
LATEST_CHECKPOINT=""

mkdir -p "$RUN_ROOT"

if [ "$KABR_ENABLE_FLASH_ATTN" = "1" ]; then
  echo "=== Preparing flash-attn environment for $JOB_NAME ==="
  PYTHON_BIN="$(bash "$REPO/scripts/ensure_flash_attn_env.sh" "$REPO" train)"
  CLI_PREFIX=("$PYTHON_BIN")
elif [ -x "$REPO/.uvenv/bin/python" ]; then
  PYTHON_BIN="$REPO/.uvenv/bin/python"
  CLI_PREFIX=("$PYTHON_BIN")
else
  CLI_PREFIX=(/uufs/chpc.utah.edu/common/home/u1320844/.local/bin/uv run
    --project "$REPO"
    --extra train --extra infer
    python)
fi

RUN_TRAIN=("${CLI_PREFIX[@]}" -m kabr_vlm.training.finetune --config "$TMP_CFG")
RUN_EVAL=("${CLI_PREFIX[@]}" -m kabr_vlm.cli
  eval-finetuned-hf-vlm
  --annotation "$ANNOTATION"
  --image-root "$IMAGE_ROOT"
  --classes "$CLASSES"
  --model-id "$MODEL_ID"
  --adapter-path "$TRAIN_OUT"
  --task behavior
  --prompt-mode label
  --output "$EVAL_OUT")

if [ "$ATTENTION_EXPERIMENT" != "baseline" ]; then
  RUN_EVAL+=(--attention-experiment "$ATTENTION_EXPERIMENT")
fi

LATEST_CHECKPOINT=$(/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python - <<'PY' "$TRAIN_OUT"
import sys
from pathlib import Path

train_out = Path(sys.argv[1])
latest = None
latest_step = -1
for candidate in train_out.glob("checkpoint-*"):
    if not candidate.is_dir():
        continue
    try:
        step = int(candidate.name.split("-", 1)[1])
    except (IndexError, ValueError):
        continue
    if step > latest_step:
        latest_step = step
        latest = candidate
print("" if latest is None else latest)
PY
)

if [ -n "$LATEST_CHECKPOINT" ]; then
  echo "=== Resuming train for $JOB_NAME from $LATEST_CHECKPOINT ==="
  RUN_TRAIN+=(--resume-from-checkpoint "$LATEST_CHECKPOINT")
fi

echo "=== Running VLM train full job for $JOB_NAME ==="
"${RUN_TRAIN[@]}" 2>&1 | tee "$RUN_ROOT/train.log"

if [ ! -d "$TRAIN_OUT" ]; then
  echo "Missing adapter directory after train: $TRAIN_OUT"
  exit 1
fi

echo "=== Running VLM eval full job for $JOB_NAME ==="
"${RUN_EVAL[@]}" 2>&1 | tee "$RUN_ROOT/eval.log"

echo "Done."
echo "Job name: $JOB_NAME"
echo "Train dir: $TRAIN_OUT"
echo "Eval output: $EVAL_OUT"
