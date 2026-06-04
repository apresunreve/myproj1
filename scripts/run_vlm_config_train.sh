#!/usr/bin/env bash
# Generic VLM LoRA training launcher that trusts the config's data paths
# (no hard overrides to KABR or baboonland JSONLs). Used by the joint /
# multi-dataset training path.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <config_path> <job_name> <model_id>"
  exit 1
fi

REPO=/scratch/general/vast/j.yan/myproj1
CONFIG_PATH=$1
JOB_NAME=$2
MODEL_ID=$3

CACHE_ROOT=${CACHE_ROOT:-/scratch/general/nfs1/u1320844}
UV_CACHE_DIR=${UV_CACHE_DIR:-$CACHE_ROOT/uv-cache}
HF_HOME=${HF_HOME:-$CACHE_ROOT/huggingface-cache}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-$CACHE_ROOT/xdg-cache}
KABR_ENABLE_FLASH_ATTN=${KABR_ENABLE_FLASH_ATTN:-0}
PYTHON_BIN=""
CLI_PREFIX=()

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"
export UV_CACHE_DIR HF_HOME XDG_CACHE_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

cd "$REPO"

TMP_CFG=$(mktemp /tmp/"$JOB_NAME".runtime.XXXXXX.yaml)
cleanup() { rm -f "$TMP_CFG"; }
trap cleanup EXIT

/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python - <<'PY' "$CONFIG_PATH" "$TMP_CFG" "$MODEL_ID"
import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
model_id = sys.argv[3]

cfg = yaml.safe_load(src.read_text())
cfg["model"]["name"] = model_id
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

RUN_ROOT=$(/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python - <<'PY' "$TMP_CFG"
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(cfg["training"]["output_dir"]).parent)
PY
)
TRAIN_OUT="$RUN_ROOT/checkpoint"
mkdir -p "$RUN_ROOT"

if [ "$KABR_ENABLE_FLASH_ATTN" = "1" ]; then
  PYTHON_BIN="$(bash "$REPO/scripts/ensure_flash_attn_env.sh" "$REPO" train)"
  CLI_PREFIX=("$PYTHON_BIN")
elif [ -x "$REPO/.venv/bin/python" ]; then
  PYTHON_BIN="$REPO/.venv/bin/python"
  CLI_PREFIX=("$PYTHON_BIN")
else
  CLI_PREFIX=(/uufs/chpc.utah.edu/common/home/u1320844/.local/bin/uv run
    --project "$REPO" --extra train --extra infer python)
fi

RUN_TRAIN=("${CLI_PREFIX[@]}" -m kabr_vlm.training.finetune --config "$TMP_CFG")

LATEST_CHECKPOINT=$(/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python - <<'PY' "$TRAIN_OUT"
import sys
from pathlib import Path

train_out = Path(sys.argv[1])
latest = None
latest_step = -1
for cand in train_out.glob("checkpoint-*"):
    if not cand.is_dir():
        continue
    try:
        step = int(cand.name.split("-", 1)[1])
    except (IndexError, ValueError):
        continue
    if step > latest_step:
        latest_step = step
        latest = cand
print("" if latest is None else latest)
PY
)

if [ -n "$LATEST_CHECKPOINT" ]; then
  echo "=== Resuming train for $JOB_NAME from $LATEST_CHECKPOINT ==="
  RUN_TRAIN+=(--resume-from-checkpoint "$LATEST_CHECKPOINT")
fi

echo "=== Running joint LoRA train: $JOB_NAME ==="
"${RUN_TRAIN[@]}" 2>&1 | tee "$RUN_ROOT/train.log"

[ -d "$TRAIN_OUT" ] || { echo "Missing adapter dir: $TRAIN_OUT"; exit 1; }

echo "Done. job=$JOB_NAME train_dir=$TRAIN_OUT"
