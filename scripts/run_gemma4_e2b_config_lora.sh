#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <config_path> <job_name> <model_id>" >&2
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
PYTHON_BIN=${PYTHON_BIN:-$REPO/.venv_minicpm_smoke/bin/python}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Gemma4 Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"
export UV_CACHE_DIR HF_HOME XDG_CACHE_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

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

RUN_TRAIN=("$PYTHON_BIN" -m kabr_vlm.training.finetune --config "$TMP_CFG")
if [ -n "$LATEST_CHECKPOINT" ]; then
  echo "=== Resuming Gemma4 train for $JOB_NAME from $LATEST_CHECKPOINT ==="
  RUN_TRAIN+=(--resume-from-checkpoint "$LATEST_CHECKPOINT")
fi

echo "=== Running Gemma4 LoRA train for $JOB_NAME ==="
"${RUN_TRAIN[@]}" 2>&1 | tee "$RUN_ROOT/train.log"

if [ ! -d "$TRAIN_OUT" ]; then
  echo "Missing adapter directory after train: $TRAIN_OUT" >&2
  exit 1
fi

echo "Done."
echo "Job name: $JOB_NAME"
echo "Train dir: $TRAIN_OUT"
