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
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-16}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-8}

TRAIN_FILE=/scratch/general/vast/j.yan/myproj1/data/processed/kabr_behavior_train.jsonl
VAL_FILE=/scratch/general/vast/j.yan/myproj1/data/processed/kabr_behavior_val_cleaned.jsonl
ANNOTATION=/scratch/general/nfs1/u1320844/KABR_files2/annotation/val.cleaned.csv
IMAGE_ROOT=/scratch/general/nfs1/u1320844/KABR_files2/dataset/image
CLASSES=/scratch/general/nfs1/u1320844/KABR_files2/annotation/classes.json
RUN_ROOT=/scratch/general/nfs1/u1320844/manual_smoke/vlm/$JOB_NAME
TRAIN_OUT=$RUN_ROOT/checkpoint
EVAL_OUT_DIR=$RUN_ROOT/eval
EVAL_OUT=$EVAL_OUT_DIR/behavior_val_predictions.jsonl
TMP_CFG=$RUN_ROOT/config.runtime.yaml
UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/general/nfs1/u1320844/uv-cache}
HF_HOME=${HF_HOME:-/scratch/general/nfs1/u1320844/huggingface-cache}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-/scratch/general/nfs1/u1320844/xdg-cache}

mkdir -p "$RUN_ROOT" "$EVAL_OUT_DIR" "$HF_HOME" "$UV_CACHE_DIR" "$XDG_CACHE_HOME"

export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export UV_CACHE_DIR
export XDG_CACHE_HOME

cd "$REPO"

cleanup() {
  rm -f "$TMP_CFG"
}
trap cleanup EXIT

/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python - <<'PY' "$CONFIG_PATH" "$TMP_CFG" "$TRAIN_FILE" "$VAL_FILE" "$TRAIN_OUT" "$MODEL_ID"
import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
train_file = sys.argv[3]
val_file = sys.argv[4]
output_dir = sys.argv[5]
model_id = sys.argv[6]

cfg = yaml.safe_load(src.read_text())
cfg["data"]["train_file"] = train_file
cfg["data"]["val_file"] = val_file
cfg["training"]["output_dir"] = output_dir
cfg["model"]["name"] = model_id
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(dst)
PY

ATTENTION_EXPERIMENT=$(/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python - <<'PY' "$TMP_CFG"
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(cfg.get("model", {}).get("attention_experiment", "baseline") or "baseline")
PY
)

if [ -x "$REPO/.uvenv/bin/python" ]; then
  RUN_TRAIN=("$REPO/.uvenv/bin/python" -m kabr_vlm.training.finetune --config "$TMP_CFG")
  RUN_EVAL=("$REPO/.uvenv/bin/python" -m kabr_vlm.cli
    eval-finetuned-hf-vlm
    --annotation "$ANNOTATION"
    --image-root "$IMAGE_ROOT"
    --classes "$CLASSES"
    --model-id "$MODEL_ID"
    --adapter-path "$TRAIN_OUT"
    --task behavior
    --prompt-mode label
    --max-samples "$EVAL_MAX_SAMPLES"
    --max-new-tokens "$EVAL_MAX_NEW_TOKENS"
    --output "$EVAL_OUT")
else
  RUN_TRAIN=(/uufs/chpc.utah.edu/common/home/u1320844/.local/bin/uv run
    --project "$REPO"
    --extra train
    python -m kabr_vlm.training.finetune --config "$TMP_CFG")
  RUN_EVAL=(/uufs/chpc.utah.edu/common/home/u1320844/.local/bin/uv run
    --project "$REPO"
    --extra infer
    python -m kabr_vlm.cli
    eval-finetuned-hf-vlm
    --annotation "$ANNOTATION"
    --image-root "$IMAGE_ROOT"
    --classes "$CLASSES"
    --model-id "$MODEL_ID"
    --adapter-path "$TRAIN_OUT"
    --task behavior
    --prompt-mode label
    --max-samples "$EVAL_MAX_SAMPLES"
    --max-new-tokens "$EVAL_MAX_NEW_TOKENS"
    --output "$EVAL_OUT")
fi

if [ "$ATTENTION_EXPERIMENT" != "baseline" ]; then
  RUN_EVAL+=( --attention-experiment "$ATTENTION_EXPERIMENT" )
fi

echo "=== Running VLM train smoke for $JOB_NAME ==="
"${RUN_TRAIN[@]}" 2>&1 | tee "$RUN_ROOT/train.log"

if [ ! -d "$TRAIN_OUT" ]; then
  echo "Missing adapter directory after train: $TRAIN_OUT"
  exit 1
fi

echo "=== Running VLM eval smoke for $JOB_NAME ==="
echo "Eval max samples: $EVAL_MAX_SAMPLES"
echo "Eval max new tokens: $EVAL_MAX_NEW_TOKENS"
"${RUN_EVAL[@]}" 2>&1 | tee "$RUN_ROOT/eval.log"

echo "Done."
echo "Job name: $JOB_NAME"
echo "Train dir: $TRAIN_OUT"
echo "Eval output: $EVAL_OUT"
