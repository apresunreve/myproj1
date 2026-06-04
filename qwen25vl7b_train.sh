#!/usr/bin/env bash
set -euo pipefail

REPO=/scratch/general/vast/j.yan/myproj1
BASE_CFG=$REPO/configs/qwen25vl7b_behavior_lora_full.yaml
OUT_DIR=${OUT_DIR:-/scratch/general/nfs1/u1320844/manual_runs/qwen25vl7b_behavior_lora_8frames}
TRAIN_FILE=/scratch/general/vast/j.yan/myproj1/data/processed/kabr_behavior_train.jsonl
VAL_FILE=/scratch/general/vast/j.yan/myproj1/data/processed/kabr_behavior_val_cleaned.jsonl
TMP_CFG=$(mktemp --suffix=.yaml)

# Optional quick-test overrides:
#   export MAX_TRAIN_SAMPLES=32
#   export MAX_VAL_SAMPLES=32
#   export EPOCHS=1
#   export GRAD_ACCUM=8
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-}
EPOCHS=${EPOCHS:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}

mkdir -p "$OUT_DIR"
cd "$REPO"

cleanup() {
  rm -f "$TMP_CFG"
}
trap cleanup EXIT

/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python - <<'PY' \
  "$BASE_CFG" "$TMP_CFG" "$TRAIN_FILE" "$VAL_FILE" "$OUT_DIR" \
  "$MAX_TRAIN_SAMPLES" "$MAX_VAL_SAMPLES" "$EPOCHS" "$GRAD_ACCUM"
import sys
from pathlib import Path
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
train_file = sys.argv[3]
val_file = sys.argv[4]
output_dir = sys.argv[5]
max_train_samples = sys.argv[6] or None
max_val_samples = sys.argv[7] or None
epochs = int(sys.argv[8])
grad_accum = int(sys.argv[9])

cfg = yaml.safe_load(src.read_text())
cfg["data"]["train_file"] = train_file
cfg["data"]["val_file"] = val_file
cfg["data"]["max_train_samples"] = int(max_train_samples) if max_train_samples else None
cfg["data"]["max_val_samples"] = int(max_val_samples) if max_val_samples else None
cfg["training"]["output_dir"] = output_dir
cfg["training"]["epochs"] = epochs
cfg["training"]["gradient_accumulation_steps"] = grad_accum

dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(dst)
PY

if [ -x "$REPO/.uvenv/bin/python" ]; then
  CMD=("$REPO/.uvenv/bin/python" -m kabr_vlm.training.finetune --config "$TMP_CFG")
else
  CMD=(/uufs/chpc.utah.edu/common/home/u1320844/.local/bin/uv run
    --project "$REPO"
    --extra train
    python -m kabr_vlm.training.finetune --config "$TMP_CFG")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee "$OUT_DIR/train.log"

echo "Done."
echo "Output dir: $OUT_DIR"
echo "Log: $OUT_DIR/train.log"
