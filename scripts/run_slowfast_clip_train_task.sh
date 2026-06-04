#!/usr/bin/env bash
# Wrapper for SlowFast clip-level training jobs run via chpc_manager.
#
# Required keys:
#   --dataset     kabr | baboonland
#   --model_key   x3d-xs | x3d-s | x3d-m | x3d-l | i3d-l
#   --output_dir  absolute path; completion file is {output_dir}/clip_level/predictions.jsonl
# Optional:
#   --max_epoch, --clip_manifest_dir, --image_root, --classes_path
set -euo pipefail

DATASET=""
MODEL_KEY=""
OUTPUT_DIR=""
MAX_EPOCH=""
CLIP_MANIFEST_DIR=""
IMAGE_ROOT=""
CLASSES_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET=$2; shift 2 ;;
        --model_key) MODEL_KEY=$2; shift 2 ;;
        --output_dir) OUTPUT_DIR=$2; shift 2 ;;
        --max_epoch) MAX_EPOCH=$2; shift 2 ;;
        --clip_manifest_dir) CLIP_MANIFEST_DIR=$2; shift 2 ;;
        --image_root) IMAGE_ROOT=$2; shift 2 ;;
        --classes_path) CLASSES_PATH=$2; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

for req in DATASET MODEL_KEY OUTPUT_DIR; do
    if [ -z "${!req}" ]; then
        echo "Missing required --${req,,}" >&2
        exit 2
    fi
done

REPO=/scratch/general/vast/j.yan/myproj1
export OUT_ROOT="$OUTPUT_DIR"
[ -n "$MAX_EPOCH" ] && export MAX_EPOCH="$MAX_EPOCH"
[ -n "$CLIP_MANIFEST_DIR" ] && export CLIP_MANIFEST_DIR="$CLIP_MANIFEST_DIR"
[ -n "$IMAGE_ROOT" ] && export IMAGE_ROOT="$IMAGE_ROOT"
[ -n "$CLASSES_PATH" ] && export CLASSES_PATH="$CLASSES_PATH"

case "$DATASET" in
    baboonland)
        exec bash "$REPO/scripts/run_baboonland_clip_video_train.sh" "$MODEL_KEY"
        ;;
    kabr)
        exec bash "$REPO/scripts/run_kabr_clip_video_train.sh" "$MODEL_KEY"
        ;;
    *)
        echo "Unknown dataset: $DATASET" >&2
        exit 2
        ;;
esac
