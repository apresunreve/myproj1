# KABR Clip-Level Evaluation Guide

This guide explains how to run evaluation on the current **derived clip-level KABR behavior dataset** used in this project.

The key idea is:

- one clip = one contiguous run of frames
- from the same `original_video_id`
- with the same behavior label

Both the VLM pipeline and the X3D pipeline evaluate on those same derived clip units.

## Dataset Inputs

KABR source files:

- annotation CSVs:
  - `/home/jhyan/KABR_files/annotation/train.csv`
  - `/home/jhyan/KABR_files/annotation/val.csv`
- classes:
  - `/home/jhyan/KABR_files/annotation/classes.json`
- images:
  - `/home/jhyan/KABR_files/dataset/image`

Current clip-level val size:

- `3761` clips

## 1. VLM Clip-Level Evaluation

The VLM path evaluates directly from the frame annotations and derives clips on the fly.

Main CLI:

- [cli.py](/home/jhyan/myproj/src/kabr_vlm/cli.py)
- command: `eval-clip-hf-vlm`

Example zero-shot evaluation:

```bash
cd /home/jhyan/myproj
./.uvenv/bin/kabr-vlm eval-clip-hf-vlm \
  --dataset kabr \
  --annotation /home/jhyan/KABR_files/annotation/val.csv \
  --image-root /home/jhyan/KABR_files/dataset/image \
  --classes /home/jhyan/KABR_files/annotation/classes.json \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --task behavior \
  --prompt-mode label \
  --max-samples 3761 \
  --max-frames-per-clip 8 \
  --max-new-tokens 16 \
  --output artifacts/full_eval/qwen25vl3b_behavior_clip_val_full.jsonl
```

Then score it:

```bash
cd /home/jhyan/myproj
./.uvenv/bin/kabr-vlm score-predictions \
  --predictions artifacts/full_eval/qwen25vl3b_behavior_clip_val_full.jsonl \
  --metrics-output artifacts/full_eval/qwen25vl3b_behavior_clip_val_full.metrics.json \
  --confusion-output artifacts/full_eval/qwen25vl3b_behavior_clip_val_full.confusion.csv \
  --classes-path /home/jhyan/KABR_files/annotation/classes.json
```

Adapter-backed VLM evaluation:

```bash
cd /home/jhyan/myproj
./.uvenv/bin/kabr-vlm eval-finetuned-hf-vlm \
  --dataset kabr \
  --annotation /home/jhyan/KABR_files/annotation/val.csv \
  --image-root /home/jhyan/KABR_files/dataset/image \
  --classes /home/jhyan/KABR_files/annotation/classes.json \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter-path artifacts/runs/qwen25vl3b_behavior_lora_full/checkpoint \
  --task behavior \
  --prompt-mode label \
  --max-samples 3761 \
  --max-frames-per-clip 8 \
  --output artifacts/runs/qwen25vl3b_behavior_lora_full/behavior_val_predictions.jsonl
```

What this produces:

- predictions JSONL
- metrics JSON
- confusion CSV

Important code paths:

- clip derivation:
  - [kabr_frames.py](/home/jhyan/myproj/src/kabr_vlm/data/kabr_frames.py)
- VLM eval logic:
  - [vlm_baselines.py](/home/jhyan/myproj/src/kabr_vlm/vlm_baselines.py)
- adapter-backed eval:
  - [qwen_tasks.py](/home/jhyan/myproj/src/kabr_vlm/training/qwen_tasks.py)

## 2. X3D Clip-Level Evaluation

The X3D path uses SlowFast and consumes prebuilt clip manifests.

Main dataset loader:

- [kabr_clips.py](/home/jhyan/SlowFast/slowfast/datasets/kabr_clips.py)

Main config:

- [X3D_CLIP_LEVEL.yaml](/home/jhyan/SlowFast/configs/KABR/X3D_CLIP_LEVEL.yaml)

### Step A: Build clip manifests

If manifests do not already exist, generate them first:

```bash
cd /home/jhyan/myproj
./.uvenv/bin/kabr-vlm prepare-charades-clip-manifests \
  --annotation-dir /home/jhyan/KABR_files/annotation \
  --image-root /home/jhyan/KABR_files/dataset/image \
  --classes /home/jhyan/KABR_files/annotation/classes.json \
  --dataset kabr \
  --output-dir /home/jhyan/KABR_files/annotation/clip_manifests \
  --copy-val-to-test
```

Expected files:

- `/home/jhyan/KABR_files/annotation/clip_manifests/train.jsonl`
- `/home/jhyan/KABR_files/annotation/clip_manifests/val.jsonl`
- `/home/jhyan/KABR_files/annotation/clip_manifests/test.jsonl`
- `/home/jhyan/KABR_files/annotation/clip_manifests/classes.json`

### Step B: Run standalone X3D evaluation from a checkpoint

Example for `X3D-L`:

```bash
cd /home/jhyan/SlowFast
/home/jhyan/myproj/.uvenv/bin/python tools/run_net.py \
  --cfg configs/KABR/X3D_CLIP_LEVEL.yaml \
  --opts \
    TRAIN.ENABLE False \
    TEST.ENABLE True \
    TEST.CHECKPOINT_FILE_PATH /home/jhyan/SlowFast/outputs/x3d_l_kabr_clips/checkpoints/checkpoint_epoch_00120.pyth \
    DATA.PATH_TO_DATA_DIR /home/jhyan/KABR_files/annotation/clip_manifests \
    OUTPUT_DIR /home/jhyan/SlowFast/outputs/x3d_l_kabr_clips \
    DATA.NUM_FRAMES 16 \
    DATA.SAMPLING_RATE 5 \
    DATA.TRAIN_CROP_SIZE 300 \
    DATA.TEST_CROP_SIZE 300 \
    DATA.TRAIN_JITTER_SCALES "[300,300]" \
    TEST.BATCH_SIZE 4 \
    DATA_LOADER.NUM_WORKERS 0 \
    LOG_MODEL_INFO False \
    X3D.DEPTH_FACTOR 5.0 \
    X3D.WIDTH_FACTOR 2.0 \
    X3D.DIM_C1 12 \
    X3D.BOTTLENECK_FACTOR 2.25
```

This writes:

- `clip_level/results.pkl`
- `clip_level/predictions.jsonl`

### Step C: Score X3D predictions

```bash
cd /home/jhyan/myproj
./.uvenv/bin/kabr-vlm score-predictions \
  --predictions /home/jhyan/SlowFast/outputs/x3d_l_kabr_clips/clip_level/predictions.jsonl \
  --metrics-output artifacts/x3d_metrics/x3d_l_behavior_clip.metrics.json \
  --confusion-output artifacts/x3d_metrics/x3d_l_behavior_clip.confusion.csv \
  --task behavior
```

Important note:

- for X3D scoring, `--task behavior` is enough because the exported prediction rows already contain the label strings

## 3. Current Default Input Representations

VLMs:

- `8` ordered sampled frames per clip by default
- plus a behavior prompt

X3D-L:

- `16` sampled frames
- `SAMPLING_RATE=5`
- fixed spatial crop `300 x 300`

So the comparison is matched at the **clip identity** level, not at identical raw tensor format.

## 4. Useful Existing Files

Existing X3D clip-level guide:

- [KABR_CLIP_LEVEL.md](/home/jhyan/SlowFast/KABR_CLIP_LEVEL.md)

Repo-level baseline runner:

- [run_x3d_l_gemma_baselines.sh](/home/jhyan/myproj/scripts/run_x3d_l_gemma_baselines.sh)

Current KABR behavior results:

- [CURRENT_BEHAVIOR_RESULTS.md](/home/jhyan/myproj/CURRENT_BEHAVIOR_RESULTS.md)

## 5. Recommended Quick Checks

After any eval run, verify:

- predictions JSONL exists
- metrics JSON exists
- confusion CSV exists
- `evaluated` matches the expected clip count

For KABR full val behavior:

- expected `evaluated = 3761`
