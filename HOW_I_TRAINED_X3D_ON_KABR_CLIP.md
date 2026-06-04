# How I Trained X3D Models on the KABR Clip Dataset

This note records the actual training path used in this repo for the supervised
`X3D` clip baselines on the derived KABR clip dataset.

The short version is:

- training was done through `SlowFast`, not the VLM training stack
- the dataset was the derived clip-manifest version of KABR
- the entrypoint in this repo was [`run_kabr_clip_video_train.sh`](/scratch/general/vast/j.yan/myproj1/scripts/run_kabr_clip_video_train.sh)
- the concrete wrappers were:
  - [`run_x3d_xs_kabr_clip_train.sh`](/scratch/general/vast/j.yan/myproj1/scripts/run_x3d_xs_kabr_clip_train.sh)
  - [`run_x3d_s_kabr_clip_train.sh`](/scratch/general/vast/j.yan/myproj1/scripts/run_x3d_s_kabr_clip_train.sh)
  - [`run_x3d_m_kabr_clip_train.sh`](/scratch/general/vast/j.yan/myproj1/scripts/run_x3d_m_kabr_clip_train.sh)

## Dataset Setup

The X3D models were trained on the derived KABR clip dataset rooted at:

- clip dataset root: `/scratch/general/nfs1/u1320844/KABR_clip_dataset_cleaned_2026-04-07`
- clip manifests: `/scratch/general/nfs1/u1320844/KABR_clip_dataset_cleaned_2026-04-07/annotation/clip_manifests`
- image root: `/scratch/general/nfs1/u1320844/KABR_files2/dataset/image`
- classes file: `/scratch/general/nfs1/u1320844/KABR_files2/annotation/classes.json`

The required manifest files were:

- `train.jsonl`
- `val.jsonl`

Important detail:

- X3D used the same derived clip identities as the VLM clip experiments
- but it did **not** use the VLM’s sampled still-frame list directly
- instead, `SlowFast` sampled a fixed video tensor from each derived clip

## Training Code Path

The main training script was:

- [`run_kabr_clip_video_train.sh`](/scratch/general/vast/j.yan/myproj1/scripts/run_kabr_clip_video_train.sh)

That script launches `SlowFast` here:

- `SLOWFAST_ROOT=/scratch/general/vast/j.yan/SlowFast`
- command:

```bash
tools/run_net.py --cfg <SlowFast base config> --opts ...
```

It expects the custom clip dataset loader to exist in the external SlowFast tree:

- `/scratch/general/vast/j.yan/SlowFast/slowfast/datasets/kabr_clips.py`

The repo-level base config used for the clip-level setup is:

- [`x3d_clip_level.yaml`](/scratch/general/vast/j.yan/myproj1/configs/x3d_clip_level.yaml)

In practice, the training script uses model-specific upstream SlowFast configs as
the starting point and overrides them with KABR clip settings at runtime.

## Commands I Used

### X3D-XS

```bash
bash /scratch/general/vast/j.yan/myproj1/scripts/run_x3d_xs_kabr_clip_train.sh
```

### X3D-S

```bash
bash /scratch/general/vast/j.yan/myproj1/scripts/run_x3d_s_kabr_clip_train.sh
```

### X3D-M

```bash
bash /scratch/general/vast/j.yan/myproj1/scripts/run_x3d_m_kabr_clip_train.sh
```

Those wrappers just call:

```bash
bash /scratch/general/vast/j.yan/myproj1/scripts/run_kabr_clip_video_train.sh <model-key>
```

where `<model-key>` is one of:

- `x3d-xs`
- `x3d-s`
- `x3d-m`

## What `run_kabr_clip_video_train.sh` Sets

For all X3D runs, the script forces the training/eval setup to use:

- `TRAIN.DATASET kabr_clips`
- `TEST.DATASET kabr_clips`
- `DATA.PATH_TO_DATA_DIR <clip manifest dir>`
- `DATA.PATH_PREFIX <image root>`
- `MODEL.NUM_CLASSES 8`
- `MODEL.LOSS_FUNC cross_entropy`
- `MODEL.HEAD_ACT softmax`
- `TEST.SAVE_RESULTS_PATH clip_level/results.pkl`
- `TEST.SAVE_CLIP_RESULTS_PATH clip_level/predictions.jsonl`
- `SOLVER.MAX_EPOCH 120`
- `TRAIN.EVAL_PERIOD 5`
- `TRAIN.CHECKPOINT_PERIOD 5`
- `TRAIN.AUTO_RESUME True`

Other common defaults:

- `NUM_GPUS=1`
- `NUM_SHARDS=1`
- `RNG_SEED=0`
- `NUM_WORKERS=8`
- `TRAIN_MIXED_PRECISION=False`

Outputs were written under:

- `/scratch/general/nfs1/u1320844/manual_train/kabr_clip/<model_name>`

For example:

- `x3d_xs` -> `/scratch/general/nfs1/u1320844/manual_train/kabr_clip/x3d_xs`
- `x3d_s` -> `/scratch/general/nfs1/u1320844/manual_train/kabr_clip/x3d_s`
- `x3d_m` -> `/scratch/general/nfs1/u1320844/manual_train/kabr_clip/x3d_m`

Each run writes:

- `train.log`
- `checkpoints/checkpoint_epoch_*.pyth`
- `clip_level/results.pkl`
- `clip_level/predictions.jsonl`

## Model-Specific Defaults

### X3D-XS

- base SlowFast config: `configs/Kinetics/X3D_XS.yaml`
- train batch size: `16`
- test batch size: `8`
- base LR: `0.0125`
- frames: `4`
- sampling rate: `12`
- train crop: `160`
- test crop: `182`
- train jitter scales: `[182,228]`
- test ensemble views: `2`
- test spatial crops: `1`
- width factor: `2.0`
- depth factor: `2.2`
- bottleneck factor: `2.25`
- `DIM_C1=12`
- `DIM_C5=2048`

### X3D-S

- base SlowFast config: `configs/Kinetics/X3D_S.yaml`
- train batch size: `8`
- test batch size: `4`
- base LR: `0.0125`
- frames: `13`
- sampling rate: `6`
- train crop: `160`
- test crop: `182`
- train jitter scales: `[182,228]`
- test ensemble views: `2`
- test spatial crops: `1`
- width factor: `2.0`
- depth factor: `2.2`
- bottleneck factor: `2.25`
- `DIM_C1=12`
- `DIM_C5=2048`

### X3D-M

- base SlowFast config: `configs/Kinetics/X3D_M.yaml`
- train batch size: `4`
- test batch size: `4`
- base LR: `0.0125`
- frames: `16`
- sampling rate: `5`
- train crop: `224`
- test crop: `256`
- train jitter scales: `[256,320]`
- test ensemble views: `2`
- test spatial crops: `1`
- width factor: `2.0`
- depth factor: `2.2`
- bottleneck factor: `2.25`
- `DIM_C1=12`
- `DIM_C5=2048`

## Notes on X3D-L

The generic X3D training wrappers in this repo were for `XS`, `S`, and `M`.

For `X3D-L`, the training/eval flow was handled separately through:

- [`run_x3d_l_gemma_baselines.sh`](/scratch/general/vast/j.yan/myproj1/scripts/run_x3d_l_gemma_baselines.sh)

That script still used the same derived KABR clip setup and the same external
SlowFast clip dataset loader, but with explicit `X3D-L` overrides at launch time.

## How I Scored the Trained X3D Models

After training/eval, the exported X3D predictions were scored with the repo CLI:

```bash
cd /scratch/general/vast/j.yan/myproj1
/scratch/general/vast/j.yan/myproj1/.uvenv/bin/python -m kabr_vlm.cli score-predictions \
  --predictions <OUTPUT_DIR>/clip_level/predictions.jsonl \
  --metrics-output artifacts/x3d_metrics/<run_name>.metrics.json \
  --confusion-output artifacts/x3d_metrics/<run_name>.confusion.csv \
  --task behavior
```

Examples already referenced in this repo:

- `artifacts/x3d_metrics/x3d_xs_behavior_clip.metrics.json`
- `artifacts/x3d_metrics/x3d_s_behavior_clip.metrics.json`
- `artifacts/x3d_metrics/x3d_m_behavior_clip.metrics.json`

## One-Line Summary

I trained the KABR clip-level X3D baselines by launching `SlowFast` with the
custom `kabr_clips` dataset loader, using derived KABR clip manifests plus the
shared image root, through the repo wrapper
[`run_kabr_clip_video_train.sh`](/scratch/general/vast/j.yan/myproj1/scripts/run_kabr_clip_video_train.sh)
and its `X3D-XS/S/M` convenience scripts.
