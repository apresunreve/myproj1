# Current Status And Plan

## Scope

The project is currently focused on **clip-level KABR behavior recognition**.

The comparison setup is:

- same derived clip definition from KABR frame annotations
- same `val` split with `3761` clips
- same behavior label set
- same metrics pipeline: accuracy, macro F1, confusion matrix

## Dataset Construction

The current clip-level dataset is derived from the released KABR frame annotations:

- source files:
  - `/home/jhyan/KABR_files/annotation/train.csv`
  - `/home/jhyan/KABR_files/annotation/val.csv`
- one derived clip is:
  - a contiguous run of frames
  - from the same `original_video_id`
  - with the same behavior label
- a new clip starts when:
  - the original video changes
  - the behavior label changes
  - or frame ids are no longer consecutive

Current derived clip counts:

- train: `11003`
- val: `3761`

For VLMs, each clip is represented by:

- `8` ordered uniformly sampled frames by default
- one text prompt
- one generated behavior label

For X3D, each clip is represented by:

- standard X3D temporal sampling within the derived clip
- not all raw frames from the segment

## Current Reliable Behavior Results

### Zero-shot VLMs

- `Qwen2.5-VL-3B`: accuracy `0.5057`, macro F1 `0.2596`
- `Qwen2.5-VL-7B`: accuracy `0.6107`, macro F1 `0.3168`
- `LLaVA OneVision 7B`: accuracy `0.5727`, macro F1 `0.2687`
- `Gemma 3 4B`: accuracy `0.5504`, macro F1 `0.2273`
- `Idefics3-8B-Llama3`: accuracy `0.5294`, macro F1 `0.2314`

### X3D clip-level baselines

- `X3D-XS`: accuracy `0.6703`, macro F1 `0.3852`
- `X3D-S`: accuracy `0.6307`, macro F1 `0.3813`
- `X3D-M`: accuracy `0.6668`, macro F1 `0.4263`

### VLM post-training

- `Qwen 3B frozen linear`: accuracy `0.6820`, macro F1 `0.3542`
- `Qwen 3B frozen MLP`: accuracy `0.6833`, macro F1 `0.2954`
- `Qwen 3B distill MLP`: accuracy `0.6517`, macro F1 `0.2742`
- `Qwen 3B distill MLP alpha70`: accuracy `0.6200`, macro F1 `0.2134`
- `Qwen 3B LoRA plain 8 frames`: accuracy `0.8535`, macro F1 `0.4846`
- `Qwen 3B LoRA plain 4 frames`: accuracy `0.8487`, macro F1 `0.4985`

Important note:

- the existing structured-output LoRA behavior artifact is not currently being treated as a stable full benchmark result

## Current Paper Story

The strongest current empirical story is:

1. zero-shot VLMs underperform specialized video baselines on KABR behavior
2. frozen VLM features recover some useful signal, but not enough
3. LoRA post-training is the strongest VLM method we have so far
4. fine-tuned `Qwen2.5-VL-3B` beats the current `X3D-XS/S/M` clip-level baselines on this setup
5. not all post-training methods are equally effective: distillation is weaker than LoRA in the current implementation

## Current In-Progress Work

### X3D-L

We are currently training a clip-level `X3D-L` baseline on the same KABR clip setup.

Status:

- resumed from `checkpoint_epoch_00035.pyth`
- training batch size was increased from `1` to `4` to improve throughput
- resumed run uses:
  - `TRAIN.EVAL_PERIOD=1000`
  - `DATA_LOADER.NUM_WORKERS=0`
  - `BN.USE_PRECISE_STATS=False`
- reason for these changes:
  - earlier `X3D-L` runs were repeatedly killed during in-training validation
  - this did **not** look like a GPU OOM
  - reducing validation pressure made the run more stable

Current live files:

- train log:
  - `/home/jhyan/myproj/logs/baseline_expansion/x3d_l_resume_train_bs4.log`
- checkpoints:
  - `/home/jhyan/SlowFast/outputs/x3d_l_kabr_clips/checkpoints/`

### Gemma 3 4B LoRA

Gemma support is implemented and validated:

- generic HF multimodal LoRA backend works for Gemma
- smoke train succeeded
- smoke adapter-backed eval succeeded

Current state:

- full `Gemma 3 4B` LoRA behavior run has **not started yet**
- it is queued behind completion of `X3D-L`

Relevant files:

- config:
  - `/home/jhyan/myproj/configs/gemma3_4b_behavior_lora_full.yaml`
- smoke config:
  - `/home/jhyan/myproj/configs/gemma3_4b_behavior_lora_smoke.yaml`

## Immediate Plan

1. Finish `X3D-L` training with the more stable batch-4 resumed setup.
2. Run `X3D-L` full clip-level eval separately after training completes.
3. Score `X3D-L` predictions into the same metrics format used by the VLM results.
4. Launch full `Gemma 3 4B` LoRA behavior fine-tuning.
5. Run full adapter-backed `Gemma 3 4B` behavior eval.
6. Add both new rows to the behavior-only comparison table.

## Risks And Notes

- `X3D-L` is the current blocker.
- The instability so far appears operational rather than architectural.
- The `SlowFast` changes for `X3D-L` live outside this repo in `/home/jhyan/SlowFast`.
- The `myproj` repo now contains the Gemma LoRA support and the orchestration scripts, but not the external SlowFast repo history.

## Key Files

- current behavior summary:
  - [CURRENT_BEHAVIOR_SUMMARY.md](/home/jhyan/myproj/CURRENT_BEHAVIOR_SUMMARY.md)
- behavior paper manifest:
  - [paper_runs_behavior.yaml](/home/jhyan/myproj/configs/paper_runs_behavior.yaml)
- Gemma full config:
  - [gemma3_4b_behavior_lora_full.yaml](/home/jhyan/myproj/configs/gemma3_4b_behavior_lora_full.yaml)
- X3D/Gemma baseline runner:
  - [run_x3d_l_gemma_baselines.sh](/home/jhyan/myproj/scripts/run_x3d_l_gemma_baselines.sh)
