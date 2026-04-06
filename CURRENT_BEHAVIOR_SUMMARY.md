# Current KABR Behavior Setup and Results

This file summarizes the current clip-level KABR behavior setup used in this repo and the current behavior-only results as of `2026-04-05`.

## Dataset Construction

### Source data

- Raw annotations come from:
  - `/home/jhyan/KABR_files/annotation/train.csv`
  - `/home/jhyan/KABR_files/annotation/val.csv`
- Raw images come from:
  - `/home/jhyan/KABR_files/dataset/image`
- Class names come from:
  - `/home/jhyan/KABR_files/annotation/classes.json`

### What stays unchanged

- No relabeling was done.
- No train/val membership was changed.
- The original per-frame labels are preserved.

### Derived clip definition

The current clip-level dataset is derived from the released frame annotations.

One derived clip is built by grouping frames that satisfy all of the following:

- same `original_video_id`
- same behavior `label_id`
- consecutive `frame_id`

A new clip starts when any of these changes:

- `original_video_id`
- behavior label
- frame continuity

This logic is implemented in:

- [kabr_frames.py](/home/jhyan/myproj/src/kabr_vlm/data/kabr_frames.py)

### Clip counts

Current derived clip counts:

- `train`: `11003` clips
- `val`: `3761` clips

The current paper-style behavior experiments all evaluate on the full derived `val` clip set.

### Labels

Each derived clip has exactly one behavior label.

Behavior classes:

- `Walk`
- `Graze`
- `Browse`
- `Head Up`
- `Auto-Groom`
- `Trot`
- `Run`
- `Occluded`

Validation clip supports from the current full-eval metrics:

- `Walk`: `1264`
- `Graze`: `1204`
- `Browse`: `108`
- `Head Up`: `968`
- `Auto-Groom`: `35`
- `Trot`: `50`
- `Run`: `51`
- `Occluded`: `81`

### Species handling

Species are inferred from `original_video_id` prefixes:

- `G...` -> `giraffe`
- `ZP...` -> `plains zebra`
- `ZG...` -> `grevys zebra`

For the current paper direction, `behavior` is the main task. Species is not part of the main story here.

## Model Inputs

### VLM inputs

The VLMs do not consume raw video. They consume:

- one derived KABR clip
- converted to a small ordered set of sampled RGB frames
- plus a text prompt

Default behavior setup:

- `8` uniformly sampled ordered frames per clip
- prompt asks for exactly one behavior label

Uniform clip sampling is implemented in:

- [kabr_frames.py](/home/jhyan/myproj/src/kabr_vlm/data/kabr_frames.py)

There is also a `4`-frame LoRA ablation using the same clip definition and same behavior labels.

### X3D inputs

X3D uses the same derived clip IDs, but samples fixed temporal tensors inside each clip according to model config. It does not consume every frame in the clip.

## Current Reliable Behavior Results

All numbers below are on the same derived clip-level KABR `val` set (`3761` clips).

### Zero-shot VLMs

| Model | Accuracy | Macro F1 | Parsed |
| --- | ---: | ---: | ---: |
| `Qwen2.5-VL-3B-Instruct` | `0.5057` | `0.2596` | `3761 / 3761` |
| `Qwen2.5-VL-7B-Instruct` | `0.6107` | `0.3168` | `3761 / 3761` |
| `LLaVA OneVision Qwen2 7B` | `0.5727` | `0.2687` | `3761 / 3761` |
| `Gemma 3 4B` | `0.5504` | `0.2273` | `3755 / 3761` |
| `Idefics3 8B Llama3` | `0.5294` | `0.2314` | `3761 / 3761` |

Artifact examples:

- [qwen25vl7b metrics](/home/jhyan/myproj/artifacts/full_eval/qwen25vl7b_behavior_clip_val_full.metrics.json)
- [gemma3 4b metrics](/home/jhyan/myproj/artifacts/full_eval/gemma3_4b_behavior_clip_val_full.metrics.json)
- [idefics3 8b metrics](/home/jhyan/myproj/artifacts/full_eval/idefics3_8b_behavior_clip_val_full.metrics.json)

### X3D baselines

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| `X3D-XS` | `0.6703` | `0.3852` |
| `X3D-S` | `0.6307` | `0.3813` |
| `X3D-M` | `0.6668` | `0.4263` |

Artifacts:

- [x3d_xs metrics](/home/jhyan/myproj/artifacts/x3d_metrics/x3d_xs_behavior_clip.metrics.json)
- [x3d_s metrics](/home/jhyan/myproj/artifacts/x3d_metrics/x3d_s_behavior_clip.metrics.json)
- [x3d_m metrics](/home/jhyan/myproj/artifacts/x3d_metrics/x3d_m_behavior_clip.metrics.json)

### Post-training / fine-tuning results

| Method | Backbone | Accuracy | Macro F1 | Parsed |
| --- | --- | ---: | ---: | ---: |
| frozen linear | `Qwen2.5-VL-3B-Instruct` | `0.6820` | `0.3542` | `3761 / 3761` |
| frozen MLP | `Qwen2.5-VL-3B-Instruct` | `0.6833` | `0.2954` | `3761 / 3761` |
| distill MLP | `Qwen2.5-VL-3B-Instruct` | `0.6517` | `0.2742` | `3761 / 3761` |
| distill MLP, alpha=`0.7` | `Qwen2.5-VL-3B-Instruct` | `0.6200` | `0.2134` | `3761 / 3761` |
| LoRA plain, `8` frames | `Qwen2.5-VL-3B-Instruct` | `0.8535` | `0.4846` | `3761 / 3761` |
| LoRA plain, `4` frames | `Qwen2.5-VL-3B-Instruct` | `0.8487` | `0.4985` | `3761 / 3761` |

Artifacts:

- [frozen linear metrics](/home/jhyan/myproj/artifacts/paper/behavior/frozen/qwen25vl3b/linear/predictions.metrics.json)
- [frozen mlp metrics](/home/jhyan/myproj/artifacts/paper/behavior/frozen/qwen25vl3b/mlp/predictions.metrics.json)
- [distill metrics](/home/jhyan/myproj/artifacts/paper/behavior/distill/qwen25vl3b/mlp/predictions.metrics.json)
- [distill alpha70 metrics](/home/jhyan/myproj/artifacts/paper/behavior/distill/qwen25vl3b/mlp_alpha70/predictions.metrics.json)
- [LoRA 8-frame metrics](/home/jhyan/myproj/artifacts/runs/qwen25vl3b_behavior_lora_full/behavior_val_predictions.metrics.json)
- [LoRA 4-frame metrics](/home/jhyan/myproj/artifacts/paper/behavior/lora/qwen25vl3b/plain_4frames/behavior_val_predictions.metrics.json)

## Current Main Takeaways

### 1. Zero-shot VLMs are below X3D on this task

The strongest zero-shot result so far is:

- `Qwen2.5-VL-7B-Instruct`: `0.6107` accuracy, `0.3168` macro F1

That is below the best X3D clip-level result:

- `X3D-M`: `0.6668` accuracy, `0.4263` macro F1

### 2. Frozen VLM features help, but only partway

Frozen-feature training improves on zero-shot, especially in accuracy:

- frozen linear: `0.6820` / `0.3542`
- frozen MLP: `0.6833` / `0.2954`

This recovers useful signal, but still does not match the strongest LoRA result.

### 3. LoRA is currently the strongest method

Best current result:

- LoRA plain, `8` frames: `0.8535` accuracy, `0.4846` macro F1

This is currently above all X3D and zero-shot VLM runs on the matched clip-level setup.

### 4. The 4-frame LoRA ablation is competitive

- LoRA plain, `4` frames: `0.8487` accuracy, `0.4985` macro F1

This suggests the current VLM behavior setup may not need all `8` frames to stay strong.

### 5. Distillation is currently weaker than LoRA

Current distillation variants:

- baseline distill: `0.6517` / `0.2742`
- alpha `0.7`: `0.6200` / `0.2134`

Under the current implementation, distillation does not preserve the LoRA gains.

## Results That Are Not Yet Stable Enough for Main Reporting

- `behavior` structured-output LoRA currently has an artifact, but the saved metrics are not a meaningful full result and should not be treated as a stable benchmark row yet.
- `InternVL2_5-4B/8B` are still blocked by upstream loader compatibility in the current environment and are not part of the reliable current results.

## Suggested Paper Story

The current evidence supports this behavior-only claim:

> On clip-level KABR behavior recognition, zero-shot VLMs underperform specialized video models, frozen VLM features recover part of the gap, and parameter-efficient supervised adaptation via LoRA closes and surpasses that gap on the matched clip-level setup.

