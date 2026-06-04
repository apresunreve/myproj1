# NAS Implementation and Experiment Plan

*Created: 2026-05-13.*

This plan turns the NAS-style idea into a concrete, runnable experiment for the
wildlife behavior VLM project.

The recommended direction is:

> **Resource-aware adapter and temporal architecture search for wildlife VLM
> adaptation.**

The search should focus on the lightweight design choices around existing VLMs,
not on inventing a new VLM architecture. This keeps the experiment feasible and
directly connected to the current paper.

## 1. Paper goal

The goal is to add a compact third contribution:

> **Contribution C: Resource-aware adapter architecture search.**

Proposed claim:

> A compact search over temporal input length, LoRA adapter capacity/placement,
> and class-aware loss identifies Pareto-efficient VLM adaptation recipes that
> improve minority-class behavior recognition without increasing inference
> cost.

This supports the existing paper story:

1. Adapted VLMs beat specialist video models.
2. Zero-shot VLMs suffer from class collapse.
3. LoRA fine-tuning fixes much of the gap.
4. Inv-sqrt class weighting mitigates class collapse.
5. Joint LoRA closes the cross-dataset transfer gap.
6. A searched adapter/input recipe gives a principled accuracy/F1/cost
   tradeoff.

## 2. What to search

Run the first search on **Baboonland** with **Qwen2.5-VL-3B**.

Baboonland is the best search dataset because it has severe class imbalance,
which makes macro-F1 and class collapse visible. Qwen2.5-VL-3B is the best
starting backbone because the repo already has B1, B2, A1, and A3
infrastructure around it.

### Primary search space

| Axis | Values | Why it matters |
|---|---|---|
| Frames per clip | 4, 8 | Tests temporal evidence vs cost |
| LoRA rank | 8, 16 | Tests adapter capacity vs trainable parameters |
| LoRA target modules | attention only, attention + MLP | Tests where behavior adaptation should happen |
| Loss | CE, inv-sqrt CE | Tests class-collapse mitigation |

This is:

```text
2 frames x 2 ranks x 2 target-module presets x 2 losses = 16 runs
```

### Target-module presets

Current default target modules are:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

Use two presets:

| Preset | Modules |
|---|---|
| attention_only | q_proj, k_proj, v_proj, o_proj |
| attention_mlp | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

The `attention_mlp` preset is the current default.

## 3. Experimental stages

## Stage 0: Preflight checks

Before launching new search jobs:

1. Check task DB health.
2. Check whether A3 evals have completed.
3. Confirm the current Qwen2.5-VL-3B Baboonland B1/B2 results are still the
   reference values.
4. Confirm there are processed 4-frame Baboonland JSONLs, or generate them.

Known current processed files:

```text
data/processed/baboonland_behavior_train.jsonl
data/processed/baboonland_behavior_val.jsonl
```

Need for the 4-frame axis:

```text
data/processed/baboonland_behavior_train_4frames.jsonl
data/processed/baboonland_behavior_val_4frames.jsonl
```

If 4-frame Baboonland JSONLs do not exist, create them using the existing
Baboonland preparation script pattern. Keep the same clips and labels; only
change the maximum sampled frames per clip.

## Stage 1: Implementation changes

The existing config already supports:

- `data.max_frames_per_clip`
- `model.max_frames`
- `training.lora_r`
- `training.lora_alpha`
- `training.lora_dropout`
- `training.class_weighting`
- `training.class_sampler`

One code change is needed for target-module search:

### 1.1 Add `training.lora_target_modules`

File:

```text
src/kabr_vlm/config.py
```

Add to `TrainingConfig`:

```python
lora_target_modules: str = "default"
```

Allowed values:

```text
default
attention_only
attention_mlp
```

### 1.2 Resolve target-module presets during model creation

File:

```text
src/kabr_vlm/training/qwen_tasks.py
```

Add constants:

```python
ATTENTION_ONLY_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

ATTENTION_MLP_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
```

Then update `create_training_model` so the target-module selection uses:

1. InternVL-specific modules if InternVL.
2. Qwen MLA modules if `attention_experiment: mla`.
3. Otherwise, `training.lora_target_modules`.

Suggested behavior:

| Config value | Result |
|---|---|
| default | current behavior |
| attention_only | attention-only preset |
| attention_mlp | current default attention+MLP preset |

Raise a clear `ValueError` for unknown values.

### 1.3 Add a config generator script

Add:

```text
scripts/analysis/generate_nas_configs.py
```

Responsibilities:

- Read a base YAML config.
- Generate the 16 Baboonland NAS configs.
- Write them to:

```text
configs/nas/baboonland_qwen25vl3b/
```

Each generated config should have:

- Unique `project_name`.
- Unique `training.output_dir`.
- Correct `data.train_file` and `data.val_file`.
- Correct `data.max_frames_per_clip`.
- Correct `model.max_frames`.
- Correct `training.lora_r`.
- Correct `training.lora_alpha`.
- Correct `training.class_weighting`.
- Correct `training.lora_target_modules`.

Use a stable naming scheme:

```text
qwen25vl3b_bbl_nas_f{frames}_r{rank}_{targets}_{loss}.yaml
```

Example:

```text
qwen25vl3b_bbl_nas_f8_r16_attention_mlp_invsqrt.yaml
```

Suggested output directory pattern:

```text
artifacts/runs/nas/baboonland/qwen25vl3b/f{frames}_r{rank}_{targets}_{loss}/checkpoint
```

## Stage 2: Training search jobs

Launch 16 training runs through the existing task DB.

Task type:

```text
vlm_lora_train
```

Dataset:

```text
baboonland
```

Model:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

Important note:

`scripts/run_vlm_baboonland_full.sh` currently overrides the config train/val
paths to the default 8-frame Baboonland JSONLs. For the 4-frame runs, use:

```text
--dataset config
```

or:

```text
--dataset joint
```

through `scripts/run_vlm_lora_train_task.sh`, because that path calls
`scripts/run_vlm_config_train.sh` and trusts the config's data paths.

Use `dataset=config` for all NAS runs to keep behavior consistent.

Example task-add command:

```bash
cd /scratch/general/vast/j.yan/myproj1/chpc_manager
PY=/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python

$PY cli.py --config config.yaml add vlm_lora_train \
  '{"dataset":"config",
    "config_path":"/scratch/general/vast/j.yan/myproj1/configs/nas/baboonland_qwen25vl3b/qwen25vl3b_bbl_nas_f8_r16_attention_mlp_invsqrt.yaml",
    "job_name":"qwen25vl3b_bbl_nas_f8_r16_attention_mlp_invsqrt",
    "model_id":"Qwen/Qwen2.5-VL-3B-Instruct",
    "output_dir":"/scratch/general/vast/j.yan/myproj1/artifacts/runs/nas/baboonland/qwen25vl3b/f8_r16_attention_mlp_invsqrt/checkpoint"}'
```

## Stage 3: Evaluation jobs

After each training run finishes, queue Baboonland validation evals.

Task type:

```text
vlm_cross_eval
```

Target dataset:

```text
baboonland
```

Validation file:

```text
data/processed/baboonland_behavior_val.jsonl
```

For fairness, evaluate every trained adapter on the same 8-frame validation
JSONL first. If a 4-frame-trained model cannot fairly consume 8-frame eval
inputs, also evaluate on the matching 4-frame validation JSONL and report the
matched-frame result as primary.

Recommended evaluation policy:

| Training frames | Primary eval | Secondary eval |
|---:|---|---|
| 4 | 4-frame Baboonland val | 8-frame Baboonland val if compatible |
| 8 | 8-frame Baboonland val | none |

Example eval task:

```bash
cd /scratch/general/vast/j.yan/myproj1/chpc_manager
PY=/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python

$PY cli.py --config config.yaml add vlm_cross_eval \
  '{"val_jsonl":"/scratch/general/vast/j.yan/myproj1/data/processed/baboonland_behavior_val.jsonl",
    "target_dataset":"baboonland",
    "model_id":"Qwen/Qwen2.5-VL-3B-Instruct",
    "adapter_path":"/scratch/general/vast/j.yan/myproj1/artifacts/runs/nas/baboonland/qwen25vl3b/f8_r16_attention_mlp_invsqrt/checkpoint",
    "output_jsonl":"/scratch/general/vast/j.yan/myproj1/artifacts/nas_eval/baboonland/qwen25vl3b_f8_r16_attention_mlp_invsqrt_predictions.jsonl"}'
```

The metrics file will be:

```text
<output_jsonl>.metrics.json
```

This concat-style metrics path is important; do not suffix-replace it.

## Stage 4: Result aggregation

Add:

```text
scripts/analysis/summarize_nas_results.py
```

Inputs:

- Generated config directory.
- Eval metrics directory.
- Optional train logs or training metrics.

Output:

```text
artifacts/nas_eval/baboonland/qwen25vl3b_nas_summary.csv
artifacts/nas_eval/baboonland/qwen25vl3b_nas_summary.md
```

Summary columns:

| Column | Meaning |
|---|---|
| run_name | Stable NAS run name |
| frames | Frames per clip |
| lora_r | LoRA rank |
| target_modules | attention_only or attention_mlp |
| loss | CE or inv_sqrt |
| accuracy | Validation accuracy |
| macro_f1 | Validation macro-F1 |
| pred_entropy | Entropy of prediction distribution |
| num_pred_classes | Number of classes predicted at least once |
| majority_pred_rate | Fraction of predictions assigned to most common predicted class |
| trainable_params | Number of trainable LoRA parameters |
| latency_ms_per_clip | Optional latency benchmark |

### Metrics to compute

Primary:

- Accuracy.
- Macro-F1.

Class-collapse diagnostics:

- Prediction distribution entropy.
- Number of predicted classes used.
- Majority-class prediction rate.
- Per-class F1.

Resource metrics:

- Trainable parameter count.
- Inference latency per clip.
- GPU memory if easy to collect.

## Stage 5: Pareto analysis and figures

Add plots to `scripts/analysis/build_figures.py` or a separate NAS plotting
script.

Recommended figures:

### Figure C1: Macro-F1 vs accuracy

Each point is a NAS configuration.

Color:

- CE vs inv-sqrt.

Shape:

- attention_only vs attention_mlp.

Label:

- frame count and rank.

This shows whether there is an accuracy/F1 tradeoff.

### Figure C2: Macro-F1 vs trainable parameters

This is the resource-aware NAS figure.

Expected message:

> Some smaller adapters are close to larger adapters, and inv-sqrt loss can move
> a point upward without adding inference cost.

### Figure C3: Prediction distribution heatmap

Rows:

- NAS configurations.

Columns:

- Baboonland behavior classes.

Cells:

- Fraction of predictions assigned to each class.

This directly visualizes class collapse and whether the searched recipes use
more of the label space.

## Stage 6: Generalization check

After the 16-run Baboonland search, take the top 2-3 recipes and test whether
they generalize.

Select recipes by:

1. Best macro-F1.
2. Best accuracy.
3. Best Pareto point with low trainable parameters or lower frame count.

Run these recipes on one of:

### Option A: KABR in-domain

Train Qwen2.5-VL-3B on KABR with the selected recipes.

Question:

> Does the searched Baboonland recipe also work on the easier KABR dataset?

### Option B: Joint KABR + Baboonland

Train Qwen2.5-VL-3B jointly using selected recipes.

Question:

> Does the searched recipe improve the already strong joint A1/A3 setup?

Recommended choice:

Run **joint KABR + Baboonland** for the top 2 recipes, because it connects
directly to Contribution A and has higher paper value.

## 4. Concrete 16-run grid

| Run | Frames | Rank | Target modules | Loss |
|---:|---:|---:|---|---|
| 1 | 4 | 8 | attention_only | CE |
| 2 | 4 | 8 | attention_only | inv_sqrt |
| 3 | 4 | 8 | attention_mlp | CE |
| 4 | 4 | 8 | attention_mlp | inv_sqrt |
| 5 | 4 | 16 | attention_only | CE |
| 6 | 4 | 16 | attention_only | inv_sqrt |
| 7 | 4 | 16 | attention_mlp | CE |
| 8 | 4 | 16 | attention_mlp | inv_sqrt |
| 9 | 8 | 8 | attention_only | CE |
| 10 | 8 | 8 | attention_only | inv_sqrt |
| 11 | 8 | 8 | attention_mlp | CE |
| 12 | 8 | 8 | attention_mlp | inv_sqrt |
| 13 | 8 | 16 | attention_only | CE |
| 14 | 8 | 16 | attention_only | inv_sqrt |
| 15 | 8 | 16 | attention_mlp | CE |
| 16 | 8 | 16 | attention_mlp | inv_sqrt |

Run 15 is the closest to the current B1 baseline.

Run 16 is the closest to the current B2 inv-sqrt result.

## 5. Success criteria

The experiment is worth including in the paper if at least one of these happens:

1. A lower-cost adapter, such as rank 8 or attention-only, matches the rank-16
   attention+MLP baseline within about 1 accuracy point or 1 macro-F1 point.
2. Inv-sqrt loss consistently improves macro-F1 across multiple adapter
   topologies.
3. 4-frame models are much cheaper and remain competitive.
4. 8-frame models clearly dominate 4-frame models, giving evidence that
   temporal input length matters.
5. The Pareto frontier includes a non-obvious recipe, such as rank-8
   attention+MLP with inv-sqrt loss.
6. The selected recipe improves or matches joint A1/A3 when tested in the
   generalization stage.

## 6. How to report outcomes

### If the search finds a better model

Paper message:

> The searched recipe improves macro-F1 and/or accuracy over the hand-picked
> LoRA baseline while preserving inference cost.

### If the search finds a cheaper equivalent model

Paper message:

> The search identifies a lower-cost adapter with comparable performance,
> improving deployability without sacrificing behavior recognition quality.

### If the search validates the existing B2 recipe

Paper message:

> The search shows that the current B2 recipe lies on the Pareto frontier,
> providing systematic support for the selected class-aware LoRA design.

### If the search is inconclusive

Use it as an appendix or future-work result:

> We explored a compact adapter search space and found that performance is
> relatively insensitive to adapter topology compared with loss weighting and
> frame count.

## 7. Implementation checklist

### Code

- [ ] Add `training.lora_target_modules` to `src/kabr_vlm/config.py`.
- [ ] Add target-module presets to `src/kabr_vlm/training/qwen_tasks.py`.
- [ ] Add validation for unknown target-module presets.
- [ ] Add `scripts/analysis/generate_nas_configs.py`.
- [ ] Add `scripts/analysis/summarize_nas_results.py`.
- [ ] Optionally add NAS figures to `scripts/analysis/build_figures.py`.

### Data

- [ ] Confirm existing 8-frame Baboonland JSONLs.
- [ ] Create or confirm 4-frame Baboonland JSONLs.
- [ ] Confirm eval JSONL paths for 4-frame and 8-frame validation.

### Training

- [ ] Generate 16 configs under `configs/nas/baboonland_qwen25vl3b/`.
- [ ] Queue 16 `vlm_lora_train` tasks with `dataset=config`.
- [ ] Monitor task DB for failed or stuck runs.

### Evaluation

- [ ] Queue evals for each completed adapter.
- [ ] Verify metrics JSON files exist.
- [ ] Aggregate results into CSV/MD.
- [ ] Compute class-collapse diagnostics.

### Paper artifacts

- [ ] Add NAS results table.
- [ ] Add Pareto figure.
- [ ] Add class-collapse prediction distribution figure or heatmap.
- [ ] Update `PAPER_DRAFT_SUMMARY.md` with Contribution C.

## 8. Recommended order of execution

1. Implement `lora_target_modules`.
2. Generate a tiny 2-run smoke grid:
   - 8 frames, rank 8, attention_only, CE.
   - 8 frames, rank 8, attention_only, inv_sqrt.
3. Train and evaluate the smoke grid.
4. If smoke passes, generate the full 16-run grid.
5. Queue full NAS training.
6. Queue evals as runs complete.
7. Aggregate results.
8. Run top 2 recipes on joint KABR+Baboonland.
9. Add Contribution C text and figures.

## 9. Minimal paper wording

Method:

> We define a compact NAS-style search space over temporal input length, LoRA
> rank, adapter placement, and class-imbalance correction. Each candidate is a
> parameter-efficient VLM adaptation recipe trained on Baboonland and evaluated
> under a multi-objective criterion: accuracy, macro-F1, prediction-distribution
> collapse, trainable parameters, and inference latency.

Result:

> The search identifies Pareto-efficient recipes that improve minority-class
> behavior recognition without increasing inference cost. In particular,
> class-aware loss moves configurations upward in macro-F1, while rank and
> target-module choices expose the cost/performance tradeoff of VLM adaptation.

Contribution:

> We introduce resource-aware adapter architecture search for wildlife VLM
> adaptation, showing that the best recipe is not determined by backbone scale
> alone but by the interaction of temporal input, adapter capacity, and
> class-collapse mitigation.
