# Future Status Snapshot

*Created: 2026-05-15.*

This file is a quick handoff snapshot for future sessions. It summarizes the
current state of the wildlife behavior VLM project, the active paper direction,
the proposed NAS extension, and new small-VLM candidates worth considering.

For fuller context:

- [`HANDOFF.md`](HANDOFF.md): operational handoff and task DB notes.
- [`PAPER_DRAFT_SUMMARY.md`](PAPER_DRAFT_SUMMARY.md): paper-ready narrative and
  results.
- [`CURRENT_RESULTS_SUMMARY.md`](CURRENT_RESULTS_SUMMARY.md): plain-language
  summary of current results.
- [`NAS_BRAINSTORM.md`](NAS_BRAINSTORM.md): expanded NAS idea bank.
- [`NAS_IMPLEMENTATION_EXPERIMENT_PLAN.md`](NAS_IMPLEMENTATION_EXPERIMENT_PLAN.md):
  concrete NAS implementation and experiment plan.

## 1. Project in one paragraph

This project studies wildlife behavior recognition from short drone-video clips
using vision-language models (VLMs). The main result so far is that lightweight
LoRA-adapted VLMs outperform specialist video models such as X3D-L and I3D-L on
two clip-level wildlife behavior datasets: KABR-Clip and Baboonland-Clip. The
project also identifies **class collapse** as a key zero-shot VLM failure mode,
shows that inverse-square-root class-weighted training improves macro-F1 under
severe imbalance, and shows that joint KABR + Baboonland LoRA training can
match or exceed separate in-domain adapters.

## 2. Current experimental status

| Phase | Status | Notes |
|---|---|---|
| Phase 1: baseline comparison | Complete | Zero-shot VLMs, LoRA VLMs, transfer, X3D/I3D baselines |
| Phase 2: class-collapse-aware training | Complete | B2 inv-sqrt class-weighted CE is best |
| Phase 3: joint cross-dataset LoRA | Partly complete | A1 evaluated; A3 trained with evals pending as of earlier handoff |
| Phase 4: scale check / SFT ablation | Not started | Optional reviewer-defense ablation |
| Phase 5: figures | Partly complete | F1/F2/F4 done; F3/F5 not started |
| NAS extension | Planned | Implementation/experiment plan written, not yet executed |

## 3. Datasets

| Dataset | Domain | Classes | Train clips | Val clips | Imbalance |
|---|---|---:|---:|---:|---|
| KABR-Clip | Drone footage of Kenyan wildlife | 8 | 11,003 | 3,761 | Moderate |
| Baboonland-Clip | Drone footage of baboons | 13 | 20,130 | 6,198 | Severe |

Key processed files:

```text
data/processed/kabr_behavior_train.jsonl
data/processed/kabr_behavior_val_cleaned.jsonl
data/processed/baboonland_behavior_train.jsonl
data/processed/baboonland_behavior_val.jsonl
data/processed/joint_kabr_bbl_behavior_train.jsonl
```

## 4. Current headline results

### KABR-Clip

Best adapted VLMs reach about **0.86 accuracy**.

Strong rows:

| Model / setup | Acc / macro-F1 |
|---|---:|
| Gemma3-4B LoRA | 0.864 / 0.515 |
| LLaVA-OV-7B LoRA | 0.863 / 0.523 |
| Qwen2.5-VL-3B joint A1 | 0.863 / 0.514 |
| Qwen2.5-VL-3B in-domain LoRA | 0.853 / 0.485 |
| X3D-L specialist baseline | 0.797 / 0.520 |
| I3D-L specialist baseline | 0.814 / unavailable |

### Baboonland-Clip

Every adapted VLM currently beats X3D-L and I3D-L on accuracy.

Strong rows:

| Model / setup | Acc / macro-F1 |
|---|---:|
| LLaVA-OV-7B LoRA | 0.775 / 0.443 |
| LLaVA-OV-0.5B LoRA | 0.768 / 0.430 |
| Gemma3-4B LoRA | 0.756 / 0.405 |
| Qwen2.5-VL-7B LoRA | 0.755 / 0.394 |
| Qwen2.5-VL-3B B2 inv-sqrt | 0.752 / 0.374 |
| Qwen2.5-VL-3B joint A1 | 0.759 / 0.377 |
| X3D-L specialist baseline | 0.671 / unavailable |
| I3D-L specialist baseline | 0.630 / unavailable |

## 5. Current method contributions

### Contribution A: Joint cross-dataset LoRA

Train one Qwen2.5-VL-3B LoRA adapter on the union of KABR and Baboonland
training data, preserving each example's native prompt and label space.

Current A1 result:

| Setup | KABR Acc / F1 | Baboonland Acc / F1 |
|---|---:|---:|
| Joint A1, no class weighting | 0.863 / 0.514 | 0.759 / 0.377 |
| Joint A3, inv-sqrt class weighting | Eval pending | Eval pending |

Interpretation:

- Joint A1 matches or exceeds in-domain LoRA on both validation sets.
- Joint training is much better than single-dataset transfer.

### Contribution B: Class-collapse-aware training

Class collapse is when a model overpredicts a few common behavior labels and
ignores rare behaviors. This is especially severe on Baboonland.

Qwen2.5-VL-3B Baboonland results:

| Variant | Acc | Macro-F1 |
|---|---:|---:|
| B1 baseline LoRA | 0.734 | 0.327 |
| B2 inv-sqrt class-weighted CE | 0.752 | 0.374 |
| B3 focal loss | 0.715 | 0.302 |
| B4 class-balanced sampler | 0.616 | 0.349 |

Interpretation:

- Inv-sqrt class-weighted CE is the best current mitigation.
- It improves macro-F1 by +0.047 and accuracy by +0.018 over baseline B1.

### Proposed Contribution C: Resource-aware adapter architecture search

The proposed NAS-style extension is not full VLM architecture search. The
recommended version is to search the lightweight adaptation and temporal input
choices around an existing VLM.

Planned search:

| Axis | Values |
|---|---|
| Dataset | Baboonland first |
| Backbone | Qwen2.5-VL-3B |
| Frames per clip | 4, 8 |
| LoRA rank | 8, 16 |
| LoRA target modules | attention only, attention + MLP |
| Loss | CE, inv-sqrt CE |

This gives 16 primary runs.

Primary paper claim if successful:

> A compact search over temporal input length, LoRA adapter capacity/placement,
> and class-aware loss identifies Pareto-efficient VLM adaptation recipes that
> improve minority-class behavior recognition without increasing inference
> cost.

Implementation plan:

- See [`NAS_IMPLEMENTATION_EXPERIMENT_PLAN.md`](NAS_IMPLEMENTATION_EXPERIMENT_PLAN.md).

Important implementation note:

- `training.lora_target_modules` does not currently exist.
- Target modules are currently hard-coded in
  `src/kabr_vlm/training/qwen_tasks.py`.
- Add a config field and target-module presets before launching the NAS grid.

## 6. New small-VLM candidates discussed on 2026-05-15

These models were raised as possible new resource-aware / Pareto-frontier
candidates. They should be verified in the local environment before committing
to experiments.

| Model | Why it may matter |
|---|---|
| MiniCPM-V 4.6 | Efficient multimodal VLM; potentially relevant small/strong candidate |
| Qwen3.5-0.8B | Very small Qwen-family multimodal model; attractive Pareto-frontier point |
| Gemma4-E2B-it | Instruction-tuned effective-2B Gemma multimodal model; likely strong but check access/runtime |
| LFM2.5-VL-1.6B | Liquid AI small VLM; possible middle point between 0.5B and 3B models |

Recommended use:

- Treat these as **new small/edge VLM candidates** for the resource-aware story.
- Do not add all of them immediately to the main grid.
- First test zero-shot/eval loader compatibility.
- Then pick 1-2 candidates for LoRA or inference-only Pareto comparisons.

Most promising order:

1. **Qwen3.5-0.8B**: tiny Qwen-family model; useful if loader compatibility is
   close to existing Qwen code.
2. **LFM2.5-VL-1.6B**: potentially strong small VLM; good Pareto point if
   Transformers support is clean.
3. **MiniCPM-V 4.6**: promising but may require model-specific prompt/processor
   support.
4. **Gemma4-E2B-it**: useful if accessible and stable, but check license,
   model class, and runtime friction.

## 7. Immediate next steps

Recommended order:

1. Check whether A3 evals have completed in the task DB.
2. Update `PAPER_DRAFT_SUMMARY.md` Phase 3 table if A3 metrics exist.
3. Implement `training.lora_target_modules`.
4. Generate and smoke-test two NAS configs:
   - 8 frames, rank 8, attention-only, CE.
   - 8 frames, rank 8, attention-only, inv-sqrt CE.
5. If smoke passes, generate the full 16-run Baboonland NAS grid.
6. Queue NAS training with `dataset=config` so configs control frame-specific
   data paths.
7. Queue evals with `vlm_cross_eval`.
8. Aggregate NAS metrics into CSV/MD.
9. Plot Pareto frontier and prediction-distribution/class-collapse diagnostics.
10. Optionally test top 2 NAS recipes on joint KABR + Baboonland training.
11. Separately, run compatibility checks for Qwen3.5-0.8B and LFM2.5-VL-1.6B
    before expanding the model set.

## 8. Task DB notes

Task DB:

```text
/uufs/chpc.utah.edu/common/home/u1320844/task_dbs/vlm_behavior.db
```

Manager config:

```text
chpc_manager/config.yaml
```

Useful commands:

```bash
cd /scratch/general/vast/j.yan/myproj1/chpc_manager
PY=/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python

$PY cli.py --config config.yaml status
$PY cli.py --config config.yaml list
```

For NAS runs, use:

```text
task_type: vlm_lora_train
dataset: config
```

Reason:

- `dataset=baboonland` calls `scripts/run_vlm_baboonland_full.sh`, which
  overrides train/val paths to the default Baboonland JSONLs.
- `dataset=config` calls `scripts/run_vlm_config_train.sh`, which trusts the
  config paths. This is necessary for frame-count experiments.

## 9. Files most likely to change next

Code:

```text
src/kabr_vlm/config.py
src/kabr_vlm/training/qwen_tasks.py
scripts/analysis/generate_nas_configs.py
scripts/analysis/summarize_nas_results.py
```

Configs:

```text
configs/nas/baboonland_qwen25vl3b/
```

Artifacts:

```text
artifacts/runs/nas/baboonland/qwen25vl3b/
artifacts/nas_eval/baboonland/
```

Paper/status docs:

```text
PAPER_DRAFT_SUMMARY.md
CURRENT_RESULTS_SUMMARY.md
NAS_BRAINSTORM.md
NAS_IMPLEMENTATION_EXPERIMENT_PLAN.md
```

## 10. One-line future reminder

The strongest next move is to finish A3 evaluation, then run a small NAS-style
search over **frames, LoRA rank, target modules, and class-aware loss** on
Baboonland Qwen2.5-VL-3B, using the result to add a resource-aware
adapter-search contribution to the paper.
