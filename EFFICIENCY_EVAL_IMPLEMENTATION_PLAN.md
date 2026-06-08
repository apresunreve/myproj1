# Efficiency Evaluation Implementation Plan

## Goal

Build a reliable path from existing trained wildlife VLM adapters to the
`plan2.md` efficiency experiments:

```text
trained adapter
  -> verified evaluation pipeline
  -> label-likelihood confidence
  -> compressed first pass
  -> full-token/frame fallback
  -> accuracy/latency Pareto analysis
```

The first target is a low-risk evaluation scaffold. Visual-token surgery comes
after confidence and fallback are working.

## Current Verified State

### Checkpoints on disk

Usable trained LoRA adapter checkpoints are present for:

- `artifacts/runs/gemma3_4b_behavior_lora_full/checkpoint`
- `artifacts/runs/llava_onevision_qwen2_0_5b_behavior_lora_full/checkpoint`
- `artifacts/runs/llava_onevision_qwen2_7b_behavior_lora_full/checkpoint`
- `artifacts/runs/llava_onevision_1_5_4b_behavior_lora_full/checkpoint`
- `artifacts/runs/qwen25vl7b_behavior_lora_full/checkpoint`
- `artifacts/runs/mobilevlm_v2_1_7b_behavior_lora_full/checkpoint`
- `artifacts/runs/baboonland/gemma3_4b_behavior_lora/checkpoint`
- `artifacts/runs/baboonland/llava_onevision_qwen2_0_5b_behavior_lora/checkpoint`
- `artifacts/runs/baboonland/llava_onevision_qwen2_7b_behavior_lora/checkpoint`
- `artifacts/runs/baboonland/mobilevlm_v2_1_7b_behavior_lora/checkpoint`
- `artifacts/runs/baboonland/qwen25vl3b_behavior_lora_b2_invsqrt/checkpoint`
- `artifacts/runs/baboonland/qwen25vl3b_behavior_lora_b3_focal/checkpoint`
- `artifacts/runs/baboonland/qwen25vl3b_behavior_lora_b4_sampler/checkpoint`
- `artifacts/runs/joint/qwen25vl3b_joint_kabr_bbl_lora_a1/checkpoint`
- `artifacts/runs/joint/qwen25vl3b_joint_kabr_bbl_lora_a3_invsqrt/checkpoint`

Older Qwen3B KABR checkpoints are present under:

- `x3d_ckpts/runs/qwen25vl3b_behavior_lora_full/checkpoint`
- `x3d_ckpts/runs/qwen25vl3b_species_lora_full/checkpoint`

Important path mismatch:

- `configs/qwen25vl3b_behavior_lora_full.yaml` points to
  `artifacts/runs/qwen25vl3b_behavior_lora_full/checkpoint`
- The available Qwen3B KABR behavior checkpoint is actually under
  `x3d_ckpts/runs/qwen25vl3b_behavior_lora_full/checkpoint`

### Evaluation smoke status

The lightweight evaluation pipeline smoke test passes:

- Python syntax checks pass for the core eval/data modules.
- `kabr_vlm.cli`, `kabr_vlm.vlm_baselines`, and
  `kabr_vlm.training.qwen_tasks` import successfully.
- A synthetic one-clip `evaluate_records` run writes prediction JSONL, parses
  the behavior label, records latency, and scores `accuracy=1.0`.
- `scripts/eval_cross_dataset_vlm.py` loads real processed KABR and Baboonland
  validation JSONL rows and computes metrics correctly.

CUDA is available on the current machine:

```text
NVIDIA RTX PRO 4000 Blackwell
```

### MiniCPM LoRA training smoke status

MiniCPM-V 4.6 requires a Transformers build that exposes
`MiniCPMV4_6ForConditionalGeneration`. The global conda environment had
`transformers 5.5.0` but did not expose that class, so a local smoke-test venv
was created:

```text
.venv_minicpm_smoke/
```

That venv was created with system packages and then updated with:

```bash
.venv_minicpm_smoke/bin/python -m pip install \
  git+https://github.com/huggingface/transformers.git
.venv_minicpm_smoke/bin/python -m pip install peft
```

Verified inside the venv:

```text
transformers 5.10.0.dev0
MiniCPMV4_6ForConditionalGeneration available
MiniCPMV4_6Processor available
torch 2.10.0+cu128
cuda available
```

Three MiniCPM-V 4.6 LoRA smoke configs were added:

- `configs/minicpm_v46_behavior_lora_smoke.yaml`
- `configs/minicpm_v46_baboonland_behavior_lora_smoke.yaml`
- `configs/minicpm_v46_joint_kabr_bbl_lora_smoke.yaml`

All three local smoke training runs completed on this node:

| Dataset | Config | Train samples | Runtime | Train loss | Output |
|---|---|---:|---:|---:|---|
| KABR | `configs/minicpm_v46_behavior_lora_smoke.yaml` | 1 | 8.55s | 1.324 | `artifacts/smoke/minicpm_v46_behavior_lora/checkpoint` |
| Baboonland | `configs/minicpm_v46_baboonland_behavior_lora_smoke.yaml` | 1 | 2.60s | 6.602 | `artifacts/smoke/baboonland/minicpm_v46_behavior_lora/checkpoint` |
| Joint KABR+Baboonland | `configs/minicpm_v46_joint_kabr_bbl_lora_smoke.yaml` | 2 | 4.22s | 1.157 | `artifacts/smoke/joint/minicpm_v46_joint_kabr_bbl_lora/checkpoint` |

Each smoke checkpoint wrote:

- `adapter_model.safetensors`
- `adapter_config.json`
- `training_metrics.json`
- Trainer checkpoint directory

MiniCPM LoRA parameter count from the smoke runs:

```text
trainable_params = 9,486,336
total_params     = 1,309,914,352
```

Observed warnings:

- MiniCPM falls back to the torch implementation because optional fast-path
  packages are not installed.
- The node kernel is `4.18.0`, below the kernel version recommended by
  Accelerate/Transformers for avoiding possible hangs.
- `torch_dtype` is deprecated in the current Transformers source build; this is
  only a warning.

### MiniCPM full training status

Full MiniCPM-V 4.6 LoRA jobs were submitted on Granite and completed
successfully:

| Setting | Slurm job | Runtime | Train samples | Val samples | Train loss | Output |
|---|---:|---:|---:|---:|---:|---|
| KABR | `1396317` | 03:37:04 | 11003 | 3981 | 1.5082 | `artifacts/runs/minicpm_v46_behavior_lora_full/checkpoint` |
| Baboonland | `1396318` | 07:02:55 | 20130 | 6198 | 1.2258 | `artifacts/runs/baboonland/minicpm_v46_behavior_lora_full/checkpoint` |
| Joint KABR+Baboonland | `1396319` | 10:29:51 | 31133 | 6198 | 1.3108 | `artifacts/runs/joint/minicpm_v46_joint_kabr_bbl_lora_full/checkpoint` |

Each full checkpoint contains:

- `adapter_model.safetensors`
- `adapter_config.json`
- tokenizer and processor files
- `training_metrics.json`

Full-run configs and launcher:

- `configs/minicpm_v46_behavior_lora_full.yaml`
- `configs/minicpm_v46_baboonland_behavior_lora_full.yaml`
- `configs/minicpm_v46_joint_kabr_bbl_lora_full.yaml`
- `scripts/sbatch_minicpm_v46_lora_train.sbatch`

### Phase 1 adapter eval status

One-sample real adapter smoke evals passed for LLaVA, Qwen, and MiniCPM:

| Model/adapter | Dataset | Rows | Parsed | Accuracy | Output root |
|---|---|---:|---:|---:|---|
| LLaVA-OV-0.5B KABR LoRA | KABR | 1 | 1 | 1.0000 | `artifacts/smoke/efficiency_phase1_llava_0_5b` |
| Qwen2.5-VL-7B KABR LoRA | KABR | 1 | 1 | 1.0000 | `artifacts/smoke/efficiency_phase1_qwen25vl7b` |
| MiniCPM-V 4.6 KABR LoRA | KABR | 1 | 1 | 1.0000 | `artifacts/smoke/efficiency_phase1_minicpm_v46_rerun` |
| MiniCPM-V 4.6 Baboonland LoRA | Baboonland | 1 | 1 | 1.0000 | `artifacts/smoke/efficiency_phase1_minicpm_v46_rerun` |
| MiniCPM-V 4.6 joint LoRA | KABR | 1 | 1 | 1.0000 | `artifacts/smoke/efficiency_phase1_minicpm_v46_rerun` |
| MiniCPM-V 4.6 joint LoRA | Baboonland | 1 | 1 | 1.0000 | `artifacts/smoke/efficiency_phase1_minicpm_v46_rerun` |

MiniCPM eval support required a small loader update:

- `src/kabr_vlm/vlm_baselines.py` now loads MiniCPM-V 4.6 PEFT adapters by
  wrapping the full MiniCPM model with `PeftModel.from_pretrained(...)`.

MiniCPM local eval launchers:

- `scripts/run_minicpm_v46_smoke_eval_local.sh`
- `scripts/run_minicpm_v46_full_eval_local.sh`

### MiniCPM full eval results

Full MiniCPM generation evals completed on this node. All rows parsed.

| Adapter | Eval dataset | Rows | Parsed rate | Accuracy | Macro F1 | Output |
|---|---|---:|---:|---:|---:|---|
| KABR LoRA | KABR | 3981 | 1.0000 | 0.8664 | 0.5162 | `artifacts/full_eval/minicpm_v46/minicpm_v46_kabr_lora_on_kabr_full.jsonl` |
| Baboonland LoRA | Baboonland | 6198 | 1.0000 | 0.7712 | 0.4236 | `artifacts/full_eval/minicpm_v46/minicpm_v46_baboonland_lora_on_baboonland_full.jsonl` |
| Joint LoRA | KABR | 3981 | 1.0000 | 0.8644 | 0.5042 | `artifacts/full_eval/minicpm_v46/minicpm_v46_joint_lora_on_kabr_full.jsonl` |
| Joint LoRA | Baboonland | 6198 | 1.0000 | 0.7643 | 0.3948 | `artifacts/full_eval/minicpm_v46/minicpm_v46_joint_lora_on_baboonland_full.jsonl` |

Metric files:

- `artifacts/full_eval/minicpm_v46/minicpm_v46_kabr_lora_on_kabr_full.jsonl.metrics.json`
- `artifacts/full_eval/minicpm_v46/minicpm_v46_baboonland_lora_on_baboonland_full.jsonl.metrics.json`
- `artifacts/full_eval/minicpm_v46/minicpm_v46_joint_lora_on_kabr_full.jsonl.metrics.json`
- `artifacts/full_eval/minicpm_v46/minicpm_v46_joint_lora_on_baboonland_full.jsonl.metrics.json`

These full eval artifacts are generated data and should not be committed.

## Phase 1: Real Adapter Smoke Tests

Run one-sample real adapter-backed evals to verify that model loading,
processor formatting, adapter loading, generation, parsing, and output writing
work end to end.

Start with the smallest/cheapest model:

```text
llava-hf/llava-onevision-qwen2-0.5b-ov-hf
adapter: artifacts/runs/llava_onevision_qwen2_0_5b_behavior_lora_full/checkpoint
```

Then run one Qwen-family smoke:

```text
Qwen/Qwen2.5-VL-7B-Instruct
adapter: artifacts/runs/qwen25vl7b_behavior_lora_full/checkpoint
```

For Qwen3B, either update the config path or pass the actual checkpoint path
explicitly:

```text
x3d_ckpts/runs/qwen25vl3b_behavior_lora_full/checkpoint
```

Success criteria:

- one prediction JSONL row is written,
- no generation error is emitted,
- `parsed=True`,
- metric JSON and confusion CSV are written,
- command can be resumed without duplicating completed rows.

## Phase 2: Baseline Latency/Eval Grid

Use existing `bench-latency` and adapter eval paths to create clean baseline
rows before adding new inference modes.

Minimum grid:

| Model | Dataset | Frames | Mode |
|---|---|---:|---|
| Qwen2.5-VL-3B LoRA | KABR | 8 | generation |
| Qwen2.5-VL-3B LoRA | KABR | 4 | generation |
| LLaVA-OV-0.5B LoRA | KABR | 8 | generation |
| LLaVA-OV-0.5B LoRA | KABR | 4 | generation |
| Qwen2.5-VL-3B B2/B3/B4 | Baboonland | 8 | generation |
| LLaVA-OV-0.5B LoRA | Baboonland | 8 | generation |

Metrics:

- accuracy
- macro F1
- per-class F1
- median latency
- p90 latency
- peak GPU memory
- frames per clip
- parsed rate

## Phase 3: Label-Likelihood Classification

Add a non-generation inference mode for closed-set behavior classification:

```text
score(label) = log P(label | visual input, prompt)
prediction = argmax_label score(label)
```

Prediction JSONL should include:

```json
{
  "inference_mode": "label_likelihood",
  "label_scores": {},
  "prediction": {"behavior": "..."},
  "confidence": {
    "top1": "...",
    "top2": "...",
    "margin": 0.0,
    "entropy": 0.0
  }
}
```

Success criteria:

- same label set as generation eval,
- no free-form parsing needed for the predicted label,
- confidence margin and entropy are available per clip,
- accuracy is comparable to generation on small smoke subsets.

## Phase 4: Confidence-Gated Fallback

Implement two-pass inference:

```text
compressed pass
  -> label likelihood + confidence
  -> if confident: return compressed prediction
  -> else: rerun full 8-frame input
```

Initial compression should use frame count, not internal visual tokens:

- 1 frame -> 8 frame fallback
- 2 frames -> 8 frame fallback
- 4 frames -> 8 frame fallback

Sweep confidence thresholds and report:

- accuracy
- macro F1
- median latency
- p90 latency
- fallback rate
- fallback rate by class
- accuracy per millisecond

## Phase 5: Redundancy-Aware Frame Policy

Before model-internal token merging, add a cheap redundancy-aware frame selector.

Candidate policies:

- uniform frame sampling,
- first/middle/last,
- simple pixel-difference high-motion frames,
- image-embedding similarity if cached features are available.

Compare against fixed 1/2/4-frame baselines.

Success criteria:

- redundancy-aware policy beats or matches uniform sampling at the same average
  frame count,
- fallback recovers most full-frame accuracy,
- policy is model-agnostic and works for at least Qwen and LLaVA.

## Phase 6: Adapter-Aware Importance Pilot

Start with Qwen2.5-VL only.

Low-risk adapter-aware signals:

- frame occlusion sensitivity,
- label-likelihood drop when removing frame groups,
- class-level frame-position priors,
- confidence changes under frame subsets.

Only after this works should we inspect and modify visual token spans.

## Phase 7: Visual-Token Consolidation

Prototype in a new isolated package:

```text
src/kabr_vlm/efficiency/
  label_likelihood.py
  confidence.py
  frame_policy.py
  token_policy.py
```

Order of implementation:

1. Count visual tokens per model/input.
2. Identify visual token spans for Qwen2.5-VL.
3. Add random/drop-token baseline if safe.
4. Add similarity-based token merging if model internals permit it.
5. Port only stable hooks to LLaVA-OneVision and MiniCPM.

## Immediate Next Commands

Run the current lightweight smoke again:

```bash
PYTHONPATH=src python -m py_compile \
  src/kabr_vlm/data/kabr_frames.py \
  src/kabr_vlm/data/kabr.py \
  src/kabr_vlm/cli.py \
  src/kabr_vlm/training/qwen_tasks.py \
  src/kabr_vlm/vlm_baselines.py \
  scripts/eval_cross_dataset_vlm.py
```

Then run a one-sample real adapter smoke, starting with LLaVA-OV-0.5B. Store
outputs under:

```text
artifacts/smoke_eval/
```

Suggested filename pattern:

```text
artifacts/smoke_eval/{model_slug}_{dataset}_{mode}_1sample.jsonl
```

## Risks

- Some config files point to checkpoint paths that do not exist on this disk.
- Real model loading may require Hugging Face cache or network access.
- Qwen3B KABR adapter exists under `x3d_ckpts/`, not the config-declared path.
- Full-SFT configs mostly do not have checkpoint directories and should not be
  treated as available trained models.
- Token-level consolidation may be model-specific despite the desired
  model-agnostic framing.

## Definition of Done For This Stage

This stage is complete when:

- at least one real LoRA adapter smoke passes end to end,
- Qwen and LLaVA adapter checkpoint paths are documented,
- baseline generation eval can produce prediction JSONL + metrics,
- label-likelihood implementation work can begin from a known-good eval path.
