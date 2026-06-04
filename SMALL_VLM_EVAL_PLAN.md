# Small VLM Evaluation Plan

*Created: 2026-05-16.*

This plan evaluates new small VLM candidates for the wildlife behavior
recognition project. The goal is to identify useful small/efficient zero-shot
models for the paper's resource-aware and Pareto-frontier story.

## Summary

Evaluate four small VLM candidates zero-shot on both KABR-Clip and
Baboonland-Clip, then report per-dataset metrics plus combined summary metrics.

Primary model IDs:

| Slug | Model ID |
|---|---|
| `minicpm_v46` | `openbmb/MiniCPM-V-4.6` |
| `qwen35_08b` | `Qwen/Qwen3.5-0.8B` |
| `gemma4_e2b_it` | `google/gemma-4-E2B-it` |
| `lfm25_vl_16b` | `LiquidAI/LFM2.5-VL-1.6B` |

Use **zero-shot only first**. Do not start LoRA training until these models pass
compatibility and full zero-shot evaluation.

The current local environment has:

```text
transformers 5.3.0
torch 2.11.0
accelerate 1.13.0
peft 0.18.1
```

This should be new enough to attempt these through Transformers first.
MiniCPM-V 4.6 and LFM2.5-VL-1.6B should be treated as the highest loader-risk
models.

## Model references

- MiniCPM-V Transformers docs:
  <https://huggingface.co/docs/transformers/main/model_doc/minicpmv4_6>
- MiniCPM-V HF repo:
  <https://huggingface.co/openbmb/MiniCPM-V-4.6>
- Qwen3.5-0.8B HF repo:
  <https://huggingface.co/Qwen/Qwen3.5-0.8B>
- Gemma4-E2B-it HF repo:
  <https://huggingface.co/google/gemma-4-E2B-it>
- LFM2.5-VL-1.6B HF repo:
  <https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B>

## Evaluation datasets

Use `scripts/eval_cross_dataset_vlm.py` as the canonical evaluator for both
datasets.

| Dataset | Validation JSONL | Target |
|---|---|---|
| KABR-Clip | `data/processed/kabr_behavior_val_cleaned.jsonl` | `kabr` |
| Baboonland-Clip | `data/processed/baboonland_behavior_val.jsonl` | `baboonland` |

Expected validation sizes:

| Dataset | Clips |
|---|---:|
| KABR-Clip | 3,761 |
| Baboonland-Clip | 6,198 |

## Output layout

Store predictions and metrics under:

```text
artifacts/small_vlm_eval/kabr/{slug}_predictions.jsonl
artifacts/small_vlm_eval/kabr/{slug}_predictions.jsonl.metrics.json
artifacts/small_vlm_eval/baboonland/{slug}_predictions.jsonl
artifacts/small_vlm_eval/baboonland/{slug}_predictions.jsonl.metrics.json
```

Aggregate outputs:

```text
artifacts/small_vlm_eval/small_vlm_summary.csv
artifacts/small_vlm_eval/small_vlm_summary.md
```

Important: the evaluator writes metrics to:

```text
<output_jsonl>.metrics.json
```

This is concat-style, not suffix replacement.

## Key implementation changes

### 1. Add a model list

Add:

```text
configs/small_vlm_eval_models.tsv
```

Contents:

```tsv
slug	model_id
minicpm_v46	openbmb/MiniCPM-V-4.6
qwen35_08b	Qwen/Qwen3.5-0.8B
gemma4_e2b_it	google/gemma-4-E2B-it
lfm25_vl_16b	LiquidAI/LFM2.5-VL-1.6B
```

### 2. Use the existing evaluator first

Use:

```text
scripts/eval_cross_dataset_vlm.py
```

This script already supports:

- zero-shot evaluation,
- KABR and Baboonland target label spaces,
- fuzzy label extraction,
- per-record prediction JSONL,
- metrics JSON with accuracy, macro-F1, parsed rate, per-class F1, prediction
  distribution, and gold distribution.

### 3. Loader strategy

First try the existing generic pipeline path:

```python
pipeline(
    task="image-text-to-text",
    model=model_id,
    device_map="auto",
    dtype="auto",
    trust_remote_code=True,
)
```

If a model fails the generic pipeline smoke test, add a model-family branch in:

```text
src/kabr_vlm/vlm_baselines.py
```

Use the existing `HFChatGenerator` pattern with:

```python
AutoProcessor.from_pretrained(...)
AutoModelForImageTextToText.from_pretrained(...)
```

Model-specific notes:

- **Qwen3.5-0.8B**: do not force the existing
  `Qwen2_5_VLForConditionalGeneration` path unless it works. Prefer generic
  `AutoModelForImageTextToText` if the Qwen3.5 class differs.
- **MiniCPM-V 4.6**: likely needs careful processor/chat-template handling.
  Do not use GPTQ/BNB variants for the main comparison unless BF16
  `openbmb/MiniCPM-V-4.6` cannot run.
- **Gemma4-E2B-it**: check gated access and runtime class support. Use the
  existing HF token mechanism if needed.
- **LFM2.5-VL-1.6B**: try generic Transformers first; if image-token or
  processor formatting fails, add a dedicated branch.

## Evaluation sequence

### Stage 1: Compatibility smoke

Run each model on both datasets with:

```text
max_samples=16
max_new_tokens=16
```

This gives:

```text
4 models x 2 datasets = 8 smoke evals
```

Acceptance checks:

- output JSONL exists,
- output JSONL contains 16 rows,
- metrics JSON exists,
- `parsed_rate >= 0.90`,
- generation errors below 5%,
- prediction text is not systematically empty or malformed.

If a smoke test fails:

1. Inspect `pred_text`.
2. Determine whether the issue is model loading, prompt rendering, image input
   formatting, or label parsing.
3. Fix loader/prompt rendering if reasonable.
4. Rerun smoke before launching full eval.

### Stage 2: Full zero-shot evaluation

Only run full evaluation for models that pass smoke.

Full runs:

```text
4 models x 2 datasets = up to 8 full evals
```

Use:

```text
max_new_tokens=16
```

Do not set `max_samples`.

### Stage 3: Aggregation

Add:

```text
scripts/analysis/summarize_small_vlm_eval.py
```

It should read all completed metrics JSON files and write:

```text
artifacts/small_vlm_eval/small_vlm_summary.csv
artifacts/small_vlm_eval/small_vlm_summary.md
```

Required summary columns:

| Column | Meaning |
|---|---|
| slug | Model slug |
| model_id | Hugging Face model ID |
| kabr_accuracy | KABR accuracy |
| kabr_macro_f1 | KABR macro-F1 |
| kabr_parsed_rate | KABR parsed rate |
| baboonland_accuracy | Baboonland accuracy |
| baboonland_macro_f1 | Baboonland macro-F1 |
| baboonland_parsed_rate | Baboonland parsed rate |
| combined_accuracy | Total correct over total clips |
| combined_macro_f1 | Mean of KABR macro-F1 and Baboonland macro-F1 |
| mean_rank | Mean rank across KABR acc/F1 and Baboonland acc/F1 |
| status | completed, smoke_failed, full_failed, blocked |
| notes | Loader/access/OOM notes |

Combined metric definitions:

- `combined_accuracy`:

```text
(kabr_correct + baboonland_correct) / (kabr_total + baboonland_total)
```

- `combined_macro_f1`:

```text
(kabr_macro_f1 + baboonland_macro_f1) / 2
```

Use mean macro-F1 rather than pooled class F1 because the two datasets have
different label spaces.

## Task DB execution

Use the existing task DB:

```text
/uufs/chpc.utah.edu/common/home/u1320844/task_dbs/vlm_behavior.db
```

Manager config:

```text
chpc_manager/config.yaml
```

Task type:

```text
vlm_cross_eval
```

Default resources:

```text
gpu_memory_gb: 21
mem: 48G
walltime: 12:00:00
```

If any model OOMs, rerun that model on a 40GB+ partition if available or mark
the memory requirement in the summary.

### Example smoke task: KABR

```bash
cd /scratch/general/vast/j.yan/myproj1/chpc_manager
PY=/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python

$PY cli.py --config config.yaml add vlm_cross_eval \
  '{"val_jsonl":"/scratch/general/vast/j.yan/myproj1/data/processed/kabr_behavior_val_cleaned.jsonl",
    "target_dataset":"kabr",
    "model_id":"Qwen/Qwen3.5-0.8B",
    "output_jsonl":"/scratch/general/vast/j.yan/myproj1/artifacts/small_vlm_eval/kabr/qwen35_08b_predictions_smoke.jsonl",
    "max_samples":"16",
    "max_new_tokens":"16"}'
```

### Example smoke task: Baboonland

```bash
cd /scratch/general/vast/j.yan/myproj1/chpc_manager
PY=/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python

$PY cli.py --config config.yaml add vlm_cross_eval \
  '{"val_jsonl":"/scratch/general/vast/j.yan/myproj1/data/processed/baboonland_behavior_val.jsonl",
    "target_dataset":"baboonland",
    "model_id":"Qwen/Qwen3.5-0.8B",
    "output_jsonl":"/scratch/general/vast/j.yan/myproj1/artifacts/small_vlm_eval/baboonland/qwen35_08b_predictions_smoke.jsonl",
    "max_samples":"16",
    "max_new_tokens":"16"}'
```

### Example full task: KABR

```bash
cd /scratch/general/vast/j.yan/myproj1/chpc_manager
PY=/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python

$PY cli.py --config config.yaml add vlm_cross_eval \
  '{"val_jsonl":"/scratch/general/vast/j.yan/myproj1/data/processed/kabr_behavior_val_cleaned.jsonl",
    "target_dataset":"kabr",
    "model_id":"Qwen/Qwen3.5-0.8B",
    "output_jsonl":"/scratch/general/vast/j.yan/myproj1/artifacts/small_vlm_eval/kabr/qwen35_08b_predictions.jsonl",
    "max_new_tokens":"16"}'
```

### Example full task: Baboonland

```bash
cd /scratch/general/vast/j.yan/myproj1/chpc_manager
PY=/uufs/chpc.utah.edu/common/home/u1320844/miniconda3/bin/python

$PY cli.py --config config.yaml add vlm_cross_eval \
  '{"val_jsonl":"/scratch/general/vast/j.yan/myproj1/data/processed/baboonland_behavior_val.jsonl",
    "target_dataset":"baboonland",
    "model_id":"Qwen/Qwen3.5-0.8B",
    "output_jsonl":"/scratch/general/vast/j.yan/myproj1/artifacts/small_vlm_eval/baboonland/qwen35_08b_predictions.jsonl",
    "max_new_tokens":"16"}'
```

## Paper-readiness checks

After full evals complete:

1. Compare new small VLMs against existing small baselines:
   - LLaVA-OV-0.5B,
   - MobileVLM-1.7B,
   - SmolVLM-256M if available.
2. Add a compact "small VLM zero-shot frontier" table to the paper notes.
3. Inspect prediction distributions, especially on Baboonland, to identify
   class collapse.
4. Decide whether any model is strong enough to justify LoRA training.

## Success criteria

The small-VLM evaluation is worth including in the paper if at least one of the
following happens:

1. A small model beats or approaches LLaVA-OV-0.5B zero-shot on either dataset.
2. A small model has better macro-F1 than larger zero-shot models under
   Baboonland class imbalance.
3. A small model becomes a strong latency/parameter Pareto point.
4. A model shows a qualitatively different class-collapse pattern worth
   discussing.
5. The results justify a follow-up LoRA run for one small candidate.

## Assumptions

- Scope is zero-shot first, not LoRA.
- Evaluate all four candidates.
- Evaluate both KABR and Baboonland.
- Report per-dataset metrics plus combined metrics.
- Keep LoRA fine-tuning out of this plan until zero-shot ranking identifies
  worthwhile candidates.
- If Gemma4 requires gated access, use the existing HF token mechanism in
  `run_vlm_eval_task.sh`; if access is unavailable, mark it as blocked rather
  than silently replacing the model.
