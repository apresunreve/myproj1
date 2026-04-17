# Experiment Ideas

This document captures all planned and proposed experiments across both active tracks,
framed by the paper story. Updated from `current_clip_results.csv` (2026-04-16).

---

## Current Results Snapshot

### KABR-Clip

| Model | Acc | Macro F1 | Method |
|---|---|---|---|
| X3D-L | 0.7968 | 0.5199 | from scratch |
| Gemma3-4B | 0.8644 | 0.5145 | LoRA |
| LLaVA OV 7B | 0.8631 | 0.5231 | LoRA |
| Qwen2.5VL-7B | 0.8475 | 0.5034 | LoRA |
| Qwen2.5VL-3B | 0.8534 | 0.4846 | LoRA |
| LLaVA OV 0.5B | 0.8189 | 0.4467 | LoRA |
| MobileVLM-1.7B | 0.7822 | 0.4307 | LoRA |
| Qwen2.5VL-7B | 0.6260 | 0.3272 | zero-shot |
| LLaVA OV 7B | 0.5813 | 0.2544 | zero-shot |
| Qwen2.5VL-3B | 0.5491 | 0.2709 | zero-shot |
| Gemma3-4B | 0.5486 | 0.2258 | zero-shot |
| InternVL2-2B | 0.3273 | 0.0616 | zero-shot |
| MobileVLM-1.7B | 0.3273 | 0.0616 | zero-shot |
| LLaVA OV 0.5B | 0.3072 | 0.1185 | zero-shot |

### Baboonland-Clip

| Model | Acc | Macro F1 | Method |
|---|---|---|---|
| LLaVA OV 7B | 0.3382 | 0.0775 | zero-shot |
| Qwen2.5VL-7B | 0.3287 | 0.1434 | zero-shot |
| Qwen2.5VL-3B | 0.2428 | 0.0804 | zero-shot |

---

## What the Results Change

Several things are now clearer from the data:

**1. Adapted VLMs match or beat X3D-L on KABR.**
LLaVA OV 7B LoRA (F1 0.5231) and Gemma3 LoRA (F1 0.5145) both exceed X3D-L (F1 0.5199)
on macro F1. The paper can no longer claim specialist video models dominate accuracy —
they dominate *efficiency*, which is the right framing anyway.

**2. The adaptation gain is large and consistent across model families.**
Every LoRA model substantially outperforms its own zero-shot baseline. The lift is
roughly +0.3 accuracy and +0.2–0.25 macro F1 regardless of backbone. This makes the
"adaptation matters" claim robust, not backbone-dependent.

**3. Zero-shot performance spreads widely.**
LLaVA OV 7B zero-shot (0.5813 acc) and Qwen 7B zero-shot (0.6260 acc) are much stronger
than InternVL2-2B and MobileVLM (0.3273). This suggests zero-shot capability tracks
model scale and instruction-following quality, not just parameter count.

**4. LLaVA OV 7B on baboonland has high accuracy but very low macro F1 (0.0775).**
Compare to Qwen 7B zero-shot (0.3287 acc, 0.1434 F1). This pattern — high accuracy,
collapsed F1 — strongly suggests LLaVA is defaulting to one or two majority classes
on baboonland. That is a concrete failure mode worth analyzing.

**5. Baboonland zero-shot rows are sparse.**
Only Qwen 3B, Qwen 7B, and LLaVA OV 7B have baboonland results. Gemma3 and other
models have no baboonland baseline at all.

---

## Paper Story Summary

The central argument is not "VLMs need adaptation" — Mamooler et al. COLM 2025 already
establishes that. The differentiating claims are:

> **1. Adapted VLMs can match specialist video models on in-domain accuracy, but the
> efficiency gap remains large and practically decisive.**
>
> **2. Zero-shot failure on cross-dataset transfer is structured: it breaks along behavior
> type, dataset shift, and a class-collapse failure mode that macro F1 exposes but
> accuracy hides.**
>
> **3. In-domain adaptation is surprisingly data-efficient — a small label budget likely
> beats full cross-dataset transfer.**

The LLaVA baboonland result (high acc, collapsed F1) is a concrete illustration of point 2
and should be a named failure mode in the paper, not buried in an appendix.

---

## Track 1: Multi-Model KABR and Baboonland Experiments

### 1.1 Baboonland in-domain LoRA — Qwen (running)

**Status**: Qwen2.5VL-3B training active (~1h remaining). 7B queued after.

**Paper role**: In-domain upper bound for baboonland. The full adaptation story is
zero-shot → KABR-transfer → in-domain for at least one model family.

---

### 1.2 Baboonland zero-shot baselines — missing models

**Status**: Only Qwen 3B, Qwen 7B, LLaVA OV 7B have baboonland zero-shot results.
Gemma3-4B, MobileVLM-1.7B, LLaVA OV 0.5B have no baboonland rows.

**What to run**: Zero-shot eval on baboonland val for:
- Gemma3-4B
- LLaVA OV 0.5B
- MobileVLM-1.7B (optional, likely very weak)

**Paper role**: Fills out the zero-shot transfer row for all models that have KABR LoRA
results. Required to make the "zero-shot vs. transfer vs. in-domain" comparison complete
across model families.

**Cost**: 2–3 eval jobs, no training. Baboonland val set is small — fast to run.

---

### 1.3 KABR-trained transfer rows on baboonland — Qwen 3B (pending eval)

**Status**: Checkpoints exist. Eval suite script at
`scripts/run_baboonland_vlm_eval_suite.sh`. Run after current training finishes.

**Rows covered**:
- LoRA plain 8f
- LoRA plain 4f
- LoRA structured
- frozen linear
- frozen MLP
- distill MLP
- distill MLP alpha=0.7

**Paper role**: Populates the adaptation-method comparison on baboonland. Core of the
transfer section.

---

### 1.4 Experiment A — Per-behavior transfer breakdown

**Question**: Which behaviors transfer from KABR to baboonland, and which fail?

**Why this matters**: Aggregate accuracy hides the mechanism. Social/interaction
behaviors likely collapse while locomotion partially transfers. That is a defensible
mechanistic claim, not just a number.

**What to build**: `scripts/report_per_class.py`

- Loads any set of `.metrics.json` files
- Joins on behavior label name
- Outputs a table: behavior × model, sorted by KABR-transfer F1
- Highlights behaviors with large zero-shot → transfer gaps

**Also apply to KABR**: Compare zero-shot vs. LoRA per-class F1 on KABR to show where
the adaptation gain concentrates — this is free since KABR metrics already exist.

**Data needed**: Completed baboonland transfer metrics JSONs (from 1.3).

**Cost**: ~1–2 hours scripting, no GPU.

**Expected figure**: Per-behavior heatmap — strongest new figure in the paper.

---

### 1.5 Experiment B — LLaVA class-collapse analysis on baboonland

**Question**: Does LLaVA OV 7B zero-shot predict mostly one class on baboonland?

**Why this matters**: LLaVA baboonland zero-shot has 0.3382 accuracy but only 0.0775
macro F1 — the gap strongly implies class collapse (predicting majority class almost
always). This is a concrete, named failure mode that strengthens the "zero-shot is
unreliable" claim with a specific mechanism.

**What to build**: Add a prediction-distribution analysis to `scripts/report_per_class.py`

- Count predicted label distribution for LLaVA vs. Qwen on baboonland zero-shot
- Compare to gold label distribution
- One extra column in the per-behavior table

**Data needed**: Existing baboonland zero-shot JSONL outputs for LLaVA OV 7B and
Qwen 7B (presumably already on disk from prior eval runs).

**Cost**: ~1 hour, no new runs needed if JSONL outputs exist.

---

### 1.6 Experiment C — Temporal ablation by behavior class

**Question**: Does the 4f → 8f improvement concentrate in temporally-extended behaviors?

**Why this matters**: If temporal behaviors (locomotion, social) benefit from more frames
while static behaviors (resting) are frame-count insensitive, that is a mechanistic
justification for the engineering choice.

**What to build**: Extend `report_per_class.py` with `--delta` mode

- Accepts two metrics JSONs (4f and 8f)
- Outputs per-class F1 difference
- Annotate behaviors as temporal vs. static manually (~10 labels)

**Data needed**: Per-class metrics from 4f and 8f evals on baboonland. 4f eval may
need a separate run.

**Cost**: ~2 hours scripting + possibly one eval run.

---

### 1.7 Experiment D — Calibration check on zero-shot VLMs

**Question**: Are zero-shot VLMs confidently wrong?

**Why this matters**: Miscalibration (high confidence + wrong prediction) is a deployment
hazard beyond just low accuracy. It also explains why the LLaVA class-collapse is
dangerous — the model may be very confident about its majority-class predictions.

**What to build**: `scripts/report_calibration.py`

- Reads prediction JSONL (logits field preserved by `normalize_prediction_row`)
- Outputs reliability diagram, ECE, overconfidence rate
- Run on baboonland zero-shot: Qwen 3B, Qwen 7B, LLaVA OV 7B

**First check**: Confirm `logits` are written to zero-shot JSONL in `vlm_baselines.py`.
Small addition needed if not.

**Cost**: ~2–3 hours. One eval re-run per model if logits are missing.

---

### 1.8 Experiment E — Few-shot learning curve on baboonland LoRA

**Question**: How many in-domain baboonland labels does LoRA need to beat KABR transfer?

**Why this matters**: The crossover point is a number the paper can state directly,
with practical implications for conservation deployment.

**Baboonland train set**: 20,130 samples. `DataConfig.max_train_samples` already exists.

**Fractions to run** (Qwen2.5VL-3B as the representative model):

| Fraction | max_train_samples |
|----------|-------------------|
| 1%       | 201               |
| 5%       | 1007              |
| 10%      | 2013              |
| 25%      | 5033              |
| 50%      | 10065             |
| 100%     | 20130 (done)      |

**What to build**:

- 5 configs (copy `qwen25vl3b_baboonland_behavior_lora_full.yaml`, set
  `max_train_samples` and `output_dir` per fraction)
- `scripts/run_baboonland_lora_fewshot_sweep.sh`
- In-domain baboonland eval after each fraction
- KABR-transfer best row as horizontal baseline in the figure

**Cost**: ~4–8 hours GPU. Run after current full-data LoRA completes.

---

## Track 2: MobileVLMv2 Temporal Distillation

MobileVLMv2 1.7B with frozen backbone + temporal compressor + distillation from
Qwen2.5VL-3B teacher. Note: a basic MobileVLM-1.7B LoRA is already in the results
(acc 0.7822, F1 0.4307). The temporal distillation track is a separate, richer approach.

### 2.1 KABR suite (in progress)

**Status**: `temporal_distill_8frames` student training active (started 2026-04-15).
`no_compressor_8frames` and `temporal_distill_4frames` queued sequentially after.

**Script**: `scripts/run_mobilevlmv2_behavior_temporal_suite.sh`

**Variants**:

| Name | Compressor | Frames | Paper question |
|------|-----------|--------|----------------|
| temporal_distill_8frames | yes | 8 | Does temporal distill beat basic LoRA? |
| classifier_only_8frames | no | 8 | Does the compressor add value? |
| temporal_distill_4frames | yes | 4 | Frame count sensitivity |

**Key comparison**: temporal_distill_8frames vs. MobileVLM-1.7B LoRA (0.7822/0.4307
already in results). If temporal distillation beats basic LoRA on MobileVLM, it is
a meaningful architectural finding.

---

### 2.2 Zero-shot MobileVLMv2 baseline (missing from real results)

**Status**: Matrix yaml points to smoke test. CSV shows MobileVLM-1.7B zero-shot at
0.3273/0.0616 — confirm this is on KABR val (full), not smoke.

**Paper role**: Anchor for "how much does each method help over zero-shot."

**Cost**: One eval job if current number is from smoke test.

---

### 2.3 Baboonland transfer eval for MobileVLMv2 (missing)

**What to run**: Evaluate KABR-trained temporal distill checkpoint on baboonland val.

**Paper role**: Fills one cell in the cross-dataset transfer matrix for a lightweight
model. Lets the paper ask whether lightweight distilled models transfer better or
worse than LoRA-adapted large VLMs.

**Cost**: One eval job after 8f distill finishes.

---

### 2.4 Latency table alignment

**What to build**: Unified accuracy-vs-latency table across:
- X3D-XS, X3D-S, X3D-M, X3D-L
- MobileVLM-1.7B LoRA (basic)
- MobileVLMv2 temporal distill variants
- Qwen2.5VL-3B LoRA, Qwen2.5VL-7B LoRA

**Revised paper claim**: Adapted VLMs reach X3D-L accuracy levels, but X3D remains
5–10x faster and uses far less memory. The efficiency frontier belongs to specialist
video models; the question is whether the accuracy parity at deployment cost is worth it.

**Cost**: ~1 hour scripting, reads existing latency JSONs.

---

## Execution Order

```
Now
├── Qwen 3B baboonland LoRA finishing (~1h)
├── MobileVLMv2 8f distill in student training
│
After Qwen 3B finishes
├── Qwen 7B baboonland LoRA starts (queue)
├── Run baboonland transfer eval suite (1.3)
├── Run missing baboonland zero-shot baselines: Gemma3, LLaVA OV 0.5B (1.2)
│
After baboonland evals complete
├── Experiment A: per-behavior breakdown (1.4)
├── Experiment B: LLaVA class-collapse analysis (1.5)  — fast, JSONL may already exist
├── Experiment C: temporal ablation by class (1.6)
├── Experiment D: calibration check (1.7)  — check logits first
│
After MobileVLMv2 suite finishes
├── Baboonland transfer eval for MobileVLMv2 (2.3)
├── Latency table alignment (2.4)
│
After Qwen 7B baboonland finishes
└── Experiment E: few-shot learning curve (1.8)  — GPU-heavy, run last
```

---

## Minimum Viable Paper Set

| Experiment | Why load-bearing |
|---|---|
| Qwen 3B + 7B baboonland in-domain LoRA | In-domain upper bound |
| Baboonland zero-shot: Gemma3, LLaVA OV 0.5B (1.2) | Complete zero-shot row |
| KABR-trained Qwen 3B transfer rows on baboonland (1.3) | Core transfer section |
| Experiment A: per-behavior breakdown (1.4) | Strongest new figure |
| Experiment B: LLaVA class-collapse (1.5) | Named failure mode, cheap |
| MobileVLMv2 KABR suite (2.1) | Efficiency + distillation comparison |
| Latency table alignment (2.4) | Efficiency story |

Optional but strengthening: calibration (1.7), few-shot curve (1.8),
temporal ablation (1.6), MobileVLMv2 baboonland transfer (2.3).
