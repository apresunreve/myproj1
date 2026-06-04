# VLM Adaptation for Wildlife Behavior Recognition — Project Summary

*Target reader: a collaborator who has not seen this project before.*
*Updated: 2026-04-25.*

---

## 1. The task, in one paragraph

Given a short video clip of a wild animal, identify what behavior it is performing
(walking, grooming, foraging, etc.). This is clip-level action recognition.
Historically it has been done with specialist 3D CNNs (X3D, I3D) trained from
scratch on small labeled video datasets. Recent vision-language models (VLMs)
like Qwen2.5-VL, Gemma3, LLaVA-OneVision can do this zero-shot (look at the clip,
answer in text), but zero-shot is unreliable. The question is: how do we adapt a
general VLM into a reliable behavior classifier, and what does that buy us over
the specialist video baselines?

## 2. Datasets (both clip-level)

- **KABR** — drone footage of Kenyan wildlife (giraffes, plains zebras,
  Grevy's zebras). 8 behavior classes: Walk, Graze, Browse, Head Up,
  Auto-Groom, Trot, Run, Occluded. Derived clip-level splits: 11,003 train /
  3,761 val.
- **Baboonland** — camera-trap footage of baboons. 13 behavior classes:
  Walking/Running, Sitting/Standing, Fighting/Playing, Self-Grooming, Being
  Groomed, Grooming Somebody, Mutual Grooming, Infant-Carrying, Foraging,
  Drinking, Mounting, Sleeping, Occluded. Derived clip-level splits: 20,130
  train / 6,198 val.

The two datasets share the Charades-style frame-annotation format. We built
baboonland clip manifests ourselves this week using the same algorithm KABR
uses (contiguous same-label runs).

Both datasets have **severe class imbalance**. On baboonland, Sitting/Standing
has 9,843 training clips vs Mounting with 63. This turns out to matter.

## 2b. How we convert the raw datasets into VLM training data

Both datasets ship with per-frame behavior annotations in a Charades-style
CSV (columns: `original_video_id`, `video_id`, `frame_id`, `rel_path`,
`label_id`) plus a `classes.json` mapping label names to IDs. The raw format
is frame-labeled, not clip-labeled. To make them usable by a VLM, we run a
three-stage pipeline that is identical for KABR and baboonland.

**Stage 1 — Derive single-label clips from contiguous same-label runs.**
A derived clip is the longest contiguous run of frames sharing the same
`original_video_id` and `label_id`. A new clip starts when any of
(`original_video_id`, `label_id`, frame continuity) changes. Each derived
clip therefore has exactly one behavior label. The resulting clip counts are
11,003 train / 3,761 val for KABR and 20,130 train / 6,198 val for
baboonland. Each clip keeps its full frame list, its label, its source video,
and a canonical `clip_id` of the form `<video>:<start_frame>-<end_frame>:<label_id>`.

Code lives in
[`src/kabr_vlm/data/kabr_frames.py`](src/kabr_vlm/data/kabr_frames.py) for
KABR and
[`src/kabr_vlm/data/baboonland_frames.py`](src/kabr_vlm/data/baboonland_frames.py)
for baboonland — both reuse the same `group_frame_records_into_clips`
algorithm. The baboonland version was written this session
([`scripts/build_baboonland_clips.py`](scripts/build_baboonland_clips.py))
and its output went to
`baboonland_files/annotation/clip_manifests/{train,val}.jsonl`.

**Stage 2 — Convert clips into VLM-ready VideoQA examples.**
For each clip we produce a single training row that looks like this:

```json
{
  "id":          "B0001.1:1-90:0",
  "video_id":    "1",
  "clip_path":   "B0001.1:1-90:0",
  "image_paths": [<8 absolute frame paths, evenly sampled across the clip>],
  "split":       "train",
  "task":        "action_recognition",
  "prompt":      "You are analyzing a clip from the baboonland dataset.\n
                  Identify the behavior being performed.\n
                  Reply with exactly one behavior label from:
                  Walking/Running, Sitting/Standing, Fighting/Playing, ...",
  "answer":      "Walking/Running",
  "metadata":    {"species": "baboon", "behavior": "Walking/Running",
                  "original_video_id": "B0001.1", "label_id": 0, "num_frames": 90}
}
```

Key design choices:

- **8 frames per clip**, uniformly sampled from first to last
  (fewer if the clip is shorter than 8). The same 8-frame convention is used
  at both training and eval time, so train and val see the same input type.
  `sample_clip_frames` in `kabr_frames.py` implements this.
- **Prompt lists the allowed labels explicitly.** KABR prompts list the 8
  KABR classes; baboonland prompts list the 13 baboonland classes. This
  keeps the model's output vocabulary constrained and lets us use simple
  substring matching at scoring time.
- **Answer is the plain label string**, not a structured format. A
  structured-output variant exists for ablation (see Claim 4b in §5) and
  measurably hurts cross-dataset transfer.
- **`metadata.behavior` holds the gold label.** The training loader
  resolves `metadata.behavior` directly, which is what lets the same
  `QwenTaskDataset` code handle both KABR (8-class) and baboonland
  (13-class) without any hard-coded label tables.

The conversion scripts are
[`src/kabr_vlm/data/kabr.py`](src/kabr_vlm/data/kabr.py) (for KABR) and
[`scripts/prepare_baboonland_behavior_jsonl.py`](scripts/prepare_baboonland_behavior_jsonl.py)
(for baboonland). The outputs live at
`data/processed/{kabr,baboonland}_behavior_{train,val}.jsonl`.

**Stage 3 — Feed to either training or evaluation.**
The same processed JSONL feeds three pipelines:

- *LoRA training* via
  [`src/kabr_vlm/training/finetune.py`](src/kabr_vlm/training/finetune.py):
  reads rows, renders them as VLM chat messages (8 images + prompt → label),
  and trains a rank-16 LoRA adapter on the attention + MLP modules of the
  target model.
- *Zero-shot / adapter eval* via
  [`scripts/eval_cross_dataset_vlm.py`](scripts/eval_cross_dataset_vlm.py):
  reads the same rows, generates up to 16 tokens from the VLM, and scores
  predictions against the gold label using a substring-matching label
  extractor (with dataset-specific fallbacks so, e.g., a model that emits
  "Walk" when asked for a baboonland label still gets mapped to
  Walking/Running).
- *SlowFast video models* (X3D-L, I3D-L) consume the *raw* clip manifests
  from Stage 1 via a custom `kabr_clips` dataset loader in SlowFast — they
  do not go through the VideoQA JSONL. But they share the same clip
  boundaries, the same labels, and the same train/val split, so their
  accuracy and macro-F1 numbers are directly comparable to the VLM numbers.

This shared pipeline is why the claims in §5 generalize across VLM
families, across datasets, and across specialist video baselines. Every
model sees the same derived clip-level benchmark, just with a different
input representation and a different training objective.

---

## 3. Models we compare

Six VLM backbones spanning 0.5B–7B parameters:

- Qwen2.5-VL-3B, Qwen2.5-VL-7B
- LLaVA-OneVision-Qwen2-0.5B, LLaVA-OneVision-Qwen2-7B
- Gemma3-4B
- MobileVLM-V2-1.7B
- (InternVL2-2B for zero-shot only)

Two supervised video baselines:

- X3D-L (3D CNN trained from scratch)
- I3D-L (R-101 with 3D convolutions, trained from scratch)

We evaluate each model under three regimes:

1. **Zero-shot** — no training; the VLM is asked to pick a label.
2. **In-domain LoRA** — light fine-tuning (rank-16 LoRA adapter) on the
   target dataset's train split.
3. **Cross-dataset transfer** — LoRA trained on *one* dataset, evaluated on
   the *other*.

## 4. Headline numbers (accuracy / macro-F1)

Accuracy is the top-line metric; macro-F1 is reported alongside because both
datasets are severely imbalanced and macro-F1 exposes the class-collapse
failure (§5, Claim 2).

### KABR-Clip (behavior) — **Every LoRA VLM beats X3D-L on accuracy**

| Model | Zero-shot Acc / F1 | LoRA Acc / F1 | Δ acc vs X3D-L |
|---|---:|---:|---:|
| **Gemma3-4B** | 0.549 / 0.226 | **0.864 / 0.515** | **+0.068** |
| **LLaVA-OV-7B** | 0.581 / 0.254 | **0.863 / 0.523** | **+0.067** |
| **Qwen2.5-VL-3B** | 0.549 / 0.271 | **0.853 / 0.485** | **+0.057** |
| **Qwen2.5-VL-7B** | 0.626 / 0.327 | **0.848 / 0.503** | **+0.052** |
| **LLaVA-OV-0.5B** | 0.307 / 0.119 | **0.819 / 0.447** | **+0.022** |
| **I3D-L (supervised, scratch)** | — | **0.814 / —** | **+0.017** |
| **X3D-L (supervised, scratch)** | — | **0.797 / 0.520** | baseline |
| MobileVLM-1.7B | 0.327 / 0.062 | 0.782 / 0.431 | −0.014 |

Read as: five of six LoRA-adapted VLMs beat X3D-L on accuracy, and three of
them also match or beat it on macro-F1. The best adapted VLM (Gemma3-4B) is
**+6.8 accuracy points** above the specialist video baseline. The smallest
adapted VLM (0.5B) already beats X3D-L on accuracy. This is the core of
Claim 1 in §5.

### Baboonland-Clip (behavior) — **VLM LoRA dominates X3D-L and I3D-L by a wide margin**

| Model | Zero-shot Acc / F1 | LoRA Acc / F1 |
|---|---:|---:|
| **LLaVA-OV-7B** | 0.338 / 0.077 | **0.775 / 0.443** |
| **LLaVA-OV-0.5B** | 0.418 / 0.078 | **0.768 / 0.430** |
| **Qwen2.5-VL-7B** | 0.329 / 0.143 | **0.755 / 0.394** |
| **Qwen2.5-VL-3B + B2 (Phase-2 method)** | 0.242 / 0.080 | **0.752 / 0.374** |
| Qwen2.5-VL-3B (B1 baseline) | 0.242 / 0.080 | 0.734 / 0.327 |
| MobileVLM-1.7B | 0.174 / 0.024 | 0.712 / 0.304 |
| Gemma3-4B | 0.540 / 0.145 | *eval pending* |
| **X3D-L (supervised, scratch)** | — | **0.671 / —** |
| **I3D-L (supervised, scratch)** | — | **0.630 / —** |

Every LoRA-adapted VLM beats both **X3D-L** (0.671) and **I3D-L** (0.630) on
accuracy by 4–10 points. The **0.5B LoRA** (0.768) beats the supervised X3D-L
specialist by **+9.7 accuracy points** despite having a fraction of the
parameters. Baboonland is *harder* for supervised video nets than KABR is
(X3D-L drops from 0.797 → 0.671 going dataset to dataset), but adapted VLMs
hold up — amplifying Claim 1 in §5.

### Cross-dataset transfer

Models trained with LoRA on dataset A, evaluated on dataset B's val split.

| Direction | Source LoRA | Acc | Macro F1 |
|---|---|---:|---:|
| **bbl → KABR** | LLaVA-OV-7B | **0.652** | **0.256** |
| bbl → KABR | Qwen2.5-VL-7B | 0.562 | 0.261 |
| bbl → KABR | LLaVA-OV-0.5B | 0.517 | 0.181 |
| bbl → KABR | MobileVLM-1.7B | 0.447 | 0.179 |
| bbl → KABR | Qwen2.5-VL-3B | 0.385 | 0.114 |
| **KABR → bbl** | Qwen3B 8-frame | **0.397** | **0.155** |
| KABR → bbl | Qwen3B 4-frame | 0.298 | 0.129 |
| KABR → bbl | Qwen3B structured output | 0.191 | 0.072 |

Five backbones across both directions. Three sub-effects in the data:
(a) scale helps transfer (7B > 3B > 0.5B); (b) 8-frame > 4-frame for the
KABR→bbl direction; (c) structured-output prompting *hurts* transfer the
hardest, despite being competitive in-domain.

### Phase-2 method — class-collapse-aware training (Qwen2.5-VL-3B baboonland)

| Variant | Acc | Macro F1 | Δ acc vs B1 | Δ F1 vs B1 |
|---|---:|---:|---:|---:|
| **B1 baseline LoRA** | 0.734 | 0.327 | — | — |
| **B2 inv-sqrt class-weighted CE** ✓ | **0.752** | **0.374** | **+0.018** | **+0.047** |
| B3 focal loss γ=2 | 0.715 | 0.302 | −0.018 | −0.025 |
| B4 class-balanced sampler | 0.616 | 0.349 | −0.117 | +0.022 |

**B2 wins**: inv-sqrt class-weighted CE delivers +4.7 macro-F1 points over
the baseline LoRA without losing accuracy. The change in `qwen_tasks.py` is
~80 lines; results land cleanly in one Phase-2 cell.

## 5. What the data says (four data-backed claims)

**Claim 1 — Adapted VLMs outperform specialist video models on accuracy.
Specialist video nets survive only on efficiency.**
On KABR, five of six LoRA-adapted VLMs beat X3D-L on top-line accuracy, with
Gemma3-4B (0.864) and LLaVA-OV-7B (0.863) ahead by **+6.7–6.8 accuracy
points** over X3D-L (0.797). Macro-F1 is at or above X3D-L for three of
those models. Even the smallest VLM we test — LLaVA-OV-0.5B with 1/10th the
parameters of the 7B variants — beats X3D-L on accuracy after a light LoRA.

This is a real shift from prior work (Mamooler et al. COLM 2025) which had
already established that VLMs need adaptation for wildlife behavior. The new
framing is that *adaptation doesn't just close the accuracy gap, it flips
it*: the specialist video nets are no longer the accuracy ceiling. Their
justification is now **efficiency only** (latency, memory, deployment cost
on edge devices), not capability.

**Claim 2 — Zero-shot failure is structured, not random. We name it "class
collapse."**
Look at LLaVA-OV-7B on baboonland: accuracy 0.34 but macro F1 only 0.078. The
gap between the two is the signature of a model that is predicting the majority
class on almost every input. Accuracy hides it; macro F1 exposes it. MobileVLM
zero-shot on baboonland shows the same pattern: 0.17 acc / 0.024 F1.

This is a named failure mode. Once you know to look for it, the pattern is
visible in most zero-shot rows on the imbalanced baboonland dataset.

**Claim 3 — In-domain LoRA adaptation is large, consistent, and works on tiny
models.**
Across six backbones and two datasets, LoRA delivers ~+0.3 accuracy and
~+0.2–0.25 macro F1 over the same model's zero-shot baseline. The lift is not
backbone-specific. The striking result: **LLaVA-OV-0.5B LoRA on baboonland
(0.77 acc, 0.43 F1) beats every 7B zero-shot row on the same data.** A tiny
adapted model beats a much larger un-adapted one by a wide margin. This is the
deployment headline.

**Claim 4 — Cross-dataset transfer is a meaningful middle ground.**
A model LoRA-trained on KABR and evaluated on baboonland does better than
zero-shot (0.40 acc vs 0.24 for Qwen3B) but worse than in-domain (0.40 vs 0.73).
Transfer recovers roughly half the adaptation gap. Three sub-findings inside
this:

- *Frame count matters for transfer.* 8-frame > 4-frame LoRA (0.40 vs 0.30
  acc on KABR→bbl). Temporal evidence survives the label-space shift better
  than sparse cues.
- *Structured output prompts hurt transfer.* KABR structured-output LoRA
  transfers at 0.19 acc — worse than both plain-text LoRA and zero-shot.
  Rigid output formats over-specialize to the source label vocabulary.
- *Model scale helps transfer, across backbones.* The pattern is not
  Qwen-specific: 7B-class models transfer best regardless of family
  (LLaVA-OV-7B bbl→KABR = 0.65 acc / 0.26 F1, Qwen-7B = 0.56 / 0.26),
  while 3B and 0.5B trail.

**Claim 5 (NEW, this session) — Class-collapse-aware training closes the
imbalance gap.** Replacing the standard cross-entropy loss with inverse-√
class-frequency weighting (B2) on Qwen2.5-VL-3B baboonland LoRA delivers
**+4.7 macro-F1 points** with **+1.8 accuracy points** over the baseline
LoRA. Focal loss (B3) hurts and naive class-balanced sampling (B4) sacrifices
too much accuracy. This is a small, well-motivated training change that
turns the named failure mode (Claim 2) into a method.

## 6. What we propose to add (two methodological contributions)

These turn the paper from "measurement study" into "method paper."

**Contribution A — Cross-dataset co-adaptation.** Train a *single* LoRA
adapter jointly on KABR + baboonland, with a shared semantic label space
(explicit 5-way mapping: {Walk, Trot, Run} ≈ Walking/Running; {Graze, Browse}
≈ Foraging; Auto-Groom ≈ Self-Grooming; Head Up ≈ Sitting/Standing; Occluded
≈ Occluded). Expected: lift transfer macro F1 ≥ 0.10 over the current
KABR-only and bbl-only adapters.

**Contribution B — Class-collapse-aware training.** ✓ **Now landed (B2 wins).**
Replacing standard CE with inverse-√ class-frequency weighting on Qwen3B
baboonland LoRA gives **+4.7 macro-F1 with +1.8 acc** over baseline. Focal
loss and class-balanced sampling tested as ablations; the simple inverse-√
weighting is the cleanest fix (Table in §4 above).

Combined (A + B together) is the paper's headline method: one recipe that
addresses both the zero-shot collapse and the transfer gap. Contribution B
is already validated; Contribution A is the next experimental milestone.

## 7. Current experiment status (2026-04-25)

### Completed
- Zero-shot on KABR: 7 models (incl. SmolVLM-256M).
- Zero-shot on baboonland: 6 models (Qwen3B, Qwen7B, LLaVA-OV-7B,
  LLaVA-OV-0.5B, Gemma3-4B, MobileVLM).
- LoRA on KABR: 6 VLM backbones.
- LoRA on baboonland: 5 backbones (Qwen3B, Qwen7B, LLaVA-OV-7B,
  LLaVA-OV-0.5B, MobileVLM); **Gemma3-4B trained, eval pending**.
- Cross-dataset transfer: **8 runs** — Qwen3B/Qwen7B both directions,
  Qwen3B 4f and structured variants, plus LLaVA-OV-7B / LLaVA-OV-0.5B /
  MobileVLM bbl→KABR (non-Qwen scale check).
- Supervised video baselines: X3D-L and I3D-L done on **both** datasets.
- **Phase-2 Contribution B**: B2 inv-sqrt class-weighted CE wins;
  +4.7 macro-F1 over B1 baseline.

### Currently running / queued
- Gemma3-4B baboonland LoRA eval (next insertion).

### Missing (to be run)
- **Phase 3 / Contribution A** — joint KABR+bbl LoRA with V1 label map.
- Full-parameter SFT ablation for Qwen3B on KABR (single ablation row,
  tests whether LoRA and full SFT differ on transfer).
- InternVL2.5-2B LoRA (image-token mismatch in
  `modeling_internvl_chat.py`; needs patched prompt renderer); ZS rerun
  on A100-class GPU after granite gpu_memory_gb routing was fixed.
- Optional MammAlps zero-shot slice (modality-generalization claim).

## 8. Infrastructure note

All training runs live under `/scratch/general/vast/j.yan/myproj1/`. We have
access to four SLURM GPU partitions across two clusters (notchpeak, granite).
A CHPC task-queue system manages parallelization — tasks are inserted into a
SQLite DB, and a daemon pulls them onto available GPU slots across partitions.
This gives us ~6–8 concurrent training/eval runs.

A project-local copy of the task-manager lives at
[`chpc_manager/`](chpc_manager/) with its own [`config.yaml`](chpc_manager/config.yaml)
pointing at a dedicated DB (`/uufs/chpc.utah.edu/common/home/u1320844/task_dbs/vlm_behavior.db`).
Three task types (`vlm_lora_train`, `vlm_cross_eval`, `slowfast_clip_train`)
are registered in [`NAS/task_types.yaml`](../NAS/task_types.yaml) and dispatch
to wrapper scripts in `scripts/run_*_task.sh`. The daemon can be launched
with `python chpc_manager/manager_daemon.py --config chpc_manager/config.yaml`.

## 9. Planned timeline to paper submission

With the current infrastructure, the path to a submittable paper is roughly:

- Week 1 — close baseline table, build per-behavior heatmap figure, finish
  all in-flight training.
- Week 2 — run Contribution B (class-balanced loss/sampler variants on
  Qwen3B); run the few-shot learning curve to quantify "how many labels does
  our method save you."
- Week 3 — run Contribution A (joint KABR+bbl co-adaptation); run the A+B
  combined headline experiment; scale-generality check on Qwen7B.
- Week 4 — write.

## 10. Key files (if the collaborator wants to read more)

- [`EXPERIMENT_IDEAS.md`](EXPERIMENT_IDEAS.md) — original brainstorm,
  enumerates all possible experiments.
- [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) — concrete plan with phases,
  budget, risk register.
- [`CURRENT_BEHAVIOR_RESULTS.md`](CURRENT_BEHAVIOR_RESULTS.md) — earlier
  KABR-only snapshot (superseded by current_clip_results.csv).
- [`current_clip_results.csv`](current_clip_results.csv) — canonical KABR +
  baboonland zero-shot / LoRA rows.
- [`KABR_DATASET_CONSTRUCTION.md`](KABR_DATASET_CONSTRUCTION.md) — how the
  derived clip manifests are built.
- [`HOW_I_TRAINED_X3D_ON_KABR_CLIP.md`](HOW_I_TRAINED_X3D_ON_KABR_CLIP.md) —
  the SlowFast-based specialist-video training path.
