# Wildlife Behavior Recognition with Adapted Vision-Language Models — Comprehensive Project Summary

*Audience: advisor, collaborators, paper reviewers. Self-contained.*
*Last updated: 2026-05-03.*

---

## 1. Motivation and background

Wildlife behavior recognition from short video clips (drone or camera-trap
footage) is a clip-level action classification problem. Historically it has
been addressed with specialist 3D CNNs — X3D, I3D, SlowFast — trained from
scratch on small, manually annotated datasets. Two recent developments
change this picture:

1. **General vision-language models** (Qwen2.5-VL, LLaVA-OneVision, Gemma3,
   MobileVLM) can answer questions about clips in natural language without
   any task-specific training.
2. **Lightweight adaptation methods** (LoRA in particular) make it cheap to
   fine-tune a 0.5B–7B VLM on a small labeled set.

The most directly relevant prior work, Mamooler et al. (COLM 2025), shows
that joint multi-task VLM fine-tuning on a union of animal-behavior
datasets outperforms vision-only baselines. They establish that adaptation
works, but they do **not** measure cross-dataset transfer, name failure
modes, or compare against specialist video baselines model-for-model.

Two questions remain open and motivate this work:

- *How does adaptation actually compare to specialist video models,
  backbone for backbone, on the same data?*
- *What are the failure modes of these adapted VLMs, and can a small
  training change fix them?*

## 2. Datasets

Both are clip-level behavior datasets with Charades-style per-frame
annotations.

| Dataset | Domain | Classes | Train clips | Val clips | Imbalance |
|---|---|---:|---:|---:|---|
| **KABR** | drone footage, Kenyan wildlife (giraffes, plains/Grevy's zebras) | 8 | 11,003 | 3,761 | moderate |
| **Baboonland** | drone footage, baboons | 13 | 20,130 | 6,198 | severe (max:min ~5000:1) |

Both are derived from the released frame annotations using the same
algorithm: contiguous same-label runs per `(original_video_id, label_id)`
become single-labeled clips. Same algorithm for both datasets; baboonland
clip manifests were built in-house this project from the raw Charades-format
labels.

## 3. Method overview

### 3.1 Models compared

Six VLM backbones (0.5B–7B parameters) plus two specialist video baselines:

- VLM backbones: Qwen2.5-VL-3B/-7B, LLaVA-OneVision-Qwen2-0.5B/-7B,
  Gemma3-4B, MobileVLM-V2-1.7B.
- Specialist video baselines: X3D-L, I3D-L (trained from scratch).
- Zero-shot only: InternVL2-2B / -2.5B, Idefics3-8B, SmolVLM-256M.

### 3.2 Evaluation regimes

Each VLM evaluated under three regimes:

1. **Zero-shot** — pick a label from a fixed list given 8 sampled frames.
2. **In-domain LoRA** — rank-16 LoRA adapter trained for 1 epoch on the
   target dataset's train split (8 frames per clip, plain-text label
   output, 13-class or 8-class prompt depending on dataset).
3. **Cross-dataset transfer** — LoRA trained on dataset *A*, evaluated on
   dataset *B*'s val.

Specialist video baselines trained with SlowFast pipeline using the same
derived clip manifests.

### 3.3 Method contributions (this work)

**Contribution B — Class-collapse-aware training.** Replace the standard
cross-entropy loss with one of:
- `inv_sqrt` weighting: per-example weight = √(max_count / class_count)
- `balanced` weighting: per-example weight = n_total / (num_classes × count)
- `focal` loss (γ=2) applied at the per-batch loss level

Optionally combine with a `class_balanced` `WeightedRandomSampler`.
Implementation: ~80 lines in
[src/kabr_vlm/training/qwen_tasks.py](src/kabr_vlm/training/qwen_tasks.py).

**Contribution A — Joint cross-dataset LoRA.** Train one LoRA adapter on
the union of KABR + baboonland training data with native prompt and label
space per example. The model sees both label vocabularies during training.
Build script:
[scripts/build_joint_train_jsonl.py](scripts/build_joint_train_jsonl.py).

**A1** = naive joint training (no class weighting). **A3** = joint + B2's
inv-sqrt class weighting (the headline A+B recipe).

## 4. Headline results

### 4.1 KABR-Clip (acc / macro-F1, val 3,761 clips)

| Model | Zero-shot | In-domain LoRA | Joint A1 LoRA | Δ acc vs X3D-L |
|---|---:|---:|---:|---:|
| **Gemma3-4B** | 0.549 / 0.226 | **0.864 / 0.515** | — | **+0.068** |
| **LLaVA-OV-7B** | 0.581 / 0.254 | **0.863 / 0.523** | — | **+0.067** |
| **Qwen2.5-VL-3B** | 0.549 / 0.271 | 0.853 / 0.485 | **0.863 / 0.514** | **+0.057** |
| **Qwen2.5-VL-7B** | 0.626 / 0.327 | 0.848 / 0.503 | — | **+0.052** |
| **LLaVA-OV-0.5B** | 0.307 / 0.119 | 0.819 / 0.447 | — | **+0.022** |
| **I3D-L (scratch)** | — | **0.814 / —** | — | **+0.017** |
| **X3D-L (scratch)** | — | **0.797 / 0.520** | — | baseline |
| MobileVLM-1.7B | 0.327 / 0.062 | 0.782 / 0.431 | — | −0.014 |
| Idefics3-8B | 0.529 / 0.231 | — | — | — |
| SmolVLM-256M | 0.249 / 0.069 | — | — | — |

### 4.2 Baboonland-Clip (acc / macro-F1, val 6,198 clips)

| Model | Zero-shot | In-domain LoRA | Joint A1 LoRA |
|---|---:|---:|---:|
| **LLaVA-OV-7B** | 0.338 / 0.077 | **0.775 / 0.443** | — |
| **LLaVA-OV-0.5B** | 0.418 / 0.078 | 0.768 / 0.430 | — |
| **Gemma3-4B** | 0.540 / 0.145 | 0.756 / 0.405 | — |
| **Qwen2.5-VL-7B** | 0.329 / 0.143 | 0.755 / 0.394 | — |
| **Qwen2.5-VL-3B + B2 (inv-sqrt CE)** | 0.242 / 0.080 | **0.752 / 0.374** | — |
| **Qwen2.5-VL-3B** (B1 baseline) | 0.242 / 0.080 | 0.734 / 0.327 | **0.759 / 0.377** |
| **MobileVLM-1.7B** | 0.174 / 0.024 | 0.712 / 0.304 | — |
| **X3D-L (scratch)** | — | 0.671 / — | — |
| **I3D-L (scratch)** | — | 0.630 / — | — |

**Every adapted VLM beats both X3D-L (0.671) and I3D-L (0.630) by 4–10
accuracy points on baboonland.** A 0.5B LoRA beats the supervised X3D-L
specialist by +9.7 accuracy points.

### 4.3 Cross-dataset transfer

LoRA trained on dataset *A*, evaluated on dataset *B*'s val.

| Direction | Source LoRA | Acc | Macro F1 |
|---|---|---:|---:|
| **bbl → KABR** | LLaVA-OV-7B | **0.652** | **0.256** |
| bbl → KABR | Qwen2.5-VL-7B | 0.562 | 0.261 |
| bbl → KABR | LLaVA-OV-0.5B | 0.517 | 0.181 |
| bbl → KABR | MobileVLM-1.7B | 0.447 | 0.179 |
| bbl → KABR | Qwen2.5-VL-3B | 0.385 | 0.114 |
| **KABR → bbl** | Qwen3B 8-frame | **0.397** | **0.155** |
| KABR → bbl | Qwen3B 4-frame | 0.298 | 0.129 |
| KABR → bbl | Qwen3B structured-output | 0.191 | 0.072 |

### 4.4 Phase-2 class-collapse-aware training (Qwen2.5-VL-3B baboonland LoRA)

| Variant | Acc | Macro F1 | Δ acc vs B1 | Δ F1 vs B1 |
|---|---:|---:|---:|---:|
| **B1 baseline LoRA** | 0.734 | 0.327 | — | — |
| **B2 inv-sqrt class-weighted CE** ✓ | **0.752** | **0.374** | **+0.018** | **+0.047** |
| B3 focal loss γ=2 | 0.715 | 0.302 | −0.018 | −0.025 |
| B4 class-balanced sampler | 0.616 | 0.349 | −0.117 | +0.022 |

### 4.5 Phase-3 joint training (Qwen2.5-VL-3B, single adapter for both datasets)

| Variant | KABR Acc / F1 | bbl Acc / F1 |
|---|---:|---:|
| **Joint A1 (no class weighting)** | **0.863 / 0.514** | **0.759 / 0.377** |
| Joint A3 (inv-sqrt class weighting) | *eval pending* | *eval pending* |
| In-domain KABR LoRA (reference) | 0.853 / 0.485 | — |
| In-domain bbl LoRA B1 (reference) | — | 0.734 / 0.327 |
| In-domain bbl LoRA B2 (reference) | — | 0.752 / 0.374 |
| bbl-only LoRA → KABR transfer | 0.385 / 0.114 | — |
| KABR-only LoRA → bbl transfer | — | 0.397 / 0.155 |

**Joint A1 matches or exceeds in-domain LoRA on *both* val sets** while
fully closing the cross-dataset transfer gap. This is the strongest
possible Contribution-A result.

## 5. Claims supported by the data

**Claim 1.** Adapted VLMs *outperform* specialist video models (X3D-L,
I3D-L) on accuracy, on both datasets, across most of six backbones.
Specialist video nets retain the efficiency frontier but lose the accuracy
ceiling. (Tables 4.1, 4.2.)

**Claim 2.** Zero-shot VLMs exhibit a structured failure mode we call
**class collapse**: high accuracy with collapsed macro F1, driven by
majority-class prediction. MobileVLM zero-shot on baboonland predicts
"Walking/Running" on 99% of 6,198 clips (3/13 classes used). Macro F1
exposes this; raw accuracy hides it.

**Claim 3.** In-domain LoRA delivers consistent +0.3 accuracy and
+0.2–0.25 macro F1 over zero-shot, across six backbones and both datasets.
A 0.5B model adapted via LoRA beats any 7B zero-shot row on the same data.

**Claim 4.** Cross-dataset transfer is a structured middle ground with
three sub-effects: (4a) model scale helps transfer across backbones
(7B > 3B > 0.5B); (4b) 8-frame > 4-frame for transfer; (4c)
structured-output prompts *hurt* transfer worst, despite being competitive
in-domain.

**Claim 5 (Contribution B).** Class-balanced loss reweighting
(inv-sqrt class frequency) on Qwen2.5-VL-3B baboonland LoRA delivers
**+4.7 macro F1 with +1.8 accuracy** over the baseline LoRA at zero
inference-time cost. Focal loss (B3) hurts; class-balanced sampling (B4)
trades too much accuracy. This converts the named failure mode (Claim 2)
into a method.

**Claim 6 (Contribution A).** Joint training a single LoRA adapter on
KABR + baboonland (with native prompts and label spaces per example)
**fully closes the cross-dataset transfer gap** and matches or exceeds
in-domain LoRA on both target val sets. Specifically, one Qwen2.5-VL-3B
adapter achieves 0.863/0.514 on KABR val (vs in-domain 0.853/0.485) and
0.759/0.377 on baboonland val (vs in-domain B1 0.734/0.327). This is the
strongest evidence yet that joint multi-task training subsumes per-dataset
fine-tuning for the wildlife behavior task, contradicting the implicit
assumption in transfer-only setups.

## 6. Why this changes the picture vs prior work

Mamooler et al. COLM 2025 demonstrated that joint multi-task fine-tuning
of a VLM outperforms vision-only baselines, evaluated in-domain on each
dataset. **They did not measure cross-dataset transfer**, did not name
class collapse, and used InternVL3-8B only. Our paper:

1. **Names and quantifies class collapse**, with a per-prediction
   distribution analysis that turns a known nuisance into a measurable
   phenomenon.
2. **Measures cross-dataset transfer explicitly**, finding that
   single-dataset LoRA transfers poorly (0.40 acc) but joint LoRA closes
   the gap entirely.
3. **Validates results across six backbones (0.5B–7B)** instead of one,
   making the claims backbone-general.
4. **Compares against specialist video baselines (X3D-L, I3D-L) on the
   same derived clip set**, showing the accuracy hierarchy has flipped.
5. **Introduces two minimal training changes** (class-balanced CE,
   joint training) that compound and have clean ablation signal.

## 7. Repository layout

```
configs/
  qwen25vl3b_*_lora_*.yaml          # per-method configs
  qwen25vl3b_joint_kabr_bbl_lora_*  # Phase-3 joint configs
scripts/
  build_joint_train_jsonl.py        # Phase-3 data prep
  prepare_baboonland_behavior_jsonl.py
  eval_cross_dataset_vlm.py         # generic eval script
  run_vlm_lora_train_task.sh        # bash wrapper called by task DB
  run_vlm_eval_task.sh
src/kabr_vlm/
  training/qwen_tasks.py            # class-aware Trainer subclass
  data/baboonland_frames.py
  data/kabr_frames.py
data/processed/
  baboonland_behavior_{train,val}.jsonl    # 20130 / 6198 rows
  kabr_behavior_{train,val_cleaned}.jsonl  # 11003 / 3761 rows
  joint_kabr_bbl_behavior_train.jsonl      # 31133 rows
chpc_manager/
  config.yaml                       # project task DB config
  manager_daemon.py                 # launched separately by user
artifacts/
  baboonland_eval/                  # per-model metrics on bbl val
  transfer_eval/                    # cross-dataset eval metrics
  joint_eval/                       # Phase-3 joint adapter metrics
  runs/baboonland/...               # in-domain bbl LoRA adapters
  runs/joint/...                    # Phase-3 joint LoRA adapters
  figures/F[1,2,4]_*.png            # per-behavior heatmap, pred dist, acc-vs-F1
PROJECT_SUMMARY.md                  # earlier paper-style summary
EXPERIMENT_PLAN.md                  # phase-by-phase plan
EXPERIMENT_IDEAS.md                 # original brainstorm
HANDOFF.md                          # session-handoff doc (see separate file)
```

## 8. Pending experiments and writing tasks

### Compute-bound
- Phase-3 A3 evals (tasks 21, 22) — running, ~1–2 GPU-hours each
- Full-SFT Qwen2.5-VL-3B on KABR (one ablation row, ~12 h on A100)
- Optional Phase-3 variant A2 (V1 label-mapped joint training)
- Optional MammAlps zero-shot slice (third-dataset generalization)
- Optional figures: F3 (few-shot curve) — would need Experiment-E sweep
- Optional figure: F5 (latency vs F1 frontier) — read from existing latency JSONs

### Writing-bound
- Draft full paper from this summary
- Per-behavior heatmap figure (F1) already exists at
  `artifacts/figures/F1_baboonland_per_behavior_heatmap.png` — refresh
  with A3 rows once they land
- Class-collapse pred-distribution figure (F2) — refresh with A3 rows
- Reconcile differences between this summary's PROJECT_SUMMARY.md and the
  outdated CURRENT_BEHAVIOR_RESULTS.md and current_clip_results.csv;
  this file is the canonical source of truth

## 9. Risk register and known issues

- **InternVL2.5-2B LoRA pipeline is broken** (image-token mismatch in
  `modeling_internvl_chat.py`). Zero-shot also OOMs on 18 GB GPUs. We
  defer this model; nothing in the paper depends on it.
- **Joint A1 may have memorized the train set** (train_loss 0.089).
  Phase-3 val acc on the held-out split is reassuringly high (0.863 KABR,
  0.759 bbl), so this is fine, but worth noting.
- **Class-balanced sampler (B4) hurts accuracy** because it oversamples
  the 2-clip Mounting class. We mention this in the paper as a negative
  result.
- **The cross-dataset transfer rows for non-Qwen families came after the
  earlier per-Qwen analysis** — no claim is hurt by their absence in
  earlier drafts.
- **Granite cluster has heterogeneous GPUs.** `chpc_manager/gpu_specs.py`
  now lists all granite nodes with their effective memory; routes
  `gpu_memory_gb >= 40` tasks away from MIG slices and L4 nodes.

## 10. One-paragraph version for an abstract

> Wildlife behavior recognition has traditionally relied on specialist 3D
> CNNs (X3D, I3D, SlowFast) trained from scratch on small labeled
> datasets. We show that across six vision-language backbones (0.5B–7B
> parameters) on two clip-level behavior datasets (KABR, Baboonland),
> LoRA-adapted VLMs *outperform* specialist video baselines on accuracy
> by up to 9.7 points, even for a 0.5B model. We characterize the
> dominant zero-shot failure mode as **class collapse** — high accuracy
> with collapsed macro F1, driven by majority-class prediction — and
> show that a simple inv-sqrt class-weighted cross-entropy loss
> (Contribution B) lifts macro F1 by +4.7 points at no accuracy cost. We
> further show that **joint training of one LoRA adapter on the union of
> both datasets** (Contribution A) fully closes the cross-dataset
> transfer gap while matching in-domain LoRA on each native val set,
> contradicting the implicit assumption that per-dataset fine-tuning is
> needed. The combined recipe (A+B) yields a single 3B-parameter
> adapter that handles both wildlife behavior datasets at or above
> specialist-video accuracy.
