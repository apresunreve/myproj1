# Current Results Summary

*Created: 2026-05-13.*

*Efficiency update: 2026-06-19.*

This document explains the current results of the wildlife behavior recognition
project for someone who has not seen the project before.

The short version: this project tests whether modern vision-language models
(VLMs) can recognize animal behaviors from short video clips better than
traditional video classification models, and how to make that inference
efficient enough for large ecological video collections. The current evidence
says yes: after lightweight fine-tuning, several VLMs outperform specialist
video models on two wildlife behavior datasets. The newer efficiency result is
that the visual input does not always need to be processed at full cost:
compressed-frame inference, generation-logit confidence, and adaptive fallback
can preserve most quality while reducing visual compute.

## 0. Current Paper Story

The paper-level contribution is now best framed as **efficient VLM adaptation
for wildlife behavior recognition**. The mature main result is speculative
visual inference / adaptive visual compute: run a cheap compressed visual pass
first, estimate confidence from normal generation logits, and spend more visual
compute only on uncertain or class-sensitive examples.

The strongest mature efficiency result is MiniCPM-V 4.6 on BaboonLand. A
4-frame pass nearly matches 8-frame quality while reducing latency, and a
class-aware entropy fallback routes about 7% of rows to 8-frame inference with
near-8-frame accuracy and much lower p90 latency than broad fallback policies.

There is also an exploratory Qwen2.5-VL visual-token pruning extension. On a
573-row stratified BaboonLand sample, 25% cluster-representative visual-token
pruning beats random pruning at the same prune rate:

| Policy | Accuracy | Macro F1 | Median Generation Latency |
|---|---:|---:|---:|
| No-prune | 0.4119 | 0.3328 | 267.54 ms |
| Random-prune 25% | 0.3892 | 0.3154 | 239.98 ms |
| Cluster-merge-prune 25% | 0.4188 | 0.3492 | 240.36 ms |
| Entropy-gated cluster-prune 25% | 0.4154 | 0.3419 | 240.48 ms |

Bootstrap on the 573-row sample shows cluster-merge-prune 25% beats
random-prune 25% by `+0.0299` accuracy (`CI +0.0103..+0.0508`) and `+0.0340`
macro F1 (`CI +0.0117..+0.0578`). Versus no-prune, quality trends positive
but the CI crosses zero, while generation latency and prompt length decrease
consistently. This should be treated as a secondary/exploratory contribution
unless a full-validation Qwen run is added.

A higher-prune 242-row stress sweep over 30%-50% pruning reinforces the same
pattern: random and similarity-merge pruning degrade as pruning gets more
aggressive, while cluster-representative pruning remains the strongest policy.
The best unconditional high-prune point was `cluster_merge_prune@0.35` with
accuracy `0.4587`, macro F1 `0.3833`, and median generation latency
`267.37 ms` versus no-prune `0.4504`, `0.3764`, and `331.57 ms`. Entropy-gated
35% cluster pruning reached accuracy `0.4711` and macro F1 `0.3956` while
pruning `182/242` rows.

## 1. What problem is this project solving?

The task is **wildlife behavior recognition from short video clips**.

Given a short clip of an animal, the model must predict a behavior label such
as walking, resting, grooming, feeding, or vigilance. This is a classification
problem: each clip has one correct behavior label, and the model must choose
the right one.

This matters because animal behavior is important for ecology and conservation,
but manually labeling behavior in large video datasets is slow and expensive.

## 2. What kinds of models are compared?

The project compares two broad types of models.

### Vision-language models

Vision-language models, or **VLMs**, are general-purpose AI models that can
look at images and answer questions in text. Instead of training a model only
to classify video clips, we can show a VLM several frames from a clip and ask:

```text
What behavior is the animal performing?
Choose one of: walking, resting, grooming, ...
```

The VLM then generates a text answer, which is matched to one of the allowed
behavior labels.

The main VLMs tested are:

- Qwen2.5-VL-3B
- Qwen2.5-VL-7B
- LLaVA-OneVision-Qwen2-0.5B
- LLaVA-OneVision-Qwen2-7B
- Gemma3-4B
- MobileVLM-V2-1.7B

### Specialist video models

The project also compares against traditional video classification models.
These models are trained specifically for video action recognition.

The specialist video baselines are:

- X3D-L
- I3D-L

These are not language models. They are video neural networks trained to map a
clip directly to a behavior class.

## 3. What does "fine-tuning" mean here?

The project tests VLMs in two main ways.

### Zero-shot evaluation

In zero-shot evaluation, the model is not trained on the wildlife dataset. It
is simply prompted with frames and asked to choose a behavior label.

This tests whether the model already understands the behavior from general
pretraining.

### LoRA fine-tuning

LoRA is a lightweight fine-tuning method. Instead of updating the entire VLM,
the project trains a small set of adapter parameters attached to the model.

This is much cheaper than full fine-tuning and allows a general VLM to adapt to
wildlife behavior labels.

In this project, "adapted VLM" usually means:

> A VLM fine-tuned with LoRA on behavior-labeled wildlife clips.

## 4. What datasets are used?

The project uses two wildlife behavior datasets.

| Dataset | What it contains | Behavior classes | Train clips | Validation clips | Difficulty |
|---|---|---:|---:|---:|---|
| KABR-Clip | Drone video of Kenyan wildlife such as giraffes and zebras | 8 | 11,003 | 3,761 | Moderate class imbalance |
| Baboonland-Clip | Drone video of baboons | 13 | 20,130 | 6,198 | Severe class imbalance |

"Class imbalance" means some behaviors appear much more often than others. For
example, there may be many walking clips but very few grooming or vigilance
clips. This makes the task harder because a model can get decent accuracy by
overpredicting common behaviors while ignoring rare ones.

## 5. What metrics are reported?

The main metrics are **accuracy** and **macro-F1**.

### Accuracy

Accuracy measures the fraction of clips classified correctly.

Example:

```text
If the model gets 80 out of 100 clips right, accuracy = 0.80.
```

Accuracy is useful, but it can hide failure on rare classes.

### Macro-F1

Macro-F1 measures performance across classes more evenly. It computes an F1
score for each behavior class, then averages those class scores.

This matters for imbalanced wildlife data because rare behaviors should not be
ignored. A model that predicts only common behaviors can have reasonable
accuracy but poor macro-F1.

In this project, macro-F1 is especially important for detecting **class
collapse**.

## 6. What is class collapse?

Class collapse is a failure mode where a model predicts only a few common
classes instead of using the full behavior label set.

For example, on Baboonland, a zero-shot model might predict "Walking/Running"
for almost every clip. This can produce nonzero accuracy because
Walking/Running is common, but it performs badly on rare behaviors.

The project identifies class collapse as a major weakness of zero-shot VLMs for
wildlife behavior recognition.

## 7. Main KABR-Clip results

KABR-Clip validation set: 3,761 clips.

| Model | Zero-shot Acc / F1 | LoRA Acc / F1 | Joint LoRA Acc / F1 |
|---|---:|---:|---:|
| Gemma3-4B | 0.549 / 0.226 | **0.864 / 0.515** | - |
| LLaVA-OV-7B | 0.581 / 0.254 | **0.863 / 0.523** | - |
| Qwen2.5-VL-3B | 0.549 / 0.271 | 0.853 / 0.485 | **0.863 / 0.514** |
| Qwen2.5-VL-7B | 0.626 / 0.327 | 0.848 / 0.503 | - |
| LLaVA-OV-0.5B | 0.307 / 0.119 | 0.819 / 0.447 | - |
| MobileVLM-1.7B | 0.327 / 0.062 | 0.782 / 0.431 | - |
| X3D-L specialist video model | - | 0.797 / 0.520 | - |
| I3D-L specialist video model | - | 0.814 / - | - |

What this means:

- Zero-shot VLMs are not enough.
- LoRA fine-tuning gives large improvements.
- Several adapted VLMs reach about 86% accuracy.
- Several adapted VLMs beat the X3D-L specialist video model on accuracy.
- Qwen2.5-VL-3B trained jointly on both datasets performs better than its
  KABR-only LoRA version.

## 8. Main Baboonland-Clip results

Baboonland-Clip validation set: 6,198 clips.

| Model | Zero-shot Acc / F1 | LoRA Acc / F1 | Joint LoRA Acc / F1 |
|---|---:|---:|---:|
| LLaVA-OV-7B | 0.338 / 0.077 | **0.775 / 0.443** | - |
| LLaVA-OV-0.5B | 0.418 / 0.078 | 0.768 / 0.430 | - |
| Gemma3-4B | 0.540 / 0.145 | 0.756 / 0.405 | - |
| Qwen2.5-VL-7B | 0.329 / 0.143 | 0.755 / 0.394 | - |
| Qwen2.5-VL-3B + inv-sqrt class weighting | 0.242 / 0.080 | **0.752 / 0.374** | - |
| Qwen2.5-VL-3B baseline LoRA | 0.242 / 0.080 | 0.734 / 0.327 | **0.759 / 0.377** |
| MobileVLM-1.7B | 0.174 / 0.024 | 0.712 / 0.304 | - |
| X3D-L specialist video model | - | 0.671 / - | - |
| I3D-L specialist video model | - | 0.630 / - | - |

What this means:

- Baboonland is harder than KABR because it has more classes and much stronger
  class imbalance.
- Zero-shot VLMs have very poor macro-F1, showing class collapse.
- After LoRA fine-tuning, every tested VLM beats both specialist video models
  on accuracy.
- Even the small LLaVA-OV-0.5B model beats X3D-L by about 9.7 accuracy points.
- The best adapted VLM reaches 0.775 accuracy and 0.443 macro-F1.

## 9. Cross-dataset transfer results

Cross-dataset transfer asks:

> If a model is fine-tuned on one wildlife dataset, can it work on another
> wildlife dataset without being fine-tuned again?

This is important because wildlife datasets are expensive to label. If a model
trained on one species or location transfers well, it becomes much more useful.

| Direction | Source LoRA | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Baboonland -> KABR | LLaVA-OV-7B | **0.652** | 0.256 |
| Baboonland -> KABR | Qwen2.5-VL-7B | 0.562 | **0.261** |
| Baboonland -> KABR | LLaVA-OV-0.5B | 0.517 | 0.181 |
| Baboonland -> KABR | MobileVLM-1.7B | 0.447 | 0.179 |
| Baboonland -> KABR | Qwen2.5-VL-3B | 0.385 | 0.114 |
| KABR -> Baboonland | Qwen2.5-VL-3B, 8-frame | **0.397** | **0.155** |
| KABR -> Baboonland | Qwen2.5-VL-3B, 4-frame | 0.298 | 0.129 |
| KABR -> Baboonland | Qwen2.5-VL-3B, structured output | 0.191 | 0.072 |

What this means:

- A LoRA adapter trained on only one dataset does not transfer perfectly.
- Larger models usually transfer better.
- Using 8 frames works better than using 4 frames in the tested Qwen setup.
- Structured-output prompting hurt transfer in the tested Qwen setup.
- Transfer from KABR to Baboonland is especially hard.

## 10. Class-collapse-aware training

Because Baboonland is severely imbalanced, the project tested special training
methods to reduce class collapse.

The main idea is to make the training loss care more about rare behavior
classes.

Model: Qwen2.5-VL-3B.

Dataset: Baboonland.

| Variant | Accuracy | Macro-F1 | Change in accuracy | Change in macro-F1 |
|---|---:|---:|---:|---:|
| Baseline LoRA | 0.734 | 0.327 | - | - |
| Inv-sqrt class-weighted loss | **0.752** | **0.374** | +0.018 | +0.047 |
| Focal loss | 0.715 | 0.302 | -0.018 | -0.025 |
| Class-balanced sampler | 0.616 | 0.349 | -0.117 | +0.022 |

What this means:

- The best method is **inverse-square-root class weighting**.
- It improves both accuracy and macro-F1.
- It adds no inference-time cost because it only changes training.
- Focal loss performed worse than the baseline.
- Class-balanced sampling helped macro-F1 somewhat but hurt accuracy too much.

This is one of the project's main method contributions.

## 11. Joint training on both datasets

The project also tests whether one LoRA adapter can be trained on both KABR and
Baboonland together.

This is called **joint cross-dataset LoRA**.

Instead of training one adapter for KABR and another adapter for Baboonland, the
model sees examples from both datasets during training. Each example keeps its
own dataset-specific prompt and label set.

Model: Qwen2.5-VL-3B.

| Training setup | KABR Acc / F1 | Baboonland Acc / F1 |
|---|---:|---:|
| Joint LoRA, no class weighting | **0.863 / 0.514** | **0.759 / 0.377** |
| Joint LoRA, inv-sqrt class weighting | Evaluation pending | Evaluation pending |
| KABR-only LoRA reference | 0.853 / 0.485 | - |
| Baboonland-only LoRA reference | - | 0.734 / 0.327 |
| Baboonland-only LoRA evaluated on KABR | 0.385 / 0.114 | - |
| KABR-only LoRA evaluated on Baboonland | - | 0.397 / 0.155 |

What this means:

- Joint training works very well.
- One jointly trained adapter matches or exceeds separate in-domain adapters on
  both validation sets.
- Joint training is much better than training on one dataset and transferring
  directly to the other.
- This is currently the strongest evidence that multi-dataset VLM adaptation is
  useful for wildlife behavior recognition.

## 12. Current scientific claims

The current results support six main claims.

### Claim 1: Adapted VLMs beat specialist video models

After LoRA fine-tuning, VLMs outperform X3D-L and I3D-L on accuracy on both
KABR and Baboonland.

### Claim 2: Zero-shot VLMs suffer from class collapse

Without fine-tuning, VLMs often overpredict common behaviors and underperform
on rare behaviors. This is especially visible through low macro-F1.

### Claim 3: LoRA fine-tuning gives large gains

Across model backbones and datasets, LoRA fine-tuning substantially improves
both accuracy and macro-F1 over zero-shot prompting.

### Claim 4: Cross-dataset transfer is limited but structured

Adapters trained on one dataset can transfer somewhat to another, but not
nearly as well as in-domain or joint training. Model size and frame count
matter.

### Claim 5: Class-aware loss improves minority-class recognition

Inverse-square-root class-weighted loss improves Baboonland macro-F1 by 4.7
points over baseline Qwen2.5-VL-3B LoRA while also improving accuracy.

### Claim 6: Joint training closes much of the transfer gap

A single LoRA adapter trained jointly on KABR and Baboonland matches or exceeds
single-dataset adapters on both validation sets.

## 13. Possible new NAS direction

The project is now considering adding a neural architecture search, or NAS,
angle.

The recommended version is not to search an entire new VLM architecture. That
would be too expensive and would distract from the current paper. Instead, the
better idea is:

> Search the lightweight adaptation and input-design choices around existing
> VLMs.

This could be called **resource-aware adapter architecture search**.

Possible search variables:

- Which VLM backbone to use.
- How many video frames to give the model.
- What LoRA rank to use.
- Which model modules receive LoRA adapters.
- Whether to use normal loss or class-weighted loss.
- How much latency or memory each configuration costs.

The NAS-style contribution would be:

> A compact search over adapter design and temporal input choices can find
> better accuracy/macro-F1/cost tradeoffs than manually choosing one LoRA
> recipe.

More detail is in [`NAS_BRAINSTORM.md`](NAS_BRAINSTORM.md).

## 14. Current open items

1. Check whether the joint LoRA + inv-sqrt class-weighting evaluations have
   finished.
2. Update the joint training table once those results are available.
3. Rerun the figure-building script so the plots include the newest results.
4. Decide whether the NAS idea should become:
   - a real experiment,
   - a paper contribution,
   - or a future-work discussion.

## 15. One-paragraph summary

This project shows that general-purpose vision-language models, once adapted
with lightweight LoRA fine-tuning, are strong wildlife behavior classifiers.
They outperform specialist video models on two drone-based wildlife datasets,
but zero-shot VLMs suffer from class collapse under class imbalance. The project
introduces class-aware training to reduce this failure mode and joint
cross-dataset LoRA to train one adapter across multiple wildlife datasets. The
current strongest result is that joint LoRA on Qwen2.5-VL-3B matches or exceeds
single-dataset LoRA on both KABR and Baboonland, suggesting that multi-dataset
VLM adaptation is a promising direction for scalable wildlife behavior
recognition.
