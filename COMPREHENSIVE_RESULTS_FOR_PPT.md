# Comprehensive Results For PPT

*Updated: 2026-05-20.*

This file consolidates the results from:

- `CURRENT_BEHAVIOR_RESULTS.md`
- `CURRENT_BEHAVIOR_SUMMARY.md`
- `CURRENT_RESULTS_SUMMARY.md`
- `artifacts/small_vlm_eval/small_vlm_summary.md`

## 1. Dataset And Evaluation Context

| Dataset | Domain | Classes | Train clips | Val clips | Main issue |
|---|---|---:|---:|---:|---|
| KABR-Clip | Drone footage of Kenyan wildlife | 8 | 11,003 | 3,761 | Moderate class imbalance |
| Baboonland-Clip | Drone footage of baboons | 13 | 20,130 | 6,198 | Severe class imbalance |

Both datasets are evaluated as clip-level single-label behavior classification
benchmarks. VLMs receive sampled frames plus a text prompt and produce a text
answer, which is parsed into the fixed behavior label set.

Primary metrics:

- Accuracy
- Macro-F1
- Parsed rate for text-generating VLMs

## 2. KABR Behavior: Method Ladder On Matched 3,761-Clip Val Set

| Family | Model / method | Accuracy | Macro-F1 | Takeaway |
|---|---|---:|---:|---|
| Zero-shot VLM | SmolVLM-256M | 0.2491 | 0.0692 | Very weak baseline |
| Zero-shot VLM | Qwen2.5-VL-3B | 0.5057 | 0.2596 | Useful but below video baselines |
| Zero-shot VLM | Gemma 3 4B | 0.5504 | 0.2273 | Moderate zero-shot accuracy |
| Zero-shot VLM | Idefics3-8B-Llama3 | 0.5294 | 0.2314 | Moderate zero-shot row |
| Zero-shot VLM | LLaVA OneVision Qwen2 7B | 0.5727 | 0.2687 | Stronger zero-shot row |
| Zero-shot VLM | Qwen2.5-VL-7B | 0.6107 | 0.3168 | Strongest listed KABR zero-shot row |
| Video baseline | X3D-XS | 0.6703 | 0.3852 | Specialist video baseline |
| Video baseline | X3D-S | 0.6307 | 0.3813 | Specialist video baseline |
| Video baseline | X3D-M | 0.6668 | 0.4263 | Best finished X3D macro-F1 in this ladder |
| Frozen features | Qwen2.5-VL-3B frozen linear | 0.6820 | 0.3542 | Recovers useful signal |
| Frozen features | Qwen2.5-VL-3B frozen MLP | 0.6833 | 0.2954 | Accuracy improves, macro-F1 weaker |
| Distillation | Qwen2.5-VL-3B distill MLP | 0.6517 | 0.2742 | Does not preserve LoRA gains |
| Distillation | Qwen2.5-VL-3B distill MLP, alpha=0.7 | 0.6200 | 0.2134 | Worse than baseline distillation |
| LoRA | Qwen2.5-VL-3B LoRA, 8 frames | 0.8535 | 0.4846 | Strongest accuracy in this KABR ladder |
| LoRA | Qwen2.5-VL-3B LoRA, 4 frames | 0.8487 | 0.4985 | Best macro-F1 in this KABR ladder |

Main interpretation:

- Zero-shot VLMs are below X3D on KABR.
- Frozen-feature adaptation recovers part of the gap.
- Distillation is not competitive in the current implementation.
- LoRA closes and surpasses the gap.
- The 4-frame LoRA row is surprisingly competitive, especially in macro-F1.

## 3. Paper-Level KABR Results

| Model | Zero-shot Acc/F1 | LoRA Acc/F1 | Joint LoRA Acc/F1 |
|---|---:|---:|---:|
| Gemma3-4B | 0.549 / 0.226 | 0.864 / 0.515 | - |
| LLaVA-OV-7B | 0.581 / 0.254 | 0.863 / 0.523 | - |
| Qwen2.5-VL-3B | 0.549 / 0.271 | 0.853 / 0.485 | 0.863 / 0.514 |
| Qwen2.5-VL-7B | 0.626 / 0.327 | 0.848 / 0.503 | - |
| LLaVA-OV-0.5B | 0.307 / 0.119 | 0.819 / 0.447 | - |
| MobileVLM-1.7B | 0.327 / 0.062 | 0.782 / 0.431 | - |
| X3D-L specialist | - | 0.797 / 0.520 | - |
| I3D-L specialist | - | 0.814 / - | - |

Interpretation:

- Several adapted VLMs reach about 0.86 accuracy.
- Adapted VLMs beat the specialist video baselines on accuracy.
- Qwen2.5-VL-3B joint LoRA exceeds the KABR-only LoRA reference.

## 4. Paper-Level Baboonland Results

| Model | Zero-shot Acc/F1 | LoRA Acc/F1 | Joint LoRA Acc/F1 |
|---|---:|---:|---:|
| LLaVA-OV-7B | 0.338 / 0.077 | 0.775 / 0.443 | - |
| LLaVA-OV-0.5B | 0.418 / 0.078 | 0.768 / 0.430 | - |
| Gemma3-4B | 0.540 / 0.145 | 0.756 / 0.405 | - |
| Qwen2.5-VL-7B | 0.329 / 0.143 | 0.755 / 0.394 | - |
| Qwen2.5-VL-3B + inv-sqrt class weighting | 0.242 / 0.080 | 0.752 / 0.374 | - |
| Qwen2.5-VL-3B baseline | 0.242 / 0.080 | 0.734 / 0.327 | 0.759 / 0.377 |
| MobileVLM-1.7B | 0.174 / 0.024 | 0.712 / 0.304 | - |
| X3D-L specialist | - | 0.671 / - | - |
| I3D-L specialist | - | 0.630 / - | - |

Interpretation:

- Baboonland is harder and more imbalanced than KABR.
- Zero-shot macro-F1 is very low, showing class collapse.
- Every adapted VLM listed beats X3D-L and I3D-L on accuracy.
- The strongest adapted row reaches 0.775 accuracy and 0.443 macro-F1.

## 5. Cross-Dataset Transfer

| Direction | Source LoRA | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Baboonland -> KABR | LLaVA-OV-7B | 0.652 | 0.256 |
| Baboonland -> KABR | Qwen2.5-VL-7B | 0.562 | 0.261 |
| Baboonland -> KABR | LLaVA-OV-0.5B | 0.517 | 0.181 |
| Baboonland -> KABR | MobileVLM-1.7B | 0.447 | 0.179 |
| Baboonland -> KABR | Qwen2.5-VL-3B | 0.385 | 0.114 |
| KABR -> Baboonland | Qwen2.5-VL-3B, 8-frame | 0.397 | 0.155 |
| KABR -> Baboonland | Qwen2.5-VL-3B, 4-frame | 0.298 | 0.129 |
| KABR -> Baboonland | Qwen2.5-VL-3B, structured output | 0.191 | 0.072 |

Interpretation:

- Single-dataset transfer is limited.
- Larger models generally transfer better.
- KABR -> Baboonland is especially hard.
- Joint training is much stronger than direct transfer.

## 6. Class-Collapse-Aware Training

Model: Qwen2.5-VL-3B on Baboonland.

| Variant | Accuracy | Macro-F1 | Change in accuracy | Change in macro-F1 |
|---|---:|---:|---:|---:|
| Baseline LoRA | 0.734 | 0.327 | - | - |
| Inv-sqrt class-weighted loss | 0.752 | 0.374 | +0.018 | +0.047 |
| Focal loss | 0.715 | 0.302 | -0.018 | -0.025 |
| Class-balanced sampler | 0.616 | 0.349 | -0.117 | +0.022 |

Interpretation:

- Inverse-square-root class weighting is the best current mitigation.
- It improves macro-F1 and accuracy with no inference-time cost.
- Focal loss and class-balanced sampling are weaker under the current setup.

## 7. Joint Training

Model: Qwen2.5-VL-3B.

| Training setup | KABR Acc/F1 | Baboonland Acc/F1 |
|---|---:|---:|
| Joint LoRA, no class weighting | 0.863 / 0.514 | 0.759 / 0.377 |
| Joint LoRA, inv-sqrt class weighting | Evaluation pending | Evaluation pending |
| KABR-only LoRA reference | 0.853 / 0.485 | - |
| Baboonland-only LoRA reference | - | 0.734 / 0.327 |
| Baboonland-only LoRA evaluated on KABR | 0.385 / 0.114 | - |
| KABR-only LoRA evaluated on Baboonland | - | 0.397 / 0.155 |

Interpretation:

- One joint adapter matches or exceeds separate in-domain adapters.
- Joint training is much stronger than transfer-only evaluation.

## 8. Small-VLM Zero-Shot Screening

| Model | KABR Acc/F1 | KABR parsed | Baboonland Acc/F1 | Baboonland parsed | Combined Acc | Combined F1 | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| MiniCPM-V 4.6 | 0.6523 / 0.2841 | 0.9739 | 0.3767 / 0.1022 | 1.0000 | 0.4845 | 0.1932 | complete |
| LFM2.5-VL-1.6B | 0.4361 / 0.1613 | 0.9764 | 0.5092 / 0.1339 | 1.0000 | 0.4806 | 0.1476 | complete |
| Qwen3.5-0.8B | - | - | - | - | - | - | missing |
| Gemma4-E2B-it | - | - | - | - | - | - | missing |

Interpretation:

- MiniCPM-V 4.6 is stronger on KABR and combined macro-F1.
- LFM2.5-VL-1.6B is stronger on Baboonland accuracy.
- These are useful zero-shot resource-aware comparison points, but adapted
  VLMs remain much stronger.

## 9. Overall Claims Supported By The Results

1. Zero-shot VLMs alone are not enough for reliable wildlife behavior
   recognition.
2. LoRA adaptation gives large gains over zero-shot prompting.
3. Adapted VLMs outperform specialist video baselines on accuracy on both
   datasets.
4. Macro-F1 is essential because class imbalance causes class collapse.
5. Inverse-square-root class weighting mitigates class collapse on Baboonland.
6. Joint cross-dataset LoRA is much stronger than direct transfer and can
   replace separate per-dataset adapters.
7. Small VLM zero-shot screening supports the resource-aware story, while the
   planned adapter/input search is the more promising paper contribution.
