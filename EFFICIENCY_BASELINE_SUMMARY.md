# Efficiency Baseline Summary

This document records the current generation-eval baseline before adding
label-likelihood, confidence-gated fallback, or frame/token compression.

## Current MiniCPM Results

Full MiniCPM-V 4.6 LoRA evals completed on this node. All rows parsed.

| Adapter | Eval dataset | Rows | Parsed rate | Accuracy | Macro F1 |
|---|---|---:|---:|---:|---:|
| MiniCPM KABR LoRA | KABR | 3981 | 1.0000 | 0.8664 | 0.5162 |
| MiniCPM BaboonLand LoRA | BaboonLand | 6198 | 1.0000 | 0.7712 | 0.4236 |
| MiniCPM joint LoRA | KABR | 3981 | 1.0000 | 0.8644 | 0.5042 |
| MiniCPM joint LoRA | BaboonLand | 6198 | 1.0000 | 0.7643 | 0.3948 |

Outputs are under:

```text
artifacts/full_eval/minicpm_v46/
```

These outputs are generated artifacts and should not be committed.

## KABR Generation Baselines

| Model / adapter | Rows | Parsed rate | Accuracy | Macro F1 | Metrics file |
|---|---:|---:|---:|---:|---|
| MiniCPM-V 4.6 KABR LoRA | 3981 | 1.0000 | 0.8664 | 0.5162 | `artifacts/full_eval/minicpm_v46/minicpm_v46_kabr_lora_on_kabr_full.jsonl.metrics.json` |
| Gemma3-4B KABR LoRA | 3981 | 1.0000 | 0.8644 | 0.5145 | `artifacts/runs/gemma3_4b_behavior_lora_full/behavior_val_predictions.metrics.json` |
| LLaVA-OV-Qwen2-7B KABR LoRA | 3981 | 1.0000 | 0.8631 | 0.5231 | `artifacts/runs/llava_onevision_qwen2_7b_behavior_lora_full/behavior_val_predictions.metrics.json` |
| MiniCPM-V 4.6 joint LoRA | 3981 | 1.0000 | 0.8644 | 0.5042 | `artifacts/full_eval/minicpm_v46/minicpm_v46_joint_lora_on_kabr_full.jsonl.metrics.json` |
| Qwen2.5-VL-7B KABR LoRA | 3981 | 1.0000 | 0.8475 | 0.5034 | `artifacts/runs/qwen25vl7b_behavior_lora_full/behavior_val_predictions.metrics.json` |
| LLaVA-OV-Qwen2-0.5B KABR LoRA | 3981 | 1.0000 | 0.8189 | 0.4468 | `artifacts/runs/llava_onevision_qwen2_0_5b_behavior_lora_full/behavior_val_predictions.metrics.json` |
| MobileVLM-v2-1.7B KABR LoRA | 3981 | 1.0000 | 0.7822 | 0.4307 | `artifacts/runs/mobilevlm_v2_1_7b_behavior_lora_full/behavior_val_predictions.metrics.json` |
| LLaVA-OneVision-1.5-4B KABR LoRA | 3981 | 0.0000 | 0.0000 | 0.0000 | `artifacts/runs/llava_onevision_1_5_4b_behavior_lora_full/behavior_val_predictions.metrics.json` |

The LLaVA-OneVision-1.5-4B row appears to be an eval-path/parsing failure and
should not be treated as a model-quality result without rerunning.

## BaboonLand Generation Baselines

| Model / adapter | Rows | Parsed rate | Accuracy | Macro F1 | Metrics file |
|---|---:|---:|---:|---:|---|
| LLaVA-OV-Qwen2-7B BaboonLand LoRA | 6198 | 1.0000 | 0.7746 | 0.4429 | `artifacts/baboonland_eval/llava_ov_7b_baboonland_lora.metrics.json` |
| MiniCPM-V 4.6 BaboonLand LoRA | 6198 | 1.0000 | 0.7712 | 0.4236 | `artifacts/full_eval/minicpm_v46/minicpm_v46_baboonland_lora_on_baboonland_full.jsonl.metrics.json` |
| LLaVA-OV-Qwen2-0.5B BaboonLand LoRA | 6198 | 1.0000 | 0.7685 | 0.4296 | `artifacts/baboonland_eval/llava_ov_0_5b_baboonland_lora.metrics.json` |
| MiniCPM-V 4.6 joint LoRA | 6198 | 1.0000 | 0.7643 | 0.3948 | `artifacts/full_eval/minicpm_v46/minicpm_v46_joint_lora_on_baboonland_full.jsonl.metrics.json` |
| Gemma3-4B BaboonLand LoRA | 6198 | 1.0000 | 0.7556 | 0.4048 | `artifacts/baboonland_eval/gemma3_4b_baboonland_lora.jsonl.metrics.json` |
| Qwen2.5-VL-3B BaboonLand B2 invsqrt | 6198 | 1.0000 | 0.7519 | 0.3743 | `artifacts/baboonland_eval/qwen25vl3b_bbl_lora_b2_invsqrt.jsonl.metrics.json` |
| Qwen2.5-VL-3B BaboonLand B3 focal | 6198 | 1.0000 | 0.7152 | 0.3016 | `artifacts/baboonland_eval/qwen25vl3b_bbl_lora_b3_focal.jsonl.metrics.json` |
| MobileVLM-v2-1.7B BaboonLand LoRA | 6198 | 1.0000 | 0.7115 | 0.3037 | `artifacts/baboonland_eval/mobilevlm_v2_1_7b_baboonland_lora.metrics.json` |
| Qwen2.5-VL-3B BaboonLand B4 sampler | 6198 | 1.0000 | 0.6162 | 0.3486 | `artifacts/baboonland_eval/qwen25vl3b_bbl_lora_b4_sampler.jsonl.metrics.json` |

## Missing For Efficiency Experiments

The current baseline table is accuracy/F1 oriented. The JSONL evaluator now
records latency-aware fields, but older full-eval metrics need to be regenerated
or post-processed before reporting efficiency claims.

New per-row fields in `scripts/eval_cross_dataset_vlm.py`:

- `latency_ms`
- `num_input_frames`

New aggregate metrics:

- median latency per clip
- p90 latency per clip
- mean/min/max latency per clip
- peak GPU memory
- input-frame summary
- new rows per second

Validation smoke:

```text
artifacts/smoke/latency_metrics_minicpm_v46/minicpm_kabr_2sample.jsonl.metrics.json
```

The smoke confirmed:

```text
n_total=2
n_timed=2
latency_median_ms=1196.95
latency_p90_ms=2106.57
num_input_frames_mean=8
peak_gpu_memory_mb=2843.2
```

The next implementation step is to rerun compact baseline subsets with this
latency-aware evaluator, then prototype label-likelihood and confidence scoring
on one Qwen-family model.
