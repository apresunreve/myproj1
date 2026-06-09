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

## Compact Latency Baselines

Compact latency baselines were run on this node with `MAX_SAMPLES=100`,
`MAX_NEW_TOKENS=8`, and the latency-aware JSONL evaluator.

Launcher:

```text
scripts/run_compact_latency_baselines_local.sh
```

Outputs:

```text
artifacts/latency_baselines/compact_100/
```

| Model / adapter | Dataset | Rows | Accuracy | Macro F1 | Median ms | P90 ms | Peak GPU MB | Rows/sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MiniCPM-V 4.6 KABR LoRA | KABR | 100 | 0.8000 | 0.2174 | 263.37 | 330.67 | 2843.2 | 3.2612 |
| Qwen2.5-VL-7B KABR LoRA | KABR | 100 | 0.8900 | 0.3154 | 410.21 | 439.80 | 16447.3 | 2.2496 |
| LLaVA-OV-Qwen2-0.5B KABR LoRA | KABR | 100 | 0.7400 | 0.2048 | 243.94 | 268.14 | 2263.0 | 3.9011 |
| MiniCPM-V 4.6 BaboonLand LoRA | BaboonLand | 100 | 0.7700 | 0.2815 | 339.12 | 423.93 | 2866.8 | 2.7074 |
| LLaVA-OV-Qwen2-0.5B BaboonLand LoRA | BaboonLand | 100 | 0.5300 | 0.1082 | 317.61 | 322.67 | 2263.0 | 3.2122 |

Notes:

- These are compact timing baselines, not full-dataset quality estimates.
- All runs parsed 100/100 rows.
- The KABR compact subset is easier than the full KABR distribution for
  Qwen2.5-VL-7B; use the full-eval table for final accuracy comparisons.

## Missing For Efficiency Experiments

The current baseline table is accuracy/F1 oriented. The JSONL evaluator now
records latency-aware fields. Compact latency baselines are available, but
older full-eval metrics need to be regenerated or post-processed before
reporting full-dataset efficiency claims.

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

The next implementation step is to prototype label-likelihood and confidence
scoring on one Qwen-family model.

## Label-Likelihood Prototype Smoke

Initial Qwen2.5-VL label-likelihood helpers were added under:

```text
src/kabr_vlm/efficiency/
```

The prototype scores each candidate behavior label by summing assistant-label
token log probabilities under the adapter model, then derives confidence from
the closed-set score distribution.

Smoke artifact:

```text
artifacts/smoke/label_likelihood/qwen7b_kabr_1sample.log
```

One-clip Qwen2.5-VL-7B KABR adapter smoke:

```text
gold=Head Up
prediction=Head Up
top2=Browse
margin=4.6356
entropy=0.1110
```

This validates the model-loading and closed-set scoring path. The next step is
to compare label-likelihood against generation on a small multi-class subset
with accuracy, macro F1, latency, margin, and entropy summaries.

Balanced 40-clip KABR comparison:

```text
artifacts/smoke/label_likelihood/qwen7b_balanced_5pc/
```

| Mode | Rows | Accuracy | Macro F1 | Median ms | P90 ms | Notes |
|---|---:|---:|---:|---:|---:|---|
| Generation | 40 | 0.4250 | 0.3118 | 419.69 | 469.26 | Free-form generation + parsing |
| Generation + logit confidence | 40 | 0.4250 | 0.3118 | 419.53 | 466.30 | Same generation path, token-score confidence |
| Label likelihood | 40 | 0.4250 | 0.3232 | 3017.20 | 3114.47 | Sequential 8-label scoring |
| Label likelihood, batched labels | 40 | 0.4250 | 0.3232 | 3697.81 | 3810.49 | One forward batch over 8 labels |

Label-likelihood confidence summary:

```text
margin_median=2.0610
entropy_median=0.6309
```

Generation-logit confidence summary:

```text
first_token_prob_median=0.8744
first_token_prob_correct_median=0.9608
first_token_prob_incorrect_median=0.7209
mean_token_entropy_median=0.3375
mean_token_entropy_correct_median=0.1679
mean_token_entropy_incorrect_median=0.4746
```

Generation-logit threshold sweep:

```text
artifacts/smoke/generation_logit/qwen7b_balanced_5pc/threshold_sweep.json
target_retained_accuracy=0.70
```

| Metric | Threshold | Coverage | Retained accuracy | Routing load | Error capture |
|---|---:|---:|---:|---:|---:|
| first_token_prob | 0.9136 | 0.4250 | 0.7059 | 0.5750 | 0.7826 |
| mean_token_prob | 0.9378 | 0.4250 | 0.7059 | 0.5750 | 0.7826 |
| mean_token_entropy | 0.2790 | 0.4250 | 0.7059 | 0.5750 | 0.7826 |
| first_token_logit_margin | 3.0000 | 0.3500 | 0.7143 | 0.6500 | 0.8261 |
| mean_token_logit_margin | 10.2917 | 0.1250 | 0.8000 | 0.8750 | 0.9565 |

Interpretation:

- Label-likelihood matched generation accuracy on this balanced subset.
- The current prototype is much slower because it runs one forward pass per
  candidate label.
- Naively batching all labels did not improve latency on this GPU because the
  visual input is duplicated for each label in the batch.
- Generation-logit confidence adds no meaningful latency over generation and
  shows a useful smoke-test separation between correct and incorrect examples.
- A 70% retained-accuracy threshold is feasible on the smoke subset, but it
  keeps only 42.5% of clips and routes 57.5% to abstention/fallback.
- The follow-up below tests this on the full KABR validation split.

Full KABR validation generation-logit confidence:

```text
artifacts/full_eval/generation_logit/qwen7b_kabr_full/
rows=3981
accuracy=0.8644
macro_f1=0.5131
parsed_rate=1.0000
latency_median_ms=474.67
latency_p90_ms=583.93
```

Full KABR confidence separation:

```text
first_token_prob_correct_median=0.9515
first_token_prob_incorrect_median=0.7094
mean_token_entropy_correct_median=0.1406
mean_token_entropy_incorrect_median=0.4524
min_token_logit_margin_correct_median=3.5000
min_token_logit_margin_incorrect_median=1.3750
```

Full KABR threshold sweep:

```text
artifacts/full_eval/generation_logit/qwen7b_kabr_full/threshold_sweep.json
target_retained_accuracy=0.90
```

| Metric | Threshold | Coverage | Retained accuracy | Routing load | Error capture |
|---|---:|---:|---:|---:|---:|
| mean_token_logprob | -0.2734 | 0.8980 | 0.9046 | 0.1020 | 0.3685 |
| first_token_prob | 0.6295 | 0.8980 | 0.9029 | 0.1020 | 0.3574 |
| mean_token_prob | 0.7850 | 0.8980 | 0.9027 | 0.1020 | 0.3556 |
| first_token_logit_margin | 1.0000 | 0.8915 | 0.9048 | 0.1085 | 0.3741 |
| mean_token_entropy | 0.5511 | 0.8777 | 0.9004 | 0.1223 | 0.3556 |

Full-validation interpretation:

- The full KABR result is more favorable than the 40-clip smoke because the
  full split reflects the real class distribution and has a much higher base
  accuracy.
- Generation-logit confidence can raise retained accuracy from 86.4% to about
  90.5% while routing about 10.2% of clips.
- This is a viable adaptive-routing candidate: route low-confidence clips to a
  stronger fallback, label-likelihood scorer, or manual review, while accepting
  high-confidence generation outputs directly.

Adaptive routing report:

```text
artifacts/full_eval/generation_logit/qwen7b_kabr_full/adaptive_routing_mean_logprob.metrics.json
metric=mean_token_logprob
threshold=-0.2733916938100265
direction=higher
```

| Decision | Rows | Share | Accuracy/Error |
|---|---:|---:|---:|
| Accept | 3575 | 0.8980 | accepted_accuracy=0.9046 |
| Route | 406 | 0.1020 | routed_error_rate=0.4901 |

The route set catches 199 of 540 generation errors, or 36.9% of errors, while
routing 10.2% of clips. Per-class routing shows an important limitation:
confidence routing does not fix rare classes by itself. Accepted Auto-Groom
and Trot examples still have 0% accepted accuracy, so the next fallback should
target rare-class confusion explicitly, not only generic uncertainty.

Rare-class error analysis:

```text
artifacts/full_eval/generation_logit/qwen7b_kabr_full/routing_error_analysis.json
```

Accepted rare-label errors are 115 of 341 accepted errors, or 33.7%.
The largest confidently accepted rare-label failures are:

| Gold | Predicted | Accepted errors | Median mean-token logprob |
|---|---|---:|---:|
| Occluded | Graze | 55 | -0.0737 |
| Auto-Groom | Graze | 18 | -0.0868 |
| Trot | Walk | 17 | -0.1120 |
| Auto-Groom | Head Up | 10 | -0.1113 |

Rare-class-aware routing simulation:

```text
artifacts/full_eval/generation_logit/qwen7b_kabr_full/rare_class_routing_sim.metrics.json
```

| Routing policy | Coverage | Routing load | Accepted accuracy | Errors routed | Error capture |
|---|---:|---:|---:|---:|---:|
| Confidence only | 0.8980 | 0.1020 | 0.9046 | 199 | 0.3685 |
| Confidence + rare-class ambiguity | 0.6707 | 0.3293 | 0.9554 | 421 | 0.7796 |

The rare-class rule catches 112 rare-label errors and reduces accepted
rare-label errors from 115 to 51, but it routes 905 additional clips. This is
useful if the fallback path is cheap or high-value; otherwise it needs a less
aggressive operating point.

Rare-class threshold sweep:

```text
artifacts/full_eval/generation_logit/qwen7b_kabr_full/rare_class_threshold_sweep_fine.json
```

The sweep keeps the base low-confidence route and adds prediction-specific
`mean_token_logprob` thresholds for common predictions that absorb rare-label
errors: `Graze`, `Walk`, and `Head Up`.

| Policy | Graze threshold | Walk threshold | Head Up threshold | Coverage | Routing load | Accepted accuracy | Total error capture | Rare-label error capture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Confidence only | NA | NA | NA | 0.8980 | 0.1020 | 0.9046 | 0.3685 | 0.2945 |
| Balanced rare fallback | -0.1200 | -0.2400 | -0.1000 | 0.8053 | 0.1947 | 0.9314 | 0.5926 | 0.4479 |
| Rare-capture fallback | -0.0600 | -0.2000 | -0.2400 | 0.8001 | 0.1999 | 0.9268 | 0.5685 | 0.6074 |
| Aggressive rare fallback | mixed | mixed | mixed | 0.6707 | 0.3293 | 0.9554 | 0.7796 | 0.6871 |

Recommended first operating point: balanced rare fallback. It roughly doubles
routing load compared with confidence-only routing, but raises accepted
accuracy from 90.5% to 93.1% and routes 320 of 540 total errors.

Concrete fallback comparison: MiniCPM on balanced routed subset

```text
artifacts/full_eval/adaptive_fallback/minicpm_on_qwen_balanced_routed/
artifacts/full_eval/adaptive_fallback/qwen_balanced_minicpm_system/
```

MiniCPM KABR LoRA was evaluated only on the 775 clips routed by the balanced
rare fallback policy.

| Component | Rows | Accuracy | Macro F1 | Median ms | Notes |
|---|---:|---:|---:|---:|---|
| Qwen accepted set | 3206 | 0.9314 | NA | NA | accepted by routing policy |
| MiniCPM on routed set | 775 | 0.6387 | 0.4332 | 264.37 | fallback only |
| Qwen + MiniCPM adaptive system | 3981 | 0.8744 | 0.5323 | 491.65 | Qwen latency plus fallback for routed rows |

Compared with Qwen generation alone, the adaptive system improves accuracy
from 0.8644 to 0.8744 and macro F1 from 0.5131 to 0.5323. This is a modest
gain. MiniCPM improves some routed cases but still has 0 F1 for Auto-Groom and
Trot on the routed subset, so it is not a sufficient rare-class fallback by
itself.
