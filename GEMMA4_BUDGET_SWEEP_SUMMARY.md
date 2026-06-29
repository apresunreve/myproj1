# Gemma4 E2B Visual Budget Sweep

Date: 2026-06-29

This sweep tests whether Gemma4-E2B LoRA can trade visual compute for latency by
using the native Gemma4 `max_soft_tokens` processor budget. This is input-side
visual-token reduction: lower budgets resize/patchify fewer image patches and
insert fewer `<image_soft_token>` embeddings into the language-model prefix.

Artifacts are generated outputs and should not be committed:

```text
artifacts/smoke_eval/visual_tokens/gemma4_e2b_lora_budget_sweep_compact/
logs/gemma4_e2b_lora_budget_sweep_compact/
```

## Implementation Status

- [x] Extend `scripts/eval_gemma4_soft_token_budget.py` to support LoRA adapters.
- [x] Add KABR and BaboonLand label-space support.
- [x] Add summary JSON writing.
- [x] Add compact local sweep launcher:
  `scripts/run_gemma4_e2b_budget_sweep_compact_local.sh`.
- [x] Run compact balanced sweep for:
  - [x] KABR, frames `1, 2, 4`, budgets `70, 140, 280`
  - [x] BaboonLand, frames `1, 2, 4`, budgets `70, 140, 280`

## Compact Sweep Results

KABR uses `5` examples per class, `40` rows total. BaboonLand uses up to `5`
examples per class, `62` rows total because one class has only two validation
examples.

| Dataset | Frames | Soft tokens | N | Acc | Macro-F1 | Gen p50 ms | Enc p50 ms | Mean image tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KABR | 1 | 70 | 40 | 0.1250 | 0.0321 | 124.15 | 3.22 | 63.00 |
| KABR | 1 | 140 | 40 | 0.2500 | 0.1389 | 133.83 | 4.28 | 130.00 |
| KABR | 1 | 280 | 40 | 0.2750 | 0.1754 | 181.54 | 6.49 | 266.00 |
| KABR | 2 | 70 | 40 | 0.2500 | 0.1329 | 132.88 | 4.92 | 122.85 |
| KABR | 2 | 140 | 40 | 0.2750 | 0.1602 | 176.39 | 7.36 | 253.50 |
| KABR | 2 | 280 | 40 | 0.3250 | 0.2409 | 305.04 | 11.81 | 518.70 |
| KABR | 4 | 70 | 40 | 0.2750 | 0.1713 | 172.13 | 9.20 | 242.55 |
| KABR | 4 | 140 | 40 | 0.3250 | 0.1949 | 293.73 | 13.91 | 500.50 |
| KABR | 4 | 280 | 40 | 0.3750 | 0.2593 | 640.02 | 21.73 | 1024.10 |
| BaboonLand | 1 | 70 | 62 | 0.1452 | 0.0525 | 232.07 | 3.64 | 63.00 |
| BaboonLand | 1 | 140 | 62 | 0.1613 | 0.0593 | 242.00 | 4.76 | 130.00 |
| BaboonLand | 1 | 280 | 62 | 0.1613 | 0.0731 | 288.82 | 6.89 | 266.00 |
| BaboonLand | 2 | 70 | 62 | 0.1935 | 0.0946 | 238.26 | 5.28 | 124.98 |
| BaboonLand | 2 | 140 | 62 | 0.1935 | 0.1087 | 281.00 | 7.94 | 257.90 |
| BaboonLand | 2 | 280 | 62 | 0.2903 | 0.2128 | 416.54 | 12.53 | 527.71 |
| BaboonLand | 4 | 70 | 62 | 0.1935 | 0.1090 | 275.82 | 9.59 | 248.95 |
| BaboonLand | 4 | 140 | 62 | 0.2097 | 0.1415 | 401.58 | 13.79 | 513.71 |
| BaboonLand | 4 | 280 | 62 | 0.4194 | 0.3289 | 741.31 | 21.88 | 1051.13 |

## Interpretation

- Gemma4 visual budget pruning works mechanically and gives a clear latency
  response.
- For KABR, `2` frames / `280` soft tokens is the best compact cheap point:
  it keeps `0.3250` accuracy versus `0.3750` for `4` frames / `280` tokens,
  while reducing median generation latency from `640.02 ms` to `305.04 ms`.
- For BaboonLand, the best compact cheap point is weaker. `2` frames / `280`
  tokens reaches `0.2903` accuracy versus `0.4194` for `4` frames / `280`
  tokens. The task appears more sensitive to full temporal/visual context.
- The very low budgets (`70` tokens) are useful for latency characterization,
  but too weak as stand-alone predictors on these balanced compact sets.

## Routing Probe

Using `4` frames / `280` soft tokens as the full Gemma4 fallback on the same
compact rows:

| Dataset | Cheap setting | Cheap acc | Full acc | Agreement with full | Best first-token-prob route |
|---|---|---:|---:|---:|---|
| KABR | 2 frames / 280 tokens | 0.3250 | 0.3750 | 0.7750 | Full acc with 0.2000 fallback load |
| KABR | 4 frames / 140 tokens | 0.3250 | 0.3750 | 0.8750 | Full acc with 0.3250 fallback load |
| BaboonLand | 2 frames / 280 tokens | 0.2903 | 0.4194 | 0.7100 | Full acc with 0.7100 fallback load |
| BaboonLand | 4 frames / 140 tokens | 0.2097 | 0.4194 | 0.5970 | Full acc with 0.5650 fallback load |

The KABR result is promising: a confidence-routed `2f/280` cheap pass may save
about half the median generation latency on accepted rows while routing only a
small fraction to full Gemma4 on this compact set. BaboonLand likely needs a
stronger confidence policy or a different fallback model; otherwise the fallback
load is too high to be an efficiency win.

## Next Actions

- [ ] Run a larger KABR sweep for `2f/280` cheap pass plus `4f/280` Gemma4
  fallback, with first-token probability and entropy routing thresholds.
- [ ] Test BaboonLand with `2f/280` cheap Gemma4 plus MiniCPM fallback instead
  of full Gemma4 fallback.
- [ ] Compare Gemma4 budget routing against the existing Qwen/MiniCPM adaptive
  routing story on accuracy, macro-F1, fallback load, and latency.
- [ ] If the larger KABR result holds, add Gemma4 as evidence that adaptive
  visual compute applies beyond Qwen/MiniCPM.
