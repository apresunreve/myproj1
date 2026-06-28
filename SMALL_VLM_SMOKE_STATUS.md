# Small VLM Smoke Status

*Created: 2026-05-16.*

This records the first local GPU smoke test for the small VLM candidates in
[`SMALL_VLM_EVAL_PLAN.md`](SMALL_VLM_EVAL_PLAN.md).

Environment:

```text
GPU: NVIDIA RTX 4000 Ada, 19 GB
transformers: 5.3.0
torch: 2.11.0
```

Smoke settings:

```text
max_samples: 1
max_new_tokens: 16
datasets: KABR-Clip and Baboonland-Clip
```

## Results

| Model | KABR smoke | Baboonland smoke | Status |
|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | Loaded, parsed 1/1 | Loaded, parsed 0/1 | Loader works; Baboonland output is verbose and does not reach a label in 16 tokens |
| `LiquidAI/LFM2.5-VL-1.6B` | Loaded, parsed 1/1 | Loaded, parsed 1/1 | Loader works through generic Transformers path |
| `openbmb/MiniCPM-V-4.6` | Failed before generation | Not run after KABR failure | Local Transformers does not recognize `model_type=minicpmv4_6` |
| `google/gemma-4-E2B-it` | Previously failed before generation | Loaded, parsed 1/1 in a new local check | Resolved in `.venv_minicpm_smoke` with Transformers `5.10.0.dev0`; one-row BaboonLand generation and soft-token budget smoke pass |

Generated smoke summary:

```text
artifacts/small_vlm_eval/small_vlm_smoke_summary.csv
artifacts/small_vlm_eval/small_vlm_smoke_summary.md
```

Representative outputs:

```text
Qwen3.5 KABR:      "The behavior being performed is \"standing\"..." -> Head Up
Qwen3.5 Baboonland:"Based on the visual evidence..." -> unparsed
LFM2.5 KABR:       "Feeding." -> Graze
LFM2.5 Baboonland: "Walking/Running" -> Walking/Running
```

## Next actions

1. Run 16-sample smoke for `LiquidAI/LFM2.5-VL-1.6B` on both datasets.
2. For `Qwen/Qwen3.5-0.8B`, test either:
   - larger `max_new_tokens`, or
   - a stricter short-answer prompt wrapper for Baboonland.
3. For MiniCPM-V 4.6 and Gemma4-E2B-it, keep using the newer
   `.venv_minicpm_smoke` Transformers environment for follow-up checks.
4. Do not queue full validation runs until each model has acceptable smoke
   parsed rate.
