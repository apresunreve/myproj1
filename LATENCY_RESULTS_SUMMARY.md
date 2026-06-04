# Latency Benchmark Results: 4 vs 8 Frames

This summarizes the KABR zero-shot latency benchmark runs for the six evaluated
VLMs using 4 and 8 frames per clip.

## Run setup

| Setting | Value |
|---|---|
| Dataset | KABR validation subset |
| Task | behavior |
| Mode | zero-shot |
| Timed clips requested | 50 |
| Warmup clips | 3 |
| GPU | RTX PRO 4000 Blackwell, 24 GB |
| 4-frame outputs | `artifacts/latency_4f/` |
| 8-frame outputs | `artifacts/latency/` |

Note: the summary JSON files report `n_clips=53` with `warmup_clips=3`,
although the commands were launched with `--n-clips 50 --warmup-clips 3`.
This suggests the current summary counter includes warmup clips.

## 4-frame results

| Model | Median ms/clip | P90 ms/clip | Accuracy | Peak GPU MB |
|---|---:|---:|---:|---:|
| `llava_ov_0_5b` | 172.74 | 183.31 | 0.3774 | 1,954 |
| `qwen25vl_3b` | 271.61 | 289.36 | 0.3585 | 7,295 |
| `minicpm_v46` | 311.34 | 314.43 | 0.0189 | 2,783 |
| `qwen25vl_7b` | 387.70 | 411.02 | 0.5283 | 17,896 |
| `gemma3_4b` | 718.54 | 746.84 | 0.5094 | 8,649 |
| `llava_ov_7b` | 766.29 | 786.66 | 0.4340 | 17,398 |

## 8-frame results

| Model | Median ms/clip | P90 ms/clip | Accuracy | Peak GPU MB |
|---|---:|---:|---:|---:|
| `llava_ov_0_5b` | 288.88 | 301.03 | 0.3208 | 2,192 |
| `minicpm_v46` | 390.86 | 400.79 | 0.0000 | 2,814 |
| `qwen25vl_3b` | 408.75 | 426.25 | 0.3585 | 7,419 |
| `qwen25vl_7b` | 591.50 | 618.86 | 0.5283 | 17,896 |
| `gemma3_4b` | 1239.27 | 1269.58 | 0.4906 | 9,086 |
| `llava_ov_7b` | 1480.91 | 1507.17 | 0.4340 | 17,398 |

## 4-frame vs 8-frame tradeoff

| Model | 4f median | 8f median | 4f speedup | 4f acc | 8f acc | Acc delta |
|---|---:|---:|---:|---:|---:|---:|
| `llava_ov_0_5b` | 172.74 | 288.88 | 1.67x | 0.3774 | 0.3208 | +0.0566 |
| `qwen25vl_3b` | 271.61 | 408.75 | 1.50x | 0.3585 | 0.3585 | +0.0000 |
| `minicpm_v46` | 311.34 | 390.86 | 1.26x | 0.0189 | 0.0000 | +0.0189 |
| `qwen25vl_7b` | 387.70 | 591.50 | 1.53x | 0.5283 | 0.5283 | +0.0000 |
| `gemma3_4b` | 718.54 | 1239.27 | 1.72x | 0.5094 | 0.4906 | +0.0188 |
| `llava_ov_7b` | 766.29 | 1480.91 | 1.93x | 0.4340 | 0.4340 | +0.0000 |

## Takeaways

- Reducing from 8 frames to 4 frames improved median latency for every model.
- The speedup ranged from `1.26x` for `minicpm_v46` to `1.93x` for
  `llava_ov_7b`.
- Accuracy did not drop for any model on this 50-clip KABR subset; several
  models showed small positive deltas, likely due to subset variance and
  generation instability rather than a guaranteed accuracy improvement.
- `qwen25vl_7b` had the best accuracy in both settings at `0.5283`, with
  median latency improving from `591.50 ms` at 8 frames to `387.70 ms` at
  4 frames.
- The two 7B models used the most memory, around 17.4-17.9 GB, but fit on the
  24 GB GPU in both settings.
