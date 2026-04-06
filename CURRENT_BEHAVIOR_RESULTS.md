# Current Behavior Results

This file summarizes the current **clip-level KABR behavior** results on the matched derived-clip validation split (`3761` clips).

All rows below use the same behavior label set and the same scorer. The comparison is therefore meaningful across:

- zero-shot VLMs
- supervised video models (`X3D`)
- VLM post-training methods

## Dataset / Eval Setting

- Task: `behavior`
- Split: derived clip-level `val`
- Number of clips: `3761`
- Main metrics:
  - `accuracy`
  - `macro_f1`

## Current Finished Results

| Family | Model / Method | Accuracy | Macro F1 | Notes |
| --- | --- | ---: | ---: | --- |
| Zero-shot VLM | SmolVLM-256M | 0.2491 | 0.0692 | very weak baseline |
| Zero-shot VLM | Qwen2.5-VL-3B | 0.5057 | 0.2596 | completed full eval |
| Zero-shot VLM | Gemma 3 4B | 0.5504 | 0.2273 | completed full eval |
| Zero-shot VLM | Idefics3-8B-Llama3 | 0.5294 | 0.2314 | resumed full eval completed |
| Zero-shot VLM | LLaVA OneVision Qwen2 7B | 0.5727 | 0.2687 | completed full eval |
| Zero-shot VLM | Qwen2.5-VL-7B | 0.6107 | 0.3168 | strongest zero-shot VLM so far |
| Video baseline | X3D-XS | 0.6703 | 0.3852 | clip-level X3D baseline |
| Video baseline | X3D-S | 0.6307 | 0.3813 | clip-level X3D baseline |
| Video baseline | X3D-M | 0.6668 | 0.4263 | strongest finished X3D so far |
| VLM post-training | Qwen2.5-VL-3B frozen linear | 0.6820 | 0.3542 | minimal adaptation baseline |
| VLM post-training | Qwen2.5-VL-3B frozen MLP | 0.6833 | 0.2954 | stronger accuracy than macro F1 |
| VLM post-training | Qwen2.5-VL-3B distill MLP | 0.6517 | 0.2742 | weaker than frozen baselines |
| VLM post-training | Qwen2.5-VL-3B distill MLP (`alpha=0.7`) | 0.6200 | 0.2134 | did not improve distillation |
| VLM post-training | Qwen2.5-VL-3B LoRA plain (`8` frames) | 0.8535 | 0.4846 | strongest finished behavior result overall |
| VLM post-training | Qwen2.5-VL-3B LoRA plain (`4` frames) | 0.8487 | 0.4985 | best macro F1 so far |

## Current Takeaways

- Zero-shot VLMs are clearly below the best finished X3D baseline on this setup.
- Frozen-feature adaptation recovers some useful behavior signal.
- Distillation has not worked well in the current implementation.
- LoRA is the strongest post-training method we have so far.
- `Qwen2.5-VL-3B` LoRA beats all finished zero-shot VLM and X3D baselines on this matched clip-level setup.

## In Progress / Not Counted As Final Rows

- `X3D-L` is currently still training and does **not** have a final behavior row yet.
- `Gemma 3 4B` LoRA behavior fine-tuning has support implemented and smoke-tested, but the full run has **not** been completed yet.
- The current structured-output behavior LoRA artifact is not being treated as a reliable benchmark row.

## Metric File References

- `Qwen2.5-VL-7B` zero-shot:
  - [qwen25vl7b_behavior_clip_val_full.metrics.json](/home/jhyan/myproj/artifacts/full_eval/qwen25vl7b_behavior_clip_val_full.metrics.json)
- `Gemma 3 4B` zero-shot:
  - [gemma3_4b_behavior_clip_val_full.metrics.json](/home/jhyan/myproj/artifacts/full_eval/gemma3_4b_behavior_clip_val_full.metrics.json)
- `Idefics3-8B-Llama3` zero-shot:
  - [idefics3_8b_behavior_clip_val_full.metrics.json](/home/jhyan/myproj/artifacts/full_eval/idefics3_8b_behavior_clip_val_full.metrics.json)
- `X3D-M`:
  - [x3d_m_behavior_clip.metrics.json](/home/jhyan/myproj/artifacts/x3d_metrics/x3d_m_behavior_clip.metrics.json)
- `Qwen2.5-VL-3B` frozen linear:
  - [predictions.metrics.json](/home/jhyan/myproj/artifacts/paper/behavior/frozen/qwen25vl3b/linear/predictions.metrics.json)
- `Qwen2.5-VL-3B` LoRA plain (`8` frames):
  - [behavior_val_predictions.metrics.json](/home/jhyan/myproj/artifacts/runs/qwen25vl3b_behavior_lora_full/behavior_val_predictions.metrics.json)
- `Qwen2.5-VL-3B` LoRA plain (`4` frames):
  - [behavior_val_predictions.metrics.json](/home/jhyan/myproj/artifacts/paper/behavior/lora/qwen25vl3b/plain_4frames/behavior_val_predictions.metrics.json)
