# Efficiency Action Items

## Completed

- [x] Read `plan2.md` and map it to an implementation path.
- [x] Verify trained adapter checkpoints on disk.
- [x] Verify the lightweight eval pipeline with smoke tests.
- [x] Create `EFFICIENCY_EVAL_IMPLEMENTATION_PLAN.md`.
- [x] Create MiniCPM-V 4.6 LoRA smoke configs for:
  - [x] KABR
  - [x] BaboonLand
  - [x] Joint KABR+BaboonLand
- [x] Run local MiniCPM-V 4.6 LoRA smoke training on this node.
- [x] Create full MiniCPM-V 4.6 LoRA configs for:
  - [x] KABR
  - [x] BaboonLand
  - [x] Joint KABR+BaboonLand
- [x] Submit MiniCPM full LoRA training jobs to Slurm.
- [x] Confirm MiniCPM Slurm training jobs completed successfully.
- [x] Add MiniCPM adapter eval support in `src/kabr_vlm/vlm_baselines.py`.
- [x] Run Phase 1 real-adapter smoke evals:
  - [x] LLaVA-OV-0.5B KABR adapter on KABR
  - [x] Qwen2.5-VL-7B KABR adapter on KABR
  - [x] MiniCPM KABR adapter on KABR
  - [x] MiniCPM BaboonLand adapter on BaboonLand
  - [x] MiniCPM joint adapter on KABR
  - [x] MiniCPM joint adapter on BaboonLand
- [x] Run full MiniCPM generation evals:
  - [x] KABR adapter on KABR
  - [x] BaboonLand adapter on BaboonLand
  - [x] Joint adapter on KABR
  - [x] Joint adapter on BaboonLand
- [x] Document MiniCPM training and full-eval results.
- [x] Create `EFFICIENCY_BASELINE_SUMMARY.md`.
- [x] Add latency-aware fields to `scripts/eval_cross_dataset_vlm.py`.
- [x] Smoke test latency-aware MiniCPM eval.
- [x] Commit and push current docs, configs, scripts, and eval loader changes.

## Current Baseline Results

- [x] MiniCPM KABR LoRA:
  - Accuracy: `0.8664`
  - Macro F1: `0.5162`
  - Parsed rate: `1.0000`
- [x] MiniCPM BaboonLand LoRA:
  - Accuracy: `0.7712`
  - Macro F1: `0.4236`
  - Parsed rate: `1.0000`
- [x] MiniCPM joint LoRA on KABR:
  - Accuracy: `0.8644`
  - Macro F1: `0.5042`
  - Parsed rate: `1.0000`
- [x] MiniCPM joint LoRA on BaboonLand:
  - Accuracy: `0.7643`
  - Macro F1: `0.3948`
  - Parsed rate: `1.0000`

## Next Action Items

- [ ] Run compact latency baselines with the updated evaluator.
  - [ ] MiniCPM-V 4.6 KABR LoRA on KABR
  - [ ] Qwen2.5-VL KABR LoRA on KABR
  - [ ] LLaVA-OV-0.5B KABR LoRA on KABR
  - [ ] MiniCPM-V 4.6 BaboonLand LoRA on BaboonLand
  - [ ] LLaVA-OV-0.5B BaboonLand LoRA on BaboonLand
- [ ] Summarize compact latency baselines.
  - [ ] Accuracy
  - [ ] Macro F1
  - [ ] Parsed rate
  - [ ] Median latency
  - [ ] P90 latency
  - [ ] Peak GPU memory
  - [ ] Rows per second
- [ ] Decide whether to rerun full latency baselines.
  - [ ] MiniCPM full evals
  - [ ] Qwen full evals
  - [ ] LLaVA full evals

## Label-Likelihood Prototype

- [ ] Create `src/kabr_vlm/efficiency/`.
- [ ] Add `src/kabr_vlm/efficiency/label_likelihood.py`.
- [ ] Add `src/kabr_vlm/efficiency/confidence.py`.
- [ ] Implement Qwen2.5-VL label scoring first.
- [ ] Score each closed-set behavior label directly.
- [ ] Return:
  - [ ] `label_scores`
  - [ ] top-1 label
  - [ ] top-2 label
  - [ ] confidence margin
  - [ ] entropy
- [ ] Run a small Qwen label-likelihood smoke test.
- [ ] Compare label-likelihood against generation on:
  - [ ] accuracy
  - [ ] macro F1
  - [ ] latency
  - [ ] confidence quality

## Eval Mode Plumbing

- [ ] Extend eval scripts or CLI to support inference modes:
  - [ ] `generation`
  - [ ] `label_likelihood`
  - [ ] `adaptive_fallback`
- [ ] Add inference mode to prediction JSONL records.
- [ ] Add label-likelihood confidence fields to prediction records.
- [ ] Add metrics summary fields for confidence distributions.

## Compressed-Frame Inference

- [ ] Add fixed frame-count evaluation modes.
  - [ ] 1 frame
  - [ ] 2 frames
  - [ ] 4 frames
  - [ ] 8 frames
- [ ] Compare fixed compressed inputs against full 8-frame generation.
- [ ] Record latency and accuracy for each frame count.

## Confidence-Gated Fallback

- [ ] Implement compressed-first inference.
- [ ] Add fallback logic:
  - [ ] run compressed frame pass
  - [ ] compute confidence
  - [ ] accept if confidence is high
  - [ ] rerun full 8-frame input if confidence is low
- [ ] Sweep confidence thresholds.
- [ ] Report:
  - [ ] accuracy
  - [ ] macro F1
  - [ ] median latency
  - [ ] p90 latency
  - [ ] fallback rate
  - [ ] fallback rate by class
  - [ ] accuracy per millisecond

## Frame Selection Policies

- [ ] Implement uniform frame sampling baseline.
- [ ] Implement first/middle/last frame selection.
- [ ] Implement simple high-motion frame selection.
- [ ] Implement redundancy filtering if useful.
- [ ] Compare frame policies at the same average frame count.

## Adapter-Aware Importance

- [ ] Start with Qwen2.5-VL only.
- [ ] Measure frame occlusion sensitivity.
- [ ] Measure label-likelihood drop when removing frames.
- [ ] Measure confidence changes under frame subsets.
- [ ] Estimate class-specific frame-position priors.
- [ ] Use these signals to guide compressed-frame policy.

## Visual-Token Consolidation

- [ ] Count visual tokens per model/input.
- [ ] Identify Qwen2.5-VL visual token spans.
- [ ] Add random visual-token dropping baseline.
- [ ] Test similarity-based visual-token merging if model internals permit.
- [ ] Port stable hooks to:
  - [ ] LLaVA-OneVision
  - [ ] MiniCPM-V 4.6

## Commit Hygiene

- [x] Ignore generated artifacts, checkpoints, logs, and virtualenvs.
- [x] Push latest committed work to `origin/main`.
- [ ] Commit future code changes before launching long full runs.
- [ ] Do not commit:
  - [ ] `artifacts/`
  - [ ] `logs/`
  - [ ] model checkpoints
  - [ ] local virtualenvs
  - [ ] Slurm output logs
