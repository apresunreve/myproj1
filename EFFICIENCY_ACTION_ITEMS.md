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
- [x] Run compact latency baselines with the updated evaluator.
- [x] Summarize compact latency baselines.
- [x] Create first Qwen label-likelihood prototype.
- [x] Run one-clip Qwen label-likelihood smoke test.
- [x] Run balanced 40-clip Qwen label-likelihood vs generation comparison.
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

- [x] Run compact latency baselines with the updated evaluator.
  - [x] MiniCPM-V 4.6 KABR LoRA on KABR
  - [x] Qwen2.5-VL KABR LoRA on KABR
  - [x] LLaVA-OV-0.5B KABR LoRA on KABR
  - [x] MiniCPM-V 4.6 BaboonLand LoRA on BaboonLand
  - [x] LLaVA-OV-0.5B BaboonLand LoRA on BaboonLand
- [x] Summarize compact latency baselines.
  - [x] Accuracy
  - [x] Macro F1
  - [x] Parsed rate
  - [x] Median latency
  - [x] P90 latency
  - [x] Peak GPU memory
  - [x] Rows per second
- [ ] Decide whether to rerun full latency baselines.
  - [ ] MiniCPM full evals
  - [ ] Qwen full evals
  - [ ] LLaVA full evals

## Label-Likelihood Prototype

- [x] Create `src/kabr_vlm/efficiency/`.
- [x] Add `src/kabr_vlm/efficiency/label_likelihood.py`.
- [x] Add `src/kabr_vlm/efficiency/confidence.py`.
- [x] Implement Qwen2.5-VL label scoring first.
- [x] Score each closed-set behavior label directly.
- [ ] Return:
  - [x] `label_scores`
  - [x] top-1 label
  - [x] top-2 label
  - [x] confidence margin
  - [x] entropy
- [x] Run a small Qwen label-likelihood smoke test.
- [x] Compare label-likelihood against generation on:
  - [x] accuracy
  - [x] macro F1
  - [x] latency
  - [x] confidence quality
- [x] Add configurable label-batch scoring.
- [x] Test naive all-label batching.
- [ ] Replace naive label batching with an efficient method that avoids
  duplicating the visual input per label.
- [ ] Investigate visual-prefix caching for label scoring.
- [x] Investigate generation-logit confidence as a cheaper fallback signal.
- [x] Analyze whether generation-logit confidence separates correct from
  incorrect predictions on the 40-clip smoke subset.
- [x] Evaluate generation-logit confidence thresholds for abstention or routing
  on the 40-clip smoke subset.
- [x] Collect generation-logit confidence on the full KABR validation split.
- [x] Re-run threshold analysis on the full KABR validation split to check
  stability.
- [x] Implement an adaptive-routing evaluator using the selected confidence
  threshold.
- [x] Design rare-class-aware fallback for confidently wrong Auto-Groom, Trot,
  and Occluded cases.
- [x] Tune rare-class fallback thresholds to reduce routing load while keeping
  rare-label error capture useful.
- [x] Materialize row-level routing outputs for balanced and rare-capture
  fallback policies.
- [x] Compare adaptive routing against a concrete fallback model/scorer on the
  routed subset.
- [x] Evaluate MiniCPM KABR LoRA as a fallback on the balanced routed subset.
- [x] Evaluate Qwen label-likelihood as a stronger fallback on the balanced
  routed subset.
- [x] Evaluate a rare-class-specific fallback for Auto-Groom and
  Trot.
- [x] Inspect/train-data support and prompts for Auto-Groom and Trot; generic
  fallback models are not recovering these classes.
- [x] Submit class-balanced MiniCPM KABR LoRA training job.
- [x] Monitor class-balanced MiniCPM job and evaluate on full KABR validation.
- [x] Compare class-balanced MiniCPM rare-label recall against standard MiniCPM
  and Qwen adaptive routing.
- [x] Build a routing policy that uses class-balanced MiniCPM only as a
  rare-class specialist, not as the default full-dataset predictor.
- [x] Evaluate class-balanced MiniCPM as fallback on the balanced Qwen routed
  subset.
- [x] Evaluate a rare-specialist cascade: Qwen generation accepts high
  confidence rows, class-balanced MiniCPM handles rare-ambiguity rows, and
  standard MiniCPM handles generic low-confidence rows.
- [x] Tune routing decision assignment because Trot recovery needs
  class-balanced fallback on some low-confidence rows too.
- [x] Run deterministic held-out split validation within KABR val for the
  tuned fallback assignment.
- [ ] Validate the tuned prediction-set fallback policy off this validation
  split before treating it as a stable result.
- [x] Validate generation-logit confidence routing on another dataset
  (BaboonLand Qwen2.5-VL-3B).
- [x] Evaluate MiniCPM BaboonLand as a real fallback for BaboonLand
  confidence-routed Qwen rows.
- [x] Analyze BaboonLand accepted rare-label errors after entropy routing.
- [x] Add a generic prediction-metric routing sweep script for dataset-specific
  rare/confusion routing.
- [x] Sweep BaboonLand prediction-specific entropy routing thresholds.
- [x] Evaluate BaboonLand Qwen + MiniCPM systems with prediction-specific
  rare/confusion routing.
- [ ] Train or select a stronger BaboonLand rare-class specialist; generic
  MiniCPM fallback improves macro F1 slightly but still does not beat full
  MiniCPM macro F1.
- [ ] Validate the KABR tuned rare-fallback model assignment on another split
  before treating it as stable.
- [ ] Validate the KABR rare-class routing thresholds on another split before
  treating them as stable.

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
