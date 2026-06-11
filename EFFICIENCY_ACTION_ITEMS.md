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
- [x] Validate BaboonLand prediction-set fallback assignment with deterministic
  half-splits.
- [x] Materialize the tuned BaboonLand prediction-set fallback system.
- [x] Add a class-balanced MiniCPM BaboonLand LoRA config.
- [x] Submit class-balanced MiniCPM BaboonLand LoRA training job.
  - Slurm job: `1411581` (`minicpm-bbl-classbal`)
  - Logs: `logs/minicpm-bbl-classbal.1411581.out`,
    `logs/minicpm-bbl-classbal.1411581.err`
- [x] Monitor class-balanced MiniCPM BaboonLand job to completion.
  - Status: `COMPLETED`, exit code `0:0`, elapsed `07:03:14`
  - Checkpoint:
    `artifacts/runs/baboonland/minicpm_v46_behavior_lora_class_balanced_full/checkpoint`
- [x] Evaluate class-balanced MiniCPM BaboonLand on the full BaboonLand
  validation split.
  - Output:
    `artifacts/full_eval/minicpm_v46_class_balanced_baboonland/minicpm_v46_baboonland_class_balanced_full.jsonl`
  - Accuracy: `0.6621`
  - Macro F1: `0.4011`
- [x] Compare class-balanced MiniCPM BaboonLand as a fallback specialist in the
  tuned BaboonLand prediction-set cascade.
  - Class-balanced as all fallback rows: accuracy `0.6655`, macro F1 `0.4072`
  - Not useful as a full fallback replacement.
- [x] Add and run dual-fallback assignment validation for BaboonLand.
  - Script: `scripts/validate_dual_fallback_assignment.py`
  - Policy: use class-balanced MiniCPM only when routed-row Qwen prediction is
    `Drinking`, `Fighting/Playing`, or `Foraging`; otherwise use standard
    MiniCPM fallback.
  - Full validation metrics: accuracy `0.7722`, macro F1 `0.4448`
  - 20 half-split holdout mean: accuracy `0.7728`, macro F1 `0.4420`
- [x] Sweep BaboonLand prediction-set fallback load/quality tradeoffs.
  - Script: `scripts/sweep_prediction_set_fallback_tradeoff.py`
  - Best macro-F1 policy still routes 5553/6198 rows: accuracy `0.7722`,
    macro F1 `0.4448`
  - Best lower-load accuracy point routes 1852/6198 rows with extra fallback
    predictions `Being Groomed`, `Grooming Somebody`, and `Sleeping`: accuracy
    `0.7759`, macro F1 `0.4275`
  - Materialized low-load metrics:
    `artifacts/full_eval/adaptive_fallback/baboonland_qwen3b_entropy085_dual_minicpm_tradeoff/lowload_metrics.json`
- [x] Bootstrap-validate BaboonLand quality-first vs efficiency-first policies.
  - Script: `scripts/bootstrap_policy_uncertainty.py`
  - Grouping: ID prefix before first colon; `5577` groups, `2000` bootstrap
    samples.
  - Quality-first macro F1 CI: `0.4180..0.4691`, fallback load CI:
    `0.8880..0.9040`
  - Efficiency-first macro F1 CI: `0.4008..0.4503`, fallback load CI:
    `0.2873..0.3106`
  - Paired efficiency-minus-quality differences: accuracy `+0.0037`
    (`p>0=0.9730`), macro F1 `-0.0175` (`p<0=0.9980`), fallback load
    `-0.5971`.
- [x] Test MiniCPM joint adapter as an additional BaboonLand fallback
  specialist.
  - Joint as all fallback rows is worse: accuracy `0.7661`, macro F1 `0.4105`
  - A mapped three-fallback policy improves high-load accuracy: standard
    MiniCPM by default, class-balanced MiniCPM for `Drinking`, and joint
    MiniCPM for `Being Groomed`, `Fighting/Playing`, `Foraging`, `Mounting`,
    `Sleeping`, and `Walking/Running`.
  - Mapped policy: accuracy `0.7778`, macro F1 `0.4425`, fallback load
    `0.8959`.
  - Bootstrap versus quality-macro policy: accuracy diff `+0.0056`
    (`P(diff>0)=0.9990`), macro-F1 diff `-0.0024` with CI crossing zero.
- [ ] Train or select a stronger/lightweight BaboonLand rare-class specialist;
  the dual MiniCPM fallback improves macro F1 but still routes most rows, so it
  is not yet a latency-saving cascade.
- [x] Validate the KABR tuned rare-fallback model assignment on deterministic
  half-splits.
  - Existing balanced route: holdout accuracy `0.8719`, holdout macro F1
    `0.6291`
- [x] Bootstrap-validate KABR quality-first vs efficiency-first policies.
  - `10%` fallback-load policy: accuracy `0.8731`, macro F1 `0.6107`
  - `19.5%` fallback-load policy: accuracy `0.8716`, macro F1 `0.6354`
  - `25%` fallback-load policy: accuracy `0.8684`, macro F1 `0.6537`
  - Grouping: ID prefix before first colon; `3315` groups, `3000` bootstrap
    samples.
- [x] Tune KABR rare-class routing thresholds across representative fallback
  loads.
  - Best current KABR rare macro-F1 point uses thresholds:
    `Graze=-0.04`, `Walk=-0.22`, `Head Up=-0.16`
  - It routes `995/3981` rows and improves rare labels: Auto-Groom F1 `0.2154`,
    Trot F1 `0.5455`, Occluded F1 `0.4151`.

## Eval Mode Plumbing

- [ ] Extend eval scripts or CLI to support inference modes:
  - [ ] `generation`
  - [ ] `label_likelihood`
  - [ ] `adaptive_fallback`
- [ ] Add inference mode to prediction JSONL records.
- [ ] Add label-likelihood confidence fields to prediction records.
- [ ] Add metrics summary fields for confidence distributions.

## Compressed-Frame Inference

- [x] Add fixed frame-count evaluation mode to `scripts/eval_cross_dataset_vlm.py`.
  - CLI: `--max_input_frames`
  - Sampling: uniform over each row's `image_paths`
- [x] Smoke test fixed compressed inputs against 8-frame generation on a
  balanced BaboonLand subset.
  - Subset: `242` rows, up to `20` per class; `Mounting` has only `2`
    validation rows.
  - `1` frame: accuracy `0.3223`, macro F1 `0.2636`, median `212.00 ms`
  - `2` frames: accuracy `0.3595`, macro F1 `0.2945`, median `223.21 ms`
  - `4` frames: accuracy `0.3884`, macro F1 `0.3124`, median `246.57 ms`
  - `8` frames: accuracy `0.4256`, macro F1 `0.3392`, median `302.87 ms`
- [x] Compare fixed compressed inputs against full 8-frame generation on the
  full BaboonLand validation split.
  - Standard MiniCPM BaboonLand LoRA, `6198` validation rows.
  - `2` frames: accuracy `0.7462`, macro F1 `0.3728`, median `225.18 ms`,
    p90 `236.55 ms`
  - `4` frames: accuracy `0.7688`, macro F1 `0.4145`, median `247.11 ms`,
    p90 `252.42 ms`
  - `8` frames: accuracy `0.7709`, macro F1 `0.4212`, median `304.54 ms`,
    p90 `340.65 ms`
  - Tradeoff: `2` frames reduces median latency by about `26.1%` and p90
    latency by about `30.6%`, but loses `0.0247` accuracy and `0.0484`
    macro F1.
- [x] Bootstrap the full BaboonLand fixed-frame comparison.
  - Output:
    `artifacts/full_eval/frame_subsample/minicpm_baboonland_2v4v8_bootstrap.json`
  - Grouping: ID prefix before first colon; `5577` groups, `2000` bootstrap
    samples.
  - `2`-frame accuracy CI: `0.7353..0.7571`, macro F1 CI:
    `0.3490..0.3956`
  - `4`-frame accuracy CI: `0.7585..0.7786`, macro F1 CI:
    `0.3890..0.4390`
  - `8`-frame accuracy CI: `0.7607..0.7811`, macro F1 CI:
    `0.3964..0.4448`
  - Paired `2`-minus-`8` differences: accuracy `-0.0246`
    (`CI -0.0323..-0.0169`), macro F1 `-0.0483`
    (`CI -0.0706..-0.0264`).
  - Paired `4`-minus-`8` differences: accuracy `-0.0021`
    (`CI -0.0069..0.0027`), macro F1 `-0.0067`
    (`CI -0.0214..0.0062`).
- [x] Run the full BaboonLand `4`-frame midpoint evaluation.
- [x] Record latency and accuracy for each full-validation frame count.

## Confidence-Gated Fallback

- [x] Implement first compressed-frame fallback proxy using 4-frame predicted
  label sets.
  - Cheap pass: standard MiniCPM BaboonLand LoRA with `4` uniformly sampled
    frames.
  - Fallback: same adapter with `8` uniformly sampled frames.
  - Sweep artifact:
    `artifacts/full_eval/frame_subsample/frames4_to_8_policy/prediction_set_tradeoff.json`
  - Low-load policy routes 4-frame predictions `Grooming Somebody` and
    `Self-Grooming` to 8-frame fallback: fallback load `0.0323`, accuracy
    `0.7704`, macro F1 `0.4185`, median `247.19 ms`, p90 `253.64 ms`.
  - Accuracy policy routes `Grooming Somebody`, `Occluded`, `Self-Grooming`,
    and `Walking/Running`: fallback load `0.3804`, accuracy `0.7748`, macro F1
    `0.4201`, median `251.97 ms`, p90 `513.09 ms`.
  - Quality policy routes `Grooming Somebody`, `Occluded`, `Self-Grooming`,
    `Sitting/Standing`, and `Walking/Running`: fallback load `0.9450`,
    accuracy `0.7720`, macro F1 `0.4282`, median `552.62 ms`.
- [x] Bootstrap compressed-first 4-frame to 8-frame fallback policies.
  - Output:
    `artifacts/full_eval/frame_subsample/frames4_to_8_policy/cascade_policy_bootstrap.json`
  - Grouping: ID prefix before first colon; `5577` groups, `2000` samples.
  - Accuracy policy vs 8-frame: accuracy diff `+0.0039`
    (`CI 0.0006..0.0071`), macro-F1 diff `-0.0011`
    (`CI -0.0128..0.0096`).
  - Quality policy vs 8-frame: macro-F1 diff `+0.0069`
    (`CI 0.0009..0.0136`), but latency is worse because it reruns most rows.
- [x] Test a deployable frame-agreement gate.
  - Script: `scripts/evaluate_frame_agreement_fallback.py`
  - Rule: run both `2`-frame and `4`-frame MiniCPM; accept the `4`-frame
    prediction when the two predictions agree; otherwise fall back to `8`
    frames.
  - Output:
    `artifacts/full_eval/frame_subsample/frames2_4_agreement_to_8/agreement_metrics.json`
  - Result: fallback load `0.0807`, accuracy `0.7690`, macro F1 `0.4162`,
    median `474.62 ms`, p90 `619.22 ms`.
  - Interpretation: deployable, but not latency-efficient because every row
    pays for both the `2`-frame and `4`-frame passes.
- [x] Smoke test single-pass MiniCPM generation-logit confidence with `4`
  input frames.
  - Script: `scripts/eval_qwen_generation_logit_smoke.py`
  - Output:
    `artifacts/smoke_eval/frame_subsample/minicpm_baboonland_4frames_genlogit_20.jsonl.metrics.json`
  - Result: `20` rows, accuracy `0.7000`, parsed rate `1.0000`, median
    `225.38 ms`, p90 `245.81 ms`.
  - Confidence separation: first-token probability median `0.9855` for
    correct rows vs `0.6190` for incorrect rows; mean-token entropy median
    `0.0338` for correct rows vs `0.2668` for incorrect rows.
- [ ] Replace the validation-set-tuned predicted-label proxy with a deployable
  confidence gate.
- [x] Run full MiniCPM `4`-frame generation-logit confidence on BaboonLand.
  - Output:
    `artifacts/full_eval/frame_subsample/minicpm_baboonland_4frames_genlogit_full.jsonl.metrics.json`
  - Accuracy `0.7688`, macro F1 `0.4145`, parsed rate `1.0000`, median
    `245.95 ms`, p90 `253.94 ms`.
- [x] Sweep deployable single-pass entropy gates for `4`-frame to `8`-frame
  fallback.
  - `0.80` retained-accuracy target: entropy threshold `0.299233`, fallback
    load `0.0910`, system accuracy `0.7693`, macro F1 `0.4149`, median
    `246.69 ms`, p90 `415.74 ms`.
  - `0.85` retained-accuracy target: entropy threshold `0.168531`, fallback
    load `0.2323`, system accuracy `0.7707`, macro F1 `0.4204`, median
    `248.10 ms`, p90 `552.57 ms`.
- [x] Bootstrap the deployable entropy-gated fallback systems.
  - Output:
    `artifacts/full_eval/frame_subsample/minicpm_baboonland_4f_entropy_to_8f_bootstrap.json`
  - `0.85` entropy gate vs 8-frame: accuracy diff `-0.0002`
    (`CI -0.0011..0.0008`), macro-F1 diff `-0.0007`
    (`CI -0.0051..0.0022`).
- [ ] Add fallback logic:
  - [x] run compressed frame pass
  - [x] compute deployable confidence
  - [x] accept if confidence is high
  - [x] rerun full 8-frame input when the proxy gate routes the row
- [x] Sweep confidence thresholds.
- [ ] Report:
  - [x] accuracy
  - [x] macro F1
  - [x] median latency
  - [x] p90 latency
  - [x] fallback rate
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
