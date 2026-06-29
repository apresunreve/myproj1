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
- [x] Add Gemma4 E2B LoRA training support.
  - [x] Add Gemma4-safe LoRA target module regex so PEFT does not attach to
    unsupported vision `Gemma4ClippableLinear` modules.
  - [x] Add Gemma4 KABR and BaboonLand full training configs and launchers.
  - [x] Add Slurm manifests for Gemma4 full training and retry jobs.
- [x] Run Gemma4 E2B LoRA smoke training on this node.
- [x] Run Gemma4 E2B full LoRA training:
  - [x] KABR full LoRA completed on Slurm job `1548607_1`.
  - [x] BaboonLand full LoRA completed on Slurm job `1548641_1`.
- [x] Run Gemma4 E2B full validation on this GPU node.
  - [x] KABR adapter on KABR.
  - [x] BaboonLand adapter on BaboonLand.

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
- [x] Gemma4 E2B KABR LoRA:
  - Accuracy: `0.8392`
  - Macro F1: `0.4862`
  - Parsed rate: `1.0000`
  - Median latency: `705.3 ms`
  - P90 latency: `839.75 ms`
  - Metrics:
    `artifacts/eval/gemma4_e2b_lora_full/kabr_behavior_val_predictions.jsonl.metrics.json`
- [x] Gemma4 E2B BaboonLand LoRA:
  - Accuracy: `0.7354`
  - Macro F1: `0.3273`
  - Parsed rate: `1.0000`
  - Median latency: `773.1 ms`
  - P90 latency: `884.75 ms`
  - Metrics:
    `artifacts/eval/gemma4_e2b_lora_full/baboonland_behavior_val_predictions.jsonl.metrics.json`

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
- [ ] Decide whether Gemma4 E2B belongs in the main baseline table or in a
  broader-model-support appendix.
- [ ] If Gemma4 is included as a main baseline, run a comparable 8-frame eval
  or explicitly label the current row as the 4-frame setup used for training.

## Label-Likelihood Prototype

- [x] Create `src/kabr_vlm/efficiency/`.
- [x] Add `src/kabr_vlm/efficiency/label_likelihood.py`.
- [x] Add `src/kabr_vlm/efficiency/confidence.py`.
- [x] Implement Qwen2.5-VL label scoring first.
- [x] Score each closed-set behavior label directly.
- [x] Return:
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
- [x] Add configurable frame-selection strategies to
  `scripts/eval_cross_dataset_vlm.py`.
  - CLI: `--frame_selection`
  - Strategies: `uniform`, `first`, `last`, `middle`,
    `first_middle_last`, `stride`
  - Prediction records now include `frame_selection` and
    `selected_frame_indices`.
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
- [x] Replace the validation-set-tuned predicted-label proxy with a deployable
  confidence gate.
  - Evidence: full 4-frame MiniCPM generation-logit confidence, global entropy
    gates, class-aware entropy gates, and deterministic held-out validation are
    documented below.
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
- [x] Sweep class-aware entropy thresholds to reduce p90 latency.
  - Script: `scripts/sweep_confidence_fallback_system.py`
  - Sweep:
    `artifacts/full_eval/frame_subsample/minicpm_baboonland_classaware_entropy_sweep.json`
  - Selected thresholds: `Walking/Running=0.30`, `Occluded=0.24`,
    `Grooming Somebody=0.16`, `Self-Grooming=0.30`.
  - Selected system:
    `artifacts/full_eval/frame_subsample/minicpm_baboonland_4f_classaware_entropy_to_8f_system.metrics.json`
  - Result: fallback load `0.0707`, accuracy `0.7736`, macro F1 `0.4208`,
    median `246.57 ms`, p90 `274.71 ms`.
  - Bootstrap vs 8-frame: accuracy diff `+0.0027`
    (`CI -0.0011..0.0064`), macro-F1 diff `-0.0004`
    (`CI -0.0129..0.0112`).
- [x] Validate class-aware threshold selection with deterministic held-out
  folds.
  - Script: `scripts/validate_confidence_fallback_split.py`
  - Output:
    `artifacts/full_eval/frame_subsample/minicpm_baboonland_classaware_entropy_split_validation.json`
  - 5-fold held-out mean: accuracy `0.7726`, macro F1 `0.4081`, fallback
    load `0.0671`, median `246.54 ms`, p90 `288.07 ms`.
  - Interpretation: the low fallback load and accuracy are stable; the
    in-sample macro-F1 improvement should not be claimed as validated.
- [x] Probe higher-load held-out macro-F1 variants.
  - Four-prediction route with caps `0.15`, `0.20`, and `0.25`: selected
    essentially the same policy, held-out macro F1 `0.4082`, fallback load
    `0.0783`, p90 `325.21 ms`.
  - Five-prediction coarse route adding `Sitting/Standing`: cap `0.15` gives
    macro F1 `0.4168`, fallback load `0.1428`, p90 `549.98 ms`; cap `0.25`
    gives macro F1 `0.4185`, fallback load `0.2090`, p90 `553.92 ms`.
  - Interpretation: confidence routing can recover held-out macro F1 only as a
    higher-load quality point, not as the low-p90 efficiency point.
- [x] Add fallback logic:
  - [x] run compressed frame pass
  - [x] compute deployable confidence
  - [x] accept if confidence is high
  - [x] rerun full 8-frame input when the proxy gate routes the row
- [x] Sweep confidence thresholds.
- [x] Report:
  - [x] accuracy
  - [x] macro F1
  - [x] median latency
  - [x] p90 latency
  - [x] fallback rate
  - [x] fallback rate by class
  - [x] accuracy per millisecond

## Frame Selection Policies

- [x] Implement uniform frame sampling baseline.
- [x] Implement first/middle/last frame selection.
- [x] Smoke-test 4-frame selection policies on the same 100-row BaboonLand
  subset.
  - `uniform`: accuracy `0.7100`, macro F1 `0.2388`, median `236.00 ms`,
    p90 `251.91 ms`
  - `first`: accuracy `0.7500`, macro F1 `0.2437`, median `238.66 ms`,
    p90 `245.53 ms`
  - `middle`: accuracy `0.7500`, macro F1 `0.2680`, median `239.34 ms`,
    p90 `242.14 ms`
  - `last`: accuracy `0.7700`, macro F1 `0.2585`, median `235.50 ms`,
    p90 `237.23 ms`
  - `first_middle_last`: accuracy `0.7300`, macro F1 `0.2431`, median
    `238.08 ms`, p90 `254.94 ms`
  - `stride`: accuracy `0.7300`, macro F1 `0.2462`, median `236.31 ms`,
    p90 `238.92 ms`
- [x] Implement simple high-motion frame selection.
  - Strategy: grayscale thumbnail frame-difference score; keep endpoints and
    fill remaining slots from the highest-motion adjacent transitions.
  - Smoke result: accuracy `0.7100`, macro F1 `0.2388`, generation median
    `235.83 ms`, generation p90 `240.38 ms`.
  - Selection overhead: median `17.34 ms`, p90 `35.40 ms`, mean `35.15 ms`.
- [x] Implement redundancy filtering if useful.
  - Strategy: keep endpoints, then greedily add frames with the largest
    minimum thumbnail distance from already selected frames.
  - Smoke result: accuracy `0.7200`, macro F1 `0.2655`, generation median
    `236.14 ms`, generation p90 `240.81 ms`.
  - Selection overhead: median `22.31 ms`, p90 `23.02 ms`, mean `22.01 ms`.
- [x] Compare frame policies at the same average frame count on a smoke subset.
- [x] Run full-validation frame-policy comparison for the best smoke candidates
  (`last` for accuracy and `middle` for macro F1).
  - `last`: accuracy `0.7525`, macro F1 `0.3914`, median `240.64 ms`,
    p90 `254.42 ms`.
  - `middle`: accuracy `0.7633`, macro F1 `0.4111`, median `242.61 ms`,
    p90 `250.64 ms`.
  - Uniform 4-frame remains better: accuracy `0.7688`, macro F1 `0.4145`,
    median `247.11 ms`, p90 `252.42 ms`.
  - Bootstrap vs uniform: `middle` accuracy diff `-0.0055`
    (`CI -0.0118..0.0010`), macro-F1 diff `-0.0034`
    (`CI -0.0185..0.0122`); `last` is significantly worse on both accuracy
    and macro F1.
- [x] Analyze class-specific frame-subset effects from existing MiniCPM
  artifacts.
  - Script: `scripts/analyze_frame_subset_effects.py`
  - Output:
    `artifacts/full_eval/frame_selection/minicpm_baboonland_frame_subset_effects.json`
  - Finding: middle/last frames help grooming-related labels but lose many
    `Walking/Running` and `Occluded` cases, so uniform remains the best
    default compressed-frame policy.

## Adapter-Aware Importance

- [x] Start with Qwen2.5-VL only.
  - Initial smoke uses Qwen2.5-VL-3B BaboonLand B2 generation-logit
    confidence on the first 20 BaboonLand validation rows.
- [x] Measure frame occlusion sensitivity.
  - Initial smoke: materialized 5 original BaboonLand rows plus 40
    leave-one-frame-out variants for Qwen2.5-VL-3B B2.
  - Scripts: `scripts/materialize_frame_ablation_jsonl.py`,
    `scripts/analyze_frame_ablation_effects.py`
  - Result: original and ablated variants were all correct; no single-frame
    omission changed predictions on this easy subset, but later-frame drops
    slightly increased entropy.
- [x] Measure label-likelihood drop when removing frames.
  - Initial Qwen2.5-VL-3B B2 smoke: 2 original BaboonLand rows plus 16
    leave-one-frame-out variants, scored with BaboonLand label likelihood.
  - Result: predictions stayed correct, but dropping later frames reduced
    label-likelihood margin and increased entropy; dropping early frames
    slightly improved margin/lowered entropy on this tiny subset.
- [x] Measure confidence changes under frame subsets.
  - Script: `scripts/analyze_confidence_subset_changes.py`
  - Artifacts:
    `artifacts/smoke_eval/qwen_frame_confidence/qwen3b_baboonland_b2_2frames_20.jsonl.metrics.json`,
    `artifacts/smoke_eval/qwen_frame_confidence/qwen3b_baboonland_b2_4frames_20.jsonl.metrics.json`,
    `artifacts/smoke_eval/qwen_frame_confidence/qwen3b_baboonland_b2_8frames_20.jsonl.metrics.json`,
    `artifacts/smoke_eval/qwen_frame_confidence/qwen3b_baboonland_b2_frame_confidence_changes_20.json`.
  - Result: 4-frame Qwen was best on this smoke subset (`0.7500`
    accuracy); 2-frame and 8-frame both scored `0.7000`.
- [x] Estimate class-specific frame-position priors from existing MiniCPM
  full-validation artifacts.
  - Middle/last frames recover some grooming cases but lose many
    `Walking/Running` and `Occluded` examples.
- [x] Use these signals to guide compressed-frame policy.
  - Current policy direction: keep uniform 4-frame as the default compressed
    pass; avoid global middle/last replacement; use class-aware entropy routing
    to recover hard rows; target future occlusion analysis at threshold-near or
    4-frame/8-frame-disagreement rows instead of easy rows.

## Visual-Token Consolidation

- [x] Count visual tokens per model/input.
  - Script: `scripts/inspect_qwen_visual_tokens.py`
  - Qwen2.5-VL-3B processor-only BaboonLand smoke, first validation row:
    2/4/8 selected frames expand to 308/616/1232 visual placeholder tokens.
  - Per-frame expansion on this sample is 154 image placeholder tokens, with
    `image_grid_thw=(1, 22, 28)` and a 2x2 merge from 616 vision-grid cells.
- [x] Identify Qwen2.5-VL visual token spans.
  - Qwen exposes `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`,
    and `<|video_pad|>` token IDs.
  - For the first BaboonLand row at 2 frames, each image span is 156 prompt
    tokens: start marker, 154 image placeholders, end marker.
- [x] Add random visual-token dropping baseline.
  - Script: `scripts/eval_qwen_random_visual_token_drop_smoke.py`
  - Implementation keeps the prompt sequence intact and zeros a random subset
    of visual placeholder embeddings after Qwen builds multimodal embeddings.
    This avoids breaking Qwen's placeholder-count/visual-feature alignment.
  - First BaboonLand B2 adapter smoke, 1 row, 4 frames: 0% and 50% visual
    embedding dropout both parsed and predicted correctly. The 50% condition
    zeroed 308 of 616 visual placeholder embeddings.
  - Balanced 13-row BaboonLand B2 adapter smoke, 1 row/class, 4 frames,
    drop fractions 0/0.25/0.5: all three settings had `2/13` correct and
    parsed rate `1.0`. Prediction changes vs no-drop were `1/13` at 25%
    dropout and `4/13` at 50% dropout.
  - Balanced-smoke prediction distributions stayed concentrated in
    `Walking/Running` and `Sitting/Standing`, so this is mainly a technical
    perturbation check rather than a reliable quality estimate.
  - Larger balanced smoke, requested 5 rows/class and obtained 62 rows per
    drop setting: no-drop accuracy `0.4032`, 25% dropout `0.3710`, 50%
    dropout `0.3548`, all with parsed rate `1.0`.
  - Prediction changes vs no-drop: 25% dropout changed `7/62` predictions
    and broke 2 previously correct rows; 50% dropout changed `10/62` and
    broke 3 previously correct rows. Neither setting corrected any no-drop
    errors in this sample.
  - Treat this as a technical smoke only; it is not enough to claim quality
    robustness or latency improvement.
- [x] Test similarity-based visual-token merging if model internals permit.
  - Script: `scripts/eval_qwen_visual_token_consolidation_smoke.py`
  - Compares `no_drop`, `random_zero`, `similarity_zero`, and
    `similarity_merge` on the same selected rows.
  - Implementation ranks visual placeholder embeddings by maximum cosine
    similarity to another visual token. `similarity_zero` zeros the most
    redundant embeddings; `similarity_merge` replaces them with the nearest
    kept visual embedding.
  - One-row BaboonLand B2 adapter smoke passed for all policies. The selected
    redundant tokens had high max-cosine similarity (`~0.89-0.92`).
  - Balanced smoke, requested 5 rows/class and obtained 62 rows per setting:
    no-drop accuracy `0.4032`; random-zero `0.3710` at 25% and `0.3387` at
    50%; similarity-zero `0.3871` at 25% and `0.3065` at 50%;
    similarity-merge `0.4194` at 25% and `0.4677` at 50%.
  - At 50%, `similarity_merge` changed `14/62` predictions, fixed 5 no-drop
    errors, and broke 1 no-drop correct row. Fixed gold classes were
    `Sitting/Standing`, `Grooming Somebody`, `Infant-Carrying` (2), and
    `Drinking`; the broken row was `Foraging`.
  - This is an embedding-level perturbation, not true latency compression,
    because sequence length is unchanged. It is still a useful positive signal:
    redundancy-aware merging is much less destructive than random or
    similarity-ranked zeroing and may even regularize this small sample.
- [x] Prototype true shorter-prefix Qwen visual-token pruning.
  - Script: `scripts/eval_qwen_visual_token_prune_smoke.py`
  - Implementation builds Qwen multimodal embeddings, then removes selected
    visual placeholder positions from `input_ids`, `attention_mask`, and
    `inputs_embeds` before generation. This shortens the LLM prefix, though it
    still pays the vision-tower cost before pruning.
  - One-row BaboonLand B2 adapter smoke passed: 50% pruning reduced prompt
    length from 720 to 412 tokens and kept the row correct.
  - Balanced smoke, requested 5 rows/class and obtained 62 rows per setting:
    no-prune accuracy `0.4032`, median latency `191.34 ms`, mean prompt
    tokens `712.45`; random-prune 25% accuracy `0.4194`, median `152.79 ms`,
    prompt tokens `560.32`; random-prune 50% accuracy `0.4032`, median
    `148.20 ms`, prompt tokens `408.18`.
  - Similarity-prune was worse: 25% accuracy `0.3710`, 50% accuracy `0.3226`.
    It broke more no-prune correct rows than it fixed, especially at 50%.
  - Added `similarity_merge_prune`, which merges each pruned high-similarity
    visual embedding into its nearest kept neighbor before deleting it from
    the prefix. This preserved the shorter sequence but did not recover the
    embedding-level `similarity_merge` gain: 25% accuracy `0.3871`, 50%
    accuracy `0.3387`. At 50%, it fixed 1 no-prune error but broke 5
    no-prune correct rows.
  - Interpretation: shorter-prefix pruning is technically viable and can lower
    LLM-side latency, but removing the most similar tokens is too destructive
    even when their information is averaged into kept neighbors. The best next
    direction is cluster-representative pruning or confidence-gated random
    pruning, not high-similarity deletion.
- [x] Add cluster-representative shorter-prefix Qwen visual-token pruning.
  - Script: `scripts/eval_qwen_visual_token_prune_smoke.py`
  - Policy: `cluster_merge_prune`.
  - Implementation groups contiguous Qwen visual placeholder spans, keeps
    k-center representatives within each span, averages dropped embeddings
    into their nearest kept token, then removes the dropped positions from
    `input_ids`, `attention_mask`, and `inputs_embeds`.
  - One-row BaboonLand B2 adapter smoke passed: both 25% and 50% pruning
    preserved the correct prediction. Prompt length dropped from `720` tokens
    to `568` at 25% and `412` at 50%.
  - Balanced smoke, requested 5 rows/class and obtained 62 rows:
    no-prune accuracy `0.4194`, median latency `186.16 ms`, mean prompt
    tokens `712.45`; cluster-merge-prune 25% accuracy `0.4355`, median
    `186.19 ms`, prompt tokens `562.29`; cluster-merge-prune 50% accuracy
    `0.4032`, median `185.70 ms`, prompt tokens `408.18`.
  - At 25%, cluster-merge-prune changed 5/62 predictions, fixed 2 no-prune
    errors (`Grooming Somebody`, `Drinking`), and broke 1 no-prune correct
    row (`Foraging`). At 50%, it fixed 1 `Drinking` row but broke 2
    `Foraging` rows.
  - Interpretation: cluster-representative selection is less destructive than
    high-similarity deletion and gives a small quality signal at 25%, but this
    smoke did not show a median-latency gain despite shorter prompts. Next
    step is confidence-gated pruning or a larger timed run that separates
    vision-tower, prefix, and decode latency.
- [x] Add confidence fields and a post-hoc gated-pruning analyzer.
  - Scripts: `scripts/eval_qwen_visual_token_prune_smoke.py`,
    `scripts/analyze_qwen_prune_gating.py`
  - Pruning records now include generation-logit confidence summaries
    (`mean_token_entropy`, `first_token_prob`, token logprobs/margins) and
    stage timings for encoding, embedding construction, token selection,
    merge, prune, and generation.
  - Confidence-enabled balanced smoke reproduced the same quality pattern:
    no-prune accuracy `0.4194`; cluster-merge-prune 25% accuracy `0.4355`;
    cluster-merge-prune 50% accuracy `0.4032`.
  - Entropy-gated 25% cluster pruning with `Foraging` treated as fragile:
    threshold `0.3326`, pruned `47/62` rows, accuracy `0.4516`, fixed 2
    no-prune errors, broke 0 no-prune correct rows, and reduced mean prompt
    length to `597.23` tokens.
  - First-token-probability gates at comparable quantiles pruned up to `45/62`
    rows but did not change predictions or accuracy from no-prune.
  - Interpretation: no-prune entropy is a better pruning gate than first-token
    probability on this smoke subset. The next test should rerun the entropy
    gate on a larger stratified sample and measure end-to-end latency with
    batching or longer generations, where prompt shortening is more likely to
    matter.
- [x] Run larger stratified Qwen visual-token pruning validation.
  - Output:
    `artifacts/smoke_eval/visual_tokens/qwen3b_b2_visual_token_cluster_prune_conf_balanced20pc.jsonl`
  - Sample: requested 20 rows/class and obtained 242 BaboonLand rows, with
    three settings per row (`726` records total).
  - No-prune: accuracy `0.4421`, median generation latency `271.43 ms`, mean
    prompt tokens `716.13`.
  - Random-prune 25%: accuracy `0.4174`, median generation latency
    `243.96 ms`, mean prompt tokens `563.09`; changed 30 predictions, fixed 8
    no-prune errors, and broke 14 no-prune correct rows.
  - Cluster-merge-prune 25%: accuracy `0.4587`, median generation latency
    `244.60 ms`, mean prompt tokens `565.07`; changed 17 predictions, fixed 9
    no-prune errors, and broke 5 no-prune correct rows.
  - Entropy-gated cluster-merge-prune 25%, excluding no-prune `Foraging`
    predictions as fragile: threshold `0.3235`, pruned `182/242` rows,
    accuracy `0.4628`, median generation latency `244.68 ms`, mean prompt
    tokens `602.29`; fixed 5 no-prune errors and broke 0 no-prune correct
    rows.
  - Fixed labels under the entropy gate: `Being Groomed` (1),
    `Grooming Somebody` (2), `Mutual Grooming` (1), and `Drinking` (1).
  - Interpretation: the larger smoke confirms that cluster-representative
    pruning is better than random pruning and that entropy gating preserves
    the useful fixes while avoiding observed breakages. This is now a stronger
    extension result, though it is still smaller than a full-validation claim.
- [x] Bootstrap the larger Qwen visual-token pruning validation.
  - Script: `scripts/bootstrap_qwen_prune_policy.py`
  - Output:
    `artifacts/smoke_eval/visual_tokens/qwen3b_b2_cluster_prune_policy_bootstrap_balanced20pc.json`
  - Grouping: ID prefix before first colon; `229` groups, `2000` bootstrap
    samples.
  - Entropy-gated cluster-prune 25% observed accuracy `0.4628`; bootstrap
    accuracy mean `0.4629`, CI `0.4042..0.5242`.
  - Paired entropy-gated minus no-prune accuracy difference: mean `+0.0205`,
    CI `+0.0041..+0.0381`, `P(diff>0)=0.9925`.
  - Paired entropy-gated minus no-prune median generation-latency difference:
    mean `-26.40 ms`, CI `-27.45..-24.85`, `P(diff<0)=1.0000`.
  - Paired entropy-gated minus no-prune mean prompt-token difference:
    mean `-113.83`, CI `-122.82..-104.46`.
  - Paired cluster-merge-prune 25% minus random-prune 25% accuracy
    difference: mean `+0.0417`, CI `+0.0122..+0.0756`,
    `P(diff>0)=0.9940`.
  - Interpretation: on this stratified sample, entropy-gated cluster pruning
    has a positive paired accuracy interval versus no-prune and a consistent
    latency/prompt-length reduction. This is strong enough to report as an
    exploratory Qwen extension, while the MiniCPM adaptive visual-compute
    result remains the mature main contribution.
- [x] Run broader 50-per-class Qwen visual-token pruning validation.
  - Output:
    `artifacts/smoke_eval/visual_tokens/qwen3b_b2_visual_token_cluster_prune_conf_balanced50pc.jsonl`
  - Sample: requested 50 rows/class and obtained 573 BaboonLand rows, with
    three settings per row (`1719` records total).
  - No-prune: accuracy `0.4119`, macro F1 `0.3328`, median generation latency
    `267.54 ms`, mean prompt tokens `717.55`.
  - Random-prune 25%: accuracy `0.3892`, macro F1 `0.3154`, median generation
    latency `239.98 ms`, mean prompt tokens `564.16`; fixed 10 no-prune errors
    but broke 23 no-prune correct rows.
  - Cluster-merge-prune 25%: accuracy `0.4188`, macro F1 `0.3492`, median
    generation latency `240.36 ms`, mean prompt tokens `566.15`; fixed 16
    no-prune errors and broke 12 no-prune correct rows.
  - Entropy-gated cluster-merge-prune 25%, excluding no-prune `Foraging`
    predictions as fragile: threshold `0.3192`, pruned `430/573` rows,
    accuracy `0.4154`, macro F1 `0.3419`, median generation latency
    `240.48 ms`, mean prompt tokens `603.88`; fixed 8 no-prune errors and
    broke 6 no-prune correct rows.
- [x] Bootstrap the 50-per-class Qwen visual-token pruning validation.
  - Output:
    `artifacts/smoke_eval/visual_tokens/qwen3b_b2_cluster_prune_policy_bootstrap_balanced50pc.json`
  - Grouping: ID prefix before first colon; `549` groups, `2000` bootstrap
    samples.
  - Cluster-merge-prune 25% minus no-prune: accuracy diff mean `+0.0069`,
    CI `-0.0121..+0.0247`; macro-F1 diff mean `+0.0164`, CI
    `-0.0025..+0.0380`; median generation-latency diff mean `-27.16 ms`, CI
    `-27.37..-26.91`.
  - Entropy-gated cluster-prune 25% minus no-prune: accuracy diff mean
    `+0.0035`, CI `-0.0088..+0.0159`; macro-F1 diff mean `+0.0092`, CI
    `-0.0043..+0.0238`; median generation-latency diff mean `-27.00 ms`, CI
    `-27.29..-26.61`.
  - Cluster-merge-prune 25% minus random-prune 25%: accuracy diff mean
    `+0.0299`, CI `+0.0103..+0.0508`; macro-F1 diff mean `+0.0340`, CI
    `+0.0117..+0.0578`.
  - Interpretation: the broader validation confirms cluster-representative
    pruning is meaningfully better than random pruning at the same prune rate
    and consistently reduces latency/prompt length versus no-prune. The
    quality gain over no-prune is positive but not decisive at this sample
    size, so Qwen pruning should remain an exploratory/secondary contribution
    unless a full-validation run is added.
- [x] Run higher-rate Qwen visual-token pruning sweep.
  - Output:
    `artifacts/smoke_eval/visual_tokens/qwen3b_b2_visual_token_high_prune_sweep_balanced20pc.jsonl`
  - Sample: requested 20 rows/class and obtained 242 BaboonLand rows, with
    no-prune plus random, similarity-merge, and cluster-merge pruning at
    `30%`, `35%`, `40%`, `45%`, and `50%` (`3872` records total).
  - No-prune baseline on this run: accuracy `0.4504`, macro F1 `0.3764`,
    median generation latency `331.57 ms`, mean prompt tokens `716.13`.
  - Best high-prune unconditional setting: `cluster_merge_prune@0.35`,
    accuracy `0.4587`, macro F1 `0.3833`, median generation latency
    `267.37 ms` (`-19.4%`), mean prompt tokens `505.45` (`-29.4%`);
    fixed 8 no-prune errors and broke 6 no-prune correct rows.
  - More aggressive unconditional cluster pruning still reduced latency but
    gave up quality: `0.40` accuracy `0.4380`, `0.45` accuracy `0.4298`,
    and `0.50` accuracy `0.4215`.
  - Random pruning degraded monotonically from accuracy `0.4380` at `30%`
    to `0.4008` at `50%`. Similarity-merge pruning was worse, falling to
    accuracy `0.3678` at `50%`.
  - Entropy-gated cluster pruning with no-prune `Foraging` treated as fragile
    improved the best point further: at `35%` pruning, threshold `0.3231`
    pruned `182/242` rows, accuracy `0.4711`, macro F1 `0.3956`,
    median generation latency `268.49 ms`, and mean prompt tokens `557.35`.
  - Interpretation: higher pruning rates make the latency/prompt tradeoff
    clearer. Cluster-representative pruning is the only tested high-prune
    policy that stays near no-prune quality, with the best smoke-test point
    around `35%`. Above `40%`, latency improves but quality drops enough that
    the result is best treated as a stress test, not a recommended setting.
- [x] Verify 35% Qwen cluster pruning on the full BaboonLand validation set.
  - Output:
    `artifacts/full_eval/visual_tokens/qwen3b_b2_visual_token_cluster35_full.jsonl`
  - Sample: all `6198` BaboonLand validation rows, with no-prune, random-prune
    35%, and cluster-merge-prune 35% (`18594` records total).
  - No-prune: accuracy `0.7249`, macro F1 `0.3591`, median generation latency
    `334.16 ms`, mean prompt tokens `719.32`.
  - Random-prune 35%: accuracy `0.7012`, macro F1 `0.3459`, median generation
    latency `268.44 ms`, mean prompt tokens `504.56`; fixed 122 no-prune
    errors but broke 269 no-prune correct rows.
  - Cluster-merge-prune 35%: accuracy `0.7231`, macro F1 `0.3678`, median
    generation latency `268.91 ms`, mean prompt tokens `507.55`; fixed 127
    no-prune errors and broke 138 no-prune correct rows.
  - Entropy-gated cluster-prune 35%, using the full-run median entropy threshold
    `0.130276` and treating no-prune `Foraging` predictions as fragile:
    pruned `3099/6198` rows, produced identical predictions to no-prune,
    median generation latency `276.00 ms`, and mean prompt tokens `613.37`.
  - Bootstrap output:
    `artifacts/full_eval/visual_tokens/qwen3b_b2_cluster35_full_policy_bootstrap.json`
  - Bootstrap, cluster 35% minus random 35%: accuracy diff `+0.0220`, CI
    `+0.0158..+0.0283`; macro-F1 diff `+0.0221`, CI `+0.0098..+0.0354`.
  - Bootstrap, cluster 35% minus no-prune: accuracy diff `-0.0017`, CI
    `-0.0066..+0.0032`; macro-F1 diff `+0.0089`, CI `-0.0049..+0.0248`;
    median generation-latency diff `-65.25 ms`, CI `-65.36..-65.15`.
  - Bootstrap, entropy-gated cluster 35% minus no-prune: accuracy and macro-F1
    differences exactly `0.0`; median generation-latency diff `-58.08 ms`, CI
    `-58.79..-57.17`; prompt-token diff `-105.98`, CI
    `-108.73..-103.30`.
  - Interpretation: the full run verifies the pruning mechanism as an
    efficiency method, not an accuracy-improvement method. Cluster pruning is
    decisively better than random pruning at the same rate and nearly preserves
    full accuracy while cutting latency. The paper-safe operating point is the
    entropy-gated 50%-load policy because it matched no-prune predictions in
    this full run while reducing latency.
- [x] Run an 8-frame stratified Qwen pruning check to test the frame-count
  mismatch hypothesis.
  - Output:
    `artifacts/smoke_eval/visual_tokens/qwen3b_b2_visual_token_cluster35_8frames_balanced50pc.jsonl`
  - Sample: requested 50 rows/class and obtained `573` BaboonLand rows, with
    no-prune, random-prune 35%, and cluster-merge-prune 35% (`1719` records).
  - No-prune: accuracy `0.4468`, macro F1 `0.3648`, median generation latency
    `504.04 ms`, mean prompt tokens `1333.38`.
  - Random-prune 35%: accuracy `0.4293`, macro F1 `0.3521`, median generation
    latency `389.15 ms`, mean prompt tokens `906.06`; fixed 7 no-prune errors
    and broke 17 no-prune correct rows.
  - Cluster-merge-prune 35%: accuracy `0.4380`, macro F1 `0.3561`, median
    generation latency `386.31 ms`, mean prompt tokens `912.99`; fixed 9
    no-prune errors and broke 14 no-prune correct rows.
  - Entropy-gated cluster-prune 35% at median no-prune entropy threshold
    `0.180715`, with no-prune `Foraging` treated as fragile: pruned
    `287/573` rows, matched no-prune accuracy and macro F1 exactly, changed
    zero predictions, and reduced median generation latency to `421.16 ms`.
  - Bootstrap output:
    `artifacts/smoke_eval/visual_tokens/qwen3b_b2_cluster35_8frames_policy_bootstrap_balanced50pc.json`
  - Bootstrap, cluster 35% minus random 35%: accuracy diff `+0.0086`, CI
    `-0.0071..+0.0261`; macro-F1 diff `+0.0039`, CI
    `-0.0135..+0.0212`; median generation-latency diff `-2.81 ms`.
  - Bootstrap, cluster 35% minus no-prune: accuracy diff `-0.0089`, CI
    `-0.0248..+0.0070`; macro-F1 diff `-0.0092`, CI `-0.0277..+0.0092`;
    median generation-latency diff `-117.73 ms`.
  - Bootstrap, entropy-gated cluster 35% minus no-prune: accuracy and macro-F1
    differences exactly `0.0`; median generation-latency diff `-90.77 ms`, CI
    `-111.69..-77.73`.
  - Interpretation: on balanced 8-frame stress sampling, pruning still gives
    a large latency reduction and the entropy gate preserves no-prune
    predictions. The low absolute accuracy is because the balanced subset
    overweights rare/hard classes; it is not directly comparable to the
    full-distribution PPTX number.
- [x] Run full-dataset 8-frame Qwen pruning validation.
  - Output:
    `artifacts/full_eval/visual_tokens/qwen3b_b2_visual_token_cluster35_8frames_full.jsonl`
  - Sample: all `6198` BaboonLand validation rows, with no-prune,
    random-prune 35%, and cluster-merge-prune 35% (`18594` records total).
  - No-prune: accuracy `0.7519`, macro F1 `0.3770`, median generation latency
    `505.32 ms`, mean prompt tokens `1341.06`.
  - Random-prune 35%: accuracy `0.7402`, macro F1 `0.3790`, median generation
    latency `391.33 ms`, mean prompt tokens `911.07`; fixed 135 no-prune
    errors and broke 207 no-prune correct rows.
  - Cluster-merge-prune 35%: accuracy `0.7531`, macro F1 `0.3839`, median
    generation latency `388.50 ms`, mean prompt tokens `918.06`; fixed 112
    no-prune errors and broke 104 no-prune correct rows.
  - Entropy-gated cluster-prune 35% at the median no-prune entropy threshold
    `0.103202`, with no-prune `Foraging` treated as fragile: pruned
    `3099/6198` rows, changed 1 prediction, accuracy `0.7517`, macro F1
    `0.3770`, and median generation latency `404.84 ms`.
  - Bootstrap output:
    `artifacts/full_eval/visual_tokens/qwen3b_b2_cluster35_8frames_full_policy_bootstrap.json`
  - Bootstrap, cluster 35% minus random 35%: accuracy diff `+0.0130`, CI
    `+0.0075..+0.0184`; macro-F1 diff `+0.0047`, CI
    `-0.0129..+0.0205`; median generation-latency diff `-2.83 ms`.
  - Bootstrap, cluster 35% minus no-prune: accuracy diff `+0.0013`, CI
    `-0.0034..+0.0057`; macro-F1 diff `+0.0069`, CI `-0.0060..+0.0224`;
    median generation-latency diff `-116.83 ms`.
  - Bootstrap, entropy-gated cluster 35% minus no-prune: accuracy diff
    `-0.0002`, CI `-0.0005..0.0`; macro-F1 diff `-0.0000`, CI
    `-0.0001..0.0`; median generation-latency diff `-95.92 ms`.
  - Interpretation: full 8-frame validation confirms the pruning story under
    the paper-style frame count. Cluster-representative pruning is clearly
    better than random pruning at the same rate, is statistically
    indistinguishable from no-prune quality, and saves about 23% median
    generation latency.
- [ ] Port stable hooks to:
  - [ ] LLaVA-OneVision
  - [x] Gemma4-E2B-it
    - Added `scripts/eval_gemma4_soft_token_budget.py`.
    - Added `scripts/analyze_gemma4_budget_cascade.py`.
    - Verified `google/gemma-4-E2B-it` loads in `.venv_minicpm_smoke` with
      `transformers 5.10.0.dev0` and runs on the local L4 GPU at about
      `9.52 GB` allocated in BF16.
    - One-row 4-frame BaboonLand smoke confirmed Gemma4's native
      `max_soft_tokens` knob scales prompt/image tokens and latency:
      budget `70`: `344` prompt tokens, `252` image tokens, median
      generation latency `195.48 ms`; budget `140`: `612` prompt tokens,
      `520` image tokens, `296.12 ms`; budget `280`: `1156` prompt tokens,
      `1064` image tokens, `558.90 ms`.
    - Balanced BaboonLand 5-per-class, 4-frame zero-shot budget smoke passed:
      `artifacts/smoke_eval/visual_tokens/gemma4_e2b_baboonland_soft_budget_balanced5pc_4frames.jsonl`.
      Budget `70`: accuracy `0.1290`, macro F1 `0.0790`, parsed rate
      `0.9839`, median generation latency `195.98 ms`, mean image tokens
      `248.95`.
      Budget `140`: accuracy `0.1129`, macro F1 `0.0525`, parsed rate
      `0.9516`, median generation latency `295.77 ms`, mean image tokens
      `513.71`.
      Budget `280`: accuracy `0.2419`, macro F1 `0.1882`, parsed rate
      `0.9677`, median generation latency `567.01 ms`, mean image tokens
      `1051.13`.
    - Initial entropy-gated cascade analysis accounts for cumulative rerun
      cost. Two-stage `70 -> 140` cascades reduce mean generation latency by
      `39.2%` to `54.5%` versus always using budget `280`; three-stage
      `70 -> 140 -> 280` cascades reduce it by `2.3%` to `50.6%`, depending
      on escalation thresholds. Zero-shot accuracy remains well below trained
      Qwen/MiniCPM baselines on this balanced hard subset. Interpretation:
      Gemma4 is a strong architecture-native efficiency target, but it likely
      needs prompt tuning, LoRA, or a dataset-specific adapter before its
      adaptive-budget results are paper-quality.
    - Added Gemma4-E2B LoRA training configs and launchers for BaboonLand and
      KABR.
      - Local one-sample BaboonLand LoRA smoke passed on this node with
        `max_frames_per_clip=1`.
      - Smoke output:
        `artifacts/smoke/baboonland/gemma4_e2b_behavior_lora/checkpoint`.
      - Smoke training metrics: train loss `0.3893`, trainable parameters
        `24,158,208`, total parameters `5,128,455,712`.
      - Slurm array submitted on `granite-gpu`: job `1548549`, tasks
        `1548549_1` KABR Gemma4-E2B full LoRA and `1548549_2` BaboonLand
        Gemma4-E2B full LoRA.
  - [x] MiniCPM-V 4.6
    - Added `scripts/inspect_minicpm_visual_tokens.py`.
    - Added `scripts/eval_minicpm_visual_token_prune_smoke.py`.
    - KABR one-row MiniCPM LoRA smoke passed at 2 frames and 4 frames with
      no-prune, random-prune 35%, and cluster-merge-prune 35%.
    - 2-frame smoke: 126 visual tokens, 44 pruned at 35%, prompt shortened
      from 200 to 156 tokens; all policies predicted `Head Up` correctly.
    - 4-frame smoke: 252 visual tokens, 88 pruned at 35%, prompt shortened
      from 338 to 250 tokens; all policies predicted `Head Up` correctly.
    - Balanced KABR 5-per-class, 4-frame smoke passed:
      `artifacts/smoke_eval/visual_tokens/minicpm_v46_kabr_visual_token_prune_balanced5pc_4frames.jsonl`.
      No-prune: accuracy `0.3500`, macro F1 `0.2123`, median generation
      latency `188.56 ms`, mean prompt tokens `327.65`.
      Random-prune 35%: accuracy `0.3500`, macro F1 `0.2123`, median
      generation latency `178.66 ms`, mean prompt tokens `242.95`, zero
      prediction changes.
      Cluster-merge-prune 35%: accuracy `0.3500`, macro F1 `0.2103`, median
      generation latency `178.70 ms`, mean prompt tokens `242.95`, one
      prediction change (`Run` to `Walk` on a gold `Trot` row), with no fixed
      or broken no-prune correct rows.
    - Balanced BaboonLand 5-per-class, 4-frame smoke passed:
      `artifacts/smoke_eval/visual_tokens/minicpm_v46_baboonland_visual_token_prune_balanced5pc_4frames.jsonl`.
      The joint validation file provided `62` BaboonLand rows, so the run
      produced `186` records across no-prune, random-prune 35%, and
      cluster-merge-prune 35%.
      No-prune: accuracy `0.3226`, macro F1 `0.2379`, median generation
      latency `307.09 ms`, mean prompt tokens `363.66`.
      Random-prune 35%: accuracy `0.3548`, macro F1 `0.2471`, median
      generation latency `300.53 ms`, mean prompt tokens `276.73`, seven
      prediction changes, three fixed no-prune errors, and one broken
      no-prune correct row.
      Cluster-merge-prune 35%: accuracy `0.3226`, macro F1 `0.2382`, median
      generation latency `301.59 ms`, mean prompt tokens `276.73`, three
      prediction changes, one fixed no-prune error, and one broken no-prune
      correct row.
      Interpretation: MiniCPM shorter-prefix visual-token pruning now works on
      both KABR and BaboonLand adapters. This smoke confirms technical support
      and prompt-token reduction, but the tiny balanced subset should not be
      used as the final latency/quality claim.
    - Next MiniCPM step: run a larger balanced smoke or full validation before
      making any quality/latency claim.

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
