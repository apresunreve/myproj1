# Session Handoff: Efficiency and Routing Work

Last updated: 2026-06-11

This handoff is for future sessions continuing the VLM efficiency, routing, and frame-compression work. It summarizes the current repository state, decisions already made, key numerical results, and the next useful actions.

Update on 2026-06-12:

- Added `scripts/inspect_qwen_visual_tokens.py` for processor-only Qwen visual-token/span inspection.
- Added `scripts/eval_qwen_random_visual_token_drop_smoke.py` for a Qwen random visual-token embedding dropout smoke baseline.
- Qwen2.5-VL-3B BaboonLand processor-only smoke on the first validation row:
  - 2 frames: 408 prompt tokens, 308 visual placeholder tokens.
  - 4 frames: 720 prompt tokens, 616 visual placeholder tokens.
  - 8 frames: 1344 prompt tokens, 1232 visual placeholder tokens.
  - Per selected frame: 154 `<|image_pad|>` placeholders on this sample.
  - Each image span is bounded by `<|vision_start|>` and `<|vision_end|>`.
- Qwen2.5-VL-3B BaboonLand B2 adapter random visual-token dropout smoke:
  - 1 row, 4 frames, 616 visual placeholder embeddings.
  - 0% dropout: parsed/correct, median latency 364.37 ms.
  - 50% dropout: zeroed 308 visual placeholder embeddings, parsed/correct, median latency 160.88 ms.
  - Balanced 13-row smoke, 1 row per BaboonLand class, 4 frames, drop fractions 0/0.25/0.5:
    all settings had 2/13 correct with parsed rate 1.0. Prediction changes vs no-drop were 1/13 at 25% dropout and 4/13 at 50% dropout.
  - Larger balanced smoke requested 5 rows per class and produced 62 rows per drop setting:
    no-drop accuracy 0.4032, 25% dropout 0.3710, 50% dropout 0.3548, parsed rate 1.0 for all. Prediction changes vs no-drop were 7/62 at 25% and 10/62 at 50%; neither dropout setting corrected no-drop errors.
  - This is a technical smoke only, not a quality or latency claim. Naive random zeroing mostly hurts quality; the next useful step is similarity- or confidence-aware token consolidation, not random deletion.

## Workspace and Git Hygiene

- Main execution workspace: `/scratch/general/vast/j.yan/myproj1`
- Clean commit/push clone: `/scratch/general/vast/j.yan/myproj1_commit_tmp`
- Use the clean clone for commits and pushes. The main execution workspace has had local run artifacts and unrelated tracked-file churn; do not use it as the source of truth for Git state unless you inspect it carefully.
- Do not commit generated outputs, checkpoints, logs, Slurm stdout/stderr, virtualenvs, or large data files.
- Current pushed branch: `origin/main`
- Recent pushed commits:
  - `f267b82 Record compressed frame policy decision`
  - `7bafeab Add Qwen label-likelihood frame ablation smoke`
  - `536a10e Add Qwen frame ablation smoke`
  - `69d2507 Add Qwen frame confidence smoke`
  - `9ed936e Analyze MiniCPM frame subset effects`
  - `b405c2d Add low-redundancy frame selection smoke`
  - `88e6db2 Update efficiency checklist status`
  - `e676a81 Add high-motion frame selection smoke`

Useful checks:

```bash
cd /scratch/general/vast/j.yan/myproj1_commit_tmp
git status --short
git log --oneline -8

cd /scratch/general/vast/j.yan/myproj1
pgrep -af 'eval|train|python|sbatch|srun'
nvidia-smi
```

At the last status check, no relevant background evaluation/training jobs were running and the local GPU was idle.

## Main Documents

- `EFFICIENCY_ACTION_ITEMS.md`: checklist and running progress tracker.
- `EFFICIENCY_BASELINE_SUMMARY.md`: baseline and efficiency result summary.
- `EFFICIENCY_EVAL_IMPLEMENTATION_PLAN.md`: implementation plan created earlier from `plan2.md`.
- This file: session-level handoff for fast future context.

## Implemented Code

Important scripts and modules now present:

- `scripts/inspect_qwen_visual_tokens.py`
  - Counts Qwen prompt tokens, visual placeholder tokens, image/video grid metadata, and visual marker spans without loading the full model by default.
- `scripts/eval_qwen_random_visual_token_drop_smoke.py`
  - Builds Qwen multimodal embeddings, zeros a random subset of visual placeholder embeddings, and generates through the model for a small baseline.
- `scripts/eval_cross_dataset_vlm.py`
  - Supports fixed frame count via `--max_input_frames`.
  - Supports frame selection strategies: `uniform`, `first`, `last`, `middle`, `first_middle_last`, `stride`, `high_motion`, `low_redundancy`.
  - Writes `frame_selection`, `selected_frame_indices`, `frame_selection_latency_ms`.
  - Summarizes frame-selection latency.
- `scripts/sweep_confidence_fallback_system.py`
  - Sweeps confidence/entropy fallback systems.
- `scripts/validate_confidence_fallback_split.py`
  - Deterministic held-out split validation for tuned fallback policies.
- `scripts/analyze_frame_subset_effects.py`
  - Analyzes class-level changes between frame subsets.
- `scripts/analyze_confidence_subset_changes.py`
  - Analyzes confidence changes between frame subsets.
- `scripts/materialize_frame_ablation_jsonl.py`
  - Builds frame leave-one-out ablation JSONL inputs.
- `scripts/analyze_frame_ablation_effects.py`
  - Summarizes frame-ablation prediction/confidence effects.
- `scripts/eval_qwen_label_likelihood_smoke.py`
  - Supports `--labels`, including BaboonLand labels.

Repo integrity fix:

- `.gitignore` now allows `src/kabr_vlm/data/*.py`.
- Tracked data helper modules:
  - `src/kabr_vlm/data/baboonland_frames.py`
  - `src/kabr_vlm/data/kabr.py`
  - `src/kabr_vlm/data/kabr_frames.py`

## Baseline MiniCPM Results

Full MiniCPM-V 4.6 LoRA evaluations:

| Setting | Accuracy | Macro F1 |
| --- | ---: | ---: |
| KABR adapter on KABR | 0.8664 | 0.5162 |
| BaboonLand adapter on BaboonLand | 0.7712 | 0.4236 |
| Joint adapter on KABR | 0.8644 | 0.5042 |
| Joint adapter on BaboonLand | 0.7643 | 0.3948 |

MiniCPM BaboonLand compact frame baselines:

| Frame Count | Accuracy | Macro F1 | Median Latency | P90 Latency |
| --- | ---: | ---: | ---: | ---: |
| 2-frame uniform | 0.7462 | 0.3728 | 225.18 ms | 236.55 ms |
| 4-frame uniform | 0.7688 | 0.4145 | 247.11 ms | 252.42 ms |
| 8-frame uniform | 0.7709 | 0.4212 | 304.54 ms | 340.65 ms |

Bootstrap 4-frame vs 8-frame:

- Accuracy diff: `-0.0021`, CI `-0.0069..0.0027`
- Macro-F1 diff: `-0.0067`, CI `-0.0214..0.0062`

Interpretation: 4-frame uniform is the current compressed default. It keeps most of the 8-frame quality while reducing median latency substantially.

## MiniCPM Confidence Routing

Full 4-frame MiniCPM generation-logit confidence:

- Accuracy: `0.7688`
- Macro F1: `0.4145`
- Parsed rate: `1.0`
- Median latency: `245.95 ms`
- P90 latency: `253.94 ms`
- First-token probability median:
  - Correct rows: `0.9839`
  - Incorrect rows: `0.8504`
- Mean-token entropy median:
  - Correct rows: `0.0329`
  - Incorrect rows: `0.1711`

Global entropy gate, 4-frame first pass to 8-frame fallback:

| Threshold Quantile | Threshold | Fallback Load | Accuracy | Macro F1 | Median | P90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80 | 0.299233 | 0.0910 | 0.7693 | 0.4149 | 246.69 ms | 415.74 ms |
| 0.85 | 0.168531 | 0.2323 | 0.7707 | 0.4204 | 248.10 ms | 552.57 ms |

Bootstrap for 0.85 global gate vs 8-frame:

- Accuracy diff: `-0.0002`, CI `-0.0011..0.0008`
- Macro-F1 diff: `-0.0007`, CI `-0.0051..0.0022`

Class-aware entropy gate:

- Thresholds:
  - `Walking/Running = 0.30`
  - `Occluded = 0.24`
  - `Grooming Somebody = 0.16`
  - `Self-Grooming = 0.30`
- Fallback load: `0.0707`
- Accuracy: `0.7736`
- Macro F1: `0.4208`
- Median latency: `246.57 ms`
- P90 latency: `274.71 ms`
- Bootstrap vs 8-frame:
  - Accuracy diff: `+0.0027`, CI `-0.0011..0.0064`
  - Macro-F1 diff: `-0.0004`, CI `-0.0129..0.0112`

Held-out split validation, 5-fold, fallback cap `0.10`:

- Mean accuracy: `0.7726`
- Mean macro F1: `0.4081`
- Fallback load: `0.0671`
- Median latency: `246.54 ms`
- P90 latency: `288.07 ms`

Interpretation: low-load accuracy is stable, but the macro-F1 improvement did not validate cleanly. Treat class-aware routing as a latency-efficient deployment policy, not a proven macro-F1 improvement.

Higher-load probes:

- 4-prediction cap `0.15/0.20/0.25`: held-out macro F1 about `0.4082`, load `0.0783`, p90 `325.21 ms`.
- 5-prediction coarse policy adding `Sitting/Standing`:
  - Cap `0.15`: macro F1 `0.4168`, load `0.1428`, p90 `549.98 ms`.
  - Cap `0.25`: macro F1 `0.4185`, load `0.2090`, p90 `553.92 ms`.

Reporting metric already documented:

- Class-aware quality per median second: accuracy/s `3.1376`, macro/s `1.7066`
- 4-frame only: accuracy/s `3.1258`, macro/s `1.6852`
- 8-frame only: accuracy/s `2.5313`, macro/s `1.3831`

## Frame Selection Results

Smoke test on the first 100 BaboonLand rows, MiniCPM 4 frames:

| Strategy | Accuracy | Macro F1 | Median | P90 | Selection Overhead |
| --- | ---: | ---: | ---: | ---: | --- |
| uniform | 0.7100 | 0.2388 | 236.00 ms | 251.91 ms | n/a |
| first | 0.7500 | 0.2437 | 238.66 ms | 245.53 ms | n/a |
| middle | 0.7500 | 0.2680 | 239.34 ms | 242.14 ms | n/a |
| last | 0.7700 | 0.2585 | 235.50 ms | 237.23 ms | n/a |
| first_middle_last | 0.7300 | 0.2431 | 238.08 ms | 254.94 ms | n/a |
| stride | 0.7300 | 0.2462 | 236.31 ms | 238.92 ms | n/a |
| high_motion | 0.7100 | 0.2388 | 235.83 ms | 240.38 ms | median 17.34 ms, p90 35.40 ms |
| low_redundancy | 0.7200 | 0.2655 | 236.14 ms | 240.81 ms | median 22.31 ms, p90 23.02 ms |

Full frame-selection comparison:

| Strategy | Accuracy | Macro F1 | Median | P90 |
| --- | ---: | ---: | ---: | ---: |
| 4-frame uniform | 0.7688 | 0.4145 | 247.11 ms | 252.42 ms |
| 4-frame middle | 0.7633 | 0.4111 | 242.61 ms | 250.64 ms |
| 4-frame last | 0.7525 | 0.3914 | 240.64 ms | 254.42 ms |

Bootstrap vs uniform:

- Middle:
  - Accuracy diff `-0.0055`, CI `-0.0118..0.0010`, `P(diff<0)=0.9450`
  - Macro diff `-0.0034`, CI `-0.0185..0.0122`, `P(diff<0)=0.6700`
- Last:
  - Accuracy diff `-0.0162`, CI `-0.0230..-0.0095`, `P(diff<0)=1.0000`
  - Macro diff `-0.0230`, CI `-0.0400..-0.0063`, `P(diff<0)=0.9985`

Class-specific frame-subset effects:

- 2-frame uniform vs 4-frame uniform:
  - Gains: `Occluded +17`, `Sitting/Standing +11`, `Fighting/Playing +2`
  - Losses: `Walking/Running -123`, `Being Groomed -15`, `Grooming Somebody -13`, `Drinking -10`
- 8-frame uniform vs 4-frame uniform:
  - Gains: `Being Groomed +17`, `Grooming Somebody +12`, `Sitting/Standing +11`
  - Losses: `Walking/Running -21`, `Occluded -9`
- 4-frame middle vs uniform:
  - Gains: `Sitting/Standing +26`, `Being Groomed +11`, `Grooming Somebody +11`
  - Losses: `Walking/Running -77`, `Occluded -4`, `Foraging -3`
- 4-frame last vs uniform:
  - Gains: `Being Groomed +16`, `Sitting/Standing +12`, `Grooming Somebody +3`
  - Losses: `Walking/Running -105`, `Occluded -20`, `Fighting/Playing -4`

Interpretation: use a class/confidence-conditional policy, not a global replacement of uniform sampling.

## Qwen Adapter-Aware Checks

Qwen2.5-VL-3B BaboonLand B2 generation-logit frame confidence, first 20 rows:

| Frames | Accuracy | Median Latency | Entropy Median | First-Token Prob Median | Mean Token Prob Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.7000 | 196.37 ms | 0.1182 | 0.9305 | 0.9768 |
| 4 | 0.7500 | 254.54 ms | 0.1108 | 0.9341 | 0.9780 |
| 8 | 0.7000 | 410.25 ms | 0.1154 | 0.9309 | 0.9769 |

Paired vs 4-frame:

- 2-frame change rate `0.1000`, net correct gain `-1`, mean entropy delta `-0.0443`, first-token probability delta `+0.0779`.
- 8-frame change rate `0.1500`, net correct gain `-1`, mean entropy delta `-0.0115`, first-token probability delta `+0.0033`.

Interpretation: more frames did not monotonically improve this small Qwen sample, and confidence did not monotonically correlate with frame count.

Qwen leave-one-frame-out generation-logit smoke:

- First 5 BaboonLand validation rows.
- Inputs: 5 originals plus 40 ablations.
- Original accuracy: `1.0000`
- Ablated variant accuracy: `1.0000`
- Prediction changes: `0`
- Entropy deltas by dropped index relative to original:
  - `0: -0.0014`
  - `1: +0.0036`
  - `2: -0.0015`
  - `3: +0.0002`
  - `4: -0.0020`
  - `5: +0.0013`
  - `6: +0.0034`
  - `7: +0.0048`

Interpretation: no prediction changes on easy rows; future occlusion should target hard rows, such as threshold-near rows or 4-frame/8-frame disagreement rows.

Qwen label-likelihood frame-ablation smoke:

- First 2 BaboonLand rows.
- Inputs: 2 originals plus 16 leave-one-out variants.
- Original accuracy: `1.0000`
- Ablated variant accuracy: `1.0000`
- Median latency: `4086.71 ms`
- Entropy/margin deltas by dropped frame:
  - `drop0: entropy -0.0295, margin +0.3750`
  - `drop1: entropy -0.0190, margin +0.3096`
  - `drop2: entropy -0.0160, margin +0.1833`
  - `drop3: entropy +0.0008, margin -0.0696`
  - `drop4: entropy +0.0011, margin -0.0696`
  - `drop5: entropy +0.0132, margin -0.1905`
  - `drop6: entropy +0.0189, margin -0.2386`
  - `drop7: entropy +0.0078, margin -0.1263`

Interpretation: later frame drops reduce label-likelihood margin and increase entropy, but label-likelihood is too slow for deployment in the current implementation.

## Current Policy Decision

- Keep uniform 4-frame sampling as the default compressed pass.
- Do not replace uniform sampling globally with middle, last, high-motion, or low-redundancy selection.
- Use single-pass generation-logit entropy for deployable confidence routing.
- Use class-aware 4-frame to 8-frame fallback if prioritizing low p90 latency with near-8-frame accuracy.
- Treat label-likelihood and frame occlusion as analysis tools, not the deployed routing signal.
- For future frame-importance work, target hard rows rather than easy first rows.

## Remaining To-Do List

Near-term:

- [ ] Count visual tokens per model/input.
- [ ] Identify Qwen2.5-VL visual token spans.
- [ ] Add a random visual-token dropping baseline.
- [ ] Test similarity-based visual-token merging if model internals permit.
- [ ] Port stable visual-token hooks to LLaVA-OneVision.
- [ ] Port stable visual-token hooks to MiniCPM-V 4.6.
- [ ] Build a hard-row frame-occlusion set from threshold-near or 4-frame/8-frame-disagreement rows.
- [ ] Validate whether hard-row occlusion identifies actionable frame-position or token-importance structure.

Evaluation and productization:

- [ ] Extend the unified eval CLI to support explicit inference modes:
  - `generation`
  - `label_likelihood`
  - `adaptive_fallback`
- [ ] Add inference mode to prediction JSONL records.
- [ ] Add label-likelihood confidence fields to prediction records.
- [ ] Add metrics summary fields for confidence distributions.
- [ ] Validate tuned prediction-set fallback policy off the current validation split if a separate split or external validation is available.
- [ ] Decide whether full latency baselines need to be rerun for MiniCPM, Qwen, and LLaVA.

Lower priority:

- [ ] Replace naive label batching with an efficient method that avoids duplicate visual input.
- [ ] Investigate visual-prefix caching for label scoring.
- [ ] Train or select a stronger lightweight BaboonLand rare-class specialist.

Commit hygiene:

- [ ] Commit future code changes before launching long full runs.
- [ ] Continue excluding `artifacts/`, `logs/`, model checkpoints, virtualenvs, and Slurm logs.

## Recommended Next Step

Start visual-token consolidation on Qwen2.5-VL:

1. Add a small inspection script that runs a single Qwen sample and records tokenized input length, visual token locations/spans, number of image/video tokens, and generated latency.
2. Run it on one KABR and one BaboonLand example with 2, 4, and 8 frames.
3. Use that output to decide whether random visual-token dropping can be implemented at the input/embedding boundary without breaking model internals.

This is the best next step because frame-count compression is already near its practical limit for MiniCPM, and the current plan's remaining novelty is visual-token consolidation rather than additional global frame-selection heuristics.
