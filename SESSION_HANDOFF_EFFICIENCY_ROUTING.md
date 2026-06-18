# Session Handoff: Efficiency and Routing Work

Last updated: 2026-06-15

This handoff is for future sessions continuing the VLM efficiency, routing, and frame-compression work. It summarizes the current repository state, decisions already made, key numerical results, and the next useful actions.

## Paper Big Picture

This project is about efficient vision-language modeling for wildlife behavior recognition. Modern VLMs can recognize animal behavior from ecological video frames, but full multimodal inference is expensive and often unnecessary. The paper adapts VLMs with LoRA on KABR and BaboonLand, then studies how to allocate visual compute only where it matters. The central idea is **speculative visual inference / adaptive visual compute**: run a cheap compressed visual pass first, estimate confidence from normal generation logits, and use additional visual evidence only for uncertain or class-sensitive examples. This is similar in spirit to speculative decoding, but the speculation is over visual evidence rather than text-token drafts.

Main contribution framing:

- LoRA adaptation and evaluation of multimodal foundation models for wildlife behavior recognition on KABR, BaboonLand, and joint settings.
- A compressed-frame inference study showing that 4 uniformly sampled frames can preserve most of the quality of 8-frame inference while reducing latency.
- A lightweight confidence signal from generation logits, avoiding expensive label-likelihood scoring for deployment-time routing.
- A class-aware adaptive fallback policy that routes only selected uncertain or fragile predictions to fuller visual input.
- A latency-quality evaluation protocol reporting accuracy, macro F1, fallback load, median/p90 latency, bootstrap uncertainty, and class-level behavior.
- An exploratory Qwen2.5-VL visual-token pruning track that tests whether adaptive visual compute can move inside the VLM prefix itself.

## Current Snapshot

- Strongest paper-ready efficiency contribution: **speculative visual inference / adaptive visual compute**.
- Deployable policy direction: run a cheap compressed 4-frame visual pass first, estimate confidence from generation logits, and rerun with fuller visual input only for uncertain or class-sensitive cases.
- Best mature result remains MiniCPM BaboonLand 4-frame to 8-frame confidence routing:
  - 4-frame uniform nearly matches 8-frame quality at lower median latency.
  - Class-aware entropy fallback routes about 7% of rows to 8 frames.
  - Accuracy/macro F1 are comparable to full 8-frame while keeping p90 latency much lower than broad fallback policies.
- Qwen visual-token work is now a useful extension track:
  - Qwen2.5-VL expands 4 selected frames to about 616 visual placeholder tokens.
  - True shorter-prefix pruning works technically and reduces LLM-side prompt length/latency.
  - Blind or high-similarity deletion is not reliable enough yet. Cluster-representative pruning is now implemented and has a modest 25% smoke signal, but confidence-gated pruning or larger timed runs are still needed.

At the latest check on 2026-06-15, no relevant training/eval/Slurm jobs were running and the GPU was idle.

## Fast Resume Summary

- Current best mature story: MiniCPM speculative visual inference / adaptive visual compute.
- Current experimental extension: Qwen2.5-VL visual-token pruning after vision-feature construction.
- Latest Qwen visual-token pruning status:
  - 242-row stratified smoke: entropy-gated 25% `cluster_merge_prune` improved accuracy over no-prune by mean `+0.0205`, CI `+0.0041..+0.0381`.
  - 573-row broader stratified smoke: cluster-merge-prune 25% improved macro F1 over no-prune by mean `+0.0164`, CI `-0.0025..+0.0380`, and reduced median generation latency by about `27 ms`; the quality CI crosses zero.
  - 573-row cluster-merge-prune 25% clearly beats random-prune 25%: accuracy diff mean `+0.0299`, CI `+0.0103..+0.0508`; macro-F1 diff mean `+0.0340`, CI `+0.0117..+0.0578`.
- Caution: Qwen token pruning is now a solid exploratory extension, especially versus random pruning, but not a full-validation main claim.
- Immediate next work: either run a full BaboonLand B2 Qwen validation if GPU time permits, or consolidate the paper with MiniCPM adaptive visual compute as the main contribution and Qwen token pruning as a secondary/exploratory extension.

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
- Added `scripts/eval_qwen_visual_token_consolidation_smoke.py` for similarity-aware embedding-level token consolidation.
  - The script compares `no_drop`, `random_zero`, `similarity_zero`, and `similarity_merge`.
  - It ranks redundant visual placeholder embeddings by maximum cosine similarity to another visual token.
  - Balanced 62-row BaboonLand B2 smoke: no-drop accuracy 0.4032; random-zero 0.3710 at 25% and 0.3387 at 50%; similarity-zero 0.3871 at 25% and 0.3065 at 50%; similarity-merge 0.4194 at 25% and 0.4677 at 50%.
  - At 50%, similarity-merge changed 14/62 predictions, fixed 5 no-drop errors, and broke 1 no-drop correct row.
  - This is a positive signal for redundancy-aware consolidation, but it is not yet real latency compression because Qwen still processes the same sequence length.
- Added `scripts/eval_qwen_visual_token_prune_smoke.py` for true shorter-prefix Qwen visual-token pruning.
  - It builds multimodal embeddings, then removes selected visual placeholder positions from `input_ids`, `attention_mask`, and `inputs_embeds` before generation.
  - One-row smoke passed: 50% pruning reduced the prompt from 720 to 412 tokens and preserved the correct prediction.
  - Balanced 62-row smoke: no-prune accuracy 0.4032, median latency 191.34 ms, mean prompt tokens 712.45; random-prune 25% accuracy 0.4194, median 152.79 ms, prompt tokens 560.32; random-prune 50% accuracy 0.4032, median 148.20 ms, prompt tokens 408.18.
  - Similarity-prune hurt quality: 0.3710 at 25% and 0.3226 at 50%. Deleting the most similar tokens is not the right policy; similarity-aware merge/replacement is a better direction.
  - Added true shorter-prefix `similarity_merge_prune`, which averages pruned high-similarity embeddings into nearest kept embeddings before deletion. It still hurt quality: 0.3871 at 25% and 0.3387 at 50%. Conclusion: embedding-level merge is a useful diagnostic, but shorter-prefix merge-prune needs cluster representatives or confidence gating rather than deleting the most similar tokens.
- Added `cluster_merge_prune` to `scripts/eval_qwen_visual_token_prune_smoke.py`.
  - It groups contiguous visual placeholder spans, keeps k-center representatives within each span, merges dropped embeddings into nearest kept representatives, and then removes dropped positions from the prefix.
  - One-row BaboonLand B2 adapter smoke passed at 25% and 50%.
  - Balanced 62-row smoke: no-prune accuracy 0.4194, median latency 186.16 ms, mean prompt tokens 712.45; cluster-merge-prune 25% accuracy 0.4355, median 186.19 ms, prompt tokens 562.29; cluster-merge-prune 50% accuracy 0.4032, median 185.70 ms, prompt tokens 408.18.
  - At 25%, cluster-merge-prune changed 5/62 predictions, fixed 2 no-prune errors (`Grooming Somebody`, `Drinking`), and broke 1 no-prune correct row (`Foraging`). At 50%, it fixed 1 `Drinking` row and broke 2 `Foraging` rows.
  - Interpretation: 25% cluster pruning is more promising than high-similarity deletion, but the current smoke did not show a median-latency gain despite shorter prompts. Treat it as a policy signal, not a finished efficiency result.
- Added confidence/timing fields to `scripts/eval_qwen_visual_token_prune_smoke.py` and added `scripts/analyze_qwen_prune_gating.py`.
  - Each pruning record now includes generation-logit confidence summaries and stage timings for encoding, embedding, selection, merge, prune, and generation.
  - Confidence-enabled balanced 62-row smoke reproduced the unconditional pruning pattern: no-prune accuracy 0.4194; cluster-merge-prune 25% accuracy 0.4355; cluster-merge-prune 50% accuracy 0.4032.
  - Post-hoc entropy-gated 25% cluster pruning with `Foraging` excluded as a fragile predicted class pruned 47/62 rows, reached accuracy 0.4516, fixed 2 no-prune errors, broke 0 no-prune correct rows, and reduced mean prompt length to 597.23 tokens.
  - First-token-probability gates pruned up to 45/62 rows but did not change predictions or accuracy from no-prune on this subset.
  - Interpretation: entropy is the better pruning gate in this smoke. This is still not a final latency claim because median generation latency was effectively unchanged in the small unbatched run.
- Ran a larger stratified Qwen visual-token pruning validation.
  - Output: `artifacts/smoke_eval/visual_tokens/qwen3b_b2_visual_token_cluster_prune_conf_balanced20pc.jsonl`
  - Sample: requested 20 rows/class and obtained 242 BaboonLand rows, with 726 generated records across no-prune, random-prune 25%, and cluster-merge-prune 25%.
  - No-prune: accuracy 0.4421, median generation latency 271.43 ms, mean prompt tokens 716.13.
  - Random-prune 25%: accuracy 0.4174, median generation latency 243.96 ms, mean prompt tokens 563.09; fixed 8 no-prune errors but broke 14 correct rows.
  - Cluster-merge-prune 25%: accuracy 0.4587, median generation latency 244.60 ms, mean prompt tokens 565.07; fixed 9 no-prune errors and broke 5 correct rows.
  - Entropy-gated cluster-merge-prune 25%, excluding no-prune `Foraging` predictions as fragile: threshold 0.3235, pruned 182/242 rows, accuracy 0.4628, median generation latency 244.68 ms, mean prompt tokens 602.29; fixed 5 no-prune errors and broke 0 correct rows.
  - Interpretation: the larger smoke confirms that cluster-representative pruning beats random pruning and that entropy gating removes the observed breakages. This is the strongest Qwen visual-token result so far.
- Bootstrapped the larger Qwen visual-token pruning validation.
  - Script: `scripts/bootstrap_qwen_prune_policy.py`
  - Output: `artifacts/smoke_eval/visual_tokens/qwen3b_b2_cluster_prune_policy_bootstrap_balanced20pc.json`
  - Grouping: ID prefix before first colon; 229 groups, 2000 bootstrap samples.
  - Entropy-gated cluster-prune 25% accuracy mean 0.4629, CI 0.4042..0.5242.
  - Paired entropy-gated minus no-prune accuracy diff: mean +0.0205, CI +0.0041..+0.0381, P(diff>0)=0.9925.
  - Paired entropy-gated minus no-prune median generation-latency diff: mean -26.40 ms, CI -27.45..-24.85, P(diff<0)=1.0000.
  - Paired entropy-gated minus no-prune prompt-token diff: mean -113.83 tokens, CI -122.82..-104.46.
  - Paired cluster-merge-prune 25% minus random-prune 25% accuracy diff: mean +0.0417, CI +0.0122..+0.0756, P(diff>0)=0.9940.
  - Interpretation: the uncertainty analysis supports the qualitative claim that entropy-gated cluster pruning improves this stratified Qwen sample while reducing prompt length and generation latency. Treat as an exploratory extension unless a full-validation run is added.
- Ran a broader 50-per-class Qwen visual-token pruning validation.
  - Output: `artifacts/smoke_eval/visual_tokens/qwen3b_b2_visual_token_cluster_prune_conf_balanced50pc.jsonl`
  - Sample: requested 50 rows/class and obtained 573 BaboonLand rows, with 1719 generated records.
  - No-prune: accuracy 0.4119, macro F1 0.3328, median generation latency 267.54 ms, mean prompt tokens 717.55.
  - Random-prune 25%: accuracy 0.3892, macro F1 0.3154, median generation latency 239.98 ms, mean prompt tokens 564.16; fixed 10 no-prune errors and broke 23 correct rows.
  - Cluster-merge-prune 25%: accuracy 0.4188, macro F1 0.3492, median generation latency 240.36 ms, mean prompt tokens 566.15; fixed 16 no-prune errors and broke 12 correct rows.
  - Entropy-gated cluster-prune 25%: threshold 0.3192, pruned 430/573 rows, accuracy 0.4154, macro F1 0.3419, median generation latency 240.48 ms, mean prompt tokens 603.88; fixed 8 no-prune errors and broke 6 correct rows.
- Bootstrapped the 50-per-class Qwen visual-token pruning validation.
  - Output: `artifacts/smoke_eval/visual_tokens/qwen3b_b2_cluster_prune_policy_bootstrap_balanced50pc.json`
  - Grouping: ID prefix before first colon; 549 groups, 2000 bootstrap samples.
  - Cluster-merge-prune 25% minus no-prune: accuracy diff mean +0.0069, CI -0.0121..+0.0247; macro-F1 diff mean +0.0164, CI -0.0025..+0.0380; median generation-latency diff mean -27.16 ms, CI -27.37..-26.91.
  - Entropy-gated cluster-prune 25% minus no-prune: accuracy diff mean +0.0035, CI -0.0088..+0.0159; macro-F1 diff mean +0.0092, CI -0.0043..+0.0238; median generation-latency diff mean -27.00 ms, CI -27.29..-26.61.
  - Cluster-merge-prune 25% minus random-prune 25%: accuracy diff mean +0.0299, CI +0.0103..+0.0508; macro-F1 diff mean +0.0340, CI +0.0117..+0.0578.
  - Interpretation: the broader validation confirms cluster-representative pruning is better than random pruning and consistently reduces generation latency/prompt length versus no-prune. Quality versus no-prune trends positive but is not decisive, so this remains a secondary/exploratory contribution.

## Workspace and Git Hygiene

- Main execution workspace: `/scratch/general/vast/j.yan/myproj1`
- Clean commit/push clone: `/scratch/general/vast/j.yan/myproj1_commit_tmp`
- Use the clean clone for commits and pushes. The main execution workspace has had local run artifacts and unrelated tracked-file churn; do not use it as the source of truth for Git state unless you inspect it carefully.
- Do not commit generated outputs, checkpoints, logs, Slurm stdout/stderr, virtualenvs, or large data files.
- Current pushed branch: `origin/main`
- Recent pushed commits:
  - `4650bb5 Add confidence gated Qwen pruning analysis`
  - `b705d1c Add Qwen cluster visual token pruning`
  - `fd9de75 Refresh efficiency session handoff`
  - `275f13c Test Qwen similarity merge pruning`
  - `d53913e Add Qwen shorter-prefix token prune smoke`
  - `352d91d Add Qwen similarity token consolidation smoke`
  - `c1d03c7 Record larger Qwen token-drop smoke`
  - `5581504 Record Qwen token-drop balanced smoke`
  - `8b8342b Add balanced sampling to Qwen token-drop smoke`
  - `3129bcb Add Qwen visual token inspection smokes`
  - `b41d3bd Add efficiency session handoff`

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

## Latest Artifacts

- Confidence-enabled Qwen cluster-prune balanced smoke:
  `artifacts/smoke_eval/visual_tokens/qwen3b_b2_visual_token_cluster_prune_conf_balanced65.jsonl`
- Larger stratified Qwen cluster-prune validation:
  `artifacts/smoke_eval/visual_tokens/qwen3b_b2_visual_token_cluster_prune_conf_balanced20pc.jsonl`
- Larger stratified Qwen cluster-prune metrics:
  `artifacts/smoke_eval/visual_tokens/qwen3b_b2_visual_token_cluster_prune_conf_balanced20pc.jsonl.metrics.json`
- Larger entropy-gated 25% cluster-prune sweep:
  `artifacts/smoke_eval/visual_tokens/qwen3b_b2_cluster_prune_conf_gate_entropy25_balanced20pc.json`
- Larger first-token-probability gated 25% cluster-prune sweep:
  `artifacts/smoke_eval/visual_tokens/qwen3b_b2_cluster_prune_conf_gate_firstprob25_balanced20pc.json`
- Larger Qwen pruning policy bootstrap:
  `artifacts/smoke_eval/visual_tokens/qwen3b_b2_cluster_prune_policy_bootstrap_balanced20pc.json`
- 50-per-class Qwen cluster-prune validation:
  `artifacts/smoke_eval/visual_tokens/qwen3b_b2_visual_token_cluster_prune_conf_balanced50pc.jsonl`
- 50-per-class Qwen cluster-prune metrics:
  `artifacts/smoke_eval/visual_tokens/qwen3b_b2_visual_token_cluster_prune_conf_balanced50pc.jsonl.metrics.json`
- 50-per-class entropy-gated cluster-prune sweep:
  `artifacts/smoke_eval/visual_tokens/qwen3b_b2_cluster_prune_conf_gate_entropy25_balanced50pc.json`
- 50-per-class first-token-probability gated cluster-prune sweep:
  `artifacts/smoke_eval/visual_tokens/qwen3b_b2_cluster_prune_conf_gate_firstprob25_balanced50pc.json`
- 50-per-class Qwen pruning policy bootstrap:
  `artifacts/smoke_eval/visual_tokens/qwen3b_b2_cluster_prune_policy_bootstrap_balanced50pc.json`
- Metrics for that smoke:
  `artifacts/smoke_eval/visual_tokens/qwen3b_b2_visual_token_cluster_prune_conf_balanced65.jsonl.metrics.json`
- Entropy-gated 25% cluster-prune sweep:
  `artifacts/smoke_eval/visual_tokens/qwen3b_b2_cluster_prune_conf_gate_entropy25.json`
- First-token-probability gated 25% cluster-prune sweep:
  `artifacts/smoke_eval/visual_tokens/qwen3b_b2_cluster_prune_conf_gate_firstprob25.json`

Useful command to rerun the larger validation:

```bash
cd /scratch/general/vast/j.yan/myproj1_commit_tmp
.venv/bin/python scripts/eval_qwen_visual_token_prune_smoke.py \
  --val-jsonl /scratch/general/vast/j.yan/myproj1/data/processed/baboonland_behavior_val.jsonl \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter-path /scratch/general/vast/j.yan/myproj1/artifacts/runs/baboonland/qwen25vl3b_behavior_lora_b2_invsqrt/checkpoint \
  --samples-per-class 20 \
  --max-input-frames 4 \
  --prune-fractions 0.25 \
  --policies no_prune,random_prune,cluster_merge_prune \
  --max-new-tokens 12 \
  --output-jsonl /scratch/general/vast/j.yan/myproj1/artifacts/smoke_eval/visual_tokens/qwen3b_b2_visual_token_cluster_prune_conf_balanced20pc.jsonl

.venv/bin/python scripts/analyze_qwen_prune_gating.py \
  --input-jsonl /scratch/general/vast/j.yan/myproj1/artifacts/smoke_eval/visual_tokens/qwen3b_b2_visual_token_cluster_prune_conf_balanced20pc.jsonl \
  --policy cluster_merge_prune \
  --prune-fraction 0.25 \
  --metric mean_token_entropy \
  --direction le \
  --quantiles 0.25,0.5,0.75 \
  --fragile-labels Foraging \
  --output-json /scratch/general/vast/j.yan/myproj1/artifacts/smoke_eval/visual_tokens/qwen3b_b2_cluster_prune_conf_gate_entropy25_balanced20pc.json
```

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
- `scripts/eval_qwen_visual_token_consolidation_smoke.py`
  - Compares random visual zeroing against similarity-ranked zeroing and similarity-ranked merge/replacement.
- `scripts/eval_qwen_visual_token_prune_smoke.py`
  - Shortens Qwen's LLM prefix by removing selected visual placeholder embeddings after vision feature construction.
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

- [x] Count visual tokens per model/input.
- [x] Identify Qwen2.5-VL visual token spans.
- [x] Add a random visual-token dropping baseline.
- [x] Test similarity-based visual-token merging if model internals permit.
- [x] Prototype true shorter-prefix visual-token pruning.
- [ ] Convert similarity-aware embedding merge into a true shorter-sequence or lower-compute implementation.
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

Choose between **fuller Qwen validation** and **paper write-up consolidation**:

1. If GPU time is available, run a full BaboonLand B2 validation for the cluster-prune policy.
2. If not, treat the 573-row bootstrapped result as an exploratory extension and move to paper organization.
3. Keep the main paper contribution centered on MiniCPM adaptive visual compute/speculative visual inference.
4. Present Qwen visual-token pruning as evidence that the same adaptive-compute principle may also apply inside the VLM visual-token prefix.
5. Avoid overstating the Qwen result as full-validation proof unless the broader run is completed.

Why this is next: the Qwen uncertainty analysis is now positive on a stratified sample, so the main remaining decision is scope: invest GPU time in a broader Qwen run or consolidate the paper around the already mature MiniCPM result.
