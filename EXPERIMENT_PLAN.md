# Experiment Plan — Upcoming Work

Last updated: 2026-04-20

## Where we are now

Baseline results table is ~70% populated across KABR and baboonland clip-level
behavior recognition, on six VLM backbones and two specialist video models. The
paper has four data-backed claims (see `EXPERIMENT_IDEAS.md` and current CSV):

1. Adapted VLMs match specialist video models on accuracy; efficiency is the gap.
2. Zero-shot failure is *structured* (class collapse, exposed by macro F1).
3. In-domain LoRA lift is large, robust, cross-backbone, and works on tiny models.
4. Cross-dataset transfer is a meaningful middle ground; frame-count, prompt
   format, and scale all affect it measurably.

The next phase proposes **two small methodological contributions** that
directly improve Claim 2 and Claim 4 rather than a new architecture:

- **Contribution A — Cross-dataset co-adaptation.** One LoRA adapter trained
  on KABR + baboonland jointly with a shared semantic label space.
- **Contribution B — Class-collapse-aware training.** Loss / sampling changes
  that directly target the macro-F1 collapse named in Claim 2.

---

## Pending in-flight jobs (must finish first)

| Job | Model | Purpose | Cluster | State |
|---|---|---|---|---|
| 1154344 | Gemma3-4B | baboonland LoRA train | granite | R |
| 1154345 | X3D-L | baboonland SlowFast train (scratch, 120 epochs) | granite | R (epoch 40/120) |
| 1154346 | I3D-L | baboonland SlowFast train | granite | R (epoch 115/120) |
| 1154347 | LLaVA-OV 7B | baboonland LoRA **eval** | granite | PD |
| 1154985 | I3D-L | **KABR** SlowFast train (matches baboonland) | granite | PD |

After these complete we have full in-domain LoRA rows for all six VLM families
plus specialist baselines on both datasets.

---

## Phase 1 — Close the current baseline table (this week)

Goals: finish in-flight jobs; add the two missing zero-shot rows and the
supervised-video eval rows; produce the per-behavior analysis from data
already on disk.

1. **Zero-shot baboonland** for Gemma3-4B and InternVL2-2B
   (eval-only, cheap) — uses existing `eval_cross_dataset_vlm.py` with
   `--target_dataset baboonland` and no adapter.
2. **Evaluate completed LoRA adapters** on their native val sets:
   - LLaVA-OV-7B baboonland LoRA (queued, 1154347)
   - Gemma3-4B baboonland LoRA (after 1154344 finishes)
3. **Score X3D-L and I3D-L predictions** (baboonland + KABR) once training
   completes. Uses `kabr-vlm score-predictions` on the SlowFast
   `clip_level/predictions.jsonl`. Need a baboonland-compatible scorer
   variant since local CLI is KABR-hardcoded — small patch.
4. **Build per-behavior heatmap** (Experiment A from `EXPERIMENT_IDEAS.md`).
   Uses existing prediction JSONLs in `artifacts/baboonland_eval/` and
   `artifacts/transfer_eval/`. No GPU. Produces the paper's flagship figure.
5. **Quantify class collapse** (Experiment B). Count predicted-label
   distribution vs gold for each zero-shot row; compute a "collapse score"
   (e.g. fraction of predictions concentrated in the top-1 gold class).
   No GPU; fast.

---

## Phase 2 — Contribution B: Class-collapse-aware training

Backbone: **Qwen2.5VL-3B** (cheapest; already in the baseline table).

| Run | Change vs B1 | Purpose |
|---|---|---|
| B1 | baseline LoRA, standard CE | reference; matches existing 0.734/0.327 |
| B2 | class-balanced CE (1/√class_count weighting) | direct imbalance fix |
| B3 | focal loss γ=2 | hard-example upweighting |
| B4 | oversampling rare classes (class-balanced sampler) | sampling-side fix |
| B5 | best-of-B2..B4 repeated across 3 seeds | noise floor |

All on baboonland-clip (imbalance severe: Sitting/Standing 9843 vs Mounting 63).
Apply the winning recipe to KABR too for generality.

**Success:** ≥ 0.03 macro F1 lift over B1 without an accuracy regression.

### Implementation

- New config `configs/qwen25vl3b_baboonland_behavior_lora_balanced_v1.yaml`
  (variants per loss/sampler).
- Patch required in `src/kabr_vlm/training/qwen_tasks.py` (or a wrapper
  around `Trainer`) to support weighted CE / focal / balanced sampler. All
  doable in ~150 lines.
- Existing `scripts/run_vlm_baboonland_full.sh` + sbatch wrapper used
  unchanged.

### Budget

5 training runs × ~8 h on granite-gpu-guest = ~40 GPU-h.
5 eval runs × ~1 h = 5 h.

---

## Phase 3 — Contribution A: Cross-dataset co-adaptation

Backbone: **Qwen2.5VL-3B**, same as Phase 2.

**First: label-space design decision** (no GPU, no jobs).
- **Option V1 — simple behavior mapping.** Explicit table: {Walk, Trot, Run}
  ≡ Walking/Running; {Graze, Browse} ≡ Foraging; Occluded ≡ Occluded;
  KABR Auto-Groom ≡ bbl Self-Grooming; KABR Head Up ≡ bbl Sitting/Standing.
  Baboonland-only classes (Fighting/Playing, Being Groomed, …) are kept
  unmapped; KABR-only samples never see these labels as positives.
- **Option V2 — text-embedding label space.** Compute embeddings for each
  class name; add a projection loss that encourages
  `emb("Walk")` ≈ `emb("Walking/Running")`. Architecturally heavier but
  preserves all fine distinctions.

Pick **V1 first** (paper-clean, fewer ML moving parts). V2 is a later
ablation if V1 works and reviewer pressure requires it.

| Run | Train on | Test on | Purpose |
|---|---|---|---|
| A0 | KABR LoRA (existing) | bbl val (V1 mapped) | transfer baseline |
| A0' | bbl LoRA (existing) | KABR val (V1 mapped) | transfer baseline rev |
| A1 | KABR+bbl interleaved, V1 | each val | joint training effect |
| A2 | KABR+bbl interleaved, V1 + shared semantic head | each val | semantic head effect |
| A3 | A2 + class-balanced loss (Phase 2 winner) | each val | **combined method — headline** |
| A4 | KABR LoRA → bbl few-shot fine-tune at 1/5/25/100 % bbl train | bbl val | few-shot curve |

**Success:**
- A3 lifts transfer direction macro F1 ≥ 0.10 over A0/A0'.
- A3 does not regress in-domain by more than 0.02 acc vs B5.

### Budget

- 3 new training runs (A1, A2, A3) × ~14 h (joint is ~1.5× data) = ~42 h
- 4 new evals (A0 existing; A1, A2, A3 on both targets) × ~1 h = ~8 h
- A4: 4 fine-tune runs × ~3 h = 12 h
- Total: ~62 GPU-h

---

## Phase 4 — Scale & generality checks

Rerun A3 on two additional backbones:
- Qwen2.5VL-7B
- LLaVA-OneVision 7B

Also: one **Qwen2.5VL-3B full-parameter SFT on KABR** as a single ablation row,
to (a) verify LoRA ≈ full SFT in-domain (expected), and (b) test whether full
SFT hurts cross-dataset transfer (plausible — if true, it's a paper-positive
finding supporting Contribution A).

### Budget

- 2 × A3 at 7B scale ≈ 30 h each = 60 h
- 1 × Qwen3B full SFT KABR ≈ 18 h
- Associated evals ≈ 6 h
- Total: ~84 GPU-h

---

## Phase 5 — Figures / analysis (no GPU)

All derived from JSONLs on disk after Phases 2–4.

- **F1 Per-behavior heatmap.** behavior × (zero-shot | transfer | in-domain | A3) × model. Flagship figure.
- **F2 Pred-distribution plot.** Visualizes class collapse: gold histogram vs predicted histogram for zero-shot / transfer / A3.
- **F3 Few-shot curve.** A4 results, with KABR-transfer as horizontal asymptote and in-domain as target. The "how many labels does your method save" figure.
- **F4 Acc vs macro-F1 scatter.** Every row in the results table as a point; diagonal shows balanced predictions, horizontal drift shows collapse. Single-glance summary.
- **F5 Latency vs F1 frontier.** VLM methods + X3D/I3D + our A3. Efficiency story.

---

## Timeline (with 2 concurrent GPUs)

- **Days 1–2:** Phase 1 wraps up (in-flight jobs); Phase 1.4–1.5 analyses done.
- **Days 3–5:** Phase 2 (B1–B5). Phase 5 F2/F4 buildable once B3 or B4 done.
- **Days 5–9:** Phase 3 (A0–A4). Phase 5 F1/F3 buildable once A3 done.
- **Days 9–13:** Phase 4 (scale + SFT). Phase 5 F5 finalizable.
- **Days 13–16:** writing.

Critical-path items:
1. Patch `Trainer` for weighted CE / focal / sampler (Phase 2).
2. Implement V1 label mapping + joint dataset iterator (Phase 3).
3. Write `score_baboonland.py` or adapt `kabr-vlm score-predictions` for
   baboonland (Phase 1.3) so SlowFast predictions can be scored without
   hand-mapping.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Joint training regresses in-domain (catastrophic forgetting) | medium | small KABR mixing ratio + per-dataset early stopping |
| V1 label mapping disputed by reviewers | medium | add V2 as appendix; report both |
| Class-balanced loss trades acc for F1 | low-medium | always report both; frame as controllable knob |
| B5 gains within noise | medium | 3 seeds on B1 and B5; report ±std |
| SlowFast KABR I3D-L underperforms X3D | low | just a baseline; underperform is fine |
| Granite 12h slots too short for A3 joint training | high | checkpoint every 500 steps; --requeue ensures resume |

---

## Open decisions (need user input before Phase 2 starts)

1. V1 label map — accept the proposed 5-way mapping, or edit it?
2. Target conference/deadline — drives whether Phase 4 is included or cut.
3. Whether to fold Track 2 MobileVLMv2 temporal distillation from
   `EXPERIMENT_IDEAS.md` into this paper, or save it for follow-up work.
   Given previous distill attempts underperformed LoRA, my recommendation is
   to *cut* it from this paper.
