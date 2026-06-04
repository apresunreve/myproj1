# Model-Agnostic Efficiency Directions for Wildlife VLM Adapters

## Goal

Develop efficiency improvements that can be applied across multiple trained VLM
adapters, including Qwen2.5-VL, LLaVA-OneVision, MiniCPM, Gemma, and related models.

The preferred intervention point is the common VLM interface:

```text
8-frame clip
  -> vision encoder / projector
  -> visual tokens
  -> efficiency module
  -> LLM + trained LoRA adapter
  -> behavior label
```

This avoids model-specific architectural changes and focuses on reducing the visual
token burden before LLM prefill.

---

## Direction 1: Adapter-Aware Visual Token Consolidation

### Core Idea

Given a trained LoRA adapter, estimate which visual tokens matter for behavior
classification and consolidate the rest.

```text
8-frame clip
  -> VLM visual tokens
  -> token importance scoring
  -> merge/drop low-importance redundant tokens
  -> LLM with trained adapter
  -> label
```

### Possible Importance Signals

- Gradient of the correct label log-probability with respect to visual tokens
- Attention from output/label tokens to visual tokens
- Occlusion sensitivity: remove token groups and measure score drop
- Class-level token priors averaged over training clips

### Novelty

The compression policy is not generic. It is derived from the trained behavior
adapter, making it task-aware while remaining applicable across VLM backbones.

### Experiments

- Apply to Qwen2.5-VL-3B, LLaVA-OneVision-0.5B, and MiniCPM.
- Compare against random pruning, uniform pruning, and fixed token merging.
- Report accuracy, macro F1, latency, memory, and visual token count.

### Main Claim

A trained behavior adapter exposes task-specific token importance that enables
cross-model visual-token compression.

---

## Direction 2: Redundancy-Gated Token Merging

### Core Idea

Measure similarity among visual tokens and merge redundant tokens before LLM prefill.

For frame-aligned visual tokens:

```text
z[t, i] = token i in frame t
similarity = cosine(z[t, i], z[t+1, i])
```

If tokens are highly similar:

```text
z_merged = average(z[t, i], z[t+1, i])
```

### Why It Fits Wildlife Behavior Clips

Short wildlife clips often contain redundant background, static posture, and repeated
body regions. Behavior-relevant motion may occupy only a subset of tokens.

### Variants

- Fixed threshold merging: `0.90`, `0.95`, `0.98`
- Adaptive threshold based on clip-level redundancy
- Merge low-motion tokens while preserving high-motion tokens
- Merge by temporal token trajectory rather than by individual adjacent pairs

### Experiments

- Merge vs drop comparison
- Fixed threshold vs adaptive threshold
- Per-class compression tolerance
- Cross-model evaluation on Qwen, LLaVA, and MiniCPM

### Main Claim

Short wildlife clips contain behavior-dependent token redundancy that can be
consolidated across different VLM backbones.

---

## Direction 3: Confidence-Preserving Two-Pass Inference

### Core Idea

Run compressed-token inference first. If confidence is high, return the label. If
confidence is low, rerun with full visual tokens.

```text
compressed visual tokens
  -> prediction + confidence
  -> if confident: return label
  -> if uncertain: rerun full-token inference
```

### Confidence Signals

- Top-1 vs top-2 label log-probability margin
- Entropy over candidate behavior labels
- Agreement across prompt variants
- Normalized label likelihood

### Experiments

- Sweep fallback threshold
- Report accuracy vs average latency
- Report fallback rate by behavior class
- Compare across Qwen, LLaVA, and MiniCPM

### Main Claim

Most wildlife clips can be classified with compressed visual evidence, while hard
clips recover full-token accuracy through fallback.

---

## Direction 4: Label-Likelihood Classification Instead of Generation

### Core Idea

Since the task is closed-set behavior classification, score each candidate behavior
label directly instead of generating free-form text.

```text
score(label) = log P(label | visual tokens, prompt)
prediction = argmax_label score(label)
```

### Benefits

- Avoids verbose or malformed outputs
- Reduces decoding overhead
- Produces better confidence values
- Enables thresholded fallback and cascaded inference
- Standardizes evaluation across VLMs

### Main Claim

Closed-set label scoring is more stable and efficient than free-form generation for
wildlife behavior classification.

---

## Direction 5: Post-Hoc Adapter Compression

### Core Idea

Compress trained rank-16 LoRA adapters after training using SVD.

For each LoRA layer:

```text
Delta W = B A
Delta W ~= U_r S_r V_r
```

Evaluate compressed ranks:

```text
16 -> 8 -> 4 -> 2 -> 1
```

### Benefits

- Smaller adapters
- Faster adapter loading and switching
- Lower memory for multi-adapter deployment
- Useful for dataset-specific and joint-adapter comparisons

### Limitation

This may not provide large latency gains if LoRA weights are merged into the base
model before inference. Its main value is adapter storage and deployment efficiency.

### Main Claim

Rank-16 behavior adapters are likely overparameterized for 8-class or 13-class
classification and can be compressed with minimal accuracy loss.

---

## Direction 6: Class-Conditional Token Budgeting

### Core Idea

Different behavior classes require different amounts of visual evidence. Learn a
class-conditioned token budget from training data.

Example policy:

```text
standing      -> 25% tokens
grazing       -> 35% tokens
walking       -> 50% tokens
mounting      -> 75% tokens
fighting      -> 75-100% tokens
```

### Inference Procedure

1. Run a cheap compressed pass.
2. Get top candidate classes.
3. Select the token budget associated with those candidate classes.
4. Rerun only if confidence is insufficient.

### Experiments

- Per-class accuracy vs token budget
- Learned class budget vs fixed global budget
- Top-1 budget vs top-2 maximum budget
- Oracle class budget as an upper bound

### Main Claim

Behavior classes have different visual-token requirements, and class-conditioned
budgets improve the accuracy/latency tradeoff.

---

## Recommended Main Contribution

The strongest combined direction is:

## Adapter-Guided Adaptive Token Inference

A model-agnostic inference optimization for trained wildlife VLM adapters.

### Pipeline

```text
8-frame clip
  -> extract visual tokens
  -> compute redundancy and adapter-aware importance
  -> consolidate low-importance redundant tokens
  -> classify by label likelihood
  -> fallback to full tokens if confidence is low
```

### Ablations

| Variant | Purpose |
|---|---|
| Full tokens | Baseline |
| Random pruning | Weak compression baseline |
| Uniform token pruning | Simple structured baseline |
| Fixed redundancy merging | Tests token redundancy |
| Adaptive redundancy merging | Tests clip-level adaptation |
| Adapter-aware consolidation | Tests trained-adapter signal |
| Adapter-aware + fallback | Deployment-oriented version |

### Models

- Qwen2.5-VL-3B + LoRA
- LLaVA-OneVision-0.5B + LoRA
- MiniCPM + LoRA
- Optional: Gemma3-4B or LLaVA-OneVision-7B

### Metrics

- Accuracy
- Macro F1
- Per-class F1
- Visual token count
- Median latency
- Peak memory
- Accuracy per millisecond
- Fallback rate

### Main Paper Claim

A trained adapter can guide adaptive visual-token consolidation across multiple VLM
backbones, improving inference efficiency for short wildlife behavior classification.

---

## Minimal Experiment Package

1. Full-token baseline for each trained adapter.
2. Token redundancy characterization by behavior class.
3. Fixed token budget compression baselines.
4. Adapter-aware consolidation vs adapter-agnostic consolidation.
5. Confidence-gated fallback curve.
6. Cross-model evaluation on at least Qwen2.5-VL-3B, LLaVA-OneVision-0.5B, and MiniCPM.



