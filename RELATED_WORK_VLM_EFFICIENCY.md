# Related Work: Efficient VLMs for Video and Multi-Frame Inputs (2023–2025)

This review covers methods for making Vision-Language Models efficient at inference
time on video and multi-frame inputs, organized by research area. Each method is
described concretely: what it does, what numbers it reports, and where it falls short.
The final section synthesizes gaps and suggests directions for a paper on efficient
wildlife behavior VLMs.

---

## Project Context: Models Used

This project fine-tunes and evaluates VLMs for wildlife behavior clip classification
on two datasets:
- **KABR-Clip** — 8 behavior classes (Kenyan wildlife: giraffe, plains zebra, Grevy's
  zebra), 11,003 train / 3,761 val clips
- **Baboonland-Clip** — 13 behavior classes (baboons, drone footage), 20,130 train /
  6,198 val clips

Inputs are 8 frames sampled uniformly per clip. Output is a single behavior label.
All VLM fine-tuning uses LoRA (rank-16) on the LLM layers.

### Primary evaluated models (zero-shot + LoRA fine-tuning)

| Model | Params | Architecture notes | KABR LoRA acc | Baboonland LoRA acc |
|---|---|---|---|---|
| Qwen2.5-VL-3B | 3B | Window ViT, MLP merger (4×), MRoPE | 0.853 | 0.734 |
| Qwen2.5-VL-7B | 7B | Same family, larger LLM | 0.848 | 0.755 |
| LLaVA-OneVision-0.5B | 0.5B | SigLIP ViT, bilinear pool 729→196/frame | 0.819 | 0.768 |
| LLaVA-OneVision-7B | 7B | Same family, larger LLM | 0.863 | 0.775 |
| Gemma3-4B | 4B | Fixed 256 tok/img, 5:1 local/global LLM attn | 0.864 | 0.756 |
| MobileVLM-V2-1.7B | 1.7B | LDPv2 projector (4× avg pool) | 0.782 | 0.712 |

Qwen2.5-VL-3B was also fine-tuned with a **joint LoRA adapter** trained on KABR +
Baboonland simultaneously, achieving 0.863 on KABR and 0.759 on Baboonland — matching
in-domain LoRA on both datasets with a single adapter.

X3D-L (3D CNN trained from scratch) is the specialist video baseline: 0.797 on KABR,
0.671 on Baboonland. All LoRA-adapted VLMs above beat X3D-L on both datasets.

### Small VLMs screened (zero-shot only, LoRA not yet started)

| Model | Params | KABR zero-shot acc | Baboonland zero-shot acc | Notes |
|---|---|---|---|---|
| MiniCPM-V 4.6 | ~4B | 0.652 | 0.377 | Perceiver resampler, 3D video resampler |
| LFM2.5-VL-1.6B | 1.6B | 0.436 | 0.509 | Hybrid gated-conv + GQA backbone |
| Qwen3.5-0.8B | 0.8B | — | — | Verbose output, needs prompt fix |
| Gemma4-E2B-it | ~2B | — | — | Unsupported loader, needs fix |

### Latency benchmarks run (zero-shot, KABR val, 50 clips)

| Model | 4-frame median ms | 8-frame median ms | Speedup 4f→8f |
|---|---|---|---|
| LLaVA-OV-0.5B | 173 | 289 | 1.67× |
| Qwen2.5-VL-3B | 272 | 409 | 1.50× |
| MiniCPM-V 4.6 | 311 | 391 | 1.26× |
| Qwen2.5-VL-7B | 388 | 592 | 1.53× |
| Gemma3-4B | 719 | 1239 | 1.72× |
| LLaVA-OV-7B | 766 | 1481 | 1.93× |

GPU: RTX PRO 4000 Blackwell (24 GB). Baboonland and LoRA-adapted benchmarks not yet
run.

---

## Area 1: Visual Token Compression at the Projector / Encoder

### Motivation

A 336×336 image through a CLIP ViT-L/14 produces 576 patch tokens. Modern
high-resolution VLMs tile images and can produce 2,880+ tokens per image. These
tokens dominate prefill cost: the LLM backbone processes all of them through every
attention layer. The attack points are (a) inside the vision encoder before the
projector, (b) at the projector boundary, and (c) inside the early LLM layers.

---

### 1.1 FastV — ECCV 2024 (Oral)

**Paper:** "An Image is Worth 1/2 Tokens After Layer 2" (Chen et al., ECCV 2024)

**Core observation:** In LLM layers 1–2, visual tokens receive ~50% of attention mass.
By deeper layers they receive only ~0.21% — nearly zero. Text tokens dominate after
layer 2.

**Method:** At layer K (default K=2), score every visual token by cumulative attention
it receives. Drop the bottom R% permanently. Inference-only, no training.

**Numbers:**
- K=2, R=50%: ~45% FLOPs reduction; GQA 62.0→60.4, MME-Perception 1510→1511
- Latency on A-OKVQA-13B: 0.124 s vs 0.203 s

**Limitations:**
- Attention scores at layer 2 are noisy ("attention shift," "attention dispersion")
- At 64-token budget: MMBench drops from 64.7 to 48.0 (−25% relative)
- Video benchmarks: FrameFusion reports FastV causes 19.7% relative performance drop
- Not designed for multi-frame inputs

**Position:** FastV was the first training-free plug-in token pruner. Its weakness
(unreliable LLM-internal attention signal) motivated all subsequent work.

---

### 1.2 FasterVLM — arXiv December 2024

**Paper:** "[CLS] Attention is All You Need for Training-Free Visual Token Pruning"
(Zhang et al., arXiv 2412.01818)

**Core insight:** FastV fails because LLM cross-attention is unreliable. The vision
encoder's own CLS token reliably aggregates patch importance via self-attention during
encoding — and this is already computed at zero extra cost.

**Method:** Score visual tokens by their attention weight from the [CLS] token in the
final ViT layer. Prune low-scoring tokens before the projector.

**Numbers at 95% token reduction on LLaVA-1.5-7B:**
- FasterVLM: 89.41% of baseline averaged across 10 tasks
- FastV: 74.93% of baseline
- VQAv2: 66.75 (FasterVLM) vs 55.92 (FastV)
- POPE: 67.24 vs 35.47

**Limitations:** CLS encodes global semantics, not fine-grained spatial detail.
OCR and counting tasks degrade at extreme ratios. No cross-frame temporal compression.

**Relationship to FastV:** Supersedes FastV at high compression ratios while remaining
training-free.

---

### 1.3 LLaVA-PruMerge — arXiv March 2024

**Paper:** "LLaVA-PruMerge: Adaptive Token Reduction" (Shang et al., arXiv 2403.15388)

**Method:** Two stages at the vision encoder output:
1. **Prune:** Select top-k tokens by CLS similarity (as in FasterVLM)
2. **Merge:** Cluster pruned tokens by cosine similarity; average each cluster into
   its nearest kept token. Information is preserved, not discarded.

**Numbers (LLaVA-1.5):**
- 14–18× average token compression (down to 5.5–25% of 576 tokens)
- 4–10× FLOPs reduction for LLM prefill
- Performance on VQAv2, TextVQA, POPE, MME: "comparable" to full model

**Limitations:** The "comparable" claim obscures non-trivial drops at extreme ratios.
The merging step adds overhead that partially offsets savings.

**Position:** PruMerge introduced the prune-then-merge paradigm that VisionZip refined.

---

### 1.4 VisionZip — CVPR 2025

**Paper:** "VisionZip: Longer is Better but Not Necessary in Vision Language Models"
(He et al., CVPR 2025, arXiv 2412.04467)

**Method:** Two-step selection from the vision encoder output:
1. **Dominant tokens:** Top-m by CLS attention score (kept verbatim)
2. **Contextual tokens:** Remaining tokens clustered by cosine similarity into c groups;
   each cluster averaged into one representative

Typical config: m=54 dominant + c=10 contextual = 64 total from 576 (11.1% retention).
Training-free, or with optional lightweight projector fine-tune.

**Numbers at 64-token budget (training-free):**

| Benchmark | Baseline | VisionZip | FastV | SparseVLM |
|-----------|----------|-----------|-------|-----------|
| TextVQA   | 58.2     | 55.5      | 47.8  | 51.8      |
| MMBench   | 64.7     | 60.1      | 48.0  | 56.2      |
| POPE      | 85.9     | 77.0      | 48.0  | 75.1      |
| SeedBench | 58.6     | 52.2      | 51.9  | 51.1      |

- 2× memory savings when applied during training
- Official Qwen2.5-VL support added to the repo in May 2025

**Limitations:**
- Tasks requiring fine-grained spatial detail (OCR) degrade more
- Per-frame compression only — no cross-frame temporal handling
- Performance below 5% retention degrades non-trivially

**Position:** Best training-free encoder-side method at 10–20% token retention.

---

### 1.5 SparseVLM — ICML 2025

**Paper:** "SparseVLM: Visual Token Sparsification for Efficient VLM Inference"
(Zhang et al., ICML 2025, arXiv 2410.04417)

**Core innovation:** CLS-based methods ignore the text prompt. SparseVLM scores visual
tokens using **text-token cross-attention** inside the LLM — which visual patches the
question actually looks at. Plus: adaptive per-layer sparsification ratios and token
"recycling" (pruned tokens compressed into compact summaries rather than discarded).

**Numbers on LLaVA-7B at 64 tokens:**
- 62.8% FLOPs reduction, 43.1% CUDA latency reduction
- Accuracy retention: 97%
- TextVQA: 53.4 vs FastV 47.8; POPE: 77.5 vs FastV 38.2
- On VideoLLaVA: 95.0% accuracy vs FastV's 80.3%

**Limitations:** Text-guided scoring requires the prompt at encoding time — incompatible
with image-caching pipelines. Token recycling adds overhead.

**Position:** Best training-free method for VQA tasks with a clear text query; this
exactly matches the wildlife classification setting where the prompt is fixed.

---

### 1.6 LLaVA-Mini — ICLR 2025

**Paper:** "LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One
Vision Token" (Guo et al., ICLR 2025, arXiv 2501.03895)

**Method:** Modality Pre-Fusion — N_fusion Transformer blocks process the concatenation
of all visual and text tokens before the main LLM. The text-token outputs become
"fusion tokens" (text enriched with visual knowledge). Only 1–64 visual tokens then
enter the main LLM backbone alongside these fusion tokens. A cross-attention query
module compresses all visual tokens to the target count.

**Numbers:**
- 1 vision token: VQAv2 77.6% vs LLaVA-1.5 78.5% (−0.9%); 77% FLOPs reduction
- Video: processes 10,000+ frames (3+ hours) on 24GB GPU
- Inference latency: 40 ms vs 100 ms for image understanding

**Limitations:** Requires training from scratch or extensive fine-tuning — not
plug-and-play. The pre-fusion stage adds ~2.42 GB overhead.

**Position:** The extreme end of the token reduction spectrum. Shows that near-zero
token counts are possible if the architecture is re-designed, but the training cost
is substantial.

---

### 1.7 PVC — Progressive Visual Token Compression (CVPR 2025)

**Paper:** "PVC: Progressive Visual Token Compression for Unified Image and Video
Processing" (Yang et al., CVPR 2025, arXiv 2412.09613)

**Method:** Adds causal temporal attention to the last 8 ViT layers — each frame
encodes information **complementary** to what was already captured in prior frames.
An Adaptive Compression Module then reduces each frame from 256 to 64 tokens.
Images are treated as "static 4-frame videos."

**Numbers (8B model):**
- MVBench: 73.8 (vs LLaVA-Video 58.6, InternVL2 66.4)
- VideoMME: 64.1
- Only 6.3% slower than single-image baseline despite processing 4 frames

**Limitations:** Requires training the ViT with modified causal temporal attention.

---

### How Area 1 Methods Relate to Each Other

```
CLS-attention scoring:   FasterVLM → VisionZip (adds contextual merging)
LLM-internal scoring:    FastV (pioneer) — superseded above at high ratios
Text-guided scoring:     SparseVLM (best for fixed-prompt classification tasks)
Prune-then-merge:        PruMerge → VisionZip (cleaner formulation)
Pre-fusion extreme:      LLaVA-Mini (requires training, enables 1-token extreme)
Causal temporal + arch:  PVC (CVPR 2025, state-of-the-art on video, requires training)
```

At 20–50% retention: all training-free methods perform comparably.
At 10% retention: VisionZip and SparseVLM substantially outperform FastV.
At <5% retention: only re-trained architectures (LLaVA-Mini, PVC) maintain performance.

---

## Area 2: Temporal Redundancy in Video VLMs

### Motivation

Adjacent video frames have high inter-token cosine similarity: FrameFusion measures
~0.62 for spatially corresponding token pairs vs ~0.28 for non-corresponding pairs
(2.2× higher). In deeper model layers these similarities remain high and their rankings
are consistent across layers — a stable, predictable redundancy signal. For wildlife
behavior clips, the level depends on behavior: an animal standing still may have
0.95+ adjacent-frame similarity; a running animal may be 0.7.

---

### 2.1 FrameFusion — arXiv January 2025

**Paper:** "FrameFusion: Combining Similarity and Importance for Video Token Reduction
on Large Vision Language Models" (Li et al., arXiv 2501.01986)

**Method:** Two stages inside the LLM forward pass:
1. **Merge (shallow layers):** Compute cosine similarity between spatially corresponding
   tokens across adjacent frames. Tokens exceeding a similarity threshold are merged
   (averaged). One copy is removed. Cascaded across multiple early layers.
2. **Prune (deep layers):** Cumulative attention scores from prior layers identify
   least-important merged tokens; these are permanently dropped.

**Numbers (6 LVLMs 2B–72B, 5 video benchmarks, 30% token budget):**
- Average performance drop: <3%
- LLaVA-Video-7B: 2.4% relative drop vs 15.4% (PruMerge), 19.7% (FastV), 35.8%
  (StreamingLLM)
- End-to-end speedup: 1.6–3.6× across model sizes
- Token reduction: 70%
- Tested explicitly on Qwen2-VL-7B (same family as Qwen2.5-VL-3B)

**Limitations:**
- Mixed results on image-only benchmarks
- Assumes frame alignment — problematic for panning/zooming cameras
- No evaluation on fine-grained action recognition tasks

---

### 2.2 HoliTom — NeurIPS 2025

**Paper:** "HoliTom: Holistic Token Merging for Fast Video Large Language Models"
(Shao et al., NeurIPS 2025, arXiv 2505.21334)

**Method:** Three stages:
1. **Global Temporal Segmentation (outer-LLM):** Dynamic programming segments the
   video into chunks that maximize within-chunk redundancy. Tokens appearing across
   all frames of a segment above similarity threshold τ are merged into the first frame.
2. **Spatial-Temporal Merging (outer-LLM):** Density peak clustering further merges
   spatially redundant tokens within each retained set.
3. **Inner-LLM Similarity Merging:** Low-attention tokens inside the LLM are averaged
   with their most similar retained neighbors rather than discarded.

**Numbers (LLaVA-OneVision-7B at 10% token retention):**
- FLOPs: 6.9% of original
- TTFT speedup: 2.28×; decoding throughput: 1.32×
- Average accuracy: 57.9 (vs full model 58.4 = 99.1% retained)
- Best at 10% budget vs FastV (57.8), VisionZip (53.5)

**Limitations:** Dynamic programming segmentation adds preprocessing overhead.
Benefits diminish for very short clips. Not evaluated on fine-grained action
classification.

---

### 2.3 EVS — Efficient Video Sampling (arXiv October 2025)

**Paper:** "Efficient Video Sampling: Pruning Temporally Redundant Tokens"
(arXiv 2510.14624)

**Method:** Simpler than cosine-similarity methods — operates on raw patch pixel
differences. Compute L1 difference of corresponding patches between consecutive frames.
Threshold at the q-th percentile; discard patches below threshold (static), retain
above (dynamic). Preserve positional identity without architecture change.

**Numbers (32 frames, plug-in mode):**
- VideoMME: 61.30 vs 65.50 baseline (−4.2 pp)
- MVBench: 69.08 vs 71.80 (−2.72 pp)
- TTFT speedup: up to 4× at q=0.75
- With uptraining: performance largely recovers
- On MVBench at 8 frames: 64.95% vs ToMe 63.73%

**Limitations:** At q=0.95, accuracy drops up to 20.4% without uptraining. JPEG
artifacts or subtle pose changes may fool L1 detection. Works best for long videos
with genuine temporal stasis; for continuous-motion clips (galloping), few patches
will be pruned.

---

### 2.4 Token Merging (ToMe) Applied to Video

**Background (ICLR 2023):** "Token Merging: Your ViT But Faster" (Bolya et al.) —
the spatial baseline. Bipartite matching within a frame, merge the r most similar
cross-set pairs. Yields 1.5–2× ViT throughput with minimal accuracy drop.

**Video extension — TempMe (ICLR 2025):**
"TempMe: Video Temporal Token Merging for Efficient Text-Video Retrieval" extends
ToMe temporally across frames using multi-granularity progressive merging. Applied to
CLIP-ViT-B/16 on MSRVTT. 95% token reduction, 1.8× speedup, with accuracy improvement
(designed for retrieval not generation). Note: targets CLIP-style encoders, not
autoregressive VLMs directly.

**EVS comparison:** At 8 frames on MVBench, EVS (64.95%) outperforms ToMe (63.73%),
suggesting raw patch-difference provides a better signal than learned-feature cosine
for short clips.

---

### 2.5 StreamingLLM — ICLR 2024 (note: bad for video)

**Paper:** "Efficient Streaming Language Models with Attention Sinks" (Xiao et al.,
ICLR 2024)

Visual tokens at the beginning of the prompt naturally function as "attention sinks."
StreamingLLM's sliding window discards mid-sequence visual tokens as the window
advances, causing severe accuracy drops on video VQA. FrameFusion confirms 35.8%
relative performance drop — the worst of all compared methods. Not recommended for
VLM video inference.

---

## Area 3: Architecture-Level Efficiency in Recent Small VLMs

The 2024–2025 model releases each make explicit architectural choices to reduce visual
token count. These are "baked-in" efficiency rather than post-hoc compression.

---

### 3.1 Qwen2.5-VL — February 2025

**Paper:** Qwen2.5-VL Technical Report (arXiv 2502.13923)

**Window Attention in ViT:** Replaces full O(N²) attention with window attention in
most ViT layers, cutting ViT cost to O(N·W). Full cross-window attention interleaved
at lower frequency.

**MRoPE:** Each visual token gets a 3D position embedding (time_id, height_id,
width_id) with absolute timestamps in seconds. Enables temporal reasoning over real
video duration without prompt engineering.

**MLP Merger:** Adjacent 2×2 ViT patches are grouped, concatenated (4× dimension),
and projected by a two-layer MLP — achieving 4× spatial compression before the LLM.
For 8 frames at 336×336: ~8 × 256 = 2,048 visual tokens entering the LLM.

**Relevance:** Qwen2.5-VL-3B is already architecturally efficient. External token
compression methods operate on top of the already-compressed 256 tokens/frame.

---

### 3.2 LLaVA-OneVision — 2024

**Paper:** "LLaVA-OneVision: Easy Visual Task Transfer" (Li et al., arXiv 2408.03326)

**Video token budget:** SigLIP ViT-SO/14-336px → 729 tokens/crop → 2×2 bilinear
pooling → 196 tokens/frame. For 8 frames: 8 × 196 = 1,568 tokens.

Bilinear pooling is fixed, simple, and content-agnostic — it discards fine spatial
detail uniformly regardless of content importance.

---

### 3.3 Gemma3-4B — March 2025

**Paper:** Gemma 3 Technical Report (arXiv 2503.19786)

**Fixed 256 visual token budget:** Hard budget regardless of resolution. No dynamic
content-adaptive compression.

**5:1 Local/Global Attention in LLM:** 5 local sliding-window attention layers (1024
token window) per 1 global full-attention layer. KV cache footprint drops from ~60%
to under 15% of memory. For 8-frame wildlife clips: 8 × 256 = 2,048 visual tokens;
most layers only see the 1024-token window, so old tokens fall out naturally.

---

### 3.4 MiniCPM-V 4.5 — September 2025

**Paper:** "MiniCPM-V 4.5: Cooking Efficient MLLMs" (arXiv 2509.18154)

**3D Resampler:** Extends a Perceiver-style resampler to 3D (spatial × temporal).
Jointly compresses up to 6 consecutive frames into 64 tokens — the same count used
for a single image. This is a 96× compression relative to processing 6 frames at
1,024 tokens each.

**Numbers:**
- Only 46.7% GPU memory and 8.7% inference time vs Qwen2.5-VL-7B
- A 6-second clip at 2fps, 448×448: 128 visual tokens
- VideoMME: 67.9; OpenCompass: 77.0

**Relevance:** The 3D Resampler is architecturally the cleanest solution for short-clip
classification — joint spatio-temporal compression into a fixed small budget.

---

### 3.5 MobileVLM V2 — February 2024

**Paper:** "MobileVLM V2: Faster and Stronger Baseline" (arXiv 2402.03766)

**LDPv2 (Lightweight Downsample Projector v2):** Pointwise conv → 4× average pooling
→ PEG with skip connections. Only 6.32M parameters. Achieves 37.4 tokens/s on A100.

Demonstrates that simple average pooling at the projector is competitive with complex
attention-based compression for classification tasks.

---

### 3.6 LFM2.5-VL — 2025

NAS-optimized hybrid backbone: 10 double-gated short-range convolution blocks + 6
Grouped Query Attention (GQA) blocks per 16-block cycle. The convolution blocks
contribute no KV cache; GQA reduces key/value heads. Result: KV cache dramatically
reduced without any post-hoc compression. 2× faster prefill/decode on CPUs. 40,400
output tokens/second on a single H100 at high concurrency.

---

## Area 4: KV Cache Compression for VLMs

### Why This Matters Less for Short Classification

For single-label classification with 1–5 output tokens, the generation phase is very
short. The bottleneck is prefill (processing all visual tokens). KV compression
methods mostly accelerate decoding and are most impactful for long-form generation
or multi-turn conversation. The methods are described here for completeness.

---

### 4.1 H2O — NeurIPS 2023

**Paper:** "H2O: Heavy-Hitter Oracle" (Zhang et al., NeurIPS 2023)

~5% of tokens receive >95% of cumulative attention ("heavy hitters"). H2O maintains
a fixed KV cache of C_heavy + C_recent positions, evicting everything else. Up to 29×
throughput improvement on OPT-6.7B vs DeepSpeed. 5% retention sufficient on text LLMs.

**VLM limitation:** Treats all tokens equally regardless of modality; may evict
visually important tokens. Replaced by VL-Cache for VLM use cases.

---

### 4.2 SnapKV — NeurIPS 2024

**Paper:** "SnapKV: LLM Knows What You are Looking for Before Generation"
(Li et al., NeurIPS 2024, arXiv 2404.14469)

Uses attention from the last W tokens of the prompt (the instruction/question) to
predict which prefix positions will matter for generation. Per-head selection with
max-pooling over the observation window. Compresses to 1,024 KV positions from
arbitrary length. 3.6× faster generation; 8.2× lower memory on 16K-token inputs.
More accurate than H2O at the same cache budget.

---

### 4.3 VL-Cache — ICLR 2025

**Paper:** "VL-Cache: Sparsity and Modality-Aware KV Cache Compression"
(Tu et al., ICLR 2025, arXiv 2410.23317)

First KV compression method designed specifically for VLMs. Key finding: language
tokens primarily attend to post-vision text tokens, not visual tokens. Visual token
attention sparsity varies by layer (some layers 99% sparse, others 70%). Method:
"Accumulated Post-vision Attention" scoring + layer-adaptive budget allocation.

**Numbers:**
- 10% KV retention → 98% task accuracy
- Decoding speedup: up to 7.08×; end-to-end: 2.33×; GPU memory: −90%

**Limitation for this project:** Short single-label classification outputs limit the
wall-clock decoding savings. Prefill compression (Areas 1–2) is more impactful.

---

## Area 5: Training-Efficient Adaptation (LoRA Variants)

### 5.1 Standard LoRA and Rank Effects

For simple classification (K=8–15 classes, single-label output), rank r=4–8 with
α=16–32 is typically sufficient. Analysis of trained adapters finds that effective
rank (singular value decay) is often lower than chosen r — suggesting redundancy.

**Where to apply in VLMs:**
- LLM transformer layers (Q, K, V, projection): consistent gains
- Vision encoder LoRA: only important when visual distribution is far
  out-of-distribution (wildlife imagery is a plausible case)
- Projector/connector: small, can often be fully fine-tuned

---

### 5.2 DoRA — ICML 2024 (Oral)

**Paper:** "DoRA: Weight-Decomposed Low-Rank Adaptation" (Liu et al., ICML 2024,
arXiv 2402.09353)

Decomposes weights into magnitude and direction, applies LoRA updates to direction
only, learns per-column magnitude scalars separately. Zero inference overhead (scalars
merge into weights at deployment). Consistently outperforms LoRA on LLaVA and
VL-BART. Supported in HuggingFace PEFT.

---

### 5.3 LoRA+ — February 2024

**Paper:** "LoRA+: Efficient Low Rank Adaptation" (Hayou et al., arXiv 2402.12354)

Theoretical analysis shows the A matrix (random-initialized) should use a lower
learning rate than the B matrix (zero-initialized). Fix: set η_B = λ · η_A, λ ≫ 1
(typically λ=16). One-line change to any training script. Improves performance and
fine-tuning speed, especially for models with large embedding dimensions.

---

### 5.4 VeRA — ICLR 2024

**Paper:** "VeRA: Vector-based Random Matrix Adaptation" (Kopiczko et al., ICLR 2024,
arXiv 2310.11454)

Shares a single random matrix pair (frozen) across all layers; each layer learns only
small trainable scaling vectors d and b. W = W₀ + (b ⊙ B)(d ⊙ A). Matches LoRA
performance with an order of magnitude fewer trainable parameters on GLUE. Attractive
for multi-adapter scenarios (one set of random matrices, tiny per-adapter vectors).

---

### 5.5 AdaLoRA — 2023

**Paper:** "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning"
(Zhang et al., arXiv 2303.10512)

SVD-parameterizes adapters as PΛQᵀ; singular values pruned during training to reach
target average rank. Distributes rank budget to layers that need it most. Outperforms
uniform-rank LoRA at the same parameter count. Adds training complexity; likely
overkill for simple K-class classification.

---

### 5.6 Post-Hoc SVD Compression of LoRA Adapters

**Papers:** PHLoRA (Amazon 2025); LoRA-Squeeze (arXiv 2602.10993); multiple
OpenReview submissions 2024

After training a LoRA adapter at rank r, the learned update BA typically has effective
rank much lower than r (rapid singular value decay). SVD truncation to rank r' < r
compresses the adapter with minimal accuracy loss.

**Finding:** Trained LoRA adapters for simple tasks need only r'=1–2 to capture most
expressive power. Training-free SVD compression consistently reduces adapter size 4–8×
without notable quality loss. PHLoRA extends this to full-rank fine-tuning: decompose
ΔW = W_finetuned − W₀ via truncated SVD to extract a LoRA adapter post-hoc.

**Relevance for wildlife project:** LoRA adapters trained at r=16 for 8-class
classification almost certainly have effective rank r'=2–4. Post-hoc SVD truncation
to r'=4 would reduce adapter size 4× and enable faster multi-adapter switching.

---

## Synthesis

### Three Most Active Research Directions

**Direction 1: Training-free visual token compression (most active, 2024–2025)**

The field has converged on: identify redundant tokens via attention signal → prune
or merge. Competition is on which signal is best (CLS > LLM cross-attention for
spatial tasks; text-guided > CLS for VQA), how to preserve information (merge > prune;
recycle > discard), and where to operate (before LLM = efficient, no task context;
inside LLM = uses task context, costs early-layer compute). Key progression:
FastV → FasterVLM / VisionZip → SparseVLM → HoliTom. 10+ major-venue papers per year.

**Direction 2: Video temporal redundancy exploitation (rapidly growing, 2025)**

Adjacent frames are structurally similar. FrameFusion, HoliTom, EVS, PVC, TempMe all
target this in 2024–2025. Key unsolved question: how to handle variable redundancy
(static scene vs fast action). Methods with fixed budgets over-compress dynamic
content and under-compress static content. Adaptive approaches are emerging but
immature. Current best: 70–93% FLOPs reduction with <3% accuracy drop on standard
benchmarks.

**Direction 3: Architecture-level efficiency (ongoing, model-release-driven)**

Every major 2025 VLM release (Qwen2.5-VL, Gemma3, MiniCPM-V 4.5) includes baked-in
efficiency: window attention in ViT, 4× spatial compression projector or Perceiver
resampler, hybrid local/global LLM attention. These "bake in" efficiency at design
time. Post-hoc token compression then operates on top of already-compressed
representations. The convergent design: SigLIP ViT + 4× compression projector +
hybrid LLM attention.

---

### What Has NOT Been Done for Short-Clip Behavior Recognition

**Gap 1: The short-clip regime is uncharacterized.**
The entire temporal redundancy literature evaluates on long videos (32–100 frames,
minutes to hours). FrameFusion: 32-frame YouTube clips. HoliTom: 100-frame long
videos. For 8-frame, 4-second wildlife behavior clips from a static camera, the
temporal dynamics are qualitatively different. No paper measures what fraction of
visual tokens can be dropped from 8-frame wildlife clips without accuracy loss.

**Gap 2: Token compression for closed-set behavior classification is unstudied.**
All token compression papers benchmark on VQA-style tasks (VQAv2, TextVQA, VideoMME)
requiring open-ended visual recall. Wildlife behavior classification is a closed-set
K-class problem where discriminative signal may be concentrated in a small set of
motion-related tokens. Which tokens actually matter for "mounting" vs "foraging" vs
"walking" has never been studied.

**Gap 3: Training-inference compression consistency is ignored.**
All methods apply compression at inference time to models trained on full-token inputs.
VisionZip notes that applying compression during training recovers near-full
performance — but no behavior classification paper exploits this. Training with
compressed tokens + deploying with compressed tokens likely outperforms the standard
"train full, deploy compressed" approach.

**Gap 4: Edge/field deployment context is absent.**
The entire literature benchmarks on A100/H100 GPUs. Wildlife monitoring involves
field deployment on embedded hardware (NVIDIA Jetson, Raspberry Pi). The relative
efficiency of different methods (window attention vs KV compression vs token pruning)
changes substantially at different hardware tiers. No paper studies VLM efficiency
for wildlife in an embedded/field deployment scenario.

---

## Suggested Research Directions

### Direction A: Content-Adaptive Token Budget Calibration for Behavior Recognition

**Hypothesis:** The number of visual tokens needed to classify a wildlife behavior
varies by behavior type. "Drinking" requires attention to head/neck only (20–40
tokens); "chasing" requires whole-scene motion (150+ tokens). Fixed-budget methods
are suboptimal because they apply the same compression to all behaviors.

**Approach:** Apply SparseVLM-style text-guided compression with behavior-specific
prompts. Compare fixed-budget VisionZip, FasterVLM, and text-guided compression
across all KABR and Baboonland behaviors. Measure per-class accuracy vs token budget.
Characterize which behaviors tolerate extreme compression and which do not.

**Why novel:** No existing paper studies per-class token budget sensitivity for
wildlife behavior recognition. The result would provide principled design rules for
efficient deployment.

---

### Direction B: Temporal Budget Calibration for Short Behavioral Clips

**Hypothesis:** Cross-frame temporal redundancy in 8-frame wildlife clips is
behavior-dependent and species-dependent. Methods applying fixed temporal reduction
rates (FrameFusion: 70%) over-compress active behaviors and under-compress sedentary
ones.

**Approach:** Measure actual inter-frame cosine similarity statistics across KABR
behaviors. Design a behavior-prior-aware compression that selects a compression rate
based on a lightweight motion estimate. Compare to uniform-rate FrameFusion, EVS,
and baseline.

**Why novel:** FrameFusion designed for 32-frame videos; nobody has studied the
8-frame regime with behavior-heterogeneous content.

---

### Direction C: Compression-Aware LoRA Fine-Tuning

**Hypothesis:** If a VLM is fine-tuned with LoRA on full-token inputs but deployed
with compressed inputs (VisionZip 64 tokens instead of 256), the adapter has learned
features from patterns the compressed representation drops. Re-training with the same
compression applied during fine-tuning should recover 2–5% accuracy and reduce
inference cost 2–4×.

**Approach:** Fine-tune Qwen2.5-VL-3B with LoRA in three conditions: (a) full tokens,
(b) VisionZip 64-token compression during training, (c) FrameFusion temporal
compression during training. Evaluate all six combinations (3 training × 2 inference
conditions) on KABR and Baboonland.

**Why novel:** Training-inference compression consistency is acknowledged in VisionZip
but never systematically studied for behavior classification VLMs.

---

### Direction D: Behavior-Class-Aware Frame Selection

**Hypothesis:** For low-motion behaviors (resting, grooming, ruminating), fewer than
4 frames carry all discriminative information. For high-motion behaviors (galloping,
fighting, mounting), all 8 frames are needed. A simple frame selector — score frames
by inter-frame cosine distance, select the K most diverse ones — should match or
exceed 8-frame accuracy at 4× lower cost for the low-motion majority.

**Approach:** Compute pairwise cosine similarity between frame CLS tokens for all
KABR/Baboonland clips. Analyze distribution by behavior class. Implement a greedy
max-diversity frame selector (no training required). Compare to 4-frame uniform
sampling at the same computational cost. Hypothesis: the selected frames outperform
uniform 4-frame sampling by 3–5% on low-motion behaviors.

**Why novel:** Frame selection papers exist for long video QA (30–100 frames), not
for 8-frame short-clip behavior recognition where the temporal redundancy is
behavior-class-dependent.

---

## References

- FastV (ECCV 2024): arXiv 2403.06764
- FasterVLM (Dec 2024): arXiv 2412.01818
- LLaVA-PruMerge (2024): arXiv 2403.15388
- VisionZip (CVPR 2025): arXiv 2412.04467
- SparseVLM (ICML 2025): arXiv 2410.04417
- LLaVA-Mini (ICLR 2025): arXiv 2501.03895
- PVC (CVPR 2025): arXiv 2412.09613
- FrameFusion (Jan 2025): arXiv 2501.01986
- HoliTom (NeurIPS 2025): arXiv 2505.21334
- EVS (Oct 2025): arXiv 2510.14624
- TempMe (ICLR 2025): arXiv 2409.01156
- StreamingLLM (ICLR 2024): arXiv 2309.17453
- H2O (NeurIPS 2023): arXiv 2306.14048
- SnapKV (NeurIPS 2024): arXiv 2404.14469
- VL-Cache (ICLR 2025): arXiv 2410.23317
- DoRA (ICML 2024): arXiv 2402.09353
- LoRA+ (2024): arXiv 2402.12354
- VeRA (ICLR 2024): arXiv 2310.11454
- AdaLoRA (2023): arXiv 2303.10512
- LoRA-Squeeze (2025): arXiv 2602.10993
- Qwen2.5-VL (Feb 2025): arXiv 2502.13923
- Gemma3 (Mar 2025): arXiv 2503.19786
- MiniCPM-V 4.5 (Sep 2025): arXiv 2509.18154
- MobileVLM V2 (Feb 2024): arXiv 2402.03766
- LFM2.5 Technical Report (2025): arXiv 2511.23404
- LLaVA-OneVision (2024): arXiv 2408.03326
- Efficient VLM Survey (2025): arXiv 2504.09724
