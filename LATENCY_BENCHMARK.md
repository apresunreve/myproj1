# Latency Benchmarking Guide

This document describes how to benchmark inference latency (ms/clip) for all
evaluated VLMs on KABR and Baboonland, and how to plot the accuracy vs.
latency Pareto curves used in the efficiency contribution.

---

## What is measured

| Metric | Description |
|---|---|
| `latency_ms` | Wall-clock time per clip: image preprocessing + forward pass + token decode |
| `latency_median_ms` | Median over all timed clips (robust to outliers) |
| `latency_p90_ms` | 90th-percentile latency |
| `clips_per_sec` | Throughput = 1000 / median_ms |
| `peak_gpu_mb` | Peak GPU memory allocated after the run |
| `accuracy` | Clip-level accuracy on the timed val subset |

Timing is done with `torch.cuda.Event` on GPU (sub-millisecond precision) or
`time.perf_counter` on CPU. A configurable warmup phase runs N silent clips
before timing starts to heat the CUDA kernels and JIT cache.

---

## Where timing lives in the code

### `src/kabr_vlm/vlm_baselines.py`

**`run_hf_generation(generator, image_paths, prompt, max_new_tokens)`**
— returns `(text: str, latency_ms: float)`.  
All generator classes (HFChatGenerator, LlavaNextVideoGenerator,
InternVLGenerator, MobileVLMGenerator, pipeline) pass through this single
function, so timing covers every model uniformly.

**`evaluate_records(..., warmup_clips=0)`**
— new optional parameter. Runs `warmup_clips` silent clips before the timed
loop. Each timed clip writes `latency_ms` into its JSONL record. The return
dict gains `latency_median_ms`, `latency_p90_ms`, `latency_mean_ms`,
`n_timed`.

### `src/kabr_vlm/cli.py`

**`bench-latency` command** — wraps `evaluate_records` with GPU memory
tracking and writes a compact `*.latency.json` summary alongside the
prediction JSONL.

---

## CLI: `bench-latency`

```
kabr-vlm bench-latency \
  --annotation  <val.csv>        \
  --image-root  <image_dir>      \
  --classes     <classes.json>   \
  --model-id    <HF model id>    \
  --dataset     kabr             \   # kabr | baboonland
  --task        behavior         \
  --n-clips     100              \   # clips to time (after warmup)
  --warmup-clips 5               \   # silent warmup clips
  --max-frames-per-clip 8        \
  --output      artifacts/latency/qwen25vl_3b_kabr_zero-shot.jsonl
```

Add `--adapter-path <dir>` to benchmark a LoRA-adapted model instead of
zero-shot.

### Output files

| File | Content |
|---|---|
| `<output>.jsonl` | Per-clip predictions with `latency_ms` field added |
| `<output>.latency.json` | Compact summary: model, dataset, mode, n_clips, median/p90/mean latency, clips_per_sec, peak_gpu_mb, accuracy |

Example `latency.json`:
```json
{
  "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
  "adapter_path": null,
  "mode": "zero-shot",
  "dataset": "kabr",
  "task": "behavior",
  "n_clips": 100,
  "warmup_clips": 5,
  "max_frames_per_clip": 8,
  "accuracy": 0.549,
  "latency_median_ms": 412.5,
  "latency_p90_ms": 510.3,
  "latency_mean_ms": 418.1,
  "clips_per_sec": 2.42,
  "peak_gpu_mb": 8432.0,
  "predictions_path": "artifacts/latency/qwen25vl_3b_kabr_zero-shot.jsonl"
}
```

---

## Running all benchmarks

### Zero-shot (all 6 models × 2 datasets)

```bash
bash scripts/run_latency_benchmark.sh
```

This loops over the 6 evaluated VLMs and runs each on KABR val and Baboonland
val in zero-shot mode. Already-completed runs are skipped automatically.

### Including LoRA adapters

```bash
INCLUDE_LORA=1 bash scripts/run_latency_benchmark.sh
```

This additionally benchmarks Qwen2.5-VL-3B with the KABR LoRA adapter and
with the joint KABR+Baboonland adapter.

### Environment overrides

| Variable | Default | Description |
|---|---|---|
| `N_CLIPS` | 100 | Clips to time per run |
| `WARMUP` | 5 | Silent warmup clips |
| `FRAMES` | 8 | Frames per clip |
| `INCLUDE_LORA` | 0 | Set to 1 to bench LoRA adapters |
| `KABR_ROOT` | `/scratch/general/nfs1/u1320844/KABR_files2` | KABR data root |
| `BBL_ROOT` | `/home/jhyan/baboon_data/charades` | Baboonland data root |
| `QWEN3B_KABR_ADAPTER` | `artifacts/paper/behavior/lora/qwen25vl3b/plain/checkpoint` | KABR LoRA adapter |
| `QWEN3B_JOINT_ADAPTER` | `artifacts/paper/behavior/lora/joint_a1/checkpoint` | Joint LoRA adapter |

Logs go to `logs/latency_benchmark/`, summaries to `artifacts/latency/`.

---

## Plotting the Pareto curve

After the benchmark runs complete:

```bash
python scripts/plot_pareto.py
```

Reads all `*.latency.json` files from `artifacts/latency/` and produces:

- `artifacts/latency/pareto.pdf` — side-by-side scatter plot of accuracy vs.
  median latency for KABR and Baboonland, with Pareto fronts drawn per mode
  (zero-shot / LoRA)
- `artifacts/latency/pareto.csv` — flat CSV of all benchmark results for
  tables

Options:

```bash
python scripts/plot_pareto.py \
  --latency-dir artifacts/latency \
  --output      artifacts/latency/pareto.pdf \
  --dataset     kabr              # optional: single-dataset view
```

---

## Connection to the efficiency contribution

The latency benchmarks directly enable:

**Option A — Token budget sweep**  
Run `bench-latency` with different `--max-frames-per-clip` values (4, 2, 1)
and, for Qwen2.5-VL, with `min_pixels` / `max_pixels` overrides at inference.
Each configuration produces one point on the Pareto curve.

**Option B — NAS + latency**  
The NAS grid eval scripts already call `evaluate_records`. Because
`latency_ms` is now written into every prediction JSONL, the NAS grid
automatically records latency per run with no additional code change. Use
`plot_pareto.py` to visualize the 3-axis (accuracy × latency × trainable
params) Pareto surface.

**Interpreting the curve**  
The paper claim is: "For wildlife behavior clips, X tokens/frame captures Y%
of full accuracy at Z× speedup over the VLM default token budget." The Pareto
curve from these runs fills in the X, Y, Z values empirically.

---

## Expected runtime

On a single A100-80GB:

| Setup | Estimated time |
|---|---|
| 6 models × 2 datasets × 100 clips zero-shot | ~3–4 hours total |
| +2 LoRA runs (Qwen3B) | +30 min |
| Plotting | <1 min |

Smaller `N_CLIPS=50` halves the time with minimal impact on median estimates.
