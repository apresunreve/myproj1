#!/usr/bin/env python
import argparse
import json
import statistics
import time
from contextlib import ExitStack
from pathlib import Path

from PIL import Image

from kabr_vlm.data.kabr_frames import group_frame_records_into_clips, read_kabr_frame_annotations, sample_clip_frames
from kabr_vlm.vlm_baselines import (
    _is_phi3_processor,
    _model_device,
    _phi3_prompt_text,
    _render_messages_for_processor,
    build_prompt,
    load_hf_pipeline,
)


def _gpu_sync() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _prepare_qwen_inputs(generator, image_paths: list[str], prompt: str):
    with ExitStack() as stack:
        images = [stack.enter_context(Image.open(path)).convert("RGB") for path in image_paths]
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": path} for path in image_paths],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if _is_phi3_processor(generator.processor):
            prompt_text = _phi3_prompt_text(prompt, len(image_paths), add_generation_prompt=True)
            encoded = generator.processor(
                text=prompt_text,
                images=images,
                return_tensors="pt",
                padding=True,
            )
        else:
            prompt_text = _render_messages_for_processor(
                generator.processor,
                messages,
                add_generation_prompt=True,
            )
            encoded = generator.processor(
                text=[prompt_text],
                images=[images],
                return_tensors="pt",
                padding=True,
            )
    device = _model_device(generator.model)
    encoded.pop("mm_token_type_ids", None)
    encoded = {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}
    return encoded


def _run_once(generator, image_paths: list[str], prompt: str, max_new_tokens: int, force_exact_new_tokens: bool):
    import torch

    encoded = _prepare_qwen_inputs(generator, image_paths, prompt)
    generate_kwargs = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if force_exact_new_tokens:
        generate_kwargs["min_new_tokens"] = max_new_tokens
    if _is_phi3_processor(generator.processor):
        generate_kwargs["use_cache"] = False

    _gpu_sync()
    start = time.perf_counter()
    with torch.inference_mode():
        generated = generator.model.generate(**generate_kwargs)
    _gpu_sync()
    elapsed = time.perf_counter() - start

    prompt_len = encoded["input_ids"].shape[1]
    completion = generated[:, prompt_len:]
    decoded = generator.processor.batch_decode(
        completion,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    completion_tokens = int(completion.shape[1])
    prompt_tokens = int(prompt_len)
    return {
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_second": (completion_tokens / elapsed) if elapsed > 0 else 0.0,
        "text": decoded,
    }


def _benchmark_variant(
    *,
    model_id: str,
    attention_experiment: str | None,
    image_paths: list[str],
    prompt: str,
    max_new_tokens: int,
    warmup_runs: int,
    benchmark_runs: int,
    force_exact_new_tokens: bool,
):
    generator = load_hf_pipeline(model_id, attention_experiment=attention_experiment)
    for _ in range(warmup_runs):
        _run_once(generator, image_paths, prompt, max_new_tokens, force_exact_new_tokens)

    runs = [
        _run_once(generator, image_paths, prompt, max_new_tokens, force_exact_new_tokens)
        for _ in range(benchmark_runs)
    ]

    elapsed_values = [run["elapsed_seconds"] for run in runs]
    tps_values = [run["tokens_per_second"] for run in runs]
    token_values = [run["completion_tokens"] for run in runs]
    prompt_values = [run["prompt_tokens"] for run in runs]
    texts = [run["text"] for run in runs]

    result = {
        "attention_experiment": attention_experiment or "baseline",
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "elapsed_seconds": elapsed_values,
        "tokens_per_second": tps_values,
        "completion_tokens": token_values,
        "prompt_tokens": prompt_values,
        "mean_elapsed_seconds": statistics.mean(elapsed_values),
        "median_elapsed_seconds": statistics.median(elapsed_values),
        "mean_tokens_per_second": statistics.mean(tps_values),
        "median_tokens_per_second": statistics.median(tps_values),
        "mean_completion_tokens": statistics.mean(token_values),
        "sample_output": texts[-1] if texts else "",
    }

    del generator
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen2.5-VL 3B baseline vs MLA decode speed on one KABR clip.")
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--task", default="behavior", choices=["behavior", "species", "joint"])
    parser.add_argument("--prompt-mode", default="label", choices=["plain", "structured", "label"])
    parser.add_argument("--max-frames-per-clip", type=int, default=8)
    parser.add_argument("--clip-index", type=int, default=0, help="0-based clip index after annotation grouping.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument(
        "--force-exact-new-tokens",
        action="store_true",
        help="Set min_new_tokens=max_new_tokens for a steadier decode-length comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_kabr_frame_annotations(args.annotation, args.image_root, args.classes)
    clips = group_frame_records_into_clips(rows)
    if not clips:
        raise RuntimeError("No clips found in the provided annotation file.")
    if args.clip_index < 0 or args.clip_index >= len(clips):
        raise IndexError(f"clip-index {args.clip_index} is out of range for {len(clips)} clips.")

    clip = clips[args.clip_index]
    image_paths = sample_clip_frames(clip, max_frames=args.max_frames_per_clip)
    prompt = build_prompt(args.task, unit="clip", prompt_mode=args.prompt_mode)

    baseline = _benchmark_variant(
        model_id=args.model_id,
        attention_experiment=None,
        image_paths=image_paths,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        force_exact_new_tokens=args.force_exact_new_tokens,
    )
    mla = _benchmark_variant(
        model_id=args.model_id,
        attention_experiment="mla",
        image_paths=image_paths,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        force_exact_new_tokens=args.force_exact_new_tokens,
    )

    summary = {
        "model_id": args.model_id,
        "task": args.task,
        "prompt_mode": args.prompt_mode,
        "max_frames_per_clip": args.max_frames_per_clip,
        "max_new_tokens": args.max_new_tokens,
        "force_exact_new_tokens": args.force_exact_new_tokens,
        "clip": {
            "clip_index": args.clip_index,
            "clip_id": clip.clip_id,
            "original_video_id": clip.original_video_id,
            "video_id": clip.video_id,
            "label_name": clip.label_name,
            "species": clip.species,
            "clip_length": len(clip.frame_ids),
            "num_input_frames": len(image_paths),
            "image_paths": image_paths,
        },
        "baseline": baseline,
        "mla": mla,
        "comparison": {
            "speedup_vs_baseline_mean_tps": (
                mla["mean_tokens_per_second"] / baseline["mean_tokens_per_second"]
                if baseline["mean_tokens_per_second"] > 0
                else None
            ),
            "speedup_vs_baseline_median_tps": (
                mla["median_tokens_per_second"] / baseline["median_tokens_per_second"]
                if baseline["median_tokens_per_second"] > 0
                else None
            ),
            "latency_ratio_mean": (
                baseline["mean_elapsed_seconds"] / mla["mean_elapsed_seconds"]
                if mla["mean_elapsed_seconds"] > 0
                else None
            ),
            "latency_ratio_median": (
                baseline["median_elapsed_seconds"] / mla["median_elapsed_seconds"]
                if mla["median_elapsed_seconds"] > 0
                else None
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["comparison"], indent=2))
    print(f"Wrote benchmark summary to {args.output}")


if __name__ == "__main__":
    main()
