#!/usr/bin/env python
"""Smoke-test random Qwen visual-token embedding dropout.

This is an analysis baseline, not a deployment path. Qwen2.5-VL requires the
number of image placeholder tokens to match the visual features produced by the
vision tower, so this script keeps the token sequence intact and zeros a random
subset of visual placeholder embeddings after multimodal embedding construction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from kabr_vlm.vlm_baselines import (  # noqa: E402
    _model_device,
    _render_messages_for_processor,
    load_hf_adapter_generator,
    load_hf_pipeline,
    match_candidate,
    normalize_text,
)


BABOONLAND_CLASSES = [
    "Walking/Running",
    "Sitting/Standing",
    "Fighting/Playing",
    "Self-Grooming",
    "Being Groomed",
    "Grooming Somebody",
    "Mutual Grooming",
    "Infant-Carrying",
    "Foraging",
    "Drinking",
    "Mounting",
    "Sleeping",
    "Occluded",
]


def load_rows(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def uniform_subsample(items: list[str], max_items: int | None) -> tuple[list[str], list[int]]:
    if max_items is None or max_items <= 0 or len(items) <= max_items:
        return list(items), list(range(len(items)))
    if max_items == 1:
        index = len(items) // 2
        return [items[index]], [index]
    last = len(items) - 1
    indices = [round(index * last / (max_items - 1)) for index in range(max_items)]
    return [items[index] for index in indices], indices


def parse_drop_fractions(value: str) -> list[float]:
    fractions = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not fractions:
        raise ValueError("--drop-fractions must contain at least one value")
    for fraction in fractions:
        if fraction < 0.0 or fraction >= 1.0:
            raise ValueError("drop fractions must be in [0, 1)")
    return fractions


def encode_inputs(processor: Any, image_paths: list[str], prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": path} for path in image_paths],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt_text = _render_messages_for_processor(processor, messages, add_generation_prompt=True)
    with ExitStack() as stack:
        images = [stack.enter_context(Image.open(path)).convert("RGB") for path in image_paths]
        encoded = processor(
            text=[prompt_text],
            images=[images],
            return_tensors="pt",
            padding=True,
        )
    return encoded


def tokenizer_token_id(processor: Any, token: str) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"Could not resolve token id for {token}")
    return int(token_id)


def qwen_inner_model(model: Any):
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = getattr(base_model, "model", None)
    if inner is None:
        raise ValueError("Expected a Qwen2.5-VL model with a `.model` inner module")
    return inner


def build_multimodal_inputs_embeds(model: Any, encoded: dict[str, Any]):
    import torch

    inner = qwen_inner_model(model)
    input_ids = encoded["input_ids"]
    inputs_embeds = inner.get_input_embeddings()(input_ids)
    pixel_values = encoded.get("pixel_values")
    image_grid_thw = encoded.get("image_grid_thw")
    if pixel_values is not None:
        image_embeds = inner.get_image_features(pixel_values, image_grid_thw).pooler_output
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = inner.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    pixel_values_videos = encoded.get("pixel_values_videos")
    video_grid_thw = encoded.get("video_grid_thw")
    if pixel_values_videos is not None:
        video_embeds = inner.get_video_features(pixel_values_videos, video_grid_thw).pooler_output
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = inner.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            video_features=video_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    return inputs_embeds


def apply_random_visual_dropout(
    *,
    inputs_embeds,
    input_ids,
    image_token_id: int,
    drop_fraction: float,
    rng: random.Random,
):
    import torch

    mask = input_ids == image_token_id
    flat_positions = mask.nonzero(as_tuple=False)
    n_visual = int(flat_positions.shape[0])
    n_drop = int(math.floor(n_visual * drop_fraction))
    dropped_positions: list[int] = []
    if n_drop <= 0:
        return inputs_embeds, n_visual, 0, dropped_positions

    selected_indices = sorted(rng.sample(range(n_visual), n_drop))
    dropped_positions = [int(flat_positions[index, 1].item()) for index in selected_indices]
    dropped_mask = torch.zeros(mask.shape, dtype=torch.bool, device=mask.device)
    for index in selected_indices:
        batch_idx, seq_idx = flat_positions[index]
        dropped_mask[batch_idx, seq_idx] = True
    masked_embeds = inputs_embeds.clone()
    masked_embeds[dropped_mask] = 0
    return masked_embeds, n_visual, n_drop, dropped_positions


def cuda_sync() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def generate_from_embeds(
    *,
    generator: Any,
    encoded: dict[str, Any],
    inputs_embeds,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    model_inputs = {
        key: value
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "image_grid_thw", "video_grid_thw"}
    }
    model_inputs["inputs_embeds"] = inputs_embeds
    cuda_sync()
    start = time.perf_counter()
    with torch.inference_mode():
        generated = generator.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    cuda_sync()
    latency_ms = (time.perf_counter() - start) * 1000.0
    prompt_len = int(encoded["input_ids"].shape[1])
    completion = generated[:, prompt_len:] if generated.shape[1] > prompt_len else generated
    text = generator.processor.batch_decode(
        completion,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return {
        "generated_text": text,
        "latency_ms": latency_ms,
        "completion_tokens": int(completion.shape[1]),
    }


def extract_baboonland_label(text: str) -> str | None:
    matched = match_candidate(text, BABOONLAND_CLASSES)
    if matched is not None:
        return matched
    normalized = normalize_text(text)
    if "groom" in normalized:
        if "mutual" in normalized:
            return "Mutual Grooming"
        if "being" in normalized or "received" in normalized:
            return "Being Groomed"
        if "somebody" in normalized or "other" in normalized or "another" in normalized:
            return "Grooming Somebody"
        return "Self-Grooming"
    if "walk" in normalized or "run" in normalized or "trot" in normalized:
        return "Walking/Running"
    if "sit" in normalized or "stand" in normalized:
        return "Sitting/Standing"
    if "fight" in normalized or "play" in normalized:
        return "Fighting/Playing"
    if "forag" in normalized or "graz" in normalized or "brows" in normalized:
        return "Foraging"
    if "drink" in normalized:
        return "Drinking"
    if "mount" in normalized:
        return "Mounting"
    if "sleep" in normalized:
        return "Sleeping"
    if "occlu" in normalized or "hidden" in normalized or "obscur" in normalized:
        return "Occluded"
    return None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_fraction: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_fraction.setdefault(str(record["drop_fraction"]), []).append(record)
    summary: dict[str, Any] = {"n_records": len(records), "by_drop_fraction": {}}
    for fraction, group in by_fraction.items():
        latencies = [float(row["latency_ms"]) for row in group]
        correct = sum(1 for row in group if row["correct"])
        parsed = sum(1 for row in group if row["pred_label"] is not None)
        summary["by_drop_fraction"][fraction] = {
            "n": len(group),
            "accuracy": correct / len(group) if group else 0.0,
            "parsed_rate": parsed / len(group) if group else 0.0,
            "latency_median_ms": round(statistics.median(latencies), 2) if latencies else None,
            "latency_mean_ms": round(statistics.mean(latencies), 2) if latencies else None,
            "visual_tokens_mean": round(statistics.mean(row["num_visual_tokens"] for row in group), 2),
            "dropped_visual_tokens_mean": round(statistics.mean(row["num_dropped_visual_tokens"] for row in group), 2),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-rows", type=int, default=1)
    parser.add_argument("--max-input-frames", type=int, default=4)
    parser.add_argument("--drop-fractions", default="0,0.25,0.5")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    if args.adapter_path:
        generator = load_hf_adapter_generator(args.model_id, args.adapter_path)
    else:
        generator = load_hf_pipeline(args.model_id)
    device = _model_device(generator.model)
    image_token_id = tokenizer_token_id(generator.processor, "<|image_pad|>")
    rows = load_rows(Path(args.val_jsonl), args.max_rows)
    drop_fractions = parse_drop_fractions(args.drop_fractions)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            image_paths, frame_indices = uniform_subsample(row["image_paths"], args.max_input_frames)
            encoded = encode_inputs(generator.processor, image_paths, row["prompt"])
            encoded = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            encoded.pop("mm_token_type_ids", None)
            base_embeds = build_multimodal_inputs_embeds(generator.model, encoded)
            gold_label = row.get("answer") or row.get("metadata", {}).get("behavior")
            for drop_fraction in drop_fractions:
                rng = random.Random(args.seed + row_index * 1009 + int(drop_fraction * 10000))
                masked_embeds, n_visual, n_drop, dropped_positions = apply_random_visual_dropout(
                    inputs_embeds=base_embeds,
                    input_ids=encoded["input_ids"],
                    image_token_id=image_token_id,
                    drop_fraction=drop_fraction,
                    rng=rng,
                )
                generated = generate_from_embeds(
                    generator=generator,
                    encoded=encoded,
                    inputs_embeds=masked_embeds,
                    max_new_tokens=args.max_new_tokens,
                )
                pred_label = extract_baboonland_label(generated["generated_text"])
                record = {
                    "id": row.get("id"),
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "correct": pred_label == gold_label,
                    "drop_fraction": drop_fraction,
                    "num_visual_tokens": n_visual,
                    "num_dropped_visual_tokens": n_drop,
                    "selected_frame_indices": frame_indices,
                    "dropped_visual_positions": dropped_positions,
                    **generated,
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                records.append(record)

    summary = summarize(records)
    output_path.with_suffix(output_path.suffix + ".metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

