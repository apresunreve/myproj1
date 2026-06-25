#!/usr/bin/env python
"""Inspect MiniCPM-V 4.6 visual-token expansion for clip JSONLs.

This follows the same processor path as `HFChatGenerator` in
`src/kabr_vlm/vlm_baselines.py`. By default it is processor-only. Use
`--run-model-inspection` to load MiniCPM and verify that the visual features
inserted by the model match the image placeholder positions in the text prefix.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
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
)


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


def uniform_subsample_indices(num_items: int, max_items: int | None) -> list[int]:
    if num_items <= 0:
        return []
    if max_items is None or max_items <= 0 or num_items <= max_items:
        return list(range(num_items))
    if max_items == 1:
        return [num_items // 2]
    last = num_items - 1
    return [round(index * last / (max_items - 1)) for index in range(max_items)]


def parse_frame_counts(value: str) -> list[int | None]:
    frame_counts: list[int | None] = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item in {"all", "full", "none"}:
            frame_counts.append(None)
        else:
            frame_counts.append(int(item))
    if not frame_counts:
        raise ValueError("--frame-counts must contain at least one value")
    return frame_counts


def build_prompt_text(processor: Any, image_paths: list[str], prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": path} for path in image_paths],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return _render_messages_for_processor(processor, messages, add_generation_prompt=True)


def encode_inputs(processor: Any, image_paths: list[str], prompt: str):
    prompt_text = build_prompt_text(processor, image_paths, prompt)
    with ExitStack() as stack:
        images = [stack.enter_context(Image.open(path)).convert("RGB") for path in image_paths]
        encoded = processor(
            text=[prompt_text],
            images=[images],
            return_tensors="pt",
            padding=True,
        )
    return prompt_text, encoded


def tensor_to_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def shape_of(value: Any) -> list[int] | str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return type(value).__name__
    return [int(dim) for dim in shape]


def tokenizer_token_ids(processor: Any) -> dict[str, int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    names = [
        "image_token",
        "image_start_token",
        "image_end_token",
        "slice_start_token",
        "slice_end_token",
        "video_token",
    ]
    token_ids: dict[str, int] = {}
    for name in names:
        token = getattr(processor, name, None)
        if not isinstance(token, str):
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            token_ids[name] = int(token_id)
    return token_ids


def contiguous_runs(indices: list[int]) -> list[dict[str, int]]:
    if not indices:
        return []
    runs: list[dict[str, int]] = []
    start = prev = indices[0]
    for index in indices[1:]:
        if index == prev + 1:
            prev = index
            continue
        runs.append({"start": start, "end": prev, "length": prev - start + 1})
        start = prev = index
    runs.append({"start": start, "end": prev, "length": prev - start + 1})
    return runs


def summarize_encoded(processor: Any, encoded: Any) -> dict[str, Any]:
    input_ids = tensor_to_list(encoded["input_ids"])[0]
    attention_mask = tensor_to_list(encoded.get("attention_mask"))[0]
    token_ids = tokenizer_token_ids(processor)
    counts = {
        name: input_ids.count(token_id)
        for name, token_id in token_ids.items()
    }
    image_token_id = token_ids.get("image_token")
    video_token_id = token_ids.get("video_token")
    image_positions = [
        index for index, token_id in enumerate(input_ids)
        if image_token_id is not None and token_id == image_token_id
    ]
    video_positions = [
        index for index, token_id in enumerate(input_ids)
        if video_token_id is not None and token_id == video_token_id
    ]

    encoded_shapes = {key: shape_of(value) for key, value in encoded.items()}
    target_sizes = tensor_to_list(encoded.get("target_sizes")) if "target_sizes" in encoded else None
    target_products = None
    if isinstance(target_sizes, list):
        target_products = [
            int(item[0]) * int(item[1])
            for item in target_sizes
            if isinstance(item, list) and len(item) == 2
        ]

    return {
        "prompt_tokens": len(input_ids),
        "attention_tokens": int(sum(attention_mask)),
        "special_token_ids": token_ids,
        "special_token_counts": counts,
        "image_placeholder_tokens": len(image_positions),
        "video_placeholder_tokens": len(video_positions),
        "image_placeholder_runs": contiguous_runs(image_positions),
        "video_placeholder_runs": contiguous_runs(video_positions),
        "encoded_shapes": encoded_shapes,
        "target_sizes": target_sizes,
        "target_size_products": target_products,
    }


def minicpm_inner_model(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    if hasattr(model, "model"):
        return model.model
    return model


def move_encoded_to_device(encoded: Any, device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoded.items()
        if key != "mm_token_type_ids"
    }


def inspect_model_embedding_path(model: Any, encoded: Any) -> dict[str, Any]:
    import torch

    inner = minicpm_inner_model(model)
    device = _model_device(model)
    encoded = move_encoded_to_device(encoded, device)
    input_ids = encoded["input_ids"]
    inputs_embeds = inner.get_input_embeddings()(input_ids)

    result: dict[str, Any] = {
        "input_ids_shape": shape_of(input_ids),
        "base_inputs_embeds_shape": shape_of(inputs_embeds),
        "image_token_id": int(getattr(inner.config, "image_token_id", -1)),
        "video_token_id": int(getattr(inner.config, "video_token_id", -1)),
        "image_features_shape": None,
        "image_placeholder_mask_true": 0,
        "image_placeholder_token_positions": 0,
        "image_feature_count": 0,
        "image_feature_placeholder_match": None,
        "supports_inputs_embeds_forward": True,
    }

    pixel_values = encoded.get("pixel_values")
    target_sizes = encoded.get("target_sizes")
    if pixel_values is not None:
        with torch.inference_mode():
            vision_output = inner.get_image_features(pixel_values[:1], target_sizes)
        image_features = torch.cat(vision_output.pooler_output, dim=0).to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        mask = inner.get_placeholder_mask(
            input_ids,
            inputs_embeds,
            image_features,
            inner.config.image_token_id,
        )
        token_mask = mask.any(dim=-1) if mask.ndim == 3 else mask
        result["image_features_shape"] = shape_of(image_features)
        result["image_placeholder_mask_true"] = int(mask.sum().item())
        result["image_placeholder_token_positions"] = int(token_mask.sum().item())
        result["image_feature_count"] = int(image_features.shape[0])
        result["image_feature_placeholder_match"] = int(token_mask.sum().item()) == int(image_features.shape[0])

    return result


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_frame_count: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_frame_count.setdefault(str(record["requested_frame_count"]), []).append(record)

    summary: dict[str, Any] = {"n_records": len(records), "by_frame_count": {}}
    for frame_count, group in by_frame_count.items():
        prompt_tokens = [int(row["token_summary"]["prompt_tokens"]) for row in group]
        image_tokens = [int(row["token_summary"]["image_placeholder_tokens"]) for row in group]
        summary["by_frame_count"][frame_count] = {
            "n": len(group),
            "prompt_tokens_mean": round(statistics.mean(prompt_tokens), 2),
            "prompt_tokens_min": min(prompt_tokens),
            "prompt_tokens_max": max(prompt_tokens),
            "image_placeholder_tokens_mean": round(statistics.mean(image_tokens), 2),
            "image_placeholder_tokens_min": min(image_tokens),
            "image_placeholder_tokens_max": max(image_tokens),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--model-id", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--frame-counts", default="2,4,8")
    parser.add_argument("--max-rows", type=int, default=1)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--run-model-inspection", action="store_true")
    args = parser.parse_args()

    frame_counts = parse_frame_counts(args.frame_counts)
    rows = load_rows(Path(args.val_jsonl), max_rows=args.max_rows)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.run_model_inspection:
        if args.adapter_path:
            generator = load_hf_adapter_generator(args.model_id, args.adapter_path)
        else:
            generator = load_hf_pipeline(args.model_id)
        processor = generator.processor
    else:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        generator = None

    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            image_paths_all = list(row["image_paths"])
            for frame_count in frame_counts:
                indices = uniform_subsample_indices(len(image_paths_all), frame_count)
                image_paths = [image_paths_all[index] for index in indices]
                prompt_text, encoded = encode_inputs(processor, image_paths, row["prompt"])
                record: dict[str, Any] = {
                    "id": row.get("id"),
                    "requested_frame_count": frame_count if frame_count is not None else "all",
                    "num_input_frames": len(image_paths),
                    "selected_frame_indices": indices,
                    "prompt_text_chars": len(prompt_text),
                    "gold_label": row.get("answer") or row.get("metadata", {}).get("behavior"),
                    "token_summary": summarize_encoded(processor, encoded),
                    "model_embedding_path": None,
                }
                if generator is not None:
                    record["model_embedding_path"] = inspect_model_embedding_path(generator.model, encoded)
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                records.append(record)

    summary = summarize_records(records)
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
