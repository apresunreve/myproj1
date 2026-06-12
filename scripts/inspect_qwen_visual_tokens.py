#!/usr/bin/env python
"""Inspect Qwen2.5-VL visual-token expansion for existing clip JSONLs.

The script follows the same chat-template and processor path used by the VLM
evaluator, but it can run processor-only. Use `--run-generation` only when
latency timing is needed, because that loads the full model.
"""

from __future__ import annotations

import argparse
import json
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
)

QWEN_VISUAL_TOKENS = [
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
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


def tokenizer_token_ids(processor: Any) -> dict[str, int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    token_ids: dict[str, int] = {}
    for token in QWEN_VISUAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            token_ids[token] = int(token_id)
    return token_ids


def tensor_to_nested_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def find_spans(input_ids: list[int], start_id: int | None, end_id: int | None) -> list[dict[str, int]]:
    if start_id is None or end_id is None:
        return []
    spans: list[dict[str, int]] = []
    open_start: int | None = None
    for index, token_id in enumerate(input_ids):
        if token_id == start_id:
            open_start = index
        elif token_id == end_id and open_start is not None:
            spans.append(
                {
                    "start": open_start,
                    "end": index,
                    "length": index - open_start + 1,
                }
            )
            open_start = None
    return spans


def summarize_encoded(processor: Any, encoded: Any) -> dict[str, Any]:
    input_ids = tensor_to_nested_list(encoded["input_ids"])[0]
    attention_mask = tensor_to_nested_list(encoded.get("attention_mask"))[0]
    token_ids = tokenizer_token_ids(processor)
    counts = {
        token: input_ids.count(token_id)
        for token, token_id in token_ids.items()
    }
    image_pad_count = counts.get("<|image_pad|>", 0)
    video_pad_count = counts.get("<|video_pad|>", 0)
    spans = find_spans(
        input_ids,
        token_ids.get("<|vision_start|>"),
        token_ids.get("<|vision_end|>"),
    )
    grid_summary: dict[str, Any] = {}
    for key in ("image_grid_thw", "video_grid_thw"):
        if key in encoded:
            grid = tensor_to_nested_list(encoded[key])
            grid_summary[key] = grid
            grid_summary[f"{key}_products"] = [
                int(item[0]) * int(item[1]) * int(item[2])
                for item in grid
                if isinstance(item, list) and len(item) == 3
            ]

    return {
        "prompt_tokens": len(input_ids),
        "attention_tokens": int(sum(attention_mask)),
        "special_token_ids": token_ids,
        "special_token_counts": counts,
        "visual_span_count": len(spans),
        "visual_spans": spans,
        "visual_placeholder_tokens": image_pad_count + video_pad_count,
        **grid_summary,
    }


def cuda_sync() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_generation(generator: Any, encoded: Any, max_new_tokens: int) -> dict[str, Any]:
    import torch

    device = _model_device(generator.model)
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
    model_inputs.pop("mm_token_type_ids", None)
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
    prompt_len = int(model_inputs["input_ids"].shape[1])
    completion = generated[:, prompt_len:]
    text = generator.processor.batch_decode(
        completion,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return {
        "generation_latency_ms": latency_ms,
        "completion_tokens": int(completion.shape[1]),
        "generated_text": text,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_frame_count: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_frame_count.setdefault(str(record["requested_frame_count"]), []).append(record)

    summary: dict[str, Any] = {"n_records": len(records), "by_frame_count": {}}
    for frame_count, group in by_frame_count.items():
        prompt_tokens = [int(row["token_summary"]["prompt_tokens"]) for row in group]
        visual_tokens = [int(row["token_summary"]["visual_placeholder_tokens"]) for row in group]
        generation_latencies = [
            float(row["generation"]["generation_latency_ms"])
            for row in group
            if row.get("generation") is not None
        ]
        item: dict[str, Any] = {
            "n": len(group),
            "prompt_tokens_mean": round(statistics.mean(prompt_tokens), 2),
            "prompt_tokens_min": min(prompt_tokens),
            "prompt_tokens_max": max(prompt_tokens),
            "visual_placeholder_tokens_mean": round(statistics.mean(visual_tokens), 2),
            "visual_placeholder_tokens_min": min(visual_tokens),
            "visual_placeholder_tokens_max": max(visual_tokens),
        }
        if generation_latencies:
            item["generation_latency_median_ms"] = round(statistics.median(generation_latencies), 2)
            item["generation_latency_mean_ms"] = round(statistics.mean(generation_latencies), 2)
        summary["by_frame_count"][frame_count] = item
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--frame-counts", default="2,4,8")
    parser.add_argument("--max-rows", type=int, default=1)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    frame_counts = parse_frame_counts(args.frame_counts)
    rows = load_rows(Path(args.val_jsonl), max_rows=args.max_rows)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.run_generation:
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
                    "generation": None,
                }
                if generator is not None:
                    record["generation"] = run_generation(generator, encoded, args.max_new_tokens)
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                records.append(record)

    summary = summarize_records(records)
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

