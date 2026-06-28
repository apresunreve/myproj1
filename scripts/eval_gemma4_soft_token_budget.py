#!/usr/bin/env python
"""Evaluate Gemma4 adaptive visual soft-token budgets.

Gemma4 exposes `max_soft_tokens` in its processor. Unlike post-hoc visual-token
pruning, this changes the visual budget before the model constructs its
multimodal prefix.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_qwen_random_visual_token_drop_smoke import (  # noqa: E402
    BABOONLAND_CLASSES,
    load_balanced_rows,
    load_rows,
    uniform_subsample,
)
from kabr_vlm.vlm_baselines import match_candidate  # noqa: E402


def parse_budgets(value: str) -> list[int]:
    budgets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not budgets:
        raise ValueError("--soft-token-budgets must contain at least one value")
    supported = {70, 140, 280, 560, 1120}
    invalid = sorted(set(budgets) - supported)
    if invalid:
        raise ValueError(f"Unsupported Gemma4 soft-token budgets: {invalid}. Supported: {sorted(supported)}")
    return budgets


def cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def tensor_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def summarize_token_scores(processor, token_ids: list[int], score_steps: tuple[Any, ...]) -> dict[str, float | int | None]:
    import torch

    tokenizer = getattr(processor, "tokenizer", None)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    token_probs: list[float] = []
    token_logprobs: list[float] = []
    token_logit_margins: list[float] = []
    token_entropies: list[float] = []
    first_token_prob: float | None = None
    first_token_logit_margin: float | None = None

    for index, token_id in enumerate(token_ids[: len(score_steps)]):
        if token_id in special_ids:
            continue
        logits = score_steps[index][0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        token_prob = float(probs[token_id].detach().cpu())
        token_logprob = float(log_probs[token_id].detach().cpu())
        top_values = torch.topk(logits, k=2).values
        logit_margin = float((top_values[0] - top_values[1]).detach().cpu())
        entropy = float((-(probs * log_probs).sum()).detach().cpu())
        if first_token_prob is None:
            first_token_prob = token_prob
            first_token_logit_margin = logit_margin
        token_probs.append(token_prob)
        token_logprobs.append(token_logprob)
        token_logit_margins.append(logit_margin)
        token_entropies.append(entropy)

    return {
        "n_scored_tokens": len(token_probs),
        "mean_token_prob": mean_or_none(token_probs),
        "min_token_prob": min(token_probs) if token_probs else None,
        "mean_token_logprob": mean_or_none(token_logprobs),
        "sum_token_logprob": sum(token_logprobs) if token_logprobs else None,
        "mean_token_logit_margin": mean_or_none(token_logit_margins),
        "min_token_logit_margin": min(token_logit_margins) if token_logit_margins else None,
        "mean_token_entropy": mean_or_none(token_entropies),
        "first_token_prob": first_token_prob,
        "first_token_logit_margin": first_token_logit_margin,
    }


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def render_prompt(processor, image_paths: list[str], prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": path} for path in image_paths],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_one(
    *,
    model,
    processor,
    image_paths: list[str],
    prompt: str,
    budget: int,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    import torch

    prompt_text = render_prompt(processor, image_paths, prompt)
    with ExitStack() as stack:
        images = [stack.enter_context(Image.open(path)).convert("RGB") for path in image_paths]
        encode_start = time.perf_counter()
        encoded = processor(
            text=[prompt_text],
            images=images,
            max_soft_tokens=budget,
            return_tensors="pt",
        )
        encode_latency_ms = (time.perf_counter() - encode_start) * 1000.0

    prompt_tokens = int(encoded["input_ids"].shape[1])
    image_token_id = int(model.config.image_token_id)
    image_tokens = int((encoded["input_ids"] == image_token_id).sum())
    encoded = tensor_to_device(dict(encoded), device)
    cuda_sync()
    generation_start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    cuda_sync()
    generation_latency_ms = (time.perf_counter() - generation_start) * 1000.0

    completion = generated.sequences[:, prompt_tokens:]
    text = processor.batch_decode(
        completion,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    token_ids = [int(token_id) for token_id in completion[0].detach().cpu().tolist()]
    return {
        "generated_text": text,
        "token_ids": token_ids,
        "completion_tokens": int(completion.shape[1]),
        "prompt_tokens": prompt_tokens,
        "image_tokens": image_tokens,
        "encode_latency_ms": encode_latency_ms,
        "generation_latency_ms": generation_latency_ms,
        "confidence": summarize_token_scores(processor, token_ids, generated.scores),
    }


def macro_f1(records: list[dict[str, Any]], labels: list[str]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for row in records if row["gold_label"] == label and row["pred_label"] == label)
        fp = sum(1 for row in records if row["gold_label"] != label and row["pred_label"] == label)
        fn = sum(1 for row in records if row["gold_label"] == label and row["pred_label"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def median_confidence(records: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row["confidence"][key])
        for row in records
        if row.get("confidence", {}).get(key) is not None
    ]
    return round(statistics.median(values), 6) if values else None


def summarize(records: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_records": len(records), "by_soft_token_budget": {}}
    by_budget: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_budget.setdefault(int(record["max_soft_tokens"]), []).append(record)
    for budget, group in sorted(by_budget.items()):
        generation_latencies = [float(row["generation_latency_ms"]) for row in group]
        encode_latencies = [float(row["encode_latency_ms"]) for row in group]
        summary["by_soft_token_budget"][str(budget)] = {
            "n": len(group),
            "accuracy": sum(1 for row in group if row["correct"]) / len(group) if group else 0.0,
            "macro_f1": macro_f1(group, labels),
            "parsed_rate": sum(1 for row in group if row["pred_label"] is not None) / len(group) if group else 0.0,
            "generation_latency_median_ms": round(statistics.median(generation_latencies), 2),
            "generation_latency_mean_ms": round(statistics.mean(generation_latencies), 2),
            "encode_latency_median_ms": round(statistics.median(encode_latencies), 2),
            "prompt_tokens_mean": round(statistics.mean(row["prompt_tokens"] for row in group), 2),
            "image_tokens_mean": round(statistics.mean(row["image_tokens"] for row in group), 2),
            "confidence_mean_token_entropy_median": median_confidence(group, "mean_token_entropy"),
            "confidence_first_token_prob_median": median_confidence(group, "first_token_prob"),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--model-id", default="google/gemma-4-E2B-it")
    parser.add_argument("--max-rows", type=int, default=1)
    parser.add_argument("--samples-per-class", type=int, default=0)
    parser.add_argument("--max-input-frames", type=int, default=4)
    parser.add_argument("--soft-token-budgets", default="70,140,280")
    parser.add_argument("--warmup-budget", type=int, default=140)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    budgets = parse_budgets(args.soft_token_budgets)
    if args.samples_per_class > 0:
        rows = load_balanced_rows(Path(args.val_jsonl), args.samples_per_class, BABOONLAND_CLASSES)
    else:
        rows = load_rows(Path(args.val_jsonl), args.max_rows)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    if rows and args.warmup_budget > 0:
        warmup_paths, _ = uniform_subsample(rows[0]["image_paths"], args.max_input_frames)
        _ = generate_one(
            model=model,
            processor=processor,
            image_paths=warmup_paths,
            prompt=rows[0]["prompt"],
            budget=args.warmup_budget,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            image_paths, frame_indices = uniform_subsample(row["image_paths"], args.max_input_frames)
            gold_label = row.get("answer") or row.get("metadata", {}).get("behavior")
            for budget in budgets:
                generated = generate_one(
                    model=model,
                    processor=processor,
                    image_paths=image_paths,
                    prompt=row["prompt"],
                    budget=budget,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                )
                pred_label = match_candidate(generated["generated_text"], BABOONLAND_CLASSES)
                record = {
                    "id": row["id"],
                    "row_index": row_index,
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "correct": pred_label == gold_label,
                    "max_soft_tokens": budget,
                    "max_input_frames": args.max_input_frames,
                    "frame_indices": frame_indices,
                    **generated,
                }
                records.append(record)
                handle.write(json.dumps(record) + "\n")
                handle.flush()

    print(json.dumps(summarize(records, BABOONLAND_CLASSES), indent=2))


if __name__ == "__main__":
    main()
