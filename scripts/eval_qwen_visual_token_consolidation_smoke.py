#!/usr/bin/env python
"""Compare Qwen visual-token consolidation policies on a small BaboonLand sample.

This smoke keeps Qwen's prompt length fixed because Qwen2.5-VL expects visual
placeholder tokens to align with vision features. The policies perturb the
visual placeholder embeddings after multimodal embedding construction:

- no_drop: unmodified embeddings.
- random_zero: zero a random subset of visual embeddings.
- similarity_zero: zero the most redundant visual embeddings, ranked by each
  token's maximum cosine similarity to another visual token.
- similarity_merge: replace the most redundant visual embeddings with their
  nearest kept visual neighbor.

This is still an analysis baseline. It tests whether redundancy-aware token
selection is less destructive than random deletion before attempting a true
shorter-sequence implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_qwen_random_visual_token_drop_smoke import (  # noqa: E402
    BABOONLAND_CLASSES,
    build_multimodal_inputs_embeds,
    cuda_sync,
    encode_inputs,
    extract_baboonland_label,
    generate_from_embeds,
    load_balanced_rows,
    load_rows,
    parse_drop_fractions,
    tokenizer_token_id,
    uniform_subsample,
)
from kabr_vlm.vlm_baselines import _model_device, load_hf_adapter_generator, load_hf_pipeline  # noqa: E402

SUPPORTED_POLICIES = {"no_drop", "random_zero", "similarity_zero", "similarity_merge"}


def parse_policies(value: str) -> list[str]:
    policies = [item.strip() for item in value.split(",") if item.strip()]
    if not policies:
        raise ValueError("--policies must contain at least one policy")
    unsupported = sorted(set(policies) - SUPPORTED_POLICIES)
    if unsupported:
        raise ValueError(f"Unsupported policies: {unsupported}. Supported: {sorted(SUPPORTED_POLICIES)}")
    if "no_drop" not in policies:
        policies.insert(0, "no_drop")
    return policies


def visual_token_positions(input_ids, image_token_id: int):
    return (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()


def similarity_rank(visual_embeds):
    import torch

    if visual_embeds.shape[0] <= 1:
        return torch.arange(visual_embeds.shape[0], device=visual_embeds.device), torch.zeros(
            visual_embeds.shape[0], device=visual_embeds.device
        )
    normalized = torch.nn.functional.normalize(visual_embeds.float(), dim=-1)
    sims = normalized @ normalized.transpose(0, 1)
    sims.fill_diagonal_(-float("inf"))
    max_sim, _nearest = sims.max(dim=1)
    return torch.argsort(max_sim, descending=True), max_sim


def apply_policy(
    *,
    inputs_embeds,
    input_ids,
    image_token_id: int,
    policy: str,
    drop_fraction: float,
    rng: random.Random,
) -> tuple[Any, dict[str, Any]]:
    import torch

    positions = visual_token_positions(input_ids, image_token_id)
    n_visual = int(positions.numel())
    n_drop = int(math.floor(n_visual * drop_fraction))
    if policy == "no_drop" or n_drop <= 0:
        return inputs_embeds, {
            "num_visual_tokens": n_visual,
            "num_modified_visual_tokens": 0,
            "modified_visual_positions": [],
            "mean_selected_max_similarity": None,
        }

    modified = inputs_embeds.clone()
    visual_embeds = inputs_embeds[0, positions]
    selected_local: list[int]
    selected_similarities: list[float] = []

    if policy == "random_zero":
        selected_local = sorted(rng.sample(range(n_visual), n_drop))
    elif policy in {"similarity_zero", "similarity_merge"}:
        ranked, max_sim = similarity_rank(visual_embeds)
        selected_local = sorted(int(index.item()) for index in ranked[:n_drop])
        selected_similarities = [float(max_sim[index].item()) for index in selected_local]
    else:
        raise ValueError(f"Unsupported policy: {policy}")

    selected_set = set(selected_local)
    selected_positions = [int(positions[index].item()) for index in selected_local]
    if policy.endswith("_zero"):
        for position in selected_positions:
            modified[0, position] = 0
    elif policy == "similarity_merge":
        normalized = torch.nn.functional.normalize(visual_embeds.float(), dim=-1)
        sims = normalized @ normalized.transpose(0, 1)
        sims.fill_diagonal_(-float("inf"))
        for local_index, position in zip(selected_local, selected_positions, strict=True):
            candidate_sims = sims[local_index].clone()
            if len(selected_set) < n_visual:
                for dropped_index in selected_set:
                    candidate_sims[dropped_index] = -float("inf")
            nearest_index = int(candidate_sims.argmax().item())
            modified[0, position] = visual_embeds[nearest_index].to(modified.dtype)

    return modified, {
        "num_visual_tokens": n_visual,
        "num_modified_visual_tokens": n_drop,
        "modified_visual_positions": selected_positions,
        "mean_selected_max_similarity": (
            round(statistics.mean(selected_similarities), 6) if selected_similarities else None
        ),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"{record['policy']}@{record['drop_fraction']}"
        groups.setdefault(key, []).append(record)

    summary: dict[str, Any] = {"n_records": len(records), "by_policy_fraction": {}}
    baseline_by_id = {row["id"]: row for row in records if row["policy"] == "no_drop"}
    for key, group in groups.items():
        latencies = [float(row["latency_ms"]) for row in group]
        correct = sum(1 for row in group if row["correct"])
        parsed = sum(1 for row in group if row["pred_label"] is not None)
        changed = 0
        fixed = 0
        broken = 0
        for row in group:
            base = baseline_by_id.get(row["id"])
            if base is None or row["policy"] == "no_drop":
                continue
            if row["pred_label"] != base["pred_label"]:
                changed += 1
            if not base["correct"] and row["correct"]:
                fixed += 1
            if base["correct"] and not row["correct"]:
                broken += 1
        similarities = [
            float(row["mean_selected_max_similarity"])
            for row in group
            if row.get("mean_selected_max_similarity") is not None
        ]
        summary["by_policy_fraction"][key] = {
            "n": len(group),
            "accuracy": correct / len(group) if group else 0.0,
            "parsed_rate": parsed / len(group) if group else 0.0,
            "latency_median_ms": round(statistics.median(latencies), 2) if latencies else None,
            "latency_mean_ms": round(statistics.mean(latencies), 2) if latencies else None,
            "prediction_changes_vs_no_drop": changed,
            "fixed_no_drop_errors": fixed,
            "broken_no_drop_correct": broken,
            "visual_tokens_mean": round(statistics.mean(row["num_visual_tokens"] for row in group), 2),
            "modified_visual_tokens_mean": round(
                statistics.mean(row["num_modified_visual_tokens"] for row in group), 2
            ),
            "mean_selected_max_similarity": round(statistics.mean(similarities), 6) if similarities else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-rows", type=int, default=1)
    parser.add_argument("--samples-per-class", type=int, default=0)
    parser.add_argument("--max-input-frames", type=int, default=4)
    parser.add_argument("--drop-fractions", default="0.25,0.5")
    parser.add_argument(
        "--policies",
        default="no_drop,random_zero,similarity_zero,similarity_merge",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    if args.adapter_path:
        generator = load_hf_adapter_generator(args.model_id, args.adapter_path)
    else:
        generator = load_hf_pipeline(args.model_id)
    device = _model_device(generator.model)
    image_token_id = tokenizer_token_id(generator.processor, "<|image_pad|>")
    if args.samples_per_class > 0:
        rows = load_balanced_rows(Path(args.val_jsonl), args.samples_per_class, BABOONLAND_CLASSES)
    else:
        rows = load_rows(Path(args.val_jsonl), args.max_rows)
    drop_fractions = parse_drop_fractions(args.drop_fractions)
    policies = parse_policies(args.policies)
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
            work_items: list[tuple[str, float]] = [("no_drop", 0.0)]
            work_items.extend(
                (policy, drop_fraction)
                for policy in policies
                if policy != "no_drop"
                for drop_fraction in drop_fractions
            )
            for policy, drop_fraction in work_items:
                rng = random.Random(
                    args.seed + row_index * 1009 + int(drop_fraction * 10000) + sum(ord(c) for c in policy)
                )
                modified_embeds, policy_stats = apply_policy(
                    inputs_embeds=base_embeds,
                    input_ids=encoded["input_ids"],
                    image_token_id=image_token_id,
                    policy=policy,
                    drop_fraction=drop_fraction,
                    rng=rng,
                )
                cuda_sync()
                generated = generate_from_embeds(
                    generator=generator,
                    encoded=encoded,
                    inputs_embeds=modified_embeds,
                    max_new_tokens=args.max_new_tokens,
                )
                pred_label = extract_baboonland_label(generated["generated_text"])
                record = {
                    "id": row.get("id"),
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "correct": pred_label == gold_label,
                    "policy": policy,
                    "drop_fraction": drop_fraction,
                    "selected_frame_indices": frame_indices,
                    **policy_stats,
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

