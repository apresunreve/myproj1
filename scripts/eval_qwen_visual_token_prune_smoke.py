#!/usr/bin/env python
"""Smoke-test true shorter-prefix Qwen visual-token pruning.

This script builds Qwen multimodal embeddings, then removes selected visual
placeholder positions from `input_ids`, `attention_mask`, and `inputs_embeds`
before generation. It therefore shortens the LLM prefix, unlike the embedding
zero/merge smoke. It does not reduce vision-tower compute yet because visual
features are still built before pruning.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
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
    load_balanced_rows,
    load_rows,
    parse_drop_fractions,
    tokenizer_token_id,
    uniform_subsample,
)
from eval_qwen_visual_token_consolidation_smoke import similarity_rank  # noqa: E402
from kabr_vlm.vlm_baselines import _model_device, load_hf_adapter_generator, load_hf_pipeline  # noqa: E402

SUPPORTED_POLICIES = {
    "no_prune",
    "random_prune",
    "similarity_prune",
    "similarity_merge_prune",
    "cluster_merge_prune",
}


def parse_policies(value: str) -> list[str]:
    policies = [item.strip() for item in value.split(",") if item.strip()]
    if not policies:
        raise ValueError("--policies must contain at least one policy")
    unsupported = sorted(set(policies) - SUPPORTED_POLICIES)
    if unsupported:
        raise ValueError(f"Unsupported policies: {unsupported}. Supported: {sorted(SUPPORTED_POLICIES)}")
    if "no_prune" not in policies:
        policies.insert(0, "no_prune")
    return policies


def visual_token_positions(input_ids, image_token_id: int):
    return (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()


def contiguous_visual_runs(positions) -> list[list[int]]:
    """Return local visual-token indices grouped by contiguous prompt spans."""

    if int(positions.numel()) == 0:
        return []
    runs: list[list[int]] = []
    current = [0]
    for local_idx in range(1, int(positions.numel())):
        prev_pos = int(positions[local_idx - 1].item())
        this_pos = int(positions[local_idx].item())
        if this_pos == prev_pos + 1:
            current.append(local_idx)
        else:
            runs.append(current)
            current = [local_idx]
    runs.append(current)
    return runs


def k_center_representatives(visual_embeds, local_indices: list[int], keep_count: int) -> list[int]:
    import torch

    if keep_count >= len(local_indices):
        return local_indices
    if keep_count <= 0:
        return []

    local_tensor = torch.tensor(local_indices, dtype=torch.long, device=visual_embeds.device)
    run_embeds = visual_embeds[local_tensor].float()
    normalized = torch.nn.functional.normalize(run_embeds, dim=-1)
    centroid = torch.nn.functional.normalize(normalized.mean(dim=0, keepdim=True), dim=-1)
    first = int((normalized @ centroid.squeeze(0)).argmax().item())
    selected = [first]
    min_distance = 1.0 - (normalized @ normalized[first])

    while len(selected) < keep_count:
        next_local = int(min_distance.argmax().item())
        if next_local in selected:
            break
        selected.append(next_local)
        distance_to_next = 1.0 - (normalized @ normalized[next_local])
        min_distance = torch.minimum(min_distance, distance_to_next)

    return sorted(local_indices[index] for index in selected)


def select_cluster_pruned_positions(
    *,
    inputs_embeds,
    positions,
    prune_fraction: float,
) -> tuple[list[int], list[float]]:
    import torch

    visual_embeds = inputs_embeds[0, positions]
    normalized = torch.nn.functional.normalize(visual_embeds.float(), dim=-1)
    pruned_local: list[int] = []
    kept_local: list[int] = []
    for run in contiguous_visual_runs(positions):
        n_prune = int(math.floor(len(run) * prune_fraction))
        keep_count = len(run) - n_prune
        run_kept = k_center_representatives(visual_embeds, run, keep_count)
        run_kept_set = set(run_kept)
        kept_local.extend(run_kept)
        pruned_local.extend(index for index in run if index not in run_kept_set)

    if not pruned_local or not kept_local:
        return [], []

    pruned_tensor = torch.tensor(pruned_local, dtype=torch.long, device=visual_embeds.device)
    kept_tensor = torch.tensor(kept_local, dtype=torch.long, device=visual_embeds.device)
    similarities = normalized[pruned_tensor] @ normalized[kept_tensor].transpose(0, 1)
    max_similarities = similarities.max(dim=1).values.tolist()
    pruned_positions = [int(positions[index].item()) for index in sorted(pruned_local)]
    return pruned_positions, [float(value) for value in max_similarities]


def select_pruned_positions(
    *,
    inputs_embeds,
    input_ids,
    image_token_id: int,
    policy: str,
    prune_fraction: float,
    rng: random.Random,
) -> tuple[list[int], dict[str, Any]]:
    positions = visual_token_positions(input_ids, image_token_id)
    n_visual = int(positions.numel())
    n_prune = int(math.floor(n_visual * prune_fraction))
    if policy == "no_prune" or n_prune <= 0:
        return [], {
            "num_visual_tokens": n_visual,
            "num_pruned_visual_tokens": 0,
            "mean_selected_max_similarity": None,
        }
    if policy == "random_prune":
        local_indices = sorted(rng.sample(range(n_visual), n_prune))
        similarities: list[float] = []
    elif policy in {"similarity_prune", "similarity_merge_prune"}:
        visual_embeds = inputs_embeds[0, positions]
        ranked, max_sim = similarity_rank(visual_embeds)
        local_indices = sorted(int(index.item()) for index in ranked[:n_prune])
        similarities = [float(max_sim[index].item()) for index in local_indices]
    elif policy == "cluster_merge_prune":
        pruned_positions, similarities = select_cluster_pruned_positions(
            inputs_embeds=inputs_embeds,
            positions=positions,
            prune_fraction=prune_fraction,
        )
        return pruned_positions, {
            "num_visual_tokens": n_visual,
            "num_pruned_visual_tokens": len(pruned_positions),
            "mean_selected_max_similarity": (
                round(statistics.mean(similarities), 6) if similarities else None
            ),
        }
    else:
        raise ValueError(f"Unsupported policy: {policy}")

    pruned_positions = [int(positions[index].item()) for index in local_indices]
    return pruned_positions, {
        "num_visual_tokens": n_visual,
        "num_pruned_visual_tokens": n_prune,
        "mean_selected_max_similarity": (
            round(statistics.mean(similarities), 6) if similarities else None
        ),
    }


def merge_pruned_visual_embeddings(
    *,
    inputs_embeds,
    input_ids,
    image_token_id: int,
    pruned_positions: list[int],
):
    import torch

    if not pruned_positions:
        return inputs_embeds

    positions = visual_token_positions(input_ids, image_token_id)
    position_to_local = {int(position.item()): local_idx for local_idx, position in enumerate(positions)}
    selected_local = sorted(position_to_local[position] for position in pruned_positions)
    selected_set = set(selected_local)
    if len(selected_set) >= int(positions.numel()):
        return inputs_embeds

    visual_embeds = inputs_embeds[0, positions]
    normalized = torch.nn.functional.normalize(visual_embeds.float(), dim=-1)
    sims = normalized @ normalized.transpose(0, 1)
    sims.fill_diagonal_(-float("inf"))

    assignments: dict[int, list[int]] = {}
    for local_idx in selected_local:
        candidate_sims = sims[local_idx].clone()
        for dropped_idx in selected_set:
            candidate_sims[dropped_idx] = -float("inf")
        nearest_idx = int(candidate_sims.argmax().item())
        assignments.setdefault(nearest_idx, []).append(local_idx)

    merged = inputs_embeds.clone()
    for kept_idx, dropped_indices in assignments.items():
        source_indices = [kept_idx, *dropped_indices]
        merged_embed = visual_embeds[source_indices].float().mean(dim=0).to(merged.dtype)
        kept_position = int(positions[kept_idx].item())
        merged[0, kept_position] = merged_embed
    return merged


def prune_encoded_prefix(encoded: dict[str, Any], inputs_embeds, pruned_positions: list[int]):
    import torch

    if not pruned_positions:
        return encoded["input_ids"], encoded["attention_mask"], inputs_embeds
    seq_len = int(encoded["input_ids"].shape[1])
    keep = torch.ones(seq_len, dtype=torch.bool, device=encoded["input_ids"].device)
    keep[torch.tensor(pruned_positions, dtype=torch.long, device=keep.device)] = False
    pruned_input_ids = encoded["input_ids"][:, keep]
    pruned_attention_mask = encoded["attention_mask"][:, keep]
    pruned_inputs_embeds = inputs_embeds[:, keep, :]
    return pruned_input_ids, pruned_attention_mask, pruned_inputs_embeds


def generate_from_pruned_prefix(
    *,
    generator: Any,
    input_ids,
    attention_mask,
    inputs_embeds,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    cuda_sync()
    start = time.perf_counter()
    with torch.inference_mode():
        generated = generator.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    cuda_sync()
    latency_ms = (time.perf_counter() - start) * 1000.0
    prompt_len = int(input_ids.shape[1])
    sequences = generated.sequences
    completion = sequences[:, prompt_len:] if sequences.shape[1] > prompt_len else sequences
    text = generator.processor.batch_decode(
        completion,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    token_ids = [int(token_id) for token_id in completion[0].detach().cpu().tolist()]
    return {
        "generated_text": text,
        "latency_ms": latency_ms,
        "generation_latency_ms": latency_ms,
        "completion_tokens": int(completion.shape[1]),
        "prompt_tokens_after_prune": prompt_len,
        "token_ids": token_ids,
        "confidence": summarize_token_scores(generator.processor, token_ids, generated.scores),
    }


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
        "mean_token_prob": _mean(token_probs),
        "min_token_prob": min(token_probs) if token_probs else None,
        "mean_token_logprob": _mean(token_logprobs),
        "sum_token_logprob": sum(token_logprobs) if token_logprobs else None,
        "mean_token_logit_margin": _mean(token_logit_margins),
        "min_token_logit_margin": min(token_logit_margins) if token_logit_margins else None,
        "mean_token_entropy": _mean(token_entropies),
        "first_token_prob": first_token_prob,
        "first_token_logit_margin": first_token_logit_margin,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(f"{record['policy']}@{record['prune_fraction']}", []).append(record)

    baseline_by_id = {row["id"]: row for row in records if row["policy"] == "no_prune"}
    summary: dict[str, Any] = {"n_records": len(records), "by_policy_fraction": {}}
    for key, group in groups.items():
        latencies = [float(row["latency_ms"]) for row in group]
        generation_latencies = [float(row["generation_latency_ms"]) for row in group if row.get("generation_latency_ms")]
        correct = sum(1 for row in group if row["correct"])
        parsed = sum(1 for row in group if row["pred_label"] is not None)
        changed = 0
        fixed = 0
        broken = 0
        for row in group:
            base = baseline_by_id.get(row["id"])
            if base is None or row["policy"] == "no_prune":
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
            "generation_latency_median_ms": (
                round(statistics.median(generation_latencies), 2) if generation_latencies else None
            ),
            "prediction_changes_vs_no_prune": changed,
            "fixed_no_prune_errors": fixed,
            "broken_no_prune_correct": broken,
            "visual_tokens_mean": round(statistics.mean(row["num_visual_tokens"] for row in group), 2),
            "pruned_visual_tokens_mean": round(
                statistics.mean(row["num_pruned_visual_tokens"] for row in group), 2
            ),
            "prompt_tokens_after_prune_mean": round(
                statistics.mean(row["prompt_tokens_after_prune"] for row in group), 2
            ),
            "mean_selected_max_similarity": round(statistics.mean(similarities), 6) if similarities else None,
            "confidence_mean_token_entropy_median": _median_confidence(group, "mean_token_entropy"),
            "confidence_first_token_prob_median": _median_confidence(group, "first_token_prob"),
        }
    return summary


def _median_confidence(group: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row["confidence"][key])
        for row in group
        if row.get("confidence", {}).get(key) is not None
    ]
    return round(statistics.median(values), 6) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-rows", type=int, default=1)
    parser.add_argument("--samples-per-class", type=int, default=0)
    parser.add_argument("--max-input-frames", type=int, default=4)
    parser.add_argument("--prune-fractions", default="0.25,0.5")
    parser.add_argument(
        "--policies",
        default="no_prune,random_prune,similarity_prune,similarity_merge_prune",
    )
    parser.add_argument("--seed", type=int, default=29)
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
    prune_fractions = parse_drop_fractions(args.prune_fractions)
    policies = parse_policies(args.policies)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            image_paths, frame_indices = uniform_subsample(row["image_paths"], args.max_input_frames)
            encode_start = time.perf_counter()
            encoded = encode_inputs(generator.processor, image_paths, row["prompt"])
            encoded = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            encoded.pop("mm_token_type_ids", None)
            cuda_sync()
            encode_latency_ms = (time.perf_counter() - encode_start) * 1000.0
            embed_start = time.perf_counter()
            base_embeds = build_multimodal_inputs_embeds(generator.model, encoded)
            cuda_sync()
            embed_latency_ms = (time.perf_counter() - embed_start) * 1000.0
            gold_label = row.get("answer") or row.get("metadata", {}).get("behavior")
            work_items: list[tuple[str, float]] = [("no_prune", 0.0)]
            work_items.extend(
                (policy, prune_fraction)
                for policy in policies
                if policy != "no_prune"
                for prune_fraction in prune_fractions
            )
            for policy, prune_fraction in work_items:
                rng = random.Random(
                    args.seed + row_index * 1009 + int(prune_fraction * 10000) + sum(ord(c) for c in policy)
                )
                select_start = time.perf_counter()
                pruned_positions, policy_stats = select_pruned_positions(
                    inputs_embeds=base_embeds,
                    input_ids=encoded["input_ids"],
                    image_token_id=image_token_id,
                    policy=policy,
                    prune_fraction=prune_fraction,
                    rng=rng,
                )
                cuda_sync()
                selection_latency_ms = (time.perf_counter() - select_start) * 1000.0
                merge_start = time.perf_counter()
                policy_embeds = (
                    merge_pruned_visual_embeddings(
                        inputs_embeds=base_embeds,
                        input_ids=encoded["input_ids"],
                        image_token_id=image_token_id,
                        pruned_positions=pruned_positions,
                    )
                    if policy in {"similarity_merge_prune", "cluster_merge_prune"}
                    else base_embeds
                )
                cuda_sync()
                merge_latency_ms = (time.perf_counter() - merge_start) * 1000.0
                prune_start = time.perf_counter()
                input_ids, attention_mask, inputs_embeds = prune_encoded_prefix(
                    encoded,
                    policy_embeds,
                    pruned_positions,
                )
                cuda_sync()
                prune_latency_ms = (time.perf_counter() - prune_start) * 1000.0
                generated = generate_from_pruned_prefix(
                    generator=generator,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=args.max_new_tokens,
                )
                pred_label = extract_baboonland_label(generated["generated_text"])
                record = {
                    "id": row.get("id"),
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "correct": pred_label == gold_label,
                    "policy": policy,
                    "prune_fraction": prune_fraction,
                    "selected_frame_indices": frame_indices,
                    "pruned_visual_positions": pruned_positions,
                    "encode_latency_ms": encode_latency_ms,
                    "embed_latency_ms": embed_latency_ms,
                    "selection_latency_ms": selection_latency_ms,
                    "merge_latency_ms": merge_latency_ms,
                    "prune_latency_ms": prune_latency_ms,
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
