#!/usr/bin/env python
"""Analyze confidence-gated visual-token pruning from Qwen prune smoke records."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_csv(value: str, *, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile over an empty list")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def infer_thresholds(records: list[dict[str, Any]], metric: str, quantiles: list[float]) -> list[float]:
    values = [
        float(row["confidence"][metric])
        for row in records
        if row.get("confidence", {}).get(metric) is not None
    ]
    return [quantile(values, q) for q in quantiles]


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def should_prune(value: float, threshold: float, direction: str) -> bool:
    if direction == "le":
        return value <= threshold
    if direction == "ge":
        return value >= threshold
    raise ValueError(f"Unsupported direction: {direction}")


def summarize_system(
    *,
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    metric: str,
    threshold: float,
    direction: str,
    fragile_labels: set[str],
) -> dict[str, Any]:
    selected = []
    pruned_count = 0
    fixed = 0
    broken = 0
    changed = 0
    for row_id, base in baseline.items():
        cand = candidate[row_id]
        metric_value = base.get("confidence", {}).get(metric)
        use_pruned = (
            metric_value is not None
            and base.get("pred_label") not in fragile_labels
            and should_prune(float(metric_value), threshold, direction)
        )
        row = cand if use_pruned else base
        selected.append(row)
        if use_pruned:
            pruned_count += 1
            if cand.get("pred_label") != base.get("pred_label"):
                changed += 1
            if not base.get("correct") and cand.get("correct"):
                fixed += 1
            if base.get("correct") and not cand.get("correct"):
                broken += 1

    correct = sum(1 for row in selected if row.get("correct"))
    parsed = sum(1 for row in selected if row.get("pred_label") is not None)
    latencies = [float(row["latency_ms"]) for row in selected if row.get("latency_ms") is not None]
    generation_latencies = [
        float(row["generation_latency_ms"])
        for row in selected
        if row.get("generation_latency_ms") is not None
    ]
    prompt_tokens = [
        float(row["prompt_tokens_after_prune"])
        for row in selected
        if row.get("prompt_tokens_after_prune") is not None
    ]
    return {
        "threshold": threshold,
        "direction": direction,
        "n": len(selected),
        "accuracy": correct / len(selected) if selected else 0.0,
        "parsed_rate": parsed / len(selected) if selected else 0.0,
        "pruned_load": pruned_count / len(selected) if selected else 0.0,
        "n_pruned": pruned_count,
        "prediction_changes_on_pruned": changed,
        "fixed_no_prune_errors": fixed,
        "broken_no_prune_correct": broken,
        "latency_median_ms": round(median(latencies), 2) if latencies else None,
        "latency_mean_ms": round(mean(latencies), 2) if latencies else None,
        "generation_latency_median_ms": (
            round(median(generation_latencies), 2) if generation_latencies else None
        ),
        "prompt_tokens_after_prune_mean": round(mean(prompt_tokens), 2) if prompt_tokens else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--policy", default="cluster_merge_prune")
    parser.add_argument("--prune-fraction", type=float, default=0.25)
    parser.add_argument("--metric", default="mean_token_entropy")
    parser.add_argument("--direction", choices=["le", "ge"], default="le")
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--quantiles", default="0.25,0.5,0.75")
    parser.add_argument("--fragile-labels", default="Foraging")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    records = load_records(Path(args.input_jsonl))
    baseline = {row["id"]: row for row in records if row["policy"] == "no_prune"}
    candidate = {
        row["id"]: row
        for row in records
        if row["policy"] == args.policy and abs(float(row["prune_fraction"]) - args.prune_fraction) < 1e-9
    }
    missing = sorted(set(baseline) - set(candidate))
    if missing:
        raise ValueError(f"Candidate policy is missing {len(missing)} baseline ids; first missing id: {missing[0]}")

    if args.thresholds:
        thresholds = parse_csv(args.thresholds, cast=float)
    else:
        thresholds = infer_thresholds(baseline.values(), args.metric, parse_csv(args.quantiles, cast=float))
    fragile_labels = set(parse_csv(args.fragile_labels))

    systems = [
        summarize_system(
            baseline=baseline,
            candidate=candidate,
            metric=args.metric,
            threshold=threshold,
            direction=args.direction,
            fragile_labels=fragile_labels,
        )
        for threshold in thresholds
    ]
    output = {
        "input_jsonl": args.input_jsonl,
        "policy": args.policy,
        "prune_fraction": args.prune_fraction,
        "metric": args.metric,
        "fragile_labels": sorted(fragile_labels),
        "baseline_accuracy": sum(1 for row in baseline.values() if row.get("correct")) / len(baseline),
        "systems": systems,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
