#!/usr/bin/env python
"""Bootstrap uncertainty for Qwen visual-token pruning policies."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
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


def group_key(row_id: str) -> str:
    return row_id.split(":", 1)[0]


def parse_fragile_labels(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def make_policy_rows(
    records: list[dict[str, Any]],
    *,
    prune_fraction: float,
    entropy_threshold: float,
    fragile_labels: set[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    fraction_tag = f"{int(round(prune_fraction * 100))}"
    by_policy: dict[str, dict[str, dict[str, Any]]] = {
        "no_prune": {},
        f"random_prune_{fraction_tag}": {},
        f"cluster_merge_prune_{fraction_tag}": {},
        f"entropy_gated_cluster_merge_prune_{fraction_tag}": {},
    }
    no_prune = {row["id"]: row for row in records if row["policy"] == "no_prune"}
    random_prune = {
        row["id"]: row
        for row in records
        if row["policy"] == "random_prune" and abs(float(row["prune_fraction"]) - prune_fraction) < 1e-9
    }
    cluster_prune = {
        row["id"]: row
        for row in records
        if row["policy"] == "cluster_merge_prune" and abs(float(row["prune_fraction"]) - prune_fraction) < 1e-9
    }
    missing_random = sorted(set(no_prune) - set(random_prune))
    missing_cluster = sorted(set(no_prune) - set(cluster_prune))
    if missing_random or missing_cluster:
        raise ValueError(
            "Missing paired pruning rows: "
            f"random={missing_random[:1]}, cluster={missing_cluster[:1]}"
        )

    by_policy["no_prune"] = no_prune
    by_policy[f"random_prune_{fraction_tag}"] = random_prune
    by_policy[f"cluster_merge_prune_{fraction_tag}"] = cluster_prune
    gated = {}
    for row_id, base in no_prune.items():
        entropy = base.get("confidence", {}).get("mean_token_entropy")
        use_pruned = (
            entropy is not None
            and float(entropy) <= entropy_threshold
            and base.get("pred_label") not in fragile_labels
        )
        gated[row_id] = cluster_prune[row_id] if use_pruned else base
    by_policy[f"entropy_gated_cluster_merge_prune_{fraction_tag}"] = gated
    return by_policy


def summarize(rows: list[dict[str, Any]], baseline_rows: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    from collections import Counter

    n = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    gold_counts = Counter(row.get("gold_label") for row in rows)
    pred_counts = Counter(row.get("pred_label") for row in rows)
    tp_counts = Counter(row.get("gold_label") for row in rows if row.get("pred_label") == row.get("gold_label"))
    f1_values = []
    for label, support in gold_counts.items():
        tp = tp_counts[label]
        fp = pred_counts[label] - tp
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    latencies = [float(row["generation_latency_ms"]) for row in rows if row.get("generation_latency_ms") is not None]
    prompt_tokens = [
        float(row["prompt_tokens_after_prune"])
        for row in rows
        if row.get("prompt_tokens_after_prune") is not None
    ]
    pruned = sum(1 for row in rows if row.get("policy") != "no_prune")
    fixed = 0
    broken = 0
    changed = 0
    if baseline_rows is not None:
        for row in rows:
            base = baseline_rows[row["id"]]
            if row.get("pred_label") != base.get("pred_label"):
                changed += 1
            if not base.get("correct") and row.get("correct"):
                fixed += 1
            if base.get("correct") and not row.get("correct"):
                broken += 1
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "pruned_load": pruned / n if n else 0.0,
        "generation_latency_median_ms": statistics.median(latencies) if latencies else None,
        "generation_latency_mean_ms": statistics.mean(latencies) if latencies else None,
        "prompt_tokens_mean": statistics.mean(prompt_tokens) if prompt_tokens else None,
        "prediction_changes_vs_no_prune": changed,
        "fixed_no_prune_errors": fixed,
        "broken_no_prune_correct": broken,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "accuracy",
        "macro_f1",
        "pruned_load",
        "generation_latency_median_ms",
        "generation_latency_mean_ms",
        "prompt_tokens_mean",
        "prediction_changes_vs_no_prune",
        "fixed_no_prune_errors",
        "broken_no_prune_correct",
    ]
    output = {}
    for key in keys:
        values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
        output[key] = {
            "mean": round(statistics.mean(values), 6) if values else None,
            "ci95": [
                round(percentile(values, 0.025), 6),
                round(percentile(values, 0.975), 6),
            ]
            if values
            else None,
        }
    return output


def summarize_differences(
    samples: dict[str, list[dict[str, Any]]],
    *,
    baseline_name: str,
) -> dict[str, Any]:
    keys = [
        "accuracy",
        "macro_f1",
        "generation_latency_median_ms",
        "generation_latency_mean_ms",
        "prompt_tokens_mean",
        "pruned_load",
    ]
    output = {}
    baseline_samples = samples[baseline_name]
    for policy_name, policy_samples in samples.items():
        if policy_name == baseline_name:
            continue
        policy_output = {}
        for key in keys:
            values = [
                float(policy_sample[key]) - float(baseline_sample[key])
                for policy_sample, baseline_sample in zip(policy_samples, baseline_samples)
                if policy_sample.get(key) is not None and baseline_sample.get(key) is not None
            ]
            policy_output[key] = {
                "mean": round(statistics.mean(values), 6) if values else None,
                "ci95": [
                    round(percentile(values, 0.025), 6),
                    round(percentile(values, 0.975), 6),
                ]
                if values
                else None,
                "p_gt_0": round(sum(1 for value in values if value > 0) / len(values), 6)
                if values
                else None,
                "p_lt_0": round(sum(1 for value in values if value < 0) / len(values), 6)
                if values
                else None,
            }
        output[f"{policy_name}_minus_{baseline_name}"] = policy_output
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--prune-fraction", type=float, default=0.25)
    parser.add_argument("--entropy-threshold", type=float, default=0.3235232369916048)
    parser.add_argument("--fragile-labels", default="Foraging")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    records = load_records(Path(args.input_jsonl))
    policies = make_policy_rows(
        records,
        prune_fraction=args.prune_fraction,
        entropy_threshold=args.entropy_threshold,
        fragile_labels=parse_fragile_labels(args.fragile_labels),
    )
    baseline_rows = policies["no_prune"]
    groups: dict[str, list[str]] = defaultdict(list)
    for row_id in baseline_rows:
        groups[group_key(row_id)].append(row_id)
    group_names = sorted(groups)

    observed = {
        name: summarize(list(rows.values()), baseline_rows if name != "no_prune" else None)
        for name, rows in policies.items()
    }
    samples: dict[str, list[dict[str, Any]]] = {name: [] for name in policies}
    rng = random.Random(args.seed)
    for _ in range(args.n_bootstrap):
        sampled_groups = [rng.choice(group_names) for _ in group_names]
        sampled_ids = [row_id for group in sampled_groups for row_id in groups[group]]
        for name, rows_by_id in policies.items():
            sampled_rows = [rows_by_id[row_id] for row_id in sampled_ids]
            samples[name].append(
                summarize(sampled_rows, baseline_rows if name != "no_prune" else None)
            )

    output = {
        "input_jsonl": args.input_jsonl,
        "prune_fraction": args.prune_fraction,
        "entropy_threshold": args.entropy_threshold,
        "fragile_labels": sorted(parse_fragile_labels(args.fragile_labels)),
        "n_rows": len(baseline_rows),
        "n_groups": len(group_names),
        "n_bootstrap": args.n_bootstrap,
        "observed": observed,
        "bootstrap": {name: summarize_samples(policy_samples) for name, policy_samples in samples.items()},
        "paired_differences": {
            "vs_no_prune": summarize_differences(samples, baseline_name="no_prune"),
            f"vs_random_prune_{int(round(args.prune_fraction * 100))}": summarize_differences(
                samples,
                baseline_name=f"random_prune_{int(round(args.prune_fraction * 100))}",
            ),
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
