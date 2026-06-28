#!/usr/bin/env python
"""Analyze adaptive Gemma4 soft-token budget cascades from row-level outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

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


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def metric_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get("confidence", {}).get(metric)
    return float(value) if value is not None else None


def summarize_policy(records: list[dict[str, Any]], *, all_records_by_id: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    latencies = [
        float(row.get("effective_generation_latency_ms", row["generation_latency_ms"]))
        for row in records
    ]
    budgets = [int(row["max_soft_tokens"]) for row in records]
    baseline_280 = [
        all_records_by_id[row["id"]][280]
        for row in records
        if row["id"] in all_records_by_id and 280 in all_records_by_id[row["id"]]
    ]
    base_latency = statistics.mean(float(row["generation_latency_ms"]) for row in baseline_280) if baseline_280 else None
    mean_latency = statistics.mean(latencies) if latencies else None
    return {
        "n": len(records),
        "accuracy": sum(1 for row in records if row["correct"]) / len(records) if records else 0.0,
        "macro_f1": macro_f1(records, BABOONLAND_CLASSES),
        "parsed_rate": sum(1 for row in records if row["pred_label"] is not None) / len(records) if records else 0.0,
        "mean_generation_latency_ms": round(mean_latency, 2) if mean_latency is not None else None,
        "median_generation_latency_ms": round(statistics.median(latencies), 2) if latencies else None,
        "mean_soft_token_budget": round(statistics.mean(budgets), 2) if budgets else None,
        "budget_counts": {str(budget): budgets.count(budget) for budget in sorted(set(budgets))},
        "mean_executed_budget_count": round(
            statistics.mean(len(row.get("executed_budgets", [row["max_soft_tokens"]])) for row in records),
            2,
        )
        if records
        else None,
        "mean_latency_delta_vs_280_pct": (
            round((mean_latency - base_latency) / base_latency * 100.0, 2)
            if mean_latency is not None and base_latency
            else None
        ),
    }


def select_cascade(
    *,
    by_id: dict[str, dict[int, dict[str, Any]]],
    metric: str,
    threshold_70: float,
    threshold_140: float | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row_id, budgets in by_id.items():
        row70 = budgets[70]
        value70 = metric_value(row70, metric)
        if value70 is not None and value70 <= threshold_70:
            selected.append(with_execution_cost(row70, [row70]))
            continue
        row140 = budgets.get(140, row70)
        if threshold_140 is None:
            selected.append(with_execution_cost(row140, [row70, row140]))
            continue
        value140 = metric_value(row140, metric)
        if value140 is not None and value140 <= threshold_140:
            selected.append(with_execution_cost(row140, [row70, row140]))
        else:
            row280 = budgets.get(280, row140)
            selected.append(with_execution_cost(row280, [row70, row140, row280]))
    return selected


def with_execution_cost(selected: dict[str, Any], executed: list[dict[str, Any]]) -> dict[str, Any]:
    row = dict(selected)
    row["executed_budgets"] = [int(item["max_soft_tokens"]) for item in executed]
    row["effective_generation_latency_ms"] = sum(float(item["generation_latency_ms"]) for item in executed)
    return row


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile of empty values")
    values = sorted(values)
    index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--metric", default="mean_token_entropy")
    parser.add_argument("--quantiles", default="0.5,0.65,0.8")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    records = load_records(Path(args.input_jsonl))
    by_id: dict[str, dict[int, dict[str, Any]]] = {}
    for record in records:
        by_id.setdefault(record["id"], {})[int(record["max_soft_tokens"])] = record
    complete = {row_id: budgets for row_id, budgets in by_id.items() if {70, 140, 280}.issubset(budgets)}
    quantiles = [float(item.strip()) for item in args.quantiles.split(",") if item.strip()]
    metric70 = [metric_value(budgets[70], args.metric) for budgets in complete.values()]
    metric140 = [metric_value(budgets[140], args.metric) for budgets in complete.values()]
    metric70 = [value for value in metric70 if value is not None]
    metric140 = [value for value in metric140 if value is not None]

    result: dict[str, Any] = {
        "input": args.input_jsonl,
        "metric": args.metric,
        "n_complete_rows": len(complete),
        "policies": {},
    }
    for budget in [70, 140, 280]:
        selected = [with_execution_cost(budgets[budget], [budgets[budget]]) for budgets in complete.values()]
        result["policies"][f"static_{budget}"] = summarize_policy(selected, all_records_by_id=complete)
    for q70 in quantiles:
        threshold70 = quantile(metric70, q70)
        selected = select_cascade(
            by_id=complete,
            metric=args.metric,
            threshold_70=threshold70,
            threshold_140=None,
        )
        result["policies"][f"cascade_70_140_q{q70:g}"] = {
            "threshold_70": threshold70,
            **summarize_policy(selected, all_records_by_id=complete),
        }
        for q140 in quantiles:
            threshold140 = quantile(metric140, q140)
            selected = select_cascade(
                by_id=complete,
                metric=args.metric,
                threshold_70=threshold70,
                threshold_140=threshold140,
            )
            result["policies"][f"cascade_70_140_280_q{q70:g}_{q140:g}"] = {
                "threshold_70": threshold70,
                "threshold_140": threshold140,
                **summarize_policy(selected, all_records_by_id=complete),
            }

    print(json.dumps(result, indent=2))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
