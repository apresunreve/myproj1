#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if isinstance(row_id, str):
                rows[row_id] = row
    return rows


def pred(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    value = row.get("pred_label") or row.get("prediction")
    return str(value) if value is not None else None


def latency(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    value = row.get("latency_ms") or row.get("total_latency_ms")
    return float(value) if isinstance(value, int | float) else None


def infer_labels(records: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for row in records:
        for key in ("gold_label", "prediction"):
            value = row.get(key)
            if isinstance(value, str) and value not in seen:
                seen.add(value)
                labels.append(value)
    return labels


def all_label_sets(labels: list[str], max_size: int | None) -> list[set[str]]:
    largest = len(labels) if max_size is None else min(max_size, len(labels))
    return [set(combo) for size in range(largest + 1) for combo in combinations(labels, size)]


def build_records(
    routing_rows: dict[str, dict[str, Any]],
    primary_rows: dict[str, dict[str, Any]],
    secondary_rows: dict[str, dict[str, Any]] | None,
    extra_fallback_predictions: set[str],
    secondary_fallback_predictions: set[str],
    route_decision_field: str,
    accept_value: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_id, routing_row in routing_rows.items():
        gold = routing_row.get("gold_label")
        qwen_prediction = routing_row.get("prediction")
        route_decision = routing_row.get(route_decision_field)
        use_fallback = route_decision != accept_value or qwen_prediction in extra_fallback_predictions
        if not use_fallback:
            prediction = qwen_prediction
            source = "qwen_accept"
            fallback_latency = None
        elif secondary_rows is not None and qwen_prediction in secondary_fallback_predictions:
            fallback_row = secondary_rows.get(row_id)
            prediction = pred(fallback_row)
            fallback_latency = latency(fallback_row)
            source = "secondary_fallback"
        else:
            fallback_row = primary_rows.get(row_id)
            prediction = pred(fallback_row)
            fallback_latency = latency(fallback_row)
            source = "primary_fallback"
        routing_latency = latency(routing_row)
        total_latency = None
        if routing_latency is not None:
            total_latency = routing_latency + (fallback_latency or 0.0)
        records.append(
            {
                "id": row_id,
                "gold_label": gold,
                "prediction": prediction,
                "correct": prediction == gold,
                "source": source,
                "qwen_prediction": qwen_prediction,
                "route_decision": route_decision,
                "routing_latency_ms": routing_latency,
                "fallback_latency_ms": fallback_latency,
                "total_latency_ms": total_latency,
            }
        )
    return records


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def compute_metrics(records: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    n_total = len(records)
    n_correct = sum(1 for row in records if row["correct"])
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    for row in records:
        gold = row["gold_label"]
        prediction = row["prediction"]
        per_class[gold]["support"] += 1
        if prediction == gold:
            per_class[gold]["tp"] += 1
        elif prediction is not None:
            per_class[gold]["fn"] += 1
            per_class[prediction]["fp"] += 1
        else:
            per_class[gold]["fn"] += 1
    per_class_f1 = {}
    f1_values = []
    for label in labels:
        stats = per_class[label]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        per_class_f1[label] = {**stats, "f1": f1}
        f1_values.append(f1)
    source_counts = Counter(row["source"] for row in records)
    fallback_rows = source_counts.get("primary_fallback", 0) + source_counts.get("secondary_fallback", 0)
    latencies = [
        float(row["total_latency_ms"])
        for row in records
        if isinstance(row.get("total_latency_ms"), int | float)
    ]
    metrics = {
        "accuracy": n_correct / n_total if n_total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "fallback_rows": fallback_rows,
        "fallback_load": fallback_rows / n_total if n_total else 0.0,
        "source_counts": dict(source_counts),
        "per_class_f1": per_class_f1,
    }
    if latencies:
        metrics.update(
            {
                "n_timed": len(latencies),
                "latency_median_ms": round(statistics.median(latencies), 2),
                "latency_p90_ms": round(percentile(latencies, 0.9), 2),
                "latency_mean_ms": round(statistics.mean(latencies), 2),
            }
        )
    return metrics


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for row in sorted(rows, key=lambda item: (item["fallback_rows"], -item["macro_f1"], -item["accuracy"])):
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = (
                other["fallback_rows"] <= row["fallback_rows"]
                and other["macro_f1"] >= row["macro_f1"]
                and other["accuracy"] >= row["accuracy"]
            )
            strictly_better = (
                other["fallback_rows"] < row["fallback_rows"]
                or other["macro_f1"] > row["macro_f1"]
                or other["accuracy"] > row["accuracy"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return frontier


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-jsonl", required=True)
    parser.add_argument("--primary-fallback-jsonl", required=True)
    parser.add_argument("--secondary-fallback-jsonl")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--route-decision-field", default="route_decision")
    parser.add_argument("--accept-value", default="accept")
    parser.add_argument("--labels", default="auto")
    parser.add_argument("--candidate-predictions", default="auto")
    parser.add_argument("--secondary-fallback-predictions", default="")
    parser.add_argument("--max-policy-size", type=int, default=None)
    parser.add_argument("--selected-policy", default="")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-metrics-json")
    args = parser.parse_args()

    routing_rows = load_jsonl_by_id(Path(args.routing_jsonl))
    primary_rows = load_jsonl_by_id(Path(args.primary_fallback_jsonl))
    secondary_rows = load_jsonl_by_id(Path(args.secondary_fallback_jsonl)) if args.secondary_fallback_jsonl else None
    if args.candidate_predictions == "auto":
        candidate_predictions = sorted(
            {
                str(row.get("prediction"))
                for row in routing_rows.values()
                if row.get(args.route_decision_field) == args.accept_value and isinstance(row.get("prediction"), str)
            }
        )
    else:
        candidate_predictions = [item.strip() for item in args.candidate_predictions.split(",") if item.strip()]
    secondary_predictions = {item.strip() for item in args.secondary_fallback_predictions.split(",") if item.strip()}
    labels_probe = build_records(
        routing_rows,
        primary_rows,
        secondary_rows,
        set(),
        secondary_predictions,
        args.route_decision_field,
        args.accept_value,
    )
    labels = infer_labels(labels_probe) if args.labels == "auto" else [item.strip() for item in args.labels.split(",") if item.strip()]
    results = []
    for policy in all_label_sets(candidate_predictions, args.max_policy_size):
        records = build_records(
            routing_rows,
            primary_rows,
            secondary_rows,
            policy,
            secondary_predictions,
            args.route_decision_field,
            args.accept_value,
        )
        metrics = compute_metrics(records, labels)
        results.append({"policy": sorted(policy), **{k: v for k, v in metrics.items() if k != "per_class_f1"}})
    output = {
        "routing_jsonl": args.routing_jsonl,
        "primary_fallback_jsonl": args.primary_fallback_jsonl,
        "secondary_fallback_jsonl": args.secondary_fallback_jsonl,
        "route_decision_field": args.route_decision_field,
        "accept_value": args.accept_value,
        "candidate_predictions": candidate_predictions,
        "secondary_fallback_predictions": sorted(secondary_predictions),
        "num_policies": len(results),
        "top_by_macro_f1": sorted(results, key=lambda row: (row["macro_f1"], row["accuracy"]), reverse=True)[:25],
        "top_by_accuracy": sorted(results, key=lambda row: (row["accuracy"], row["macro_f1"]), reverse=True)[:25],
        "pareto_frontier": pareto_frontier(results),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if args.selected_policy:
        selected_policy = {item.strip() for item in args.selected_policy.split(",") if item.strip()}
        selected_records = build_records(
            routing_rows,
            primary_rows,
            secondary_rows,
            selected_policy,
            secondary_predictions,
            args.route_decision_field,
            args.accept_value,
        )
        selected_metrics = compute_metrics(selected_records, labels)
        selected_metrics["policy"] = sorted(selected_policy)
        if args.output_jsonl:
            write_jsonl(Path(args.output_jsonl), selected_records)
        if args.output_metrics_json:
            metrics_path = Path(args.output_metrics_json)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(selected_metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    best = output["top_by_macro_f1"][0]
    print(
        f"wrote {output_path}\n"
        f"num_policies={len(results)} best_policy={best['policy']} "
        f"best_macro_f1={best['macro_f1']:.4f} best_acc={best['accuracy']:.4f} "
        f"fallback_rows={best['fallback_rows']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
