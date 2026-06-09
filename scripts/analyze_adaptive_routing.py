#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def confidence_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get("confidence", {}).get(metric)
    if value is None:
        return None
    return float(value)


def should_accept(row: dict[str, Any], metric: str, threshold: float, direction: str) -> bool:
    value = confidence_value(row, metric)
    if value is None:
        return False
    if direction == "higher":
        return value >= threshold
    if direction == "lower":
        return value <= threshold
    raise ValueError(f"Unsupported direction: {direction}")


def accuracy(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row.get("correct", False)) / len(rows)


def per_class_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int | float | None]]:
    labels = sorted({str(row.get("gold_label")) for row in rows})
    summary = {}
    for label in labels:
        label_rows = [row for row in rows if row.get("gold_label") == label]
        accepted_rows = [row for row in label_rows if row.get("route_decision") == "accept"]
        routed_rows = [row for row in label_rows if row.get("route_decision") == "route"]
        summary[label] = {
            "n_total": len(label_rows),
            "n_accepted": len(accepted_rows),
            "n_routed": len(routed_rows),
            "coverage": len(accepted_rows) / len(label_rows) if label_rows else None,
            "accepted_accuracy": accuracy(accepted_rows),
            "routed_error_rate": (
                sum(1 for row in routed_rows if not row.get("correct", False)) / len(routed_rows)
                if routed_rows
                else None
            ),
        }
    return summary


def analyze(
    records: list[dict[str, Any]],
    *,
    metric: str,
    threshold: float,
    direction: str,
    fallback_accuracy: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    routed_records = []
    for row in records:
        accepted = should_accept(row, metric, threshold, direction)
        routed_record = dict(row)
        routed_record["route_decision"] = "accept" if accepted else "route"
        routed_record["route_metric"] = metric
        routed_record["route_threshold"] = threshold
        routed_record["route_metric_value"] = confidence_value(row, metric)
        routed_records.append(routed_record)

    accepted_rows = [row for row in routed_records if row["route_decision"] == "accept"]
    routed_rows = [row for row in routed_records if row["route_decision"] == "route"]
    total = len(routed_records)
    base_correct = sum(1 for row in routed_records if row.get("correct", False))
    accepted_correct = sum(1 for row in accepted_rows if row.get("correct", False))
    routed_correct = sum(1 for row in routed_rows if row.get("correct", False))
    routed_errors = len(routed_rows) - routed_correct
    route_counts = Counter(row["route_decision"] for row in routed_records)

    expected_system_accuracy = None
    if fallback_accuracy is not None:
        expected_system_accuracy = (
            accepted_correct + fallback_accuracy * len(routed_rows)
        ) / total if total else None

    metrics = {
        "metric": metric,
        "threshold": threshold,
        "direction": direction,
        "n_total": total,
        "base_accuracy": base_correct / total if total else None,
        "n_accepted": len(accepted_rows),
        "n_routed": len(routed_rows),
        "coverage": len(accepted_rows) / total if total else None,
        "routing_load": len(routed_rows) / total if total else None,
        "accepted_accuracy": accuracy(accepted_rows),
        "routed_accuracy": accuracy(routed_rows),
        "routed_error_rate": routed_errors / len(routed_rows) if routed_rows else None,
        "errors_routed": routed_errors,
        "total_errors": total - base_correct,
        "error_capture_rate": routed_errors / (total - base_correct) if total - base_correct else None,
        "route_counts": dict(route_counts),
        "fallback_accuracy_assumption": fallback_accuracy,
        "expected_system_accuracy_with_fallback": expected_system_accuracy,
        "per_class": per_class_summary(routed_records),
    }
    return routed_records, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-metrics-json", required=True)
    parser.add_argument("--metric", default="mean_token_logprob")
    parser.add_argument("--threshold", type=float, default=-0.2733916938100265)
    parser.add_argument("--direction", choices=["higher", "lower"], default="higher")
    parser.add_argument(
        "--fallback-accuracy",
        type=float,
        default=None,
        help="Optional assumed accuracy for routed clips, used only for expected system accuracy.",
    )
    args = parser.parse_args()

    records = load_jsonl(Path(args.input_jsonl))
    routed_records, metrics = analyze(
        records,
        metric=args.metric,
        threshold=args.threshold,
        direction=args.direction,
        fallback_accuracy=args.fallback_accuracy,
    )

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in routed_records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    output_metrics = Path(args.output_metrics_json)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(
        f"wrote routed predictions: {output_jsonl}\n"
        f"wrote metrics: {output_metrics}\n"
        f"base_accuracy={metrics['base_accuracy']:.4f} "
        f"accepted_accuracy={metrics['accepted_accuracy']:.4f} "
        f"coverage={metrics['coverage']:.4f} routing_load={metrics['routing_load']:.4f} "
        f"error_capture_rate={metrics['error_capture_rate']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
