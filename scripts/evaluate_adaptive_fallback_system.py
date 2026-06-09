#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


KABR_CLASSES = ["Walk", "Graze", "Browse", "Head Up", "Auto-Groom", "Trot", "Run", "Occluded"]


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
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


def _latency(row: dict[str, Any]) -> float | None:
    value = row.get("latency_ms")
    if isinstance(value, int | float):
        return float(value)
    return None


def infer_labels(records: list[dict[str, Any]]) -> list[str]:
    labels = []
    seen = set()
    for row in records:
        for key in ("gold_label", "prediction"):
            value = row.get(key)
            if isinstance(value, str) and value not in seen:
                seen.add(value)
                labels.append(value)
    return labels


def build_system_records(
    routing_rows: dict[str, dict[str, Any]],
    fallback_rows: dict[str, dict[str, Any]],
    route_decision_field: str,
    accept_value: str,
) -> list[dict[str, Any]]:
    records = []
    for row_id, routing_row in routing_rows.items():
        decision = routing_row.get(route_decision_field)
        gold = routing_row.get("gold_label")
        qwen_latency = _latency(routing_row)
        if decision == accept_value:
            prediction = routing_row.get("prediction")
            pred_text = routing_row.get("pred_text")
            source = "qwen_accept"
            fallback_latency = None
        else:
            fallback_row = fallback_rows.get(row_id)
            if fallback_row is None:
                prediction = None
                pred_text = "__missing_fallback__"
                fallback_latency = None
            else:
                prediction = fallback_row.get("pred_label") or fallback_row.get("prediction")
                pred_text = fallback_row.get("pred_text") or prediction
                fallback_latency = _latency(fallback_row)
            source = "fallback"
        total_latency = None
        if qwen_latency is not None:
            total_latency = qwen_latency + (fallback_latency or 0.0)
        records.append(
            {
                "id": row_id,
                "gold_label": gold,
                "prediction": prediction,
                "pred_text": pred_text,
                "correct": prediction == gold,
                "source": source,
                "route_decision": decision,
                "qwen_latency_ms": qwen_latency,
                "fallback_latency_ms": fallback_latency,
                "total_latency_ms": total_latency,
            }
        )
    return records


def compute_metrics(records: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    n_total = len(records)
    n_correct = sum(1 for row in records if row["correct"])
    pred_counts = Counter(row["prediction"] or "__missing__" for row in records)
    gold_counts = Counter(row["gold_label"] for row in records)
    source_counts = Counter(row["source"] for row in records)
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    for row in records:
        gold = row["gold_label"]
        pred = row["prediction"]
        per_class[gold]["support"] += 1
        if pred == gold:
            per_class[gold]["tp"] += 1
        elif pred is not None:
            per_class[gold]["fn"] += 1
            per_class[pred]["fp"] += 1
        else:
            per_class[gold]["fn"] += 1

    per_class_f1 = {}
    f1_values = []
    for label in labels:
        stats = per_class[label]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else 0.0
        per_class_f1[label] = {**stats, "f1": f1}
        f1_values.append(f1)

    latencies = [row["total_latency_ms"] for row in records if isinstance(row.get("total_latency_ms"), int | float)]
    latency_summary = {}
    if latencies:
        ordered = sorted(float(value) for value in latencies)
        p90_idx = max(0, min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1))
        latency_summary = {
            "latency_mean_ms": round(statistics.mean(ordered), 2),
            "latency_median_ms": round(statistics.median(ordered), 2),
            "latency_p90_ms": round(ordered[p90_idx], 2),
        }

    fallback_records = [row for row in records if row["source"] == "fallback"]
    accepted_records = [row for row in records if row["source"] == "qwen_accept"]
    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "source_counts": dict(source_counts),
        "accepted_accuracy": (
            sum(1 for row in accepted_records if row["correct"]) / len(accepted_records)
            if accepted_records
            else None
        ),
        "fallback_accuracy": (
            sum(1 for row in fallback_records if row["correct"]) / len(fallback_records)
            if fallback_records
            else None
        ),
        "pred_distribution": dict(pred_counts),
        "gold_distribution": dict(gold_counts),
        "per_class_f1": per_class_f1,
        **latency_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-jsonl", required=True)
    parser.add_argument("--fallback-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-metrics-json", required=True)
    parser.add_argument("--route-decision-field", default="rare_fallback_route_decision")
    parser.add_argument("--accept-value", default="accept")
    parser.add_argument("--labels", default="auto", help="Use 'auto' or comma-separated labels.")
    args = parser.parse_args()

    routing_rows = load_jsonl_by_id(Path(args.routing_jsonl))
    fallback_rows = load_jsonl_by_id(Path(args.fallback_jsonl))
    records = build_system_records(routing_rows, fallback_rows, args.route_decision_field, args.accept_value)
    labels = infer_labels(records) if args.labels == "auto" else [item.strip() for item in args.labels.split(",") if item.strip()]
    metrics = compute_metrics(records, labels)
    metrics["labels"] = labels

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    output_metrics = Path(args.output_metrics_json)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        f"wrote system records: {output_jsonl}\n"
        f"wrote metrics: {output_metrics}\n"
        f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"fallback_accuracy={metrics['fallback_accuracy']:.4f} "
        f"latency_median_ms={metrics.get('latency_median_ms')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
