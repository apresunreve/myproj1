#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
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


def prediction(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    value = row.get("pred_label") or row.get("prediction")
    return str(value) if value is not None else None


def latency(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    value = row.get("latency_ms") or row.get("total_latency_ms")
    return float(value) if isinstance(value, int | float) else None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def infer_labels(records: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for row in records:
        for value in (row.get("gold_label"), row.get("prediction")):
            if isinstance(value, str) and value not in seen:
                seen.add(value)
                labels.append(value)
    return labels


def build_records(
    first_rows: dict[str, dict[str, Any]],
    second_rows: dict[str, dict[str, Any]],
    fallback_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    common_ids = sorted(set(first_rows) & set(second_rows) & set(fallback_rows))
    records: list[dict[str, Any]] = []
    for row_id in common_ids:
        first_row = first_rows[row_id]
        second_row = second_rows[row_id]
        fallback_row = fallback_rows[row_id]
        first_pred = prediction(first_row)
        second_pred = prediction(second_row)
        agree = first_pred is not None and first_pred == second_pred
        if agree:
            pred = second_pred
            source = "qwen_accept"
            fallback_latency = None
        else:
            pred = prediction(fallback_row)
            source = "fallback"
            fallback_latency = latency(fallback_row)
        first_latency = latency(first_row)
        second_latency = latency(second_row)
        total_latency = None
        if first_latency is not None and second_latency is not None:
            total_latency = first_latency + second_latency + (fallback_latency or 0.0)
        gold = second_row.get("gold_label") or first_row.get("gold_label") or fallback_row.get("gold_label")
        records.append(
            {
                "id": row_id,
                "gold_label": gold,
                "prediction": pred,
                "correct": pred == gold,
                "source": source,
                "first_prediction": first_pred,
                "second_prediction": second_pred,
                "agreement": agree,
                "first_latency_ms": first_latency,
                "second_latency_ms": second_latency,
                "fallback_latency_ms": fallback_latency,
                "total_latency_ms": total_latency,
            }
        )
    return records


def compute_metrics(records: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    n_total = len(records)
    n_correct = sum(1 for row in records if row["correct"])
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
    f1_values = []
    per_class_f1 = {}
    for label in labels:
        stats = per_class[label]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        per_class_f1[label] = {**stats, "f1": f1}
        f1_values.append(f1)
    source_counts = Counter(row["source"] for row in records)
    latencies = [
        float(row["total_latency_ms"])
        for row in records
        if isinstance(row.get("total_latency_ms"), int | float)
    ]
    metrics: dict[str, Any] = {
        "n_total": n_total,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "source_counts": dict(source_counts),
        "fallback_rows": source_counts.get("fallback", 0),
        "fallback_load": source_counts.get("fallback", 0) / n_total if n_total else 0.0,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-jsonl", required=True)
    parser.add_argument("--second-jsonl", required=True)
    parser.add_argument("--fallback-jsonl", required=True)
    parser.add_argument("--labels", default="auto")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-metrics-json", required=True)
    args = parser.parse_args()

    records = build_records(
        load_jsonl_by_id(Path(args.first_jsonl)),
        load_jsonl_by_id(Path(args.second_jsonl)),
        load_jsonl_by_id(Path(args.fallback_jsonl)),
    )
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
        f"wrote {output_jsonl}\n"
        f"wrote {output_metrics}\n"
        f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"fallback_load={metrics['fallback_load']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
