#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
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
    value = row.get("prediction") or row.get("pred_label")
    return str(value) if value is not None else None


def latency(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    value = row.get("latency_ms") or row.get("total_latency_ms")
    return float(value) if isinstance(value, int | float) else None


def confidence_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get("confidence", {}).get(metric)
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


def parse_threshold_grid(raw: str) -> list[float | None]:
    thresholds: list[float | None] = [None]
    for item in raw.split(","):
        item = item.strip()
        if item:
            thresholds.append(float(item))
    return thresholds


def should_route(value: float | None, threshold: float | None, direction: str) -> bool:
    if value is None or threshold is None:
        return False
    if direction == "higher":
        return value > threshold
    if direction == "lower":
        return value < threshold
    raise ValueError(f"unsupported direction: {direction}")


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
    routing_rows: dict[str, dict[str, Any]],
    fallback_rows: dict[str, dict[str, Any]],
    *,
    thresholds: dict[str, float | None],
    metric: str,
    direction: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_id in sorted(set(routing_rows) & set(fallback_rows)):
        route_row = routing_rows[row_id]
        fallback_row = fallback_rows[row_id]
        route_prediction = prediction(route_row)
        value = confidence_value(route_row, metric)
        threshold = thresholds.get(str(route_prediction))
        use_fallback = should_route(value, threshold, direction)
        if use_fallback:
            pred = prediction(fallback_row)
            source = "fallback"
            fallback_latency = latency(fallback_row)
        else:
            pred = route_prediction
            source = "qwen_accept"
            fallback_latency = None
        route_latency = latency(route_row)
        total_latency = None
        if route_latency is not None:
            total_latency = route_latency + (fallback_latency or 0.0)
        gold = route_row.get("gold_label")
        records.append(
            {
                "id": row_id,
                "gold_label": gold,
                "prediction": pred,
                "correct": pred == gold,
                "source": source,
                "route_prediction": route_prediction,
                "route_metric": metric,
                "route_metric_value": value,
                "route_threshold": threshold,
                "route_latency_ms": route_latency,
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
    fallback_rows = source_counts.get("fallback", 0)
    latencies = [
        float(row["total_latency_ms"])
        for row in records
        if isinstance(row.get("total_latency_ms"), int | float)
    ]
    metrics: dict[str, Any] = {
        "n_total": n_total,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "fallback_rows": fallback_rows,
        "fallback_load": fallback_rows / n_total if n_total else 0.0,
        "source_counts": dict(source_counts),
        "fallback_prediction_counts": dict(Counter(str(row["route_prediction"]) for row in records if row["source"] == "fallback")),
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-jsonl", required=True)
    parser.add_argument("--fallback-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--metric", default="mean_token_entropy")
    parser.add_argument("--direction", choices=["higher", "lower"], default="higher")
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--threshold-grid", required=True)
    parser.add_argument("--labels", default="auto")
    parser.add_argument("--selected-thresholds", default="")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-metrics-json")
    args = parser.parse_args()

    routing_rows = load_jsonl_by_id(Path(args.routing_jsonl))
    fallback_rows = load_jsonl_by_id(Path(args.fallback_jsonl))
    threshold_grid = parse_threshold_grid(args.threshold_grid)
    labels_probe = build_records(
        routing_rows,
        fallback_rows,
        thresholds={prediction: None for prediction in args.predictions},
        metric=args.metric,
        direction=args.direction,
    )
    labels = infer_labels(labels_probe) if args.labels == "auto" else [item.strip() for item in args.labels.split(",") if item.strip()]

    sweep_rows = []
    for values in itertools.product(threshold_grid, repeat=len(args.predictions)):
        thresholds = dict(zip(args.predictions, values, strict=True))
        records = build_records(
            routing_rows,
            fallback_rows,
            thresholds=thresholds,
            metric=args.metric,
            direction=args.direction,
        )
        metrics = compute_metrics(records, labels)
        sweep_rows.append({"thresholds": thresholds, **{key: value for key, value in metrics.items() if key != "per_class_f1"}})
    sweep_rows.sort(key=lambda row: (row["fallback_load"], -row["accuracy"], -row["macro_f1"]))
    output = {
        "routing_jsonl": args.routing_jsonl,
        "fallback_jsonl": args.fallback_jsonl,
        "metric": args.metric,
        "direction": args.direction,
        "predictions": args.predictions,
        "threshold_grid": args.threshold_grid,
        "n_candidates": len(sweep_rows),
        "top_by_accuracy": sorted(sweep_rows, key=lambda row: (row["accuracy"], row["macro_f1"], -row["fallback_load"]), reverse=True)[:25],
        "top_by_macro_f1": sorted(sweep_rows, key=lambda row: (row["macro_f1"], row["accuracy"], -row["fallback_load"]), reverse=True)[:25],
        "low_p90": sorted(
            [row for row in sweep_rows if row.get("fallback_load", 1.0) <= 0.10],
            key=lambda row: (row["macro_f1"], row["accuracy"], -row["fallback_load"]),
            reverse=True,
        )[:25],
        "sweep": sweep_rows,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if args.selected_thresholds:
        thresholds: dict[str, float | None] = {}
        for item in args.selected_thresholds.split(","):
            item = item.strip()
            if not item:
                continue
            label, raw_value = item.split("=", 1)
            thresholds[label.strip()] = None if raw_value.strip().lower() == "none" else float(raw_value)
        for prediction in args.predictions:
            thresholds.setdefault(prediction, None)
        records = build_records(
            routing_rows,
            fallback_rows,
            thresholds=thresholds,
            metric=args.metric,
            direction=args.direction,
        )
        metrics = compute_metrics(records, labels)
        metrics["thresholds"] = thresholds
        if args.output_jsonl:
            write_jsonl(Path(args.output_jsonl), records)
        if args.output_metrics_json:
            metrics_path = Path(args.output_metrics_json)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    best = output["top_by_macro_f1"][0]
    print(
        f"wrote {output_path}\n"
        f"best_macro_f1={best['macro_f1']:.4f} accuracy={best['accuracy']:.4f} "
        f"fallback_load={best['fallback_load']:.4f} p90={best.get('latency_p90_ms')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
