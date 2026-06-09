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


def latency(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    value = row.get("latency_ms")
    return float(value) if isinstance(value, int | float) else None


def prediction(row: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if row is None:
        return None, "__missing_fallback__"
    pred = row.get("pred_label") or row.get("prediction")
    text = row.get("pred_text") or pred
    return pred, text


def build_records(
    routing_rows: dict[str, dict[str, Any]],
    low_conf_rows: dict[str, dict[str, Any]],
    rare_rows: dict[str, dict[str, Any]],
    rare_fallback_qwen_predictions: set[str] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for row_id, routing_row in routing_rows.items():
        decision = routing_row.get("rare_fallback_route_decision")
        gold = routing_row.get("gold_label")
        qwen_latency = latency(routing_row)
        fallback_latency = None
        if decision == "accept":
            pred = routing_row.get("prediction")
            text = routing_row.get("pred_text")
            source = "qwen_accept"
        elif rare_fallback_qwen_predictions is not None:
            use_rare = routing_row.get("prediction") in rare_fallback_qwen_predictions
            fallback_row = (rare_rows if use_rare else low_conf_rows).get(row_id)
            pred, text = prediction(fallback_row)
            fallback_latency = latency(fallback_row)
            source = "rare_fallback" if use_rare else "low_conf_fallback"
        elif decision == "route_low_confidence":
            fallback_row = low_conf_rows.get(row_id)
            pred, text = prediction(fallback_row)
            fallback_latency = latency(fallback_row)
            source = "low_conf_fallback"
        elif decision == "route_rare_ambiguity":
            fallback_row = rare_rows.get(row_id)
            pred, text = prediction(fallback_row)
            fallback_latency = latency(fallback_row)
            source = "rare_fallback"
        else:
            pred = None
            text = "__unknown_route_decision__"
            source = "unknown"
        total_latency = qwen_latency + (fallback_latency or 0.0) if qwen_latency is not None else None
        records.append(
            {
                "id": row_id,
                "gold_label": gold,
                "prediction": pred,
                "pred_text": text,
                "correct": pred == gold,
                "source": source,
                "route_decision": decision,
                "qwen_latency_ms": qwen_latency,
                "fallback_latency_ms": fallback_latency,
                "total_latency_ms": total_latency,
            }
        )
    return records


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    n_total = len(records)
    n_correct = sum(1 for row in records if row["correct"])
    pred_counts = Counter(row["prediction"] or "__missing__" for row in records)
    gold_counts = Counter(row["gold_label"] for row in records)
    source_counts = Counter(row["source"] for row in records)
    per_source_correct = Counter(row["source"] for row in records if row["correct"])
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
    for label in KABR_CLASSES:
        stats = per_class[label]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else 0.0
        per_class_f1[label] = {**stats, "f1": f1}
        f1_values.append(f1)

    latencies = [float(row["total_latency_ms"]) for row in records if isinstance(row.get("total_latency_ms"), int | float)]
    latency_summary = {}
    if latencies:
        ordered = sorted(latencies)
        p90_idx = max(0, min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1))
        latency_summary = {
            "latency_mean_ms": round(statistics.mean(ordered), 2),
            "latency_median_ms": round(statistics.median(ordered), 2),
            "latency_p90_ms": round(ordered[p90_idx], 2),
        }

    source_accuracy = {
        source: per_source_correct[source] / count if count else None
        for source, count in source_counts.items()
    }
    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "source_counts": dict(source_counts),
        "source_accuracy": source_accuracy,
        "pred_distribution": dict(pred_counts),
        "gold_distribution": dict(gold_counts),
        "per_class_f1": per_class_f1,
        **latency_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-jsonl", required=True)
    parser.add_argument("--low-confidence-fallback-jsonl", required=True)
    parser.add_argument("--rare-fallback-jsonl", required=True)
    parser.add_argument(
        "--rare-fallback-qwen-predictions",
        nargs="*",
        help=(
            "If set, choose the rare fallback for routed rows whose original "
            "Qwen prediction is in this label list; otherwise choose the "
            "low-confidence fallback. By default, fallback choice follows "
            "rare_fallback_route_decision."
        ),
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-metrics-json", required=True)
    args = parser.parse_args()

    records = build_records(
        load_jsonl_by_id(Path(args.routing_jsonl)),
        load_jsonl_by_id(Path(args.low_confidence_fallback_jsonl)),
        load_jsonl_by_id(Path(args.rare_fallback_jsonl)),
        set(args.rare_fallback_qwen_predictions) if args.rare_fallback_qwen_predictions is not None else None,
    )
    metrics = compute_metrics(records)

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
        f"latency_median_ms={metrics.get('latency_median_ms')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
