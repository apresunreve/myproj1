#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("fallback name cannot be empty")
    return name, Path(path)


def parse_mapping(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError("mapping entries must be LABEL=FALLBACK_NAME")
        label, fallback_name = item.split("=", 1)
        label = label.strip()
        fallback_name = fallback_name.strip()
        if not label or not fallback_name:
            raise argparse.ArgumentTypeError("mapping labels and fallback names cannot be empty")
        mapping[label] = fallback_name
    return mapping


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


def pred(row: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if row is None:
        return None, "__missing_fallback__"
    value = row.get("pred_label") or row.get("prediction")
    text = row.get("pred_text") or value
    return str(value) if value is not None else None, str(text) if text is not None else None


def latency(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    value = row.get("latency_ms")
    return float(value) if isinstance(value, int | float) else None


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
    fallback_rows: dict[str, dict[str, dict[str, Any]]],
    default_fallback: str,
    prediction_fallback_map: dict[str, str],
    route_decision_field: str,
    accept_value: str,
) -> list[dict[str, Any]]:
    records = []
    for row_id, routing_row in routing_rows.items():
        gold = routing_row.get("gold_label")
        qwen_prediction = routing_row.get("prediction")
        route_decision = routing_row.get(route_decision_field)
        qwen_latency = latency(routing_row)
        fallback_latency = None
        fallback_name = None
        if route_decision == accept_value:
            prediction = qwen_prediction
            pred_text = routing_row.get("pred_text") or prediction
            source = "qwen_accept"
        else:
            fallback_name = prediction_fallback_map.get(str(qwen_prediction), default_fallback)
            fallback_row = fallback_rows[fallback_name].get(row_id)
            prediction, pred_text = pred(fallback_row)
            fallback_latency = latency(fallback_row)
            source = f"fallback:{fallback_name}"
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
                "fallback_name": fallback_name,
                "qwen_prediction": qwen_prediction,
                "route_decision": route_decision,
                "qwen_latency_ms": qwen_latency,
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
    fallback_rows = sum(count for source, count in source_counts.items() if source != "qwen_accept")
    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "fallback_rows": fallback_rows,
        "fallback_load": fallback_rows / n_total if n_total else 0.0,
        "source_counts": dict(source_counts),
        "fallback_prediction_counts": dict(
            Counter(str(row["qwen_prediction"]) for row in records if row["source"] != "qwen_accept")
        ),
        "per_class_f1": per_class_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-jsonl", required=True)
    parser.add_argument("--fallback", action="append", required=True, type=parse_named_path, help="NAME=PATH")
    parser.add_argument("--default-fallback", required=True)
    parser.add_argument("--prediction-fallback", action="append", default=[], help="QWEN_PREDICTION=FALLBACK_NAME")
    parser.add_argument("--route-decision-field", default="route_decision")
    parser.add_argument("--accept-value", default="accept")
    parser.add_argument("--labels", default="auto")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-metrics-json", required=True)
    args = parser.parse_args()

    fallback_paths = dict(args.fallback)
    if args.default_fallback not in fallback_paths:
        raise SystemExit(f"default fallback {args.default_fallback!r} was not provided")
    prediction_fallback_map = parse_mapping(args.prediction_fallback)
    unknown = sorted(set(prediction_fallback_map.values()) - set(fallback_paths))
    if unknown:
        raise SystemExit(f"mapping refers to unknown fallback names: {unknown}")

    records = build_records(
        load_jsonl_by_id(Path(args.routing_jsonl)),
        {name: load_jsonl_by_id(path) for name, path in fallback_paths.items()},
        args.default_fallback,
        prediction_fallback_map,
        args.route_decision_field,
        args.accept_value,
    )
    labels = infer_labels(records) if args.labels == "auto" else [item.strip() for item in args.labels.split(",") if item.strip()]
    metrics = compute_metrics(records, labels)
    metrics["labels"] = labels
    metrics["default_fallback"] = args.default_fallback
    metrics["prediction_fallback_map"] = prediction_fallback_map

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
