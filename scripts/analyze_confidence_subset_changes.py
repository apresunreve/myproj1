#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_system_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("system must be NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("system name cannot be empty")
    return name, Path(path)


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


def prediction(row: dict[str, Any]) -> str | None:
    value = row.get("prediction") or row.get("pred_label")
    return str(value) if value is not None else None


def confidence_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get("confidence", {}).get(metric)
    return float(value) if isinstance(value, int | float) else None


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", action="append", required=True, type=parse_system_arg)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--metric", action="append", default=["mean_token_entropy", "first_token_prob", "mean_token_prob"])
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    rows_by_system = {name: load_jsonl_by_id(path) for name, path in args.system}
    if args.reference not in rows_by_system:
        raise SystemExit(f"unknown reference: {args.reference}")
    common_ids = sorted(set.intersection(*(set(rows) for rows in rows_by_system.values())))
    if not common_ids:
        raise SystemExit("systems have no common ids")

    systems = {}
    for name, rows in rows_by_system.items():
        records = [rows[row_id] for row_id in common_ids]
        systems[name] = {
            "accuracy": sum(1 for row in records if prediction(row) == row.get("gold_label")) / len(records),
            "prediction_distribution": {},
            "confidence": {
                metric: summarize([value for row in records if (value := confidence_value(row, metric)) is not None])
                for metric in args.metric
            },
        }
        pred_counts: dict[str, int] = {}
        for row in records:
            pred = prediction(row) or "__unparsed__"
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        systems[name]["prediction_distribution"] = pred_counts

    comparisons = {}
    reference_rows = rows_by_system[args.reference]
    for name, rows in rows_by_system.items():
        if name == args.reference:
            continue
        changed = 0
        target_only_correct = 0
        reference_only_correct = 0
        both_correct = 0
        both_wrong = 0
        metric_deltas: dict[str, list[float]] = {metric: [] for metric in args.metric}
        changed_metric_deltas: dict[str, list[float]] = {metric: [] for metric in args.metric}
        examples = {"target_only_correct": [], "reference_only_correct": []}
        for row_id in common_ids:
            ref = reference_rows[row_id]
            target = rows[row_id]
            gold = ref.get("gold_label")
            ref_pred = prediction(ref)
            target_pred = prediction(target)
            ref_correct = ref_pred == gold
            target_correct = target_pred == gold
            pred_changed = ref_pred != target_pred
            changed += int(pred_changed)
            if ref_correct and target_correct:
                both_correct += 1
            elif target_correct:
                target_only_correct += 1
                if len(examples["target_only_correct"]) < 20:
                    examples["target_only_correct"].append({"id": row_id, "gold": gold, args.reference: ref_pred, name: target_pred})
            elif ref_correct:
                reference_only_correct += 1
                if len(examples["reference_only_correct"]) < 20:
                    examples["reference_only_correct"].append({"id": row_id, "gold": gold, args.reference: ref_pred, name: target_pred})
            else:
                both_wrong += 1
            for metric in args.metric:
                ref_value = confidence_value(ref, metric)
                target_value = confidence_value(target, metric)
                if ref_value is None or target_value is None:
                    continue
                delta = target_value - ref_value
                metric_deltas[metric].append(delta)
                if pred_changed:
                    changed_metric_deltas[metric].append(delta)
        comparisons[name] = {
            "prediction_change_rate": changed / len(common_ids),
            "target_only_correct": target_only_correct,
            "reference_only_correct": reference_only_correct,
            "net_target_correct_gain": target_only_correct - reference_only_correct,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "metric_deltas": {metric: summarize(values) for metric, values in metric_deltas.items()},
            "metric_deltas_when_prediction_changed": {
                metric: summarize(values) for metric, values in changed_metric_deltas.items()
            },
            "examples": examples,
        }

    output = {
        "systems": {name: str(path) for name, path in args.system},
        "reference": args.reference,
        "common_rows": len(common_ids),
        "metrics": args.metric,
        "system_summaries": systems,
        "comparisons_vs_reference": comparisons,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path} common_rows={len(common_ids)} reference={args.reference}", flush=True)
    for name, summary in systems.items():
        print(f"{name}: accuracy={summary['accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
