#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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


def infer_labels(rows_by_system: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for rows in rows_by_system.values():
        for row in rows.values():
            for value in (row.get("gold_label"), prediction(row)):
                if isinstance(value, str) and value not in seen:
                    seen.add(value)
                    labels.append(value)
    return labels


def compute_metrics(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0, "correct": 0})
    n_correct = 0
    for row in rows:
        gold = row.get("gold_label")
        pred = prediction(row)
        per_class[gold]["support"] += 1
        if pred == gold:
            n_correct += 1
            per_class[gold]["tp"] += 1
            per_class[gold]["correct"] += 1
        elif pred is not None:
            per_class[gold]["fn"] += 1
            per_class[pred]["fp"] += 1
        else:
            per_class[gold]["fn"] += 1

    per_class_out = {}
    f1_values = []
    for label in labels:
        stats = per_class[label]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        support = stats["support"]
        per_class_out[label] = {
            **stats,
            "accuracy": stats["correct"] / support if support else 0.0,
            "f1": f1,
        }
        f1_values.append(f1)

    return {
        "n_total": len(rows),
        "accuracy": n_correct / len(rows) if rows else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class_out,
        "pred_distribution": dict(Counter(prediction(row) or "__unparsed__" for row in rows)),
    }


def compare_to_reference(
    *,
    reference_name: str,
    target_name: str,
    reference_rows: dict[str, dict[str, Any]],
    target_rows: dict[str, dict[str, Any]],
    ids: list[str],
    labels: list[str],
) -> dict[str, Any]:
    per_gold = defaultdict(lambda: {"support": 0, "same_prediction": 0, "ref_only_correct": 0, "target_only_correct": 0, "both_correct": 0, "both_wrong": 0})
    transition_counts: Counter[tuple[str | None, str | None, str]] = Counter()
    examples = {"target_only_correct": [], "ref_only_correct": []}

    for row_id in ids:
        ref = reference_rows[row_id]
        target = target_rows[row_id]
        gold = str(ref.get("gold_label"))
        ref_pred = prediction(ref)
        target_pred = prediction(target)
        ref_correct = ref_pred == gold
        target_correct = target_pred == gold
        stats = per_gold[gold]
        stats["support"] += 1
        if ref_pred == target_pred:
            stats["same_prediction"] += 1
        if ref_correct and target_correct:
            stats["both_correct"] += 1
        elif ref_correct and not target_correct:
            stats["ref_only_correct"] += 1
            if len(examples["ref_only_correct"]) < 25:
                examples["ref_only_correct"].append({"id": row_id, "gold": gold, reference_name: ref_pred, target_name: target_pred})
        elif target_correct and not ref_correct:
            stats["target_only_correct"] += 1
            if len(examples["target_only_correct"]) < 25:
                examples["target_only_correct"].append({"id": row_id, "gold": gold, reference_name: ref_pred, target_name: target_pred})
        else:
            stats["both_wrong"] += 1
        if ref_pred != target_pred:
            transition_counts[(ref_pred, target_pred, gold)] += 1

    per_gold_out = {}
    for label in labels:
        stats = per_gold[label]
        support = stats["support"]
        per_gold_out[label] = {
            **stats,
            "net_target_correct_gain": stats["target_only_correct"] - stats["ref_only_correct"],
            "prediction_change_rate": 1.0 - (stats["same_prediction"] / support if support else 0.0),
        }

    return {
        "target": target_name,
        "reference": reference_name,
        "per_gold": per_gold_out,
        "top_prediction_transitions": [
            {"reference_prediction": ref, "target_prediction": target, "gold_label": gold, "count": count}
            for (ref, target, gold), count in transition_counts.most_common(50)
        ],
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", action="append", required=True, type=parse_system_arg, help="NAME=PATH JSONL")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--labels", default="auto")
    args = parser.parse_args()

    rows_by_system = {name: load_jsonl_by_id(path) for name, path in args.system}
    if args.reference not in rows_by_system:
        raise SystemExit(f"unknown reference: {args.reference}")
    common_ids = sorted(set.intersection(*(set(rows) for rows in rows_by_system.values())))
    if not common_ids:
        raise SystemExit("systems have no common ids")
    labels = infer_labels(rows_by_system) if args.labels == "auto" else [item.strip() for item in args.labels.split(",") if item.strip()]

    filtered = {
        name: [rows[row_id] for row_id in common_ids]
        for name, rows in rows_by_system.items()
    }
    output = {
        "systems": {name: str(path) for name, path in args.system},
        "reference": args.reference,
        "common_rows": len(common_ids),
        "labels": labels,
        "metrics": {name: compute_metrics(rows, labels) for name, rows in filtered.items()},
        "comparisons_vs_reference": {
            name: compare_to_reference(
                reference_name=args.reference,
                target_name=name,
                reference_rows=rows_by_system[args.reference],
                target_rows=rows,
                ids=common_ids,
                labels=labels,
            )
            for name, rows in rows_by_system.items()
            if name != args.reference
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path} common_rows={len(common_ids)} reference={args.reference}", flush=True)
    for name, metrics in output["metrics"].items():
        print(f"{name}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
