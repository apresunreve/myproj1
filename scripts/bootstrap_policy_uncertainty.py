#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import statistics
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prediction(row: dict[str, Any]) -> str | None:
    value = row.get("prediction") or row.get("pred_label")
    return str(value) if value is not None else None


def group_key(row_id: str, mode: str) -> str:
    if mode == "id":
        return row_id
    if mode == "prefix_before_colon":
        return row_id.split(":", 1)[0]
    if mode == "prefix_before_second_colon":
        parts = row_id.split(":")
        return ":".join(parts[:2]) if len(parts) > 1 else row_id
    raise ValueError(f"unknown group mode: {mode}")


def infer_labels(system_rows: dict[str, list[dict[str, Any]]]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for rows in system_rows.values():
        for row in rows:
            for value in (row.get("gold_label"), prediction(row)):
                if isinstance(value, str) and value not in seen:
                    seen.add(value)
                    labels.append(value)
    return labels


def fallback_count(rows: list[dict[str, Any]]) -> int | None:
    if not any("source" in row for row in rows):
        return None
    return sum(1 for row in rows if row.get("source") != "qwen_accept")


def compute_metrics(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    n_total = len(rows)
    n_correct = sum(1 for row in rows if prediction(row) == row.get("gold_label"))
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    for row in rows:
        gold = row.get("gold_label")
        pred = prediction(row)
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

    n_fallback = fallback_count(rows)
    metrics = {
        "n_total": n_total,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class_f1": per_class_f1,
        "source_counts": dict(Counter(str(row.get("source", "__missing__")) for row in rows)),
    }
    if n_fallback is not None:
        metrics["fallback_rows"] = n_fallback
        metrics["fallback_load"] = n_fallback / n_total if n_total else 0.0
    return metrics


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "ci95_low": percentile(values, 0.025),
        "ci95_high": percentile(values, 0.975),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", action="append", required=True, type=parse_system_arg, help="NAME=PATH JSONL system output")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--labels", default="auto")
    parser.add_argument("--reference", help="System name to compare against. Defaults to first system.")
    parser.add_argument("--group-mode", default="prefix_before_colon", choices=["id", "prefix_before_colon", "prefix_before_second_colon"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    system_rows = {name: load_jsonl(path) for name, path in args.system}
    if len(system_rows) < 2:
        raise SystemExit("provide at least two --system arguments")
    row_ids_by_system = {name: {str(row.get("id")) for row in rows} for name, rows in system_rows.items()}
    common_ids = set.intersection(*row_ids_by_system.values())
    if not common_ids:
        raise SystemExit("systems have no common ids")

    filtered_rows = {
        name: [row for row in rows if str(row.get("id")) in common_ids]
        for name, rows in system_rows.items()
    }
    labels = infer_labels(filtered_rows) if args.labels == "auto" else [item.strip() for item in args.labels.split(",") if item.strip()]
    reference = args.reference or next(iter(system_rows))
    if reference not in system_rows:
        raise SystemExit(f"unknown reference system: {reference}")

    by_id = {name: {str(row.get("id")): row for row in rows} for name, rows in filtered_rows.items()}
    groups: dict[str, list[str]] = defaultdict(list)
    for row_id in sorted(common_ids):
        groups[group_key(row_id, args.group_mode)].append(row_id)
    group_names = sorted(groups)
    rng = random.Random(args.seed)

    point_metrics = {name: compute_metrics(rows, labels) for name, rows in filtered_rows.items()}
    boot_metrics = {name: defaultdict(list) for name in filtered_rows}
    boot_per_class = {name: {label: [] for label in labels} for name in filtered_rows}
    boot_diffs = {
        name: {"accuracy": [], "macro_f1": [], "fallback_load": []}
        for name in filtered_rows
        if name != reference
    }

    for _ in range(args.n_bootstrap):
        sampled_groups = [rng.choice(group_names) for _ in group_names]
        sampled_ids = [row_id for name in sampled_groups for row_id in groups[name]]
        sampled = {
            name: [by_id[name][row_id] for row_id in sampled_ids]
            for name in filtered_rows
        }
        sampled_metrics = {name: compute_metrics(rows, labels) for name, rows in sampled.items()}
        for name, metrics in sampled_metrics.items():
            for metric in ("accuracy", "macro_f1", "fallback_load"):
                if metric in metrics:
                    boot_metrics[name][metric].append(float(metrics[metric]))
            for label in labels:
                boot_per_class[name][label].append(float(metrics["per_class_f1"][label]["f1"]))
        ref_metrics = sampled_metrics[reference]
        for name, diffs in boot_diffs.items():
            metrics = sampled_metrics[name]
            for metric in ("accuracy", "macro_f1", "fallback_load"):
                if metric in metrics and metric in ref_metrics:
                    diffs[metric].append(float(metrics[metric]) - float(ref_metrics[metric]))

    output = {
        "systems": {name: str(path) for name, path in args.system},
        "reference": reference,
        "common_rows": len(common_ids),
        "group_mode": args.group_mode,
        "num_groups": len(group_names),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "labels": labels,
        "point_metrics": point_metrics,
        "bootstrap": {
            name: {metric: summarize(values) for metric, values in metrics.items()}
            for name, metrics in boot_metrics.items()
        },
        "per_class_bootstrap_f1": {
            name: {label: summarize(values) for label, values in label_values.items()}
            for name, label_values in boot_per_class.items()
        },
        "paired_differences_vs_reference": {
            name: {
                metric: {
                    **summarize(values),
                    "prob_gt_0": sum(1 for value in values if value > 0.0) / len(values) if values else 0.0,
                    "prob_lt_0": sum(1 for value in values if value < 0.0) / len(values) if values else 0.0,
                }
                for metric, values in diffs.items()
            }
            for name, diffs in boot_diffs.items()
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        f"wrote {output_path}\n"
        f"common_rows={len(common_ids)} groups={len(group_names)} reference={reference}",
        flush=True,
    )
    for name, metrics in point_metrics.items():
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"fallback_load={metrics.get('fallback_load')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
