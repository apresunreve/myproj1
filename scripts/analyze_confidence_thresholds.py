#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_HIGHER_IS_BETTER = [
    "first_token_prob",
    "mean_token_prob",
    "min_token_prob",
    "first_token_logit_margin",
    "mean_token_logit_margin",
    "min_token_logit_margin",
    "mean_token_logprob",
    "sum_token_logprob",
]

DEFAULT_LOWER_IS_BETTER = [
    "mean_token_entropy",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _value(row: dict[str, Any], key: str) -> float | None:
    confidence = row.get("confidence", {})
    value = confidence.get(key)
    if value is None:
        return None
    return float(value)


def _quantile_thresholds(values: list[float], max_thresholds: int) -> list[float]:
    if not values:
        return []
    unique_values = sorted(set(values))
    if len(unique_values) <= max_thresholds:
        return unique_values
    ordered = sorted(values)
    thresholds = set()
    for idx in range(max_thresholds):
        position = round(idx * (len(ordered) - 1) / max(1, max_thresholds - 1))
        thresholds.add(ordered[position])
    return sorted(thresholds)


def _summarize_partition(records: list[dict[str, Any]], kept: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    total_errors = sum(1 for row in records if not row.get("correct", False))
    kept_ids = {row.get("id") for row in kept}
    abstained = [row for row in records if row.get("id") not in kept_ids]
    kept_correct = sum(1 for row in kept if row.get("correct", False))
    kept_errors = len(kept) - kept_correct
    abstained_errors = sum(1 for row in abstained if not row.get("correct", False))
    return {
        "n_kept": len(kept),
        "n_abstained": len(abstained),
        "coverage": len(kept) / total if total else 0.0,
        "routing_load": len(abstained) / total if total else 0.0,
        "retained_accuracy": kept_correct / len(kept) if kept else None,
        "kept_error_rate": kept_errors / len(kept) if kept else None,
        "abstained_error_rate": abstained_errors / len(abstained) if abstained else None,
        "errors_caught": abstained_errors,
        "total_errors": total_errors,
        "error_capture_rate": abstained_errors / total_errors if total_errors else None,
    }


def sweep_metric(
    records: list[dict[str, Any]],
    key: str,
    *,
    direction: str,
    max_thresholds: int,
) -> list[dict[str, Any]]:
    values = [_value(row, key) for row in records]
    thresholds = _quantile_thresholds([value for value in values if value is not None], max_thresholds)
    rows = []
    for threshold in thresholds:
        if direction == "higher":
            kept = [row for row in records if (value := _value(row, key)) is not None and value >= threshold]
        elif direction == "lower":
            kept = [row for row in records if (value := _value(row, key)) is not None and value <= threshold]
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        summary = _summarize_partition(records, kept)
        rows.append(
            {
                "metric": key,
                "direction": direction,
                "threshold": threshold,
                **summary,
            }
        )
    if direction == "higher":
        return sorted(rows, key=lambda row: (-row["coverage"], row["threshold"]))
    return sorted(rows, key=lambda row: (-row["coverage"], -row["threshold"]))


def choose_operating_point(rows: list[dict[str, Any]], target_retained_accuracy: float) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row["n_kept"] > 0
        and row["retained_accuracy"] is not None
        and row["retained_accuracy"] >= target_retained_accuracy
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row["coverage"],
            row["retained_accuracy"],
            row["error_capture_rate"] or 0.0,
        ),
    )


def analyze(
    records: list[dict[str, Any]],
    *,
    target_retained_accuracy: float,
    max_thresholds: int,
) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for row in records if row.get("correct", False))
    sweeps = []
    best_by_metric = {}
    metric_specs = [(key, "higher") for key in DEFAULT_HIGHER_IS_BETTER]
    metric_specs.extend((key, "lower") for key in DEFAULT_LOWER_IS_BETTER)
    for key, direction in metric_specs:
        metric_rows = sweep_metric(records, key, direction=direction, max_thresholds=max_thresholds)
        if not metric_rows:
            continue
        sweeps.extend(metric_rows)
        best_by_metric[key] = choose_operating_point(metric_rows, target_retained_accuracy)

    viable = [row for row in best_by_metric.values() if row is not None]
    overall_best = None
    if viable:
        overall_best = max(
            viable,
            key=lambda row: (
                row["coverage"],
                row["retained_accuracy"],
                row["error_capture_rate"] or 0.0,
            ),
        )
    return {
        "n_total": total,
        "base_accuracy": correct / total if total else 0.0,
        "target_retained_accuracy": target_retained_accuracy,
        "overall_best": overall_best,
        "best_by_metric": best_by_metric,
        "sweeps": sweeps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--target-retained-accuracy", type=float, default=0.70)
    parser.add_argument("--max-thresholds", type=int, default=50)
    args = parser.parse_args()

    records = load_jsonl(Path(args.input_jsonl))
    result = analyze(
        records,
        target_retained_accuracy=args.target_retained_accuracy,
        max_thresholds=args.max_thresholds,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    best = result["overall_best"]
    if best is None:
        print(
            f"wrote {output_path}\n"
            f"base_accuracy={result['base_accuracy']:.4f}; no threshold met "
            f"target_retained_accuracy={args.target_retained_accuracy:.4f}",
            flush=True,
        )
        return
    print(
        f"wrote {output_path}\n"
        f"base_accuracy={result['base_accuracy']:.4f} "
        f"best_metric={best['metric']} threshold={best['threshold']:.6g} "
        f"coverage={best['coverage']:.4f} retained_accuracy={best['retained_accuracy']:.4f} "
        f"routing_load={best['routing_load']:.4f} error_capture_rate={best['error_capture_rate']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
