#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from sweep_confidence_fallback_system import (
    confidence_value,
    infer_labels,
    latency,
    load_jsonl_by_id,
    prediction,
)


def stable_fold(row_id: str, num_folds: int, seed: str) -> int:
    digest = hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % num_folds


def parse_threshold_grid(raw: str) -> list[float | None]:
    thresholds: list[float | None] = [None]
    for item in raw.split(","):
        item = item.strip()
        if item:
            thresholds.append(float(item))
    return thresholds


def make_examples(
    routing_rows: dict[str, dict[str, Any]],
    fallback_rows: dict[str, dict[str, Any]],
    row_ids: list[str],
    *,
    metric: str,
    num_folds: int,
    seed: str,
) -> list[dict[str, Any]]:
    examples = []
    for row_id in row_ids:
        route_row = routing_rows[row_id]
        fallback_row = fallback_rows[row_id]
        route_latency = latency(route_row)
        fallback_latency = latency(fallback_row)
        examples.append(
            {
                "id": row_id,
                "fold": stable_fold(row_id, num_folds, seed),
                "gold": route_row.get("gold_label"),
                "route_prediction": prediction(route_row),
                "fallback_prediction": prediction(fallback_row),
                "value": confidence_value(route_row, metric),
                "route_latency": route_latency,
                "fallback_latency": fallback_latency,
            }
        )
    return examples


def make_arrays(examples: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    unknown = len(labels)
    return {
        "fold": np.array([item["fold"] for item in examples], dtype=np.int16),
        "gold": np.array([label_to_idx.get(item["gold"], unknown) for item in examples], dtype=np.int16),
        "route_prediction": np.array([label_to_idx.get(item["route_prediction"], unknown) for item in examples], dtype=np.int16),
        "fallback_prediction": np.array([label_to_idx.get(item["fallback_prediction"], unknown) for item in examples], dtype=np.int16),
        "value": np.array(
            [
                float(item["value"]) if isinstance(item["value"], int | float) else np.nan
                for item in examples
            ],
            dtype=np.float32,
        ),
        "route_latency": np.array(
            [
                float(item["route_latency"]) if isinstance(item["route_latency"], int | float) else np.nan
                for item in examples
            ],
            dtype=np.float32,
        ),
        "fallback_latency": np.array(
            [
                float(item["fallback_latency"]) if isinstance(item["fallback_latency"], int | float) else np.nan
                for item in examples
            ],
            dtype=np.float32,
        ),
        "label_to_idx": label_to_idx,
        "num_labels": len(labels),
    }


def slice_arrays(arrays: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {
        key: value[mask] if isinstance(value, np.ndarray) else value
        for key, value in arrays.items()
    }


def compute_array_metrics(
    arrays: dict[str, Any],
    labels: list[str],
    *,
    thresholds: dict[str, float | None],
    direction: str,
) -> dict[str, Any]:
    n_total = int(arrays["gold"].shape[0])
    use_fallback = np.zeros(n_total, dtype=bool)
    values = arrays["value"]
    label_to_idx = arrays["label_to_idx"]
    for label, threshold in thresholds.items():
        if threshold is None or label not in label_to_idx:
            continue
        label_mask = arrays["route_prediction"] == label_to_idx[label]
        value_mask = np.isfinite(values)
        if direction == "higher":
            use_fallback |= label_mask & value_mask & (values > threshold)
        elif direction == "lower":
            use_fallback |= label_mask & value_mask & (values < threshold)
        else:
            raise ValueError(f"unsupported direction: {direction}")

    pred = np.where(use_fallback, arrays["fallback_prediction"], arrays["route_prediction"])
    gold = arrays["gold"]
    n_correct = int(np.count_nonzero(pred == gold))
    fallback_rows = int(np.count_nonzero(use_fallback))

    f1_values = []
    for label_idx in range(len(labels)):
        tp = int(np.count_nonzero((pred == label_idx) & (gold == label_idx)))
        fp = int(np.count_nonzero((pred == label_idx) & (gold != label_idx)))
        fn = int(np.count_nonzero((pred != label_idx) & (gold == label_idx)))
        f1_values.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)

    route_latency = arrays["route_latency"]
    fallback_latency = np.where(np.isfinite(arrays["fallback_latency"]), arrays["fallback_latency"], 0.0)
    latencies_array = route_latency + np.where(use_fallback, fallback_latency, 0.0)
    latencies = latencies_array[np.isfinite(latencies_array)]
    metrics: dict[str, Any] = {
        "n_total": n_total,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "fallback_rows": fallback_rows,
        "fallback_load": fallback_rows / n_total if n_total else 0.0,
    }
    if latencies.size:
        metrics.update(
            {
                "n_timed": int(latencies.size),
                "latency_median_ms": round(float(np.median(latencies)), 2),
                "latency_p90_ms": round(float(np.quantile(latencies, 0.9)), 2),
                "latency_mean_ms": round(float(np.mean(latencies)), 2),
            }
        )
    return metrics


def select_candidate(
    candidates: list[dict[str, Any]],
    *,
    max_fallback_load: float | None,
    objective: str,
) -> dict[str, Any]:
    pool = candidates
    if max_fallback_load is not None:
        limited = [row for row in candidates if row["fallback_load"] <= max_fallback_load]
        if limited:
            pool = limited
    if objective == "macro_f1":
        key = lambda row: (row["macro_f1"], row["accuracy"], -row["fallback_load"])
    elif objective == "accuracy":
        key = lambda row: (row["accuracy"], row["macro_f1"], -row["fallback_load"])
    else:
        raise ValueError(f"unsupported objective: {objective}")
    return max(pool, key=key)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


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
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", default="confidence-split-v1")
    parser.add_argument("--objective", choices=["macro_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--max-fallback-load", type=float, default=0.10)
    args = parser.parse_args()

    routing_rows = load_jsonl_by_id(Path(args.routing_jsonl))
    fallback_rows = load_jsonl_by_id(Path(args.fallback_jsonl))
    row_ids = sorted(set(routing_rows) & set(fallback_rows))
    if args.num_folds < 2:
        raise ValueError("--num-folds must be at least 2")
    if not row_ids:
        raise ValueError("no overlapping row ids between routing and fallback JSONLs")

    labels_probe = [
        {"gold_label": routing_rows[row_id].get("gold_label"), "prediction": prediction(routing_rows[row_id])}
        for row_id in row_ids
    ]
    labels = infer_labels(labels_probe) if args.labels == "auto" else [item.strip() for item in args.labels.split(",") if item.strip()]
    threshold_grid = parse_threshold_grid(args.threshold_grid)
    examples = make_examples(
        routing_rows,
        fallback_rows,
        row_ids,
        metric=args.metric,
        num_folds=args.num_folds,
        seed=args.seed,
    )
    arrays = make_arrays(examples, labels)

    fold_outputs = []
    for fold_idx in range(args.num_folds):
        train_arrays = slice_arrays(arrays, arrays["fold"] != fold_idx)
        validation_arrays = slice_arrays(arrays, arrays["fold"] == fold_idx)

        candidates = []
        for values in itertools.product(threshold_grid, repeat=len(args.predictions)):
            thresholds = dict(zip(args.predictions, values, strict=True))
            train_metrics = compute_array_metrics(
                train_arrays,
                labels,
                thresholds=thresholds,
                direction=args.direction,
            )
            candidates.append(
                {
                    "thresholds": thresholds,
                    "accuracy": train_metrics["accuracy"],
                    "macro_f1": train_metrics["macro_f1"],
                    "fallback_load": train_metrics["fallback_load"],
                    "latency_p90_ms": train_metrics.get("latency_p90_ms"),
                }
            )

        selected = select_candidate(
            candidates,
            max_fallback_load=args.max_fallback_load,
            objective=args.objective,
        )
        validation_metrics = compute_array_metrics(
            validation_arrays,
            labels,
            thresholds=selected["thresholds"],
            direction=args.direction,
        )
        fold_outputs.append(
            {
                "fold": fold_idx,
                "n_train": int(train_arrays["gold"].shape[0]),
                "n_validation": int(validation_arrays["gold"].shape[0]),
                "selected_train": selected,
                "validation": validation_metrics,
            }
        )

    summary_keys = [
        "accuracy",
        "macro_f1",
        "fallback_load",
        "latency_median_ms",
        "latency_p90_ms",
        "latency_mean_ms",
    ]
    summary = {
        key: summarize(
            [
                float(fold["validation"][key])
                for fold in fold_outputs
                if isinstance(fold["validation"].get(key), int | float)
            ]
        )
        for key in summary_keys
    }
    output = {
        "routing_jsonl": args.routing_jsonl,
        "fallback_jsonl": args.fallback_jsonl,
        "metric": args.metric,
        "direction": args.direction,
        "predictions": args.predictions,
        "threshold_grid": args.threshold_grid,
        "num_folds": args.num_folds,
        "seed": args.seed,
        "objective": args.objective,
        "max_fallback_load": args.max_fallback_load,
        "n_total": len(row_ids),
        "summary": summary,
        "folds": fold_outputs,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        f"wrote {output_path}\n"
        f"holdout_accuracy_mean={summary['accuracy']['mean']:.4f} "
        f"holdout_macro_f1_mean={summary['macro_f1']['mean']:.4f} "
        f"holdout_fallback_load_mean={summary['fallback_load']['mean']:.4f} "
        f"holdout_p90_mean={summary['latency_p90_ms']['mean']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
