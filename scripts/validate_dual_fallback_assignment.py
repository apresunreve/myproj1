#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
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


def pred(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    value = row.get("pred_label") or row.get("prediction")
    return str(value) if value is not None else None


def latency(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    value = row.get("latency_ms")
    return float(value) if isinstance(value, int | float) else None


def stable_bucket(row_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{row_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def infer_labels(records: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for row in records:
        for key in ("gold_label", "prediction"):
            value = row.get(key)
            if isinstance(value, str) and value not in seen:
                seen.add(value)
                labels.append(value)
    return labels


def all_label_sets(labels: list[str], max_size: int | None) -> list[set[str]]:
    largest = len(labels) if max_size is None else min(max_size, len(labels))
    return [set(combo) for size in range(largest + 1) for combo in combinations(labels, size)]


def build_records(
    ids: set[str] | None,
    routing_rows: dict[str, dict[str, Any]],
    primary_rows: dict[str, dict[str, Any]],
    secondary_rows: dict[str, dict[str, Any]],
    secondary_qwen_predictions: set[str],
    route_decision_field: str,
    accept_value: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_id, routing_row in routing_rows.items():
        if ids is not None and row_id not in ids:
            continue
        gold = routing_row.get("gold_label")
        qwen_prediction = routing_row.get("prediction")
        route_decision = routing_row.get(route_decision_field)
        qwen_latency = latency(routing_row)
        if route_decision == accept_value:
            prediction = qwen_prediction
            source = "qwen_accept"
            fallback_latency = None
        elif qwen_prediction in secondary_qwen_predictions:
            fallback_row = secondary_rows.get(row_id)
            prediction = pred(fallback_row)
            source = "secondary_fallback"
            fallback_latency = latency(fallback_row)
        else:
            fallback_row = primary_rows.get(row_id)
            prediction = pred(fallback_row)
            source = "primary_fallback"
            fallback_latency = latency(fallback_row)
        total_latency = None
        if qwen_latency is not None:
            total_latency = qwen_latency + (fallback_latency or 0.0)
        records.append(
            {
                "id": row_id,
                "gold_label": gold,
                "prediction": prediction,
                "correct": prediction == gold,
                "source": source,
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
    f1_values = []
    per_class_f1 = {}
    for label in labels:
        stats = per_class[label]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        per_class_f1[label] = {**stats, "f1": f1}
        f1_values.append(f1)
    latencies = sorted(
        float(row["total_latency_ms"]) for row in records if isinstance(row.get("total_latency_ms"), int | float)
    )
    latency_summary = {}
    if latencies:
        p90_idx = max(0, min(len(latencies) - 1, math.ceil(0.9 * len(latencies)) - 1))
        latency_summary = {
            "latency_median_ms": latencies[len(latencies) // 2],
            "latency_p90_ms": latencies[p90_idx],
        }
    fallback_records = [row for row in records if row["source"] != "qwen_accept"]
    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "source_counts": dict(Counter(row["source"] for row in records)),
        "fallback_accuracy": (
            sum(1 for row in fallback_records if row["correct"]) / len(fallback_records)
            if fallback_records
            else None
        ),
        "secondary_qwen_prediction_counts": dict(
            Counter(str(row["qwen_prediction"]) for row in records if row["source"] == "secondary_fallback")
        ),
        "per_class_f1": per_class_f1,
        **latency_summary,
    }


def objective(metrics: dict[str, Any], accuracy_weight: float) -> float:
    return float(metrics["macro_f1"]) + accuracy_weight * float(metrics["accuracy"])


def choose_policy(
    ids: set[str],
    routing_rows: dict[str, dict[str, Any]],
    primary_rows: dict[str, dict[str, Any]],
    secondary_rows: dict[str, dict[str, Any]],
    labels: list[str],
    candidate_sets: list[set[str]],
    route_decision_field: str,
    accept_value: str,
    accuracy_weight: float,
) -> tuple[set[str], dict[str, Any]]:
    best_set: set[str] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score = -math.inf
    for label_set in candidate_sets:
        metrics = compute_metrics(
            build_records(
                ids,
                routing_rows,
                primary_rows,
                secondary_rows,
                label_set,
                route_decision_field,
                accept_value,
            ),
            labels,
        )
        score = objective(metrics, accuracy_weight)
        tie_break = (metrics["accuracy"], -len(label_set), sorted(label_set))
        best_tie = (
            best_metrics["accuracy"] if best_metrics is not None else -math.inf,
            -len(best_set) if best_set is not None else -math.inf,
            sorted(best_set) if best_set is not None else [],
        )
        if score > best_score or (score == best_score and tie_break > best_tie):
            best_score = score
            best_set = set(label_set)
            best_metrics = metrics
    assert best_set is not None and best_metrics is not None
    return best_set, best_metrics


def split_ids(row_ids: set[str], holdout_fraction: float, salt: str) -> tuple[set[str], set[str]]:
    tune_ids: set[str] = set()
    holdout_ids: set[str] = set()
    for row_id in row_ids:
        if stable_bucket(row_id, salt) < holdout_fraction:
            holdout_ids.add(row_id)
        else:
            tune_ids.add(row_id)
    return tune_ids, holdout_ids


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-jsonl", required=True)
    parser.add_argument("--primary-fallback-jsonl", required=True)
    parser.add_argument("--secondary-fallback-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--route-decision-field", default="route_decision")
    parser.add_argument("--accept-value", default="accept")
    parser.add_argument("--labels", default="auto")
    parser.add_argument("--candidate-predictions", default="auto")
    parser.add_argument("--max-policy-size", type=int, default=None)
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--num-splits", type=int, default=20)
    parser.add_argument("--salt-prefix", default="dual-fallback-assignment")
    parser.add_argument("--accuracy-weight", type=float, default=0.0)
    args = parser.parse_args()

    routing_rows = load_jsonl_by_id(Path(args.routing_jsonl))
    primary_rows = load_jsonl_by_id(Path(args.primary_fallback_jsonl))
    secondary_rows = load_jsonl_by_id(Path(args.secondary_fallback_jsonl))
    full_ids = set(routing_rows)
    base_records = build_records(
        None,
        routing_rows,
        primary_rows,
        secondary_rows,
        set(),
        args.route_decision_field,
        args.accept_value,
    )
    labels = infer_labels(base_records) if args.labels == "auto" else [item.strip() for item in args.labels.split(",") if item.strip()]
    if args.candidate_predictions == "auto":
        candidate_predictions = sorted(
            {
                str(row.get("prediction"))
                for row in routing_rows.values()
                if row.get(args.route_decision_field) != args.accept_value and isinstance(row.get("prediction"), str)
            }
        )
    else:
        candidate_predictions = [item.strip() for item in args.candidate_predictions.split(",") if item.strip()]
    candidate_sets = all_label_sets(candidate_predictions, args.max_policy_size)

    split_reports = []
    for split_idx in range(args.num_splits):
        tune_ids, holdout_ids = split_ids(full_ids, args.holdout_fraction, f"{args.salt_prefix}-{split_idx}")
        policy, tune_metrics = choose_policy(
            tune_ids,
            routing_rows,
            primary_rows,
            secondary_rows,
            labels,
            candidate_sets,
            args.route_decision_field,
            args.accept_value,
            args.accuracy_weight,
        )
        holdout_metrics = compute_metrics(
            build_records(
                holdout_ids,
                routing_rows,
                primary_rows,
                secondary_rows,
                policy,
                args.route_decision_field,
                args.accept_value,
            ),
            labels,
        )
        split_reports.append(
            {
                "split": split_idx,
                "policy": sorted(policy),
                "tune_rows": len(tune_ids),
                "holdout_rows": len(holdout_ids),
                "tune_metrics": tune_metrics,
                "holdout_metrics": holdout_metrics,
            }
        )

    base_metrics = compute_metrics(base_records, labels)
    secondary_all_metrics = compute_metrics(
        build_records(
            full_ids,
            routing_rows,
            primary_rows,
            secondary_rows,
            set(candidate_predictions),
            args.route_decision_field,
            args.accept_value,
        ),
        labels,
    )
    tuned_full_policy, tuned_full_metrics = choose_policy(
        full_ids,
        routing_rows,
        primary_rows,
        secondary_rows,
        labels,
        candidate_sets,
        args.route_decision_field,
        args.accept_value,
        args.accuracy_weight,
    )
    tuned_full_records = build_records(
        None,
        routing_rows,
        primary_rows,
        secondary_rows,
        tuned_full_policy,
        args.route_decision_field,
        args.accept_value,
    )
    if args.output_jsonl:
        write_jsonl(Path(args.output_jsonl), tuned_full_records)
    holdout_macro = [row["holdout_metrics"]["macro_f1"] for row in split_reports]
    holdout_acc = [row["holdout_metrics"]["accuracy"] for row in split_reports]
    policy_counts = Counter(tuple(row["policy"]) for row in split_reports)
    output = {
        "routing_jsonl": args.routing_jsonl,
        "primary_fallback_jsonl": args.primary_fallback_jsonl,
        "secondary_fallback_jsonl": args.secondary_fallback_jsonl,
        "route_decision_field": args.route_decision_field,
        "accept_value": args.accept_value,
        "labels": labels,
        "candidate_predictions": candidate_predictions,
        "holdout_fraction": args.holdout_fraction,
        "num_splits": args.num_splits,
        "accuracy_weight": args.accuracy_weight,
        "base_primary_metrics": base_metrics,
        "secondary_all_metrics": secondary_all_metrics,
        "tuned_full_policy": sorted(tuned_full_policy),
        "tuned_full_metrics": tuned_full_metrics,
        "holdout_macro_f1_mean": sum(holdout_macro) / len(holdout_macro) if holdout_macro else 0.0,
        "holdout_accuracy_mean": sum(holdout_acc) / len(holdout_acc) if holdout_acc else 0.0,
        "holdout_macro_f1_min": min(holdout_macro) if holdout_macro else 0.0,
        "holdout_macro_f1_max": max(holdout_macro) if holdout_macro else 0.0,
        "policy_counts": {"|".join(policy): count for policy, count in policy_counts.items()},
        "splits": split_reports,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        f"wrote {output_path}\n"
        f"base_primary_macro_f1={base_metrics['macro_f1']:.4f} base_primary_acc={base_metrics['accuracy']:.4f}\n"
        f"secondary_all_macro_f1={secondary_all_metrics['macro_f1']:.4f} "
        f"secondary_all_acc={secondary_all_metrics['accuracy']:.4f}\n"
        f"tuned_full_policy={sorted(tuned_full_policy)} tuned_macro_f1={tuned_full_metrics['macro_f1']:.4f} "
        f"tuned_acc={tuned_full_metrics['accuracy']:.4f}\n"
        f"holdout_macro_f1_mean={output['holdout_macro_f1_mean']:.4f} "
        f"holdout_accuracy_mean={output['holdout_accuracy_mean']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
