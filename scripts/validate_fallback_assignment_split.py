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


def pred(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    value = row.get("pred_label") or row.get("prediction")
    return str(value) if value is not None else None


def stable_bucket(row_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{row_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def all_label_sets(labels: list[str]) -> list[set[str]]:
    result = []
    for size in range(len(labels) + 1):
        for combo in combinations(labels, size):
            result.append(set(combo))
    return result


def build_records(
    ids: set[str] | None,
    routing_rows: dict[str, dict[str, Any]],
    standard_rows: dict[str, dict[str, Any]],
    rare_rows: dict[str, dict[str, Any]],
    rare_qwen_predictions: set[str],
) -> list[dict[str, Any]]:
    records = []
    for row_id, routing_row in routing_rows.items():
        if ids is not None and row_id not in ids:
            continue
        gold = routing_row.get("gold_label")
        decision = routing_row.get("rare_fallback_route_decision")
        if decision == "accept":
            prediction = routing_row.get("prediction")
            source = "qwen_accept"
        else:
            use_rare = routing_row.get("prediction") in rare_qwen_predictions
            fallback_row = (rare_rows if use_rare else standard_rows).get(row_id)
            prediction = pred(fallback_row)
            source = "rare_fallback" if use_rare else "standard_fallback"
        records.append(
            {
                "id": row_id,
                "gold_label": gold,
                "prediction": prediction,
                "correct": prediction == gold,
                "source": source,
                "qwen_prediction": routing_row.get("prediction"),
                "route_decision": decision,
            }
        )
    return records


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
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
            if prediction in KABR_CLASSES:
                per_class[prediction]["fp"] += 1
        else:
            per_class[gold]["fn"] += 1

    f1_values = []
    per_class_f1 = {}
    for label in KABR_CLASSES:
        stats = per_class[label]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else 0.0
        per_class_f1[label] = {**stats, "f1": f1}
        f1_values.append(f1)

    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "source_counts": dict(Counter(row["source"] for row in records)),
        "qwen_prediction_counts": dict(Counter(row["qwen_prediction"] for row in records if row["source"] != "qwen_accept")),
        "per_class_f1": per_class_f1,
    }


def objective(metrics: dict[str, Any], accuracy_weight: float) -> float:
    return float(metrics["macro_f1"]) + accuracy_weight * float(metrics["accuracy"])


def choose_policy(
    tune_ids: set[str],
    routing_rows: dict[str, dict[str, Any]],
    standard_rows: dict[str, dict[str, Any]],
    rare_rows: dict[str, dict[str, Any]],
    candidate_sets: list[set[str]],
    accuracy_weight: float,
) -> tuple[set[str], dict[str, Any]]:
    best_set: set[str] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score = -math.inf
    for label_set in candidate_sets:
        metrics = compute_metrics(build_records(tune_ids, routing_rows, standard_rows, rare_rows, label_set))
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


def split_ids(routing_rows: dict[str, dict[str, Any]], holdout_fraction: float, salt: str) -> tuple[set[str], set[str]]:
    tune_ids: set[str] = set()
    holdout_ids: set[str] = set()
    for row_id in routing_rows:
        if stable_bucket(row_id, salt) < holdout_fraction:
            holdout_ids.add(row_id)
        else:
            tune_ids.add(row_id)
    return tune_ids, holdout_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-jsonl", required=True)
    parser.add_argument("--standard-fallback-jsonl", required=True)
    parser.add_argument("--rare-fallback-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--salt-prefix", default="fallback-split")
    parser.add_argument("--accuracy-weight", type=float, default=0.0)
    args = parser.parse_args()

    routing_rows = load_jsonl_by_id(Path(args.routing_jsonl))
    standard_rows = load_jsonl_by_id(Path(args.standard_fallback_jsonl))
    rare_rows = load_jsonl_by_id(Path(args.rare_fallback_jsonl))
    candidate_sets = all_label_sets(KABR_CLASSES)

    routed_ids = {
        row_id
        for row_id, row in routing_rows.items()
        if row.get("rare_fallback_route_decision") != "accept"
    }
    full_ids = set(routing_rows)
    split_reports = []
    for split_idx in range(args.num_splits):
        salt = f"{args.salt_prefix}-{split_idx}"
        tune_ids, holdout_ids = split_ids(routing_rows, args.holdout_fraction, salt)
        policy, tune_metrics = choose_policy(
            tune_ids,
            routing_rows,
            standard_rows,
            rare_rows,
            candidate_sets,
            args.accuracy_weight,
        )
        holdout_metrics = compute_metrics(build_records(holdout_ids, routing_rows, standard_rows, rare_rows, policy))
        split_reports.append(
            {
                "split": split_idx,
                "policy": sorted(policy),
                "tune_rows": len(tune_ids),
                "holdout_rows": len(holdout_ids),
                "tune_routed_rows": len(tune_ids & routed_ids),
                "holdout_routed_rows": len(holdout_ids & routed_ids),
                "tune_metrics": tune_metrics,
                "holdout_metrics": holdout_metrics,
            }
        )

    standard_all = compute_metrics(build_records(full_ids, routing_rows, standard_rows, rare_rows, set()))
    rare_all = compute_metrics(build_records(full_ids, routing_rows, standard_rows, rare_rows, set(KABR_CLASSES)))
    tuned_full_policy, tuned_full_metrics = choose_policy(
        full_ids,
        routing_rows,
        standard_rows,
        rare_rows,
        candidate_sets,
        args.accuracy_weight,
    )
    holdout_macro = [row["holdout_metrics"]["macro_f1"] for row in split_reports]
    holdout_acc = [row["holdout_metrics"]["accuracy"] for row in split_reports]
    policy_counts = Counter(tuple(row["policy"]) for row in split_reports)
    output = {
        "routing_jsonl": args.routing_jsonl,
        "standard_fallback_jsonl": args.standard_fallback_jsonl,
        "rare_fallback_jsonl": args.rare_fallback_jsonl,
        "holdout_fraction": args.holdout_fraction,
        "num_splits": args.num_splits,
        "accuracy_weight": args.accuracy_weight,
        "standard_fallback_all": standard_all,
        "rare_fallback_all": rare_all,
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
        f"standard_macro_f1={standard_all['macro_f1']:.4f} standard_acc={standard_all['accuracy']:.4f}\n"
        f"rare_macro_f1={rare_all['macro_f1']:.4f} rare_acc={rare_all['accuracy']:.4f}\n"
        f"tuned_full_policy={sorted(tuned_full_policy)} tuned_macro_f1={tuned_full_metrics['macro_f1']:.4f} "
        f"tuned_acc={tuned_full_metrics['accuracy']:.4f}\n"
        f"holdout_macro_f1_mean={output['holdout_macro_f1_mean']:.4f} "
        f"holdout_accuracy_mean={output['holdout_accuracy_mean']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
