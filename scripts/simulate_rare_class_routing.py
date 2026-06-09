#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


RARE_LABELS = {"Auto-Groom", "Trot", "Occluded"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def confidence_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get("confidence", {}).get(metric)
    if value is None:
        return None
    return float(value)


def base_low_confidence_route(row: dict[str, Any]) -> bool:
    return row.get("route_decision") == "route"


def rare_ambiguity_route_handpicked(row: dict[str, Any]) -> bool:
    prediction = row.get("prediction")
    confidence = row.get("confidence", {})
    mean_logprob = confidence_value(row, "mean_token_logprob")
    first_prob = confidence_value(row, "first_token_prob")
    min_margin = confidence_value(row, "min_token_logit_margin")
    mean_entropy = confidence_value(row, "mean_token_entropy")

    if prediction == "Graze":
        if mean_logprob is not None and mean_logprob < -0.08:
            return True
        if mean_entropy is not None and mean_entropy > 0.24:
            return True
    if prediction == "Walk":
        if mean_logprob is not None and mean_logprob < -0.11:
            return True
        if min_margin is not None and min_margin < 3.0:
            return True
    if prediction == "Head Up":
        if mean_logprob is not None and mean_logprob < -0.11:
            return True
        if first_prob is not None and first_prob < 0.90:
            return True
    return False


def rare_ambiguity_route_prediction_metric(
    row: dict[str, Any],
    *,
    metric: str,
    thresholds: dict[str, float | None],
) -> bool:
    threshold = thresholds.get(str(row.get("prediction")))
    if threshold is None:
        return False
    value = confidence_value(row, metric)
    return value is not None and value < threshold


def analyze(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    metric: str,
    thresholds: dict[str, float | None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    routed_rows = []
    for row in rows:
        low_conf = base_low_confidence_route(row)
        if mode == "handpicked":
            rare_ambiguity = rare_ambiguity_route_handpicked(row)
        elif mode == "prediction_metric":
            rare_ambiguity = rare_ambiguity_route_prediction_metric(row, metric=metric, thresholds=thresholds)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        next_row = dict(row)
        if low_conf:
            decision = "route_low_confidence"
        elif rare_ambiguity:
            decision = "route_rare_ambiguity"
        else:
            decision = "accept"
        next_row["rare_fallback_route_decision"] = decision
        next_row["rare_fallback_routed"] = decision != "accept"
        routed_rows.append(next_row)

    accepted = [row for row in routed_rows if not row["rare_fallback_routed"]]
    routed = [row for row in routed_rows if row["rare_fallback_routed"]]
    low_conf_routed = [row for row in routed_rows if row["rare_fallback_route_decision"] == "route_low_confidence"]
    rare_routed = [row for row in routed_rows if row["rare_fallback_route_decision"] == "route_rare_ambiguity"]
    total_errors = sum(1 for row in routed_rows if not row.get("correct", False))
    routed_errors = sum(1 for row in routed if not row.get("correct", False))
    rare_gold_accepted_errors = [
        row
        for row in accepted
        if row.get("gold_label") in RARE_LABELS and not row.get("correct", False)
    ]
    rare_gold_routed_errors = [
        row
        for row in routed
        if row.get("gold_label") in RARE_LABELS and not row.get("correct", False)
    ]
    route_counts = Counter(row["rare_fallback_route_decision"] for row in routed_rows)
    metrics = {
        "mode": mode,
        "metric": metric,
        "prediction_thresholds": thresholds,
        "n_total": len(routed_rows),
        "base_accuracy": sum(1 for row in routed_rows if row.get("correct", False)) / len(routed_rows),
        "n_accepted": len(accepted),
        "n_routed": len(routed),
        "coverage": len(accepted) / len(routed_rows),
        "routing_load": len(routed) / len(routed_rows),
        "accepted_accuracy": (
            sum(1 for row in accepted if row.get("correct", False)) / len(accepted)
            if accepted
            else None
        ),
        "routed_error_rate": routed_errors / len(routed) if routed else None,
        "errors_routed": routed_errors,
        "total_errors": total_errors,
        "error_capture_rate": routed_errors / total_errors if total_errors else None,
        "route_counts": dict(route_counts),
        "low_confidence_route_errors": sum(1 for row in low_conf_routed if not row.get("correct", False)),
        "rare_ambiguity_route_errors": sum(1 for row in rare_routed if not row.get("correct", False)),
        "rare_gold_accepted_errors": len(rare_gold_accepted_errors),
        "rare_gold_routed_errors": len(rare_gold_routed_errors),
        "rare_gold_error_capture_rate": (
            len(rare_gold_routed_errors) / (len(rare_gold_routed_errors) + len(rare_gold_accepted_errors))
            if rare_gold_routed_errors or rare_gold_accepted_errors
            else None
        ),
        "accepted_error_predictions": dict(Counter(str(row.get("prediction")) for row in accepted if not row.get("correct", False))),
        "routed_error_predictions": dict(Counter(str(row.get("prediction")) for row in routed if not row.get("correct", False))),
    }
    return routed_rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-metrics-json", required=True)
    parser.add_argument("--mode", choices=["handpicked", "prediction_metric"], default="handpicked")
    parser.add_argument("--metric", default="mean_token_logprob")
    parser.add_argument("--graze-threshold", type=float, default=None)
    parser.add_argument("--walk-threshold", type=float, default=None)
    parser.add_argument("--head-up-threshold", type=float, default=None)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input_jsonl))
    thresholds = {
        "Graze": args.graze_threshold,
        "Walk": args.walk_threshold,
        "Head Up": args.head_up_threshold,
    }
    routed_rows, metrics = analyze(rows, mode=args.mode, metric=args.metric, thresholds=thresholds)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in routed_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    output_metrics = Path(args.output_metrics_json)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(
        f"wrote routed predictions: {output_jsonl}\n"
        f"wrote metrics: {output_metrics}\n"
        f"accepted_accuracy={metrics['accepted_accuracy']:.4f} "
        f"coverage={metrics['coverage']:.4f} routing_load={metrics['routing_load']:.4f} "
        f"error_capture_rate={metrics['error_capture_rate']:.4f} "
        f"rare_gold_error_capture_rate={metrics['rare_gold_error_capture_rate']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
