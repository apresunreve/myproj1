#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any


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


def parse_thresholds(raw: str) -> list[float | None]:
    thresholds: list[float | None] = [None]
    for item in raw.split(","):
        item = item.strip()
        if item:
            thresholds.append(float(item))
    return thresholds


def metric_routes(value: float | None, threshold: float | None, direction: str) -> bool:
    if value is None or threshold is None:
        return False
    if direction == "higher":
        return value > threshold
    if direction == "lower":
        return value < threshold
    raise ValueError(f"Unsupported direction: {direction}")


def route_reason(
    row: dict[str, Any],
    *,
    thresholds: dict[str, float | None],
    metric: str,
    direction: str,
    base_route_field: str,
    base_route_value: str,
) -> str:
    if row.get(base_route_field) == base_route_value:
        return "route_low_confidence"
    prediction = str(row.get("prediction"))
    threshold = thresholds.get(prediction)
    value = confidence_value(row, metric)
    if metric_routes(value, threshold, direction):
        return f"route_prediction_metric_{prediction}"
    return "accept"


def summarize(
    rows: list[dict[str, Any]],
    *,
    thresholds: dict[str, float | None],
    metric: str,
    direction: str,
    rare_labels: set[str],
    base_route_field: str,
    base_route_value: str,
) -> dict[str, Any]:
    reasons = [
        route_reason(
            row,
            thresholds=thresholds,
            metric=metric,
            direction=direction,
            base_route_field=base_route_field,
            base_route_value=base_route_value,
        )
        for row in rows
    ]
    accepted = [row for row, reason in zip(rows, reasons, strict=True) if reason == "accept"]
    routed = [row for row, reason in zip(rows, reasons, strict=True) if reason != "accept"]
    base_routed = [row for row, reason in zip(rows, reasons, strict=True) if reason == "route_low_confidence"]
    extra_routed = [row for row, reason in zip(rows, reasons, strict=True) if reason.startswith("route_prediction_metric_")]

    total_errors = sum(1 for row in rows if not row.get("correct", False))
    routed_errors = sum(1 for row in routed if not row.get("correct", False))
    accepted_correct = sum(1 for row in accepted if row.get("correct", False))
    rare_gold_errors = [row for row in rows if row.get("gold_label") in rare_labels and not row.get("correct", False)]
    rare_gold_routed_errors = [
        row
        for row, reason in zip(rows, reasons, strict=True)
        if row.get("gold_label") in rare_labels and not row.get("correct", False) and reason != "accept"
    ]

    return {
        "thresholds": thresholds,
        "n_total": len(rows),
        "n_accepted": len(accepted),
        "n_routed": len(routed),
        "coverage": len(accepted) / len(rows) if rows else 0.0,
        "routing_load": len(routed) / len(rows) if rows else 0.0,
        "accepted_accuracy": accepted_correct / len(accepted) if accepted else None,
        "errors_routed": routed_errors,
        "error_capture_rate": routed_errors / total_errors if total_errors else None,
        "rare_gold_errors_routed": len(rare_gold_routed_errors),
        "rare_gold_error_capture_rate": (
            len(rare_gold_routed_errors) / len(rare_gold_errors)
            if rare_gold_errors
            else None
        ),
        "base_routed": len(base_routed),
        "extra_routed": len(extra_routed),
        "extra_routed_errors": sum(1 for row in extra_routed if not row.get("correct", False)),
    }


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                other["routing_load"] <= row["routing_load"]
                and (other["accepted_accuracy"] or 0.0) >= (row["accepted_accuracy"] or 0.0)
                and (other["rare_gold_error_capture_rate"] or 0.0) >= (row["rare_gold_error_capture_rate"] or 0.0)
                and (
                    other["routing_load"] < row["routing_load"]
                    or (other["accepted_accuracy"] or 0.0) > (row["accepted_accuracy"] or 0.0)
                    or (other["rare_gold_error_capture_rate"] or 0.0) > (row["rare_gold_error_capture_rate"] or 0.0)
                )
            ):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda row: (row["routing_load"], -(row["accepted_accuracy"] or 0.0)))


def choose_operating_points(rows: list[dict[str, Any]], max_routes: list[float], min_accs: list[float]) -> dict[str, Any | None]:
    chosen: dict[str, Any | None] = {}
    for max_route in max_routes:
        for min_acc in min_accs:
            name = f"route_le_{max_route:.2f}_acc_ge_{min_acc:.2f}".replace(".", "p")
            candidates = [
                row
                for row in rows
                if row["routing_load"] <= max_route and (row["accepted_accuracy"] or 0.0) >= min_acc
            ]
            chosen[name] = (
                max(
                    candidates,
                    key=lambda row: (
                        row["rare_gold_error_capture_rate"] or 0.0,
                        row["accepted_accuracy"] or 0.0,
                        -row["routing_load"],
                    ),
                )
                if candidates
                else None
            )

    for max_route in max_routes:
        name = f"max_rare_capture_route_le_{max_route:.2f}".replace(".", "p")
        candidates = [row for row in rows if row["routing_load"] <= max_route]
        chosen[name] = (
            max(
                candidates,
                key=lambda row: (
                    row["rare_gold_error_capture_rate"] or 0.0,
                    row["accepted_accuracy"] or 0.0,
                    -row["routing_load"],
                ),
            )
            if candidates
            else None
        )
    return chosen


def materialize(
    rows: list[dict[str, Any]],
    *,
    thresholds: dict[str, float | None],
    metric: str,
    direction: str,
    base_route_field: str,
    base_route_value: str,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        next_row = dict(row)
        decision = route_reason(
            row,
            thresholds=thresholds,
            metric=metric,
            direction=direction,
            base_route_field=base_route_field,
            base_route_value=base_route_value,
        )
        next_row["prediction_metric_route_decision"] = decision
        next_row["prediction_metric_routed"] = decision != "accept"
        output.append(next_row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--materialize-jsonl", default=None)
    parser.add_argument("--materialize-choice", default=None)
    parser.add_argument("--metric", default="mean_token_entropy")
    parser.add_argument("--direction", choices=["higher", "lower"], default="higher")
    parser.add_argument("--threshold-grid", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--rare-labels", nargs="+", required=True)
    parser.add_argument("--base-route-field", default="route_decision")
    parser.add_argument("--base-route-value", default="route")
    parser.add_argument("--max-routes", default="0.30,0.35,0.40,0.45")
    parser.add_argument("--min-accepted-accs", default="0.86,0.88,0.90")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input_jsonl))
    grid = parse_thresholds(args.threshold_grid)
    rare_labels = set(args.rare_labels)
    sweep_rows = []
    for values in itertools.product(grid, repeat=len(args.predictions)):
        thresholds = dict(zip(args.predictions, values, strict=True))
        sweep_rows.append(
            summarize(
                rows,
                thresholds=thresholds,
                metric=args.metric,
                direction=args.direction,
                rare_labels=rare_labels,
                base_route_field=args.base_route_field,
                base_route_value=args.base_route_value,
            )
        )
    sweep_rows.sort(key=lambda row: (row["routing_load"], -(row["accepted_accuracy"] or 0.0)))
    max_routes = [float(item.strip()) for item in args.max_routes.split(",") if item.strip()]
    min_accs = [float(item.strip()) for item in args.min_accepted_accs.split(",") if item.strip()]
    chosen = choose_operating_points(sweep_rows, max_routes=max_routes, min_accs=min_accs)
    result = {
        "metric": args.metric,
        "direction": args.direction,
        "predictions": args.predictions,
        "rare_labels": sorted(rare_labels),
        "threshold_grid": args.threshold_grid,
        "base_route_field": args.base_route_field,
        "base_route_value": args.base_route_value,
        "n_candidates": len(sweep_rows),
        "chosen": chosen,
        "pareto_front": pareto_front(sweep_rows),
        "sweep": sweep_rows,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    materialized = None
    if args.materialize_jsonl:
        if not args.materialize_choice:
            raise ValueError("--materialize-choice is required with --materialize-jsonl")
        choice = chosen.get(args.materialize_choice)
        if choice is None:
            raise ValueError(f"Choice {args.materialize_choice!r} was not found or had no candidate")
        materialized = materialize(
            rows,
            thresholds=choice["thresholds"],
            metric=args.metric,
            direction=args.direction,
            base_route_field=args.base_route_field,
            base_route_value=args.base_route_value,
        )
        materialize_path = Path(args.materialize_jsonl)
        materialize_path.parent.mkdir(parents=True, exist_ok=True)
        with materialize_path.open("w", encoding="utf-8") as handle:
            for row in materialized:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"wrote {output_path}", flush=True)
    if materialized is not None:
        print(f"wrote materialized routing: {args.materialize_jsonl}", flush=True)


if __name__ == "__main__":
    main()
