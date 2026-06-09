#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any


RARE_LABELS = {"Auto-Groom", "Trot", "Occluded"}
DEFAULT_PREDICTIONS = ["Graze", "Walk", "Head Up"]


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


def route_reason(row: dict[str, Any], thresholds: dict[str, float | None], metric: str) -> str:
    if row.get("route_decision") == "route":
        return "low_confidence"
    prediction = str(row.get("prediction"))
    threshold = thresholds.get(prediction)
    if threshold is None:
        return "accept"
    value = confidence_value(row, metric)
    if value is not None and value < threshold:
        return f"rare_ambiguity_{prediction}"
    return "accept"


def summarize(rows: list[dict[str, Any]], thresholds: dict[str, float | None], metric: str) -> dict[str, Any]:
    total = len(rows)
    reasons = [route_reason(row, thresholds, metric) for row in rows]
    accepted = [row for row, reason in zip(rows, reasons, strict=True) if reason == "accept"]
    routed = [row for row, reason in zip(rows, reasons, strict=True) if reason != "accept"]
    low_conf_routed = [row for row, reason in zip(rows, reasons, strict=True) if reason == "low_confidence"]
    rare_routed = [row for row, reason in zip(rows, reasons, strict=True) if reason.startswith("rare_ambiguity_")]
    total_errors = sum(1 for row in rows if not row.get("correct", False))
    routed_errors = sum(1 for row in routed if not row.get("correct", False))
    accepted_correct = sum(1 for row in accepted if row.get("correct", False))
    rare_gold_errors = [row for row in rows if row.get("gold_label") in RARE_LABELS and not row.get("correct", False)]
    rare_gold_routed_errors = [
        row
        for row, reason in zip(rows, reasons, strict=True)
        if row.get("gold_label") in RARE_LABELS and not row.get("correct", False) and reason != "accept"
    ]
    return {
        "thresholds": thresholds,
        "n_total": total,
        "n_accepted": len(accepted),
        "n_routed": len(routed),
        "coverage": len(accepted) / total if total else 0.0,
        "routing_load": len(routed) / total if total else 0.0,
        "accepted_accuracy": accepted_correct / len(accepted) if accepted else None,
        "errors_routed": routed_errors,
        "error_capture_rate": routed_errors / total_errors if total_errors else None,
        "rare_gold_errors_routed": len(rare_gold_routed_errors),
        "rare_gold_error_capture_rate": (
            len(rare_gold_routed_errors) / len(rare_gold_errors)
            if rare_gold_errors
            else None
        ),
        "low_confidence_routed": len(low_conf_routed),
        "rare_ambiguity_routed": len(rare_routed),
        "rare_ambiguity_errors_routed": sum(1 for row in rare_routed if not row.get("correct", False)),
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


def choose_operating_points(rows: list[dict[str, Any]]) -> dict[str, Any | None]:
    targets = {
        "route_le_15_acc_ge_92": lambda row: row["routing_load"] <= 0.15 and (row["accepted_accuracy"] or 0.0) >= 0.92,
        "route_le_20_acc_ge_93": lambda row: row["routing_load"] <= 0.20 and (row["accepted_accuracy"] or 0.0) >= 0.93,
        "route_le_25_acc_ge_94": lambda row: row["routing_load"] <= 0.25 and (row["accepted_accuracy"] or 0.0) >= 0.94,
        "max_rare_capture_under_20_route": lambda row: row["routing_load"] <= 0.20,
    }
    chosen: dict[str, Any | None] = {}
    for name, predicate in targets.items():
        candidates = [row for row in rows if predicate(row)]
        if not candidates:
            chosen[name] = None
            continue
        if name.startswith("max_rare"):
            chosen[name] = max(
                candidates,
                key=lambda row: (
                    row["rare_gold_error_capture_rate"] or 0.0,
                    row["accepted_accuracy"] or 0.0,
                    -row["routing_load"],
                ),
            )
        else:
            chosen[name] = max(
                candidates,
                key=lambda row: (
                    row["accepted_accuracy"] or 0.0,
                    row["rare_gold_error_capture_rate"] or 0.0,
                    -row["routing_load"],
                ),
            )
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--metric", default="mean_token_logprob")
    parser.add_argument("--threshold-grid", default="-0.24,-0.20,-0.16,-0.12,-0.08,-0.04")
    parser.add_argument("--predictions", nargs="+", default=DEFAULT_PREDICTIONS)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input_jsonl))
    grid = parse_thresholds(args.threshold_grid)
    sweep_rows = []
    for values in itertools.product(grid, repeat=len(args.predictions)):
        thresholds = dict(zip(args.predictions, values, strict=True))
        sweep_rows.append(summarize(rows, thresholds, args.metric))
    sweep_rows.sort(key=lambda row: (row["routing_load"], -(row["accepted_accuracy"] or 0.0)))
    result = {
        "metric": args.metric,
        "predictions": args.predictions,
        "threshold_grid": args.threshold_grid,
        "n_candidates": len(sweep_rows),
        "chosen": choose_operating_points(sweep_rows),
        "pareto_front": pareto_front(sweep_rows),
        "sweep": sweep_rows,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    chosen = result["chosen"]
    best = chosen.get("route_le_20_acc_ge_93") or chosen.get("route_le_25_acc_ge_94") or chosen.get("route_le_15_acc_ge_92")
    if best is None:
        print(f"wrote {output_path}; no preferred operating point found", flush=True)
    else:
        print(
            f"wrote {output_path}\n"
            f"candidate routing_load={best['routing_load']:.4f} "
            f"accepted_accuracy={best['accepted_accuracy']:.4f} "
            f"rare_gold_error_capture_rate={best['rare_gold_error_capture_rate']:.4f} "
            f"thresholds={best['thresholds']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
