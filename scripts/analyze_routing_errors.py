#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
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


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _round(value: float | None, ndigits: int = 4) -> float | None:
    return None if value is None else round(value, ndigits)


def group_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [confidence_value(row, metric) for row in rows]
    values = [value for value in values if value is not None]
    n_correct = sum(1 for row in rows if row.get("correct", False))
    route_counts = Counter(row.get("route_decision", "unknown") for row in rows)
    return {
        "n": len(rows),
        "n_correct": n_correct,
        "accuracy": n_correct / len(rows) if rows else None,
        "n_errors": len(rows) - n_correct,
        "route_counts": dict(route_counts),
        "confidence_mean": _round(_mean(values)),
        "confidence_median": _round(_median(values)),
        "confidence_min": _round(min(values) if values else None),
        "confidence_max": _round(max(values) if values else None),
    }


def analyze(
    rows: list[dict[str, Any]],
    *,
    rare_labels: set[str],
    metric: str,
    top_k: int,
) -> dict[str, Any]:
    errors = [row for row in rows if not row.get("correct", False)]
    accepted = [row for row in rows if row.get("route_decision") == "accept"]
    accepted_errors = [row for row in accepted if not row.get("correct", False)]

    confusion_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in errors:
        gold = str(row.get("gold_label"))
        pred = str(row.get("prediction"))
        route = str(row.get("route_decision", "unknown"))
        confusion_groups[(gold, pred, route)].append(row)

    top_confusions = []
    for (gold, pred, route), group_rows in confusion_groups.items():
        top_confusions.append(
            {
                "gold_label": gold,
                "prediction": pred,
                "route_decision": route,
                **group_summary(group_rows, metric),
            }
        )
    top_confusions.sort(key=lambda row: (row["n_errors"], row["confidence_median"] or -999.0), reverse=True)

    rare_rows = [row for row in rows if row.get("gold_label") in rare_labels]
    rare_errors = [row for row in rare_rows if not row.get("correct", False)]
    rare_accepted_errors = [
        row
        for row in rare_errors
        if row.get("route_decision") == "accept"
    ]

    rare_by_gold = {}
    for label in sorted(rare_labels):
        label_rows = [row for row in rows if row.get("gold_label") == label]
        label_errors = [row for row in label_rows if not row.get("correct", False)]
        label_accepted_errors = [row for row in label_errors if row.get("route_decision") == "accept"]
        pred_counts = Counter(str(row.get("prediction")) for row in label_errors)
        rare_by_gold[label] = {
            **group_summary(label_rows, metric),
            "error_predictions": dict(pred_counts),
            "accepted_errors": group_summary(label_accepted_errors, metric),
        }

    accepted_common_prediction_errors = Counter(str(row.get("prediction")) for row in accepted_errors)
    accepted_rare_gold_errors = Counter(str(row.get("gold_label")) for row in rare_accepted_errors)

    return {
        "n_total": len(rows),
        "metric": metric,
        "rare_labels": sorted(rare_labels),
        "overall": group_summary(rows, metric),
        "accepted": group_summary(accepted, metric),
        "accepted_errors": group_summary(accepted_errors, metric),
        "rare_gold": group_summary(rare_rows, metric),
        "rare_gold_errors": group_summary(rare_errors, metric),
        "rare_gold_accepted_errors": group_summary(rare_accepted_errors, metric),
        "accepted_common_prediction_errors": dict(accepted_common_prediction_errors),
        "accepted_rare_gold_errors": dict(accepted_rare_gold_errors),
        "rare_by_gold": rare_by_gold,
        "top_confusions": top_confusions[:top_k],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--metric", default="mean_token_logprob")
    parser.add_argument("--rare-labels", nargs="+", default=["Auto-Groom", "Trot", "Occluded"])
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input_jsonl))
    result = analyze(
        rows,
        rare_labels=set(args.rare_labels),
        metric=args.metric,
        top_k=args.top_k,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    rare = result["rare_gold_accepted_errors"]
    accepted_errors = result["accepted_errors"]
    print(
        f"wrote {output_path}\n"
        f"accepted_errors={accepted_errors['n']} "
        f"rare_gold_accepted_errors={rare['n']} "
        f"rare_gold_accepted_error_share={(rare['n'] / accepted_errors['n']) if accepted_errors['n'] else 0.0:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
