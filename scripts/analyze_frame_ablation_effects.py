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


def prediction(row: dict[str, Any]) -> str | None:
    value = row.get("prediction") or row.get("pred_label")
    return str(value) if value is not None else None


def source_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("ablation_source_id"), str):
        return metadata["ablation_source_id"]
    return str(row.get("id", "")).split("|ablation=", 1)[0]


def ablation_mode(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("ablation_mode"), str):
        return metadata["ablation_mode"]
    row_id = str(row.get("id", ""))
    if "|ablation=original" in row_id:
        return "original"
    if "|ablation=drop" in row_id:
        return "leave_one_out"
    return "unknown"


def dropped_frame_index(row: dict[str, Any]) -> int | None:
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("dropped_frame_index"), int):
        return int(metadata["dropped_frame_index"])
    row_id = str(row.get("id", ""))
    if "|ablation=drop" not in row_id:
        return None
    try:
        return int(row_id.rsplit("drop", 1)[1])
    except ValueError:
        return None


def confidence_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get("confidence", {}).get(metric)
    return float(value) if isinstance(value, int | float) else None


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--metric", action="append", default=["mean_token_entropy", "first_token_prob", "mean_token_prob"])
    args = parser.parse_args()

    rows = load_jsonl(Path(args.jsonl))
    originals = {source_id(row): row for row in rows if ablation_mode(row) == "original"}
    variants = [row for row in rows if ablation_mode(row) == "leave_one_out"]
    if not originals:
        raise SystemExit("no original rows found")

    by_position = defaultdict(lambda: {"n": 0, "prediction_changed": 0, "target_only_correct": 0, "reference_only_correct": 0, "metric_deltas": {metric: [] for metric in args.metric}})
    by_source = defaultdict(lambda: {"n": 0, "prediction_changed": 0, "reference_prediction": None, "gold_label": None})
    examples = {"prediction_changed": [], "target_only_correct": [], "reference_only_correct": []}

    for row in variants:
        src = source_id(row)
        ref = originals.get(src)
        if ref is None:
            continue
        idx = dropped_frame_index(row)
        if idx is None:
            continue
        gold = ref.get("gold_label")
        ref_pred = prediction(ref)
        target_pred = prediction(row)
        ref_correct = ref_pred == gold
        target_correct = target_pred == gold
        changed = ref_pred != target_pred
        pos_stats = by_position[idx]
        pos_stats["n"] += 1
        pos_stats["prediction_changed"] += int(changed)
        pos_stats["target_only_correct"] += int(target_correct and not ref_correct)
        pos_stats["reference_only_correct"] += int(ref_correct and not target_correct)
        for metric in args.metric:
            ref_value = confidence_value(ref, metric)
            target_value = confidence_value(row, metric)
            if ref_value is not None and target_value is not None:
                pos_stats["metric_deltas"][metric].append(target_value - ref_value)
        src_stats = by_source[src]
        src_stats["n"] += 1
        src_stats["prediction_changed"] += int(changed)
        src_stats["reference_prediction"] = ref_pred
        src_stats["gold_label"] = gold
        if changed and len(examples["prediction_changed"]) < 30:
            examples["prediction_changed"].append({"id": row.get("id"), "gold": gold, "original": ref_pred, "ablated": target_pred, "dropped_frame_index": idx})
        if target_correct and not ref_correct and len(examples["target_only_correct"]) < 30:
            examples["target_only_correct"].append({"id": row.get("id"), "gold": gold, "original": ref_pred, "ablated": target_pred, "dropped_frame_index": idx})
        if ref_correct and not target_correct and len(examples["reference_only_correct"]) < 30:
            examples["reference_only_correct"].append({"id": row.get("id"), "gold": gold, "original": ref_pred, "ablated": target_pred, "dropped_frame_index": idx})

    output = {
        "jsonl": args.jsonl,
        "n_originals": len(originals),
        "n_variants": len(variants),
        "original_accuracy": sum(1 for row in originals.values() if prediction(row) == row.get("gold_label")) / len(originals),
        "variant_accuracy": sum(1 for row in variants if prediction(row) == row.get("gold_label")) / len(variants) if variants else 0.0,
        "by_dropped_frame_index": {
            str(idx): {
                **{key: value for key, value in stats.items() if key != "metric_deltas"},
                "prediction_change_rate": stats["prediction_changed"] / stats["n"] if stats["n"] else 0.0,
                "net_ablation_correct_gain": stats["target_only_correct"] - stats["reference_only_correct"],
                "metric_deltas": {metric: summarize(values) for metric, values in stats["metric_deltas"].items()},
            }
            for idx, stats in sorted(by_position.items())
        },
        "most_sensitive_sources": [
            {"source_id": src, **stats, "prediction_change_rate": stats["prediction_changed"] / stats["n"] if stats["n"] else 0.0}
            for src, stats in sorted(by_source.items(), key=lambda item: (item[1]["prediction_changed"], item[1]["n"]), reverse=True)
        ],
        "dropped_frame_change_counts": dict(Counter(str(dropped_frame_index(row)) for row in variants if source_id(row) in originals and prediction(row) != prediction(originals[source_id(row)]))),
        "examples": examples,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        f"wrote {output_path} originals={output['n_originals']} variants={output['n_variants']} "
        f"original_accuracy={output['original_accuracy']:.4f} variant_accuracy={output['variant_accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
