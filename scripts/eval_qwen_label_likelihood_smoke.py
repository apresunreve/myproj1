#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from kabr_vlm.efficiency.label_likelihood import QwenLabelLikelihoodScorer  # noqa: E402
from kabr_vlm.vlm_baselines import BEHAVIOR_LABELS  # noqa: E402


def load_balanced_rows(path: Path, labels: list[str], samples_per_class: int) -> list[dict]:
    buckets: dict[str, list[dict]] = {label: [] for label in labels}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("answer") or row.get("metadata", {}).get("behavior")
            if label in buckets and len(buckets[label]) < samples_per_class:
                buckets[label].append(row)
            if all(len(rows) >= samples_per_class for rows in buckets.values()):
                break

    selected: list[dict] = []
    for label in labels:
        selected.extend(buckets[label])
    return selected


def _safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _safe_p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1))
    return ordered[idx]


def compute_metrics(records: list[dict], labels: list[str]) -> dict:
    n_total = len(records)
    n_correct = sum(1 for row in records if row["prediction"] == row["gold_label"])
    pred_counts = Counter(row["prediction"] for row in records)
    gold_counts = Counter(row["gold_label"] for row in records)
    tp_counts = Counter(row["gold_label"] for row in records if row["prediction"] == row["gold_label"])

    per_class = {}
    f1_values = []
    for label in labels:
        tp = tp_counts[label]
        fp = pred_counts[label] - tp
        fn = gold_counts[label] - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": gold_counts[label],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if gold_counts[label]:
            f1_values.append(f1)

    latencies = [float(row["latency_ms"]) for row in records if row.get("latency_ms") is not None]
    margins = [float(row["confidence"]["margin"]) for row in records]
    entropies = [float(row["confidence"]["entropy"]) for row in records]

    return {
        "inference_mode": "label_likelihood",
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
        "gold_distribution": dict(gold_counts),
        "pred_distribution": dict(pred_counts),
        "latency_median_ms": None if _safe_median(latencies) is None else round(_safe_median(latencies), 2),
        "latency_p90_ms": None if _safe_p90(latencies) is None else round(_safe_p90(latencies), 2),
        "latency_mean_ms": None if _safe_mean(latencies) is None else round(_safe_mean(latencies), 2),
        "margin_median": None if _safe_median(margins) is None else round(_safe_median(margins), 4),
        "margin_mean": None if _safe_mean(margins) is None else round(_safe_mean(margins), 4),
        "entropy_median": None if _safe_median(entropies) is None else round(_safe_median(entropies), 4),
        "entropy_mean": None if _safe_mean(entropies) is None else round(_safe_mean(entropies), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-jsonl", default="data/processed/kabr_behavior_val_cleaned.jsonl")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter-path", default="artifacts/runs/qwen25vl7b_behavior_lora_full/checkpoint")
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    rows = load_balanced_rows(Path(args.val_jsonl), BEHAVIOR_LABELS, args.samples_per_class)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scorer = QwenLabelLikelihoodScorer.from_adapter(args.model_id, args.adapter_path)
    records = []
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows, start=1):
            gold = row.get("answer") or row.get("metadata", {}).get("behavior")
            t0 = time.perf_counter()
            result = scorer.predict(row["image_paths"], row["prompt"], BEHAVIOR_LABELS)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            record = {
                "id": row["id"],
                "inference_mode": "label_likelihood",
                "gold_label": gold,
                "prediction": result.prediction,
                "correct": result.prediction == gold,
                "label_scores": result.label_scores,
                "confidence": {
                    "top1": result.confidence.top1,
                    "top2": result.confidence.top2,
                    "margin": result.confidence.margin,
                    "entropy": result.confidence.entropy,
                    "probabilities": result.confidence.probabilities,
                },
                "latency_ms": round(latency_ms, 2),
                "num_input_frames": len(row["image_paths"]),
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            records.append(record)
            if idx % 10 == 0 or idx == len(rows):
                acc = sum(1 for item in records if item["correct"]) / len(records)
                print(f"[{idx}/{len(rows)}] accuracy_so_far={acc:.4f}", flush=True)

    metrics = compute_metrics(records, BEHAVIOR_LABELS)
    metrics_path = Path(str(output_path) + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        f"wrote predictions: {output_path}\nwrote metrics: {metrics_path}\n"
        f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"latency_median_ms={metrics['latency_median_ms']} "
        f"margin_median={metrics['margin_median']} entropy_median={metrics['entropy_median']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
