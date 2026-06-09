#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from kabr_vlm.efficiency.generation_logits import QwenGenerationLogitScorer  # noqa: E402
from kabr_vlm.vlm_baselines import BEHAVIOR_LABELS, extract_label, match_candidate, normalize_text  # noqa: E402


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


def load_rows(path: Path, max_samples: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def infer_labels(rows: list[dict]) -> list[str]:
    labels = []
    seen = set()
    for row in rows:
        label = row.get("answer") or row.get("metadata", {}).get("behavior")
        if isinstance(label, str) and label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def parse_labels_arg(value: str | None, rows: list[dict]) -> list[str]:
    if value is None or value.strip().lower() == "auto":
        return infer_labels(rows)
    labels = [item.strip() for item in value.split(",") if item.strip()]
    if not labels:
        raise ValueError("--labels must be 'auto' or a comma-separated non-empty label list")
    return labels


def extract_kabr_label(text: str) -> str | None:
    matched = match_candidate(text, BEHAVIOR_LABELS)
    if matched is not None:
        return matched
    normalized = normalize_text(text)
    if "walking" in normalized or "walk" in normalized:
        return "Walk"
    if "running" in normalized:
        return "Run"
    if "trot" in normalized:
        return "Trot"
    if "graz" in normalized:
        return "Graze"
    if "brows" in normalized or "forag" in normalized or "feed" in normalized or "eat" in normalized:
        return "Graze"
    if "groom" in normalized:
        return "Auto-Groom"
    if "head" in normalized and "up" in normalized:
        return "Head Up"
    if "occlu" in normalized or "hidden" in normalized or "obscur" in normalized:
        return "Occluded"
    return None


def extract_prediction(text: str, labels: list[str]) -> str | None:
    if labels == BEHAVIOR_LABELS:
        return extract_kabr_label(text)
    return extract_label(text, labels)


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


def _rounded(value: float | None, ndigits: int = 4) -> float | None:
    return None if value is None else round(value, ndigits)


def _confidence_metric(records: list[dict], key: str) -> dict:
    values = [float(row["confidence"][key]) for row in records if row["confidence"].get(key) is not None]
    correct = [
        float(row["confidence"][key])
        for row in records
        if row["correct"] and row["confidence"].get(key) is not None
    ]
    incorrect = [
        float(row["confidence"][key])
        for row in records
        if not row["correct"] and row["confidence"].get(key) is not None
    ]
    return {
        "mean": _rounded(_safe_mean(values)),
        "median": _rounded(_safe_median(values)),
        "correct_mean": _rounded(_safe_mean(correct)),
        "incorrect_mean": _rounded(_safe_mean(incorrect)),
        "correct_median": _rounded(_safe_median(correct)),
        "incorrect_median": _rounded(_safe_median(incorrect)),
    }


def compute_metrics(records: list[dict], labels: list[str]) -> dict:
    n_total = len(records)
    n_correct = sum(1 for row in records if row["correct"])
    n_parsed = sum(1 for row in records if row["prediction"] is not None)
    pred_counts = Counter(row["prediction"] or "__unparsed__" for row in records)
    gold_counts = Counter(row["gold_label"] for row in records)
    tp_counts = Counter(row["gold_label"] for row in records if row["prediction"] == row["gold_label"])
    latencies = [float(row["latency_ms"]) for row in records if row.get("latency_ms") is not None]
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

    return {
        "inference_mode": "generation_logit_confidence",
        "n_total": n_total,
        "n_correct": n_correct,
        "n_parsed": n_parsed,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "parsed_rate": n_parsed / n_total if n_total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
        "gold_distribution": dict(gold_counts),
        "pred_distribution": dict(pred_counts),
        "latency_median_ms": _rounded(_safe_median(latencies), 2),
        "latency_p90_ms": _rounded(_safe_p90(latencies), 2),
        "latency_mean_ms": _rounded(_safe_mean(latencies), 2),
        "confidence_summary": {
            "mean_token_prob": _confidence_metric(records, "mean_token_prob"),
            "min_token_prob": _confidence_metric(records, "min_token_prob"),
            "mean_token_logprob": _confidence_metric(records, "mean_token_logprob"),
            "sum_token_logprob": _confidence_metric(records, "sum_token_logprob"),
            "mean_token_logit_margin": _confidence_metric(records, "mean_token_logit_margin"),
            "min_token_logit_margin": _confidence_metric(records, "min_token_logit_margin"),
            "mean_token_entropy": _confidence_metric(records, "mean_token_entropy"),
            "first_token_prob": _confidence_metric(records, "first_token_prob"),
            "first_token_logit_margin": _confidence_metric(records, "first_token_logit_margin"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-jsonl", default="data/processed/kabr_behavior_val_cleaned.jsonl")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter-path", default="artifacts/runs/qwen25vl7b_behavior_lora_full/checkpoint")
    parser.add_argument("--labels", default="auto", help="Use 'auto' or a comma-separated closed-set label list.")
    parser.add_argument("--samples-per-class", type=int, default=5, help="Use <=0 to evaluate rows in file order.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional row limit when --samples-per-class <=0.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    candidate_rows = load_rows(Path(args.val_jsonl), max_samples=args.max_samples if args.samples_per_class <= 0 else None)
    labels = parse_labels_arg(args.labels, candidate_rows)
    if args.samples_per_class > 0:
        rows = load_balanced_rows(Path(args.val_jsonl), labels, args.samples_per_class)
    else:
        rows = candidate_rows
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    records = []
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record_id = record.get("id")
                if isinstance(record_id, str):
                    done_ids.add(record_id)
                    records.append(record)
        print(f"resume: loaded {len(done_ids)} existing rows from {output_path}", flush=True)
    pending_rows = [row for row in rows if row.get("id") not in done_ids]
    print(f"loaded={len(rows)} pending={len(pending_rows)} labels={labels}", flush=True)

    scorer = QwenGenerationLogitScorer.from_adapter(args.model_id, args.adapter_path)
    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for idx, row in enumerate(pending_rows, start=1):
            gold = row.get("answer") or row.get("metadata", {}).get("behavior")
            t0 = time.perf_counter()
            result = scorer.generate_with_confidence(
                row["image_paths"],
                row["prompt"],
                max_new_tokens=args.max_new_tokens,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            prediction = extract_prediction(result.text, labels)
            record = {
                "id": row["id"],
                "inference_mode": "generation_logit_confidence",
                "gold_label": gold,
                "pred_text": result.text,
                "prediction": prediction,
                "correct": prediction == gold,
                "token_ids": result.token_ids,
                "confidence": result.confidence,
                "latency_ms": round(latency_ms, 2),
                "num_input_frames": len(row["image_paths"]),
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            records.append(record)
            if idx % 10 == 0 or idx == len(pending_rows):
                acc = sum(1 for item in records if item["correct"]) / len(records)
                print(f"[{idx}/{len(pending_rows)}] total={len(records)} accuracy_so_far={acc:.4f}", flush=True)

    metrics = compute_metrics(records, labels)
    metrics["labels"] = labels
    metrics_path = Path(str(output_path) + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        f"wrote predictions: {output_path}\nwrote metrics: {metrics_path}\n"
        f"accuracy={metrics['accuracy']:.4f} parsed_rate={metrics['parsed_rate']:.4f} "
        f"latency_median_ms={metrics['latency_median_ms']} "
        f"first_token_prob={metrics['confidence_summary']['first_token_prob']['median']} "
        f"mean_token_prob={metrics['confidence_summary']['mean_token_prob']['median']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
