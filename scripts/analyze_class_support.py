#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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


def gold_label(row: dict[str, Any]) -> str:
    return str(row.get("answer") or row.get("metadata", {}).get("behavior") or row.get("gold_label"))


def dataset_summary(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    label_counts = Counter(gold_label(row) for row in rows)
    videos_by_label: dict[str, set[str]] = defaultdict(set)
    species_by_label: dict[str, Counter] = defaultdict(Counter)
    frame_counts_by_label: dict[str, list[int]] = defaultdict(list)
    prompt_counts = Counter(str(row.get("prompt", "")) for row in rows)
    for row in rows:
        label = gold_label(row)
        metadata = row.get("metadata", {})
        video_id = str(metadata.get("original_video_id") or row.get("video_id") or "")
        if video_id:
            videos_by_label[label].add(video_id)
        species = metadata.get("species")
        if species:
            species_by_label[label][str(species)] += 1
        num_frames = metadata.get("num_frames")
        if isinstance(num_frames, int | float):
            frame_counts_by_label[label].append(int(num_frames))
    labels = sorted(label_counts)
    per_label = {}
    for label in labels:
        frame_counts = frame_counts_by_label[label]
        per_label[label] = {
            "rows": label_counts[label],
            "share": label_counts[label] / len(rows) if rows else 0.0,
            "videos": len(videos_by_label[label]),
            "top_species": dict(species_by_label[label].most_common(5)),
            "num_frames_min": min(frame_counts) if frame_counts else None,
            "num_frames_max": max(frame_counts) if frame_counts else None,
        }
    return {
        "path": str(path),
        "n_rows": len(rows),
        "n_prompts": len(prompt_counts),
        "per_label": per_label,
    }


def prediction_summary(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    gold_counts = Counter(str(row.get("gold_label")) for row in rows)
    pred_counts = Counter(str(row.get("prediction") or row.get("pred_label")) for row in rows)
    correct_by_label = Counter(
        str(row.get("gold_label"))
        for row in rows
        if (row.get("prediction") or row.get("pred_label")) == row.get("gold_label")
    )
    per_label = {}
    for label in sorted(gold_counts):
        per_label[label] = {
            "support": gold_counts[label],
            "correct": correct_by_label[label],
            "recall": correct_by_label[label] / gold_counts[label] if gold_counts[label] else 0.0,
        }
    return {
        "path": str(path),
        "n_rows": len(rows),
        "gold_distribution": dict(gold_counts),
        "pred_distribution": dict(pred_counts),
        "per_label": per_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", nargs="+", required=True)
    parser.add_argument("--prediction-jsonl", nargs="*", default=[])
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    result = {
        "datasets": [dataset_summary(Path(path)) for path in args.dataset_jsonl],
        "predictions": [prediction_summary(Path(path)) for path in args.prediction_jsonl],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
