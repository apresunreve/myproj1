#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--mode", choices=["leave_one_out"], default="leave_one_out")
    args = parser.parse_args()

    rows = load_rows(Path(args.input_jsonl), args.max_samples)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            image_paths = list(row.get("image_paths", []))
            if args.include_original:
                original = dict(row)
                original["id"] = f"{row['id']}|ablation=original"
                original.setdefault("metadata", {})
                original["metadata"] = {**original["metadata"], "ablation_source_id": row["id"], "ablation_mode": "original"}
                handle.write(json.dumps(original, ensure_ascii=True) + "\n")
                n_written += 1
            for idx in range(len(image_paths)):
                variant = dict(row)
                variant["id"] = f"{row['id']}|ablation=drop{idx}"
                variant["image_paths"] = [path for frame_idx, path in enumerate(image_paths) if frame_idx != idx]
                variant.setdefault("metadata", {})
                variant["metadata"] = {
                    **variant["metadata"],
                    "ablation_source_id": row["id"],
                    "ablation_mode": "leave_one_out",
                    "dropped_frame_index": idx,
                    "original_num_frames": len(image_paths),
                }
                handle.write(json.dumps(variant, ensure_ascii=True) + "\n")
                n_written += 1
    print(f"wrote {n_written} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
