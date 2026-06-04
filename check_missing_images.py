#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def iter_kabr_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            return
        for line_num, raw_line in enumerate(f, start=2):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                yield {
                    "line_num": line_num,
                    "parse_error": f"expected 5 columns, got {len(parts)}",
                    "raw": line,
                }
                continue
            original_video_id, video_id, frame_id, rel_path, label_id = parts
            yield {
                "line_num": line_num,
                "original_video_id": original_video_id,
                "video_id": video_id,
                "frame_id": int(frame_id),
                "rel_path": rel_path,
                "label_id": int(label_id),
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/scratch/general/nfs1/u1320844/KABR_files2",
        help="KABR root directory",
    )
    parser.add_argument(
        "--output-dir",
        default="kabr_file_check_fast",
        help="Directory for reports",
    )
    args = parser.parse_args()

    root = Path(args.root)
    image_root = root / "dataset" / "image"
    annotation_root = root / "annotation"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for split in ["train", "val"]:
        csv_path = annotation_root / f"{split}.csv"
        bad_rows = []
        counts = {
            "total_rows": 0,
            "ok": 0,
            "missing": 0,
            "zero_bytes": 0,
            "parse_error": 0,
        }

        for row in iter_kabr_rows(csv_path):
            counts["total_rows"] += 1

            if "parse_error" in row:
                counts["parse_error"] += 1
                bad_rows.append(row)
                continue

            abs_path = image_root / row["rel_path"]

            if not abs_path.exists():
                counts["missing"] += 1
                status = "missing"
                size = None
            else:
                size = abs_path.stat().st_size
                if size == 0:
                    counts["zero_bytes"] += 1
                    status = "zero_bytes"
                else:
                    counts["ok"] += 1
                    status = "ok"

            if status != "ok":
                bad_rows.append(
                    {
                        "line_num": row["line_num"],
                        "original_video_id": row["original_video_id"],
                        "video_id": row["video_id"],
                        "frame_id": row["frame_id"],
                        "label_id": row["label_id"],
                        "rel_path": row["rel_path"],
                        "abs_path": str(abs_path),
                        "status": status,
                        "size": size,
                    }
                )

        report_path = output_dir / f"{split}_bad_files.jsonl"
        with report_path.open("w", encoding="utf-8") as f:
            for row in bad_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

        summary[split] = {
            **counts,
            "report": str(report_path.resolve()),
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
