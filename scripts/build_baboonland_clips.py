"""Build Baboonland derived clip manifests from Charades-style frame annotations.

Mirrors the KABR derived-clip construction described in KABR_DATASET_CONSTRUCTION.md:
contiguous same-label runs on (original_video_id, label_id) with consecutive frame_id.
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from kabr_vlm.data.baboonland_frames import (  # noqa: E402
    group_frame_records_into_clips,
    read_baboonland_frame_annotations,
)


def _write_jsonl(path: Path, clips) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for clip in clips:
            f.write(json.dumps(asdict(clip), separators=(",", ":")) + "\n")


def _summarize(clips, split: str) -> dict:
    per_class = Counter(c.label_name for c in clips)
    lengths = [len(c.frame_ids) for c in clips]
    return {
        "split": split,
        "num_clips": len(clips),
        "num_unique_videos": len({c.original_video_id for c in clips}),
        "frames_total": sum(lengths),
        "frames_min": min(lengths) if lengths else 0,
        "frames_max": max(lengths) if lengths else 0,
        "frames_mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "clips_per_class": dict(per_class.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        default="/scratch/general/nfs1/u1320844/baboon2/charades",
    )
    parser.add_argument(
        "--output_root",
        default="/scratch/general/vast/j.yan/myproj1/baboonland_files",
    )
    args = parser.parse_args()

    source = Path(args.source_root)
    ann_dir = source / "annotation"
    image_root = source / "dataset" / "image"
    classes_path = ann_dir / "classes.json"

    output_root = Path(args.output_root)
    manifest_dir = output_root / "annotation" / "clip_manifests"
    summary_path = output_root / "annotation" / "clip_manifests_summary.json"

    summaries = []
    for split_name in ("train", "val"):
        ann_path = ann_dir / f"{split_name}.csv"
        print(f"[{split_name}] reading {ann_path}", flush=True)
        rows = read_baboonland_frame_annotations(ann_path, image_root, classes_path)
        print(f"[{split_name}] frame rows: {len(rows)}", flush=True)

        clips = group_frame_records_into_clips(rows)
        print(f"[{split_name}] derived clips: {len(clips)}", flush=True)

        out = manifest_dir / f"{split_name}.jsonl"
        _write_jsonl(out, clips)
        print(f"[{split_name}] wrote {out}", flush=True)

        summaries.append(_summarize(clips, split_name))

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"wrote summary: {summary_path}", flush=True)
    for s in summaries:
        print(json.dumps(s, indent=2), flush=True)


if __name__ == "__main__":
    main()
