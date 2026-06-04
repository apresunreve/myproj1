"""Produce data/processed/baboonland_behavior_{train,val}.jsonl for LoRA training.

Reads the raw Charades-style baboonland annotations, groups into derived
single-label clips, samples frames, and writes VideoQAExample-shaped rows with
the action_recognition prompt and label answer. Matches the schema consumed by
src/kabr_vlm/training/qwen_tasks.py (same shape as data/processed/kabr_behavior_*.jsonl).
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from kabr_vlm.data.baboonland_frames import (  # noqa: E402
    group_frame_records_into_clips,
    read_baboonland_frame_annotations,
)
from kabr_vlm.data.kabr_frames import sample_clip_frames  # noqa: E402

BABOONLAND_CLASSES = [
    "Walking/Running",
    "Sitting/Standing",
    "Fighting/Playing",
    "Self-Grooming",
    "Being Groomed",
    "Grooming Somebody",
    "Mutual Grooming",
    "Infant-Carrying",
    "Foraging",
    "Drinking",
    "Mounting",
    "Sleeping",
    "Occluded",
]


def build_prompt() -> str:
    labels = ", ".join(BABOONLAND_CLASSES)
    return (
        "You are analyzing a clip from the baboonland dataset.\n"
        "Identify the behavior being performed.\n"
        f"Reply with exactly one behavior label from: {labels}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_root",
        default="/scratch/general/nfs1/u1320844/baboon2/charades",
    )
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "data" / "processed"),
    )
    parser.add_argument("--max_frames_per_clip", type=int, default=8)
    args = parser.parse_args()

    source = Path(args.source_root)
    ann_dir = source / "annotation"
    image_root = source / "dataset" / "image"
    classes_path = ann_dir / "classes.json"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = build_prompt()

    for split in ("train", "val"):
        ann_path = ann_dir / f"{split}.csv"
        print(f"[{split}] reading {ann_path}", flush=True)
        rows = read_baboonland_frame_annotations(ann_path, image_root, classes_path)
        print(f"[{split}] frame rows: {len(rows)}", flush=True)
        clips = group_frame_records_into_clips(rows)
        print(f"[{split}] clips: {len(clips)}", flush=True)

        out = output_dir / f"baboonland_behavior_{split}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for clip in clips:
                sampled = sample_clip_frames(clip, max_frames=args.max_frames_per_clip)
                example = {
                    "id": clip.clip_id,
                    "video_id": clip.video_id,
                    "clip_path": clip.clip_id,
                    "image_paths": sampled,
                    "split": clip.split,
                    "task": "action_recognition",
                    "prompt": prompt_text,
                    "answer": clip.label_name,
                    "metadata": {
                        "species": clip.species,
                        "behavior": clip.label_name,
                        "original_video_id": clip.original_video_id,
                        "label_id": clip.label_id,
                        "num_frames": len(clip.frame_ids),
                    },
                }
                handle.write(json.dumps(example, ensure_ascii=True) + "\n")
        print(f"[{split}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
