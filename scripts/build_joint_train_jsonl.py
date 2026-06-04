"""Build joint KABR + baboonland training JSONL by interleaving the two
processed JSONLs. Each example keeps its native prompt and label space —
the model learns both vocabularies in one adapter.

Optionally writes a per-source ID prefix for traceability.
"""

import argparse
import json
import random
from pathlib import Path


def load(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kabr_train",
        default="/scratch/general/vast/j.yan/myproj1/data/processed/kabr_behavior_train.jsonl",
    )
    parser.add_argument(
        "--baboonland_train",
        default="/scratch/general/vast/j.yan/myproj1/data/processed/baboonland_behavior_train.jsonl",
    )
    parser.add_argument(
        "--output",
        default="/scratch/general/vast/j.yan/myproj1/data/processed/joint_kabr_bbl_behavior_train.jsonl",
    )
    parser.add_argument(
        "--baboonland_val",
        default="/scratch/general/vast/j.yan/myproj1/data/processed/baboonland_behavior_val.jsonl",
        help="Used as joint val to monitor the harder dataset's loss.",
    )
    parser.add_argument(
        "--val_output",
        default="/scratch/general/vast/j.yan/myproj1/data/processed/joint_kabr_bbl_behavior_val.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    kabr = load(Path(args.kabr_train))
    bbl = load(Path(args.baboonland_train))
    print(f"loaded KABR train: {len(kabr)} examples")
    print(f"loaded baboonland train: {len(bbl)} examples")

    # Tag for traceability and so we can stratify if needed.
    for row in kabr:
        row.setdefault("metadata", {})["source_dataset"] = "kabr"
        row["id"] = f"kabr::{row['id']}"
    for row in bbl:
        row.setdefault("metadata", {})["source_dataset"] = "baboonland"
        row["id"] = f"bbl::{row['id']}"

    combined = kabr + bbl
    rng = random.Random(args.seed)
    rng.shuffle(combined)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"wrote {len(combined)} interleaved rows -> {out}")
    print(f"  kabr fraction: {len(kabr) / len(combined):.3f}")
    print(f"  bbl  fraction: {len(bbl) / len(combined):.3f}")

    # Use baboonland val as the joint-training val (its imbalance is the harder
    # signal; loss-based early-stop / monitoring is more informative there).
    val_rows = load(Path(args.baboonland_val))
    for row in val_rows:
        row.setdefault("metadata", {})["source_dataset"] = "baboonland"
        row["id"] = f"bbl::{row['id']}"
    val_out = Path(args.val_output)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    with val_out.open("w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"wrote val {len(val_rows)} rows -> {val_out}")


if __name__ == "__main__":
    main()
