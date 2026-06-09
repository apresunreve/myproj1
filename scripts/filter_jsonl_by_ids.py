#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_ids(path: Path, *, decision_field: str | None, decision_values: set[str] | None) -> set[str]:
    ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if decision_field is not None and decision_values is not None:
                if str(row.get(decision_field)) not in decision_values:
                    continue
            row_id = row.get("id")
            if isinstance(row_id, str):
                ids.add(row_id)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--ids-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--decision-field", default=None)
    parser.add_argument("--decision-values", nargs="+", default=None)
    args = parser.parse_args()

    decision_values = set(args.decision_values) if args.decision_values else None
    ids = load_ids(
        Path(args.ids_jsonl),
        decision_field=args.decision_field,
        decision_values=decision_values,
    )
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with Path(args.input_jsonl).open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") in ids:
                dst.write(json.dumps(row, ensure_ascii=True) + "\n")
                written += 1
    print(f"loaded_ids={len(ids)} wrote_rows={written} output={output_path}", flush=True)


if __name__ == "__main__":
    main()
