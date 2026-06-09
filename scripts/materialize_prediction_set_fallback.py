#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--fallback-predictions", nargs="+", required=True)
    parser.add_argument("--base-route-field", default="route_decision")
    parser.add_argument("--accept-value", default="accept")
    parser.add_argument("--output-route-field", default="prediction_set_route_decision")
    parser.add_argument("--output-routed-field", default="prediction_set_routed")
    args = parser.parse_args()

    fallback_predictions = set(args.fallback_predictions)
    rows = load_jsonl(Path(args.input_jsonl))
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    route_counts: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            next_row = dict(row)
            if next_row.get(args.base_route_field) != args.accept_value:
                decision = "route_low_confidence"
            elif next_row.get("prediction") in fallback_predictions:
                decision = f"route_prediction_set_{next_row.get('prediction')}"
            else:
                decision = "accept"
            next_row[args.output_route_field] = decision
            next_row[args.output_routed_field] = decision != "accept"
            route_counts[decision] = route_counts.get(decision, 0) + 1
            handle.write(json.dumps(next_row, ensure_ascii=True) + "\n")

    print(f"wrote {output_path}", flush=True)
    print(json.dumps({"route_counts": route_counts}, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
