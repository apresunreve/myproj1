#!/usr/bin/env python3
"""Summarize small-VLM zero-shot evaluation metrics.

Reads metrics files written by scripts/eval_cross_dataset_vlm.py under
artifacts/small_vlm_eval/{kabr,baboonland}/ and emits a compact CSV/Markdown
summary with per-dataset and combined metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = ("kabr", "baboonland")


def read_model_list(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            slug = (row.get("slug") or "").strip()
            model_id = (row.get("model_id") or "").strip()
            if slug and model_id:
                rows.append({"slug": slug, "model_id": model_id})
    return rows


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def rank_values(rows: list[dict], key: str) -> dict[str, int | None]:
    present = [(row["slug"], row.get(key)) for row in rows if isinstance(row.get(key), (int, float))]
    present.sort(key=lambda item: item[1], reverse=True)
    ranks: dict[str, int | None] = {row["slug"]: None for row in rows}
    for rank, (slug, _) in enumerate(present, start=1):
        ranks[slug] = rank
    return ranks


def fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_rows(model_rows: list[dict[str, str]], eval_root: Path, prediction_suffix: str) -> list[dict]:
    rows: list[dict] = []
    for model in model_rows:
        slug = model["slug"]
        out: dict = {
            "slug": slug,
            "model_id": model["model_id"],
            "status": "completed",
            "notes": "",
        }
        total_correct = 0
        total_count = 0
        f1_values: list[float] = []
        missing: list[str] = []

        for dataset in DATASETS:
            metrics_path = eval_root / dataset / f"{slug}_predictions{prediction_suffix}.jsonl.metrics.json"
            metrics = load_metrics(metrics_path)
            if metrics is None:
                missing.append(dataset)
                out[f"{dataset}_accuracy"] = None
                out[f"{dataset}_macro_f1"] = None
                out[f"{dataset}_parsed_rate"] = None
                out[f"{dataset}_n_total"] = None
                out[f"{dataset}_n_correct"] = None
                continue

            accuracy = metrics.get("accuracy")
            macro_f1 = metrics.get("macro_f1")
            parsed_rate = metrics.get("parsed_rate")
            n_total = int(metrics.get("n_total") or 0)
            n_correct = int(metrics.get("n_correct") or 0)

            out[f"{dataset}_accuracy"] = accuracy
            out[f"{dataset}_macro_f1"] = macro_f1
            out[f"{dataset}_parsed_rate"] = parsed_rate
            out[f"{dataset}_n_total"] = n_total
            out[f"{dataset}_n_correct"] = n_correct

            total_correct += n_correct
            total_count += n_total
            if isinstance(macro_f1, (int, float)):
                f1_values.append(float(macro_f1))

        out["combined_accuracy"] = (total_correct / total_count) if total_count else None
        out["combined_macro_f1"] = (sum(f1_values) / len(f1_values)) if f1_values else None
        if missing:
            out["status"] = "partial" if len(missing) < len(DATASETS) else "missing"
            out["notes"] = "missing metrics: " + ",".join(missing)
        rows.append(out)

    rank_keys = ("kabr_accuracy", "kabr_macro_f1", "baboonland_accuracy", "baboonland_macro_f1")
    ranks_by_key = {key: rank_values(rows, key) for key in rank_keys}
    for row in rows:
        ranks = [ranks_by_key[key][row["slug"]] for key in rank_keys]
        ranks = [rank for rank in ranks if rank is not None]
        row["mean_rank"] = (sum(ranks) / len(ranks)) if ranks else None
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "slug",
        "model_id",
        "kabr_accuracy",
        "kabr_macro_f1",
        "kabr_parsed_rate",
        "baboonland_accuracy",
        "baboonland_macro_f1",
        "baboonland_parsed_rate",
        "combined_accuracy",
        "combined_macro_f1",
        "mean_rank",
        "status",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(rows: list[dict], path: Path) -> None:
    headers = [
        "Model",
        "KABR Acc/F1",
        "KABR Parsed",
        "Bbl Acc/F1",
        "Bbl Parsed",
        "Combined Acc",
        "Combined F1",
        "Mean Rank",
        "Status",
    ]
    lines = [
        "# Small VLM Evaluation Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in sorted(rows, key=lambda r: (r.get("mean_rank") is None, r.get("mean_rank") or 9999)):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["slug"],
                    f"{fmt(row.get('kabr_accuracy'))} / {fmt(row.get('kabr_macro_f1'))}",
                    fmt(row.get("kabr_parsed_rate")),
                    f"{fmt(row.get('baboonland_accuracy'))} / {fmt(row.get('baboonland_macro_f1'))}",
                    fmt(row.get("baboonland_parsed_rate")),
                    fmt(row.get("combined_accuracy")),
                    fmt(row.get("combined_macro_f1")),
                    fmt(row.get("mean_rank")),
                    row.get("status") or "",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Notes:")
    for row in rows:
        if row.get("notes"):
            lines.append(f"- {row['slug']}: {row['notes']}")
    if not any(row.get("notes") for row in rows):
        lines.append("- None.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-list", default="configs/small_vlm_eval_models.tsv")
    parser.add_argument("--eval-root", default="artifacts/small_vlm_eval")
    parser.add_argument("--output-csv", default="artifacts/small_vlm_eval/small_vlm_summary.csv")
    parser.add_argument("--output-md", default="artifacts/small_vlm_eval/small_vlm_summary.md")
    parser.add_argument(
        "--prediction-suffix",
        default="",
        help="Suffix between '<slug>_predictions' and '.jsonl' (for example: '_smoke').",
    )
    args = parser.parse_args()

    model_rows = read_model_list(Path(args.model_list))
    rows = build_rows(model_rows, Path(args.eval_root), args.prediction_suffix)
    write_csv(rows, Path(args.output_csv))
    write_markdown(rows, Path(args.output_md))
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
