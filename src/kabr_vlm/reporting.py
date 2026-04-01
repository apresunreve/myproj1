import csv
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PaperRun(BaseModel):
    name: str
    task: str
    family: str
    method: str
    model: str
    output_format: str = "plain"
    metrics_path: Path
    training_metrics_path: Path | None = None
    checkpoint_path: Path | None = None
    notes: str | None = None
    runtime_hours: float | None = None
    trainable_params: int | None = None


class PaperManifest(BaseModel):
    runs: list[PaperRun] = Field(default_factory=list)


def load_paper_manifest(path: str | Path) -> PaperManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return PaperManifest.model_validate(raw)


def _size_mb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (1024 * 1024)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_paper_results(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    manifest = load_paper_manifest(manifest_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    main_rows = []
    per_class_rows = []
    summary_rows = []

    for run in manifest.runs:
        metrics = _load_json(run.metrics_path)
        training_metrics = _load_json(run.training_metrics_path)
        evaluated = metrics.get("evaluated", 0)
        parsed = metrics.get("parsed", 0)
        parse_rate = (parsed / evaluated) if evaluated else 0.0
        runtime_hours = run.runtime_hours
        if runtime_hours is None and training_metrics.get("train_runtime") is not None:
            runtime_hours = float(training_metrics["train_runtime"]) / 3600.0
        trainable_params = run.trainable_params
        if trainable_params is None:
            trainable_params = training_metrics.get("trainable_params")
        checkpoint_size_mb = _size_mb(run.checkpoint_path)
        main_rows.append(
            {
                "task": run.task,
                "family": run.family,
                "method": run.method,
                "model": run.model,
                "run_name": run.name,
                "output_format": run.output_format,
                "accuracy": metrics.get("accuracy", 0.0),
                "macro_f1": metrics.get("macro_f1", 0.0),
                "parse_rate": parse_rate,
                "evaluated": evaluated,
                "runtime_hours": runtime_hours,
                "trainable_params": trainable_params,
                "checkpoint_size_mb": checkpoint_size_mb,
                "metrics_path": str(run.metrics_path),
                "notes": run.notes or "",
            }
        )
        per_class = metrics.get("per_class", {})
        for label, label_metrics in per_class.items():
            per_class_rows.append(
                {
                    "task": run.task,
                    "run_name": run.name,
                    "family": run.family,
                    "method": run.method,
                    "model": run.model,
                    "label": label,
                    "precision": label_metrics.get("precision", 0.0),
                    "recall": label_metrics.get("recall", 0.0),
                    "f1": label_metrics.get("f1", 0.0),
                    "support": label_metrics.get("support", 0),
                }
            )
        summary_rows.append(
            {
                "name": run.name,
                "task": run.task,
                "family": run.family,
                "method": run.method,
                "model": run.model,
                "output_format": run.output_format,
                "metrics_path": str(run.metrics_path),
                "training_metrics_path": str(run.training_metrics_path) if run.training_metrics_path else None,
                "checkpoint_path": str(run.checkpoint_path) if run.checkpoint_path else None,
            }
        )

    main_csv = destination / "main_table.csv"
    with main_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(main_rows[0].keys()) if main_rows else [])
        if main_rows:
            writer.writeheader()
            writer.writerows(main_rows)

    per_class_csv = destination / "per_class_f1.csv"
    with per_class_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class_rows[0].keys()) if per_class_rows else [])
        if per_class_rows:
            writer.writeheader()
            writer.writerows(per_class_rows)

    main_md = destination / "main_table.md"
    with main_md.open("w", encoding="utf-8") as handle:
        handle.write("| run_name | task | family | method | model | accuracy | macro_f1 | parse_rate | runtime_hours | trainable_params | checkpoint_size_mb |\n")
        handle.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in main_rows:
            runtime_hours = "" if row["runtime_hours"] is None else f"{row['runtime_hours']:.2f}"
            trainable_params = "" if row["trainable_params"] is None else row["trainable_params"]
            checkpoint_size_mb = "" if row["checkpoint_size_mb"] is None else f"{row['checkpoint_size_mb']:.2f}"
            handle.write(
                f"| {row['run_name']} | {row['task']} | {row['family']} | {row['method']} | {row['model']} | "
                f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['parse_rate']:.4f} | "
                f"{runtime_hours} | {trainable_params} | {checkpoint_size_mb} |\n"
            )

    summary_json = destination / "summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump({"runs": summary_rows}, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    return {
        "main_table_csv": main_csv,
        "main_table_md": main_md,
        "per_class_csv": per_class_csv,
        "summary_json": summary_json,
    }
