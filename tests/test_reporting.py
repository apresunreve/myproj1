import json
from pathlib import Path

from kabr_vlm.reporting import aggregate_paper_results


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_aggregate_paper_results_computes_transfer_outputs(tmp_path: Path) -> None:
    in_domain_metrics = {
        "evaluated": 10,
        "parsed": 10,
        "accuracy": 0.8,
        "macro_f1": 0.7,
        "per_class": {
            "A": {"precision": 1.0, "recall": 0.8, "f1": 0.9, "support": 5},
            "B": {"precision": 0.6, "recall": 0.6, "f1": 0.5, "support": 5},
        },
    }
    transfer_metrics = {
        "evaluated": 10,
        "parsed": 10,
        "accuracy": 0.5,
        "macro_f1": 0.4,
        "per_class": {
            "A": {"precision": 0.5, "recall": 0.4, "f1": 0.3, "support": 5},
            "B": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 5},
        },
    }

    _write_json(tmp_path / "in_domain.metrics.json", in_domain_metrics)
    _write_json(tmp_path / "transfer.metrics.json", transfer_metrics)

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "task_filter: behavior",
                "skip_missing_metrics: true",
                "runs:",
                "  - name: qwen_kabr_in_domain",
                "    task: behavior",
                "    family: vlm",
                "    method: lora",
                "    model: Qwen/Qwen2.5-VL-3B-Instruct",
                "    train_dataset: kabr",
                "    test_dataset: kabr",
                f"    metrics_path: {tmp_path / 'in_domain.metrics.json'}",
                "  - name: qwen_kabr_to_baboonland",
                "    task: behavior",
                "    family: vlm",
                "    method: lora",
                "    model: Qwen/Qwen2.5-VL-3B-Instruct",
                "    train_dataset: kabr",
                "    test_dataset: baboonland",
                f"    metrics_path: {tmp_path / 'transfer.metrics.json'}",
                "  - name: missing_future_row",
                "    task: behavior",
                "    family: vlm",
                "    method: lora",
                "    model: Qwen/Qwen2.5-VL-3B-Instruct",
                "    train_dataset: baboonland",
                "    test_dataset: kabr",
                f"    metrics_path: {tmp_path / 'does_not_exist.metrics.json'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    outputs = aggregate_paper_results(manifest, tmp_path / "out")

    transfer_gap_csv = Path(outputs["transfer_gap_csv"]).read_text(encoding="utf-8")
    transfer_per_class_csv = Path(outputs["transfer_per_class_csv"]).read_text(encoding="utf-8")
    transfer_summary_md = Path(outputs["transfer_summary_md"]).read_text(encoding="utf-8")
    summary_json = json.loads(Path(outputs["summary_json"]).read_text(encoding="utf-8"))

    assert "qwen_kabr_to_baboonland" in transfer_gap_csv
    assert "0.29999999999999993" in transfer_gap_csv or "0.3" in transfer_gap_csv
    assert "A" in transfer_per_class_csv
    assert "Best cross-dataset result" in transfer_summary_md
    assert summary_json["num_rows"] == 2
    assert summary_json["num_transfer_rows"] == 1
