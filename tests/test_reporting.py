import json
from pathlib import Path

from kabr_vlm.reporting import aggregate_paper_results


def test_aggregate_paper_results_writes_tables(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "task": "behavior",
                "evaluated": 10,
                "parsed": 9,
                "accuracy": 0.7,
                "macro_f1": 0.6,
                "per_class": {
                    "Walk": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 5},
                },
                "labels": ["Walk"],
                "confusion_matrix": {"Walk": {"Walk": 5}},
            }
        ),
        encoding="utf-8",
    )
    train_metrics_path = tmp_path / "training_metrics.json"
    train_metrics_path.write_text(
        json.dumps({"train_runtime": 3600, "trainable_params": 1234}),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "checkpoint.bin"
    checkpoint_path.write_bytes(b"abcd")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "runs:",
                "  - name: demo-run",
                "    task: behavior",
                "    family: vlm",
                "    method: lora",
                "    model: Qwen/Qwen2.5-VL-3B-Instruct",
                "    output_format: plain",
                f"    metrics_path: {metrics_path}",
                f"    training_metrics_path: {train_metrics_path}",
                f"    checkpoint_path: {checkpoint_path}",
            ]
        ),
        encoding="utf-8",
    )

    outputs = aggregate_paper_results(manifest_path, tmp_path / "tables")

    assert outputs["main_table_csv"].exists()
    assert outputs["per_class_csv"].exists()
    assert outputs["summary_json"].exists()
    assert "demo-run" in outputs["main_table_md"].read_text(encoding="utf-8")
