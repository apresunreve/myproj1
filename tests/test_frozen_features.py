import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from kabr_vlm.config import load_config
from kabr_vlm.training.frozen_features import evaluate_frozen_feature_model


def test_evaluate_frozen_feature_model_uses_cached_features(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    row = {
        "id": "clip-1",
        "video_id": "video-1",
        "image_paths": ["/tmp/fake.jpg"],
        "split": "val",
        "task": "action_recognition",
        "prompt": "Choose one behavior.",
        "answer": "Walk",
        "metadata": {},
    }
    train_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    val_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: test-frozen-eval",
                "seed: 42",
                "data:",
                f"  train_file: {train_path}",
                f"  val_file: {val_path}",
                "  task_mode: action_recognition",
                "  input_type: clip",
                "  max_frames_per_clip: 8",
                "model:",
                "  name: Qwen/Qwen2.5-VL-3B-Instruct",
                "  max_frames: 8",
                "  precision: bf16",
                "  prompt_mode: label",
                "  output_format: plain",
                "training:",
                f"  output_dir: {tmp_path / 'output'}",
                "  method: frozen_linear",
                "  feature_batch_size: 2",
                "  classifier_type: linear",
                "  classifier_epochs: 1",
                "  dry_run: false",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    features_dir = config.training.output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ids": ["clip-1"],
            "labels": ["Walk"],
            "features": torch.tensor([[3.0, -1.0]], dtype=torch.float32),
            "task_mode": "action_recognition",
        },
        features_dir / "val.pt",
    )

    checkpoint_path = config.training.output_dir / "classifier.pt"
    torch.save(
        {
            "state_dict": {
                "network.weight": torch.tensor([[1.0, 0.0]] + [[0.0, 0.0]] * 7, dtype=torch.float32),
                "network.bias": torch.zeros(8, dtype=torch.float32),
            },
            "input_dim": 2,
            "class_names": ["Walk", "Graze", "Browse", "Head Up", "Auto-Groom", "Trot", "Run", "Occluded"],
            "classifier_type": "linear",
            "hidden_dim": 256,
            "dropout": 0.1,
            "task_mode": "action_recognition",
        },
        checkpoint_path,
    )

    summary = evaluate_frozen_feature_model(
        config=config,
        checkpoint_path=checkpoint_path,
        output_path=config.training.output_dir / "predictions.jsonl",
    )

    assert summary["evaluated"] == 1
    assert summary["accuracy"] == 1.0
