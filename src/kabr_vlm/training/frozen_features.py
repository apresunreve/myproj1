import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from kabr_vlm.config import AppConfig
from kabr_vlm.metrics import score_prediction_records, write_confusion_csv, write_metrics
from kabr_vlm.training.qwen_tasks import (
    TASK_TO_EVAL_NAME,
    QwenTaskDataset,
    choose_precision_flags,
    label_space_for_examples,
    load_jsonl_examples,
)


def _images_to_rgb(image_paths: list[str]) -> list[Image.Image]:
    return [Image.open(path).convert("RGB") for path in image_paths]


def _device_for_model(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _checkpoint_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (1024 * 1024)


@dataclass
class FeatureBatch:
    ids: list[str]
    labels: list[str]
    features: torch.Tensor


class FrozenFeatureExtractor:
    def __init__(self, config: AppConfig) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        bf16, fp16 = choose_precision_flags(config)
        if bf16:
            dtype = torch.bfloat16
        elif fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(
            config.model.resolved_processor_id,
            trust_remote_code=True,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model.base_model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def extract(self, examples: list[Any], batch_size: int) -> FeatureBatch:
        batches: list[torch.Tensor] = []
        labels: list[str] = []
        ids: list[str] = []
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            batch_images = [_images_to_rgb(example.image_paths) for example in chunk]
            try:
                prompts = [
                    self.processor.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": [
                                    *[{"type": "image", "image": path} for path in example.image_paths],
                                    {"type": "text", "text": example.prompt},
                                ],
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for example in chunk
                ]
                encoded = self.processor(
                    text=prompts,
                    images=batch_images,
                    return_tensors="pt",
                    padding=True,
                )
            finally:
                for image_list in batch_images:
                    for image in image_list:
                        image.close()

            device = _device_for_model(self.model)
            encoded = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            with torch.inference_mode():
                outputs = self.model(**encoded, output_hidden_states=True, return_dict=True)
            hidden = outputs.hidden_states[-1].detach().float().cpu()
            mask = encoded["attention_mask"].detach().cpu()
            pooled_rows = []
            for row_index in range(hidden.shape[0]):
                length = int(mask[row_index].sum().item())
                pooled_rows.append(hidden[row_index, length - 1])
            batches.append(torch.stack(pooled_rows))
            labels.extend(example.label for example in chunk)
            ids.extend(example.example_id for example in chunk)
        return FeatureBatch(
            ids=ids,
            labels=labels,
            features=torch.cat(batches, dim=0) if batches else torch.empty(0, 0),
        )


class FeatureClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, classifier_type: str, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if classifier_type == "linear":
            self.network = nn.Linear(input_dim, output_dim)
        elif classifier_type == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            raise ValueError(f"Unsupported classifier_type: {classifier_type}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _encode_labels(labels: list[str], class_names: list[str]) -> torch.Tensor:
    label_to_index = {label: index for index, label in enumerate(class_names)}
    return torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)


def _evaluate_classifier(model: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        logits = model(features)
        loss = F.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
    return float(loss.item()), accuracy


def _train_classifier(
    model: nn.Module,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[nn.Module, dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = None
    best_val_loss = float("inf")
    history: dict[str, float] = {}

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_features)
        loss = F.cross_entropy(logits, train_targets)
        loss.backward()
        optimizer.step()
        val_loss, val_accuracy = _evaluate_classifier(model, val_features, val_targets)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history = {
                "train_loss": float(loss.item()),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _write_feature_tensor(path: Path, batch: FeatureBatch, task_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ids": batch.ids,
            "labels": batch.labels,
            "features": batch.features,
            "task_mode": task_mode,
        },
        path,
    )


def _load_feature_tensor(path: Path) -> tuple[FeatureBatch, str]:
    payload = torch.load(path, map_location="cpu")
    batch = FeatureBatch(
        ids=list(payload["ids"]),
        labels=list(payload["labels"]),
        features=payload["features"],
    )
    return batch, str(payload["task_mode"])


def train_frozen_feature_model(config: AppConfig) -> dict[str, Any]:
    train_examples = load_jsonl_examples(config.data.train_file, config.data.max_train_samples)
    val_examples = load_jsonl_examples(config.data.val_file, config.data.max_val_samples)
    train_dataset = QwenTaskDataset(train_examples)
    val_dataset = QwenTaskDataset(val_examples)

    extractor = FrozenFeatureExtractor(config)
    train_batch = extractor.extract([train_dataset[index] for index in range(len(train_dataset))], config.training.feature_batch_size)
    val_batch = extractor.extract([val_dataset[index] for index in range(len(val_dataset))], config.training.feature_batch_size)

    features_dir = config.training.output_dir / "features"
    _write_feature_tensor(features_dir / "train.pt", train_batch, config.data.task_mode)
    _write_feature_tensor(features_dir / "val.pt", val_batch, config.data.task_mode)

    class_names = label_space_for_examples(train_examples, config.data.task_mode)
    train_targets = _encode_labels(train_batch.labels, class_names)
    val_targets = _encode_labels(val_batch.labels, class_names)
    classifier = FeatureClassifier(
        input_dim=train_batch.features.shape[1],
        output_dim=len(class_names),
        classifier_type=config.training.classifier_type,
        hidden_dim=config.training.classifier_hidden_dim,
        dropout=config.training.classifier_dropout,
    )
    classifier, history = _train_classifier(
        model=classifier,
        train_features=train_batch.features,
        train_targets=train_targets,
        val_features=val_batch.features,
        val_targets=val_targets,
        epochs=config.training.classifier_epochs,
        learning_rate=config.training.classifier_learning_rate,
        weight_decay=config.training.classifier_weight_decay,
    )

    checkpoint_path = config.training.output_dir / "classifier.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": classifier.state_dict(),
            "input_dim": train_batch.features.shape[1],
            "class_names": class_names,
            "classifier_type": config.training.classifier_type,
            "hidden_dim": config.training.classifier_hidden_dim,
            "dropout": config.training.classifier_dropout,
            "task_mode": config.data.task_mode,
        },
        checkpoint_path,
    )
    metrics = {
        "method": config.training.method,
        "task_mode": config.data.task_mode,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "classifier_type": config.training.classifier_type,
        "trainable_params": sum(parameter.numel() for parameter in classifier.parameters() if parameter.requires_grad),
        "checkpoint_size_mb": _checkpoint_size_mb(checkpoint_path),
        **history,
    }
    with (config.training.output_dir / "training_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return metrics


def _load_classifier_checkpoint(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    model = FeatureClassifier(
        input_dim=payload["input_dim"],
        output_dim=len(payload["class_names"]),
        classifier_type=payload["classifier_type"],
        hidden_dim=payload["hidden_dim"],
        dropout=payload["dropout"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def evaluate_frozen_feature_model(
    config: AppConfig,
    checkpoint_path: str | Path,
    output_path: str | Path,
    max_samples: int | None = None,
) -> dict[str, Any]:
    examples = load_jsonl_examples(config.data.val_file, max_samples)
    dataset = QwenTaskDataset(examples)
    cache_path = config.training.output_dir / "features" / "val.pt"
    if max_samples is None and cache_path.exists():
        batch, cached_task_mode = _load_feature_tensor(cache_path)
        if cached_task_mode != config.data.task_mode or len(batch.ids) != len(dataset):
            extractor = FrozenFeatureExtractor(config)
            batch = extractor.extract([dataset[index] for index in range(len(dataset))], config.training.feature_batch_size)
        else:
            expected_ids = [example.example_id for example in dataset._examples]
            if batch.ids != expected_ids:
                extractor = FrozenFeatureExtractor(config)
                batch = extractor.extract([dataset[index] for index in range(len(dataset))], config.training.feature_batch_size)
    else:
        extractor = FrozenFeatureExtractor(config)
        batch = extractor.extract([dataset[index] for index in range(len(dataset))], config.training.feature_batch_size)
    classifier, payload = _load_classifier_checkpoint(Path(checkpoint_path))
    with torch.inference_mode():
        logits = classifier(batch.features)
        predictions = logits.argmax(dim=-1).tolist()

    eval_name = TASK_TO_EVAL_NAME[config.data.task_mode]
    label_key = "species" if eval_name == "species" else "behavior"
    class_names = payload["class_names"]
    records = []
    for example, pred_index in zip(dataset._examples, predictions, strict=True):
        gold = {label_key: example.label}
        pred_label = class_names[pred_index]
        prediction = {label_key: pred_label}
        records.append(
            {
                "id": example.example_id,
                "task": eval_name,
                "gold": gold,
                "prediction": prediction,
                "parsed": True,
                "raw_output": pred_label,
                "method": config.training.method,
                "model_id": config.model.base_model_id,
            }
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")

    metrics = score_prediction_records(records)
    metrics_path = destination.with_suffix(".metrics.json")
    confusion_path = destination.with_suffix(".confusion.csv")
    write_metrics(metrics_path, metrics)
    write_confusion_csv(confusion_path, metrics)
    return {
        "evaluated": metrics["evaluated"],
        "parsed": metrics["parsed"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "metrics_path": str(metrics_path),
        "confusion_path": str(confusion_path),
    }
