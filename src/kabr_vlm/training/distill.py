import json
import traceback
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from kabr_vlm.config import AppConfig
from kabr_vlm.training.frozen_features import (
    FeatureBatch,
    FeatureClassifier,
    FrozenFeatureExtractor,
    _checkpoint_size_mb,
    _encode_labels,
    _load_classifier_checkpoint,
    _write_feature_tensor,
    evaluate_frozen_feature_model,
)
from kabr_vlm.training.qwen_tasks import TASK_TO_LABELS, QwenTaskDataset, choose_precision_flags, load_jsonl_examples


def _device_for_model(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _write_status(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"updated_at": _timestamp(), **payload}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _load_feature_tensor(path: Path) -> FeatureBatch:
    payload = torch.load(path, map_location="cpu")
    return FeatureBatch(
        ids=list(payload["ids"]),
        labels=list(payload["labels"]),
        features=payload["features"],
    )


def _load_teacher(config: AppConfig):
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if config.model.adapter_path is None:
        raise ValueError("Distillation requires model.adapter_path pointing to the teacher LoRA adapter.")

    bf16, fp16 = choose_precision_flags(config)
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    processor = AutoProcessor.from_pretrained(config.model.resolved_processor_id, trust_remote_code=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model.base_model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(config.model.adapter_path))
    model.eval()
    return model, processor


def _score_candidate_answers(
    model,
    processor,
    prompt: str,
    image_paths: list[str],
    candidates: list[str],
) -> torch.Tensor:
    device = _device_for_model(model)
    scores = []
    for answer in candidates:
        with ExitStack() as stack:
            images = [stack.enter_context(Image.open(path)).convert("RGB") for path in image_paths]
            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": path} for path in image_paths],
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            full_messages = [
                *prompt_messages,
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            prompt_text = processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            encoded_full = processor(
                text=[full_text],
                images=[images],
                return_tensors="pt",
                padding=True,
            )
            encoded_prompt = processor(
                text=[prompt_text],
                images=[images],
                return_tensors="pt",
                padding=True,
            )

        encoded_full = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded_full.items()
        }
        encoded_prompt = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded_prompt.items()
        }
        with torch.inference_mode():
            outputs = model(**encoded_full, return_dict=True)

        logits = outputs.logits[:, :-1, :]
        target_ids = encoded_full["input_ids"][:, 1:]
        attention_mask = encoded_full["attention_mask"][:, 1:]
        token_log_probs = F.log_softmax(logits.float(), dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        prompt_len = int(encoded_prompt["attention_mask"][0].sum().item())
        answer_mask = attention_mask[0].clone()
        if prompt_len > 0:
            answer_mask[: prompt_len - 1] = 0
        scores.append((token_log_probs[0] * answer_mask).sum().cpu())

        del outputs, logits, target_ids, attention_mask, token_log_probs, encoded_full, encoded_prompt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.stack(scores).cpu()


def _extract_or_load_features(
    extractor: FrozenFeatureExtractor,
    dataset: QwenTaskDataset,
    batch_size: int,
    path: Path,
    task_mode: str,
    split_name: str,
) -> FeatureBatch:
    if path.exists():
        _log(f"Loading cached {split_name} features from {path}.")
        return _load_feature_tensor(path)

    _log(f"Extracting {split_name} features for {len(dataset)} examples.")
    batch = extractor.extract([dataset[index] for index in range(len(dataset))], batch_size)
    _write_feature_tensor(path, batch, task_mode)
    _log(f"Saved {split_name} features to {path}.")
    return batch


def _compute_or_load_teacher_probs(
    dataset: QwenTaskDataset,
    class_names: list[str],
    teacher_model,
    teacher_processor,
    temperature: float,
    cache_path: Path,
    log_every: int = 25,
) -> torch.Tensor:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    saved_probs: list[torch.Tensor] = []
    start_index = 0
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        cached_tensor = payload.get("teacher_probs")
        if cached_tensor is not None:
            saved_probs = [row.clone() for row in cached_tensor]
            start_index = len(saved_probs)
            _log(
                f"Resuming teacher probability cache from {cache_path} at "
                f"{start_index}/{len(dataset)} examples."
            )

    total = len(dataset)
    for index in range(start_index, total):
        example = dataset._examples[index]
        _log(f"Scoring teacher candidates for example {index + 1}/{total}: {example.example_id}")
        scores = _score_candidate_answers(
            teacher_model,
            teacher_processor,
            example.prompt,
            example.image_paths,
            class_names,
        )
        saved_probs.append(F.softmax(scores / temperature, dim=-1).cpu())

        completed = index + 1
        if completed % log_every == 0 or completed == total:
            stacked = torch.stack(saved_probs)
            temp_path = cache_path.with_suffix(".tmp")
            torch.save(
                {
                    "teacher_probs": stacked,
                    "completed": completed,
                    "total": total,
                    "class_names": class_names,
                },
                temp_path,
            )
            temp_path.replace(cache_path)
            _log(f"Saved teacher probability cache at {completed}/{total} examples to {cache_path}.")

    return torch.stack(saved_probs)


def train_distilled_feature_model(config: AppConfig) -> dict[str, Any]:
    status_path = config.training.output_dir / "distill_status.json"
    try:
        _write_status(
            status_path,
            state="starting",
            method=config.training.method,
            task_mode=config.data.task_mode,
            output_dir=str(config.training.output_dir),
        )
        train_examples = load_jsonl_examples(config.data.train_file, config.data.max_train_samples)
        val_examples = load_jsonl_examples(config.data.val_file, config.data.max_val_samples)
        train_dataset = QwenTaskDataset(train_examples)
        val_dataset = QwenTaskDataset(val_examples)
        _log(
            f"Loaded distillation datasets: train={len(train_dataset)} examples, "
            f"val={len(val_dataset)} examples."
        )

        features_dir = config.training.output_dir / "features"
        extractor = FrozenFeatureExtractor(config)
        _write_status(
            status_path,
            state="feature_extraction",
            train_samples=len(train_dataset),
            val_samples=len(val_dataset),
            features_dir=str(features_dir),
        )
        train_batch = _extract_or_load_features(
            extractor=extractor,
            dataset=train_dataset,
            batch_size=config.training.feature_batch_size,
            path=features_dir / "train.pt",
            task_mode=config.data.task_mode,
            split_name="train",
        )
        val_batch = _extract_or_load_features(
            extractor=extractor,
            dataset=val_dataset,
            batch_size=config.training.feature_batch_size,
            path=features_dir / "val.pt",
            task_mode=config.data.task_mode,
            split_name="val",
        )

        class_names = TASK_TO_LABELS[config.data.task_mode]
        train_targets = _encode_labels(train_batch.labels, class_names)
        val_targets = _encode_labels(val_batch.labels, class_names)
        classifier = FeatureClassifier(
            input_dim=train_batch.features.shape[1],
            output_dim=len(class_names),
            classifier_type=config.training.classifier_type,
            hidden_dim=config.training.classifier_hidden_dim,
            dropout=config.training.classifier_dropout,
        )
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=config.training.classifier_learning_rate,
            weight_decay=config.training.classifier_weight_decay,
        )
        best_state = None
        best_val_loss = float("inf")
        history: dict[str, float] = {}

        _write_status(
            status_path,
            state="teacher_scoring",
            train_samples=len(train_dataset),
            val_samples=len(val_dataset),
        )
        teacher_model, teacher_processor = _load_teacher(config)
        teacher_cache_path = config.training.output_dir / "teacher_probs.pt"
        teacher_train_probs = _compute_or_load_teacher_probs(
            dataset=train_dataset,
            class_names=class_names,
            teacher_model=teacher_model,
            teacher_processor=teacher_processor,
            temperature=config.training.distill_temperature,
            cache_path=teacher_cache_path,
        )
        del teacher_model, teacher_processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _write_status(
            status_path,
            state="classifier_training",
            teacher_cache_path=str(teacher_cache_path),
            teacher_cache_examples=int(teacher_train_probs.shape[0]),
        )
        for epoch in range(config.training.classifier_epochs):
            classifier.train()
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(train_batch.features)
            hard_loss = F.cross_entropy(logits, train_targets)
            student_log_probs = F.log_softmax(logits / config.training.distill_temperature, dim=-1)
            soft_loss = F.kl_div(student_log_probs, teacher_train_probs, reduction="batchmean")
            loss = (
                config.training.distill_alpha * soft_loss * (config.training.distill_temperature ** 2)
                + (1.0 - config.training.distill_alpha) * hard_loss
            )
            loss.backward()
            optimizer.step()

            classifier.eval()
            with torch.inference_mode():
                val_logits = classifier(val_batch.features)
                val_loss = F.cross_entropy(val_logits, val_targets).item()
                val_accuracy = (val_logits.argmax(dim=-1) == val_targets).float().mean().item()
            _log(
                f"Classifier epoch {epoch + 1}/{config.training.classifier_epochs}: "
                f"train_loss={loss.item():.4f} hard_loss={hard_loss.item():.4f} "
                f"soft_loss={soft_loss.item():.4f} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
                history = {
                    "train_loss": float(loss.item()),
                    "hard_loss": float(hard_loss.item()),
                    "soft_loss": float(soft_loss.item()),
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_accuracy),
                }

        if best_state is not None:
            classifier.load_state_dict(best_state)

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
                "teacher_adapter_path": str(config.model.adapter_path),
                "distill_temperature": config.training.distill_temperature,
                "distill_alpha": config.training.distill_alpha,
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
        _write_status(
            status_path,
            state="completed",
            checkpoint_path=str(checkpoint_path),
            metrics_path=str(config.training.output_dir / "training_metrics.json"),
        )
        return metrics
    except Exception as exc:
        _log(f"Distillation failed: {exc}")
        _log(traceback.format_exc())
        _write_status(
            status_path,
            state="failed",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise


def evaluate_distilled_feature_model(
    config: AppConfig,
    checkpoint_path: str | Path,
    output_path: str | Path,
    max_samples: int | None = None,
) -> dict[str, Any]:
    return evaluate_frozen_feature_model(
        config=config,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        max_samples=max_samples,
    )
