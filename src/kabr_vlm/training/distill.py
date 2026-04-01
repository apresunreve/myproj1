import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from kabr_vlm.config import AppConfig
from kabr_vlm.training.frozen_features import (
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


def train_distilled_feature_model(config: AppConfig) -> dict[str, Any]:
    train_examples = load_jsonl_examples(config.data.train_file, config.data.max_train_samples)
    val_examples = load_jsonl_examples(config.data.val_file, config.data.max_val_samples)
    train_dataset = QwenTaskDataset(train_examples)
    val_dataset = QwenTaskDataset(val_examples)

    extractor = FrozenFeatureExtractor(config)
    teacher_model, teacher_processor = _load_teacher(config)

    train_batch = extractor.extract([train_dataset[index] for index in range(len(train_dataset))], config.training.feature_batch_size)
    val_batch = extractor.extract([val_dataset[index] for index in range(len(val_dataset))], config.training.feature_batch_size)
    features_dir = config.training.output_dir / "features"
    _write_feature_tensor(features_dir / "train.pt", train_batch, config.data.task_mode)
    _write_feature_tensor(features_dir / "val.pt", val_batch, config.data.task_mode)

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

    teacher_train_probs = []
    for example in train_dataset._examples:
        scores = _score_candidate_answers(teacher_model, teacher_processor, example.prompt, example.image_paths, class_names)
        teacher_train_probs.append(F.softmax(scores / config.training.distill_temperature, dim=-1))
    teacher_train_probs = torch.stack(teacher_train_probs)

    for _ in range(config.training.classifier_epochs):
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
    return metrics


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
