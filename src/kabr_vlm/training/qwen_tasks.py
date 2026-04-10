import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from PIL import Image

from kabr_vlm.config import AppConfig
from kabr_vlm.datasets import SPECIES_LABELS, get_dataset_spec
from kabr_vlm.metrics import score_prediction_records, write_confusion_csv, write_metrics
from kabr_vlm.vlm_baselines import BEHAVIOR_LABELS, parse_prediction


TASK_TO_LABELS = {
    "animal_recognition": SPECIES_LABELS,
    "action_recognition": BEHAVIOR_LABELS,
}

TASK_TO_LABEL_KEY = {
    "animal_recognition": "species",
    "action_recognition": "behavior",
}

TASK_TO_EVAL_NAME = {
    "animal_recognition": "species",
    "action_recognition": "behavior",
}


def task_prompt(task_mode: str) -> str:
    if task_mode == "animal_recognition":
        options = ", ".join(SPECIES_LABELS)
        return (
            "You are classifying an ordered set of wildlife drone frames from the KABR dataset. "
            f"Choose exactly one species label from: {options}. "
            "Reply with exactly one label and no extra text."
        )
    if task_mode == "action_recognition":
        options = ", ".join(BEHAVIOR_LABELS)
        return (
            "You are classifying an ordered set of wildlife drone frames from the KABR dataset. "
            f"Choose exactly one behavior label from: {options}. "
            "Reply with exactly one label and no extra text."
        )
    raise ValueError(f"Unsupported task_mode: {task_mode}")


def label_space_for_examples(examples: list[dict[str, Any]], task_mode: str) -> list[str]:
    if examples:
        metadata = examples[0].get("metadata", {})
        label_space = metadata.get("label_space", {}) if isinstance(metadata, dict) else {}
        if task_mode == "action_recognition":
            behavior_labels = label_space.get("behavior")
            if isinstance(behavior_labels, list) and behavior_labels:
                return list(behavior_labels)
        if task_mode == "animal_recognition":
            species_labels = label_space.get("species")
            if isinstance(species_labels, list) and species_labels:
                return list(species_labels)
        dataset = metadata.get("dataset") if isinstance(metadata, dict) else None
        if isinstance(dataset, str):
            spec = get_dataset_spec(dataset)
            if task_mode == "animal_recognition" and spec.species_labels:
                return list(spec.species_labels)
    return list(TASK_TO_LABELS[task_mode])


def load_jsonl_examples(path: Path, max_examples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def choose_precision_flags(config: AppConfig) -> tuple[bool, bool]:
    import torch

    if config.training.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return True, False
    if config.training.fp16 and torch.cuda.is_available():
        return False, True
    return False, False


def build_task_messages(prompt: str, image_paths: list[str], answer: str | None = None) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [
        *[{"type": "image", "image": path} for path in image_paths],
        {"type": "text", "text": prompt},
    ]
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


@dataclass
class QwenTaskExample:
    example_id: str
    image_paths: list[str]
    prompt: str
    answer: str
    label: str
    task_mode: str
    metadata: dict[str, Any]


class QwenTaskDataset:
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self._examples = []
        for row in examples:
            task_mode = row["task"]
            label_key = TASK_TO_LABEL_KEY[task_mode]
            metadata = dict(row.get("metadata", {}))
            label = metadata.get(label_key)
            if not isinstance(label, str):
                label_space = metadata.get("label_space", {}) if isinstance(metadata, dict) else {}
                parsed = parse_prediction(
                    TASK_TO_EVAL_NAME[task_mode],
                    row["answer"].strip(),
                    behavior_labels=label_space.get("behavior"),
                    species_labels=label_space.get("species"),
                )
                label = parsed.get(label_key)
            if label is None:
                raise ValueError(f"Could not parse label from answer for example {row['id']}: {row['answer']}")
            self._examples.append(
                QwenTaskExample(
                    example_id=row["id"],
                    image_paths=list(row["image_paths"]),
                    prompt=row["prompt"].strip(),
                    answer=row["answer"].strip(),
                    label=label,
                    task_mode=task_mode,
                    metadata=metadata,
                )
            )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> QwenTaskExample:
        return self._examples[index]


def _open_images(image_paths: list[str]) -> list[Image.Image]:
    return [Image.open(path).convert("RGB") for path in image_paths]


class QwenTaskCollator:
    def __init__(self, processor, max_seq_length: int) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __call__(self, features: list[QwenTaskExample]) -> dict[str, Any]:
        batch_images = [_open_images(feature.image_paths) for feature in features]
        try:
            full_texts = [
                self.processor.apply_chat_template(
                    build_task_messages(feature.prompt, feature.image_paths, answer=feature.answer),
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for feature in features
            ]
            prompt_texts = [
                self.processor.apply_chat_template(
                    build_task_messages(feature.prompt, feature.image_paths),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for feature in features
            ]
            full_batch = self.processor(
                text=full_texts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
            )
            prompt_batch = self.processor(
                text=prompt_texts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
            )
        finally:
            for image_list in batch_images:
                for image in image_list:
                    image.close()

        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100
        for row_index in range(labels.shape[0]):
            prompt_len = int(prompt_batch["attention_mask"][row_index].sum().item())
            labels[row_index, :prompt_len] = -100
        full_batch["labels"] = labels
        return full_batch


def _hf_auth_kwargs() -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return {}
    return {"token": token}


def create_lora_model(config: AppConfig):
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Fine-tuning requires `transformers`, `torch`, and `peft`. "
            "Install with `uv pip install -e \".[train]\"`."
        ) from exc

    bf16, fp16 = choose_precision_flags(config)
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    auth_kwargs = _hf_auth_kwargs()
    processor = AutoProcessor.from_pretrained(
        config.model.resolved_processor_id,
        trust_remote_code=True,
        **auth_kwargs,
    )
    if config.model.base_model_id.startswith("Qwen/"):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model.base_model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            **auth_kwargs,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            config.model.base_model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            **auth_kwargs,
        )
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=config.training.lora_r,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    return model, processor


def train_hf_task_model(config: AppConfig) -> dict[str, Any]:
    try:
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "Fine-tuning requires `transformers`. Install with `uv pip install -e \".[train]\"`."
        ) from exc

    train_examples = load_jsonl_examples(config.data.train_file, config.data.max_train_samples)
    val_examples = load_jsonl_examples(config.data.val_file, config.data.max_val_samples)
    model, processor = create_lora_model(config)

    train_dataset = QwenTaskDataset(train_examples)
    val_dataset = QwenTaskDataset(val_examples)
    collator = QwenTaskCollator(processor=processor, max_seq_length=config.training.max_seq_length)
    bf16, fp16 = choose_precision_flags(config)

    args = TrainingArguments(
        output_dir=str(config.training.output_dir),
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        eval_strategy=config.training.eval_strategy,
        save_strategy=config.training.save_strategy,
        logging_steps=config.training.logging_steps,
        save_total_limit=config.training.save_total_limit,
        remove_unused_columns=False,
        report_to="none",
        seed=config.training.seed,
        dataloader_num_workers=0,
        save_only_model=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    result = trainer.train()
    trainer.save_model(str(config.training.output_dir))
    processor.save_pretrained(str(config.training.output_dir))
    metrics = {
        "train_runtime": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "task_mode": config.data.task_mode,
        "output_format": config.model.output_format,
        "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_params": sum(parameter.numel() for parameter in model.parameters()),
    }
    with (config.training.output_dir / "training_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return metrics


def train_qwen_task_model(config: AppConfig) -> dict[str, Any]:
    return train_hf_task_model(config)


def evaluate_hf_adapter(
    annotation_path: str | Path,
    image_root: str | Path,
    classes_path: str | Path,
    model_id: str,
    adapter_path: str | Path,
    task: str,
    output_path: str | Path,
    max_samples: int | None = None,
    seed: int = 42,
    max_new_tokens: int = 8,
    max_frames_per_clip: int = 8,
    prompt_mode: str = "label",
    dataset: str = "kabr",
) -> dict[str, Any]:
    from kabr_vlm.data.kabr_frames import group_frame_records_into_clips, read_charades_frame_annotations
    from kabr_vlm.metrics import load_prediction_records
    from kabr_vlm.vlm_baselines import load_hf_adapter_generator

    rows = read_charades_frame_annotations(annotation_path, image_root, classes_path, dataset=dataset)
    generator = load_hf_adapter_generator(model_id=model_id, adapter_path=adapter_path)

    summary = evaluate_clip_records_with_generator(
        rows=rows,
        generator=generator,
        model_id=model_id,
        task=task,
        max_samples=max_samples or len(group_frame_records_into_clips(rows)),
        seed=seed,
        max_new_tokens=max_new_tokens,
        max_frames_per_clip=max_frames_per_clip,
        output_path=output_path,
        prompt_mode=prompt_mode,
        dataset=dataset,
    )
    metrics_rows = load_prediction_records(output_path)
    metrics = score_prediction_records(metrics_rows)
    metrics_path = Path(output_path).with_suffix(".metrics.json")
    confusion_path = Path(output_path).with_suffix(".confusion.csv")
    write_metrics(metrics_path, metrics)
    write_confusion_csv(confusion_path, metrics)
    summary["metrics_path"] = str(metrics_path)
    summary["confusion_path"] = str(confusion_path)
    summary["macro_f1"] = metrics["macro_f1"]
    return summary


def evaluate_qwen_adapter(
    annotation_path: str | Path,
    image_root: str | Path,
    classes_path: str | Path,
    model_id: str,
    adapter_path: str | Path,
    task: str,
    output_path: str | Path,
    max_samples: int | None = None,
    seed: int = 42,
    max_new_tokens: int = 8,
    max_frames_per_clip: int = 8,
    prompt_mode: str = "label",
    dataset: str = "kabr",
) -> dict[str, Any]:
    return evaluate_hf_adapter(
        annotation_path=annotation_path,
        image_root=image_root,
        classes_path=classes_path,
        model_id=model_id,
        adapter_path=adapter_path,
        task=task,
        output_path=output_path,
        max_samples=max_samples,
        seed=seed,
        max_new_tokens=max_new_tokens,
        max_frames_per_clip=max_frames_per_clip,
        prompt_mode=prompt_mode,
        dataset=dataset,
    )


def evaluate_clip_records_with_generator(
    rows,
    generator,
    model_id: str,
    task: str,
    max_samples: int,
    seed: int,
    max_new_tokens: int,
    max_frames_per_clip: int,
    output_path: str | Path,
    prompt_mode: str,
    dataset: str = "kabr",
):
    from kabr_vlm.data.kabr_frames import group_frame_records_into_clips, sample_records
    from kabr_vlm.vlm_baselines import build_prompt, evaluate_records, behavior_labels_for_rows

    clips = group_frame_records_into_clips(rows)
    sample = sample_records(clips, max_samples=max_samples, seed=seed)
    behavior_labels = behavior_labels_for_rows(rows)
    prompt = build_prompt(task, unit="clip", prompt_mode=prompt_mode, dataset=dataset, behavior_labels=behavior_labels)
    return evaluate_records(
        records=sample,
        generator=generator,
        prompt=prompt,
        task=task,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        output_path=output_path,
        unit="clip",
        max_frames_per_clip=max_frames_per_clip,
        dataset=dataset,
        behavior_labels=behavior_labels,
        species_labels=list(get_dataset_spec(dataset).species_labels),
    )
