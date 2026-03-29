import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from kabr_vlm.config import AppConfig
from kabr_vlm.metrics import score_prediction_records, write_confusion_csv, write_metrics


SPECIES_PROMPT = (
    "You are classifying an ordered set of wildlife drone frames from the KABR dataset. "
    "Choose exactly one species label from: giraffe, plains zebra, grevys zebra. "
    "Reply with exactly one label and no extra text."
)


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


def build_species_messages(image_paths: list[str], answer: str | None = None) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [
        *[{"type": "image", "image": path} for path in image_paths],
        {"type": "text", "text": SPECIES_PROMPT},
    ]
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


@dataclass
class QwenSpeciesExample:
    example_id: str
    image_paths: list[str]
    answer: str
    metadata: dict[str, Any]


class QwenSpeciesDataset:
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self._examples = [
            QwenSpeciesExample(
                example_id=row["id"],
                image_paths=list(row["image_paths"]),
                answer=row["answer"].strip(),
                metadata=dict(row.get("metadata", {})),
            )
            for row in examples
        ]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> QwenSpeciesExample:
        return self._examples[index]


def _open_images(image_paths: list[str]) -> list[Image.Image]:
    return [Image.open(path).convert("RGB") for path in image_paths]


class QwenSpeciesCollator:
    def __init__(self, processor, max_seq_length: int) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __call__(self, features: list[QwenSpeciesExample]) -> dict[str, Any]:
        batch_images = [_open_images(feature.image_paths) for feature in features]
        try:
            full_texts = [
                self.processor.apply_chat_template(
                    build_species_messages(feature.image_paths, answer=feature.answer),
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for feature in features
            ]
            prompt_texts = [
                self.processor.apply_chat_template(
                    build_species_messages(feature.image_paths),
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


def create_lora_model(config: AppConfig):
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
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

    processor = AutoProcessor.from_pretrained(
        config.model.resolved_processor_id,
        trust_remote_code=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model.base_model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
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


def train_qwen_species_model(config: AppConfig) -> dict[str, Any]:
    try:
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "Fine-tuning requires `transformers`. Install with `uv pip install -e \".[train]\"`."
        ) from exc

    train_examples = load_jsonl_examples(config.data.train_file, config.data.max_train_samples)
    val_examples = load_jsonl_examples(config.data.val_file, config.data.max_val_samples)
    model, processor = create_lora_model(config)

    train_dataset = QwenSpeciesDataset(train_examples)
    val_dataset = QwenSpeciesDataset(val_examples)
    collator = QwenSpeciesCollator(processor=processor, max_seq_length=config.training.max_seq_length)
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
    }
    with (config.training.output_dir / "training_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return metrics


def evaluate_qwen_species_adapter(
    annotation_path: str | Path,
    image_root: str | Path,
    classes_path: str | Path,
    model_id: str,
    adapter_path: str | Path,
    output_path: str | Path,
    max_samples: int | None = None,
    seed: int = 42,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    from kabr_vlm.data.kabr_frames import read_kabr_frame_annotations
    from kabr_vlm.metrics import load_prediction_records
    from kabr_vlm.vlm_baselines import load_hf_qwen_generator

    rows = read_kabr_frame_annotations(annotation_path, image_root, classes_path)
    generator = load_hf_qwen_generator(model_id=model_id, adapter_path=adapter_path)
    from kabr_vlm.data.kabr_frames import group_frame_records_into_clips

    summary = evaluate_clip_records_with_generator(
        rows=rows,
        generator=generator,
        model_id=model_id,
        task="species",
        max_samples=max_samples or len(group_frame_records_into_clips(rows)),
        seed=seed,
        max_new_tokens=max_new_tokens,
        max_frames_per_clip=8,
        output_path=output_path,
        prompt_mode="label",
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
):
    from kabr_vlm.data.kabr_frames import group_frame_records_into_clips, sample_records
    from kabr_vlm.vlm_baselines import build_prompt, evaluate_records

    clips = group_frame_records_into_clips(rows)
    sample = sample_records(clips, max_samples=max_samples, seed=seed)
    prompt = build_prompt(task, unit="clip", prompt_mode=prompt_mode)
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
    )
