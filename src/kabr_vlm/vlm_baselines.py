import json
import re
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from PIL import Image

from kabr_vlm.data.kabr_frames import (
    KABRClipRecord,
    KABRFrameRecord,
    SPECIES_LABELS,
    group_frame_records_into_clips,
    read_kabr_frame_annotations,
    sample_clip_frames,
    sample_records,
)

BEHAVIOR_LABELS = ["Walk", "Graze", "Browse", "Head Up", "Auto-Groom", "Trot", "Run", "Occluded"]
SUPPORTED_TASKS = {"behavior", "species", "joint"}
SUPPORTED_PROMPT_MODES = {"plain", "structured", "label"}


def build_prompt(task: str, unit: str, prompt_mode: str) -> str:
    noun = "drone frame" if unit == "frame" else "drone clip represented by ordered frames"
    if prompt_mode not in SUPPORTED_PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")

    if task == "behavior":
        options = ", ".join(BEHAVIOR_LABELS)
        if prompt_mode == "label":
            return (
                f"You are classifying a {noun} from the KABR wildlife dataset. "
                f"Choose exactly one behavior label from: {options}. "
                "Reply with exactly one label and no extra text."
            )
        if prompt_mode == "structured":
            return (
                f"You are classifying a {noun} from the KABR wildlife dataset. "
                f"Choose exactly one behavior label from: {options}. "
                'Reply with code: recognize(entity_type="action", value="<label>").'
            )
        return (
            f"You are classifying a {noun} from the KABR wildlife dataset. "
            f"Choose exactly one behavior label from: {options}. "
            'Reply with JSON: {"behavior": "<label>"}'
        )

    if task == "species":
        options = ", ".join(SPECIES_LABELS)
        if prompt_mode == "label":
            return (
                f"You are classifying a {noun} from the KABR wildlife dataset. "
                f"Choose exactly one species label from: {options}. "
                "Reply with exactly one label and no extra text."
            )
        if prompt_mode == "structured":
            return (
                f"You are classifying a {noun} from the KABR wildlife dataset. "
                f"Choose exactly one species label from: {options}. "
                'Reply with code: recognize(entity_type="animal", value="<label>").'
            )
        return (
            f"You are classifying a {noun} from the KABR wildlife dataset. "
            f"Choose exactly one species label from: {options}. "
            'Reply with JSON: {"species": "<label>"}'
        )

    if task == "joint":
        behavior_options = ", ".join(BEHAVIOR_LABELS)
        species_options = ", ".join(SPECIES_LABELS)
        if prompt_mode == "structured":
            return (
                f"You are classifying a {noun} from the KABR wildlife dataset. "
                f"Choose one species from: {species_options}. "
                f"Choose one behavior from: {behavior_options}. "
                'Reply with two lines:\n'
                'recognize(entity_type="animal", value="<species>")\n'
                'recognize(entity_type="action", value="<behavior>", condition=\'animal == "<species>"\')'
            )
        return (
            f"You are classifying a {noun} from the KABR wildlife dataset. "
            f"Choose one species from: {species_options}. "
            f"Choose one behavior from: {behavior_options}. "
            'Reply with JSON: {"species": "<label>", "behavior": "<label>"}'
        )

    raise ValueError(f"Unsupported task: {task}")


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def match_candidate(value: str, candidates: list[str]) -> str | None:
    normalized = normalize_text(value)
    for candidate in candidates:
        if normalized == normalize_text(candidate):
            return candidate
    for candidate in candidates:
        if normalize_text(candidate) in normalized:
            return candidate
    return None


def extract_label(text: str, candidates: list[str]) -> str | None:
    payload = extract_json_object(text)
    if payload is not None:
        for value in payload.values():
            if isinstance(value, str):
                matched = match_candidate(value, candidates)
                if matched is not None:
                    return matched
    return match_candidate(text, candidates)


def parse_prediction(task: str, text: str) -> dict[str, str | None]:
    if task == "behavior":
        return {"behavior": extract_label(text, BEHAVIOR_LABELS)}
    if task == "species":
        return {"species": extract_label(text, SPECIES_LABELS)}

    payload = extract_json_object(text)
    behavior = None
    species = None
    if payload is not None:
        raw_behavior = payload.get("behavior")
        raw_species = payload.get("species")
        if isinstance(raw_behavior, str):
            behavior = match_candidate(raw_behavior, BEHAVIOR_LABELS)
        if isinstance(raw_species, str):
            species = match_candidate(raw_species, SPECIES_LABELS)
    if behavior is None:
        behavior = extract_label(text, BEHAVIOR_LABELS)
    if species is None:
        species = extract_label(text, SPECIES_LABELS)
    return {"behavior": behavior, "species": species}


def load_hf_pipeline(model_id: str):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for evaluation. Install with `uv pip install -e \".[infer]\"`."
        ) from exc

    return pipeline(
        task="image-text-to-text",
        model=model_id,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True,
    )


def load_hf_qwen_generator(model_id: str, adapter_path: str | Path | None = None):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Qwen adapter inference requires `transformers`, `torch`, and `peft`. "
            "Install with `uv pip install -e \".[infer,train]\"`."
        ) from exc

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return QwenGenerator(model=model, processor=processor)


class QwenGenerator:
    def __init__(self, model, processor) -> None:
        self.model = model
        self.processor = processor

    def __call__(self, image_paths: list[str], prompt: str, max_new_tokens: int) -> str:
        import torch

        with ExitStack() as stack:
            images = [stack.enter_context(Image.open(path)).convert("RGB") for path in image_paths]
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": path} for path in image_paths],
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self.processor(
                text=[prompt_text],
                images=[images],
                return_tensors="pt",
                padding=True,
            )
        encoded = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        prompt_len = encoded["input_ids"].shape[1]
        completion = generated[:, prompt_len:]
        return self.processor.batch_decode(
            completion,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def run_hf_generation(generator, image_paths: list[str], prompt: str, max_new_tokens: int) -> str:
    if callable(generator) and not hasattr(generator, "model"):
        return generator(image_paths=image_paths, prompt=prompt, max_new_tokens=max_new_tokens)
    if callable(generator) and hasattr(generator, "model"):
        return generator(image_paths=image_paths, prompt=prompt, max_new_tokens=max_new_tokens)

    with ExitStack() as stack:
        images = [stack.enter_context(Image.open(path)).convert("RGB") for path in image_paths]
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        output = generator(text=messages, max_new_tokens=max_new_tokens)

    generated = output[0]["generated_text"]
    if isinstance(generated, list):
        for item in reversed(generated):
            if isinstance(item, dict) and item.get("role") == "assistant":
                content = item.get("content")
                if isinstance(content, str):
                    return content
        return json.dumps(generated)
    if isinstance(generated, str):
        return generated
    return str(generated)


def gold_for_record(task: str, species: str, behavior: str) -> dict[str, str]:
    if task == "behavior":
        return {"behavior": behavior}
    if task == "species":
        return {"species": species}
    return {"behavior": behavior, "species": species}


def is_correct(task: str, gold: dict[str, str], pred: dict[str, str | None]) -> bool:
    if task == "behavior":
        return pred["behavior"] == gold["behavior"]
    if task == "species":
        return pred["species"] == gold["species"]
    return pred["behavior"] == gold["behavior"] and pred["species"] == gold["species"]


def evaluate_frame_records(
    rows: list[KABRFrameRecord],
    model_id: str,
    task: str,
    max_samples: int,
    seed: int,
    max_new_tokens: int,
    output_path: str | Path,
    prompt_mode: str = "plain",
) -> dict[str, float | int | str]:
    if task not in SUPPORTED_TASKS:
        supported = ", ".join(sorted(SUPPORTED_TASKS))
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {supported}")

    sample = sample_records(rows, max_samples=max_samples, seed=seed)
    generator = load_hf_pipeline(model_id)
    prompt = build_prompt(task, unit="frame", prompt_mode=prompt_mode)
    return evaluate_records(
        records=sample,
        generator=generator,
        prompt=prompt,
        task=task,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        output_path=output_path,
        unit="frame",
    )


def evaluate_clip_records(
    rows: list[KABRFrameRecord],
    model_id: str,
    task: str,
    max_samples: int,
    seed: int,
    max_new_tokens: int,
    max_frames_per_clip: int,
    output_path: str | Path,
    prompt_mode: str = "plain",
) -> dict[str, float | int | str]:
    if task not in SUPPORTED_TASKS:
        supported = ", ".join(sorted(SUPPORTED_TASKS))
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {supported}")

    clips = group_frame_records_into_clips(rows)
    sample = sample_records(clips, max_samples=max_samples, seed=seed)
    generator = load_hf_pipeline(model_id)
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


def evaluate_records(
    records: list[KABRFrameRecord] | list[KABRClipRecord],
    generator,
    prompt: str,
    task: str,
    model_id: str,
    max_new_tokens: int,
    output_path: str | Path,
    unit: str,
    max_frames_per_clip: int = 8,
) -> dict[str, float | int | str]:
    parsed = 0
    correct = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in records:
            if isinstance(row, KABRFrameRecord):
                image_paths = [row.image_path]
                input_id = f"{row.original_video_id}:{row.frame_id}"
                gold = gold_for_record(task, species=row.species, behavior=row.label_name)
                extra = {
                    "original_video_id": row.original_video_id,
                    "video_id": row.video_id,
                    "frame_ids": [row.frame_id],
                    "num_input_frames": 1,
                }
            else:
                image_paths = sample_clip_frames(row, max_frames=max_frames_per_clip)
                input_id = row.clip_id
                gold = gold_for_record(task, species=row.species, behavior=row.label_name)
                extra = {
                    "original_video_id": row.original_video_id,
                    "video_id": row.video_id,
                    "frame_ids": row.frame_ids,
                    "num_input_frames": len(image_paths),
                    "clip_length": len(row.frame_ids),
                }

            raw_text = run_hf_generation(
                generator=generator,
                image_paths=image_paths,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            pred = parse_prediction(task, raw_text)
            parsed_ok = all(value is not None for value in pred.values())
            if parsed_ok:
                parsed += 1
            correct_flag = is_correct(task, gold, pred)
            if correct_flag:
                correct += 1

            record = {
                "input_id": input_id,
                "unit": unit,
                "task": task,
                "model_id": model_id,
                "gold": gold,
                "prediction": pred,
                "parsed": parsed_ok,
                "correct": correct_flag,
                "raw_text": raw_text,
                "input_image_paths": image_paths,
                **extra,
            }
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")

    total = len(records)
    return {
        "evaluated": total,
        "parsed": parsed,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "output_path": str(output_path),
    }


def evaluate_hf_vlm_on_kabr(
    annotation_path: str | Path,
    image_root: str | Path,
    classes_path: str | Path,
    model_id: str,
    task: str,
    max_samples: int,
    seed: int,
    max_new_tokens: int,
    output_path: str | Path,
    prompt_mode: str = "plain",
) -> dict[str, float | int | str]:
    rows = read_kabr_frame_annotations(annotation_path, image_root, classes_path)
    return evaluate_frame_records(
        rows=rows,
        model_id=model_id,
        task=task,
        max_samples=max_samples,
        seed=seed,
        max_new_tokens=max_new_tokens,
        output_path=output_path,
        prompt_mode=prompt_mode,
    )
