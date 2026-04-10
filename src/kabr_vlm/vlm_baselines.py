import json
import os
import re
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from PIL import Image

from kabr_vlm.data.kabr_frames import (
    KABRClipRecord,
    KABRFrameRecord,
    group_frame_records_into_clips,
    read_charades_frame_annotations,
    read_kabr_frame_annotations,
    sample_clip_frames,
    sample_records,
)
from kabr_vlm.datasets import SPECIES_LABELS, get_dataset_spec, ordered_label_names

BEHAVIOR_LABELS = ["Walk", "Graze", "Browse", "Head Up", "Auto-Groom", "Trot", "Run", "Occluded"]
SUPPORTED_TASKS = {"behavior", "species", "joint"}
SUPPORTED_PROMPT_MODES = {"plain", "structured", "label"}
INTERNVL_IMAGE_SIZE = 448
INTERNVL_MAX_TILES_PER_FRAME = 1


def _record_input_id(record: KABRFrameRecord | KABRClipRecord) -> str:
    if isinstance(record, KABRFrameRecord):
        return f"{record.original_video_id}:{record.frame_id}"
    return record.clip_id


def _load_existing_prediction_state(output_path: Path) -> tuple[set[str], int, int]:
    if not output_path.exists():
        return set(), 0, 0

    completed_ids: set[str] = set()
    parsed = 0
    correct = 0
    with output_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            input_id = row.get("input_id")
            if isinstance(input_id, str):
                completed_ids.add(input_id)
            if row.get("parsed") is True:
                parsed += 1
            if row.get("correct") is True:
                correct += 1
    return completed_ids, parsed, correct


def build_prompt(
    task: str,
    unit: str,
    prompt_mode: str,
    dataset: str = "kabr",
    behavior_labels: list[str] | None = None,
    species_labels: list[str] | None = None,
) -> str:
    noun = "frame" if unit == "frame" else "clip represented by ordered frames"
    if prompt_mode not in SUPPORTED_PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
    spec = get_dataset_spec(dataset)
    dataset_name = spec.prompt_name
    behavior_candidates = behavior_labels or BEHAVIOR_LABELS
    species_candidates = species_labels or list(spec.species_labels) or SPECIES_LABELS

    if task == "behavior":
        if "behavior" not in spec.supported_tasks and "action_recognition" not in spec.supported_tasks:
            raise ValueError(f"Task '{task}' is not supported for dataset '{dataset}'.")
        options = ", ".join(behavior_candidates)
        if prompt_mode == "label":
            return (
                f"You are classifying a {noun} from the {dataset_name}. "
                f"Choose exactly one behavior label from: {options}. "
                "Reply with exactly one label and no extra text."
            )
        if prompt_mode == "structured":
            return (
                f"You are classifying a {noun} from the {dataset_name}. "
                f"Choose exactly one behavior label from: {options}. "
                'Reply with code: recognize(entity_type="action", value="<label>").'
            )
        return (
            f"You are classifying a {noun} from the {dataset_name}. "
            f"Choose exactly one behavior label from: {options}. "
            'Reply with JSON: {"behavior": "<label>"}'
        )

    if task == "species":
        if "species" not in spec.supported_tasks and "animal_recognition" not in spec.supported_tasks:
            raise ValueError(f"Task '{task}' is not supported for dataset '{dataset}'.")
        options = ", ".join(species_candidates)
        if prompt_mode == "label":
            return (
                f"You are classifying a {noun} from the {dataset_name}. "
                f"Choose exactly one species label from: {options}. "
                "Reply with exactly one label and no extra text."
            )
        if prompt_mode == "structured":
            return (
                f"You are classifying a {noun} from the {dataset_name}. "
                f"Choose exactly one species label from: {options}. "
                'Reply with code: recognize(entity_type="animal", value="<label>").'
            )
        return (
            f"You are classifying a {noun} from the {dataset_name}. "
            f"Choose exactly one species label from: {options}. "
            'Reply with JSON: {"species": "<label>"}'
        )

    if task == "joint":
        if "joint" not in spec.supported_tasks and "joint_animal_action_recognition" not in spec.supported_tasks:
            raise ValueError(f"Task '{task}' is not supported for dataset '{dataset}'.")
        behavior_options = ", ".join(behavior_candidates)
        species_options = ", ".join(species_candidates)
        if prompt_mode == "structured":
            return (
                f"You are classifying a {noun} from the {dataset_name}. "
                f"Choose one species from: {species_options}. "
                f"Choose one behavior from: {behavior_options}. "
                'Reply with two lines:\n'
                'recognize(entity_type="animal", value="<species>")\n'
                'recognize(entity_type="action", value="<behavior>", condition=\'animal == "<species>"\')'
            )
        return (
            f"You are classifying a {noun} from the {dataset_name}. "
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


def parse_prediction(
    task: str,
    text: str,
    behavior_labels: list[str] | None = None,
    species_labels: list[str] | None = None,
) -> dict[str, str | None]:
    behavior_candidates = behavior_labels or BEHAVIOR_LABELS
    species_candidates = species_labels or SPECIES_LABELS
    if task == "behavior":
        return {"behavior": extract_label(text, behavior_candidates)}
    if task == "species":
        return {"species": extract_label(text, species_candidates)}

    payload = extract_json_object(text)
    behavior = None
    species = None
    if payload is not None:
        raw_behavior = payload.get("behavior")
        raw_species = payload.get("species")
        if isinstance(raw_behavior, str):
            behavior = match_candidate(raw_behavior, behavior_candidates)
        if isinstance(raw_species, str):
            species = match_candidate(raw_species, species_candidates)
    if behavior is None:
        behavior = extract_label(text, behavior_candidates)
    if species is None:
        species = extract_label(text, species_candidates)
    return {"behavior": behavior, "species": species}


def behavior_labels_for_rows(rows: list[KABRFrameRecord] | list[KABRClipRecord]) -> list[str]:
    label_names = {row.label_id: row.label_name for row in rows}
    return [label_names[index] for index in sorted(label_names)]


def load_hf_pipeline(model_id: str):
    if model_id.startswith("OpenGVLab/InternVL2_5-"):
        return load_hf_internvl_generator(model_id)

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
        **_hf_auth_kwargs(),
    )


def load_hf_internvl_generator(model_id: str):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "InternVL inference requires `transformers` and `torch`. "
            "Install with `uv pip install -e \".[infer]\"`."
        ) from exc

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        device_map=None,
        trust_remote_code=True,
        **_hf_auth_kwargs(),
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
        **_hf_auth_kwargs(),
    )
    return InternVLGenerator(model=model, tokenizer=tokenizer, torch_dtype=torch_dtype)


def load_hf_adapter_generator(model_id: str, adapter_path: str | Path | None = None):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Adapter inference requires `transformers`, `torch`, and `peft`. "
            "Install with `uv pip install -e \".[infer,train]\"`."
        ) from exc

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    auth_kwargs = _hf_auth_kwargs()
    if model_id.startswith("Qwen/"):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            **auth_kwargs,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            **auth_kwargs,
        )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **auth_kwargs)
    return HFChatGenerator(model=model, processor=processor)


def load_hf_qwen_generator(model_id: str, adapter_path: str | Path | None = None):
    return load_hf_adapter_generator(model_id=model_id, adapter_path=adapter_path)


class HFChatGenerator:
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
        device = _model_device(self.model)
        encoded = {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}
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


class QwenGenerator(HFChatGenerator):
    pass


class InternVLGenerator:
    def __init__(self, model, tokenizer, torch_dtype) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.torch_dtype = torch_dtype

    def __call__(self, image_paths: list[str], prompt: str, max_new_tokens: int) -> str:
        import torch

        pixel_batches: list[torch.Tensor] = []
        num_patches_list: list[int] = []
        for image_path in image_paths:
            pixel_values = _internvl_load_image(
                image_path,
                input_size=INTERNVL_IMAGE_SIZE,
                max_num=INTERNVL_MAX_TILES_PER_FRAME,
            )
            num_patches_list.append(pixel_values.shape[0])
            pixel_batches.append(pixel_values)

        pixel_values = torch.cat(pixel_batches, dim=0) if pixel_batches else None
        if pixel_values is not None:
            device = _model_device(self.model)
            pixel_values = pixel_values.to(device=device, dtype=self.torch_dtype)

        if len(image_paths) <= 1:
            question = f"<image>\n{prompt}"
            chat_kwargs: dict[str, Any] = {}
        else:
            frame_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(image_paths)))
            question = f"{frame_prefix}{prompt}"
            chat_kwargs = {"num_patches_list": num_patches_list}

        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
            **chat_kwargs,
        )
        if isinstance(response, tuple):
            response = response[0]
        return str(response).strip()


def _model_device(model) -> str:
    if hasattr(model, "device"):
        return str(model.device)
    try:
        return str(next(model.parameters()).device)
    except (AttributeError, StopIteration):
        return "cpu"


def _hf_auth_kwargs() -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return {}
    return {"token": token}


def _internvl_build_transform(input_size: int):
    try:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
    except ImportError as exc:
        raise RuntimeError(
            "InternVL inference requires `torchvision`. Install with `uv pip install -e \".[infer]\"`."
        ) from exc

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _internvl_find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = INTERNVL_IMAGE_SIZE,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    target_aspect_ratio = _internvl_find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized = image.resize((target_width, target_height))
    processed_images: list[Image.Image] = []
    tiles_per_row = target_width // image_size
    for index in range(blocks):
        box = (
            (index % tiles_per_row) * image_size,
            (index // tiles_per_row) * image_size,
            ((index % tiles_per_row) + 1) * image_size,
            ((index // tiles_per_row) + 1) * image_size,
        )
        processed_images.append(resized.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _internvl_load_image(
    image_path: str,
    input_size: int = INTERNVL_IMAGE_SIZE,
    max_num: int = 12,
):
    import torch

    transform = _internvl_build_transform(input_size=input_size)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        processed_images = _internvl_dynamic_preprocess(
            image=image,
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num,
        )
    pixel_values = [transform(tile) for tile in processed_images]
    return torch.stack(pixel_values)


def run_hf_generation(generator, image_paths: list[str], prompt: str, max_new_tokens: int) -> str:
    if isinstance(generator, (HFChatGenerator, InternVLGenerator)):
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
    resume: bool = False,
    dataset: str = "kabr",
) -> dict[str, float | int | str]:
    if task not in SUPPORTED_TASKS:
        supported = ", ".join(sorted(SUPPORTED_TASKS))
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {supported}")

    sample = sample_records(rows, max_samples=max_samples, seed=seed)
    generator = load_hf_pipeline(model_id)
    behavior_labels = behavior_labels_for_rows(rows)
    prompt = build_prompt(task, unit="frame", prompt_mode=prompt_mode, dataset=dataset, behavior_labels=behavior_labels)
    return evaluate_records(
        records=sample,
        generator=generator,
        prompt=prompt,
        task=task,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        output_path=output_path,
        unit="frame",
        resume=resume,
        dataset=dataset,
        behavior_labels=behavior_labels,
        species_labels=list(get_dataset_spec(dataset).species_labels),
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
    resume: bool = False,
    dataset: str = "kabr",
) -> dict[str, float | int | str]:
    if task not in SUPPORTED_TASKS:
        supported = ", ".join(sorted(SUPPORTED_TASKS))
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {supported}")

    clips = group_frame_records_into_clips(rows)
    sample = sample_records(clips, max_samples=max_samples, seed=seed)
    generator = load_hf_pipeline(model_id)
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
        resume=resume,
        dataset=dataset,
        behavior_labels=behavior_labels,
        species_labels=list(get_dataset_spec(dataset).species_labels),
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
    resume: bool = False,
    dataset: str = "kabr",
    behavior_labels: list[str] | None = None,
    species_labels: list[str] | None = None,
) -> dict[str, float | int | str]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(records)

    completed_ids: set[str] = set()
    parsed = 0
    correct = 0
    mode = "w"
    if resume:
        completed_ids, parsed, correct = _load_existing_prediction_state(output_path)
        mode = "a"

    with output_path.open(mode, encoding="utf-8") as handle:
        for row in records:
            record_id = _record_input_id(row)
            if record_id in completed_ids:
                continue
            if isinstance(row, KABRFrameRecord):
                image_paths = [row.image_path]
                input_id = record_id
                gold = gold_for_record(task, species=row.species, behavior=row.label_name)
                extra = {
                    "original_video_id": row.original_video_id,
                    "video_id": row.video_id,
                    "frame_ids": [row.frame_id],
                    "num_input_frames": 1,
                }
            else:
                image_paths = sample_clip_frames(row, max_frames=max_frames_per_clip)
                input_id = record_id
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
            pred = parse_prediction(task, raw_text, behavior_labels=behavior_labels, species_labels=species_labels)
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
                "dataset": dataset,
                "model_id": model_id,
                "gold": gold,
                "prediction": pred,
                "parsed": parsed_ok,
                "correct": correct_flag,
                "raw_text": raw_text,
                "input_image_paths": image_paths,
                "label_space": {
                    "behavior": behavior_labels or [],
                    "species": species_labels or [],
                },
                **extra,
            }
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")
            handle.flush()
            completed_ids.add(input_id)

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
    resume: bool = False,
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
        resume=resume,
        dataset="kabr",
    )
