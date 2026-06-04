import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
from contextlib import contextmanager
from types import MethodType

from PIL import Image

from kabr_vlm.config import AppConfig
from kabr_vlm.datasets import SPECIES_LABELS, get_dataset_spec
from kabr_vlm.metrics import score_prediction_records, write_confusion_csv, write_metrics
from kabr_vlm.experimental.mla import apply_qwen25vl_mla_rewrite
from kabr_vlm.vlm_baselines import (
    BEHAVIOR_LABELS,
    MobileVLMSentencePieceTokenizer,
    _ensure_mobilevlm_upstream,
    _has_google_protobuf,
    _internvl_load_image,
    _mobilevlm_snapshot_path,
    _patch_mobilevlm_vision_tower_delay_load,
    parse_prediction,
)


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


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

QWEN_MLA_LORA_TARGET_MODULES = [
    "q_proj",
    "o_proj",
    "kv_latent_proj",
    "k_up_proj",
    "v_up_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

INTERNVL_LORA_TARGET_MODULES = [
    "wqkv",
    "wo",
    "w1",
    "w2",
    "w3",
]


def find_latest_trainer_checkpoint(output_dir: Path) -> Path | None:
    latest_checkpoint: Path | None = None
    latest_step = -1
    if not output_dir.exists():
        return None
    for candidate in output_dir.iterdir():
        if not candidate.is_dir():
            continue
        prefix = "checkpoint-"
        if not candidate.name.startswith(prefix):
            continue
        try:
            step = int(candidate.name[len(prefix):])
        except ValueError:
            continue
        if step > latest_step:
            latest_step = step
            latest_checkpoint = candidate
    return latest_checkpoint


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


def _is_internvl_tokenizer(processor) -> bool:
    return processor.__class__.__name__.endswith("Tokenizer")


def _is_phi3_processor(processor) -> bool:
    return processor.__class__.__name__ == "Phi3VProcessor"


def _phi3_prompt_text(prompt: str, image_count: int, answer: str | None = None, add_generation_prompt: bool = False) -> str:
    image_tokens = "\n".join(f"<|image_{idx}|>" for idx in range(1, image_count + 1))
    text = f"{image_tokens}\nUSER: {prompt}"
    if answer is not None:
        text += f"\nASSISTANT: {answer}"
    elif add_generation_prompt:
        text += "\nASSISTANT:"
    return text


class MobileVLMProcessorBundle:
    def __init__(self, tokenizer, image_processor, model_config, conv_mode: str = "v1") -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def save_pretrained(self, output_dir: str) -> None:
        self.tokenizer.save_pretrained(output_dir)
        self.image_processor.save_pretrained(output_dir)


def _is_mobilevlm_processor(processor) -> bool:
    return isinstance(processor, MobileVLMProcessorBundle)


def _safe_save_processor(processor, output_dir: Path) -> None:
    if _is_internvl_tokenizer(processor):
        processor.save_pretrained(str(output_dir))
        return
    if hasattr(processor, "chat_template"):
        processor.save_pretrained(str(output_dir))
        return

    # Some remote-code processors (for example Phi3VProcessor) don't define
    # `chat_template`, but transformers' generic save_pretrained currently
    # assumes the attribute exists.
    setattr(processor, "chat_template", None)
    try:
        processor.save_pretrained(str(output_dir))
    finally:
        try:
            delattr(processor, "chat_template")
        except AttributeError:
            pass


def _render_messages_for_processor(processor, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except AttributeError:
            pass

    text_parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = message.get("content", [])
        if isinstance(content, list):
            chunks = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
            text = "\n".join(chunk for chunk in chunks if chunk)
        else:
            text = str(content)
        if text:
            text_parts.append(f"{role}: {text}")
    if add_generation_prompt:
        text_parts.append("ASSISTANT:")
    return "\n".join(text_parts)


def _internvl_user_question(prompt: str, image_paths: list[str]) -> str:
    if len(image_paths) <= 1:
        return f"<image>\n{prompt}"
    frame_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(image_paths)))
    return f"{frame_prefix}{prompt}"


def _internvl_expand_image_tokens(text: str, num_patches_list: list[int], num_image_token: int) -> str:
    for num_patches in num_patches_list:
        image_tokens = "<img>" + "<IMG_CONTEXT>" * (num_image_token * num_patches) + "</img>"
        text = text.replace("<image>", image_tokens, 1)
    return text


class QwenTaskCollator:
    def __init__(self, processor, max_seq_length: int) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __call__(self, features: list[QwenTaskExample]) -> dict[str, Any]:
        batch_images = [_open_images(feature.image_paths) for feature in features]
        try:
            if _is_internvl_tokenizer(self.processor):
                import torch

                full_texts: list[str] = []
                prompt_texts: list[str] = []
                pixel_batches: list[torch.Tensor] = []
                image_flags: list[list[int]] = []
                num_image_token = int(getattr(self.processor, "_kabr_internvl_num_image_token"))
                for feature in features:
                    question = _internvl_user_question(feature.prompt, feature.image_paths)
                    full_text = self.processor.apply_chat_template(
                        [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": feature.answer},
                        ],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    prompt_text = self.processor.apply_chat_template(
                        [{"role": "user", "content": question}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    num_patches_list: list[int] = []
                    for image_path in feature.image_paths:
                        pixel_values = _internvl_load_image(image_path, max_num=1)
                        num_patches_list.append(pixel_values.shape[0])
                        pixel_batches.append(pixel_values)
                        image_flags.extend([[1]] * pixel_values.shape[0])
                    full_texts.append(_internvl_expand_image_tokens(full_text, num_patches_list, num_image_token))
                    prompt_texts.append(_internvl_expand_image_tokens(prompt_text, num_patches_list, num_image_token))

                full_batch = self.processor(
                    full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                )
                prompt_batch = self.processor(
                    prompt_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                )
                full_batch["pixel_values"] = torch.cat(pixel_batches, dim=0)
                full_batch["image_flags"] = torch.tensor(image_flags, dtype=torch.long)
            elif _is_mobilevlm_processor(self.processor):
                import torch
                from mobilevlm.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
                from mobilevlm.conversation import conv_templates
                from mobilevlm.utils import process_images, tokenizer_image_token

                if len(features) != 1:
                    raise ValueError("MobileVLM training is currently supported only with batch_size=1 in this collator.")

                input_ids_rows: list[torch.Tensor] = []
                prompt_id_rows: list[torch.Tensor] = []
                label_rows: list[torch.Tensor] = []
                image_tensors: list[torch.Tensor] = []
                pad_token_id = int(self.processor.tokenizer.pad_token_id)

                for feature, images in zip(features, batch_images):
                    image_tokens = "\n".join(DEFAULT_IMAGE_TOKEN for _ in feature.image_paths)
                    user_prompt = f"{image_tokens}\n{feature.prompt}" if image_tokens else feature.prompt

                    full_conv = conv_templates[self.processor.conv_mode].copy()
                    full_conv.append_message(full_conv.roles[0], user_prompt)
                    full_conv.append_message(full_conv.roles[1], feature.answer)
                    full_text = full_conv.get_prompt()

                    prompt_conv = conv_templates[self.processor.conv_mode].copy()
                    prompt_conv.append_message(prompt_conv.roles[0], user_prompt)
                    prompt_conv.append_message(prompt_conv.roles[1], None)
                    prompt_text = prompt_conv.get_prompt()

                    full_ids = tokenizer_image_token(
                        full_text,
                        self.processor.tokenizer,
                        IMAGE_TOKEN_INDEX,
                        return_tensors="pt",
                    )[: self.max_seq_length]
                    prompt_ids = tokenizer_image_token(
                        prompt_text,
                        self.processor.tokenizer,
                        IMAGE_TOKEN_INDEX,
                        return_tensors="pt",
                    )[: self.max_seq_length]

                    labels = full_ids.clone()
                    labels[: prompt_ids.shape[0]] = -100
                    input_ids_rows.append(full_ids)
                    prompt_id_rows.append(prompt_ids)
                    label_rows.append(labels)
                    image_tensors.append(process_images(images, self.processor.image_processor, self.processor.model_config))

                max_len = max(row.shape[0] for row in input_ids_rows)
                batch_size = len(input_ids_rows)
                input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
                attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
                prompt_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
                labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
                for row_index, (row_ids, row_prompt_ids, row_labels) in enumerate(zip(input_ids_rows, prompt_id_rows, label_rows)):
                    row_len = row_ids.shape[0]
                    input_ids[row_index, :row_len] = row_ids
                    attention_mask[row_index, :row_len] = 1
                    prompt_attention_mask[row_index, : row_prompt_ids.shape[0]] = 1
                    labels[row_index, :row_len] = row_labels

                full_batch = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "images": image_tensors[0],
                }
                prompt_batch = {"attention_mask": prompt_attention_mask}
            elif _is_phi3_processor(self.processor):
                if len(features) != 1:
                    raise ValueError("Phi3VProcessor is currently only supported with batch_size=1 in this collator.")
                feature = features[0]
                full_text = _phi3_prompt_text(feature.prompt, len(feature.image_paths), answer=feature.answer)
                prompt_text = _phi3_prompt_text(feature.prompt, len(feature.image_paths), add_generation_prompt=True)
                full_batch = self.processor(
                    text=full_text,
                    images=batch_images[0],
                    return_tensors="pt",
                    padding=True,
                )
                prompt_batch = self.processor(
                    text=prompt_text,
                    images=batch_images[0],
                    return_tensors="pt",
                    padding=True,
                )
            else:
                full_texts = [
                    _render_messages_for_processor(
                        self.processor,
                        build_task_messages(feature.prompt, feature.image_paths, answer=feature.answer),
                        add_generation_prompt=False,
                    )
                    for feature in features
                ]
                prompt_texts = [
                    _render_messages_for_processor(
                        self.processor,
                        build_task_messages(feature.prompt, feature.image_paths),
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
        full_batch.pop("mm_token_type_ids", None)
        full_batch["labels"] = labels
        return full_batch


def _hf_auth_kwargs() -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return {}
    return {"token": token}


def _is_qwen_vl_model(model_id: str) -> bool:
    return model_id.startswith("Qwen/")


def _is_phi3_vision_model(model_id: str) -> bool:
    return model_id.startswith("microsoft/Phi-3.5-vision-")


def _is_internvl_model(model_id: str) -> bool:
    return model_id.startswith("OpenGVLab/InternVL2_5-")


def _is_llava_onevision_15_model(model_id: str) -> bool:
    return "LLaVA-OneVision-1.5-" in model_id


def _is_mobilevlm_model(model_id: str) -> bool:
    return model_id.startswith("mtgv/MobileVLM")


def _validate_attention_experiment(model_id: str, attention_experiment: str | None) -> None:
    if attention_experiment in {None, "", "baseline"}:
        return
    if attention_experiment != "mla":
        raise ValueError(
            f"Unsupported attention experiment '{attention_experiment}'. "
            "Supported values are: baseline, mla."
        )
    if not _is_qwen_vl_model(model_id):
        raise ValueError(
            "The MLA attention experiment is currently implemented only for Qwen2.5-VL models."
        )


def _flash_attention_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def _preferred_attn_implementation() -> str:
    return "flash_attention_2" if _flash_attention_available() else "eager"


def _phi3_load_kwargs() -> dict[str, str]:
    return {"attn_implementation": _preferred_attn_implementation()}


def _patch_transformers_cache_utils_compat() -> None:
    import transformers.cache_utils as cache_utils

    if hasattr(cache_utils, "SlidingWindowCache"):
        return

    class SlidingWindowCache(cache_utils.StaticCache):
        pass

    cache_utils.SlidingWindowCache = SlidingWindowCache


def _patch_transformers_rope_utils_compat() -> None:
    import transformers.modeling_rope_utils as rope_utils

    if "default" in rope_utils.ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None):
        if config is None:
            raise ValueError("config is required for default RoPE initialization")
        if hasattr(config, "standardize_rope_params"):
            config.standardize_rope_params()
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        if layer_type is not None and isinstance(rope_parameters, dict):
            rope_parameters = rope_parameters.get(layer_type, {})
        base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base
            ** (rope_utils.torch.arange(0, dim, 2, dtype=rope_utils.torch.int64).to(device=device, dtype=rope_utils.torch.float) / dim)
        )
        return inv_freq, 1.0

    rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _llava_onevision_15_config(model_id: str, processor, **auth_kwargs):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, **auth_kwargs)
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    for target in (cfg, getattr(cfg, "text_config", None)):
        if target is None:
            continue
        if getattr(target, "pad_token_id", None) is None:
            setattr(target, "pad_token_id", pad_token_id)
        if getattr(target, "eos_token_id", None) is None:
            setattr(target, "eos_token_id", eos_token_id)
        if getattr(target, "bos_token_id", None) is None and bos_token_id is not None:
            setattr(target, "bos_token_id", bos_token_id)
    return cfg


def _phi3_config(model_id: str, **auth_kwargs):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        **auth_kwargs,
    )
    attn_implementation = _preferred_attn_implementation()
    setattr(cfg, "_attn_implementation", attn_implementation)
    setattr(cfg, "_attn_implementation_internal", attn_implementation)
    if hasattr(cfg, "attn_implementation"):
        setattr(cfg, "attn_implementation", attn_implementation)
    if hasattr(cfg, "use_flash_attention_2"):
        setattr(cfg, "use_flash_attention_2", attn_implementation == "flash_attention_2")
    return cfg


@contextmanager
def _force_torch_cpu_constructors(torch_module):
    original_linspace = torch_module.linspace

    def _cpu_linspace(*args, **kwargs):
        kwargs.setdefault("device", "cpu")
        return original_linspace(*args, **kwargs)

    torch_module.linspace = _cpu_linspace
    try:
        yield
    finally:
        torch_module.linspace = original_linspace


def _patch_transformers_tied_weights_compat():
    from transformers import modeling_utils
    from transformers.modeling_utils import PreTrainedModel

    if getattr(PreTrainedModel, "_kabr_tied_weights_compat_patched", False):
        return

    original = PreTrainedModel.mark_tied_weights_as_initialized
    original_get_tied_weight_keys = modeling_utils._get_tied_weight_keys

    def _wrapped(self, loading_info):
        if not hasattr(self, "all_tied_weights_keys"):
            try:
                self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)
            except Exception:
                self.all_tied_weights_keys = {}
        return original(self, loading_info)

    def _get_tied_weight_keys_compat(module):
        tied_weight_keys: list[str] = []
        for name, submodule in module.named_modules():
            tied = getattr(submodule, "_tied_weights_keys", {}) or {}
            if isinstance(tied, list):
                keys = tied
            else:
                keys = tied.keys()
            tied_weight_keys.extend([f"{name}.{k}" if name else k for k in keys])
        return tied_weight_keys

    PreTrainedModel.mark_tied_weights_as_initialized = _wrapped
    modeling_utils._get_tied_weight_keys = _get_tied_weight_keys_compat
    modeling_utils._kabr_original_get_tied_weight_keys = original_get_tied_weight_keys
    PreTrainedModel._kabr_tied_weights_compat_patched = True


def _ensure_generation_config(model) -> None:
    try:
        from transformers import GenerationConfig
    except ImportError:
        return

    if hasattr(model, "generation_config") and getattr(model, "generation_config") is not None:
        return
    config = getattr(model, "config", None)
    if config is None:
        return
    model.generation_config = GenerationConfig.from_model_config(config)


def _ensure_generation_mixin(model) -> None:
    try:
        from transformers.generation.utils import GenerationMixin
    except ImportError:
        return

    cls = model.__class__
    if hasattr(cls, "generate"):
        return

    for name, value in GenerationMixin.__dict__.items():
        if name.startswith("__"):
            continue
        if hasattr(cls, name):
            continue
        setattr(cls, name, value)


def _is_internlm2_language_model(model) -> bool:
    cls_name = model.__class__.__name__
    if "InternLM2" in cls_name:
        return True
    config = getattr(model, "config", None)
    architectures = getattr(config, "architectures", None)
    if architectures:
        return any("InternLM2" in arch for arch in architectures)
    return False


def _patch_internlm2_prepare_inputs_for_generation(model) -> None:
    if not _is_internlm2_language_model(model):
        return
    if getattr(model, "_kabr_prepare_inputs_compat_patched", False):
        return

    def _prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_length = past_key_values.get_seq_length()
            elif hasattr(past_key_values, "seen_tokens"):
                past_length = past_key_values.seen_tokens
            else:
                past_length = past_key_values[0][0].shape[2]

            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    model.prepare_inputs_for_generation = MethodType(_prepare_inputs_for_generation, model)
    model._kabr_prepare_inputs_compat_patched = True


def _normalize_legacy_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0:
        return None
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    if hasattr(past_key_values, "layers"):
        legacy = []
        for layer in past_key_values:
            if isinstance(layer, tuple):
                keys, values = layer[:2]
                if keys is None or values is None:
                    return None
                legacy.append((keys, values))
            else:
                legacy.append(layer)
        return tuple(legacy)
    return past_key_values


def _patch_internlm2_forward_compat(language_model) -> None:
    if not _is_internlm2_language_model(language_model):
        return
    if getattr(language_model, "_kabr_forward_compat_patched", False):
        return

    original_forward = language_model.forward

    def _forward(self, *args, **kwargs):
        if "past_key_values" in kwargs:
            kwargs["past_key_values"] = _normalize_legacy_past_key_values(kwargs["past_key_values"])
        return original_forward(*args, **kwargs)

    language_model.forward = MethodType(_forward, language_model)

    inner_model = getattr(language_model, "model", None)
    if inner_model is not None and not getattr(inner_model, "_kabr_forward_compat_patched", False):
        original_inner_forward = inner_model.forward

        def _inner_forward(self, *args, **kwargs):
            if "past_key_values" in kwargs:
                kwargs["past_key_values"] = _normalize_legacy_past_key_values(kwargs["past_key_values"])
            return original_inner_forward(*args, **kwargs)

        inner_model.forward = MethodType(_inner_forward, inner_model)
        inner_model._kabr_forward_compat_patched = True

    language_model._kabr_forward_compat_patched = True


def _patch_internvl_generate_compat(model) -> None:
    if getattr(model, "_kabr_generate_compat_patched", False):
        return

    def _generate(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        visual_features=None,
        generation_config=None,
        output_hidden_states=None,
        **generate_kwargs,
    ):
        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            bsz, seq_len, hidden = input_embeds.shape
            input_embeds = input_embeds.reshape(bsz * seq_len, hidden)

            flat_input_ids = input_ids.reshape(bsz * seq_len)
            selected = flat_input_ids == self.img_context_token_id
            if selected.sum() == 0:
                raise ValueError("InternVL generate expected at least one image context token.")
            input_embeds[selected] = vit_embeds.reshape(-1, hidden).to(input_embeds.device)
            input_embeds = input_embeds.reshape(bsz, seq_len, hidden)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        try:
            from copy import deepcopy
        except ImportError:
            deepcopy = None

        base_generation_config = generation_config
        if base_generation_config is None:
            base_generation_config = getattr(self.language_model, "generation_config", None)
        if deepcopy is not None and base_generation_config is not None:
            base_generation_config = deepcopy(base_generation_config)

        if base_generation_config is not None:
            for key in ("max_new_tokens", "do_sample", "eos_token_id", "bos_token_id", "pad_token_id"):
                if key in generate_kwargs:
                    setattr(base_generation_config, key, generate_kwargs.pop(key))
            setattr(base_generation_config, "use_cache", False)

        generate_kwargs.pop("use_cache", None)
        return self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=base_generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=False,
            **generate_kwargs,
        )

    model.generate = MethodType(_generate, model)
    model._kabr_generate_compat_patched = True


def _load_training_model_and_processor(config: AppConfig):
    try:
        import torch
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
            AutoProcessor,
            AutoConfig,
            Qwen2_5_VLForConditionalGeneration,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Fine-tuning requires `transformers` and `torch`. Install with `uv pip install -e \".[train]\"`."
        ) from exc

    bf16, fp16 = choose_precision_flags(config)
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    _validate_attention_experiment(config.model.base_model_id, config.model.attention_experiment)
    auth_kwargs = _hf_auth_kwargs()
    if _is_internvl_model(config.model.base_model_id):
        processor = AutoTokenizer.from_pretrained(
            config.model.resolved_processor_id,
            trust_remote_code=True,
            use_fast=False,
            **auth_kwargs,
        )
    elif _is_mobilevlm_model(config.model.base_model_id):
        _ensure_mobilevlm_upstream()
        _patch_mobilevlm_vision_tower_delay_load()
        try:
            from mobilevlm.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
            from mobilevlm.model.mobilellama import MobileLlamaForCausalLM
            from transformers import LlamaTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "MobileVLM fine-tuning requires the upstream MobileVLM code and Llama tokenizer support."
            ) from exc

        if _has_google_protobuf():
            tokenizer = LlamaTokenizer.from_pretrained(
                config.model.resolved_processor_id,
                use_fast=False,
                legacy=True,
                **auth_kwargs,
            )
        else:
            snapshot_path = _mobilevlm_snapshot_path(config.model.resolved_processor_id, **auth_kwargs)
            tokenizer = MobileVLMSentencePieceTokenizer(
                snapshot_path / "tokenizer.model",
                add_bos_token=True,
                add_eos_token=False,
                bos_token="<s>",
                eos_token="</s>",
                unk_token="<unk>",
                pad_token="<unk>",
            )
        model = MobileLlamaForCausalLM.from_pretrained(
            config.model.base_model_id,
            low_cpu_mem_usage=False,
            torch_dtype=dtype,
            device_map=None,
            **auth_kwargs,
        )

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded or getattr(vision_tower, "vision_tower", None) is None:
            vision_tower.load_model()
        elif "v2" in getattr(model.config, "mm_projector_type", "ldpnet"):
            vision_tower.load_image_processor()
        processor = MobileVLMProcessorBundle(
            tokenizer=tokenizer,
            image_processor=vision_tower.image_processor,
            model_config=model.config,
        )
    else:
        processor = AutoProcessor.from_pretrained(
            config.model.resolved_processor_id,
            trust_remote_code=True,
            **auth_kwargs,
        )
    if _is_mobilevlm_model(config.model.base_model_id):
        pass
    elif _is_qwen_vl_model(config.model.base_model_id):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model.base_model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            **auth_kwargs,
        )
        if config.model.attention_experiment == "mla":
            model = apply_qwen25vl_mla_rewrite(model)
    elif _is_internvl_model(config.model.base_model_id):
        _patch_transformers_tied_weights_compat()
        # InternVL remote model init touches tensor constructors like
        # torch.linspace(...).item() during __init__. In some HF load paths a
        # higher-level device mode still redirects constructor calls to `meta`,
        # so we temporarily pin the constructor we hit here to CPU.
        with _force_torch_cpu_constructors(torch):
            model = AutoModel.from_pretrained(
                config.model.base_model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                device_map=None,
                _fast_init=False,
                use_flash_attn=_flash_attention_available(),
                **auth_kwargs,
            )
        _patch_internlm2_prepare_inputs_for_generation(model.language_model)
        _patch_internlm2_forward_compat(model.language_model)
        _patch_internvl_generate_compat(model)
        _ensure_generation_mixin(model.language_model)
        _ensure_generation_config(model.language_model)
        setattr(processor, "_kabr_internvl_num_image_token", int(model.num_image_token))
    elif _is_phi3_vision_model(config.model.base_model_id):
        phi_cfg = _phi3_config(config.model.base_model_id, **auth_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            config.model.base_model_id,
            config=phi_cfg,
            torch_dtype=dtype,
            trust_remote_code=True,
            **_phi3_load_kwargs(),
            **auth_kwargs,
        )
    elif _is_llava_onevision_15_model(config.model.base_model_id):
        _patch_transformers_cache_utils_compat()
        _patch_transformers_rope_utils_compat()
        llava_cfg = _llava_onevision_15_config(config.model.base_model_id, processor, **auth_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            config.model.base_model_id,
            config=llava_cfg,
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
        if _is_internvl_model(config.model.base_model_id):
            model.language_model.gradient_checkpointing_enable()
            if hasattr(model.language_model, "enable_input_require_grads"):
                model.language_model.enable_input_require_grads()
            model.language_model.config.use_cache = False
        elif _is_mobilevlm_model(config.model.base_model_id):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            model.config.use_cache = False
        else:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model.config.use_cache = False

    return model, processor


def create_training_model(config: AppConfig):
    model, processor = _load_training_model_and_processor(config)
    if config.training.method == "full":
        return model, processor
    if config.training.method != "lora":
        raise ValueError(f"Unsupported training.method={config.training.method!r}. Expected 'lora' or 'full'.")

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "LoRA fine-tuning requires `peft`. Install with `uv pip install -e \".[train]\"`."
        ) from exc

    if _is_internvl_model(config.model.base_model_id):
        target_modules = INTERNVL_LORA_TARGET_MODULES
    elif _is_qwen_vl_model(config.model.base_model_id) and config.model.attention_experiment == "mla":
        target_modules = QWEN_MLA_LORA_TARGET_MODULES
    else:
        target_modules = DEFAULT_LORA_TARGET_MODULES
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=config.training.lora_r,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    if _is_internvl_model(config.model.base_model_id):
        model.language_model = get_peft_model(model.language_model, lora_config)
        _ensure_generation_mixin(model.language_model)
        _ensure_generation_config(model.language_model)
    else:
        model = get_peft_model(model, lora_config)
    return model, processor


def create_lora_model(config: AppConfig):
    if config.training.method != "lora":
        raise ValueError("create_lora_model requires training.method='lora'.")
    return create_training_model(config)


def train_hf_task_model(config: AppConfig, resume_from_checkpoint: str | None = None) -> dict[str, Any]:
    try:
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "Fine-tuning requires `transformers`. Install with `uv pip install -e \".[train]\"`."
        ) from exc

    train_examples = load_jsonl_examples(config.data.train_file, config.data.max_train_samples)
    val_examples = load_jsonl_examples(config.data.val_file, config.data.max_val_samples)
    model, processor = create_training_model(config)

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
        save_steps=config.training.save_steps,
        logging_steps=config.training.logging_steps,
        save_total_limit=config.training.save_total_limit,
        remove_unused_columns=False,
        report_to="none",
        seed=config.training.seed,
        dataloader_num_workers=0,
    )
    trainer_cls = Trainer
    trainer_kwargs: dict[str, Any] = {}

    sampler_mode = getattr(config.training, "class_sampler", "uniform")
    weighting_mode = getattr(config.training, "class_weighting", "none")

    if sampler_mode == "class_balanced" or weighting_mode != "none":
        import math
        import torch
        from collections import Counter

        labels = [ex.label for ex in train_dataset]
        counts = Counter(labels)
        unique_labels = sorted(counts)
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        num_classes = len(unique_labels)
        n_total = sum(counts.values())
        max_count = max(counts.values())

        if weighting_mode == "balanced":
            class_weight_vec = torch.tensor(
                [n_total / (num_classes * counts[lbl]) for lbl in unique_labels],
                dtype=torch.float32,
            )
        elif weighting_mode == "inv_sqrt":
            class_weight_vec = torch.tensor(
                [math.sqrt(max_count / counts[lbl]) for lbl in unique_labels],
                dtype=torch.float32,
            )
        else:
            class_weight_vec = None

        per_example_weight = torch.tensor(
            [
                class_weight_vec[label_to_idx[lbl]].item()
                if class_weight_vec is not None
                else 1.0
                for lbl in labels
            ],
            dtype=torch.float32,
        )

        focal_gamma = (
            float(getattr(config.training, "focal_gamma", 2.0))
            if weighting_mode == "focal"
            else None
        )

        class _ClassAwareTrainer(Trainer):  # type: ignore[misc]
            _cb_per_example_weight = per_example_weight
            _cb_label_index = [label_to_idx[lbl] for lbl in labels]
            _cb_sampler_mode = sampler_mode
            _cb_focal_gamma = focal_gamma

            def _get_train_sampler(self, train_dataset=None):  # type: ignore[override]
                if self._cb_sampler_mode != "class_balanced":
                    return super()._get_train_sampler(train_dataset)
                g = torch.Generator().manual_seed(int(self.args.seed))
                ds_len = len(self.train_dataset)
                # Weighted sampler: per-example inverse frequency, with replacement.
                sampler_weights = torch.tensor(
                    [
                        1.0 / counts[lbl]
                        for lbl in labels[:ds_len]
                    ],
                    dtype=torch.double,
                )
                return torch.utils.data.WeightedRandomSampler(
                    weights=sampler_weights,
                    num_samples=ds_len,
                    replacement=True,
                    generator=g,
                )

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
                outputs = model(**inputs)
                loss = outputs.loss
                if self._cb_focal_gamma is not None and loss is not None:
                    # Focal scaling on the example's aggregate loss:
                    #   focal_loss = (1 - exp(-loss))^gamma * loss.
                    # This is a scalar (batch-mean) approximation since the trainer
                    # already reduces per-example. Reduces impact of easy batches.
                    import torch as _t
                    with _t.no_grad():
                        p_t = _t.exp(-loss.detach())
                        scale = (1.0 - p_t) ** self._cb_focal_gamma
                    loss = scale * loss
                return (loss, outputs) if return_outputs else loss

        trainer_cls = _ClassAwareTrainer
        print(
            f"class-aware trainer: sampler={sampler_mode}  weighting={weighting_mode}"
            f"  classes={num_classes}  counts={dict(counts)}",
            flush=True,
        )

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        **trainer_kwargs,
    )
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if _is_internvl_model(config.model.base_model_id) and config.training.method == "lora":
        model.language_model.save_pretrained(str(config.training.output_dir))
    else:
        trainer.save_model(str(config.training.output_dir))
    _safe_save_processor(processor, config.training.output_dir)
    metrics = {
        "train_runtime": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "task_mode": config.data.task_mode,
        "output_format": config.model.output_format,
        "attention_experiment": config.model.attention_experiment,
        "method": config.training.method,
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
    attention_experiment: str | None = None,
) -> dict[str, Any]:
    from kabr_vlm.data.kabr_frames import group_frame_records_into_clips, read_charades_frame_annotations
    from kabr_vlm.metrics import load_prediction_records
    from kabr_vlm.vlm_baselines import load_hf_adapter_generator

    rows = read_charades_frame_annotations(annotation_path, image_root, classes_path, dataset=dataset)
    generator = load_hf_adapter_generator(
        model_id=model_id,
        adapter_path=adapter_path,
        attention_experiment=attention_experiment,
    )

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


def evaluate_hf_full_model(
    annotation_path: str | Path,
    image_root: str | Path,
    classes_path: str | Path,
    model_id: str,
    checkpoint_path: str | Path,
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
    from kabr_vlm.vlm_baselines import load_hf_full_generator

    rows = read_charades_frame_annotations(annotation_path, image_root, classes_path, dataset=dataset)
    generator = load_hf_full_generator(model_id=model_id, checkpoint_path=checkpoint_path)

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
    attention_experiment: str | None = None,
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
        attention_experiment=attention_experiment,
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
