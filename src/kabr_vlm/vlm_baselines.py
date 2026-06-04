import json
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from contextlib import ExitStack
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any

from PIL import Image

try:
    from PIL import UnidentifiedImageError
except ImportError:  # Pillow < 7 fallback
    UnidentifiedImageError = OSError

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
from kabr_vlm.experimental.mla import apply_qwen25vl_mla_rewrite

BEHAVIOR_LABELS = ["Walk", "Graze", "Browse", "Head Up", "Auto-Groom", "Trot", "Run", "Occluded"]
SUPPORTED_TASKS = {"behavior", "species", "joint"}
SUPPORTED_PROMPT_MODES = {"plain", "structured", "label"}
INTERNVL_IMAGE_SIZE = 448
INTERNVL_MAX_TILES_PER_FRAME = 1


def _is_qwen_vl_model(model_id: str) -> bool:
    return model_id.startswith("Qwen/")


def _is_qwen25_vl_model(model_id: str) -> bool:
    return model_id.startswith("Qwen/Qwen2.5-VL-")


def _is_internvl_model(model_id: str) -> bool:
    return model_id.startswith("OpenGVLab/InternVL2_5-")


def _is_phi3_vision_model(model_id: str) -> bool:
    return model_id.startswith("microsoft/Phi-3.5-vision-")


def _is_mobilevlm_model(model_id: str) -> bool:
    return model_id.startswith("mtgv/MobileVLM")


def _is_llava_next_video_model(model_id: str) -> bool:
    return model_id.startswith("llava-hf/LLaVA-NeXT-Video-")


def _is_minicpm_v46_model(model_id: str) -> bool:
    return model_id.startswith("openbmb/MiniCPM-V-4.6")


def _flash_attention_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def _preferred_attn_implementation() -> str:
    return "flash_attention_2" if _flash_attention_available() else "eager"


def _preferred_video_attn_implementation() -> str:
    if _flash_attention_available():
        return "flash_attention_2"
    try:
        import torch
    except ImportError:
        return "eager"
    if torch.cuda.is_available():
        return "sdpa"
    return "eager"


def _uniform_subsample_sequence(items: list[Any], max_items: int) -> list[Any]:
    if max_items <= 0:
        raise ValueError("max_items must be >= 1")
    if len(items) <= max_items:
        return list(items)
    if max_items == 1:
        return [items[len(items) // 2]]

    step = (len(items) - 1) / float(max_items - 1)
    indices: list[int] = []
    for idx in range(max_items):
        candidate = int(round(idx * step))
        candidate = min(candidate, len(items) - 1)
        if indices and candidate <= indices[-1]:
            candidate = min(indices[-1] + 1, len(items) - 1)
        indices.append(candidate)
    return [items[index] for index in indices]


def _is_llava_onevision_15_model(model_id: str) -> bool:
    return "LLaVA-OneVision-1.5-" in model_id


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
            "The MLA attention-swap pilot is currently implemented only for Qwen2.5-VL models."
        )


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


def filter_readable_image_paths(image_paths: list[str]) -> tuple[list[str], list[str]]:
    readable: list[str] = []
    skipped: list[str] = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                image.verify()
            readable.append(path)
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            skipped.append(path)
    return readable, skipped


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


def _is_phi3_processor(processor) -> bool:
    return processor.__class__.__name__ == "Phi3VProcessor"


def _phi3_prompt_text(prompt: str, image_count: int, add_generation_prompt: bool) -> str:
    image_tokens = "\n".join(f"<|image_{idx}|>" for idx in range(1, image_count + 1))
    text = f"{image_tokens}\nUSER: {prompt}"
    if add_generation_prompt:
        text += "\nASSISTANT:"
    return text


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


def load_hf_pipeline(model_id: str, attention_experiment: str | None = None):
    _validate_attention_experiment(model_id, attention_experiment)
    if _is_internvl_model(model_id):
        return load_hf_internvl_generator(model_id)
    if _is_mobilevlm_model(model_id):
        return load_hf_mobilevlm_generator(model_id)
    if _is_llava_next_video_model(model_id):
        return load_hf_llava_next_video_generator(model_id)
    if _is_minicpm_v46_model(model_id):
        return load_hf_minicpm_v46_generator(model_id)
    if _is_qwen25_vl_model(model_id):
        return load_hf_qwen_generator(
            model_id,
            attention_experiment=attention_experiment,
        )

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


def load_hf_minicpm_v46_generator(model_id: str):
    try:
        import torch
        import transformers
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "MiniCPM-V 4.6 inference requires `transformers` and `torch`. "
            "Install with `uv pip install -e \".[infer]\"`."
        ) from exc

    model_cls = getattr(transformers, "MiniCPMV4_6ForConditionalGeneration", None)
    if model_cls is None:
        raise RuntimeError(
            "This Transformers install does not expose MiniCPMV4_6ForConditionalGeneration. "
            "Upgrade Transformers to a build with MiniCPM-V 4.6 support before evaluating "
            f"{model_id}."
        )

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = model_cls.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        **_hf_auth_kwargs(),
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        **_hf_auth_kwargs(),
    )
    return HFChatGenerator(model=model, processor=processor)


def load_hf_llava_next_video_generator(model_id: str):
    try:
        import torch
        from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "LLaVA-NeXT-Video inference requires `transformers` and `torch`. "
            "Install with `uv pip install -e \".[infer]\"`."
        ) from exc

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation=_preferred_video_attn_implementation(),
        **_hf_auth_kwargs(),
    )
    processor = LlavaNextVideoProcessor.from_pretrained(
        model_id,
        use_fast=False,
        trust_remote_code=True,
        **_hf_auth_kwargs(),
    )
    return LlavaNextVideoGenerator(model=model, processor=processor)


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
    _patch_transformers_tied_weights_compat()
    with _force_torch_cpu_constructors(torch):
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            device_map=None,
            trust_remote_code=True,
            _fast_init=False,
            use_flash_attn=False,
            **_hf_auth_kwargs(),
        ).eval()
    _patch_internlm2_prepare_inputs_for_generation(model.language_model)
    _patch_internlm2_forward_compat(model.language_model)
    _patch_internvl_generate_compat(model)
    _ensure_generation_mixin(model.language_model)
    _ensure_generation_config(model.language_model)
    if torch.cuda.is_available():
        model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
        **_hf_auth_kwargs(),
    )
    return InternVLGenerator(model=model, tokenizer=tokenizer, torch_dtype=torch_dtype)


def load_hf_mobilevlm_generator(model_id: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "MobileVLM inference requires `torch` and `transformers`. Install with `uv pip install -e \".[infer]\"`."
        ) from exc

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model, tokenizer, image_processor = load_hf_mobilevlm_components(
        model_id=model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        eval_mode=True,
    )
    return MobileVLMGenerator(model=model, tokenizer=tokenizer, image_processor=image_processor)


def load_hf_mobilevlm_components(
    model_id: str | Path,
    torch_dtype,
    *,
    tokenizer_source: str | Path | None = None,
    adapter_path: str | Path | None = None,
    device_map=None,
    eval_mode: bool = False,
):
    try:
        from peft import PeftModel
        from transformers import LlamaTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "MobileVLM loading requires `transformers`, `torch`, and `peft`. "
            "Install with `uv pip install -e \".[infer,train]\"`."
        ) from exc

    _ensure_mobilevlm_upstream()
    _patch_mobilevlm_vision_tower_delay_load()
    try:
        from mobilevlm.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
        from mobilevlm.model.mobilellama import MobileLlamaForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import the upstream MobileVLM code after fetching it."
        ) from exc

    auth_kwargs = _hf_auth_kwargs()
    tokenizer_source = tokenizer_source or model_id
    if _has_google_protobuf():
        tokenizer = LlamaTokenizer.from_pretrained(
            str(tokenizer_source),
            use_fast=False,
            legacy=True,
            **auth_kwargs,
        )
    else:
        tokenizer_path = Path(str(tokenizer_source))
        if tokenizer_path.exists() and (tokenizer_path / "tokenizer.model").exists():
            snapshot_path = tokenizer_path
        else:
            snapshot_path = _mobilevlm_snapshot_path(str(tokenizer_source), **auth_kwargs)
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
        str(model_id),
        low_cpu_mem_usage=False,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **auth_kwargs,
    )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))

    vision_model_owner = model.get_base_model() if hasattr(model, "get_base_model") else model
    vision_tower = vision_model_owner.get_vision_tower()
    if not vision_tower.is_loaded or getattr(vision_tower, "vision_tower", None) is None:
        vision_tower.load_model()
    elif "v2" in getattr(vision_model_owner.config, "mm_projector_type", "ldpnet"):
        vision_tower.load_image_processor()

    try:
        model_device = next(model.parameters()).device
        vision_tower.to(device=model_device, dtype=torch_dtype)
    except (AttributeError, StopIteration):
        pass

    image_processor = vision_tower.image_processor
    if eval_mode and hasattr(model, "eval"):
        model = model.eval()
    return model, tokenizer, image_processor


def _load_hf_generator_from_source(
    model_id: str,
    *,
    model_source: str | Path | None = None,
    adapter_path: str | Path | None = None,
    attention_experiment: str | None = None,
):
    _validate_attention_experiment(model_id, attention_experiment)
    try:
        import torch
        from peft import PeftModel
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
            "Adapter inference requires `transformers`, `torch`, and `peft`. "
            "Install with `uv pip install -e \".[infer,train]\"`."
        ) from exc

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    auth_kwargs = _hf_auth_kwargs()
    model_source = str(model_source or model_id)
    processor_source = model_source
    processor = None
    if _is_internvl_model(model_id):
        _patch_transformers_tied_weights_compat()
        # InternVL remote init calls torch.linspace(...).item() during model
        # construction. Some HF load paths still redirect constructors to
        # `meta`, so pin the constructor we hit here to CPU.
        with _force_torch_cpu_constructors(torch):
            model = AutoModel.from_pretrained(
                model_source,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=False,
                device_map=None,
                trust_remote_code=True,
                use_flash_attn=False,
                **auth_kwargs,
            ).eval()
        _patch_internlm2_prepare_inputs_for_generation(model.language_model)
        _patch_internlm2_forward_compat(model.language_model)
        _patch_internvl_generate_compat(model)
        _ensure_generation_mixin(model.language_model)
        _ensure_generation_config(model.language_model)
        if adapter_path is not None:
            model.language_model = PeftModel.from_pretrained(model.language_model, str(adapter_path))
            _ensure_generation_mixin(model.language_model)
            _ensure_generation_config(model.language_model)
        if torch.cuda.is_available():
            model = model.cuda()
        tokenizer = AutoTokenizer.from_pretrained(
            processor_source,
            trust_remote_code=True,
            use_fast=False,
            **auth_kwargs,
        )
        return InternVLGenerator(model=model, tokenizer=tokenizer, torch_dtype=torch_dtype)
    if _is_mobilevlm_model(model_id):
        model, tokenizer, image_processor = load_hf_mobilevlm_components(
            model_id=model_source,
            torch_dtype=torch_dtype,
            tokenizer_source=processor_source,
            adapter_path=adapter_path,
            device_map="auto" if torch.cuda.is_available() else None,
            eval_mode=True,
        )
        return MobileVLMGenerator(model=model, tokenizer=tokenizer, image_processor=image_processor)
    if _is_minicpm_v46_model(model_id):
        if adapter_path is not None:
            raise RuntimeError("MiniCPM-V 4.6 adapter loading is not implemented for this eval path.")
        return load_hf_minicpm_v46_generator(model_source)
    if _is_qwen25_vl_model(model_id):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            **auth_kwargs,
        )
        if attention_experiment == "mla":
            model = apply_qwen25vl_mla_rewrite(model)
    elif _is_phi3_vision_model(model_id):
        phi_cfg = _phi3_config(model_id, **auth_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            config=phi_cfg,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            **_phi3_load_kwargs(),
            **auth_kwargs,
        )
    elif _is_llava_onevision_15_model(model_id):
        _patch_transformers_cache_utils_compat()
        _patch_transformers_rope_utils_compat()
        processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True, **auth_kwargs)
        llava_cfg = _llava_onevision_15_config(model_id, processor, **auth_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            config=llava_cfg,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            **auth_kwargs,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_source,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            **auth_kwargs,
        )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    if processor is None:
        processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True, **auth_kwargs)
    return HFChatGenerator(model=model, processor=processor)


def load_hf_adapter_generator(
    model_id: str,
    adapter_path: str | Path | None = None,
    attention_experiment: str | None = None,
):
    return _load_hf_generator_from_source(
        model_id=model_id,
        adapter_path=adapter_path,
        attention_experiment=attention_experiment,
    )


def load_hf_full_generator(model_id: str, checkpoint_path: str | Path):
    return _load_hf_generator_from_source(model_id=model_id, model_source=checkpoint_path)


def load_hf_qwen_generator(
    model_id: str,
    adapter_path: str | Path | None = None,
    attention_experiment: str | None = None,
):
    return load_hf_adapter_generator(
        model_id=model_id,
        adapter_path=adapter_path,
        attention_experiment=attention_experiment,
    )


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
            if _is_phi3_processor(self.processor):
                prompt_text = _phi3_prompt_text(prompt, len(image_paths), add_generation_prompt=True)
                encoded = self.processor(
                    text=prompt_text,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
            else:
                prompt_text = _render_messages_for_processor(
                    self.processor,
                    messages,
                    add_generation_prompt=True,
                )
                encoded = self.processor(
                    text=[prompt_text],
                    images=[images],
                    return_tensors="pt",
                    padding=True,
                )
        device = _model_device(self.model)
        encoded.pop("mm_token_type_ids", None)
        encoded = {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}
        generate_kwargs = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        if _is_phi3_processor(self.processor):
            generate_kwargs["use_cache"] = False
        with torch.inference_mode():
            generated = self.model.generate(**generate_kwargs)
        prompt_len = encoded["input_ids"].shape[1]
        completion = generated[:, prompt_len:]
        return self.processor.batch_decode(
            completion,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class LlavaNextVideoGenerator:
    def __init__(self, model, processor) -> None:
        self.model = model
        self.processor = processor
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

    def _max_input_length(self) -> int | None:
        tokenizer = getattr(self.processor, "tokenizer", None)
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            return tokenizer_limit

        for cfg in (getattr(self.model, "config", None), getattr(getattr(self.model, "config", None), "text_config", None)):
            limit = getattr(cfg, "max_position_embeddings", None)
            if isinstance(limit, int) and limit > 0:
                return limit
        return None

    def __call__(self, image_paths: list[str], prompt: str, max_new_tokens: int) -> str:
        import torch

        with ExitStack() as stack:
            frames = [stack.enter_context(Image.open(path)).convert("RGB") for path in image_paths]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "video"},
                    ],
                }
            ]
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = None
            active_frames = frames
            max_input_length = self._max_input_length()
            tokenizer = getattr(self.processor, "tokenizer", None)
            original_tokenizer_limit = getattr(tokenizer, "model_max_length", None) if tokenizer is not None else None
            if tokenizer is not None and max_input_length is not None:
                tokenizer.model_max_length = max(original_tokenizer_limit or 0, 1_000_000)
            try:
                while True:
                    encoded = self.processor(
                        text=[prompt_text],
                        videos=[active_frames],
                        padding=True,
                        return_tensors="pt",
                    )
                    if max_input_length is None or encoded["input_ids"].shape[1] <= max_input_length or len(active_frames) <= 1:
                        break
                    next_frame_count = max(1, int(len(active_frames) * max_input_length / encoded["input_ids"].shape[1]))
                    if next_frame_count >= len(active_frames):
                        next_frame_count = len(active_frames) - 1
                    active_frames = _uniform_subsample_sequence(active_frames, next_frame_count)
            finally:
                if tokenizer is not None and original_tokenizer_limit is not None:
                    tokenizer.model_max_length = original_tokenizer_limit
        device = _model_device(self.model)
        encoded = {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}
        generate_kwargs = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        with torch.inference_mode():
            generated = self.model.generate(**generate_kwargs)
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

        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False, "use_cache": False}
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


class MobileVLMGenerator:
    def __init__(self, model, tokenizer, image_processor, conv_mode: str = "v1") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = conv_mode

    def __call__(self, image_paths: list[str], prompt: str, max_new_tokens: int) -> str:
        try:
            import torch
            from mobilevlm.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
            from mobilevlm.conversation import SeparatorStyle, conv_templates
            from mobilevlm.utils import (
                KeywordsStoppingCriteria,
                process_images,
                tokenizer_image_token,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MobileVLM runtime support is unavailable. The upstream MobileVLM code must be importable."
            ) from exc

        with ExitStack() as stack:
            images = [stack.enter_context(Image.open(path)).convert("RGB") for path in image_paths]
            image_tokens = "\n".join(DEFAULT_IMAGE_TOKEN for _ in image_paths)
            full_prompt = f"{image_tokens}\n{prompt}" if image_tokens else prompt

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], full_prompt)
            conv.append_message(conv.roles[1], None)
            rendered_prompt = conv.get_prompt()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            image_tensor = process_images(images, self.image_processor, self.model.config)
            try:
                image_dtype = next(self.model.parameters()).dtype
            except (AttributeError, StopIteration):
                image_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            image_device = _model_device(self.model)
            image_tensor = image_tensor.to(device=image_device, dtype=image_dtype)

            input_ids = tokenizer_image_token(
                rendered_prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(image_device)
            stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=False,
                stopping_criteria=[stopping_criteria],
            )

        prompt_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(
            output_ids[:, prompt_len:],
            skip_special_tokens=True,
        )[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        return outputs.strip()


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


def _has_google_protobuf() -> bool:
    try:
        return importlib.util.find_spec("google.protobuf") is not None
    except ModuleNotFoundError:
        return False


class _TokenizerCallResult:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class MobileVLMSentencePieceTokenizer:
    def __init__(
        self,
        model_path: str | Path,
        *,
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str | None = None,
    ) -> None:
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise RuntimeError(
                "MobileVLM tokenizer fallback requires `sentencepiece`."
            ) from exc

        self.model_path = Path(model_path)
        self.sp_model = spm.SentencePieceProcessor(model_file=str(model_path))
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token or unk_token
        self.bos_token_id = int(self.sp_model.bos_id())
        self.eos_token_id = int(self.sp_model.eos_id())
        self.unk_token_id = int(self.sp_model.unk_id())
        pad_piece_id = self.sp_model.piece_to_id(self.pad_token)
        self.pad_token_id = int(pad_piece_id if pad_piece_id >= 0 else self.unk_token_id)

    def __len__(self) -> int:
        return int(self.sp_model.vocab_size())

    def add_tokens(self, tokens, special_tokens: bool = False) -> int:
        # This fallback is only used for MobileVLM checkpoints whose configs do
        # not require extra image tokens. Keep the interface compatible anyway.
        return 0

    def __call__(self, text: str):
        input_ids = [int(token_id) for token_id in self.sp_model.encode(text, out_type=int)]
        if self.add_bos_token:
            input_ids = [self.bos_token_id, *input_ids]
        if self.add_eos_token:
            input_ids = [*input_ids, self.eos_token_id]
        return _TokenizerCallResult(input_ids)

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> list[str]:
        decoded: list[str] = []
        special_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        for sequence in sequences:
            if hasattr(sequence, "tolist"):
                token_ids = sequence.tolist()
            else:
                token_ids = list(sequence)
            token_ids = [int(token_id) for token_id in token_ids]
            if skip_special_tokens:
                token_ids = [token_id for token_id in token_ids if token_id not in special_ids]
            decoded.append(self.sp_model.decode(token_ids))
        return decoded

    def save_pretrained(self, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.model_path, output_path / "tokenizer.model")
        (output_path / "special_tokens_map.json").write_text(
            json.dumps(
                {
                    "bos_token": self.bos_token,
                    "eos_token": self.eos_token,
                    "pad_token": self.pad_token,
                    "unk_token": self.unk_token,
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (output_path / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "add_bos_token": self.add_bos_token,
                    "add_eos_token": self.add_eos_token,
                    "bos_token": self.bos_token,
                    "eos_token": self.eos_token,
                    "unk_token": self.unk_token,
                    "pad_token": self.pad_token,
                    "legacy": True,
                    "tokenizer_class": "LlamaTokenizer",
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _mobilevlm_cache_root() -> Path:
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "kabr_vlm"
    return Path.home() / ".cache" / "kabr_vlm"


def _ensure_mobilevlm_upstream() -> Path:
    cache_root = _mobilevlm_cache_root()
    repo_root = cache_root / "mobilevlm_upstream"
    marker = repo_root / "mobilevlm" / "model" / "mobilellama.py"
    if marker.exists():
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        return repo_root

    git = shutil.which("git")
    if git is None:
        raise RuntimeError(
            "MobileVLM support requires `git` to fetch the upstream MobileVLM code."
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            git,
            "clone",
            "--depth",
            "1",
            "https://github.com/Meituan-AutoML/MobileVLM",
            str(repo_root),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _patch_mobilevlm_vision_tower_delay_load() -> None:
    from mobilevlm.model.vision_encoder import CLIPVisionTower, CLIPVisionConfig

    if getattr(CLIPVisionTower, "_kabr_delay_load_patched", False):
        return

    original_init = CLIPVisionTower.__init__

    def _patched_init(self, vision_tower, args, delay_load=False):
        if not delay_load:
            return original_init(self, vision_tower, args, delay_load=delay_load)
        super(CLIPVisionTower, self).__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        self.vision_tower = None

    CLIPVisionTower.__init__ = _patched_init
    CLIPVisionTower._kabr_delay_load_patched = True


def _mobilevlm_snapshot_path(model_id: str, **auth_kwargs) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "MobileVLM loading requires `huggingface_hub` to locate model files."
        ) from exc

    return Path(
        snapshot_download(
            repo_id=model_id,
            local_files_only=False,
            **auth_kwargs,
        )
    )


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


def run_hf_generation(
    generator,
    image_paths: list[str],
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, float]:
    """Run one generation and return (text, latency_ms).

    Latency is measured with torch.cuda.Event on GPU or perf_counter on CPU,
    covering image preprocessing + forward pass + token decode.
    """
    import time
    import torch

    def _call():
        if isinstance(generator, (HFChatGenerator, LlavaNextVideoGenerator, InternVLGenerator, MobileVLMGenerator)):
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

    if torch.cuda.is_available():
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        text = _call()
        end_evt.record()
        torch.cuda.synchronize()
        latency_ms = start_evt.elapsed_time(end_evt)
    else:
        t0 = time.perf_counter()
        text = _call()
        latency_ms = (time.perf_counter() - t0) * 1000.0

    return text, latency_ms


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
    attention_experiment: str | None = None,
) -> dict[str, float | int | str]:
    if task not in SUPPORTED_TASKS:
        supported = ", ".join(sorted(SUPPORTED_TASKS))
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {supported}")

    sample = sample_records(rows, max_samples=max_samples, seed=seed)
    generator = load_hf_pipeline(model_id, attention_experiment=attention_experiment)
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
    max_frames_per_clip: int | None,
    output_path: str | Path,
    prompt_mode: str = "plain",
    resume: bool = False,
    dataset: str = "kabr",
    attention_experiment: str | None = None,
) -> dict[str, float | int | str]:
    if task not in SUPPORTED_TASKS:
        supported = ", ".join(sorted(SUPPORTED_TASKS))
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {supported}")

    clips = group_frame_records_into_clips(rows)
    sample = sample_records(clips, max_samples=max_samples, seed=seed)
    generator = load_hf_pipeline(model_id, attention_experiment=attention_experiment)
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
    max_frames_per_clip: int | None = 8,
    resume: bool = False,
    dataset: str = "kabr",
    behavior_labels: list[str] | None = None,
    species_labels: list[str] | None = None,
    warmup_clips: int = 0,
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

    # Warmup: run a few clips without recording results to heat the GPU/JIT cache.
    if warmup_clips > 0 and records:
        warmup_sample = records[:warmup_clips]
        for w_row in warmup_sample:
            w_paths = (
                [w_row.image_path]
                if isinstance(w_row, KABRFrameRecord)
                else sample_clip_frames(w_row, max_frames=max_frames_per_clip)
            )
            w_usable, _ = filter_readable_image_paths(w_paths)
            if w_usable:
                run_hf_generation(
                    generator=generator,
                    image_paths=w_usable,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )

    latencies_ms: list[float] = []

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

            usable_image_paths, skipped_image_paths = filter_readable_image_paths(image_paths)
            if not usable_image_paths:
                record = {
                    "input_id": input_id,
                    "unit": unit,
                    "task": task,
                    "model_id": model_id,
                    "gold": gold,
                    "prediction": {key: None for key in gold.keys()},
                    "parsed": False,
                    "correct": False,
                    "raw_text": "",
                    "latency_ms": None,
                    "input_image_paths": image_paths,
                    "skipped_image_paths": skipped_image_paths,
                    "error": "no_readable_images",
                    **extra,
                }
                handle.write(json.dumps(record, ensure_ascii=True))
                handle.write("\n")
                handle.flush()
                completed_ids.add(input_id)
                continue

            raw_text, latency_ms = run_hf_generation(
                generator=generator,
                image_paths=usable_image_paths,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            latencies_ms.append(latency_ms)
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
                "latency_ms": round(latency_ms, 2),
                "input_image_paths": usable_image_paths,
                "skipped_image_paths": skipped_image_paths,
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

    import statistics
    lat_summary: dict[str, Any] = {}
    if latencies_ms:
        lat_summary = {
            "latency_median_ms": round(statistics.median(latencies_ms), 2),
            "latency_p90_ms": round(sorted(latencies_ms)[int(len(latencies_ms) * 0.9)], 2),
            "latency_mean_ms": round(statistics.mean(latencies_ms), 2),
            "n_timed": len(latencies_ms),
        }

    return {
        "evaluated": total,
        "parsed": parsed,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "output_path": str(output_path),
        **lat_summary,
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
    attention_experiment: str | None = None,
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
        attention_experiment=attention_experiment,
    )
