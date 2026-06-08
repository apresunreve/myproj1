from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from kabr_vlm.efficiency.confidence import ConfidenceSummary, summarize_confidence
from kabr_vlm.vlm_baselines import _render_messages_for_processor, load_hf_adapter_generator


@dataclass(frozen=True)
class LabelLikelihoodResult:
    label_scores: dict[str, float]
    prediction: str
    confidence: ConfidenceSummary


def build_vision_messages(prompt: str, image_paths: list[str], answer: str | None = None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        *[{"type": "image", "image": path} for path in image_paths],
        {"type": "text", "text": prompt},
    ]
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


class QwenLabelLikelihoodScorer:
    """Closed-set label scoring for Qwen2.5-VL-style HF adapter generators.

    The scorer computes the log likelihood of each candidate assistant label
    under the model, conditioned on the same visual prompt used by generation
    eval. It is intentionally Qwen-first; other model families can be added
    behind the same interface once this path is validated.
    """

    def __init__(self, model, processor, label_batch_size: int | None = 1) -> None:
        self.model = model.eval()
        self.processor = processor
        self.label_batch_size = label_batch_size

    @classmethod
    def from_adapter(
        cls,
        model_id: str,
        adapter_path: str | Path,
        *,
        attention_experiment: str | None = None,
        label_batch_size: int | None = 1,
    ) -> "QwenLabelLikelihoodScorer":
        if not model_id.startswith("Qwen/Qwen2.5-VL-"):
            raise ValueError("QwenLabelLikelihoodScorer currently supports only Qwen2.5-VL model ids")
        generator = load_hf_adapter_generator(
            model_id=model_id,
            adapter_path=adapter_path,
            attention_experiment=attention_experiment,
        )
        return cls(model=generator.model, processor=generator.processor, label_batch_size=label_batch_size)

    def _label_batches(self, labels: list[str]) -> list[list[str]]:
        if self.label_batch_size is None or self.label_batch_size <= 0:
            return [labels]
        return [labels[start : start + self.label_batch_size] for start in range(0, len(labels), self.label_batch_size)]

    def score_labels(self, image_paths: list[str], prompt: str, labels: list[str]) -> dict[str, float]:
        if not labels:
            raise ValueError("labels must not be empty")

        import torch

        with ExitStack() as stack:
            images = [stack.enter_context(Image.open(path)).convert("RGB") for path in image_paths]
            prompt_messages = build_vision_messages(prompt, image_paths)
            prompt_text = _render_messages_for_processor(
                self.processor,
                prompt_messages,
                add_generation_prompt=True,
            )
            prompt_encoded = self.processor(
                text=[prompt_text],
                images=[images],
                return_tensors="pt",
                padding=True,
            )
            prompt_len = int(prompt_encoded["input_ids"].shape[1])

            device = next(self.model.parameters()).device
            scores: dict[str, float] = {}
            tokenizer = getattr(self.processor, "tokenizer", None)
            original_padding_side = getattr(tokenizer, "padding_side", None)
            if tokenizer is not None:
                tokenizer.padding_side = "right"
            try:
                for label_batch in self._label_batches(labels):
                    full_texts = [
                        _render_messages_for_processor(
                            self.processor,
                            build_vision_messages(prompt, image_paths, answer=label),
                            add_generation_prompt=False,
                        )
                        for label in label_batch
                    ]
                    batch_images = [images for _ in label_batch]
                    encoded = self.processor(
                        text=full_texts,
                        images=batch_images,
                        return_tensors="pt",
                        padding=True,
                    )
                    encoded = {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}
                    input_ids = encoded["input_ids"]
                    attention_mask = encoded.get("attention_mask")
                    if input_ids.shape[1] <= prompt_len:
                        raise RuntimeError("No continuation tokens found for label batch")

                    with torch.inference_mode():
                        outputs = self.model(**encoded)
                        log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)

                    target_ids = input_ids[:, 1:]
                    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
                    continuation_start = max(0, prompt_len - 1)
                    for row_index, label in enumerate(label_batch):
                        if attention_mask is None:
                            actual_len = int(input_ids.shape[1])
                        else:
                            actual_len = int(attention_mask[row_index].sum().detach().cpu())
                        continuation_end = actual_len - 1
                        if continuation_end <= continuation_start:
                            raise RuntimeError(f"No continuation tokens found for label {label!r}")
                        score = token_log_probs[row_index, continuation_start:continuation_end].sum()
                        scores[label] = float(score.detach().cpu())
            finally:
                if tokenizer is not None and original_padding_side is not None:
                    tokenizer.padding_side = original_padding_side

        return scores

    def predict(self, image_paths: list[str], prompt: str, labels: list[str]) -> LabelLikelihoodResult:
        label_scores = self.score_labels(image_paths=image_paths, prompt=prompt, labels=labels)
        confidence = summarize_confidence(label_scores)
        return LabelLikelihoodResult(
            label_scores=label_scores,
            prediction=confidence.top1,
            confidence=confidence,
        )
