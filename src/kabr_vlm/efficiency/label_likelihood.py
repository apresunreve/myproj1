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

    def __init__(self, model, processor) -> None:
        self.model = model.eval()
        self.processor = processor

    @classmethod
    def from_adapter(
        cls,
        model_id: str,
        adapter_path: str | Path,
        *,
        attention_experiment: str | None = None,
    ) -> "QwenLabelLikelihoodScorer":
        if not model_id.startswith("Qwen/Qwen2.5-VL-"):
            raise ValueError("QwenLabelLikelihoodScorer currently supports only Qwen2.5-VL model ids")
        generator = load_hf_adapter_generator(
            model_id=model_id,
            adapter_path=adapter_path,
            attention_experiment=attention_experiment,
        )
        return cls(model=generator.model, processor=generator.processor)

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

            scores: dict[str, float] = {}
            device = next(self.model.parameters()).device
            for label in labels:
                full_messages = build_vision_messages(prompt, image_paths, answer=label)
                full_text = _render_messages_for_processor(
                    self.processor,
                    full_messages,
                    add_generation_prompt=False,
                )
                encoded = self.processor(
                    text=[full_text],
                    images=[images],
                    return_tensors="pt",
                    padding=True,
                )
                encoded = {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}
                input_ids = encoded["input_ids"]
                if input_ids.shape[1] <= prompt_len:
                    raise RuntimeError(f"No continuation tokens found for label {label!r}")

                with torch.inference_mode():
                    outputs = self.model(**encoded)
                    log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)

                target_ids = input_ids[:, 1:]
                continuation_start = max(0, prompt_len - 1)
                continuation_log_probs = log_probs[:, continuation_start:, :].gather(
                    dim=-1,
                    index=target_ids[:, continuation_start:].unsqueeze(-1),
                )
                scores[label] = float(continuation_log_probs.sum().detach().cpu())

        return scores

    def predict(self, image_paths: list[str], prompt: str, labels: list[str]) -> LabelLikelihoodResult:
        label_scores = self.score_labels(image_paths=image_paths, prompt=prompt, labels=labels)
        confidence = summarize_confidence(label_scores)
        return LabelLikelihoodResult(
            label_scores=label_scores,
            prediction=confidence.top1,
            confidence=confidence,
        )
