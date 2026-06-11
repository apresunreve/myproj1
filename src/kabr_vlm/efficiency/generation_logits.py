from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from kabr_vlm.efficiency.label_likelihood import build_vision_messages
from kabr_vlm.vlm_baselines import _model_device, _render_messages_for_processor, load_hf_adapter_generator


@dataclass(frozen=True)
class GenerationLogitConfidenceResult:
    text: str
    token_ids: list[int]
    confidence: dict[str, float | int | None]


class QwenGenerationLogitScorer:
    """Cheap confidence proxy from normal greedy generation logits.

    This does not rescore all candidate labels. It runs the same generation
    path used by the baseline evaluator and summarizes the probabilities of
    the tokens the model actually emitted.
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
    ) -> "QwenGenerationLogitScorer":
        generator = load_hf_adapter_generator(
            model_id=model_id,
            adapter_path=adapter_path,
            attention_experiment=attention_experiment,
        )
        return cls(model=generator.model, processor=generator.processor)

    def generate_with_confidence(
        self,
        image_paths: list[str],
        prompt: str,
        *,
        max_new_tokens: int = 16,
    ) -> GenerationLogitConfidenceResult:
        import torch

        with ExitStack() as stack:
            images = [stack.enter_context(Image.open(path)).convert("RGB") for path in image_paths]
            prompt_text = _render_messages_for_processor(
                self.processor,
                build_vision_messages(prompt, image_paths),
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
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        with torch.inference_mode():
            generated = self.model.generate(**generate_kwargs)

        prompt_len = int(encoded["input_ids"].shape[1])
        completion = generated.sequences[:, prompt_len:]
        text = self.processor.batch_decode(
            completion,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        token_ids = [int(token_id) for token_id in completion[0].detach().cpu().tolist()]
        confidence = self._summarize_token_scores(token_ids, generated.scores)
        return GenerationLogitConfidenceResult(text=text, token_ids=token_ids, confidence=confidence)

    def _summarize_token_scores(self, token_ids: list[int], score_steps: tuple[Any, ...]) -> dict[str, float | int | None]:
        import torch

        tokenizer = getattr(self.processor, "tokenizer", None)
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

        token_probs: list[float] = []
        token_logprobs: list[float] = []
        token_logit_margins: list[float] = []
        token_entropies: list[float] = []
        first_token_prob: float | None = None
        first_token_logit_margin: float | None = None

        for index, token_id in enumerate(token_ids[: len(score_steps)]):
            if token_id in special_ids:
                continue
            logits = score_steps[index][0].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            token_prob = float(probs[token_id].detach().cpu())
            token_logprob = float(log_probs[token_id].detach().cpu())
            top_values = torch.topk(logits, k=2).values
            logit_margin = float((top_values[0] - top_values[1]).detach().cpu())
            entropy = float((-(probs * log_probs).sum()).detach().cpu())

            if first_token_prob is None:
                first_token_prob = token_prob
                first_token_logit_margin = logit_margin
            token_probs.append(token_prob)
            token_logprobs.append(token_logprob)
            token_logit_margins.append(logit_margin)
            token_entropies.append(entropy)

        return {
            "n_scored_tokens": len(token_probs),
            "mean_token_prob": _mean(token_probs),
            "min_token_prob": min(token_probs) if token_probs else None,
            "mean_token_logprob": _mean(token_logprobs),
            "sum_token_logprob": sum(token_logprobs) if token_logprobs else None,
            "mean_token_logit_margin": _mean(token_logit_margins),
            "min_token_logit_margin": min(token_logit_margins) if token_logit_margins else None,
            "mean_token_entropy": _mean(token_entropies),
            "first_token_prob": first_token_prob,
            "first_token_logit_margin": first_token_logit_margin,
        }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
