from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceSummary:
    top1: str
    top2: str | None
    margin: float
    entropy: float
    probabilities: dict[str, float]


def softmax_scores(label_scores: dict[str, float]) -> dict[str, float]:
    if not label_scores:
        raise ValueError("label_scores must not be empty")
    max_score = max(label_scores.values())
    exp_scores = {label: math.exp(score - max_score) for label, score in label_scores.items()}
    total = sum(exp_scores.values())
    if total <= 0:
        raise ValueError("Could not normalize label scores")
    return {label: value / total for label, value in exp_scores.items()}


def summarize_confidence(label_scores: dict[str, float]) -> ConfidenceSummary:
    if not label_scores:
        raise ValueError("label_scores must not be empty")

    ranked = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
    top1, top1_score = ranked[0]
    top2 = ranked[1][0] if len(ranked) > 1 else None
    top2_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
    margin = top1_score - top2_score if top2 is not None else float("inf")

    probabilities = softmax_scores(label_scores)
    entropy = -sum(prob * math.log(prob) for prob in probabilities.values() if prob > 0.0)

    return ConfidenceSummary(
        top1=top1,
        top2=top2,
        margin=margin,
        entropy=entropy,
        probabilities=probabilities,
    )

