"""Dataset-agnostic clip-level VLM evaluation (zero-shot or LoRA-adapter).

Scores a generated prediction against either KABR (8 classes) or baboonland
(13 classes). Fuzzy label extraction via substring matching; fall-through
heuristics bridge common cross-dataset phrasings (Walk<->Walking/Running,
Graze/Browse->Foraging, etc) so KABR-trained adapters can be scored on the
baboonland label space and vice versa.

Emits one prediction record per row + a metrics.json with accuracy, macro F1,
per-class F1, and pred/gold distributions.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from kabr_vlm.vlm_baselines import (  # noqa: E402
    load_hf_adapter_generator,
    load_hf_pipeline,
    match_candidate,
    normalize_text,
)

KABR_CLASSES = ["Walk", "Graze", "Browse", "Head Up", "Auto-Groom", "Trot", "Run", "Occluded"]

BABOONLAND_CLASSES = [
    "Walking/Running",
    "Sitting/Standing",
    "Fighting/Playing",
    "Self-Grooming",
    "Being Groomed",
    "Grooming Somebody",
    "Mutual Grooming",
    "Infant-Carrying",
    "Foraging",
    "Drinking",
    "Mounting",
    "Sleeping",
    "Occluded",
]


def _kabr_fallback(normalized: str) -> str | None:
    # map common baboonland-style outputs onto nearest KABR class
    if "walk" in normalized and "run" in normalized:
        return "Walk"
    if "running" in normalized:
        return "Run"
    if "walking" in normalized or "walk" in normalized:
        return "Walk"
    if "trot" in normalized:
        return "Trot"
    if "graz" in normalized:
        return "Graze"
    if "brows" in normalized or "forag" in normalized or "feed" in normalized or "eat" in normalized:
        return "Graze"
    if "groom" in normalized:
        return "Auto-Groom"
    if "head" in normalized and "up" in normalized:
        return "Head Up"
    if "sit" in normalized or "stand" in normalized:
        return "Head Up"
    if "occlu" in normalized or "hidden" in normalized or "obscur" in normalized:
        return "Occluded"
    return None


def _baboonland_fallback(normalized: str) -> str | None:
    if "groom" in normalized:
        if "mutual" in normalized:
            return "Mutual Grooming"
        if "being" in normalized or "received" in normalized:
            return "Being Groomed"
        if "self" in normalized or "auto" in normalized:
            return "Self-Grooming"
        if "somebody" in normalized or "other" in normalized or "another" in normalized:
            return "Grooming Somebody"
        return "Self-Grooming"
    if "walk" in normalized or "run" in normalized or "trot" in normalized:
        return "Walking/Running"
    if "sit" in normalized or "stand" in normalized or "head up" in normalized:
        return "Sitting/Standing"
    if "fight" in normalized or "play" in normalized:
        return "Fighting/Playing"
    if "carry" in normalized and "infant" in normalized:
        return "Infant-Carrying"
    if "forag" in normalized or "graz" in normalized or "brows" in normalized or "feed" in normalized or "eat" in normalized:
        return "Foraging"
    if "drink" in normalized:
        return "Drinking"
    if "mount" in normalized:
        return "Mounting"
    if "sleep" in normalized:
        return "Sleeping"
    if "occlu" in normalized or "hidden" in normalized or "obscur" in normalized:
        return "Occluded"
    return None


def _spec(target: str) -> tuple[list[str], callable]:
    if target == "kabr":
        return KABR_CLASSES, _kabr_fallback
    if target == "baboonland":
        return BABOONLAND_CLASSES, _baboonland_fallback
    raise ValueError(f"unsupported target_dataset: {target}")


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def extract_label(text: str, target: str) -> str | None:
    classes, fallback = _spec(target)
    matched = match_candidate(text, classes)
    if matched is not None:
        return matched
    normalized = normalize_text(text)
    return fallback(normalized)


def compute_metrics(records: list[dict], target: str) -> dict:
    labels, _ = _spec(target)
    n_total = len(records)
    n_correct = sum(1 for r in records if r["pred_label"] == r["gold_label"])
    n_parsed = sum(1 for r in records if r["pred_label"] is not None)
    accuracy = n_correct / n_total if n_total else 0.0
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    for r in records:
        gold = r["gold_label"]
        pred = r["pred_label"]
        per_class[gold]["support"] += 1
        if pred == gold:
            per_class[gold]["tp"] += 1
        else:
            per_class[gold]["fn"] += 1
            if pred is not None:
                per_class[pred]["fp"] += 1
    per_class_f1 = {}
    f1_sum = 0.0
    for label in labels:
        stats = per_class[label]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        per_class_f1[label] = {
            "support": stats["support"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1": f1,
        }
        f1_sum += f1
    macro_f1 = f1_sum / len(labels)
    pred_distribution = Counter(r["pred_label"] or "__unparsed__" for r in records)
    gold_distribution = Counter(r["gold_label"] for r in records)
    return {
        "target_dataset": target,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_parsed": n_parsed,
        "accuracy": accuracy,
        "parsed_rate": n_parsed / n_total if n_total else 0.0,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "pred_distribution": dict(pred_distribution),
        "gold_distribution": dict(gold_distribution),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--target_dataset", required=True, choices=["kabr", "baboonland"])
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    val_rows = load_jsonl(Path(args.val_jsonl), limit=args.max_samples)
    print(f"loaded {len(val_rows)} val rows from {args.val_jsonl}", flush=True)
    print(f"target_dataset={args.target_dataset}", flush=True)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    records: list[dict] = []
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" in d:
                    done_ids.add(d["id"])
                    records.append(d)
        print(f"resume: {len(done_ids)} rows already in {output_path}", flush=True)

    pending_rows = [r for r in val_rows if r["id"] not in done_ids]
    print(f"pending rows to evaluate: {len(pending_rows)}", flush=True)

    if pending_rows:
        if args.adapter_path:
            print(f"loading adapter generator: base={args.model_id} adapter={args.adapter_path}", flush=True)
            generator = load_hf_adapter_generator(model_id=args.model_id, adapter_path=args.adapter_path)
        else:
            print(f"loading zero-shot pipeline: {args.model_id}", flush=True)
            generator = load_hf_pipeline(model_id=args.model_id)
        is_pipeline = hasattr(generator, "task")
    else:
        generator = None
        is_pipeline = False

    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(pending_rows):
            meta = row.get("metadata", {})
            gold_label = (
                meta.get("behavior")
                if args.target_dataset == "baboonland"
                else (row.get("answer") or meta.get("behavior"))
            )
            # Both processed JSONLs (kabr + baboonland) place the plain behavior
            # label in metadata.behavior; fall back to row.answer just in case.
            if not isinstance(gold_label, str):
                gold_label = row.get("answer", "")
            image_paths = row["image_paths"]
            prompt = row["prompt"]
            try:
                if is_pipeline:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                *[{"type": "image", "url": path} for path in image_paths],
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    output = generator(messages, max_new_tokens=args.max_new_tokens, do_sample=False)
                    text = output[0]["generated_text"][-1]["content"] if isinstance(output, list) else str(output)
                else:
                    text = generator(image_paths=image_paths, prompt=prompt, max_new_tokens=args.max_new_tokens)
            except Exception as exc:  # noqa: BLE001
                text = f"__generation_error__: {exc!r}"
            pred_label = extract_label(text, args.target_dataset)
            record = {
                "id": row["id"],
                "gold_label": gold_label,
                "pred_text": text,
                "pred_label": pred_label,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            records.append(record)
            if (index + 1) % 100 == 0 or (index + 1) == len(pending_rows):
                acc_so_far = sum(1 for r in records if r["pred_label"] == r["gold_label"]) / len(records)
                print(
                    f"[{index + 1}/{len(pending_rows)}] total={len(records)} acc_so_far={acc_so_far:.4f}",
                    flush=True,
                )

    metrics = compute_metrics(records, args.target_dataset)
    metrics_path = Path(str(output_path) + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")
    print(
        f"wrote predictions: {output_path}\nwrote metrics: {metrics_path}\n"
        f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"parsed_rate={metrics['parsed_rate']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
