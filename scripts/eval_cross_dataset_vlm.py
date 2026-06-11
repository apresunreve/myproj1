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
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

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


def _mean_abs_diff(left: Image.Image, right: Image.Image) -> float:
    left_pixels = list(left.getdata())
    right_pixels = list(right.getdata())
    if not left_pixels or len(left_pixels) != len(right_pixels):
        return 0.0
    return sum(abs(int(a) - int(b)) for a, b in zip(left_pixels, right_pixels, strict=True)) / len(left_pixels)


def _load_frame_thumbnails(image_paths: list[str]) -> list[Image.Image]:
    thumbnails: list[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as image:
            thumbnails.append(image.convert("L").resize((64, 64)))
    return thumbnails


def high_motion_indices(image_paths: list[str], max_items: int) -> list[int]:
    if len(image_paths) <= max_items:
        return list(range(len(image_paths)))
    if max_items == 1:
        return [len(image_paths) // 2]

    thumbnails = _load_frame_thumbnails(image_paths)

    transition_scores = [
        (score, idx - 1, idx)
        for idx, score in enumerate(
            (_mean_abs_diff(thumbnails[i - 1], thumbnails[i]) for i in range(1, len(thumbnails))),
            start=1,
        )
    ]
    selected = {0, len(image_paths) - 1}
    for _score, left_idx, right_idx in sorted(transition_scores, reverse=True):
        if len(selected) >= max_items:
            break
        selected.add(left_idx)
        if len(selected) >= max_items:
            break
        selected.add(right_idx)
    if len(selected) < max_items:
        for idx in select_frame_indices(len(image_paths), max_items, "uniform"):
            selected.add(idx)
            if len(selected) >= max_items:
                break
    return sorted(selected)[:max_items]


def low_redundancy_indices(image_paths: list[str], max_items: int) -> list[int]:
    if len(image_paths) <= max_items:
        return list(range(len(image_paths)))
    if max_items == 1:
        return [len(image_paths) // 2]

    thumbnails = _load_frame_thumbnails(image_paths)
    selected = {0, len(image_paths) - 1}
    while len(selected) < max_items:
        best_idx = None
        best_score = -1.0
        for idx in range(len(image_paths)):
            if idx in selected:
                continue
            min_distance = min(_mean_abs_diff(thumbnails[idx], thumbnails[selected_idx]) for selected_idx in selected)
            if min_distance > best_score:
                best_score = min_distance
                best_idx = idx
        if best_idx is None:
            break
        selected.add(best_idx)
    return sorted(selected)[:max_items]


def select_frame_indices(
    num_items: int,
    max_items: int | None,
    strategy: str,
    image_paths: list[str] | None = None,
) -> list[int]:
    if num_items <= 0:
        return []
    if max_items is None or max_items <= 0 or num_items <= max_items:
        return list(range(num_items))
    if strategy == "high_motion":
        if image_paths is None:
            raise ValueError("high_motion frame selection requires image paths")
        return high_motion_indices(image_paths, max_items)
    if strategy == "low_redundancy":
        if image_paths is None:
            raise ValueError("low_redundancy frame selection requires image paths")
        return low_redundancy_indices(image_paths, max_items)
    if max_items == 1:
        if strategy == "first":
            return [0]
        if strategy == "last":
            return [num_items - 1]
        return [num_items // 2]

    if strategy == "uniform":
        last_index = num_items - 1
        return [round(i * last_index / (max_items - 1)) for i in range(max_items)]
    if strategy == "first":
        return list(range(max_items))
    if strategy == "last":
        return list(range(num_items - max_items, num_items))
    if strategy == "middle":
        start = max(0, (num_items - max_items) // 2)
        return list(range(start, start + max_items))
    if strategy == "first_middle_last":
        if max_items == 2:
            return [0, num_items - 1]
        anchors = [0, num_items // 2, num_items - 1]
        if max_items == 3:
            return sorted(set(anchors))
        remaining = [idx for idx in range(num_items) if idx not in anchors]
        needed = max_items - len(set(anchors))
        extra = uniform_subsample(remaining, needed)
        return sorted(set(anchors + extra))[:max_items]
    if strategy == "stride":
        step = max(1, num_items // max_items)
        indices = list(range(0, num_items, step))[:max_items]
        while len(indices) < max_items:
            candidate = indices[-1] + 1 if indices else 0
            if candidate >= num_items:
                candidate = num_items - 1
            if candidate not in indices:
                indices.append(candidate)
            else:
                break
        return sorted(indices)
    raise ValueError(f"unsupported frame selection strategy: {strategy}")


def uniform_subsample(items: list[Any], max_items: int | None) -> list[Any]:
    if max_items is None or max_items <= 0 or len(items) <= max_items:
        return list(items)
    if max_items == 1:
        return [items[len(items) // 2]]
    last_index = len(items) - 1
    indices = [round(i * last_index / (max_items - 1)) for i in range(max_items)]
    return [items[index] for index in indices]


def select_frames(items: list[Any], max_items: int | None, strategy: str) -> tuple[list[Any], list[int]]:
    needs_paths = strategy in {"high_motion", "low_redundancy"}
    indices = select_frame_indices(len(items), max_items, strategy, image_paths=items if needs_paths else None)
    return [items[index] for index in indices], indices


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
    latencies_ms = [
        float(r["latency_ms"])
        for r in records
        if isinstance(r.get("latency_ms"), int | float) and r["latency_ms"] >= 0
    ]
    frame_counts = [
        int(r["num_input_frames"])
        for r in records
        if isinstance(r.get("num_input_frames"), int) and r["num_input_frames"] >= 0
    ]
    frame_selection_latencies_ms = [
        float(r["frame_selection_latency_ms"])
        for r in records
        if isinstance(r.get("frame_selection_latency_ms"), int | float) and r["frame_selection_latency_ms"] >= 0
    ]
    latency_summary = {}
    if latencies_ms:
        ordered = sorted(latencies_ms)
        p90_idx = max(0, min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1))
        latency_summary = {
            "n_timed": len(latencies_ms),
            "latency_median_ms": round(statistics.median(ordered), 2),
            "latency_p90_ms": round(ordered[p90_idx], 2),
            "latency_mean_ms": round(statistics.mean(ordered), 2),
            "latency_min_ms": round(ordered[0], 2),
            "latency_max_ms": round(ordered[-1], 2),
        }
    frame_summary = {}
    if frame_counts:
        frame_summary = {
            "num_input_frames_mean": round(statistics.mean(frame_counts), 2),
            "num_input_frames_min": min(frame_counts),
            "num_input_frames_max": max(frame_counts),
        }
    frame_selection_latency_summary = {}
    if frame_selection_latencies_ms:
        ordered = sorted(frame_selection_latencies_ms)
        p90_idx = max(0, min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1))
        frame_selection_latency_summary = {
            "frame_selection_latency_median_ms": round(statistics.median(ordered), 2),
            "frame_selection_latency_p90_ms": round(ordered[p90_idx], 2),
            "frame_selection_latency_mean_ms": round(statistics.mean(ordered), 2),
            "frame_selection_latency_max_ms": round(ordered[-1], 2),
        }
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
        **latency_summary,
        **frame_summary,
        **frame_selection_latency_summary,
    }


def _cuda_peak_memory_mb() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def _reset_cuda_peak_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _synchronize_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--target_dataset", required=True, choices=["kabr", "baboonland"])
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument(
        "--max_input_frames",
        type=int,
        default=None,
        help="Subsample each row's image_paths to this many frames before evaluation.",
    )
    parser.add_argument(
        "--frame_selection",
        choices=[
            "uniform",
            "first",
            "last",
            "middle",
            "first_middle_last",
            "stride",
            "high_motion",
            "low_redundancy",
        ],
        default="uniform",
        help="Frame selection strategy used with --max_input_frames.",
    )
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

    _reset_cuda_peak_memory()
    eval_start = time.perf_counter()
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
            selection_t0 = time.perf_counter()
            image_paths, frame_indices = select_frames(row["image_paths"], args.max_input_frames, args.frame_selection)
            frame_selection_latency_ms = (time.perf_counter() - selection_t0) * 1000.0
            prompt = row["prompt"]
            latency_ms = None
            try:
                _synchronize_cuda()
                t0 = time.perf_counter()
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
                _synchronize_cuda()
                latency_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:  # noqa: BLE001
                _synchronize_cuda()
                if "t0" in locals():
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                text = f"__generation_error__: {exc!r}"
            pred_label = extract_label(text, args.target_dataset)
            record = {
                "id": row["id"],
                "gold_label": gold_label,
                "pred_text": text,
                "pred_label": pred_label,
                "latency_ms": None if latency_ms is None else round(latency_ms, 2),
                "num_input_frames": len(image_paths),
                "frame_selection": args.frame_selection,
                "selected_frame_indices": frame_indices,
                "frame_selection_latency_ms": round(frame_selection_latency_ms, 2),
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

    eval_wall_seconds = time.perf_counter() - eval_start
    metrics = compute_metrics(records, args.target_dataset)
    metrics["eval_wall_seconds"] = round(eval_wall_seconds, 2)
    if pending_rows:
        metrics["new_rows_per_second"] = round(len(pending_rows) / eval_wall_seconds, 4) if eval_wall_seconds else 0.0
    metrics["peak_gpu_memory_mb"] = _cuda_peak_memory_mb()
    metrics_path = Path(str(output_path) + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")
    print(
        f"wrote predictions: {output_path}\nwrote metrics: {metrics_path}\n"
        f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"parsed_rate={metrics['parsed_rate']:.4f} "
        f"latency_median_ms={metrics.get('latency_median_ms', 'NA')} "
        f"latency_p90_ms={metrics.get('latency_p90_ms', 'NA')} "
        f"peak_gpu_memory_mb={metrics.get('peak_gpu_memory_mb', 'NA')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
