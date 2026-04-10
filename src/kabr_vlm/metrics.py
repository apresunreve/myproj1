import csv
import json
from collections import Counter
from pathlib import Path

from kabr_vlm.vlm_baselines import BEHAVIOR_LABELS, SPECIES_LABELS


def load_prediction_records(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def task_labels(task: str) -> list[str]:
    if task == "behavior":
        return BEHAVIOR_LABELS
    if task == "species":
        return SPECIES_LABELS
    if task == "joint":
        return [f"{species} | {behavior}" for species in SPECIES_LABELS for behavior in BEHAVIOR_LABELS]
    raise ValueError(f"Unsupported task: {task}")


def infer_task_labels(rows: list[dict], task: str) -> list[str]:
    for row in rows:
        label_space = row.get("label_space")
        if isinstance(label_space, dict):
            if task == "behavior" and isinstance(label_space.get("behavior"), list) and label_space["behavior"]:
                return list(label_space["behavior"])
            if task == "species" and isinstance(label_space.get("species"), list) and label_space["species"]:
                return list(label_space["species"])
            if task == "joint":
                species = label_space.get("species")
                behavior = label_space.get("behavior")
                if isinstance(species, list) and isinstance(behavior, list) and species and behavior:
                    return [f"{species_label} | {behavior_label}" for species_label in species for behavior_label in behavior]
    return task_labels(task)


def normalize_prediction_row(row: dict, task: str | None = None) -> dict:
    if "gold" in row and "prediction" in row and "task" in row:
        return row

    resolved_task = task
    if resolved_task is None and "gold_label_name" in row and "pred_label_name" in row:
        resolved_task = "behavior"
    if resolved_task is None:
        raise ValueError("Could not infer task for prediction row.")

    if resolved_task == "behavior":
        gold = {"behavior": row.get("gold_label_name")}
        prediction = {"behavior": row.get("pred_label_name")}
    elif resolved_task == "species":
        gold = {"species": row.get("gold_label_name")}
        prediction = {"species": row.get("pred_label_name")}
    elif resolved_task == "joint":
        gold = {
            "species": row.get("gold_species"),
            "behavior": row.get("gold_label_name"),
        }
        prediction = {
            "species": row.get("pred_species"),
            "behavior": row.get("pred_label_name"),
        }
    else:
        raise ValueError(f"Unsupported task: {resolved_task}")

    parsed = all(value is not None for value in prediction.values())
    normalized = {
        "task": resolved_task,
        "gold": gold,
        "prediction": prediction,
        "parsed": parsed,
    }
    for key in (
        "clip_id",
        "original_video_id",
        "video_id",
        "gold_label_id",
        "pred_label_id",
        "logits",
        "label_space",
        "dataset",
    ):
        if key in row:
            normalized[key] = row[key]
    return normalized


def label_for_payload(task: str, payload: dict[str, str | None]) -> str | None:
    if task == "behavior":
        return payload.get("behavior")
    if task == "species":
        return payload.get("species")
    species = payload.get("species")
    behavior = payload.get("behavior")
    if species is None or behavior is None:
        return None
    return f"{species} | {behavior}"


def score_prediction_records(rows: list[dict], task: str | None = None, labels_override: list[str] | None = None) -> dict:
    if not rows:
        raise ValueError("No prediction rows supplied.")

    normalized_rows = [normalize_prediction_row(row, task=task) for row in rows]
    resolved_task = normalized_rows[0]["task"]
    labels = labels_override or infer_task_labels(normalized_rows, resolved_task)
    gold_counts = Counter()
    pred_counts = Counter()
    tp_counts = Counter()
    confusion = {gold: {pred: 0 for pred in labels} for gold in labels}

    parsed = 0
    correct = 0
    total = len(rows)

    for row in normalized_rows:
        gold = label_for_payload(resolved_task, row["gold"])
        pred = label_for_payload(resolved_task, row["prediction"])
        if gold is None:
            continue
        gold_counts[gold] += 1
        if row.get("parsed"):
            parsed += 1
        if pred in labels:
            pred_counts[pred] += 1
            confusion[gold][pred] += 1
        if pred == gold:
            correct += 1
            tp_counts[gold] += 1

    per_class = {}
    f1_values = []
    for label in labels:
        tp = tp_counts[label]
        fp = pred_counts[label] - tp
        fn = gold_counts[label] - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = gold_counts[label]
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support:
            f1_values.append(f1)

    return {
        "task": resolved_task,
        "evaluated": total,
        "parsed": parsed,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
        "labels": labels,
        "confusion_matrix": confusion,
    }


def write_metrics(path: str | Path, metrics: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_confusion_csv(path: str | Path, metrics: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels = metrics["labels"]
    confusion = metrics["confusion_matrix"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold/pred", *labels])
        for gold in labels:
            writer.writerow([gold, *[confusion[gold][pred] for pred in labels]])
