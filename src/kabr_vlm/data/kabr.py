import csv
import json
from pathlib import Path
from typing import Any

from kabr_vlm.data.kabr_frames import (
    group_frame_records_into_clips,
    read_charades_frame_annotations,
    read_kabr_frame_annotations,
    sample_clip_frames,
)
from kabr_vlm.datasets import get_dataset_spec, ordered_label_names
from kabr_vlm.prompts import build_joint_answer, build_joint_prompt, build_task_answer, build_task_prompt
from kabr_vlm.schemas import KABRAnnotation, VideoQAExample


def read_annotations(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def convert_annotations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        annotation = KABRAnnotation(
            video_id=str(row.get("video_id", index)),
            species=str(row["species"]),
            behavior=str(row["behavior"]),
            clip_path=str(row.get("clip_path", "")),
            split=str(row.get("split", "train")),
            question_id=str(row.get("question_id")) if row.get("question_id") is not None else None,
        )
        example = VideoQAExample(
            id=annotation.question_id or f"{annotation.video_id}:{index}",
            video_id=annotation.video_id,
            clip_path=annotation.clip_path,
            split=annotation.split,
            task="joint_animal_action_recognition",
            prompt=build_joint_prompt(annotation),
            answer=build_joint_answer(annotation),
            metadata={
                "species": annotation.species,
                "behavior": annotation.behavior,
            },
        )
        examples.append(example.model_dump())
    return examples


def convert_kabr_split_to_videoqa(
    annotation_path: str | Path,
    image_root: str | Path,
    classes_path: str | Path,
    task: str,
    max_frames_per_clip: int,
    output_format: str = "structured",
    dataset: str = "kabr",
) -> list[dict[str, Any]]:
    if task not in {"animal_recognition", "action_recognition", "joint_animal_action_recognition"}:
        raise ValueError(f"Unsupported task: {task}")

    rows = read_charades_frame_annotations(annotation_path, image_root, classes_path, dataset=dataset)
    clips = group_frame_records_into_clips(rows)
    unit = "clip"
    behavior_labels = ordered_label_names(classes_path)
    species_labels = list(get_dataset_spec(dataset).species_labels)
    prompt = build_joint_prompt(
        KABRAnnotation(video_id="template", species="giraffe", behavior="Walk", clip_path="", split="train")
    ) if task == "joint_animal_action_recognition" else build_task_prompt(
        task,
        unit=unit,
        output_format=output_format,
        dataset=dataset,
        behavior_labels=behavior_labels,
        species_labels=species_labels,
    )

    examples: list[dict[str, Any]] = []
    for clip in clips:
        if task == "joint_animal_action_recognition":
            annotation = KABRAnnotation(
                video_id=clip.video_id,
                species=clip.species,
                behavior=clip.label_name,
                clip_path=clip.clip_id,
                split=clip.split,
                question_id=clip.clip_id,
            )
            answer = build_joint_answer(annotation)
            prompt_text = build_joint_prompt(annotation)
        else:
            answer = build_task_answer(
                task=task,
                species=clip.species or "",
                behavior=clip.label_name,
                output_format=output_format,
            )
            prompt_text = prompt

        example = VideoQAExample(
            id=clip.clip_id,
            video_id=clip.video_id,
            clip_path=clip.clip_id,
            image_paths=sample_clip_frames(clip, max_frames=max_frames_per_clip),
            split=clip.split,
            task=task,
            prompt=prompt_text,
            answer=answer,
            metadata={
                "dataset": dataset,
                "species": clip.species,
                "behavior": clip.label_name,
                "original_video_id": clip.original_video_id,
                "label_id": clip.label_id,
                "num_frames": len(clip.frame_ids),
                "label_space": {
                    "behavior": behavior_labels,
                    "species": species_labels,
                },
            },
        )
        examples.append(example.model_dump())

    return examples


def convert_charades_split_to_clip_manifests(
    annotation_path: str | Path,
    image_root: str | Path,
    classes_path: str | Path,
    dataset: str = "kabr",
) -> list[dict[str, Any]]:
    rows = read_charades_frame_annotations(annotation_path, image_root, classes_path, dataset=dataset)
    clips = group_frame_records_into_clips(rows)
    return [
        {
            "clip_id": clip.clip_id,
            "original_video_id": clip.original_video_id,
            "video_id": clip.video_id,
            "dataset": clip.dataset,
            "species": clip.species,
            "label_id": clip.label_id,
            "label_name": clip.label_name,
            "frame_ids": clip.frame_ids,
            "rel_paths": clip.rel_paths,
            "image_paths": clip.image_paths,
            "split": clip.split,
        }
        for clip in clips
    ]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def write_clip_manifest_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, rows)
