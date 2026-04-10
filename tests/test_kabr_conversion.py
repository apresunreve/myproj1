from pathlib import Path

from kabr_vlm.data.kabr import convert_annotations, convert_kabr_split_to_videoqa
from kabr_vlm.schemas import KABRAnnotation


def test_convert_annotations_builds_joint_task_example() -> None:
    rows = [
        KABRAnnotation(
            video_id="ZG0001.1",
            species="plains zebra",
            behavior="graze",
            clip_path="ZG0001.1.mp4",
        )
    ]

    examples = convert_annotations(rows)

    assert len(examples) == 1
    example = examples[0]
    assert example.task == "joint_animal_action_recognition"
    assert 'recognize(entity_type="animal", value="plains zebra")' in example.answer
    assert 'recognize(entity_type="action", value="graze"' in example.answer


def test_convert_kabr_split_to_videoqa_builds_clip_examples() -> None:
    examples = convert_kabr_split_to_videoqa(
        annotation_path=Path("../KABR_files/annotation/val.csv"),
        image_root=Path("../KABR_files/dataset/image"),
        classes_path=Path("../KABR_files/annotation/classes.json"),
        task="action_recognition",
        max_frames_per_clip=4,
    )

    assert examples
    first = examples[0]
    assert first.task == "action_recognition"
    assert first.image_paths
    assert len(first.image_paths) <= 4


def test_convert_kabr_split_to_videoqa_builds_plain_species_examples() -> None:
    examples = convert_kabr_split_to_videoqa(
        annotation_path=Path("../KABR_files/annotation/val.csv"),
        image_root=Path("../KABR_files/dataset/image"),
        classes_path=Path("../KABR_files/annotation/classes.json"),
        task="animal_recognition",
        max_frames_per_clip=8,
        output_format="plain",
    )

    assert examples
    first = examples[0]
    assert first.task == "animal_recognition"
    assert first.answer in {"giraffe", "plains zebra", "grevys zebra"}
    assert "Reply with exactly one label" in first.prompt


def test_convert_baboonland_split_to_videoqa_builds_behavior_examples() -> None:
    examples = convert_kabr_split_to_videoqa(
        annotation_path=Path("/home/jhyan/baboon_data/charades/annotation/val.csv"),
        image_root=Path("/home/jhyan/baboon_data/charades/dataset/image"),
        classes_path=Path("/home/jhyan/baboon_data/charades/annotation/classes.json"),
        dataset="baboonland",
        task="action_recognition",
        max_frames_per_clip=4,
        output_format="plain",
    )

    assert examples
    first = examples[0]
    assert first.metadata["dataset"] == "baboonland"
    assert first.metadata["species"] is None
    assert first.answer in {
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
    }
