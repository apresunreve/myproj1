from dataclasses import dataclass
from pathlib import Path
from typing import Callable


SPECIES_LABELS = ["giraffe", "plains zebra", "grevys zebra"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    prompt_name: str
    supported_tasks: tuple[str, ...]
    species_labels: tuple[str, ...] = ()
    infer_species: Callable[[str], str] | None = None


def infer_kabr_species(original_video_id: str) -> str:
    if original_video_id.startswith("ZG"):
        return "grevys zebra"
    if original_video_id.startswith("ZP"):
        return "plains zebra"
    if original_video_id.startswith("G"):
        return "giraffe"
    raise ValueError(f"Unknown video prefix for species inference: {original_video_id}")


DATASET_SPECS = {
    "kabr": DatasetSpec(
        name="kabr",
        prompt_name="KABR wildlife dataset",
        supported_tasks=("behavior", "species", "joint", "action_recognition", "animal_recognition", "joint_animal_action_recognition"),
        species_labels=tuple(SPECIES_LABELS),
        infer_species=infer_kabr_species,
    ),
    "baboonland": DatasetSpec(
        name="baboonland",
        prompt_name="baboonland dataset",
        supported_tasks=("behavior", "action_recognition"),
        species_labels=(),
        infer_species=None,
    ),
}


def get_dataset_spec(dataset: str) -> DatasetSpec:
    normalized = dataset.strip().lower()
    if normalized not in DATASET_SPECS:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of: {supported}")
    return DATASET_SPECS[normalized]


def load_label_map(path: str | Path) -> dict[int, str]:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(value): key for key, value in raw.items()}


def ordered_label_names(path: str | Path) -> list[str]:
    label_map = load_label_map(path)
    return [label_map[index] for index in sorted(label_map)]
