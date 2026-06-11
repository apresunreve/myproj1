import json
import random
from dataclasses import dataclass
from pathlib import Path

from kabr_vlm.datasets import get_dataset_spec


SPECIES_LABELS = ["giraffe", "plains zebra", "grevys zebra"]
_VIDEO_PREFIX_TO_SPECIES = {
    "G": "giraffe",
    "ZP": "plains zebra",
    "ZG": "grevys zebra",
}


@dataclass(frozen=True)
class KABRFrameRecord:
    original_video_id: str
    video_id: str
    frame_id: int
    rel_path: str
    image_path: str
    label_id: int
    label_name: str
    species: str | None
    split: str
    dataset: str = "kabr"


@dataclass(frozen=True)
class KABRClipRecord:
    clip_id: str
    original_video_id: str
    video_id: str
    species: str | None
    label_id: int
    label_name: str
    frame_ids: list[int]
    rel_paths: list[str]
    image_paths: list[str]
    split: str
    dataset: str = "kabr"


def _infer_species(original_video_id: str) -> str:
    for prefix, species in _VIDEO_PREFIX_TO_SPECIES.items():
        if original_video_id.startswith(prefix):
            return species
    raise ValueError(f"Unsupported video id prefix for species inference: {original_video_id}")


def infer_species(original_video_id: str) -> str:
    return _infer_species(original_video_id)


def _load_label_lookup(classes_path: str | Path) -> dict[int, str]:
    payload = json.loads(Path(classes_path).read_text(encoding="utf-8"))
    return {int(label_id): label_name for label_name, label_id in payload.items()}


def load_label_map(classes_path: str | Path) -> dict[int, str]:
    return _load_label_lookup(classes_path)


def _infer_split(annotation_path: str | Path) -> str:
    name = Path(annotation_path).stem.lower()
    if "train" in name:
        return "train"
    if "val" in name:
        return "val"
    if "test" in name:
        return "test"
    return name or "unknown"


def read_kabr_frame_annotations(
    annotation_path: str | Path,
    image_root: str | Path,
    classes_path: str | Path,
) -> list[KABRFrameRecord]:
    image_root = Path(image_root)
    label_lookup = _load_label_lookup(classes_path)
    split = _infer_split(annotation_path)
    rows: list[KABRFrameRecord] = []

    with Path(annotation_path).open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            return rows
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Unexpected annotation row format: {line}")
            original_video_id, video_id, frame_id, rel_path, label_id = parts
            label_id_int = int(label_id)
            rows.append(
                KABRFrameRecord(
                    original_video_id=original_video_id,
                    video_id=video_id,
                    frame_id=int(frame_id),
                    rel_path=rel_path,
                    image_path=str(image_root / rel_path),
                    label_id=label_id_int,
                    label_name=label_lookup[label_id_int],
                    species=_infer_species(original_video_id),
                    split=split,
                    dataset="kabr",
                )
            )

    rows.sort(key=lambda row: (row.original_video_id, row.label_id, row.frame_id))
    return rows


def read_charades_frame_annotations(
    annotation_path: str | Path,
    image_root: str | Path,
    classes_path: str | Path,
    dataset: str = "kabr",
) -> list[KABRFrameRecord]:
    image_root = Path(image_root)
    label_lookup = _load_label_lookup(classes_path)
    split = _infer_split(annotation_path)
    spec = get_dataset_spec(dataset)
    rows: list[KABRFrameRecord] = []

    with Path(annotation_path).open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            return rows
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Unexpected annotation row format: {line}")
            original_video_id, video_id, frame_id, rel_path, label_id = parts
            label_id_int = int(label_id)
            species = spec.infer_species(original_video_id) if spec.infer_species is not None else None
            rows.append(
                KABRFrameRecord(
                    original_video_id=original_video_id,
                    video_id=video_id,
                    frame_id=int(frame_id),
                    rel_path=rel_path,
                    image_path=str(image_root / rel_path),
                    label_id=label_id_int,
                    label_name=label_lookup[label_id_int],
                    species=species,
                    split=split,
                    dataset=spec.name,
                )
            )

    rows.sort(key=lambda row: (row.original_video_id, row.label_id, row.frame_id))
    return rows


def group_frame_records_into_clips(rows: list[KABRFrameRecord]) -> list[KABRClipRecord]:
    if not rows:
        return []

    clips: list[KABRClipRecord] = []
    current_rows: list[KABRFrameRecord] = [rows[0]]

    def flush() -> None:
        if not current_rows:
            return
        first = current_rows[0]
        frame_ids = [row.frame_id for row in current_rows]
        clip = KABRClipRecord(
            clip_id=f"{first.original_video_id}:{frame_ids[0]}-{frame_ids[-1]}:{first.label_id}",
            original_video_id=first.original_video_id,
            video_id=first.video_id,
            species=first.species,
            label_id=first.label_id,
            label_name=first.label_name,
            frame_ids=frame_ids,
            rel_paths=[row.rel_path for row in current_rows],
            image_paths=[row.image_path for row in current_rows],
            split=first.split,
            dataset=first.dataset,
        )
        clips.append(clip)

    for row in rows[1:]:
        prev = current_rows[-1]
        contiguous = (
            row.original_video_id == prev.original_video_id
            and row.label_id == prev.label_id
            and row.frame_id == prev.frame_id + 1
        )
        if contiguous:
            current_rows.append(row)
            continue
        flush()
        current_rows = [row]

    flush()
    return clips


def sample_clip_frames(clip: KABRClipRecord, max_frames: int | None) -> list[str]:
    if max_frames is None:
        return list(clip.image_paths)
    if max_frames <= 0:
        raise ValueError("max_frames must be >= 1 when provided")
    if len(clip.image_paths) <= max_frames:
        return list(clip.image_paths)
    if max_frames == 1:
        return [clip.image_paths[len(clip.image_paths) // 2]]

    step = (len(clip.image_paths) - 1) / float(max_frames - 1)
    indices = []
    for idx in range(max_frames):
        candidate = int(round(idx * step))
        candidate = min(candidate, len(clip.image_paths) - 1)
        if indices and candidate <= indices[-1]:
            candidate = min(indices[-1] + 1, len(clip.image_paths) - 1)
        indices.append(candidate)
    return [clip.image_paths[index] for index in indices]


def sample_records(records: list, max_samples: int, seed: int) -> list:
    if max_samples is None or max_samples >= len(records):
        return list(records)
    rng = random.Random(seed)
    return rng.sample(list(records), max_samples)
