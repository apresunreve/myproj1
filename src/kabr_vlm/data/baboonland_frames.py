import json
from dataclasses import dataclass
from pathlib import Path


BABOONLAND_SPECIES = "baboon"


@dataclass(frozen=True)
class BaboonlandFrameRecord:
    original_video_id: str
    video_id: str
    frame_id: int
    rel_path: str
    image_path: str
    label_id: int
    label_name: str
    species: str
    split: str


@dataclass(frozen=True)
class BaboonlandClipRecord:
    clip_id: str
    original_video_id: str
    video_id: str
    species: str
    label_id: int
    label_name: str
    frame_ids: list[int]
    rel_paths: list[str]
    image_paths: list[str]
    split: str


def _load_label_lookup(classes_path: str | Path) -> dict[int, str]:
    payload = json.loads(Path(classes_path).read_text(encoding="utf-8"))
    return {int(label_id): label_name for label_name, label_id in payload.items()}


def _infer_split(annotation_path: str | Path) -> str:
    name = Path(annotation_path).stem.lower()
    if "train" in name:
        return "train"
    if "val" in name:
        return "val"
    if "test" in name:
        return "test"
    return name or "unknown"


def read_baboonland_frame_annotations(
    annotation_path: str | Path,
    image_root: str | Path,
    classes_path: str | Path,
) -> list[BaboonlandFrameRecord]:
    image_root = Path(image_root)
    label_lookup = _load_label_lookup(classes_path)
    split = _infer_split(annotation_path)
    rows: list[BaboonlandFrameRecord] = []

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
                BaboonlandFrameRecord(
                    original_video_id=original_video_id,
                    video_id=video_id,
                    frame_id=int(frame_id),
                    rel_path=rel_path,
                    image_path=str(image_root / rel_path),
                    label_id=label_id_int,
                    label_name=label_lookup[label_id_int],
                    species=BABOONLAND_SPECIES,
                    split=split,
                )
            )

    rows.sort(key=lambda row: (row.original_video_id, row.label_id, row.frame_id))
    return rows


def group_frame_records_into_clips(
    rows: list[BaboonlandFrameRecord],
) -> list[BaboonlandClipRecord]:
    if not rows:
        return []

    clips: list[BaboonlandClipRecord] = []
    current_rows: list[BaboonlandFrameRecord] = [rows[0]]

    def flush() -> None:
        if not current_rows:
            return
        first = current_rows[0]
        frame_ids = [row.frame_id for row in current_rows]
        clips.append(
            BaboonlandClipRecord(
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
            )
        )

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
