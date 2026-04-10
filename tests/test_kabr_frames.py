from pathlib import Path

from kabr_vlm.data.kabr_frames import infer_species, load_label_map, read_charades_frame_annotations, read_kabr_frame_annotations


def test_infer_species_from_video_prefix() -> None:
    assert infer_species("G0067.1") == "giraffe"
    assert infer_species("ZG0001.1") == "grevys zebra"
    assert infer_species("ZP0001.1") == "plains zebra"


def test_read_kabr_frame_annotations_parses_real_format() -> None:
    rows = read_kabr_frame_annotations(
        annotation_path=Path("../KABR_files/annotation/train.csv"),
        image_root=Path("../KABR_files/dataset/image"),
        classes_path=Path("../KABR_files/annotation/classes.json"),
    )

    assert rows
    first = rows[0]
    assert first.original_video_id == "ZG0001.1"
    assert first.frame_id == 1
    assert first.label_name == "Walk"
    assert first.species == "grevys zebra"


def test_load_label_map_uses_numeric_ids() -> None:
    labels = load_label_map(Path("../KABR_files/annotation/classes.json"))
    assert labels[0] == "Walk"
    assert labels[7] == "Occluded"


def test_read_baboonland_frame_annotations_parses_behavior_only_format() -> None:
    rows = read_charades_frame_annotations(
        annotation_path=Path("/home/jhyan/baboon_data/charades/annotation/train.csv"),
        image_root=Path("/home/jhyan/baboon_data/charades/dataset/image"),
        classes_path=Path("/home/jhyan/baboon_data/charades/annotation/classes.json"),
        dataset="baboonland",
    )
    assert rows
    first = rows[0]
    assert first.original_video_id == "B0001.1"
    assert first.label_name == "Walking/Running"
    assert first.species is None
    assert first.dataset == "baboonland"
