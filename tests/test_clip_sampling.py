from kabr_vlm.data.kabr_frames import (
    KABRFrameRecord,
    group_frame_records_into_clips,
    sample_clip_frames,
)


def build_row(frame_id: int, label_id: int, label_name: str) -> KABRFrameRecord:
    return KABRFrameRecord(
        original_video_id="ZG0001.1",
        video_id="1",
        frame_id=frame_id,
        rel_path=f"ZG0001.1/{frame_id}.jpg",
        image_path=f"/tmp/ZG0001.1/{frame_id}.jpg",
        label_id=label_id,
        label_name=label_name,
        species="grevys zebra",
        split="train",
    )


def test_group_frame_records_into_clips_splits_on_label_change() -> None:
    rows = [
        build_row(1, 0, "Walk"),
        build_row(2, 0, "Walk"),
        build_row(3, 1, "Graze"),
        build_row(4, 1, "Graze"),
    ]

    clips = group_frame_records_into_clips(rows)

    assert len(clips) == 2
    assert clips[0].frame_ids == [1, 2]
    assert clips[1].frame_ids == [3, 4]


def test_sample_clip_frames_keeps_order_and_boundaries() -> None:
    clip = group_frame_records_into_clips([build_row(i, 0, "Walk") for i in range(1, 11)])[0]

    sampled = sample_clip_frames(clip, max_frames=4)

    assert sampled[0].endswith("/1.jpg")
    assert sampled[-1].endswith("/10.jpg")
    assert len(sampled) == 4

