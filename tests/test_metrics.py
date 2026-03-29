from kabr_vlm.metrics import score_prediction_records


def test_score_prediction_records_behavior() -> None:
    rows = [
        {
            "task": "behavior",
            "gold": {"behavior": "Walk"},
            "prediction": {"behavior": "Walk"},
            "parsed": True,
        },
        {
            "task": "behavior",
            "gold": {"behavior": "Graze"},
            "prediction": {"behavior": "Walk"},
            "parsed": True,
        },
    ]

    metrics = score_prediction_records(rows)

    assert metrics["evaluated"] == 2
    assert metrics["parsed"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"]["Graze"]["Walk"] == 1


def test_score_prediction_records_x3d_schema() -> None:
    rows = [
        {
            "clip_id": "clip-a",
            "gold_label_id": 0,
            "gold_label_name": "Walk",
            "pred_label_id": 0,
            "pred_label_name": "Walk",
        },
        {
            "clip_id": "clip-b",
            "gold_label_id": 1,
            "gold_label_name": "Graze",
            "pred_label_id": 0,
            "pred_label_name": "Walk",
        },
    ]

    metrics = score_prediction_records(rows, task="behavior")

    assert metrics["task"] == "behavior"
    assert metrics["evaluated"] == 2
    assert metrics["parsed"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"]["Graze"]["Walk"] == 1
