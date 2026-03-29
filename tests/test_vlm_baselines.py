from kabr_vlm.vlm_baselines import parse_prediction


def test_parse_behavior_prediction_from_json() -> None:
    pred = parse_prediction("behavior", '{"behavior": "Graze"}')
    assert pred == {"behavior": "Graze"}


def test_parse_joint_prediction_from_free_text() -> None:
    pred = parse_prediction(
        "joint",
        "The animal is a plains zebra and the behavior looks like Head Up.",
    )
    assert pred == {"behavior": "Head Up", "species": "plains zebra"}


def test_parse_behavior_prediction_from_structured_output() -> None:
    pred = parse_prediction(
        "behavior",
        'recognize(entity_type="action", value="Auto-Groom")',
    )
    assert pred == {"behavior": "Auto-Groom"}
