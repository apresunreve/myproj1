from kabr_vlm.vlm_baselines import build_prompt, parse_prediction


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


def test_parse_behavior_prediction_with_custom_baboon_labels() -> None:
    pred = parse_prediction(
        "behavior",
        '{"behavior": "Mutual Grooming"}',
        behavior_labels=["Walking/Running", "Mutual Grooming", "Sleeping"],
    )
    assert pred == {"behavior": "Mutual Grooming"}


def test_build_prompt_uses_dataset_specific_behavior_labels() -> None:
    prompt = build_prompt(
        "behavior",
        unit="clip",
        prompt_mode="label",
        dataset="baboonland",
        behavior_labels=["Walking/Running", "Sleeping"],
    )
    assert "baboonland dataset" in prompt
    assert "Walking/Running" in prompt
    assert "Sleeping" in prompt
