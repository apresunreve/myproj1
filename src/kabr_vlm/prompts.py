import json

from kabr_vlm.datasets import SPECIES_LABELS, get_dataset_spec
from kabr_vlm.schemas import KABRAnnotation


def build_joint_prompt(annotation: KABRAnnotation, dataset: str = "kabr") -> str:
    dataset_name = get_dataset_spec(dataset).prompt_name
    return (
        f"You are analyzing a video clip from the {dataset_name}.\n"
        "Identify the animal species present and the behavior being performed.\n"
        "Return the result as code-like recognize(...) statements.\n"
        'Use entity_type values "animal" and "action".'
    )


def build_joint_answer(annotation: KABRAnnotation) -> str:
    animal_stmt = f'recognize(entity_type="animal", value="{annotation.species}")'
    action_stmt = (
        'recognize(entity_type="action", '
        f'value="{annotation.behavior}", condition=\'animal == "{annotation.species}"\')'
    )
    return f"{animal_stmt}\n{action_stmt}"


def build_task_prompt(
    task: str,
    unit: str = "clip",
    output_format: str = "structured",
    dataset: str = "kabr",
    behavior_labels: list[str] | None = None,
    species_labels: list[str] | None = None,
) -> str:
    if output_format not in {"structured", "plain"}:
        raise ValueError(f"Unsupported output format: {output_format}")

    spec = get_dataset_spec(dataset)
    dataset_name = spec.prompt_name
    behavior_options = behavior_labels or []
    species_options = species_labels or list(spec.species_labels) or SPECIES_LABELS

    if task == "animal_recognition":
        if output_format == "plain":
            return (
                f"You are analyzing a {unit} from the {dataset_name}.\n"
                "Identify the animal species present.\n"
                f"Reply with exactly one label from: {', '.join(species_options)}."
            )
        return (
            f"You are analyzing a {unit} from the {dataset_name}.\n"
            "Identify the animal species present.\n"
            "Return the result as a single code-like recognize(...) statement."
        )
    if task == "action_recognition":
        if output_format == "plain":
            return (
                f"You are analyzing a {unit} from the {dataset_name}.\n"
                "Identify the behavior being performed.\n"
                f"Reply with exactly one behavior label from: {', '.join(behavior_options)}."
            )
        return (
            f"You are analyzing a {unit} from the {dataset_name}.\n"
            "Identify the behavior being performed.\n"
            "Return the result as a single code-like recognize(...) statement."
        )
    if output_format == "plain":
        return (
            f"You are analyzing a {unit} from the {dataset_name}.\n"
            "Identify the animal species present and the behavior being performed.\n"
            'Reply with JSON: {"species": "<label>", "behavior": "<label>"}.'
        )
    return (
        f"You are analyzing a {unit} from the {dataset_name}.\n"
        "Identify the animal species present and the behavior being performed.\n"
        "Return the result as code-like recognize(...) statements.\n"
        'Use entity_type values "animal" and "action".'
    )


def build_task_answer(
    task: str,
    species: str,
    behavior: str,
    output_format: str = "structured",
) -> str:
    if output_format not in {"structured", "plain"}:
        raise ValueError(f"Unsupported output format: {output_format}")

    if task == "animal_recognition":
        if output_format == "plain":
            return species
        return f'recognize(entity_type="animal", value="{species}")'
    if task == "action_recognition":
        if output_format == "plain":
            return behavior
        return f'recognize(entity_type="action", value="{behavior}")'
    if output_format == "plain":
        return json.dumps({"species": species, "behavior": behavior}, ensure_ascii=True)
    animal_stmt = f'recognize(entity_type="animal", value="{species}")'
    action_stmt = (
        'recognize(entity_type="action", '
        f'value="{behavior}", condition=\'animal == "{species}"\')'
    )
    return f"{animal_stmt}\n{action_stmt}"
