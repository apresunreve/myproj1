import argparse
import json
from pathlib import Path

from kabr_vlm.config import load_config
from kabr_vlm.training.qwen_species import train_qwen_species_model


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_first_jsonl_row(path: Path) -> dict | None:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a VLM on KABR VideoQA data.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config for project: {config.project_name}")
    print(f"Model: {config.model.base_model_id}")
    print(f"Train file: {config.data.train_file}")
    print(f"Validation file: {config.data.val_file}")
    print(f"Task mode: {config.data.task_mode}")
    print(f"Input type: {config.data.input_type}")
    print(f"Prompt mode: {config.model.prompt_mode}")
    print(f"Output format: {config.model.output_format}")
    print(f"Training method: {config.training.method}")
    Path(config.training.output_dir).mkdir(parents=True, exist_ok=True)
    for split_name, split_path in [
        ("train", config.data.train_file),
        ("val", config.data.val_file),
    ]:
        if split_path.exists():
            print(f"{split_name}_rows: {count_jsonl_rows(split_path)}")
            example = read_first_jsonl_row(split_path)
            if example is not None:
                print(
                    f"{split_name}_example: id={example.get('id')} "
                    f"task={example.get('task')} num_images={len(example.get('image_paths', []))}"
                )
        else:
            print(f"{split_name}_rows: missing file")
    if config.training.dry_run:
        print("Dry run only. Dataset and config validation completed.")
        return

    if config.data.task_mode != "animal_recognition":
        raise ValueError(
            "This first fine-tuning backend currently supports only task_mode=animal_recognition."
        )
    if config.model.output_format != "plain":
        raise ValueError("This first fine-tuning backend currently supports output_format=plain.")
    if config.training.method != "lora":
        raise ValueError("This first fine-tuning backend currently supports method=lora.")

    metrics = train_qwen_species_model(config)
    print(
        "Training complete. "
        f"train_samples={metrics['train_samples']} val_samples={metrics['val_samples']} "
        f"train_loss={metrics.get('train_loss')}"
    )


if __name__ == "__main__":
    main()
