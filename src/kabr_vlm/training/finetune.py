import argparse
import json
from pathlib import Path

from kabr_vlm.config import load_config
from kabr_vlm.training.qwen_tasks import find_latest_trainer_checkpoint, train_hf_task_model


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
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint directory to resume from, or 'auto' to use the latest checkpoint under training.output_dir.",
    )
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
    print(f"Attention experiment: {config.model.attention_experiment}")
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

    if config.data.task_mode not in {"animal_recognition", "action_recognition"}:
        raise ValueError("This fine-tuning backend currently supports only species or behavior.")
    if config.training.method not in {"lora", "full"}:
        raise ValueError("This entrypoint currently supports method=lora or method=full.")

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint == "auto":
        latest = find_latest_trainer_checkpoint(Path(config.training.output_dir))
        if latest is not None:
            resume_from_checkpoint = str(latest)
            print(f"Auto-resuming from checkpoint: {resume_from_checkpoint}")
        else:
            resume_from_checkpoint = None
            print("Auto-resume requested but no checkpoint-* directory was found. Starting fresh.")
    elif resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    metrics = train_hf_task_model(config, resume_from_checkpoint=resume_from_checkpoint)
    print(
        "Training complete. "
        f"train_samples={metrics['train_samples']} val_samples={metrics['val_samples']} "
        f"train_loss={metrics.get('train_loss')}"
    )


if __name__ == "__main__":
    main()
