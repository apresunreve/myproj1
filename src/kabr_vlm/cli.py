from pathlib import Path

import typer

from kabr_vlm.data.kabr import (
    convert_annotations,
    convert_kabr_split_to_videoqa,
    read_annotations,
    write_jsonl,
)
from kabr_vlm.data.kabr_frames import read_kabr_frame_annotations
from kabr_vlm.metrics import (
    load_prediction_records,
    score_prediction_records,
    write_confusion_csv,
    write_metrics,
)
from kabr_vlm.training.qwen_species import evaluate_qwen_species_adapter
from kabr_vlm.vlm_baselines import evaluate_clip_records, evaluate_hf_vlm_on_kabr

app = typer.Typer(add_completion=False, help="Utilities for KABR VLM workflows.")


@app.command("convert-kabr")
def convert_kabr(
    annotations: Path = typer.Option(..., exists=True, readable=True, help="Input annotation CSV."),
    output: Path = typer.Option(..., help="Output JSONL file."),
) -> None:
    rows = read_annotations(annotations)
    examples = convert_annotations(rows)
    write_jsonl(output, examples)
    typer.echo(f"Wrote {len(examples)} examples to {output}")


@app.command("convert-kabr-split")
def convert_kabr_split(
    annotation: Path = typer.Option(..., exists=True, readable=True, help="KABR split annotation file."),
    image_root: Path = typer.Option(..., exists=True, file_okay=False, help="Root KABR image dir."),
    classes: Path = typer.Option(..., exists=True, readable=True, help="KABR classes.json path."),
    task: str = typer.Option("joint_animal_action_recognition", help="One of: animal_recognition, action_recognition, joint_animal_action_recognition."),
    max_frames_per_clip: int = typer.Option(8, min=1, help="Max ordered frames to keep per clip."),
    output_format: str = typer.Option("structured", help="One of: structured, plain."),
    output: Path = typer.Option(..., help="Output JSONL file."),
) -> None:
    examples = convert_kabr_split_to_videoqa(
        annotation_path=annotation,
        image_root=image_root,
        classes_path=classes,
        task=task,
        max_frames_per_clip=max_frames_per_clip,
        output_format=output_format,
    )
    write_jsonl(output, examples)
    typer.echo(f"Wrote {len(examples)} examples to {output}")


@app.command("eval-frame-hf-vlm")
def eval_frame_hf_vlm(
    annotation: Path = typer.Option(..., exists=True, readable=True, help="KABR split annotation file."),
    image_root: Path = typer.Option(..., exists=True, file_okay=False, help="Root KABR image dir."),
    classes: Path = typer.Option(..., exists=True, readable=True, help="KABR classes.json path."),
    model_id: str = typer.Option(..., help="Hugging Face VLM model id."),
    task: str = typer.Option("behavior", help="One of: behavior, species, joint."),
    prompt_mode: str = typer.Option("plain", help="One of: plain, structured."),
    max_samples: int = typer.Option(128, min=1, help="Number of frames to evaluate."),
    max_frames_per_clip: int = typer.Option(1, min=1, help="Ignored for frame evaluation; accepted for matrix compatibility."),
    seed: int = typer.Option(42, help="Random seed for row sampling."),
    max_new_tokens: int = typer.Option(64, min=1, help="Generation length."),
    output: Path = typer.Option(..., help="Prediction JSONL output path."),
) -> None:
    _ = max_frames_per_clip
    summary = evaluate_hf_vlm_on_kabr(
        annotation_path=annotation,
        image_root=image_root,
        classes_path=classes,
        model_id=model_id,
        task=task,
        max_samples=max_samples,
        seed=seed,
        max_new_tokens=max_new_tokens,
        output_path=output,
        prompt_mode=prompt_mode,
    )
    typer.echo(
        "evaluated="
        f"{summary['evaluated']} parsed={summary['parsed']} "
        f"correct={summary['correct']} accuracy={summary['accuracy']:.4f}"
    )


@app.command("eval-clip-hf-vlm")
def eval_clip_hf_vlm(
    annotation: Path = typer.Option(..., exists=True, readable=True, help="KABR split annotation file."),
    image_root: Path = typer.Option(..., exists=True, file_okay=False, help="Root KABR image dir."),
    classes: Path = typer.Option(..., exists=True, readable=True, help="KABR classes.json path."),
    model_id: str = typer.Option(..., help="Hugging Face VLM model id."),
    task: str = typer.Option("behavior", help="One of: behavior, species, joint."),
    prompt_mode: str = typer.Option("plain", help="One of: plain, structured."),
    max_samples: int = typer.Option(128, min=1, help="Number of clips to evaluate."),
    max_frames_per_clip: int = typer.Option(8, min=1, help="Ordered frames sampled per clip."),
    seed: int = typer.Option(42, help="Random seed for clip sampling."),
    max_new_tokens: int = typer.Option(64, min=1, help="Generation length."),
    output: Path = typer.Option(..., help="Prediction JSONL output path."),
) -> None:
    rows = read_kabr_frame_annotations(annotation, image_root, classes)
    summary = evaluate_clip_records(
        rows=rows,
        model_id=model_id,
        task=task,
        max_samples=max_samples,
        seed=seed,
        max_new_tokens=max_new_tokens,
        max_frames_per_clip=max_frames_per_clip,
        output_path=output,
        prompt_mode=prompt_mode,
    )
    typer.echo(
        "evaluated="
        f"{summary['evaluated']} parsed={summary['parsed']} "
        f"correct={summary['correct']} accuracy={summary['accuracy']:.4f}"
    )


@app.command("eval-finetuned-hf-vlm")
def eval_finetuned_hf_vlm(
    annotation: Path = typer.Option(..., exists=True, readable=True, help="KABR split annotation file."),
    image_root: Path = typer.Option(..., exists=True, file_okay=False, help="Root KABR image dir."),
    classes: Path = typer.Option(..., exists=True, readable=True, help="KABR classes.json path."),
    model_id: str = typer.Option(..., help="Base Hugging Face VLM model id."),
    adapter_path: Path = typer.Option(..., exists=True, help="LoRA adapter directory."),
    task: str = typer.Option("species", help="Currently only species is supported."),
    max_samples: int | None = typer.Option(None, min=1, help="Optional number of clips to evaluate."),
    seed: int = typer.Option(42, help="Random seed for clip sampling."),
    max_new_tokens: int = typer.Option(8, min=1, help="Generation length."),
    output: Path = typer.Option(..., help="Prediction JSONL output path."),
) -> None:
    if task != "species":
        raise typer.BadParameter("This first adapter-backed evaluator currently supports only task=species.")
    summary = evaluate_qwen_species_adapter(
        annotation_path=annotation,
        image_root=image_root,
        classes_path=classes,
        model_id=model_id,
        adapter_path=adapter_path,
        output_path=output,
        max_samples=max_samples,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
    typer.echo(
        "evaluated="
        f"{summary['evaluated']} parsed={summary['parsed']} "
        f"correct={summary['correct']} accuracy={summary['accuracy']:.4f} "
        f"macro_f1={summary['macro_f1']:.4f}"
    )


@app.command("score-predictions")
def score_predictions(
    predictions: Path = typer.Option(..., exists=True, readable=True, help="Prediction JSONL input."),
    metrics_output: Path = typer.Option(..., help="Metrics JSON output."),
    confusion_output: Path = typer.Option(..., help="Confusion CSV output."),
    task: str | None = typer.Option(None, help="Optional task override for non-VLM schemas, e.g. behavior."),
) -> None:
    rows = load_prediction_records(predictions)
    metrics = score_prediction_records(rows, task=task)
    write_metrics(metrics_output, metrics)
    write_confusion_csv(confusion_output, metrics)
    typer.echo(
        f"evaluated={metrics['evaluated']} parsed={metrics['parsed']} "
        f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    app()
