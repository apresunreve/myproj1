from pathlib import Path

import typer

from kabr_vlm.data.kabr import (
    convert_charades_split_to_clip_manifests,
    convert_annotations,
    convert_kabr_split_to_videoqa,
    read_annotations,
    write_clip_manifest_jsonl,
    write_jsonl,
)
from kabr_vlm.data.kabr_frames import read_charades_frame_annotations, read_kabr_frame_annotations
from kabr_vlm.datasets import ordered_label_names
from kabr_vlm.metrics import (
    load_prediction_records,
    score_prediction_records,
    write_confusion_csv,
    write_metrics,
)
from kabr_vlm.reporting import aggregate_paper_results
from kabr_vlm.config import load_config
from kabr_vlm.training.qwen_tasks import evaluate_hf_adapter, evaluate_hf_full_model
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
    dataset: str = typer.Option("kabr", help="Dataset name: kabr or baboonland."),
    task: str = typer.Option("joint_animal_action_recognition", help="One of: animal_recognition, action_recognition, joint_animal_action_recognition."),
    max_frames_per_clip: int = typer.Option(8, min=1, help="Max ordered frames to keep per clip."),
    output_format: str = typer.Option("structured", help="One of: structured, plain."),
    output: Path = typer.Option(..., help="Output JSONL file."),
) -> None:
    examples = convert_kabr_split_to_videoqa(
        annotation_path=annotation,
        image_root=image_root,
        classes_path=classes,
        dataset=dataset,
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
    dataset: str = typer.Option("kabr", help="Dataset name: kabr or baboonland."),
    model_id: str = typer.Option(..., help="Hugging Face VLM model id."),
    task: str = typer.Option("behavior", help="One of: behavior, species, joint."),
    prompt_mode: str = typer.Option("plain", help="One of: plain, structured."),
    max_samples: int = typer.Option(128, min=1, help="Number of frames to evaluate."),
    max_frames_per_clip: int = typer.Option(1, min=1, help="Ignored for frame evaluation; accepted for matrix compatibility."),
    seed: int = typer.Option(42, help="Random seed for row sampling."),
    max_new_tokens: int = typer.Option(64, min=1, help="Generation length."),
    attention_experiment: str | None = typer.Option(
        None,
        help="Optional experimental attention backend, e.g. mla.",
    ),
    resume: bool = typer.Option(False, help="Resume from an existing partial JSONL output."),
    output: Path = typer.Option(..., help="Prediction JSONL output path."),
) -> None:
    _ = max_frames_per_clip
    if dataset != "kabr":
        raise typer.BadParameter("Frame-level HF VLM eval currently supports only dataset=kabr.")
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
        resume=resume,
        attention_experiment=attention_experiment,
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
    dataset: str = typer.Option("kabr", help="Dataset name: kabr or baboonland."),
    model_id: str = typer.Option(..., help="Hugging Face VLM model id."),
    task: str = typer.Option("behavior", help="One of: behavior, species, joint."),
    prompt_mode: str = typer.Option("plain", help="One of: plain, structured."),
    max_samples: int = typer.Option(128, min=1, help="Number of clips to evaluate."),
    max_frames_per_clip: int = typer.Option(8, min=1, help="Ordered frames sampled per clip."),
    use_all_frames: bool = typer.Option(False, help="Use every frame in each clip instead of subsampling."),
    seed: int = typer.Option(42, help="Random seed for clip sampling."),
    max_new_tokens: int = typer.Option(64, min=1, help="Generation length."),
    attention_experiment: str | None = typer.Option(
        None,
        help="Optional experimental attention backend, e.g. mla.",
    ),
    resume: bool = typer.Option(False, help="Resume from an existing partial JSONL output."),
    output: Path = typer.Option(..., help="Prediction JSONL output path."),
) -> None:
    rows = read_charades_frame_annotations(annotation, image_root, classes, dataset=dataset)
    summary = evaluate_clip_records(
        rows=rows,
        model_id=model_id,
        task=task,
        max_samples=max_samples,
        seed=seed,
        max_new_tokens=max_new_tokens,
        max_frames_per_clip=None if use_all_frames else max_frames_per_clip,
        output_path=output,
        prompt_mode=prompt_mode,
        resume=resume,
        dataset=dataset,
        attention_experiment=attention_experiment,
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
    dataset: str = typer.Option("kabr", help="Dataset name: kabr or baboonland."),
    model_id: str = typer.Option(..., help="Base Hugging Face VLM model id."),
    adapter_path: Path = typer.Option(..., exists=True, help="LoRA adapter directory."),
    task: str = typer.Option("species", help="Currently supports species or behavior."),
    prompt_mode: str = typer.Option("label", help="One of: label, structured."),
    max_samples: int | None = typer.Option(None, min=1, help="Optional number of clips to evaluate."),
    seed: int = typer.Option(42, help="Random seed for clip sampling."),
    max_new_tokens: int = typer.Option(8, min=1, help="Generation length."),
    max_frames_per_clip: int = typer.Option(8, min=1, help="Ordered frames sampled per clip."),
    attention_experiment: str | None = typer.Option(
        None,
        help="Optional experimental attention backend, e.g. mla.",
    ),
    output: Path = typer.Option(..., help="Prediction JSONL output path."),
) -> None:
    if task not in {"species", "behavior"}:
        raise typer.BadParameter("This adapter-backed evaluator currently supports only task=species or behavior.")
    summary = evaluate_hf_adapter(
        annotation_path=annotation,
        image_root=image_root,
        classes_path=classes,
        model_id=model_id,
        adapter_path=adapter_path,
        task=task,
        output_path=output,
        max_samples=max_samples,
        seed=seed,
        max_new_tokens=max_new_tokens,
        max_frames_per_clip=max_frames_per_clip,
        prompt_mode=prompt_mode,
        dataset=dataset,
        attention_experiment=attention_experiment,
    )
    typer.echo(
        "evaluated="
        f"{summary['evaluated']} parsed={summary['parsed']} "
        f"correct={summary['correct']} accuracy={summary['accuracy']:.4f} "
        f"macro_f1={summary['macro_f1']:.4f}"
    )


@app.command("eval-full-hf-vlm")
def eval_full_hf_vlm(
    annotation: Path = typer.Option(..., exists=True, readable=True, help="KABR split annotation file."),
    image_root: Path = typer.Option(..., exists=True, file_okay=False, help="Root KABR image dir."),
    classes: Path = typer.Option(..., exists=True, readable=True, help="KABR classes.json path."),
    model_id: str = typer.Option(..., help="Original base Hugging Face VLM model id."),
    checkpoint_path: Path = typer.Option(..., exists=True, help="Full fine-tuned checkpoint directory."),
    task: str = typer.Option("species", help="Currently supports species or behavior."),
    prompt_mode: str = typer.Option("label", help="One of: label, structured."),
    max_samples: int | None = typer.Option(None, min=1, help="Optional number of clips to evaluate."),
    seed: int = typer.Option(42, help="Random seed for clip sampling."),
    max_new_tokens: int = typer.Option(8, min=1, help="Generation length."),
    output: Path = typer.Option(..., help="Prediction JSONL output path."),
) -> None:
    if task not in {"species", "behavior"}:
        raise typer.BadParameter("This full-checkpoint evaluator currently supports only task=species or behavior.")
    summary = evaluate_hf_full_model(
        annotation_path=annotation,
        image_root=image_root,
        classes_path=classes,
        model_id=model_id,
        checkpoint_path=checkpoint_path,
        task=task,
        output_path=output,
        max_samples=max_samples,
        seed=seed,
        max_new_tokens=max_new_tokens,
        prompt_mode=prompt_mode,
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
    classes_path: Path | None = typer.Option(None, exists=True, readable=True, help="Optional classes.json for label ordering."),
) -> None:
    rows = load_prediction_records(predictions)
    labels_override = ordered_label_names(classes_path) if classes_path is not None else None
    metrics = score_prediction_records(rows, task=task, labels_override=labels_override)
    write_metrics(metrics_output, metrics)
    write_confusion_csv(confusion_output, metrics)
    typer.echo(
        f"evaluated={metrics['evaluated']} parsed={metrics['parsed']} "
        f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
    )


@app.command("prepare-charades-clip-manifests")
def prepare_charades_clip_manifests(
    annotation_dir: Path = typer.Option(..., exists=True, file_okay=False, help="Directory containing train.csv and val.csv."),
    image_root: Path = typer.Option(..., exists=True, file_okay=False, help="Root image directory."),
    classes: Path = typer.Option(..., exists=True, readable=True, help="classes.json path."),
    dataset: str = typer.Option("kabr", help="Dataset name: kabr or baboonland."),
    output_dir: Path = typer.Option(..., help="Output directory for train/val manifests."),
    max_clips: int = typer.Option(0, min=0, help="Optional clip cap per split."),
    copy_val_to_test: bool = typer.Option(False, help="Copy val.jsonl to test.jsonl."),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val"):
        clips = convert_charades_split_to_clip_manifests(
            annotation_path=annotation_dir / f"{split_name}.csv",
            image_root=image_root,
            classes_path=classes,
            dataset=dataset,
        )
        if max_clips > 0:
            clips = clips[:max_clips]
        write_clip_manifest_jsonl(output_dir / f"{split_name}.jsonl", clips)
    import json

    with (output_dir / "classes.json").open("w", encoding="utf-8") as handle:
        json.dump({label: index for index, label in enumerate(ordered_label_names(classes))}, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    if copy_val_to_test:
        (output_dir / "test.jsonl").write_text((output_dir / "val.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    typer.echo(f"Wrote manifests to {output_dir}")


@app.command("train-frozen-features")
def train_frozen_features(
    config_path: Path = typer.Option(..., "--config", exists=True, readable=True, help="YAML config."),
) -> None:
    from kabr_vlm.training.frozen_features import train_frozen_feature_model

    config = load_config(config_path)
    metrics = train_frozen_feature_model(config)
    typer.echo(
        f"train_samples={metrics['train_samples']} val_samples={metrics['val_samples']} "
        f"val_accuracy={metrics.get('val_accuracy', 0.0):.4f}"
    )


@app.command("eval-frozen-features")
def eval_frozen_features(
    config_path: Path = typer.Option(..., "--config", exists=True, readable=True, help="YAML config."),
    checkpoint_path: Path = typer.Option(..., exists=True, readable=True, help="Frozen classifier checkpoint."),
    output: Path = typer.Option(..., help="Prediction JSONL output path."),
    max_samples: int | None = typer.Option(None, min=1, help="Optional number of examples."),
) -> None:
    from kabr_vlm.training.frozen_features import evaluate_frozen_feature_model

    config = load_config(config_path)
    summary = evaluate_frozen_feature_model(config, checkpoint_path=checkpoint_path, output_path=output, max_samples=max_samples)
    typer.echo(
        f"evaluated={summary['evaluated']} parsed={summary['parsed']} "
        f"accuracy={summary['accuracy']:.4f} macro_f1={summary['macro_f1']:.4f}"
    )


@app.command("train-distill")
def train_distill(
    config_path: Path = typer.Option(..., "--config", exists=True, readable=True, help="YAML config."),
) -> None:
    from kabr_vlm.training.distill import train_distilled_feature_model

    config = load_config(config_path)
    metrics = train_distilled_feature_model(config)
    typer.echo(
        f"train_samples={metrics['train_samples']} val_samples={metrics['val_samples']} "
        f"val_accuracy={metrics.get('val_accuracy', 0.0):.4f}"
    )


@app.command("eval-distill")
def eval_distill(
    config_path: Path = typer.Option(..., "--config", exists=True, readable=True, help="YAML config."),
    checkpoint_path: Path = typer.Option(..., exists=True, readable=True, help="Distilled classifier checkpoint."),
    output: Path = typer.Option(..., help="Prediction JSONL output path."),
    max_samples: int | None = typer.Option(None, min=1, help="Optional number of examples."),
) -> None:
    from kabr_vlm.training.distill import evaluate_distilled_feature_model

    config = load_config(config_path)
    summary = evaluate_distilled_feature_model(config, checkpoint_path=checkpoint_path, output_path=output, max_samples=max_samples)
    typer.echo(
        f"evaluated={summary['evaluated']} parsed={summary['parsed']} "
        f"accuracy={summary['accuracy']:.4f} macro_f1={summary['macro_f1']:.4f}"
    )


@app.command("aggregate-paper-results")
def aggregate_results(
    manifest: Path = typer.Option(..., exists=True, readable=True, help="YAML manifest listing experiment runs."),
    output_dir: Path = typer.Option(..., help="Output directory for paper tables."),
) -> None:
    outputs = aggregate_paper_results(manifest, output_dir)
    typer.echo(
        f"wrote main_table={outputs['main_table_csv']} per_class={outputs['per_class_csv']} summary={outputs['summary_json']}"
    )


@app.command("bench-latency")
def bench_latency(
    annotation: Path = typer.Option(..., exists=True, readable=True, help="Val annotation CSV."),
    image_root: Path = typer.Option(..., exists=True, file_okay=False, help="Root image directory."),
    classes: Path = typer.Option(..., exists=True, readable=True, help="classes.json path."),
    model_id: str = typer.Option(..., help="HuggingFace model id."),
    adapter_path: Path = typer.Option(None, exists=True, help="Optional LoRA adapter directory (omit for zero-shot)."),
    dataset: str = typer.Option("kabr", help="Dataset name: kabr or baboonland."),
    task: str = typer.Option("behavior", help="Task name passed to the prompt builder."),
    n_clips: int = typer.Option(100, min=1, help="Number of clips to time (after warmup)."),
    warmup_clips: int = typer.Option(5, min=0, help="Clips to run before timing starts."),
    max_frames_per_clip: int = typer.Option(8, min=1, help="Frames sampled per clip."),
    seed: int = typer.Option(42, help="Random seed for clip sampling."),
    output: Path = typer.Option(..., help="Path for per-clip JSONL predictions."),
    summary_output: Path = typer.Option(None, help="Optional path for latency summary JSON (default: <output>.latency.json)."),
) -> None:
    """Benchmark inference latency (ms/clip) for a VLM on KABR or Baboonland val clips.

    Runs --warmup-clips silent clips first to heat the GPU, then times --n-clips clips.
    Writes per-clip latency_ms into the prediction JSONL and a compact summary JSON.
    """
    import json
    import statistics
    import torch
    from kabr_vlm.vlm_baselines import (
        evaluate_records,
        build_prompt,
        load_hf_pipeline,
        load_hf_adapter_generator,
    )
    from kabr_vlm.data.kabr_frames import (
        read_kabr_frame_annotations,
        group_frame_records_into_clips,
        sample_records,
    )

    rows = read_kabr_frame_annotations(annotation, image_root, classes)
    clips = group_frame_records_into_clips(rows)
    sample = sample_records(clips, max_samples=n_clips + warmup_clips, seed=seed)

    if adapter_path is not None:
        generator = load_hf_adapter_generator(model_id=model_id, adapter_path=adapter_path)
        mode_label = "lora"
    else:
        generator = load_hf_pipeline(model_id)
        mode_label = "zero-shot"

    prompt = build_prompt(task, unit="clip", prompt_mode="label" if adapter_path else "plain")

    summary = evaluate_records(
        records=sample,
        generator=generator,
        prompt=prompt,
        task=task,
        model_id=model_id,
        max_new_tokens=8,
        output_path=output,
        unit="clip",
        max_frames_per_clip=max_frames_per_clip,
        warmup_clips=warmup_clips,
    )

    peak_gpu_mb = (
        round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        if torch.cuda.is_available()
        else None
    )

    latency_summary = {
        "model_id": model_id,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "mode": mode_label,
        "dataset": dataset,
        "task": task,
        "n_clips": summary["n_timed"],
        "warmup_clips": warmup_clips,
        "max_frames_per_clip": max_frames_per_clip,
        "accuracy": round(summary["accuracy"], 4),
        "latency_median_ms": summary.get("latency_median_ms"),
        "latency_p90_ms": summary.get("latency_p90_ms"),
        "latency_mean_ms": summary.get("latency_mean_ms"),
        "clips_per_sec": round(1000.0 / summary["latency_median_ms"], 2) if summary.get("latency_median_ms") else None,
        "peak_gpu_mb": peak_gpu_mb,
        "predictions_path": str(output),
    }

    summary_path = summary_output or output.with_suffix(".latency.json")
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(latency_summary, f, indent=2)

    typer.echo(
        f"model={model_id}  mode={mode_label}  dataset={dataset}  "
        f"n={latency_summary['n_clips']}  "
        f"median={latency_summary['latency_median_ms']} ms  "
        f"p90={latency_summary['latency_p90_ms']} ms  "
        f"acc={latency_summary['accuracy']:.4f}  "
        f"gpu_mb={peak_gpu_mb}"
    )
    typer.echo(f"summary → {summary_path}")


if __name__ == "__main__":
    app()
