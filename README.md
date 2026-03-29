# KABR VLM

This repository is a starting point for a KABR-based vision-language project inspired by the COLM 2025 paper "Fine-tuning Vision-Language Models for Animal Behavior Analysis".

The intended workflow is:

1. Start from KABR frame or clip annotations.
2. Convert them into clip-level multimodal supervision for prompting or fine-tuning.
3. Fine-tune a video-language model on the converted data.
4. Evaluate on behavior recognition and joint species-action tasks.

## Scope

This scaffold focuses on:

- KABR-to-VideoQA data conversion
- structured or plain-label supervision for animal and action recognition
- prompt generation for supervised VLM training
- frame-level VLM probing baselines on raw KABR frames
- clip-level VLM baselines built from ordered KABR frame segments
- aggregate scoring with JSON metrics and confusion CSV artifacts
- a working Qwen2.5-VL LoRA fine-tuning path for clip-level species classification

It does not ship KABR data. Put raw data under `data/raw/` and generated artifacts under `data/processed/` or `artifacts/`.

## Layout

```text
src/kabr_vlm/
  cli.py
  metrics.py
  vlm_baselines.py
  data/
    kabr.py
    kabr_frames.py
  training/
    finetune.py
slurm/
  kabr_eval_jobs.tsv
  submit_kabr_eval_array.sh
  run_kabr_eval_array.sh
```

## Quick Start

```bash
python3 -m pip install --user uv
~/.local/bin/uv venv .uvenv
~/.local/bin/uv pip install --python .uvenv/bin/python -e ".[dev]"
```

Convert the released KABR split into clip-level species supervision for fine-tuning:

```bash
~/.local/bin/uv run --python .uvenv/bin/python kabr-vlm convert-kabr-split \
  --annotation ../KABR_files/annotation/train.csv \
  --image-root ../KABR_files/dataset/image \
  --classes ../KABR_files/annotation/classes.json \
  --task animal_recognition \
  --max-frames-per-clip 8 \
  --output-format plain \
  --output data/processed/kabr_species_train.jsonl
```

Repeat for validation:

```bash
~/.local/bin/uv run --python .uvenv/bin/python kabr-vlm convert-kabr-split \
  --annotation ../KABR_files/annotation/val.csv \
  --image-root ../KABR_files/dataset/image \
  --classes ../KABR_files/annotation/classes.json \
  --task animal_recognition \
  --max-frames-per-clip 8 \
  --output-format plain \
  --output data/processed/kabr_species_val.jsonl
```

Run the Qwen2.5-VL LoRA fine-tuning path:

```bash
~/.local/bin/uv pip install --python .uvenv/bin/python -e ".[train]"
~/.local/bin/uv run --python .uvenv/bin/python \
  python -m kabr_vlm.training.finetune --config configs/train.example.yaml
```

Evaluate a saved adapter on clip-level KABR validation data:

```bash
~/.local/bin/uv run --python .uvenv/bin/python kabr-vlm eval-finetuned-hf-vlm \
  --annotation ../KABR_files/annotation/val.csv \
  --image-root ../KABR_files/dataset/image \
  --classes ../KABR_files/annotation/classes.json \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter-path artifacts/checkpoints/qwen25vl3b_species_lora \
  --task species \
  --output artifacts/qwen25vl3b_species_lora_val.jsonl
```

Run a frame-level VLM probing baseline on KABR validation data:

```bash
~/.local/bin/uv pip install --python .uvenv/bin/python -e ".[infer]"
~/.local/bin/uv run --python .uvenv/bin/python kabr-vlm eval-frame-hf-vlm \
  --annotation ../KABR_files/annotation/val.csv \
  --image-root ../KABR_files/dataset/image \
  --classes ../KABR_files/annotation/classes.json \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --task behavior \
  --prompt-mode plain \
  --max-samples 128 \
  --output artifacts/qwen25_vl_val_behavior.jsonl
```

Run a clip-level baseline with structured outputs:

```bash
~/.local/bin/uv run --python .uvenv/bin/python kabr-vlm eval-clip-hf-vlm \
  --annotation ../KABR_files/annotation/val.csv \
  --image-root ../KABR_files/dataset/image \
  --classes ../KABR_files/annotation/classes.json \
  --model-id HuggingFaceTB/SmolVLM-256M-Instruct \
  --task joint \
  --prompt-mode structured \
  --max-samples 16 \
  --max-frames-per-clip 8 \
  --output artifacts/smolvlm_joint_clip_val.jsonl
```

Score a prediction file:

```bash
~/.local/bin/uv run --python .uvenv/bin/python kabr-vlm score-predictions \
  --predictions artifacts/smolvlm_joint_clip_val.jsonl \
  --metrics-output artifacts/smolvlm_joint_clip_val.metrics.json \
  --confusion-output artifacts/smolvlm_joint_clip_val.confusion.csv
```

## Slurm Enumeration

The repository includes a simple Slurm array setup for enumerating multiple model/task runs over KABR validation data.

Job definitions live in:

- `slurm/kabr_eval_jobs.tsv`

Each row defines one run:

- `job_name`
- `command`
- `model_id`
- `task`
- `prompt_mode`
- `max_samples`
- `max_frames_per_clip`
- `max_new_tokens`
- `output_name`

Submit the full array from a Slurm login node:

```bash
bash slurm/submit_kabr_eval_array.sh
```

Useful environment overrides:

```bash
PARTITION=gpu GRES=gpu:1 MEM=48G TIME=08:00:00 \
bash slurm/submit_kabr_eval_array.sh
```

The array runner expects an existing virtualenv at `.venv` with the repo installed and inference dependencies available.
The array runner expects `uv` plus an existing project env at `.uvenv`.

## Supported Tasks

For evaluation:

- `species`
- `behavior`
- `joint`

For supervision conversion:

- `animal_recognition`
- `action_recognition`
- `joint_animal_action_recognition`

## Expected Annotation Columns

The converter expects a CSV with these fields:

- `video_id`
- `species`
- `behavior`
- `clip_path`

Optional fields:

- `question_id`
- `split`
- `start_time`
- `end_time`
- `notes`

If you use the released KABR split files under `../KABR_files/annotation/`, use `convert-kabr-split` instead.

## Output Formats

The converter supports:

- `structured`: code-like `recognize(...)` targets
- `plain`: plain label targets for the first species fine-tuning path

Structured example target:

```python
recognize(entity_type="animal", value="plains zebra")
recognize(entity_type="action", value="graze", condition='animal == "plains zebra"')
```

Plain species target:

```text
giraffe
```

## Next Steps

- compare frame probing against clip baselines
- run cluster sweeps with the Slurm matrix under `slurm/kabr_eval_jobs.tsv`
- extend the first LoRA species path to behavior recognition
- compare plain-label and structured-output fine-tuning
