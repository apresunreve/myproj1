UV ?= $(HOME)/.local/bin/uv

.PHONY: uv-venv install install-dev install-infer lint test convert-example

uv-venv:
	$(UV) venv .uvenv

install: uv-venv
	$(UV) pip install --python .uvenv/bin/python -e .

install-dev: uv-venv
	$(UV) pip install --python .uvenv/bin/python -e ".[dev]"

install-infer: uv-venv
	$(UV) pip install --python .uvenv/bin/python -e ".[infer]"

lint:
	$(UV) run --python .uvenv/bin/python ruff check .

test:
	$(UV) run --python .uvenv/bin/python pytest

convert-example:
	$(UV) run --python .uvenv/bin/python python -m kabr_vlm.cli convert-kabr-split \
		--annotation ../KABR_files/annotation/train.csv \
		--image-root ../KABR_files/dataset/image \
		--classes ../KABR_files/annotation/classes.json \
		--task joint_animal_action_recognition \
		--output data/processed/kabr_videoqa_train.jsonl
